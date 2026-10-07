"""CPU tests for the E2 shuffled-target control (e2_control). Synthetic bundle and random weights only; nothing
here is an E2 result."""
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import keras
import numpy as np

import arch1_data as ad
import e1_data as ed
import e1_probe as ep
import e1_train as et
import e2_control as e2
import pipeline_v2_config as v2cfg
import pl_convnext as pl
import stage34_cache_v2 as cache
import weighted_corn

try:
    from tests import v2_bundle_fixture as fx
except ImportError:
    import v2_bundle_fixture as fx

CW = list(weighted_corn.PREREGISTERED_CLASS_WEIGHTS)
PRIOR = (0.1, 0.1, 0.1, 0.1)
GATES = {k: {"PASS": True} for k in ("p_parity", "log_alias", "targets")}


class DerangementTests(unittest.TestCase):
    def test_true_permutation_without_fixed_points_and_independent_of_the_model_seed(self):
        ids = [f"{n:012x}" for n in range(2921)]
        partner = e2.derangement(ids)
        report = e2.verify_derangement(partner, ids, val_ids=["f" * 12])
        self.assertEqual((report["images"], report["fixed_points"], report["seed"]), (2921, 0, 20261001))
        self.assertTrue(all(partner[i] != i for i in ids))
        self.assertEqual(sorted(partner.values()), sorted(ids))            # every image is a partner exactly once
        self.assertEqual(list(partner), ids)
        self.assertEqual(partner, e2.derangement(ids))                     # one fixed mapping
        self.assertEqual(e2.DERANGEMENT_SEED, 20261001)
        self.assertNotEqual(partner, e2.derangement(ids, seed=1))

    def test_broken_mappings_are_refused(self):
        ids = [f"{n:012x}" for n in range(6)]
        good = e2.derangement(ids)
        with self.assertRaises(RuntimeError):
            e2.verify_derangement(dict(good, **{ids[0]: ids[0]}), ids)     # a fixed point
        doubled = dict(good)
        doubled[ids[0]] = good[ids[1]]
        with self.assertRaises(RuntimeError):
            e2.verify_derangement(doubled, ids)                            # a partner used twice
        with self.assertRaises(RuntimeError):
            e2.verify_derangement(good, ids[:-1])                          # does not cover the training set
        with self.assertRaises(RuntimeError):
            e2.verify_derangement(good, ids, val_ids=[ids[2]])             # a validation image as partner


class PairCountTests(unittest.TestCase):
    def test_weighted_pair_counts_equal_the_auroc_of_the_resampled_cells(self):
        rng = np.random.default_rng(0)
        n = 40
        positive = rng.random((n, 16, 16)) < 0.2
        positive[3] = False                                                # an image without positives
        positive[7] = True                                                 # an image without negatives
        score = (rng.normal(0, 1, (n, 16, 16)) + 0.7 * positive).astype(np.float32)
        u, n_pos, n_neg = e2.pair_counts(positive, score)
        ones = np.ones(n)
        self.assertAlmostEqual((ones @ u @ ones) / (n_pos.sum() * n_neg.sum()), ep.cell_auroc(positive, score), places=10)
        for seed in range(5):
            idx = np.random.default_rng(seed).integers(0, n, n)
            w = np.bincount(idx, minlength=n).astype(float)
            self.assertAlmostEqual((w @ u @ w) / ((w @ n_pos) * (w @ n_neg)), ep.cell_auroc(positive[idx], score[idx]), places=10)

    def test_ties_count_one_half(self):
        positive = np.array([[[True, False]], [[True, False]]])
        score = np.array([[[1.0, 1.0]], [[2.0, 0.0]]], np.float32)
        u, n_pos, n_neg = e2.pair_counts(positive, score)
        self.assertEqual(u.sum(), 0.5 + 1 + 1 + 1)                         # (1 vs 1) tie, (1 vs 0), (2 vs 1), (2 vs 0)


class RunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        b = fx.build(os.path.join(cls.tmp.name, "data"))
        cls.bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)
        cls.grade_of = dict(fx.TRAIN + fx.VAL)
        keras.utils.set_random_seed(7)
        cls.reference = keras.applications.ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True,
                                                        pooling="avg", input_shape=(512, 512, 3), name=pl.BACKBONE_NAME)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_training_targets_come_from_the_partner_and_the_own_map_is_never_opened(self):
        bundle = self.bundle
        partner = e2.derangement(bundle.train_ids)
        e2.verify_derangement(partner, bundle.train_ids, bundle.val_ids)
        entries = [(i, self.grade_of[i]) for i in bundle.train_ids]
        opened = []
        original = ed.load_pathology
        ed.load_pathology = lambda b, i: (opened.append(str(i)), original(b, i))[1]
        try:
            seq = e2.make_epoch_sequence(bundle, entries, 0, 42, 1, partner)
            order = [i for i, _ in ad.epoch_entries(entries, 42, 0, True)]
            for n, image_id in enumerate(order):
                before = len(opened)
                x, (grades, targets) = seq[n]
                self.assertEqual(opened[before:], [partner[image_id]])     # exactly the partner's file, never its own
                self.assertNotEqual(partner[image_id], image_id)
                self.assertEqual(int(grades[0]), self.grade_of[image_id])  # the grade is the image's own
                # the target is the partner's map under the IMAGE's augmentation, as E1 would transform its own map
                import improved_training_data as itd
                own_rgb = ed.load_inputs(bundle, image_id)["rgb"]
                expected = ed.augment_inputs({"rgb": own_rgb, "pathology": cache.from_uint8(original(bundle, partner[image_id]))},
                                             itd.per_image_augmentation_rng(42, 0, image_id))
                np.testing.assert_array_equal(x["rgb"][0], expected["rgb"])
                np.testing.assert_array_equal(targets[0], ed.pool_targets(expected["pathology"], bundle.channels))
                aligned = ed.augment_inputs({"rgb": own_rgb, "pathology": cache.from_uint8(original(bundle, image_id))},
                                            itd.per_image_augmentation_rng(42, 0, image_id))
                np.testing.assert_array_equal(x["rgb"][0], aligned["rgb"])  # same RGB as E1 gave this image
                self.assertFalse(np.array_equal(targets[0], ed.pool_targets(aligned["pathology"], bundle.channels)))
        finally:
            ed.load_pathology = original
        self.assertEqual(sorted(set(opened)), sorted(bundle.train_ids))    # every training map used, as a partner's
        with self.assertRaises(RuntimeError):
            e2.make_epoch_sequence(bundle, entries, 0, 42, 1, {i: i for i in bundle.train_ids})[0]

    def test_one_epoch_run_records_the_derangement_and_validates_on_aligned_targets(self):
        import multiseed_runs as msr
        bundle = self.bundle
        work = os.path.join(self.tmp.name, "e2")
        run_dir = e2.run_dir_for(os.path.join(work, "exp"), bundle.stage4_sha256, 42)
        partner = e2.derangement(bundle.train_ids)
        report = e2.verify_derangement(partner, bundle.train_ids, bundle.val_ids)
        e2.train_seed(run_dir, bundle, 42, PRIOR, self.reference, CW, partner, repo_dir=os.getcwd(),
                      staging_dir=os.path.join(work, "stage"), max_epochs=1, grade_of=self.grade_of, mixed_precision=False,
                      log=lambda *a: None)
        with open(os.path.join(run_dir, "config.json")) as fh:
            cfg = json.load(fh)
        self.assertEqual((cfg["experiment"], cfg["control_of"], cfg["derangement_seed"]), ("E2ShuffledTargets", "E1MultiTask", 20261001))
        self.assertEqual((cfg["derangement_sha256"], cfg["derangement_fixed_points"]), (report["sha256"], 0))
        self.assertTrue(cfg["derangement_shared_by_all_model_seeds"])
        self.assertEqual((cfg["lesion_loss_weight"], cfg["batch_size"], cfg["learning_rate"], cfg["monitor"]), (1.0, 2, 1e-4, "val_QWK"))
        self.assertIn("e2_", os.path.basename(run_dir))
        history = msr.read_history(run_dir)
        self.assertEqual(len(history), 1)
        self.assertIsNotNone(history[0]["val_QWK"])
        results = et.evaluate_run(run_dir, bundle, 42, PRIOR, self.reference, CW, grade_of=self.grade_of, mixed_precision=False)
        payload = e2.write_result(run_dir, results, GATES, report)
        self.assertEqual((payload["experiment"], payload["derangement"]["sha256"]), ("E2ShuffledTargets", report["sha256"]))
        self.assertEqual(e2.seed_state(run_dir), "complete")
        self.assertEqual(results["best"]["checkpoint"]["monitor"], "val_QWK")
        path, sha = ep.e1_best_weights(run_dir)                            # the probe will find E2's BEST weights
        self.assertEqual(sha, results["best"]["checkpoint"]["weights_sha256"])
        with self.assertRaises(RuntimeError):                              # E1's gates must be recorded as passed
            e2.run_sequence(os.path.join(work, "exp2"), bundle, PRIOR, self.reference, CW, repo_dir=os.getcwd(),
                            staging_root=os.path.join(work, "st"), gates={"p_parity": {"PASS": True}})


if __name__ == "__main__":
    unittest.main()
