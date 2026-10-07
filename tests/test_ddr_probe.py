"""CPU tests for ddr_probe. Synthetic masks and features only; the one test that touches the DDR folder reads the
manifest and file names (no test image is decoded, no probe is fitted on DDR). Nothing here is a result."""
import inspect
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import ddr_probe as dp
import e1_probe as ep

REPO = os.path.dirname(os.path.abspath(dp.__file__))
AUDIT = os.path.join(REPO, "datasets", "DDR", "audit")
LESION = os.path.join(REPO, "datasets", "DDR", "raw", "lesion_segmentation")
PRIOR = (0.05, 0.10, 0.08, 0.03)


class TargetTests(unittest.TestCase):
    def test_cell_targets_equal_bruteforce_on_non_divisible_sizes(self):
        rng = np.random.default_rng(0)
        for h, w in ((1728, 2592), (1934, 1956), (1382, 1380), (37, 53), (16, 16)):
            mask = (rng.random((h, w)) < 0.0005).astype(np.uint8) * 255
            t = dp.cell_targets(mask)
            self.assertEqual((t.shape, t.dtype), ((16, 16), np.float32))
            self.assertTrue(set(np.unique(t)) <= {0.0, 1.0})
            np.testing.assert_array_equal(t, dp.cell_targets_bruteforce(mask))

    def test_any_lesion_pixel_makes_its_cell_positive_and_only_that_cell(self):
        mask = np.zeros((1600, 3200), np.uint8)
        mask[999, 3199] = 255                                             # row cell 9 (100 px each), last column cell
        t = dp.cell_targets(mask)
        self.assertEqual(float(t[9, 15]), 1.0)
        self.assertEqual(int(t.sum()), 1)
        self.assertEqual(int(dp.cell_targets(np.zeros((1600, 3200), np.uint8)).sum()), 0)
        mask[:] = 1                                                       # any non-zero value counts
        self.assertEqual(int(dp.cell_targets(mask).sum()), 256)
        self.assertEqual(list(dp.cell_edges(1728)[:3]), [0, 108, 216])
        with self.assertRaises(ValueError):
            dp.cell_targets(np.zeros((8, 100), np.uint8))
        with self.assertRaises(ValueError):
            dp.cell_targets(np.zeros((32, 32, 3), np.uint8))

    def test_locked_design_constants(self):
        self.assertEqual(dp.CLASSES, ("MA", "HE", "EX", "SE"))
        self.assertEqual(dp.EXPECTED_COUNTS, {"train": 383, "val": 148, "test": 225})
        self.assertEqual(dp.EXCLUDED, ("007-5869-300.jpg",))
        self.assertEqual((ep.PROBE["learning_rate"], ep.PROBE["batch_size"], ep.PROBE["epochs"], dp.FIVE_EPOCHS), (1e-3, 16, 5, 5))
        self.assertEqual((dp.N_BOOT, dp.BOOT_SEED, dp.PRIMARY), (2000, 20260927, "e1_minus_p"))
        self.assertEqual(dp.STOPPING["monitor"].split()[0], "validation")
        self.assertEqual(dp.CONTRASTS, {"e1_minus_p": ("e1", "p"), "e2_minus_p": ("e2", "p"), "e1_minus_e2": ("e1", "e2")})


class ManifestTests(unittest.TestCase):
    @unittest.skipUnless(os.path.isdir(AUDIT), "DDR audit folder not present")
    def test_the_pinned_manifest_is_the_756_image_probe_set(self):
        manifest = dp.read_manifest(AUDIT)                                # manifest only: no image is opened
        self.assertEqual({s: len(v) for s, v in manifest.items()}, {"train": 383, "val": 148, "test": 225})
        names = [n for v in manifest.values() for n, _ in v]
        self.assertEqual(len(set(names)), 756)
        self.assertNotIn("007-5869-300.jpg", names)
        self.assertIn("007-2809-100.jpg", [n for n, _ in manifest["test"]])
        if os.path.isdir(LESION):                                         # file names only
            for split, items in manifest.items():
                for name, _ in items[:5]:
                    self.assertTrue(os.path.exists(dp.image_path(LESION, split, name)))
                    for c in dp.CLASSES:
                        self.assertTrue(os.path.exists(dp.mask_path(LESION, split, name, c)))
            self.assertIn(os.sep + "tet" + os.sep, dp.mask_path(LESION, "test", "x.jpg", "MA"))

    def test_a_changed_manifest_is_refused(self):
        if not os.path.isdir(AUDIT):
            self.skipTest("DDR audit folder not present")
        with tempfile.TemporaryDirectory() as tmp:
            text = open(os.path.join(AUDIT, dp.MANIFEST_NAME)).read().replace("007-5869-300.jpg,", "007-5869-300.jpg ,", 1)
            with open(os.path.join(tmp, dp.MANIFEST_NAME), "w", newline="") as fh:
                fh.write(text)
            with self.assertRaises(RuntimeError):
                dp.read_manifest(tmp)

    @unittest.skipUnless(os.path.isdir(AUDIT), "DDR audit folder not present")
    def test_splits_are_disjoint_and_a_changed_image_is_refused(self):
        manifest = dp.read_manifest(AUDIT)
        names = {s: {n for n, _ in v} for s, v in manifest.items()}
        self.assertFalse(names["train"] & names["val"] or names["train"] & names["test"] or names["val"] & names["test"])
        self.assertEqual(len({sha for v in manifest.values() for _, sha in v}), 756)   # no image twice under two names
        with tempfile.TemporaryDirectory() as tmp:                         # same name, other bytes: refused
            name = manifest["train"][0][0]
            os.makedirs(os.path.dirname(dp.image_path(tmp, "train", name)))
            with open(dp.image_path(tmp, "train", name), "wb") as fh:
                fh.write(b"not the pinned image")
            with self.assertRaises(RuntimeError):
                dp.read_manifest(AUDIT, tmp)

    def test_no_idrid_or_aptos_image_can_enter_the_probe(self):
        source = inspect.getsource(dp)
        for forbidden in ("verify_split", "Arch1Bundle", "idrid_root", "B. Disease Grading", "train_images", "DOWNSTREAM_SPLIT"):
            self.assertNotIn(forbidden, source)
        run = inspect.getsource(dp.probe_encoder)                         # the per-encoder step of dp.run
        self.assertEqual(run.count('f["test"]'), 1)                        # the test features are read in one place:
        self.assertIn('evaluate_probe(probe, weights[variant], f["test"]', run)   # the two final predictions
        self.assertNotIn("test", inspect.getsource(dp.train_probes).replace("the test split is not an argument", ""))


class ProbeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        w = rng.normal(0, 1, (24, 4)).astype("float32")
        self.features = rng.normal(0, 1, (96, 4, 4, 24)).astype("float16")
        logits = self.features.astype("float32") @ w - 1.5
        self.targets = (rng.random(logits.shape) < 1 / (1 + np.exp(-logits))).astype("float32")     # binary, like DDR

    def test_the_test_split_is_not_an_input_of_training(self):
        self.assertEqual(list(inspect.signature(dp.train_probes).parameters)[:6],
                         ["seed", "prior", "train_f", "train_t", "val_f", "val_t"])
        self.assertNotIn("test", inspect.signature(dp.train_probes).parameters)

    def test_five_epoch_probe_is_the_63_2_protocol_and_the_converged_probe_follows_the_stopping_rule(self):
        tr_f, tr_t, va_f, va_t = self.features[:64], self.targets[:64], self.features[64:], self.targets[64:]
        reference_logits, reference_curve, _ = ep.train_probe(42, PRIOR, tr_f, tr_t, va_f, va_t)   # §63.2 trainer
        probe, weights, behaviour = dp.train_probes(42, PRIOR, tr_f, tr_t, va_f, va_t)
        five_logits, five = dp.evaluate_probe(probe, weights["five_epoch"], va_f, va_t)
        np.testing.assert_allclose(five_logits, reference_logits, atol=1e-5)          # identical five-epoch probe
        for a, b in zip(behaviour["curve"][:6], reference_curve):
            self.assertAlmostEqual(a["val_loss"], b["val_loss"], places=5)
        self.assertGreaterEqual(behaviour["epochs_run"], 5)
        self.assertLessEqual(behaviour["epochs_run"], dp.STOPPING["max_epochs"])
        losses = [c["val_loss"] for c in behaviour["curve"][1:]]
        best = behaviour["converged_epoch"]
        self.assertAlmostEqual(losses[best - 1], behaviour["val_loss_converged"], places=10)
        self.assertLessEqual(behaviour["val_loss_converged"], behaviour["val_loss_at_five"] + 1e-12)
        if behaviour["stopped_by"] == "patience":                          # no later epoch improved by more than min_delta
            self.assertEqual(behaviour["epochs_run"] - best, dp.STOPPING["patience"]) if best >= 5 else None
            self.assertTrue(all(v >= behaviour["val_loss_converged"] - dp.STOPPING["min_delta"] for v in losses[best:]))
        conv_logits, conv = dp.evaluate_probe(probe, weights["converged"], va_f, va_t)
        self.assertAlmostEqual(conv["loss"], behaviour["val_loss_converged"], places=5)
        self.assertEqual(set(conv["scores"]), {"MA", "HE", "EX", "SE", "mean"})
        again = dp.train_probes(42, PRIOR, tr_f, tr_t, va_f, va_t)[2]
        self.assertEqual(again["converged_epoch"], behaviour["converged_epoch"])   # deterministic

    def test_stopping_rule_with_a_short_cap(self):
        tr_f, tr_t, va_f, va_t = self.features[:64], self.targets[:64], self.features[64:], self.targets[64:]
        stopping = dict(dp.STOPPING, max_epochs=6, patience=50)
        _, _, behaviour = dp.train_probes(7, PRIOR, tr_f, tr_t, va_f, va_t, stopping=stopping)
        self.assertEqual((behaviour["epochs_run"], behaviour["stopped_by"]), (6, "epoch cap"))


class AnalysisTests(unittest.TestCase):
    def _write(self, tmp, e1_gain, e2_gain):
        rng = np.random.default_rng(5)
        n = 90
        targets = (rng.random((n, 16, 16, 4)) < 0.08).astype("float32")
        ids = np.asarray([f"img{i:03d}.jpg" for i in range(n)])
        np.savez_compressed(os.path.join(tmp, "targets.npz"), test_targets=targets, test_ids=ids,
                            train_targets=targets[:4], train_ids=ids[:4], val_targets=targets[:4], val_ids=ids[:4])
        for seed in dp.SEEDS:
            for model, strength in (("p", 1.0), ("e1", 1.0 + e1_gain), ("e2", 1.0 + e2_gain)):
                z = (rng.normal(0, 1, targets.shape) + strength * targets).astype("float32")
                np.savez_compressed(os.path.join(tmp, f"probe_{model}_seed{seed}.npz"), test_ids=ids,
                                    test_logits_five_epoch=z, test_logits_converged=z + 0.01)
        return targets

    def test_criterion_is_on_the_five_epoch_result_and_reports_all_three_contrasts(self):
        with tempfile.TemporaryDirectory() as tmp:
            targets = self._write(tmp, e1_gain=0.6, e2_gain=0.0)
            r = dp.analyse(tmp, n_boot=80)
            five = r["variants"]["five_epoch"]
            self.assertEqual((r["primary_variant"], r["test_images"]), ("five_epoch", 90))
            self.assertEqual(set(five["mean"]) & set(dp.CONTRASTS), set(dp.CONTRASTS))
            self.assertTrue(five["mean"]["e1_minus_p"]["ci_excludes_zero"] and r["criterion_met"])
            self.assertEqual(five["mean"]["e1_minus_p"]["positive_seeds"], 3)
            self.assertGreater(five["mean"]["e1_minus_e2"]["mean"], 0)
            self.assertEqual(set(five["mean"]["e1_minus_p"]["per_class"]), set(dp.CLASSES))
            with np.load(os.path.join(tmp, "probe_p_seed42.npz")) as d:
                self.assertAlmostEqual(five["per_seed"][42]["p"]["mean"], ep.probe_scores(targets, d["test_logits_five_epoch"])["mean"], places=12)
            self.assertIn("converged", r["variants"])
            with open(os.path.join(tmp, "ddr_probe_result.json")) as fh:
                saved = json.load(fh)
            self.assertEqual(saved["criterion_met"], True)
            self.assertEqual(set(saved), {"test_images", "n_boot", "boot_seed", "bootstrap", "variants", "primary_variant",
                                          "criterion", "criterion_met"})
            self.assertEqual(set(saved["variants"]), set(dp.VARIANTS))
            self.assertEqual(set(saved["variants"]["five_epoch"]), {"per_seed", "mean", "criterion_met"})
            self.assertEqual(set(saved["variants"]["five_epoch"]["per_seed"]["42"]), set(dp.MODELS) | set(dp.CONTRASTS))
            again = dp.analyse(tmp, n_boot=80)                             # same seed, same resamples, same numbers
            self.assertEqual(json.loads(json.dumps(again, default=float)), json.loads(json.dumps(r, default=float)))
            other = dp.analyse(tmp, n_boot=80, boot_seed=1)
            self.assertNotEqual(other["variants"]["five_epoch"]["mean"]["e1_minus_p"]["ci"], five["mean"]["e1_minus_p"]["ci"])
            self.assertEqual(other["variants"]["five_epoch"]["mean"]["e1_minus_p"]["mean"], five["mean"]["e1_minus_p"]["mean"])

    def test_run_metadata_is_recorded_per_invocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dp.environment(REPO)
            for key in ("utc", "git_commit", "python", "platform", "numpy", "tensorflow", "keras", "gpus"):
                self.assertIn(key, env)
            dp.record_invocation(tmp, dict(env, seeds=[42]))
            runs = dp.record_invocation(tmp, dict(env, seeds=[123]))
            self.assertEqual([r["seeds"] for r in runs], [[42], [123]])

    def test_criterion_fails_when_e1_is_not_better(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, e1_gain=0.0, e2_gain=0.0)
            self.assertFalse(dp.analyse(tmp, n_boot=80)["criterion_met"])

    def test_bootstrap_weights_reproduce_the_auroc_of_a_resampled_test_set(self):
        import e2_control as e2
        rng = np.random.default_rng(1)
        pos = rng.random((30, 16, 16)) < 0.1
        z = (rng.normal(size=pos.shape) + pos).astype("float32")
        idx = np.random.default_rng(dp.BOOT_SEED).integers(0, 30, size=(4, 30))
        w = np.stack([np.bincount(i, minlength=30) for i in idx]).astype(float)
        u, a, b = e2.pair_counts(pos, z)
        fast = np.einsum("bi,ij,bj->b", w, u, w) / ((w @ a) * (w @ b))
        for k in range(4):
            self.assertAlmostEqual(fast[k], ep.cell_auroc(pos[idx[k]], z[idx[k]]), places=10)


if __name__ == "__main__":
    unittest.main()
