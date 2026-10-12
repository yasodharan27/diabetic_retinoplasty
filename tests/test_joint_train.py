"""CPU tests for joint_train. The data rules are checked on the real split manifests (labels and names only, no
image); the run mechanics on random ConvNeXt weights and synthetic 64-pixel frames. Nothing here is a result."""
import csv
import inspect
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import arch1_train as at
import eyepacs_adaptation_data as ea
import eyepacs_adaptation_train as eat
import joint_train as jt
import pipeline_v2_config as v2cfg
import weighted_corn
from test_eyepacs_adaptation_train import REF_ARRAYS, SMALL, FakeFrames

REPO = os.path.dirname(os.path.abspath(jt.__file__))


def real_parts():
    empty = set(v2cfg.APTOS_EMPTY_FOV_IDS)
    with open(os.path.join(REPO, "dataset_splits", "aptos2019_train_val_split.csv"), newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["id_code"] not in empty]
    aptos = {s: [(r["id_code"], int(r["diagnosis"])) for r in rows if r["split"] == s] for s in ("train", "val")}
    eyepacs_rows = ea.read_split(os.path.join(REPO, "dataset_splits"), ea.MANIFEST_SHA256)
    eyepacs = {s: [(r["image"], r["grade"]) for r in eyepacs_rows if r["split"] == s] for s in ("train", "val")}
    return aptos["train"], aptos["val"], eyepacs["train"], eyepacs["val"]


class DataRuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parts = real_parts()

    def test_the_parts_are_the_permitted_ones_and_validation_cannot_reach_training(self):
        aptos_train, aptos_val, eyepacs_train, eyepacs_val = self.parts
        self.assertTrue(jt.check_data(*self.parts))
        self.assertEqual((len(aptos_train), len(aptos_val), len(eyepacs_train), len(eyepacs_val)), (2921, 730, 31610, 3512))
        with self.assertRaises(jt.JointError):                          # a validation image moved into training
            jt.check_data(aptos_train[:-1] + aptos_val[:1], aptos_val, eyepacs_train, eyepacs_val)
        with self.assertRaises(jt.JointError):
            jt.check_data(aptos_train, aptos_val, eyepacs_train[:-1] + eyepacs_val[:1], eyepacs_val)
        with self.assertRaises(jt.JointError):                          # the wrong training part
            jt.check_data(aptos_train[:100], aptos_val, eyepacs_train, eyepacs_val)
        validation = {i for i, _ in aptos_val} | {i for i, _ in eyepacs_val}
        rotation = jt.eyepacs_rotation(eyepacs_train, 42)
        for epoch in (0, 1, 5, 49):
            self.assertFalse({i for i, _ in jt.epoch_entries(aptos_train, rotation, epoch)} & validation)

    def test_every_epoch_is_one_aptos_to_two_eyepacs(self):
        aptos_train, _, eyepacs_train, _ = self.parts
        rotation = jt.eyepacs_rotation(eyepacs_train, 42)
        aptos_names, eyepacs_names = {i for i, _ in aptos_train}, {i for i, _ in eyepacs_train}
        for epoch in (0, 3, 17):
            names = [i for i, _ in jt.epoch_entries(aptos_train, rotation, epoch)]
            self.assertEqual(len(names), 2921 * 3)
            self.assertEqual(len(set(names)), len(names))               # no image twice in an epoch
            self.assertEqual((sum(n in aptos_names for n in names), sum(n in eyepacs_names for n in names)), (2921, 5842))
        self.assertEqual((jt.EYEPACS_PER_APTOS, jt.PROTOCOL["batch_size"]), (2, 16))

    def test_the_rotation_is_seeded_and_does_not_depend_on_the_epoch_or_on_a_resume(self):
        aptos_train, _, eyepacs_train, _ = self.parts
        a = jt.eyepacs_rotation(eyepacs_train, 42)
        self.assertEqual(a, jt.eyepacs_rotation(list(reversed(eyepacs_train)), 42))        # not the order it is given in
        self.assertNotEqual(a, jt.eyepacs_rotation(eyepacs_train, 123))
        self.assertEqual(sorted(a), sorted((str(i), int(g)) for i, g in eyepacs_train))
        # an epoch is a pure function of (seed, epoch): epoch 7 is the same whether or not epochs 0-6 were built first
        fresh = jt.epoch_entries(aptos_train, jt.eyepacs_rotation(eyepacs_train, 42), 7)
        for e in range(7):
            jt.epoch_entries(aptos_train, a, e)
        self.assertEqual(jt.epoch_entries(aptos_train, a, 7), fresh)
        # consecutive epochs take consecutive slices: nothing repeats until the rotation is exhausted (5 full epochs)
        seen = [i for e in range(5) for i, _ in jt.epoch_entries(aptos_train, a, e)[2921:]]
        self.assertEqual(len(set(seen)), 5 * 5842)
        wrapped = {i for e in range(6) for i, _ in jt.epoch_entries(aptos_train, a, e)[2921:]}
        self.assertEqual(len(wrapped), 31610)                           # by the sixth epoch every EyePACS training image was used
        self.assertNotIn("epoch", inspect.signature(jt.eyepacs_rotation).parameters)

    def test_class_weights_are_ps_formula_on_the_mixture_that_is_actually_sampled(self):
        aptos_train, _, eyepacs_train, _ = self.parts
        self.assertEqual(jt.grade_counts(aptos_train).tolist(), [1444, 296, 791, 154, 236])
        self.assertEqual(jt.grade_counts(eyepacs_train).tolist(), [23209, 2209, 4768, 782, 642])
        for seed in jt.SEEDS:
            expected, weights, sampled = jt.class_weights(aptos_train, eyepacs_train, seed)
            self.assertTrue(np.allclose(weights, jt.APPROVED_CLASS_WEIGHTS, atol=5e-5))
            self.assertTrue(np.allclose(weights, weighted_corn.class_weights_from_counts(expected)))
            self.assertAlmostEqual(float(np.sum(np.asarray(expected) / np.sum(expected) * np.asarray(weights))), 1.0, places=9)   # the project's normalisation
            self.assertLess(sampled["max_abs_difference"], jt.WEIGHT_TOLERANCE)
            self.assertAlmostEqual(sum(sampled["mean_counts_per_epoch"]), 8763.0, places=6)
        self.assertAlmostEqual(sum(expected), 8763.0, places=6)
        self.assertEqual(weighted_corn.DEFAULT_WEIGHT_POWER, 0.5)
        with self.assertRaises(jt.JointError):                          # another mixture does not get the recorded weights
            jt.class_weights(aptos_train[:1000], eyepacs_train)
        # between P's weights (APTOS only) and the EyePACS-only weights, as a mixture must be
        p, e = weighted_corn.class_weights_from_counts(jt.grade_counts(aptos_train)), np.asarray(eat.APPROVED_CLASS_WEIGHTS)
        self.assertTrue(all(min(a, b) - 1e-3 <= w <= max(a, b) + 1e-3 for w, a, b in zip(weights, p, e)))


class SelectionTests(unittest.TestCase):
    def test_both_sets_have_equal_influence_whatever_their_sizes(self):
        self.assertEqual(jt.selection_metric(0.9, 0.5), 0.7)
        self.assertEqual(jt.selection_metric(0.9, 0.5), jt.selection_metric(0.5, 0.9))
        self.assertAlmostEqual(jt.selection_metric(0.91, 0.80) - jt.selection_metric(0.90, 0.80), jt.selection_metric(0.90, 0.81) - jt.selection_metric(0.90, 0.80))
        self.assertIn("both full", jt.SELECTION)
        self.assertNotIn("subset", jt.SELECTION)

    def test_the_callback_scores_both_full_sets_and_replaces_the_monitored_value(self):
        scored = []

        def fake_qwk(model, frames, entries):
            scored.append(len(entries))
            return 0.9 if len(entries) == 730 else 0.6
        aptos_val, eyepacs_val = [(f"a{k}", 0) for k in range(730)], [(f"{k}_left.jpeg", 0) for k in range(3512)]
        with tempfile.TemporaryDirectory() as tmp:
            callback = jt.dual_validation_callback(tmp, None, aptos_val, eyepacs_val, fake_qwk)
            logs = {"val_QWK": 0.88, "val_loss": 1.0}
            callback.on_epoch_end(3, logs)
            self.assertEqual(scored, [730, 3512])                       # the full sets, every epoch
            self.assertEqual((logs["val_QWK"], logs["val_QWK_aptos"], logs["val_QWK_eyepacs"], logs["val_QWK_keras_aptos"]), (0.75, 0.9, 0.6, 0.88))
            written = json.load(open(os.path.join(tmp, "dual_validation", "epoch_0003.json")))
            self.assertEqual((written["epoch"], written["selection_metric"]), (3, 0.75))
            with self.assertRaises(jt.JointError):
                callback.on_epoch_end(4, None)

    def test_the_continuation_rule(self):
        cases = [((-0.004, -0.015), True), ((-0.010, -0.020), True), ((-0.0101, 0.0), False), ((0.0, -0.0201), False), ((0.02, 0.05), True)]
        for (a, e), expected in cases:
            out = jt.continuation({"difference": a}, {"difference": e})
            self.assertEqual(out["continue"], expected, (a, e))
        self.assertEqual((jt.APTOS_MARGIN, jt.EYEPACS_MARGIN), (0.01, 0.02))


class JointFramesTests(unittest.TestCase):
    def test_frames_are_dispatched_by_dataset_and_unknown_images_are_refused(self):
        class Source:
            def __init__(self, value):
                self.value = value

            def frame(self, image):
                return np.full((2, 2, 3), self.value, np.float32)
        frames = jt.JointFrames(Source(0.1), Source(0.9), ["abc"], ["1_left.jpeg"])
        self.assertEqual((float(frames.frame("abc")[0, 0, 0]), float(frames.frame("1_left.jpeg")[0, 0, 0])), (np.float32(0.1), np.float32(0.9)))
        with self.assertRaises(jt.JointError):
            frames.frame("2_left.jpeg")                                  # e.g. a held-out test image: not reachable
        with self.assertRaises(jt.JointError):
            jt.JointFrames(Source(0), Source(1), ["x"], ["x"])


class AptosFake:
    def frame(self, image):
        return np.random.default_rng(int(image[1:]) + 1000).random((SMALL, SMALL, 3)).astype(np.float32)


class RunTests(unittest.TestCase):
    """Two epochs of the real loop on synthetic frames, then a resume to a third: mechanics only."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        eyepacs = FakeFrames(n=(20, 10))
        aptos_train = [(f"a{k}", k % 5) for k in range(8)]
        aptos_val = [(f"a{k}", k % 5) for k in range(100, 110)]
        cls.parts = (aptos_train, aptos_val, eyepacs.entries("train"), eyepacs.entries("val"))
        cls.frames = jt.JointFrames(AptosFake(), eyepacs, [i for i, _ in aptos_train + aptos_val], [i for i, _ in eyepacs.entries("train") + eyepacs.entries("val")])
        cls.weights = list(jt.APPROVED_CLASS_WEIGHTS)
        cls.run_dir = os.path.join(cls.tmp.name, "exp", jt.EXPERIMENT, "j_seed42")
        cls.mapping = jt.run_mapping(42, cls.weights, [1.0] * 5, {"fingerprint": "a" * 8}, {"fingerprint": "e" * 8}, repo_dir=REPO)
        cls.kw = dict(repo_dir=REPO, staging_dir=os.path.join(cls.tmp.name, "staging"), log=lambda *a: None, mixed_precision=False,
                      image_size=SMALL, workers=1)
        jt.train_seed(cls.run_dir, 42, cls.frames, cls.parts, REF_ARRAYS, cls.weights, cls.mapping, max_epochs=2, **cls.kw)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_1_every_epoch_scored_both_sets_and_the_monitor_is_their_mean(self):
        import multiseed_runs as msr
        files = sorted(os.listdir(os.path.join(self.run_dir, "dual_validation")))
        self.assertEqual(files, ["epoch_0000.json", "epoch_0001.json"])
        scores = [json.load(open(os.path.join(self.run_dir, "dual_validation", f))) for f in files]
        for s in scores:
            self.assertAlmostEqual(s["selection_metric"], 0.5 * (s["val_QWK_aptos"] + s["val_QWK_eyepacs"]))
        history = msr.read_epoch_history(self.run_dir) if hasattr(msr, "read_epoch_history") else None
        config = json.load(open(os.path.join(self.run_dir, "config.json")))
        self.assertEqual((config["batch_size"], config["checkpoint_selection"], config["eyepacs_order"]), (16, jt.SELECTION, jt.ORDER_TAG))
        _, pointer = msr.read_best(self.run_dir)
        best = max(scores, key=lambda s: (s["selection_metric"], -s["epoch"]))
        self.assertEqual(int(pointer["epoch"]), best["epoch"])                # BEST is the first epoch with the highest mean
        self.assertAlmostEqual(float(pointer["val_QWK"]), best["selection_metric"], places=6)
        self.assertTrue(history is None or len(history) == 2)

    def test_2_best_is_evaluated_on_both_sets_with_every_reported_quantity(self):
        build = lambda: eat.build_compiled_model(42, REF_ARRAYS, self.weights, jt.PROTOCOL, False, SMALL)      # noqa: E731
        out = jt.evaluate_best(self.run_dir, build, self.frames, self.parts[1], self.parts[3])
        self.assertEqual(sorted(out["sets"]), ["aptos_val", "eyepacs_val"])
        for name, n in (("aptos_val", 10), ("eyepacs_val", 10)):
            m = out["sets"][name]
            self.assertEqual(sorted(m), sorted(["qwk", "auroc_cuts", "recall_per_grade", "confusion_matrix", "referable_sensitivity",
                                                "referable_specificity", "ece_top_label", "mae"]))
            table = os.path.join(self.run_dir, "metrics", f"per_sample_best_{name}.csv")
            self.assertEqual(sum(1 for _ in open(table)) - 1, n)
        self.assertEqual(len(out["checkpoint"]["weights_sha256"]), 64)
        same = jt.compare(os.path.join(self.run_dir, "metrics", "per_sample_best_aptos_val.csv"),
                          os.path.join(self.run_dir, "metrics", "per_sample_best_aptos_val.csv"))
        self.assertEqual((same["difference"], same["ci_95"]), (0.0, [0.0, 0.0]))

    def test_3_a_resumed_run_continues_with_the_next_epoch_and_keeps_the_earlier_ones(self):
        before = {f: open(os.path.join(self.run_dir, "dual_validation", f)).read() for f in ("epoch_0000.json", "epoch_0001.json")}
        os.remove(os.path.join(self.run_dir, "checkpoints", "stop_decision.json"))      # the two-epoch cap of the first call; an interrupted run has none
        jt.train_seed(self.run_dir, 42, self.frames, self.parts, REF_ARRAYS, self.weights, self.mapping, max_epochs=3, **self.kw)
        files = sorted(os.listdir(os.path.join(self.run_dir, "dual_validation")))
        self.assertEqual(files, ["epoch_0000.json", "epoch_0001.json", "epoch_0002.json"])
        self.assertEqual({f: open(os.path.join(self.run_dir, "dual_validation", f)).read() for f in before}, before)   # earlier epochs untouched

    def test_4_later_seeds_wait_for_seed_42_and_its_rule(self):
        root = os.path.join(self.tmp.name, "gate")
        for arm in ("direct", "crop"):                                       # no seed-42 result yet, in either arm
            with self.assertRaises(jt.JointError):
                jt.run_seed(123, root, self.frames, self.parts, REF_ARRAYS, {}, {}, repo_dir=REPO, staging_root=self.tmp.name, arm=arm)
        os.makedirs(jt.run_dir_for(root, 42, "crop"))
        json.dump({"arm": "crop"}, open(os.path.join(jt.run_dir_for(root, 42, "crop"), "result.json"), "w"))
        with self.assertRaises(jt.JointError):                               # seed 2026 does not jump over seed 123
            jt.run_seed(2026, root, self.frames, self.parts, REF_ARRAYS, {}, {}, repo_dir=REPO, staging_root=self.tmp.name, arm="crop")
        first = jt.run_dir_for(root, 42)
        os.makedirs(first)
        json.dump({"continuation": {"continue": False}}, open(os.path.join(first, "result.json"), "w"))
        with self.assertRaises(jt.JointError):                               # the rule was not met
            jt.run_seed(2026, root, self.frames, self.parts, REF_ARRAYS, {}, {}, repo_dir=REPO, staging_root=self.tmp.name)
        self.assertEqual(jt.run_seed(42, root, self.frames, self.parts, REF_ARRAYS, {}, {}, repo_dir=REPO, staging_root=self.tmp.name),
                         {"continuation": {"continue": False}})              # a completed seed is returned as recorded
        with self.assertRaises(ValueError):
            jt.run_dir_for(root, 7)


class ArmTests(unittest.TestCase):
    def test_the_direct_arm_keeps_the_configuration_seed_42_was_started_with(self):
        args = (42, jt.APPROVED_CLASS_WEIGHTS, [5733.4, 704.3, 1672.2, 298.5, 354.7], {"fingerprint": "a"}, {"fingerprint": "e"})
        direct = jt.run_mapping(*args, repo_dir=REPO)
        self.assertEqual(at.config_hash(direct), "5252d2e9960b1d3097d07eef713e267749d01b3d743078a2a7fa2b0c9d003886")     # as committed in 64a4e74
        self.assertEqual(direct, jt.run_mapping(*args, repo_dir=REPO, arm="direct"))
        self.assertNotIn("arm", direct)
        self.assertEqual(jt.run_dir_for("root", 42), jt.run_dir_for("root", 42, "direct"))
        self.assertTrue(jt.run_dir_for("root", 42).endswith("JointAptosEyePACS/j_seed42"))

    def test_the_cropped_arm_differs_in_the_frames_only_and_has_its_own_directory(self):
        import crop_frames as cf
        import geometry_diagnostic as gd
        args = (42, jt.APPROVED_CLASS_WEIGHTS, [1.0] * 5, {"fingerprint": "a"}, {"fingerprint": "e"})
        direct, crop = jt.run_mapping(*args, repo_dir=REPO), jt.run_mapping(*args, repo_dir=REPO, arm="crop")
        changed = {k for k in crop if crop[k] != direct.get(k)}
        self.assertEqual(changed, {"arm", "preprocessing", "crop_rule", "crop_frames_version"})
        self.assertEqual(crop["crop_rule"], {"gray_threshold": gd.GRAY_THRESHOLD, "line_fraction": gd.LINE_FRACTION, "min_side_fraction": gd.MIN_SIDE_FRACTION})
        self.assertEqual((cf.RULE["gray_threshold"], cf.RULE["line_fraction"], cf.RULE["min_side_fraction"]), (12, 0.02, 0.10))    # the rule fixed in record 89
        self.assertTrue(jt.run_dir_for("root", 42, "crop").endswith("JointAptosEyePACS/jc_seed42"))
        self.assertNotEqual(at.config_hash(direct), at.config_hash(crop))
        with self.assertRaises(ValueError):
            jt.run_dir_for("root", 42, "other")
        with self.assertRaises(ValueError):
            jt.run_mapping(*args, repo_dir=REPO, arm="other")
        self.assertEqual({a: v["continuation_rule"] for a, v in jt.ARMS.items()}, {"direct": True, "crop": False})


class CropCacheTests(unittest.TestCase):
    """The cropped-frame cache on fabricated images: build, resume, read, refuse."""

    def setUp(self):
        import cv2
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.raw = os.path.join(self.tmp.name, "raw")
        os.makedirs(self.raw)
        rng = np.random.default_rng(0)
        self.plan = {"train": [], "val": []}
        for k in range(5):
            image = np.zeros((120, 180, 3), np.uint8)
            image[10:110, 40 + k:140 + k] = rng.integers(60, 200, (100, 100, 3))      # a lit block with dark margins
            name = f"{k}_left.jpeg"
            cv2.imwrite(os.path.join(self.raw, name), image)
            self.plan["train" if k < 3 else "val"].append((name, k % 5, os.path.join(self.raw, name)))
        self.cache = os.path.join(self.tmp.name, "cache")

    def tearDown(self):
        self.tmp.cleanup()

    def test_build_resume_and_read(self):
        import crop_frames as cf
        import geometry_diagnostic as gd
        index = cf.build(self.plan, self.cache, "toy", workers=1, shard_size=2, log=lambda *a: None)
        self.assertEqual(sorted(index["shards"]), ["frames_train_000.npy", "frames_train_001.npy", "frames_val_000.npy"])
        self.assertEqual(index["frames"], {"train": 3, "val": 2})
        box = index["shards"]["frames_train_000.npy"]["boxes"][0]
        self.assertTrue(box[2] > 20 and box[3] < 160 and box[0] > 0)                # the dark margins were cut
        frames = cf.CropFrames(self.cache, {s: [(i, g) for i, g, _ in v] for s, v in self.plan.items()})
        self.assertEqual(frames.entries("val"), [("3_left.jpeg", 3), ("4_left.jpeg", 4)])
        frame = frames.frame("0_left.jpeg")
        self.assertEqual((frame.shape, frame.dtype), ((512, 512, 3), np.float32))
        direct, crop, _, _ = gd.two_frames(self.plan["train"][0][2])
        self.assertEqual(float(np.abs(frame - crop.astype(np.float16).astype(np.float32)).max()), 0.0)      # the diagnostic's cropped frame
        self.assertGreater(float(np.abs(frame - direct).max()), 0.05)
        self.assertGreater(float(frame.mean()), float(direct.mean()))                # less dark margin in the frame
        stamp = os.path.getmtime(os.path.join(self.cache, "frames_train_000.npy"))
        os.remove(os.path.join(self.cache, "frames_val_000.npy"))                   # an interrupted build: one shard missing
        again = cf.build(self.plan, self.cache, "toy", workers=1, shard_size=2, log=lambda *a: None)
        self.assertEqual(os.path.getmtime(os.path.join(self.cache, "frames_train_000.npy")), stamp)       # finished shards are kept
        self.assertEqual(again["shards"]["frames_val_000.npy"]["sha256"], index["shards"]["frames_val_000.npy"]["sha256"])
        self.assertEqual(cf.summary(again)["left_uncut"], 0)

    def test_a_wrong_or_altered_cache_is_refused(self):
        import crop_frames as cf
        cf.build(self.plan, self.cache, "toy", workers=1, shard_size=2, log=lambda *a: None)
        expected = {s: [(i, g) for i, g, _ in v] for s, v in self.plan.items()}
        with self.assertRaises(cf.CropCacheError):                                   # other images than the run expects
            cf.CropFrames(self.cache, {"train": expected["train"][:2], "val": expected["val"]})
        with self.assertRaises(cf.CropCacheError):
            cf.CropFrames(self.cache, {"train": list(reversed(expected["train"])), "val": expected["val"]})
        block = np.load(os.path.join(self.cache, "frames_train_000.npy"), mmap_mode="r+")
        block[0, 0, 0, 0] = block[0, 0, 0, 0] + np.float16(0.25)
        block.flush()
        del block
        with self.assertRaises(cf.CropCacheError):
            cf.CropFrames(self.cache, expected)
        index_path = os.path.join(self.cache, "index.json")
        index = json.load(open(index_path))
        index["rule"]["gray_threshold"] = 20
        json.dump(index, open(index_path, "w"))
        with self.assertRaises(cf.CropCacheError):                                   # built with another rule
            cf.CropFrames(self.cache, expected, check_hashes=False)


class SequenceTests(unittest.TestCase):
    """run_arm trains one seed after another; the run itself is replaced by stand-ins (mechanics only)."""

    def setUp(self):
        import multiseed_runs as msr
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.order, self.differences = [], {"aptos": 0.0, "eyepacs": 0.0}
        self._orig = (jt.check_data, jt.class_weights, jt.compare)
        jt.check_data = lambda *parts: True
        jt.class_weights = lambda a, e, seed=None, epochs=None: ([1.0] * 5, list(jt.APPROVED_CLASS_WEIGHTS), {"weights": list(jt.APPROVED_CLASS_WEIGHTS)})
        jt.compare = lambda joint, reference: {"difference": self.differences["aptos" if "aptos" in joint else "eyepacs"], "ci_95": [0.0, 0.0],
                                               "joint_qwk": 0.9, "reference_qwk": 0.9}

        def train(run_dir, seed, frames, parts, arrays, weights, mapping, **kw):
            self.order.append((kw["arm"], seed))
            os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
            msr.write_stop_decision(run_dir, 1, "epoch_cap")
        self.kw = dict(repo_dir=REPO, staging_root=self.tmp.name, log=lambda *a: None, train_fn=train,
                       evaluate_fn=lambda *a: {"checkpoint": {}, "sets": {}})
        self.parts = ([("a0", 0)], [("a1", 0)], [("1_left.jpeg", 0)], [("2_left.jpeg", 0)])

    def tearDown(self):
        jt.check_data, jt.class_weights, jt.compare = self._orig
        self.tmp.cleanup()

    def _arm(self, arm, root):
        return jt.run_arm(arm, root, None, self.parts, lambda: None, {"aptos": {}, "eyepacs": {}},
                          lambda seed: {"p_aptos": "p", "eyepacs_only": "e"}, **self.kw)

    def test_seeds_run_in_order_one_after_another_and_finished_seeds_are_kept(self):
        root = os.path.join(self.tmp.name, "a")
        results = self._arm("crop", root)
        self.assertEqual(self.order, [("crop", 42), ("crop", 123), ("crop", 2026)])
        self.assertEqual(sorted(results), [42, 123, 2026])
        self.assertTrue(all(r["arm"] == "crop" and "continuation" not in r for r in results.values()))
        self._arm("crop", root)                                               # a second call trains nothing again
        self.assertEqual(len(self.order), 3)
        results = self._arm("direct", root)                                   # the direct arm: rule met, so all three
        self.assertEqual(self.order[3:], [("direct", 42), ("direct", 123), ("direct", 2026)])
        self.assertTrue(results[42]["continuation"]["continue"])

    def test_the_direct_arm_stops_after_seed_42_when_its_rule_is_not_met(self):
        self.differences["eyepacs"] = -0.05
        results = self._arm("direct", os.path.join(self.tmp.name, "b"))
        self.assertEqual((self.order, sorted(results)), ([("direct", 42)], [42]))
        self.assertFalse(results[42]["continuation"]["continue"])
        with self.assertRaises(ValueError):
            self._arm("other", self.tmp.name)


class ModuleTests(unittest.TestCase):
    def test_only_the_data_differs_from_p_and_no_external_or_test_set_is_reachable(self):
        self.assertEqual(jt.PROTOCOL, dict(at.P_PROTOCOL, batch_size=16))
        self.assertEqual(jt.PROTOCOL, eat.PROTOCOL)
        self.assertEqual(jt.SEEDS, (42, 123, 2026))
        source = open(os.path.join(REPO, "joint_train.py"), encoding="utf-8").read()
        code = source.split('"""', 2)[2]
        for forbidden in ("idrid", "ddr", "heldout", "read_test", "TEST_MANIFEST", "evaluate_heldout_test"):
            self.assertNotIn(forbidden, code.lower() if forbidden.islower() else code, forbidden)
        for reused in ("eat.training_loop(", "eat.build_compiled_model(", "eat.make_epoch_sequence(", "eat.predict_logits(",
                       "weighted_corn.class_weights_from_counts(", "corn.decode_logits("):
            self.assertIn(reused, source, reused)
        mapping = jt.run_mapping(42, jt.APPROVED_CLASS_WEIGHTS, [1] * 5, {}, {}, repo_dir=REPO)
        self.assertIn("ImageNet", mapping["encoder_initialisation"])
        self.assertIn("no crop", mapping["preprocessing"])


if __name__ == "__main__":
    unittest.main()
