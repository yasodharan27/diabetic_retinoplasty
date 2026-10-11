"""CPU tests for idrid_batch2_eval. Fabricated run directories, synthetic tables and fake models only: no EP model
is loaded, no IDRiD image is decoded and nothing here is an IDRiD result."""
import copy
import inspect
import json
import os
import shutil
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import arch1_posthoc as ph
import arch1_train as at
import idrid_batch1_eval as b1
import idrid_batch2_eval as e2
import idrid_grading_eval as ig

REAL_IDRID = os.path.join(e2.HERE, "datasets", "IDRiD", "grading", "raw")


class FakeNet:
    """Stands in for a model: deterministic logits from the mean of each RGB frame. `kind` 'P' takes P's three
    inputs and returns one array; 'E1' takes {'rgb'} and returns [grading logits, lesion logits]."""

    def __init__(self, kind, shift):
        self.kind, self.shift, self.seen = kind, float(shift), []

    def predict_on_batch(self, x):
        self.seen.append(sorted(x))
        rgb = x["stage5_input"][..., :3] if self.kind == "P" else x["rgb"]
        m = rgb.reshape(len(rgb), -1).mean(axis=1, dtype=np.float64)
        base = (m - 0.5) * 40.0 + self.shift
        z = np.stack([base + 2, base, base - 2, base - 4], axis=1).astype(np.float32)
        return z if self.kind == "P" else [z, None]


def fake_models(shift=0.0):
    keys = [(m, s) for m in e2.RUN_MODELS for s in e2.SEEDS]
    return {key: FakeNet(e2.ARMS[key[0]], shift + 0.3 * i) for i, key in enumerate(keys)}


def synthetic(n=103, seed=0, shifts=None):
    shifts = shifts or {}
    rng = np.random.default_rng(seed)
    grades = np.array(([0] * 34 + [1] * 5 + [2] * 32 + [3] * 19 + [4] * 13)[:n])
    for excluded, donor in ((87, 40), (88, 41), (90, 42)):          # IDRiD_088 / 089 / 091 are grade 2, as in the real set
        grades[excluded], grades[donor] = grades[donor], grades[excluded]
    ids = [f"IDRiD_{i + 1:03d}" for i in range(n)]
    tables = {}
    for s in e2.SEEDS:
        base = (grades[:, None] - np.arange(4)[None, :] - 0.5) * 2.0 + rng.normal(0, 1.2, (n, 4))
        tables[s] = {m: ph.table_arrays(pd.DataFrame(at.metrics_from_logits(ids, grades, base + shifts.get(m, 0.0))[1])) for m in e2.MODELS}
    return ids, grades, tables


def fake_drive(tmp, protocol, parent):
    """A drive root holding every pinned file of `protocol` (hashes rewritten to the stand-in files)."""
    root = lambda rel: os.path.join(tmp, *rel.split("/"))
    adapted = protocol["adapted_encoder"]
    os.makedirs(os.path.dirname(root(adapted["path"])), exist_ok=True)
    with open(root(adapted["path"]), "wb") as fh:
        fh.write(b"adapted encoder")
    adapted["sha256"] = ig.sha256_file(root(adapted["path"]))
    with open(os.path.join(root(adapted["run_dir"]), "frozen_checkpoint.json"), "w") as fh:
        json.dump({"sha256": adapted["sha256"]}, fh)
    for name, entry in protocol["checkpoints"].items():
        path, run_dir = root(entry["path"]), root(entry["run_dir"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(name.encode())
        entry["sha256"] = ig.sha256_file(path)
        files = {"config.json": {"experiment": protocol["experiment"], "arm": entry["arm"], "seed": entry["seed"],
                                 "model_type": entry["model_type"], "adapted_encoder_sha256": adapted["sha256"],
                                 "checkpoint_selection": "APTOS validation QWK (as P)", "lesion_prior": [0.07, 0.11, 0.11, 0.05]},
                 "frozen_checkpoint.json": {"sha256": entry["sha256"], "weights": entry["path"][len(entry["run_dir"]) + 1:],
                                            "best_epoch_index": entry["best_epoch_index"]},
                 "initialization.json": {"arm": entry["arm"], "encoder_is_adapted": True, "encoder_is_imagenet": False,
                                         "adapted_encoder_sha256": adapted["sha256"], "fresh_corn_head": True},
                 "result.json": {"best": {"checkpoint": {"weights_sha256": entry["sha256"], "monitor": "val_QWK"}}}}
        for filename, payload in files.items():
            with open(os.path.join(run_dir, filename), "w") as fh:
                json.dump(payload, fh)
        with open(os.path.join(run_dir, "checkpoints", "best.json"), "w") as fh:
            json.dump({"active": entry["best_slot"], "epoch": entry["best_epoch_index"]}, fh)
    pretrained = parent["checkpoints"]["convnext_pretrained"]
    os.makedirs(os.path.dirname(root(pretrained["path"])), exist_ok=True)
    with open(root(pretrained["path"]), "wb") as fh:
        fh.write(b"imagenet")
    pretrained["sha256"] = ig.sha256_file(root(pretrained["path"]))
    for s in e2.SEEDS:
        entry = protocol["stored_p_tables"][f"p_seed{s}"]
        os.makedirs(os.path.dirname(root(entry["path"])), exist_ok=True)
        with open(root(entry["path"]), "w") as fh:
            fh.write(f"p{s}")
        entry["sha256"] = ig.sha256_file(root(entry["path"]))
    return protocol, parent


class ProtocolTests(unittest.TestCase):
    def test_protocol_pins_the_locked_design(self):
        protocol, sha, parent = e2.load_protocol()
        b1_protocol, _, b1_parent = b1.load_protocol()
        self.assertEqual(protocol["parent_protocol"], b1_protocol["parent_protocol"])     # the same locked IDRiD protocol as batch 1
        self.assertEqual(parent, b1_parent)
        self.assertEqual(parent["dataset"]["primary_exclusions"], ["IDRiD_088", "IDRiD_089", "IDRiD_091"])
        self.assertEqual((parent["dataset"]["n_images"], parent["inference"]["keras_policy"], parent["inference"]["batch_size"]),
                         (103, "mixed_float16", 8))
        self.assertEqual(sorted(protocol["checkpoints"]), sorted(f"{a}_seed{s}" for a in ("p_ep", "e1_ep", "e2_ep") for s in (42, 123, 2026)))
        self.assertEqual(len({e["sha256"] for e in protocol["checkpoints"].values()}), 9)
        for name, entry in protocol["checkpoints"].items():
            self.assertEqual(len(entry["sha256"]), 64)
            self.assertTrue(entry["path"].startswith(f"experiments/EyePACSAdaptedAPTOS/{entry['arm']}_{protocol['adapted_encoder']['sha256'][:12]}_seed{entry['seed']}/"))
        self.assertFalse({e["sha256"] for e in protocol["checkpoints"].values()} & e2.baseline_hashes(parent))
        self.assertEqual(protocol["stored_p_tables"], b1_protocol["stored_p_tables"])       # P reused, the same pinned tables
        self.assertEqual(protocol["comparisons"]["primary"], ["p_ep_minus_p", "e1_ep_minus_p_ep"])
        self.assertEqual(protocol["comparisons"]["secondary_descriptive"], ["e2_ep_minus_p_ep", "e1_ep_minus_e2_ep"])
        self.assertEqual(e2.CONTRASTS, {"p_ep_minus_p": ("p_ep", "p"), "e1_ep_minus_p_ep": ("e1_ep", "p_ep"),
                                        "e2_ep_minus_p_ep": ("e2_ep", "p_ep"), "e1_ep_minus_e2_ep": ("e1_ep", "e2_ep")})
        self.assertEqual(protocol["pipeline"]["decode"], b1_protocol["pipeline"]["decode"])
        self.assertEqual((protocol["pipeline"]["stage2"], protocol["pipeline"]["rgb_frame"]),
                         (b1_protocol["pipeline"]["stage2"], b1_protocol["pipeline"]["rgb_frame"]))
        self.assertEqual(protocol["parity"]["tolerances"], b1_protocol["parity"]["tolerances"])
        self.assertEqual(protocol["confirmation_token"], "RUN_IDRID_BATCH_2_EP_ONCE")
        self.assertNotEqual(protocol["confirmation_token"], b1_protocol["confirmation_token"])
        self.assertIn("evaluation only", protocol["rules"])
        self.assertTrue(any("Stage 4" in d for d in protocol["disclosures"]) and any("clinical" in d for d in protocol["disclosures"]))
        self.assertEqual(len(sha), 64)

    def test_wrong_arms_seeds_or_parent_are_refused(self):
        real = json.load(open(e2.PROTOCOL_PATH, encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            parent_copy = os.path.join(tmp, "idrid_grading_protocol.json")
            shutil.copyfile(ig.PROTOCOL_PATH, parent_copy)

            def load(protocol):
                path = os.path.join(tmp, "p.json")
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(protocol, fh)
                return e2.load_protocol(path, parent_copy)
            load(copy.deepcopy(real))
            bad = copy.deepcopy(real)
            del bad["checkpoints"]["e2_ep_seed123"]                         # a missing seed
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            bad = copy.deepcopy(real)
            bad["checkpoints"]["e1_seed42"] = dict(bad["checkpoints"]["e1_ep_seed42"])     # an arm that is not an EP arm
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            bad = copy.deepcopy(real)
            bad["checkpoints"]["p_ep_seed42"]["arm"] = "e1_ep"                # entry filed under the wrong arm
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            bad = copy.deepcopy(real)
            bad["checkpoints"]["p_ep_seed42"]["seed"] = 123                   # ... or the wrong seed
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            bad = copy.deepcopy(real)
            bad["checkpoints"]["e1_ep_seed42"]["model_type"] = "P"            # ... or the wrong architecture
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            bad = copy.deepcopy(real)
            bad["checkpoints"]["e2_ep_seed42"]["sha256"] = bad["checkpoints"]["e1_ep_seed42"]["sha256"]   # one file for two entries
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            bad = copy.deepcopy(real)
            bad["comparisons"]["primary"] = ["e1_ep_minus_p_ep"]
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            bad = copy.deepcopy(real)
            bad["seeds"] = [42, 123]
            with self.assertRaises(e2.ProtocolError):
                load(bad)
            text = open(parent_copy, encoding="utf-8").read().replace('"batch_size": 8', '"batch_size": 16')
            with open(parent_copy, "w", encoding="utf-8") as fh:
                fh.write(text)
            with self.assertRaises(e2.ProtocolError):
                load(copy.deepcopy(real))                                     # a changed parent protocol

    @unittest.skipUnless(os.path.isdir(os.path.join(e2.HERE, "results", "EP_APTOS")), "local copies of the EP run records not present")
    def test_pins_match_the_runs_own_frozen_records_when_present(self):
        protocol, _, _ = e2.load_protocol()
        for name, entry in protocol["checkpoints"].items():
            local = os.path.join(e2.HERE, "results", "EP_APTOS", os.path.basename(entry["run_dir"]))
            frozen = json.load(open(os.path.join(local, "frozen_checkpoint.json")))
            self.assertEqual((frozen["sha256"], frozen["best_epoch_index"]), (entry["sha256"], entry["best_epoch_index"]))
            self.assertTrue(entry["path"].endswith(frozen["weights"]))


class VerifyFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        real, _, parent = e2.load_protocol()
        self.protocol, self.parent = fake_drive(self.tmp.name, copy.deepcopy(real), copy.deepcopy(parent))
        self.verify = lambda protocol=None: e2.verify_files(protocol or self.protocol, self.tmp.name, self.parent)

    def tearDown(self):
        self.tmp.cleanup()

    def _run_file(self, name, filename, **changes):
        path = os.path.join(self.tmp.name, *self.protocol["checkpoints"][name]["run_dir"].split("/"), filename)
        original = json.load(open(path))
        with open(path, "w") as fh:
            json.dump(dict(original, **changes), fh)
        return path, original

    def _refused_then_restored(self, name, filename, **changes):
        path, original = self._run_file(name, filename, **changes)
        with self.assertRaises(e2.ProtocolError):
            self.verify()
        with open(path, "w") as fh:
            json.dump(original, fh)
        self.verify()

    def test_pinned_files_are_accepted(self):
        files = self.verify()
        self.assertEqual(len(files["checkpoints"]), 9)
        self.assertEqual({files["kinds"][f"{a}_seed42"] for a in ("p_ep",)}, {"P"})
        self.assertEqual({files["kinds"][f"{a}_seed{s}"] for a in ("e1_ep", "e2_ep") for s in e2.SEEDS}, {"E1"})
        self.assertEqual(sorted(files["lesion_prior"]), sorted(f"{a}_seed{s}" for a in ("e1_ep", "e2_ep") for s in e2.SEEDS))
        self.assertEqual(sorted(files["p_tables"]), [42, 123, 2026])
        self.assertEqual(files["adapted_encoder_sha256"], self.protocol["adapted_encoder"]["sha256"])

    def test_wrong_hash_missing_or_replaced_checkpoints_are_refused(self):
        bad = copy.deepcopy(self.protocol)
        bad["checkpoints"]["e2_ep_seed123"]["sha256"] = "0" * 64
        with self.assertRaises(e2.ProtocolError):
            self.verify(bad)                                                  # wrong checkpoint hash
        path = os.path.join(self.tmp.name, *self.protocol["checkpoints"]["p_ep_seed42"]["path"].split("/"))
        os.rename(path, path + ".away")
        with self.assertRaises(e2.ProtocolError):
            self.verify()                                                     # missing checkpoint: nothing is substituted
        with open(path, "wb") as fh:
            fh.write(b"another model saved in its place")
        with self.assertRaises(e2.ProtocolError):
            self.verify()                                                     # a replaced checkpoint
        os.remove(path)
        os.rename(path + ".away", path)
        self.verify()
        self._refused_then_restored("e1_ep_seed42", "frozen_checkpoint.json", sha256="f" * 64)   # not the run's frozen checkpoint
        self._refused_then_restored("e1_ep_seed42", "result.json", best={"checkpoint": {"weights_sha256": "f" * 64, "monitor": "val_QWK"}})
        bad = copy.deepcopy(self.protocol)
        bad["checkpoints"]["e1_ep_seed42"]["best_slot"] = "best_z"
        with self.assertRaises(e2.ProtocolError):
            self.verify(bad)                                                  # the BEST pointer names another slot
        bad = copy.deepcopy(self.protocol)
        bad["stored_p_tables"]["p_seed2026"]["sha256"] = "0" * 64
        with self.assertRaises(e2.ProtocolError):
            self.verify(bad)                                                  # P table changed
        bad = copy.deepcopy(self.protocol)
        bad["adapted_encoder"]["sha256"] = "0" * 64
        with self.assertRaises(e2.ProtocolError):
            self.verify(bad)                                                  # another adapted encoder
        bad_parent = copy.deepcopy(self.parent)
        bad_parent["checkpoints"]["convnext_pretrained"]["sha256"] = "0" * 64
        with self.assertRaises(e2.ProtocolError):
            e2.verify_files(self.protocol, self.tmp.name, bad_parent)

    def test_wrong_arm_seed_lineage_or_selection_is_refused(self):
        self._refused_then_restored("p_ep_seed42", "config.json", arm="e1_ep")                       # a run of another arm
        self._refused_then_restored("p_ep_seed42", "config.json", seed=123)                          # ... of another seed
        self._refused_then_restored("e1_ep_seed123", "config.json", model_type="P")                  # ... of another architecture
        self._refused_then_restored("e1_ep_seed123", "config.json", experiment="E1MultiTask")        # ... of another experiment
        self._refused_then_restored("e2_ep_seed2026", "config.json", adapted_encoder_sha256="a" * 64)  # another encoder lineage
        self._refused_then_restored("e2_ep_seed2026", "initialization.json", encoder_is_adapted=False)
        self._refused_then_restored("e2_ep_seed2026", "initialization.json", encoder_is_imagenet=True)
        self._refused_then_restored("p_ep_seed123", "initialization.json", fresh_corn_head=False)
        self._refused_then_restored("p_ep_seed123", "config.json", checkpoint_selection="IDRiD QWK")  # not selected on APTOS validation
        self._refused_then_restored("p_ep_seed123", "result.json", best={"checkpoint": {
            "weights_sha256": self.protocol["checkpoints"]["p_ep_seed123"]["sha256"], "monitor": "val_loss"}})
        baseline = sorted(e2.baseline_hashes(self.parent))[0]
        bad = copy.deepcopy(self.protocol)
        bad["checkpoints"]["p_ep_seed42"]["sha256"] = baseline                # a pinned P / E1 / E2 checkpoint
        with self.assertRaises(e2.ProtocolError):
            self.verify(bad)


class SummaryTests(unittest.TestCase):
    def test_identical_models_have_zero_paired_differences(self):
        ids, grades, tables = synthetic()
        mask = np.array([i not in {"IDRiD_088", "IDRiD_089", "IDRiD_091"} for i in ids])
        s = e2.summarise(tables, mask, n_boot=40)
        self.assertEqual((s["n"], s["grade_counts"]), (100, [34, 5, 29, 19, 13]))
        self.assertEqual(set(s["aggregate"]), set(e2.MODELS))
        self.assertEqual(set(s["paired_differences"]), set(e2.CONTRASTS))
        for name in e2.CONTRASTS:
            d = s["paired_differences"][name]
            self.assertEqual(d["qwk"]["per_seed"], [0.0, 0.0, 0.0])
            self.assertEqual((d["qwk"]["ci_low"], d["qwk"]["ci_high"]), (0.0, 0.0))
            self.assertEqual(d["decoded_grades_differ"], [0, 0, 0])
        self.assertEqual({n: s["paired_differences"][n]["role"] for n in e2.CONTRASTS},
                         {"p_ep_minus_p": "primary", "e1_ep_minus_p_ep": "primary", "e2_ep_minus_p_ep": "secondary, descriptive",
                          "e1_ep_minus_e2_ep": "secondary, descriptive"})
        self.assertEqual(e2.summarise(tables, np.ones(103, bool), n_boot=20)["n"], 103)

    def test_differences_are_paired_per_seed_and_match_direct_metrics(self):
        ids, grades, tables = synthetic(shifts={"p_ep": 0.4, "e1_ep": 0.9, "e2_ep": -0.5})
        mask = np.array([i not in {"IDRiD_088", "IDRiD_089", "IDRiD_091"} for i in ids])
        s = e2.summarise(tables, mask, n_boot=60)
        for name, (a, b) in e2.CONTRASTS.items():
            for n, seed in enumerate(e2.SEEDS):
                direct = ph.metrics(b1.subset_table(tables[seed][a], mask))["qwk"] - ph.metrics(b1.subset_table(tables[seed][b], mask))["qwk"]
                self.assertAlmostEqual(s["paired_differences"][name]["qwk"]["per_seed"][n], direct, places=12)
            d = s["paired_differences"][name]["qwk"]
            self.assertAlmostEqual(d["mean"], float(np.mean(d["per_seed"])), places=12)
            self.assertLessEqual(d["ci_low"], d["ci_high"])
        for model in e2.MODELS:
            st = s["aggregate"][model]["qwk"]
            self.assertAlmostEqual(st["sd"], float(np.std(st["per_seed"], ddof=1)), places=12)
            for key in ("auroc_ge1", "auroc_ge2", "auroc_ge3", "auroc_ge4", "false_urgent_rate", "recall_grade0", "recall_grade4"):
                self.assertIn(key, s["aggregate"][model])
            self.assertEqual(np.asarray(s["per_seed"][42][model]["confusion_matrix"]).shape, (5, 5))
        self.assertEqual((s["n_boot"], s["boot_seed"], s["bootstrap"]), (60, ph.BOOT_SEED, "grade-stratified over the test images"))
        again = e2.summarise(tables, mask, n_boot=60)
        self.assertEqual(json.dumps(ph._jsonable(again), sort_keys=True), json.dumps(ph._jsonable(s), sort_keys=True))   # reproducible
        report = e2.report_markdown("T", "note", s, json.load(open(e2.PROTOCOL_PATH, encoding="utf-8")))
        for text in ("P-EP − P (primary)", "E1-EP − P-EP (primary)", "E2-EP − P-EP (secondary, descriptive)",
                     "E1-EP − E2-EP (secondary, descriptive)", "## Disclosures", "No equivalence or non-inferiority margin"):
            self.assertIn(text, report)

    def test_the_method_is_batch_1s(self):
        source = inspect.getsource(e2)
        for reused in ("b1.gates(", "b1.summarise(", "b1.report_markdown(", "b1.rgb_from_raw(", "b1.table_from_logits(",
                       "b1.require_gates(", "b1.read_p_tables(", "ig.read_idrid("):
            self.assertIn(reused, source)
        for redefined in ("def rgb_from_raw", "def table_from_logits", "def require_gates", "bootstrap_indices(", "stage2_rgb(",
                          "recompute_rgb_512(", "> 0.5", ".fit(", "train_on_batch", "compile("):
            self.assertNotIn(redefined, source)

    def test_preprocessing_is_the_locked_path_shared_with_batch_1(self):
        import cv2

        import ddr_probe
        import eyepacs_adaptation_data as ea
        with tempfile.TemporaryDirectory() as tmp:
            yy, xx = np.mgrid[:480, :640]
            image = np.zeros((480, 640, 3), np.uint8)
            image[((yy - 240) ** 2 + (xx - 320) ** 2) < 220 ** 2] = (40, 90, 170)
            image = np.clip(image + np.random.default_rng(0).integers(0, 12, image.shape), 0, 255).astype(np.uint8)
            path = os.path.join(tmp, "10_left.jpeg")
            cv2.imwrite(path, image)
            rgb, native = b1.rgb_from_raw(path)
            self.assertEqual((rgb.dtype, rgb.shape, native.shape), (np.float32, (512, 512, 3), (480, 640, 3)))
            np.testing.assert_array_equal(rgb, ea.frame_from_raw(path))        # the frame every EP model was trained on
            np.testing.assert_array_equal(rgb, ddr_probe.frame_from_raw(path))
        self.assertEqual(inspect.getsource(e2.run).count("b1.rgb_from_raw(path)"), 1)


class RunTests(unittest.TestCase):
    """The one-time run with fake models, a fake dataset reader and a temporary protocol: mechanics only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        tmp = self.tmp.name
        real, _, parent = e2.load_protocol()
        ids, grades, _ = synthetic()
        self.ids, self.grades = ids, grades
        protocol = copy.deepcopy(real)
        old_run = os.path.join(tmp, "drive", *protocol["stored_p_tables"]["source_run"].split("/"))
        os.makedirs(old_run)
        self.p_paths = {}
        for s in e2.SEEDS:
            base = (grades[:, None] - np.arange(4)[None, :] - 0.5) * 2.0 + np.random.default_rng(s).normal(0, 1, (103, 4))
            path = os.path.join(old_run, f"per_image_p_seed{s}.csv")
            pd.DataFrame(at.metrics_from_logits(ids, grades, base)[1]).to_csv(path, index=False)
            protocol["stored_p_tables"][f"p_seed{s}"]["sha256"] = ig.sha256_file(path)
            self.p_paths[s] = path
        self.protocol_path = os.path.join(tmp, "idrid_batch2_protocol.json")
        with open(self.protocol_path, "w", encoding="utf-8") as fh:
            json.dump(protocol, fh)
        parent_tmp = os.path.join(tmp, "idrid_grading_protocol.json")
        shutil.copyfile(ig.PROTOCOL_PATH, parent_tmp)
        self.protocol, self.sha = protocol, ig.sha256_file(self.protocol_path)
        self.read_images = []
        self._orig = (e2.verify_files, ig.read_idrid, b1.rgb_from_raw, e2.load_protocol)
        e2.load_protocol = lambda path=self.protocol_path, parent_path=None: self._orig[3](self.protocol_path, parent_tmp)
        e2.verify_files = lambda protocol, root, parent=None: {
            "checkpoints": {k: "unused" for k in protocol["checkpoints"]}, "p_tables": dict(self.p_paths), "lesion_prior": {},
            "kinds": {k: v["model_type"] for k, v in protocol["checkpoints"].items()}, "pretrained": "unused",
            "adapted_encoder_sha256": protocol["adapted_encoder"]["sha256"]}
        primary = np.array([i not in {"IDRiD_088", "IDRiD_089", "IDRiD_091"} for i in ids])
        self.primary = primary
        ig.read_idrid = lambda raw_dir, protocol: {"ids": ids, "grades": grades, "paths": [f"fake/{i}.jpg" for i in ids],
                                                   "hashes": {f"{i}.jpg": "0" * 64 for i in ids}, "primary": primary}

        def fake_rgb(path):
            self.read_images.append(path)
            n = int(os.path.basename(path)[6:9])
            return np.full((512, 512, 3), (n % 50) / 50.0, np.float32), np.zeros((60, 80, 3), np.uint8)
        b1.rgb_from_raw = fake_rgb
        self.out_root = os.path.join(tmp, "drive", "experiments", "IDRiDGrading")
        settings = dict(parent["inference"])
        self.gate = {"official": True, "PASS": True, "protocol_sha256": self.sha, "settings": settings,
                     "git_commit": ig.git_commit(), "environment": ig.environment(settings)}

    def tearDown(self):
        e2.verify_files, ig.read_idrid, b1.rgb_from_raw, e2.load_protocol = self._orig
        self.tmp.cleanup()

    def _run(self, **kw):
        args = dict(confirm=self.protocol["confirmation_token"], models=fake_models(), log=lambda *a: None)
        args.update(kw)
        return e2.run(os.path.join(self.tmp.name, "drive"), "fake_raw", self.out_root, self.gate, **args)

    def test_refuses_without_token_or_without_an_official_passed_gate(self):
        for confirm in (None, "yes", b1.load_protocol()[0]["confirmation_token"]):     # batch 1's token does not open batch 2
            with self.assertRaises(e2.ProtocolError):
                self._run(confirm=confirm)
        for change in ({"PASS": False}, {"official": False}, {"protocol_sha256": "0" * 64}, {"git_commit": "another commit"},
                       {"settings": dict(self.gate["settings"], batch_size=4)}):
            gate, self.gate = self.gate, dict(self.gate, **change)
            with self.assertRaises(e2.ProtocolError):
                self._run()
            self.gate = gate
        self.assertEqual(self.read_images, [])                               # nothing was read on any refusal
        self.assertFalse(os.path.exists(e2.lock_path(self.out_root, self.sha)))

    def test_one_run_writes_everything_reuses_p_and_then_locks(self):
        models = fake_models()
        result = self._run(models=models, command="test")
        out = result["out_dir"]
        self.assertIn("idrid_batch2_ep_", os.path.basename(out))
        self.assertEqual(len(self.read_images), 103)                         # every image read exactly once
        expected = {"lock_record.json", "image_manifest.json", "run_configuration.json", "primary_results_100.json",
                    "primary_report_100.md", "sensitivity_results_103.json", "sensitivity_report_103.md", "output_hashes.json"}
        expected |= {f"per_image_{a}_seed{s}.csv" for a in e2.RUN_MODELS for s in e2.SEEDS}
        expected |= {f"per_image_p_seed{s}_stored.csv" for s in e2.SEEDS}
        self.assertEqual(set(os.listdir(out)), expected)
        hashes = json.load(open(os.path.join(out, "output_hashes.json")))
        self.assertEqual(set(hashes), expected - {"output_hashes.json"})
        for name, sha in hashes.items():
            self.assertEqual(ig.sha256_file(os.path.join(out, name)), sha)     # output hashes are those of the files
        frame = pd.read_csv(os.path.join(out, "per_image_p_ep_seed42.csv"))
        for column in ("image_id", "true_grade", "predicted_grade", "logit_0", "logit_3", "p_gt_0", "p_gt_3", "in_primary_set"):
            self.assertIn(column, frame.columns)                             # raw logits and probabilities are kept
        self.assertEqual((len(frame), int(frame["in_primary_set"].sum())), (103, 100))
        for s in e2.SEEDS:                                                   # P is the stored table, byte for byte; never run
            self.assertEqual(ig.sha256_file(os.path.join(out, f"per_image_p_seed{s}_stored.csv")), ig.sha256_file(self.p_paths[s]))
        self.assertEqual({k[0] for k in models}, set(e2.RUN_MODELS))
        self.assertEqual({tuple(c) for c in models[("p_ep", 42)].seen}, {("reliability", "stage5_input", "stage6_input")})
        self.assertEqual({tuple(c) for c in models[("e1_ep", 42)].seen}, {("rgb",)})
        cfg = json.load(open(os.path.join(out, "run_configuration.json")))
        self.assertEqual((cfg["p_rerun"], cfg["stage3_run"], cfg["stage4_run"], cfg["protocol_sha256"], cfg["command"]),
                         (False, False, False, self.sha, "test"))
        self.assertEqual(sorted(cfg["models_run"]), sorted(f"{a}_seed{s}" for a in e2.RUN_MODELS for s in e2.SEEDS))
        self.assertEqual(cfg["checkpoints_verified"], {k: v["sha256"] for k, v in self.protocol["checkpoints"].items()})
        for key in ("environment", "git_commit", "gates", "timestamp_utc", "adapted_encoder_sha256", "excluded_from_primary"):
            self.assertIn(key, cfg)
        primary = json.load(open(os.path.join(out, "primary_results_100.json")))
        sensitivity = json.load(open(os.path.join(out, "sensitivity_results_103.json")))
        self.assertEqual((primary["n"], sensitivity["n"], primary["grade_counts"]), (100, 103, [34, 5, 29, 19, 13]))
        self.assertEqual(set(primary["paired_differences"]), set(e2.CONTRASTS))
        self.assertIn("PRIMARY RESULT (100 images)", open(os.path.join(out, "primary_report_100.md"), encoding="utf-8").read())
        self.assertIn("SENSITIVITY ANALYSIS ONLY (103 images)", open(os.path.join(out, "sensitivity_report_103.md"), encoding="utf-8").read())
        manifest = json.load(open(os.path.join(out, "image_manifest.json")))
        self.assertEqual([i for i, keep in zip(manifest["ids"], manifest["in_primary_set"]) if not keep], ["IDRiD_088", "IDRiD_089", "IDRiD_091"])
        lock = e2.lock_path(self.out_root, self.sha)
        self.assertTrue(os.path.exists(lock))
        self.assertNotEqual(os.path.basename(lock), os.path.basename(b1.lock_path(self.out_root, self.sha)))   # its own lock
        before = {n: ig.sha256_file(os.path.join(out, n)) for n in os.listdir(out)}
        with self.assertRaises(e2.ProtocolError):
            self._run()                                                      # a second run is refused ...
        self.assertEqual(len(self.read_images), 103)                         # ... before any image is read
        self.assertEqual(before, {n: ig.sha256_file(os.path.join(out, n)) for n in os.listdir(out)})   # and nothing is overwritten
        again = self._run(technical_rerun_reason="runtime crashed while writing the reports")
        self.assertNotEqual(again["out_dir"], out)                           # a recorded technical rerun gets its own directory
        self.assertEqual(json.load(open(lock))["previous_run"]["out_dir"], out)
        self.assertEqual(before, {n: ig.sha256_file(os.path.join(out, n)) for n in os.listdir(out)})

    def test_p_tables_that_are_not_the_verified_test_set_are_refused(self):
        frame = pd.read_csv(self.p_paths[123])
        frame.iloc[::-1].to_csv(self.p_paths[123], index=False)
        with self.assertRaises(e2.ProtocolError):
            self._run()

    def test_preflight_checks_files_subsets_and_whether_the_batch_was_run(self):
        report = e2.preflight(os.path.join(self.tmp.name, "drive"), "fake_raw", self.out_root, self.protocol_path)
        self.assertEqual((report["PASS"], report["images"], report["primary_images"], report["excluded_from_primary"],
                          report["primary_grade_counts"], report["already_run"]),
                         (True, 103, 100, ["IDRiD_088", "IDRiD_089", "IDRiD_091"], [34, 5, 29, 19, 13], False))
        self.assertEqual(len(report["checkpoints"]), 9)
        self.assertEqual(self.read_images, [])                               # the preflight decodes no image
        self._run()
        self.assertTrue(e2.preflight(os.path.join(self.tmp.name, "drive"), "fake_raw", self.out_root, self.protocol_path)["already_run"])
        self.primary[:] = True                                               # a reader that no longer excludes the three copies
        with self.assertRaises(e2.ProtocolError):
            e2.preflight(os.path.join(self.tmp.name, "drive"), "fake_raw", self.out_root, self.protocol_path)


class RealDataTests(unittest.TestCase):
    def test_this_module_has_no_default_path_to_idrid_images(self):
        source = inspect.getsource(e2)
        self.assertNotIn("datasets", source)
        self.assertNotIn("Original Images", source)
        for function in (e2.run, e2.preflight):
            self.assertIs(inspect.signature(function).parameters["idrid_raw_dir"].default, inspect.Parameter.empty)

    @unittest.skipUnless(os.path.isdir(REAL_IDRID), "local IDRiD grading folder not present")
    def test_the_real_image_manifest_and_subsets_are_the_locked_ones(self):
        _, _, parent = e2.load_protocol()
        data = ig.read_idrid(REAL_IDRID, parent)                              # hashes every file and the label file; decodes nothing
        primary = np.asarray(data["primary"], bool)
        self.assertEqual((len(data["ids"]), int(primary.sum())), (103, 100))
        self.assertEqual([i for i, keep in zip(data["ids"], primary) if not keep], ["IDRiD_088", "IDRiD_089", "IDRiD_091"])
        self.assertEqual(np.bincount(np.asarray(data["grades"])[primary], minlength=5).tolist(), [34, 5, 29, 19, 13])


if __name__ == "__main__":
    unittest.main()
