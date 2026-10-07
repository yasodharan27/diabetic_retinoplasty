"""CPU tests for idrid_batch1_eval. Synthetic images, synthetic logits and fake models only: no IDRiD file is
opened, no real checkpoint is loaded and P is never run. Nothing here is a result."""
import copy
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
import idrid_batch1_eval as eb
import idrid_grading_eval as ig

REAL_IDRID = os.path.join(os.path.dirname(eb.HERE) if False else eb.HERE, "datasets", "IDRiD")


class FakeNet:
    """Stands in for an E1 / E2 model: deterministic logits from the mean of each RGB frame."""

    def __init__(self, shift):
        self.shift = float(shift)
        self.calls = 0

    def predict_on_batch(self, x):
        self.calls += 1
        m = x["rgb"].reshape(len(x["rgb"]), -1).mean(axis=1, dtype=np.float64)
        base = (m - 0.5) * 40.0 + self.shift
        return [np.stack([base + 2, base, base - 2, base - 4], axis=1).astype(np.float32), None]


def fake_models(shift=0.0):
    return {(m, s): FakeNet(shift + 0.3 * i) for i, (m, s) in enumerate([(m, s) for m in eb.RUN_MODELS for s in eb.SEEDS])}


def synthetic_tables(n=103, seed=0, e1_shift=0.0, e2_shift=0.0):
    rng = np.random.default_rng(seed)
    grades = np.array(([0] * 34 + [1] * 5 + [2] * 32 + [3] * 19 + [4] * 13)[:n])
    for excluded, donor in ((87, 40), (88, 41), (90, 42)):          # IDRiD_088 / 089 / 091 are grade 2, as in the real set
        if excluded < n:
            grades[excluded], grades[donor] = grades[donor], grades[excluded]
    ids = [f"IDRiD_{i + 1:03d}" for i in range(n)]
    tables = {}
    for s in eb.SEEDS:
        base = (grades[:, None] - np.arange(4)[None, :] - 0.5) * 2.0 + rng.normal(0, 1.2, (n, 4))
        tables[s] = {"p": ph.table_arrays(pd.DataFrame(at.metrics_from_logits(ids, grades, base)[1])),
                     "e1": ph.table_arrays(pd.DataFrame(at.metrics_from_logits(ids, grades, base + e1_shift)[1])),
                     "e2": ph.table_arrays(pd.DataFrame(at.metrics_from_logits(ids, grades, base + e2_shift)[1]))}
    return ids, grades, tables


class ProtocolTests(unittest.TestCase):
    def test_protocol_pins_the_locked_design(self):
        protocol, sha, parent = eb.load_protocol()
        self.assertEqual(len(sha), 64)
        self.assertEqual(protocol["parent_protocol"]["sha256"], "f93c1237eb81ab20397dea8bc6d9079871d6308578304f793dfabee17938587e")
        self.assertEqual(parent["dataset"]["primary_exclusions"], ["IDRiD_088", "IDRiD_089", "IDRiD_091"])
        self.assertEqual((parent["dataset"]["n_images"], parent["dataset"]["grade_counts"]), (103, [34, 5, 32, 19, 13]))
        self.assertEqual(parent["inference"]["keras_policy"], "mixed_float16")
        self.assertEqual(parent["inference"]["batch_size"], 8)
        pins = {k: v["sha256"] for k, v in protocol["checkpoints"].items()}
        self.assertEqual(pins, {
            "e1_seed42": "a6101cf44629384335d767fff8c3f6270d349a676be8b07059c332cdd3bca971",
            "e1_seed123": "0667b16c9faf95e3f942f171134c5cd14128467b1cb314df96e34e0dba1ac7fc",
            "e1_seed2026": "840aa468458362463f008fecf4337d14089d06aba2b05011f2d28c4934f0d95f",
            "e2_seed42": "540ef5a1a528293d792eedafcf774acb11971c57541591f4cde8c954898b640e",
            "e2_seed123": "967429ce04d1654d1b4a40cb3292c19ddf92415d6e1878edc84003cdd90d3f9e",
            "e2_seed2026": "291dbfcd5aadffdd7869f132716ce986abb097c6db3e461cfd3d7d48342e4e3b"})
        self.assertEqual([protocol["stored_p_tables"][f"p_seed{s}"]["sha256"][:8] for s in eb.SEEDS],
                         ["579d4079", "070f1589", "cedd6563"])
        self.assertEqual(protocol["comparisons"]["primary"], ["e1_minus_p", "e2_minus_p"])
        self.assertEqual(protocol["comparisons"]["secondary_descriptive"], ["e2_minus_e1"])
        self.assertEqual((ph.N_BOOT, ph.BOOT_SEED), (2000, 20260927))
        self.assertNotIn("p", eb.RUN_MODELS)                               # P is never run by this module
        self.assertTrue(any("used once before" in d for d in protocol["disclosures"]))
        self.assertTrue(any("camera domain" in d for d in protocol["disclosures"]))

    def test_a_changed_parent_protocol_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "idrid_grading_protocol.json")
            with open(ig.PROTOCOL_PATH, encoding="utf-8") as fh:
                data = json.load(fh)
            data["inference"]["batch_size"] = 4
            with open(parent, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            with self.assertRaises(eb.ProtocolError):
                eb.load_protocol(parent_path=parent)

    def test_stored_local_p_tables_match_the_pins_when_present(self):
        protocol, _, _ = eb.load_protocol()
        local = os.path.join(eb.HERE, "results", "IDRiDGrading", "idrid_grading_f93c1237eb81_2026-10-05_13-28-16")
        if not os.path.isdir(local):
            self.skipTest("no local mirror of the earlier run")
        for s in eb.SEEDS:
            self.assertEqual(ig.sha256_file(os.path.join(local, f"per_image_p_seed{s}.csv")),
                             protocol["stored_p_tables"][f"p_seed{s}"]["sha256"])


class VerifyFilesTests(unittest.TestCase):
    def _root(self, tmp, protocol):
        for name, entry in protocol["checkpoints"].items():
            path = os.path.join(tmp, *entry["path"].split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(name.encode())
            entry["sha256"] = ig.sha256_file(path)
            run_dir = os.path.join(tmp, *entry["run_dir"].split("/"))
            with open(os.path.join(run_dir, "checkpoints", "best.json"), "w") as fh:
                json.dump({"active": entry["best_slot"], "epoch": entry["best_epoch_index"]}, fh)
            with open(os.path.join(run_dir, "result.json"), "w") as fh:
                json.dump({"best": {"checkpoint": {"weights_sha256": entry["sha256"], "monitor": "val_QWK"}},
                           "config": {"lesion_prior": [0.07, 0.11, 0.11, 0.05]}}, fh)
        for s in eb.SEEDS:
            entry = protocol["stored_p_tables"][f"p_seed{s}"]
            path = os.path.join(tmp, *entry["path"].split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as fh:
                fh.write(f"p{s}")
            entry["sha256"] = ig.sha256_file(path)
        return protocol

    def test_pinned_files_are_accepted_and_any_mismatch_is_refused(self):
        real, _, _ = eb.load_protocol()
        with tempfile.TemporaryDirectory() as tmp:
            protocol = self._root(tmp, copy.deepcopy(real))
            files = eb.verify_files(protocol, tmp)
            self.assertEqual(len(files["checkpoints"]), 6)
            self.assertEqual(sorted(files["p_tables"]), [42, 123, 2026])
            self.assertEqual(files["lesion_prior"]["e1_seed42"], [0.07, 0.11, 0.11, 0.05])
            bad = copy.deepcopy(protocol)
            bad["checkpoints"]["e2_seed123"]["sha256"] = "0" * 64
            with self.assertRaises(eb.ProtocolError):
                eb.verify_files(bad, tmp)                                  # wrong checkpoint hash
            bad = copy.deepcopy(protocol)
            bad["checkpoints"]["e1_seed42"]["best_slot"] = "best_b"
            with self.assertRaises(eb.ProtocolError):
                eb.verify_files(bad, tmp)                                  # wrong BEST slot
            bad = copy.deepcopy(protocol)
            bad["stored_p_tables"]["p_seed2026"]["sha256"] = "0" * 64
            with self.assertRaises(eb.ProtocolError):
                eb.verify_files(bad, tmp)                                  # P table changed
            run_dir = os.path.join(tmp, *protocol["checkpoints"]["e1_seed123"]["run_dir"].split("/"))
            with open(os.path.join(run_dir, "result.json"), "w") as fh:
                json.dump({"best": {"checkpoint": {"weights_sha256": "f" * 64, "monitor": "val_QWK"}},
                           "config": {"lesion_prior": [0.1] * 4}}, fh)
            with self.assertRaises(eb.ProtocolError):
                eb.verify_files(protocol, tmp)                             # not the checkpoint the run evaluated


class SummaryTests(unittest.TestCase):
    def test_identical_models_have_zero_paired_differences(self):
        ids, grades, tables = synthetic_tables()
        mask = np.array([i not in {"IDRiD_088", "IDRiD_089", "IDRiD_091"} for i in ids])
        s = eb.summarise(tables, mask, n_boot=40)
        self.assertEqual((s["n"], s["grade_counts"]), (100, [34, 5, 29, 19, 13]))
        for name in eb.CONTRASTS:
            d = s["paired_differences"][name]
            self.assertEqual(d["qwk"]["mean"], 0.0)
            self.assertEqual((d["qwk"]["ci_low"], d["qwk"]["ci_high"]), (0.0, 0.0))
            self.assertEqual(d["decoded_grades_differ"], [0, 0, 0])
        self.assertEqual(s["paired_differences"]["e1_minus_p"]["role"], "primary")
        self.assertEqual(s["paired_differences"]["e2_minus_p"]["role"], "primary")
        self.assertEqual(s["paired_differences"]["e2_minus_e1"]["role"], "secondary, descriptive")
        self.assertEqual(s["aggregate"]["e1"]["qwk"], s["aggregate"]["p"]["qwk"])
        full = eb.summarise(tables, np.ones(103, bool), n_boot=10)
        self.assertEqual((full["n"], full["grade_counts"]), (103, [34, 5, 32, 19, 13]))

    def test_differences_are_paired_per_seed_and_match_direct_metrics(self):
        ids, grades, tables = synthetic_tables(e1_shift=0.8, e2_shift=-0.8)
        mask = np.ones(103, bool)
        s = eb.summarise(tables, mask, n_boot=60)
        for seed_index, seed in enumerate(eb.SEEDS):
            direct = ph.metrics(tables[seed]["e1"])["qwk"] - ph.metrics(tables[seed]["p"])["qwk"]
            self.assertAlmostEqual(s["paired_differences"]["e1_minus_p"]["qwk"]["per_seed"][seed_index], direct, places=12)
            fu = ph.metrics(tables[seed]["e2"])["false_urgent_rate"] - ph.metrics(tables[seed]["e1"])["false_urgent_rate"]
            self.assertAlmostEqual(s["paired_differences"]["e2_minus_e1"]["false_urgent_rate"]["per_seed"][seed_index], fu, places=12)
        d = s["paired_differences"]["e2_minus_e1"]
        self.assertTrue(d["qwk"]["ci_low"] <= d["qwk"]["mean"] <= d["qwk"]["ci_high"])
        self.assertGreater(sum(d["decoded_grades_differ"]), 0)
        for g in range(5):
            self.assertIn(f"recall_grade{g}", s["aggregate"]["e1"])
            self.assertIn(f"recall_grade{g}", s["paired_differences"]["e1_minus_p"])
        text = eb.report_markdown("T", "note", s, eb.load_protocol()[0])
        for needle in ("E1 − P (primary)", "E2 − E1 (secondary, descriptive)", "used once before", "camera domain",
                       "No equivalence or non-inferiority margin"):
            self.assertIn(needle, text)


class RunTests(unittest.TestCase):
    """The one-time run with fake models, a fake dataset reader and a temporary protocol: mechanics only."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = self.tmp.name
        real, _, parent = eb.load_protocol()
        self.parent = parent
        ids, grades, tables = synthetic_tables()
        self.ids, self.grades = ids, grades
        protocol = copy.deepcopy(real)
        self.old_run = os.path.join(tmp, "drive", *protocol["stored_p_tables"]["source_run"].split("/"))
        os.makedirs(self.old_run)
        self.p_paths = {}
        for s in eb.SEEDS:
            base = (grades[:, None] - np.arange(4)[None, :] - 0.5) * 2.0 + np.random.default_rng(s).normal(0, 1, (103, 4))
            path = os.path.join(self.old_run, f"per_image_p_seed{s}.csv")
            pd.DataFrame(at.metrics_from_logits(ids, grades, base)[1]).to_csv(path, index=False)
            protocol["stored_p_tables"][f"p_seed{s}"]["sha256"] = ig.sha256_file(path)
            self.p_paths[s] = path
        self.protocol_path = os.path.join(tmp, "idrid_batch1_protocol.json")
        with open(self.protocol_path, "w", encoding="utf-8") as fh:
            json.dump(protocol, fh)
        shutil.copyfile(ig.PROTOCOL_PATH, os.path.join(tmp, "idrid_grading_protocol.json"))
        self.protocol, self.sha = protocol, ig.sha256_file(self.protocol_path)
        self.read_images = []
        self._orig = (eb.verify_files, ig.read_idrid, eb.rgb_from_raw, eb.load_protocol)
        parent_path_tmp = os.path.join(tmp, "idrid_grading_protocol.json")
        eb.load_protocol = lambda path=self.protocol_path, parent_path=None: self._orig[3](self.protocol_path, parent_path_tmp)
        eb.verify_files = lambda protocol, root: {"checkpoints": {k: "unused" for k in protocol["checkpoints"]},
                                                  "p_tables": dict(self.p_paths), "lesion_prior": {}}
        primary = np.array([i not in {"IDRiD_088", "IDRiD_089", "IDRiD_091"} for i in ids])
        ig.read_idrid = lambda raw_dir, protocol: {"ids": ids, "grades": grades, "paths": [f"fake/{i}.jpg" for i in ids],
                                                   "hashes": {f"{i}.jpg": "0" * 64 for i in ids}, "primary": primary}

        def fake_rgb(path):
            self.read_images.append(path)
            n = int(os.path.basename(path)[6:9])
            return np.full((512, 512, 3), (n % 50) / 50.0, np.float32), np.zeros((60, 80, 3), np.uint8)
        eb.rgb_from_raw = fake_rgb
        self.out_root = os.path.join(tmp, "drive", "experiments", "IDRiDGrading")
        settings = dict(parent["inference"])
        self.gate = {"official": True, "PASS": True, "protocol_sha256": self.sha, "settings": settings,
                     "git_commit": ig.git_commit(), "environment": ig.environment(settings)}

    def tearDown(self):
        eb.verify_files, ig.read_idrid, eb.rgb_from_raw, eb.load_protocol = self._orig
        self.tmp.cleanup()

    def _run(self, **kw):
        args = dict(confirm=self.protocol["confirmation_token"], models=fake_models(), log=lambda *a: None)
        args.update(kw)
        return eb.run(os.path.join(self.tmp.name, "drive"), "fake_raw", self.out_root, self.gate, **args)

    def test_refuses_without_token_or_without_an_official_passed_gate(self):
        with self.assertRaises(eb.ProtocolError):
            self._run(confirm="yes")
        for bad in ({"PASS": False}, {"official": False}, {"protocol_sha256": "0" * 64}, {"git_commit": "other"},
                    {"settings": dict(self.gate["settings"], batch_size=4)}):
            gate = dict(self.gate, **bad)
            with self.assertRaises(eb.ProtocolError):
                eb.run(os.path.join(self.tmp.name, "drive"), "fake_raw", self.out_root, gate,
                       confirm=self.protocol["confirmation_token"], models=fake_models(), log=lambda *a: None)
        self.assertEqual(self.read_images, [])                             # nothing was read
        self.assertFalse(os.path.exists(eb.lock_path(self.out_root, self.sha)))

    def test_one_run_writes_everything_reuses_p_and_then_locks(self):
        before = {n: ig.sha256_file(os.path.join(self.old_run, n)) for n in sorted(os.listdir(self.old_run))}
        models = fake_models()
        result = self._run(models=models)
        out = result["out_dir"]
        self.assertIn("idrid_batch1_e1e2_", os.path.basename(out))
        self.assertNotEqual(os.path.abspath(out), os.path.abspath(self.old_run))
        self.assertEqual({n: ig.sha256_file(os.path.join(self.old_run, n)) for n in sorted(os.listdir(self.old_run))}, before)
        self.assertEqual(sorted(os.listdir(self.old_run)), sorted(before))  # the earlier run is untouched
        self.assertEqual(len(self.read_images), 103)                        # every image once
        expected = ["image_manifest.json", "lock_record.json", "primary_report_100.md", "primary_results_100.json",
                    "run_configuration.json", "sensitivity_report_103.md", "sensitivity_results_103.json"]
        expected += [f"per_image_{m}_seed{s}.csv" for m in ("e1", "e2") for s in eb.SEEDS]
        expected += [f"per_image_p_seed{s}_stored.csv" for s in eb.SEEDS]
        self.assertEqual(sorted(os.listdir(out)), sorted(expected))
        for s in eb.SEEDS:                                                  # P is a byte copy of the stored table
            self.assertEqual(ig.sha256_file(os.path.join(out, f"per_image_p_seed{s}_stored.csv")), ig.sha256_file(self.p_paths[s]))
        self.assertEqual(set(models), {(m, s) for m in ("e1", "e2") for s in eb.SEEDS})
        with open(os.path.join(out, "run_configuration.json")) as fh:
            cfg = json.load(fh)
        self.assertEqual((cfg["p_rerun"], cfg["stage3_run"], cfg["stage4_run"]), (False, False, False))
        self.assertEqual(len(cfg["checkpoints_verified"]), 6)
        self.assertEqual(cfg["excluded_from_primary"], ["IDRiD_088", "IDRiD_089", "IDRiD_091"])
        with open(os.path.join(out, "primary_results_100.json")) as fh:
            primary = json.load(fh)
        with open(os.path.join(out, "sensitivity_results_103.json")) as fh:
            sensitivity = json.load(fh)
        self.assertEqual((primary["n"], sensitivity["n"]), (100, 103))
        self.assertEqual(set(primary["paired_differences"]), set(eb.CONTRASTS))
        frame = pd.read_csv(os.path.join(out, "per_image_e1_seed42.csv"), dtype={"image_id": str})
        self.assertEqual((len(frame), int(frame["in_primary_set"].sum())), (103, 100))
        p = (np.cumprod(1 / (1 + np.exp(-frame[[f"logit_{k}" for k in range(4)]].to_numpy())), axis=1) > 0.5).sum(axis=1)
        np.testing.assert_array_equal(p, frame["predicted_grade"].to_numpy())  # decode = count of cumulative p > 0.5
        with open(os.path.join(out, "image_manifest.json")) as fh:
            manifest = json.load(fh)
        self.assertEqual((len(manifest["ids"]), sum(manifest["in_primary_set"])), (103, 100))
        self.assertTrue(os.path.exists(eb.lock_path(self.out_root, self.sha)))
        with self.assertRaises(eb.ProtocolError):                           # a second run is refused
            self._run()
        again = self._run(technical_rerun_reason="test of the documented technical-failure path")
        with open(eb.lock_path(self.out_root, self.sha)) as fh:
            lock = json.load(fh)
        self.assertEqual(lock["technical_rerun_reason"], "test of the documented technical-failure path")
        self.assertIsNotNone(lock["previous_run"])
        self.assertNotEqual(again["out_dir"], out)

    def test_p_tables_that_are_not_the_verified_test_set_are_refused(self):
        frame = pd.read_csv(self.p_paths[42], dtype={"image_id": str}).iloc[::-1]
        frame.to_csv(self.p_paths[42], index=False)
        with self.assertRaises(eb.ProtocolError):
            self._run()


class RealDataIsNotTouchedTests(unittest.TestCase):
    def test_this_module_has_no_default_path_to_idrid_images(self):
        import inspect
        source = inspect.getsource(eb)
        self.assertNotIn("datasets", source)                               # paths are always passed in by the caller
        self.assertNotIn("load_segmentation_models", source)               # Stage 3 / 4 are never loaded
        self.assertNotIn("build_pl_model", source)                         # P is never built


if __name__ == "__main__":
    unittest.main()
