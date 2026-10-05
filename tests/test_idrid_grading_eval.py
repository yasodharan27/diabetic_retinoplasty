"""CPU tests for idrid_grading_eval.py: the protocol/lock file, the dataset checks, the guards that keep the
IDRiD test to one run, and the aggregation. No model is run and no IDRiD prediction is generated here."""
import copy
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import idrid_grading_eval as ig
import pathology_grader_fusion as pf
import pathology_grader_severity_fusion as sf
import pipeline_v2_config as v2cfg

REPO = os.path.dirname(os.path.abspath(ig.__file__))
REAL_IDRID = os.path.join(REPO, "datasets", "IDRiD", "grading", "raw")
REAL_CHECKPOINTS = os.path.join(REPO, "trained_models", "frozen_eval")
GRADES = np.array([0] * 34 + [1] * 5 + [2] * 32 + [3] * 19 + [4] * 13)
IDS = [f"IDRiD_{k:03d}" for k in range(1, 104)]


def logits_for(seed, quality):
    rng = np.random.default_rng(seed)
    return (GRADES[:, None] - np.arange(4)[None, :] - 0.5) * quality + rng.normal(0, 1.0, (len(GRADES), 4))


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.sha = ig.load_protocol()

    def test_frozen_decisions(self):
        p = self.protocol
        self.assertEqual(p["dataset"]["primary_exclusions"], ["IDRiD_088", "IDRiD_089", "IDRiD_091"])
        self.assertEqual((p["dataset"]["n_images"], p["dataset"]["grade_counts"]), (103, [34, 5, 32, 19, 13]))
        self.assertEqual(p["seeds"], [42, 123, 2026])
        self.assertEqual({k: p["fusion"][k] for k in ("fused_0", "fused_1", "fused_2", "fused_3")},
                         {"fused_0": "0.5 * P_0 + 0.5 * Path_0", "fused_1": "0.5 * P_1 + 0.5 * Path_1",
                          "fused_2": "0.5 * P_2 + 0.5 * Path_2", "fused_3": "min(P_3, fused_2)"})
        self.assertEqual(p["inference"], {"device": "cuda", "stage4_amp": True, "keras_policy": "mixed_float16",
                                          "vessel_tta": True, "stage4_inference_size": 1536, "cache_size": 512,
                                          "stage2_profile": "DR", "batch_size": 8})
        self.assertEqual((p["aggregation"]["ensemble"], p["aggregation"]["seed_selection"]), (False, False))
        self.assertIn("lesion / vessel shuffles", p["not_done_on_idrid"])
        self.assertEqual(p["confirmation_token"], ig.CONFIRMATION_TOKEN)
        self.assertEqual(len(self.sha), 64)

    def test_eight_pinned_model_files(self):
        c = self.protocol["checkpoints"]
        models = ["stage3_lwnet", "stage4_unet"] + [f"p_seed{s}" for s in ig.SEEDS] + [f"pathology_seed{s}" for s in ig.SEEDS]
        self.assertEqual(sorted(c), sorted(models + ["convnext_pretrained"]))
        for name in models + ["convnext_pretrained"]:
            self.assertRegex(c[name]["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(c[name]["bytes"], 0)
        self.assertEqual(c["stage3_lwnet"]["sha256"], v2cfg.STAGE3_LWNET_SHA256)
        self.assertEqual(c["stage4_unet"]["sha256"], "cb5fc7a8d370af7d2ae191cadaebdde858f71820d147fb76f852b757766f8ad8")
        self.assertEqual(len({c[f"p_seed{s}"]["sha256"] for s in ig.SEEDS} | {c[f"pathology_seed{s}"]["sha256"] for s in ig.SEEDS}), 6)
        for s in ig.SEEDS:
            for kind in ("p", "pathology"):
                e = c[f"{kind}_seed{s}"]
                self.assertIn(f"/{e['best_slot']}/model.weights.h5", e["path"])      # a BEST slot, never a LAST generation
                self.assertNotIn("gen_", e["path"])
                self.assertIn(f"seed_{s}" if kind == "p" else f"seed{s}", e["path"])
        self.assertEqual([c[f"p_seed{s}"]["best_epoch_index"] for s in ig.SEEDS], [18, 3, 6])
        self.assertEqual([c[f"pathology_seed{s}"]["best_epoch_index"] for s in ig.SEEDS], [42, 10, 36])

    def test_parity_tolerances_reuse_the_project_constants(self):
        tol = self.protocol["parity"]["tolerances"]
        self.assertEqual((tol["rgb_max_abs"], tol["vessel_max_abs"], tol["stage4_max_counts"]),
                         (v2cfg.RGB_PARITY_TOL, v2cfg.STAGE3_PARITY_TOL, v2cfg.CANARY_TOL_COUNTS))
        self.assertEqual((tol["logit_max_abs"], tol["probability_max_abs"], tol["grades"]), (0.05, 0.01, "identical"))
        ids = [f"{k:03d}" for k in range(40)]
        grades = [k % 5 for k in range(40)]
        subset = ig.parity_subset(ids, grades, 2)
        self.assertEqual(subset, ["000", "001", "002", "003", "004", "005", "006", "007", "008", "009"])
        self.assertEqual(subset, ig.parity_subset(ids, grades, 2))

    def test_a_changed_rule_or_exclusion_list_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mutate in (lambda p: p["fusion"].update(fused_3="0.5 * P_3 + 0.5 * Path_3"),
                           lambda p: p["dataset"]["primary_exclusions"].append("IDRiD_001"),
                           lambda p: p.update(seeds=[42, 123])):
                changed = copy.deepcopy(self.protocol)
                mutate(changed)
                path = os.path.join(tmp, "protocol.json")
                with open(path, "w") as fh:
                    json.dump(changed, fh)
                with self.assertRaises(ig.ProtocolError):
                    ig.load_protocol(path)

    @unittest.skipUnless(os.path.isdir(REAL_CHECKPOINTS), "frozen checkpoints not downloaded locally")
    def test_pinned_files_verify_and_a_changed_file_is_refused(self):
        paths = ig.verify_checkpoints(self.protocol, REAL_CHECKPOINTS)
        self.assertEqual(len(paths), 9)
        with tempfile.TemporaryDirectory() as tmp:
            entry = self.protocol["checkpoints"]["pathology_seed42"]
            target = ig.checkpoint_path(tmp, entry)
            os.makedirs(os.path.dirname(target))
            with open(target, "wb") as fh:
                fh.write(b"not the pinned weights")
            with self.assertRaises(ig.ProtocolError):
                ig.verify_checkpoints({"checkpoints": {"pathology_seed42": entry}}, tmp)
            with self.assertRaises(ig.ProtocolError):
                ig.verify_checkpoints({"checkpoints": {"stage4_unet": self.protocol["checkpoints"]["stage4_unet"]}}, tmp)


class DatasetTests(unittest.TestCase):
    def _fake(self, tmp, n=6, exclusions=("IDRiD_002", "IDRiD_003", "IDRiD_005")):
        import hashlib
        protocol, _ = ig.load_protocol()
        protocol = copy.deepcopy(protocol)
        image_dir = os.path.join(tmp, "raw", *protocol["dataset"]["image_dir"].split("/"))
        label_file = os.path.join(tmp, "raw", *protocol["dataset"]["label_file"].split("/"))
        os.makedirs(image_dir)
        os.makedirs(os.path.dirname(label_file))
        digest, grades = hashlib.sha256(), [0, 1, 2, 3, 4, 2]
        with open(label_file, "w", newline="") as fh:
            fh.write("Image name,Retinopathy grade,Risk of macular edema \n")
            for k in range(1, n + 1):
                name = f"IDRiD_{k:03d}.jpg"
                with open(os.path.join(image_dir, name), "wb") as img:
                    img.write(f"image {k}".encode())
                digest.update(f"{name}:{ig.sha256_file(os.path.join(image_dir, name))}\n".encode())
                fh.write(f"IDRiD_{k:03d},{grades[k - 1]},0\n")
        protocol["dataset"].update(n_images=n, grade_counts=np.bincount(grades, minlength=5).tolist(),
                                   images_manifest_sha256=digest.hexdigest(), label_file_sha256=ig.sha256_file(label_file),
                                   primary_exclusions=list(exclusions))
        return protocol, os.path.join(tmp, "raw"), image_dir

    def test_reads_raw_images_and_marks_the_primary_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, raw, image_dir = self._fake(tmp)
            data = ig.read_idrid(raw, protocol)
            self.assertEqual(data["ids"], [f"IDRiD_{k:03d}" for k in range(1, 7)])
            self.assertEqual(data["grades"].tolist(), [0, 1, 2, 3, 4, 2])
            self.assertEqual(data["primary"].tolist(), [True, False, False, True, False, True])
            self.assertEqual(int(data["primary"].sum()), 3)
            with open(os.path.join(image_dir, "IDRiD_004.jpg"), "wb") as fh:      # a changed image
                fh.write(b"tampered")
            with self.assertRaises(ig.ProtocolError):
                ig.read_idrid(raw, protocol)

    def test_refuses_the_processed_tree_wrong_counts_and_unknown_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            protocol, raw, _ = self._fake(tmp)
            processed = os.path.join(tmp, "processed")
            os.rename(raw, processed)
            with self.assertRaises(ig.ProtocolError):
                ig.read_idrid(processed, protocol)                                # Stage 2 would be applied twice
            os.rename(processed, raw)
            with self.assertRaises(ig.ProtocolError):
                ig.read_idrid(raw, dict(protocol, dataset=dict(protocol["dataset"], n_images=7)))
            with self.assertRaises(ig.ProtocolError):
                ig.read_idrid(raw, dict(protocol, dataset=dict(protocol["dataset"], primary_exclusions=["IDRiD_099"] * 3)))

    @unittest.skipUnless(os.path.isdir(REAL_IDRID), "IDRiD grading data not present locally")
    def test_the_real_test_set_matches_the_pinned_manifest(self):
        protocol, _ = ig.load_protocol()                                          # files and labels only: no prediction
        data = ig.read_idrid(REAL_IDRID, protocol)
        self.assertEqual((len(data["ids"]), int(data["primary"].sum())), (103, 100))
        excluded = [g for i, g in zip(data["ids"], data["grades"]) if i in protocol["dataset"]["primary_exclusions"]]
        self.assertEqual(excluded, [2, 2, 2])
        self.assertEqual(np.bincount(data["grades"][data["primary"]], minlength=5).tolist(), [34, 5, 29, 19, 13])


class AggregationTests(unittest.TestCase):
    def setUp(self):
        logits = {"p": {s: logits_for(s, 2.0) for s in ig.SEEDS}, "pathology": {s: logits_for(100 + s, 1.2) for s in ig.SEEDS}}
        self.tables, self.frames = ig.tables_from_logits(IDS, GRADES, logits)
        self.primary = np.array([i not in ("IDRiD_088", "IDRiD_089", "IDRiD_091") for i in IDS])

    def test_fused_tables_are_the_locked_rule(self):
        for s in ig.SEEDS:
            want = sf.fuse_severity(self.tables[s]["p"], self.tables[s]["pathology"])
            np.testing.assert_array_equal(self.tables[s]["fused"]["pred"], want["pred"])
            np.testing.assert_allclose(self.tables[s]["fused"]["p_gt"], want["p_gt"])
            self.assertEqual(list(self.frames[s]["fused"].columns),
                             ["image_id", "true_grade", "predicted_grade", "p_gt_0", "p_gt_1", "p_gt_2", "p_gt_3"])
            self.assertIn("logit_0", self.frames[s]["p"].columns)

    def test_summary_per_seed_mean_sd_and_no_ensemble(self):
        import arch1_posthoc as ph
        s = ig.summarise(self.tables, self.primary, n_boot=30)
        self.assertEqual((s["n"], s["grade_counts"]), (100, np.bincount(GRADES[self.primary], minlength=5).tolist()))
        self.assertEqual(set(s["per_seed"][42]), {"p", "pathology", "fused"})
        for model in ig.MODELS:
            x = [ph.metrics(ig.subset_table(self.tables[k][model], self.primary))["qwk"] for k in ig.SEEDS]
            self.assertAlmostEqual(s["aggregate"][model]["qwk"]["mean"], np.mean(x), places=12)
            self.assertAlmostEqual(s["aggregate"][model]["qwk"]["sd"], np.std(x, ddof=1), places=12)
            self.assertEqual(s["aggregate"][model]["qwk"]["per_seed"], x)
            for g in range(5):
                self.assertIn(f"recall_grade{g}", s["aggregate"][model])
            lo, hi = s["seed_mean_interval_95"][model]["qwk"]
            self.assertLessEqual(lo, hi)
            self.assertEqual(np.array(s["per_seed"][42][model]["confusion_matrix"]).sum(), 100)
        self.assertIn("ci_low", s["reference_differences"]["fused_minus_p"]["qwk"])
        full = ig.summarise(self.tables, np.ones(103, bool), n_boot=10)
        self.assertEqual((full["n"], full["grade_counts"]), (103, [34, 5, 32, 19, 13]))
        text = ig.report_markdown("T", "note", s, ig.load_protocol()[0])
        for needle in ("FINAL PIPELINE", "reference: P alone", "no ensemble, no seed selection", "fused_3 = min(P_3, fused_2)",
                       "recall, grade 4", "Confusion matrix, seed 2026"):
            self.assertIn(needle, text)


class OneRunGuardTests(unittest.TestCase):
    def setUp(self):
        self.protocol, self.sha = ig.load_protocol()

    def test_token_gate_and_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            parity = {"official": True, "PASS": True}
            with self.assertRaises(ig.ProtocolError):                             # no token
                ig.run_idrid(tmp, tmp, tmp, parity, confirm=None)
            with self.assertRaises(ig.ProtocolError):
                ig.run_idrid(tmp, tmp, tmp, parity, confirm="yes")
            for bad in ({"official": False, "PASS": True}, {"official": True, "PASS": False}, {}):
                with self.assertRaises(ig.ProtocolError):                         # the gate has not passed
                    ig.require_gate(bad, self.protocol, self.sha, self.protocol["inference"])
            other = {"official": True, "PASS": True, "protocol_sha256": "0" * 64, "settings": self.protocol["inference"]}
            with self.assertRaises(ig.ProtocolError):                             # another protocol
                ig.require_gate(other, self.protocol, self.sha, self.protocol["inference"])
            precheck = {"official": True, "PASS": True, "protocol_sha256": self.sha,
                        "settings": dict(self.protocol["inference"], device="cpu", stage4_amp=False)}
            with self.assertRaises(ig.ProtocolError):                             # other settings
                ig.require_gate(precheck, self.protocol, self.sha, self.protocol["inference"])
            self.assertEqual(os.listdir(tmp), [])                                 # nothing was written or read

    def test_a_second_run_is_refused_without_a_technical_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = ig.require_gate
            ig.require_gate = lambda *a, **k: True
            try:
                with open(ig.lock_path(tmp, self.sha), "w") as fh:
                    json.dump({"timestamp_utc": "2026-10-06_00-00-00"}, fh)
                with self.assertRaises(ig.ProtocolError) as ctx:
                    ig.run_idrid(os.path.join(tmp, "none"), os.path.join(tmp, "none"), tmp, {}, confirm=ig.CONFIRMATION_TOKEN)
                self.assertIn("already been run", str(ctx.exception))
                with self.assertRaises(ig.ProtocolError) as ctx:                  # with a reason it proceeds to the file checks
                    ig.run_idrid(os.path.join(tmp, "none"), os.path.join(tmp, "none"), tmp, {}, confirm=ig.CONFIRMATION_TOKEN,
                                 technical_rerun_reason="Drive dropped mid-run")
                self.assertIn("not found", str(ctx.exception))
                self.assertEqual(len(os.listdir(tmp)), 1)                         # still only the lock: no output folder
            finally:
                ig.require_gate = original


if __name__ == "__main__":
    unittest.main()
