"""CPU tests for ddr_probe_ep (the §68 probe applied to the EP arms). Fabricated run directories, synthetic
features and masks only: no EP checkpoint exists, no DDR test image is scored, and nothing here is a result."""
import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import ddr_probe as dp
import ddr_probe_ep as dpe
import e1_probe as ep
import ep_aptos_train as epa
import pl_convnext as pl

REPO = os.path.dirname(os.path.abspath(dpe.__file__))
AUDIT = os.path.join(REPO, "datasets", "DDR", "audit")
ADAPTED = "ad" * 32
WEIGHTS = "checkpoints/best_a/model.weights.h5"


def fake_run(root, arm, seed, *, experiment=epa.EXPERIMENT, adapted=ADAPTED, model_type=None, content=None, started=None):
    """A pinned EP run directory with a stand-in weights file."""
    run_dir = epa.run_dir_for(root, arm, adapted, seed)
    os.makedirs(os.path.join(run_dir, "checkpoints", "best_a"), exist_ok=True)
    path = os.path.join(run_dir, *WEIGHTS.split("/"))
    with open(path, "wb") as fh:
        fh.write(content or f"{arm}-{seed}-weights".encode())
    model_type = model_type or dpe.ARMS[arm][1]
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump({"experiment": experiment, "arm": arm, "seed": seed, "model_type": model_type, "adapted_encoder_sha256": adapted}, fh)
    with open(os.path.join(run_dir, "frozen_checkpoint.json"), "w") as fh:
        json.dump({"experiment": experiment, "seed": seed, "weights": WEIGHTS, "sha256": dp.sha256_file(path)}, fh)
    with open(os.path.join(run_dir, "initialization.json"), "w") as fh:
        json.dump(started or {"arm": arm, "encoder_is_adapted": True, "adapted_encoder_sha256": adapted, "fresh_corn_head": True}, fh)
    return run_dir


def fake_experiment(root, arms=dpe.REQUIRED_ARMS):
    for arm in arms:
        for seed in dpe.SEEDS:
            fake_run(root, arm, seed)
    return dpe.build_checkpoint_manifest(root, ADAPTED, arms, repo_dir=REPO)


def criterion_record(path, per_seed, ci):
    return epa.write_e2_ep_eligibility(path, {"variants": {"five_epoch": {"mean": {"e1_ep_minus_p_ep": {
        "per_seed": per_seed, "mean": float(np.mean(per_seed)), "ci": ci, "positive_seeds": int(sum(v > 0 for v in per_seed))}}}}})


class ProtocolTests(unittest.TestCase):
    def test_the_probe_is_the_recorded_one_and_only_the_encoders_differ(self):
        self.assertEqual(dpe.SEEDS, (42, 123, 2026))
        self.assertEqual((dp.N_BOOT, dp.BOOT_SEED), (2000, 20260927))                 # 11: bootstrap seed / resample count
        self.assertEqual((ep.PROBE["epochs"], ep.PROBE["learning_rate"], ep.PROBE["batch_size"], dp.FIVE_EPOCHS), (5, 1e-3, 16, 5))   # 9
        self.assertEqual({k: dp.STOPPING[k] for k in ("min_delta", "patience", "min_epochs", "max_epochs")},
                         {"min_delta": 1e-4, "patience": 5, "min_epochs": 5, "max_epochs": 100})                                     # 10
        self.assertEqual((dp.EXPECTED_COUNTS, dp.EXCLUDED, dp.CLASSES), ({"train": 383, "val": 148, "test": 225}, ("007-5869-300.jpg",),
                                                                        ("MA", "HE", "EX", "SE")))
        self.assertEqual(dp.MANIFEST_SHA256, "2f3e4a40fa17e0af8706c859002163a18d56d594ca43f856a54cac65dbdcc2d0")
        self.assertEqual((dpe.PRIMARY, dpe.CONTRASTS), ("e1_ep_minus_p_ep", {"e1_ep_minus_p_ep": ("e1_ep", "p_ep")}))
        self.assertEqual(dpe.ARMS, {"p_ep": ("p", "P"), "e1_ep": ("e1", "E1"), "e2_ep": ("e1", "E1")})   # 7: model identification
        self.assertEqual(dpe.REQUIRED_ARMS, ("p_ep", "e1_ep"))
        source = inspect.getsource(dpe)
        for reused in ("dp.read_manifest(", "dp.prepare_data(", "dp.probe_encoder(", "dp.analyse("):
            self.assertIn(reused, source)                                              # the method is ddr_probe's, not a copy
        for redefined in ("def cell_targets", "def train_probes", "def load_targets", "def frame_from_raw", "build_probe(",
                          "N_BOOT =", "BOOT_SEED =", "STOPPING =", "Adam("):
            self.assertNotIn(redefined, source)
        self.assertNotIn("n_boot", inspect.signature(dpe.analyse).parameters)          # not adjustable for the EP analysis
        self.assertNotIn("boot_seed", inspect.signature(dpe.analyse).parameters)

    def test_targets_are_the_recorded_cell_targets(self):                              # 8
        mask = np.zeros((1900, 2500), np.uint8)
        mask[5, 7] = 255
        mask[1899, 2499] = 1
        target = dp.cell_targets(mask)
        np.testing.assert_array_equal(target, dp.cell_targets_bruteforce(mask))
        self.assertEqual((target.shape, float(target.sum()), float(target[0, 0]), float(target[15, 15])), ((16, 16), 2.0, 1.0, 1.0))

    def test_the_test_split_never_enters_probe_training(self):                         # 17
        self.assertNotIn("test", inspect.signature(dp.train_probes).parameters)
        step = inspect.getsource(dp.probe_encoder)
        self.assertEqual(step.count('f["test"]'), 1)
        self.assertIn('train_probes(seed, prior, f["train"], targets["train"], f["val"], targets["val"]', step)
        self.assertNotIn("train_probes", inspect.getsource(dpe))                       # the EP runner trains only through that step

    @unittest.skipUnless(os.path.isdir(AUDIT), "DDR audit folder not present")
    def test_the_locked_ddr_manifest_is_unchanged_and_never_written(self):             # 18
        self.assertEqual(dp.sha256_file(os.path.join(AUDIT, dp.MANIFEST_NAME)), dp.MANIFEST_SHA256)
        self.assertEqual({s: len(v) for s, v in dp.read_manifest(AUDIT).items()}, {"train": 383, "val": 148, "test": 225})
        for line in inspect.getsource(dpe).splitlines():
            if "audit_dir" in line and "def run(" not in line:
                self.assertIn("dp.read_manifest(audit_dir, lesion_root)", line)        # the only use: reading it


class CheckpointManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "exp")
        self.manifest = fake_experiment(self.root)
        self.verify = lambda m, **kw: dpe.verify_checkpoint_manifest(m, self.root, repo_dir=REPO, **kw)

    def tearDown(self):
        self.tmp.cleanup()

    def change(self, index, **fields):
        m = json.loads(json.dumps(self.manifest))
        m["checkpoints"][index].update(fields)
        return m

    def test_manifest_is_parsed_and_every_checkpoint_identified(self):                 # 1, 2, 7
        m = self.manifest
        self.assertEqual((m["probe_protocol_version"], m["ddr_manifest_sha256"], m["adapted_encoder_sha256"]),
                         (dpe.PROTOCOL_VERSION, dp.MANIFEST_SHA256, ADAPTED))
        self.assertTrue(m["git_commit"])
        self.assertEqual(sorted((e["arm"], e["seed"]) for e in m["checkpoints"]),
                         sorted((a, s) for a in ("p_ep", "e1_ep") for s in (42, 123, 2026)))
        checkpoints = self.verify(m)
        self.assertEqual(set(checkpoints), {(a, s) for a in ("p_ep", "e1_ep") for s in (42, 123, 2026)})
        self.assertEqual({(a, v["kind"], v["model_type"]) for (a, _), v in checkpoints.items()}, {("p_ep", "p", "P"), ("e1_ep", "e1", "E1")})
        for (arm, seed), v in checkpoints.items():
            self.assertEqual(dp.sha256_file(v["path"]), v["sha256"])
        path = os.path.join(self.tmp.name, "probe", "checkpoints.json")
        self.assertEqual(dpe.write_checkpoint_manifest(path, m), m)
        self.assertEqual(self.verify(dpe.write_checkpoint_manifest(path, m)).keys(), checkpoints.keys())
        with self.assertRaises(RuntimeError):                                          # a written manifest is not replaced
            dpe.write_checkpoint_manifest(path, self.change(0, sha256="0" * 64))
        self.assertNotIn("sha256\": \"", inspect.getsource(dpe))                       # no checkpoint hash is hard-coded

    def test_missing_and_duplicate_seeds_are_refused(self):                            # 3, 4
        m = json.loads(json.dumps(self.manifest))
        m["checkpoints"] = [e for e in m["checkpoints"] if not (e["arm"] == "e1_ep" and e["seed"] == 123)]
        with self.assertRaises(RuntimeError):
            self.verify(m)
        m = json.loads(json.dumps(self.manifest))
        m["checkpoints"].append(dict(m["checkpoints"][0]))
        with self.assertRaises(RuntimeError):
            self.verify(m)
        with self.assertRaises(RuntimeError):                                          # a seed outside the three
            self.verify(self.change(0, seed=7))
        m = json.loads(json.dumps(self.manifest))
        m["checkpoints"] = [e for e in m["checkpoints"] if e["arm"] == "p_ep"]         # an arm missing altogether
        with self.assertRaises(RuntimeError):
            self.verify(m)

    def test_wrong_hashes_versions_and_manifests_are_refused(self):                    # 5, 6
        with self.assertRaises(RuntimeError):
            self.verify(self.change(2, sha256="1" * 64))                               # not the run's pinned hash
        run_dir = epa.run_dir_for(self.root, "p_ep", ADAPTED, 42)
        with open(os.path.join(run_dir, *WEIGHTS.split("/")), "ab") as fh:
            fh.write(b"tampered")
        with self.assertRaises(RuntimeError):
            self.verify(self.manifest)                                                 # the file no longer has its SHA-256
        fake_run(self.root, "p_ep", 42)
        self.verify(self.manifest)
        with self.assertRaises(RuntimeError):
            self.verify(dict(self.manifest, ddr_manifest_sha256="2f3e4a40fa17e0af8706c859002163a18d56d594ca43f856a54cac65dbdcc2d"))
        with self.assertRaises(RuntimeError):
            self.verify(dict(self.manifest, probe_protocol_version="ddr-ground-truth-probe/v2"))
        with self.assertRaises(ValueError):
            self.verify({k: v for k, v in self.manifest.items() if k != "git_commit"})  # commit is required
        with self.assertRaises(ValueError):
            self.verify(dict(self.manifest, git_commit=None))

    def test_incompatible_architecture_and_non_ep_checkpoints_are_refused(self):
        with self.assertRaises(RuntimeError):
            self.verify(self.change(0, model_type="E1"))                               # P-EP declared with E1's architecture
        fake_run(self.root, "e1_ep", 42, model_type="P")                               # the run itself is of another type
        with self.assertRaises(RuntimeError):
            self.verify(self.manifest)
        fake_run(self.root, "e1_ep", 42)
        with self.assertRaises(RuntimeError):
            self.verify(self.change(0, arm="p"))                                       # 'p' is not an EP arm
        baseline = sorted(dpe.baseline_checkpoint_hashes(REPO))
        self.assertGreaterEqual(len(baseline), 10)                                     # 3 P + 3 E1 + 3 E2 + ImageNet
        with self.assertRaises(RuntimeError):
            self.verify(self.change(0, sha256=baseline[0]))                            # a pinned P / E1 / E2 checkpoint
        fake_run(self.root, "p_ep", 123, experiment="E1MultiTask")                     # a run of another experiment
        with self.assertRaises(RuntimeError):
            self.verify(self.manifest)
        fake_run(self.root, "p_ep", 123, started={"arm": "p_ep", "encoder_is_adapted": False, "adapted_encoder_sha256": ADAPTED,
                                                  "fresh_corn_head": True})           # started from ImageNet
        with self.assertRaises(RuntimeError):
            self.verify(self.manifest)
        fake_run(self.root, "p_ep", 123)
        self.verify(self.manifest)
        with self.assertRaises(RuntimeError):                                          # another adapted encoder
            self.verify(dict(self.manifest, adapted_encoder_sha256="be" * 32))

    def test_e2_ep_gate(self):                                                          # 13, 14, 15
        manifest = fake_experiment(self.root, ("p_ep", "e1_ep", "e2_ep"))
        gate = os.path.join(self.tmp.name, epa.ELIGIBILITY_NAME)
        self.assertEqual(dpe.e2_ep_gate_status(gate), {"recorded": False, "eligible": False,
                                                        "reason": "no E1-EP vs P-EP criterion result is recorded"})
        with self.assertRaises(RuntimeError):                                          # nothing recorded
            self.verify(manifest, eligibility=gate, e2_ep_confirmation=dpe.E2_EP_PROBE_CONFIRMATION)
        failed = os.path.join(self.tmp.name, "failed.json")
        criterion_record(failed, [0.02, -0.01, 0.03], [0.001, 0.03])
        self.assertFalse(dpe.e2_ep_gate_status(failed)["eligible"])
        with self.assertRaises(RuntimeError):                                          # criterion failed: refused, even confirmed
            self.verify(manifest, eligibility=failed, e2_ep_confirmation=dpe.E2_EP_PROBE_CONFIRMATION)
        criterion_record(gate, [0.02, 0.01, 0.03], [0.004, 0.03])
        status = dpe.e2_ep_gate_status(gate)
        self.assertEqual((status["eligible"], status["positive_seeds"]), (True, 3))    # eligible ...
        with self.assertRaises(RuntimeError):                                          # ... but not started without the phrase
            self.verify(manifest, eligibility=gate)
        with self.assertRaises(RuntimeError):
            self.verify(manifest, eligibility=gate, e2_ep_confirmation="yes")
        self.assertEqual(len(self.verify(manifest, eligibility=gate, e2_ep_confirmation=dpe.E2_EP_PROBE_CONFIRMATION)), 9)
        self.assertEqual(len(self.verify(self.manifest)), 6)                           # P-EP and E1-EP never need the gate
        # the same rule for TRAINING E2-EP: eligibility alone starts nothing
        self.assertIn("confirmation != E2_EP_CONFIRMATION", inspect.getsource(epa.run_arm))
        self.assertIsNone(inspect.signature(epa.run_arm).parameters["confirmation"].default)


class RunAndAnalysisTests(unittest.TestCase):
    """dpe.run with the encoder step replaced by synthetic features (the probe training itself is the real
    ddr_probe code), then the analysis and the gate record."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root, cls.out = os.path.join(cls.tmp.name, "exp"), os.path.join(cls.tmp.name, "probe")
        cls.manifest = fake_experiment(cls.root)
        rng = np.random.default_rng(0)
        counts = {"train": 48, "val": 24, "test": 30}
        cls.ids = {s: [(f"{s}{i:03d}.jpg", "0" * 64) for i in range(n)] for s, n in counts.items()}
        cls.targets = {s: (rng.random((n, 16, 16, 4)) < 0.08).astype(np.float32) for s, n in counts.items()}
        cls.calls = []
        pretrained = os.path.join(cls.tmp.name, "imagenet.h5")
        with open(pretrained, "wb") as fh:
            fh.write(b"stand-in")
        cls.pretrained = pretrained

        def probe_encoder(kind, seed, weights_path, reference_arrays, frames, ids, targets, prior, log=print):
            arm = "e1_ep" if "e1_ep" in weights_path else "p_ep"
            cls.calls.append((arm, kind, int(seed)))
            strength = 2.0 if arm == "e1_ep" else 1.0                      # synthetic: E1-EP features carry more signal
            noise = np.random.default_rng([int(seed), len(arm)])
            f = {s: np.concatenate([strength * targets[s], noise.normal(0, 1, targets[s].shape[:3] + (4,))], -1).astype(np.float16)
                 for s in dp.SPLITS}
            probe, weights, behaviour = dp.train_probes(seed, prior, f["train"], targets["train"], f["val"], targets["val"])
            result, arrays = {"behaviour": behaviour}, {"test_ids": np.asarray(ids["test"])}
            for variant in dp.VARIANTS:
                logits, test = dp.evaluate_probe(probe, weights[variant], f["test"], targets["test"])
                _, val = dp.evaluate_probe(probe, weights[variant], f["val"], targets["val"])
                _, train = dp.evaluate_probe(probe, weights[variant], f["train"], targets["train"])
                result[variant] = {"test": test, "val": val, "train": train}
                arrays[f"test_logits_{variant}"] = logits
            return result, arrays
        cls.patches = [mock.patch.object(dp, "read_manifest", lambda audit, lesion=None: cls.ids),
                       mock.patch.object(dp, "prepare_data", lambda lesion, ids, work, log=print: ({}, cls.targets)),
                       mock.patch.object(dp, "probe_encoder", probe_encoder),
                       mock.patch.object(pl, "load_reference", lambda path: (None, [])),
                       mock.patch.object(pl, "WEIGHTS_SHA256", dp.sha256_file(pretrained))]
        for p in cls.patches:
            p.start()
        cls.probe_run = staticmethod(lambda **kw: dpe.run(cls.root, "lesion", "audit", cls.out, cls.manifest, pretrained, repo_dir=REPO,
                                                    work_dir=os.path.join(cls.tmp.name, "work"), log=lambda *a: None, **kw))
        cls.summary = cls.probe_run()

    @classmethod
    def tearDownClass(cls):
        for p in cls.patches:
            p.stop()
        cls.tmp.cleanup()

    def test_1_outputs_and_schema(self):                                                # 16, Part 4, Part 6
        self.assertEqual(sorted(self.calls), sorted((a, k, s) for a, k in (("p_ep", "p"), ("e1_ep", "e1")) for s in (42, 123, 2026)))
        for name in ("configuration.json", "run_metadata.json", "targets.npz", "summary.json"):
            self.assertTrue(os.path.exists(os.path.join(self.out, name)))
        config = json.load(open(os.path.join(self.out, "configuration.json")))
        self.assertEqual((config["probe_protocol_version"], config["manifest_sha256"], config["primary"], config["arms"]),
                         (dpe.PROTOCOL_VERSION, dp.MANIFEST_SHA256, "e1_ep_minus_p_ep", ["p_ep", "e1_ep"]))
        self.assertEqual(config["bootstrap"], {"n": 2000, "seed": 20260927, "unit": "test image", "stratified": False})
        self.assertEqual((config["probe"], config["stopping"]), (json.loads(json.dumps(ep.PROBE)), json.loads(json.dumps(dp.STOPPING))))
        expected = {f"{e['arm']}_seed{e['seed']}": e["sha256"] for e in self.manifest["checkpoints"]}
        self.assertEqual(config["checkpoints"], expected)
        meta = json.load(open(os.path.join(self.out, "run_metadata.json")))["invocations"][0]
        for key in ("utc", "git_commit", "python", "tensorflow", "numpy", "keras", "gpus", "platform", "manifest_sha256", "checkpoints",
                    "probe_protocol_version", "bootstrap", "checkpoint_manifest_commit"):
            self.assertIn(key, meta)
        for key, sha in expected.items():
            r = self.summary["results"][key]
            self.assertEqual((r["weights_sha256"], f"{r['arm']}_seed{r['seed']}", r["model_type"]),
                             (sha, key, dpe.ARMS[r["arm"]][1]))
            self.assertEqual(set(r["behaviour"]) >= {"converged_epoch", "epochs_run", "val_loss_at_five", "val_loss_converged", "curve"}, True)
            for variant in dp.VARIANTS:
                self.assertEqual(set(r[variant]["test"]["scores"]), {"MA", "HE", "EX", "SE", "mean"})
                self.assertIn("loss", r[variant]["val"])
            with np.load(os.path.join(self.out, f"probe_{key}.npz")) as data:
                self.assertEqual(set(data.files), {"test_ids", "test_logits_five_epoch", "test_logits_converged"})
                self.assertEqual(data["test_logits_five_epoch"].shape, (30, 16, 16, 4))

    def test_2_resume_and_one_configuration_per_directory(self):
        calls = len(self.calls)
        again = self.probe_run()                                                             # finished pairs are kept
        self.assertEqual((len(self.calls), again["results"].keys()), (calls, self.summary["results"].keys()))
        self.assertEqual(len(json.load(open(os.path.join(self.out, "run_metadata.json")))["invocations"]), 2)
        fake_run(self.root, "p_ep", 42, content=b"another p-ep checkpoint")
        other = dpe.build_checkpoint_manifest(self.root, ADAPTED, repo_dir=REPO)
        try:
            with self.assertRaises(RuntimeError):                                      # other checkpoints, same directory
                dpe.run(self.root, "lesion", "audit", self.out, other, self.pretrained, repo_dir=REPO,
                        work_dir=os.path.join(self.tmp.name, "work"), log=lambda *a: None)
        finally:
            fake_run(self.root, "p_ep", 42)
        with self.assertRaises(RuntimeError):                                          # another DDR manifest: refused before anything
            dpe.run(self.root, "lesion", "audit", os.path.join(self.tmp.name, "p2"), dict(self.manifest, ddr_manifest_sha256="0" * 64),
                    self.pretrained, repo_dir=REPO, log=lambda *a: None)

    def test_3_primary_criterion_robustness_and_gate_record(self):                      # 12, 14, Part 4
        result = dpe.analyse(self.out)
        self.assertEqual((result["primary_variant"], result["n_boot"], result["boot_seed"], result["test_images"]),
                         ("five_epoch", 2000, 20260927, 30))
        self.assertEqual(result["criterion"], dpe.CRITERION)
        five, converged = result["variants"]["five_epoch"], result["variants"]["converged"]
        contrast = five["mean"]["e1_ep_minus_p_ep"]
        self.assertEqual(set(contrast), {"mean", "ci", "per_seed", "positive_seeds", "ci_excludes_zero", "per_class"})
        self.assertEqual(len(contrast["per_seed"]), 3)
        per_seed = [five["per_seed"][s]["e1_ep"]["mean"] - five["per_seed"][s]["p_ep"]["mean"] for s in (42, 123, 2026)]
        np.testing.assert_allclose(contrast["per_seed"], per_seed, atol=1e-12)
        self.assertAlmostEqual(contrast["mean"], float(np.mean(per_seed)), places=12)
        self.assertEqual(contrast["positive_seeds"], sum(v > 0 for v in per_seed))
        met = contrast["positive_seeds"] == 3 and contrast["ci"][0] > 0
        self.assertEqual((five["criterion_met"], result["criterion_met"]), (met, met))  # the criterion is the five-epoch one
        self.assertTrue(met)                                                           # (synthetic features built that way)
        self.assertIn("e1_ep_minus_p_ep", converged["mean"])                           # robustness reported next to it
        self.assertEqual(set(five["mean"]), {"p_ep", "e1_ep", "e1_ep_minus_p_ep"})
        self.assertEqual(set(five["mean"]["p_ep"]), {"MA", "HE", "EX", "SE", "mean"})
        saved = json.load(open(os.path.join(self.out, dpe.RESULT_NAME)))
        self.assertEqual(saved, json.loads(json.dumps(dpe.analyse(self.out), default=float)))   # reproducible
        self.assertFalse(os.path.exists(os.path.join(self.out, "ddr_probe_result.json")))       # the §70 file name is not used
        # a converged result cannot stand in for the primary one
        forged = json.loads(json.dumps(saved))
        forged["variants"]["five_epoch"]["mean"]["e1_ep_minus_p_ep"].update(positive_seeds=2)
        gate = os.path.join(self.tmp.name, "gate_forged.json")
        self.assertFalse(epa.write_e2_ep_eligibility(gate, forged)["criterion_met"])   # read from five_epoch only
        gate = os.path.join(self.tmp.name, epa.ELIGIBILITY_NAME)
        record = dpe.record_e2_ep_gate(self.out, gate)
        self.assertEqual((record["criterion_met"], record["contrast"], record["primary_variant"]), (True, "e1_ep_minus_p_ep", "five_epoch"))
        self.assertTrue(dpe.e2_ep_gate_status(gate)["eligible"])
        self.assertEqual(epa.require_e2_ep_eligibility(gate)["positive_seeds"], 3)

    def test_4_descriptive_comparison_with_the_recorded_p_and_e1_probes(self):
        base = os.path.join(self.tmp.name, "baseline")
        os.makedirs(base)
        with np.load(os.path.join(self.out, "targets.npz")) as data:
            np.savez_compressed(os.path.join(base, "targets.npz"), **{k: data[k] for k in data.files})
            targets = data["test_targets"]
        rng = np.random.default_rng(3)
        for seed in dpe.SEEDS:
            for model in ("p", "e1"):
                z = (rng.normal(0, 1, targets.shape) + targets).astype(np.float32)
                np.savez_compressed(os.path.join(base, f"probe_{model}_seed{seed}.npz"), test_ids=np.asarray([n for n, _ in self.ids["test"]]),
                                    test_logits_five_epoch=z, test_logits_converged=z)
        result = dpe.analyse_against_baseline(self.out, base)
        self.assertEqual(set(result["variants"]["five_epoch"]["mean"]) & set(dpe.BASELINE_CONTRASTS), set(dpe.BASELINE_CONTRASTS))
        self.assertNotIn("criterion_met", result)
        self.assertNotIn("criterion_met", json.load(open(os.path.join(self.out, dpe.BASELINE_RESULT_NAME))))
        with np.load(os.path.join(base, "targets.npz")) as data:
            changed = {k: data[k] for k in data.files}
        changed["test_ids"] = changed["test_ids"][::-1]
        np.savez_compressed(os.path.join(base, "targets.npz"), **changed)
        with self.assertRaises(RuntimeError):                                          # other test images: no comparison
            dpe.analyse_against_baseline(self.out, base)


if __name__ == "__main__":
    unittest.main()
