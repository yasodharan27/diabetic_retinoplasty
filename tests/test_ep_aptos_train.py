"""CPU tests for ep_aptos_train (P-EP, E1-EP, E2-EP on APTOS from the pinned EyePACS-adapted encoder). Random
ConvNeXt weights, a two-epoch synthetic 'adaptation' and the 5-image synthetic bundle only: nothing here is an
experiment and nothing produces a result. The one-epoch runs prove the start-from-adapted-encoder, fresh-head,
pinning and ordering mechanics."""
import inspect
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np

import arch1_data as ad
import arch1_model as a1
import arch1_train as at
import e1_data as ed
import e1_model as em
import e1_train as e1t
import e2_control as e2
import ep_aptos_train as ep
import eyepacs_adaptation_train as eat
import pipeline_v2_config as v2cfg
import pl_convnext as pl
import weighted_corn

try:
    from tests import test_eyepacs_adaptation_train as tea
    from tests import v2_bundle_fixture as fx
except ImportError:
    import test_eyepacs_adaptation_train as tea
    import v2_bundle_fixture as fx

REPO = os.path.dirname(os.path.abspath(ep.__file__))
CW = list(weighted_corn.PREREGISTERED_CLASS_WEIGHTS)
PRIOR = (0.0718, 0.1099, 0.1089, 0.0547)
SMALL = tea.SMALL
IMAGENET = tea.REF_ARRAYS


def fake_adapted(shift=0.01):
    """An 'adapted' encoder for the structure tests: the reference arrays, perturbed."""
    arrays = [a + np.float32(shift) for a in IMAGENET]
    return {"arrays": arrays, "sha256": "c" * 64, "digest": ep.encoder_digest(arrays),
            "imagenet_digest": ep.encoder_digest(IMAGENET), "record": {}}


class ProtocolTests(unittest.TestCase):
    def test_arms_seeds_and_protocol(self):
        self.assertEqual(ep.SEEDS, (42, 123, 2026))                        # three downstream APTOS seeds
        self.assertIs(ep.PROTOCOL, at.P_PROTOCOL)                          # the exact P protocol, not a copy
        self.assertEqual((ep.PROTOCOL["batch_size"], ep.PROTOCOL["max_epochs"], ep.PROTOCOL["early_stopping_patience"]), (2, 50, 12))
        self.assertEqual({a: (v["mandatory"], v["requires"]) for a, v in ep.ARMS.items()},
                         {"p_ep": (True, ()), "e1_ep": (True, ("p_ep",)), "e2_ep": (False, ("p_ep", "e1_ep"))})
        self.assertEqual(e1t.LAMBDA, 1.0)
        self.assertEqual((ep.E2_EP_CONDITION["contrast"], ep.E2_EP_CONDITION["primary_variant"]), ("e1_ep_minus_p_ep", "five_epoch"))
        with self.assertRaises(ValueError):
            ep.run_dir_for("x", "p_ep", "a" * 64, 7)
        with self.assertRaises(ValueError):
            ep.run_dir_for("x", "e3_ep", "a" * 64, 42)

    def test_no_eyepacs_test_ddr_or_idrid_image_can_enter_an_aptos_run(self):
        source = inspect.getsource(ep)
        for forbidden in ("ddr_probe", "idrid", "FrameCache", "read_test_manifest", "evaluate_heldout_test", 'entries("test")',
                          "Label_EyeQ", "image_quality"):
            self.assertNotIn(forbidden, source)
        # the held-out result is consulted in one place, for its existence and checkpoint hash only -- never its metrics
        self.assertEqual(source.count("heldout_result("), 1)
        self.assertNotIn('heldout["metrics"]', source)

    def test_e2_ep_needs_a_recorded_criterion_that_was_met(self):
        result = lambda per_seed, ci: {"variants": {"five_epoch": {"mean": {"e1_ep_minus_p_ep": {   # noqa: E731
            "per_seed": per_seed, "mean": float(np.mean(per_seed)), "ci": ci, "positive_seeds": int(sum(v > 0 for v in per_seed))}}}}}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ep.ELIGIBILITY_NAME)
            with self.assertRaises(RuntimeError):
                ep.require_e2_ep_eligibility(path)                         # nothing recorded
            with self.assertRaises(RuntimeError):
                ep.require_e2_ep_eligibility(None)
            met = ep.write_e2_ep_eligibility(path, result([0.02, 0.03, 0.01], [0.005, 0.03]))
            self.assertTrue(met["criterion_met"])
            self.assertEqual(ep.require_e2_ep_eligibility(path)["positive_seeds"], 3)
            with self.assertRaises(RuntimeError):                          # a recorded result is not overwritten
                ep.write_e2_ep_eligibility(path, result([0.02, -0.03, 0.01], [0.005, 0.03]))
            for name, bad in (("two_seeds", result([0.02, -0.01, 0.03], [0.001, 0.03])),
                              ("interval", result([0.02, 0.01, 0.03], [-0.001, 0.03]))):
                other = os.path.join(tmp, name + ".json")
                self.assertFalse(ep.write_e2_ep_eligibility(other, bad)["criterion_met"])
                with self.assertRaises(RuntimeError):
                    ep.require_e2_ep_eligibility(other)
            forged = os.path.join(tmp, "forged.json")
            with open(forged, "w") as fh:                                  # another contrast claiming success
                json.dump(dict(met, contrast="e1_minus_p"), fh)
            with self.assertRaises(RuntimeError):
                ep.require_e2_ep_eligibility(forged)


class InitialisationTests(unittest.TestCase):
    def test_p_ep_starts_from_the_adapted_encoder_with_a_fresh_head(self):
        adapted = fake_adapted()
        for seed in (42, 123):
            model = ep.build_model("p_ep", seed, adapted, CW, mixed_precision=False, image_size=SMALL)
            facts = ep.assert_initialisation("p_ep", model, seed, adapted)
            self.assertEqual((facts["encoder_is_adapted"], facts["encoder_is_imagenet"], facts["fresh_corn_head"],
                              facts["model_type"], facts["adapted_encoder_sha256"]), (True, False, True, "P", "c" * 64))
            for a, v in zip(adapted["arrays"], model.get_layer(pl.BACKBONE_NAME).weights):
                np.testing.assert_array_equal(a, v.numpy())
            self.assertEqual(model.count_params(), pl.EXPECTED_BACKBONE_PARAMETERS["P"] + pl.EXPECTED_HEAD_PARAMETERS)
        imagenet_model = pl.build_pl_model("P", 42, IMAGENET, image_size=SMALL)     # an accidental ImageNet reset
        with self.assertRaises(RuntimeError):
            ep.assert_initialisation("p_ep", imagenet_model, 42, adapted)
        with self.assertRaises(RuntimeError):                               # a head of another seed is not fresh for this one
            ep.assert_initialisation("p_ep", ep.build_model("p_ep", 123, adapted, CW, mixed_precision=False, image_size=SMALL), 42, adapted)
        trained = ep.build_model("p_ep", 42, adapted, CW, mixed_precision=False, image_size=SMALL)
        layer = trained.get_layer("corn").get_layer("corn_logits")
        layer.set_weights([w + 0.1 for w in layer.get_weights()])           # e.g. a carried-over EyePACS head
        with self.assertRaises(RuntimeError):
            ep.assert_initialisation("p_ep", trained, 42, adapted)

    def test_e1_ep_is_e1_started_from_the_adapted_encoder(self):
        adapted = fake_adapted()
        model = ep.build_model("e1_ep", 42, adapted, CW, PRIOR, mixed_precision=False, image_size=SMALL)
        facts = ep.assert_initialisation("e1_ep", model, 42, adapted, PRIOR)
        self.assertEqual((facts["model_type"], facts["encoder_is_adapted"], facts["fresh_corn_head"], facts["fresh_lesion_head"],
                          facts["encoder_arrays"]), ("E1", True, True, True, len(IMAGENET)))
        self.assertEqual(tuple(model.output_names), (em.GRADING_OUTPUT, em.LESION_OUTPUT))
        reference = em.build_e1_model(42, PRIOR, image_size=SMALL)          # E1's own architecture and head initialisation
        self.assertEqual(model.count_params(), reference.count_params())
        self.assertEqual(model.count_params() - pl.EXPECTED_BACKBONE_PARAMETERS["P"] - pl.EXPECTED_HEAD_PARAMETERS, 3076)
        for a, b in zip(model.get_layer(em.LESION_CONV_NAME).get_weights(), reference.get_layer(em.LESION_CONV_NAME).get_weights()):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(model.get_layer(em.LESION_CONV_NAME).get_weights()[1], em.prior_logits(PRIOR), atol=1e-6)
        self.assertEqual(model.loss[1], e1t.lesion_loss) if isinstance(model.loss, (list, tuple)) else None
        p_ep = ep.build_model("p_ep", 42, adapted, CW, mixed_precision=False, image_size=SMALL)
        self.assertEqual(ep.encoder_digest(ep._encoder_arrays("e1_ep", model)), ep.encoder_digest(ep._encoder_arrays("p_ep", p_ep)))
        x = np.random.default_rng(0).random((2, SMALL, SMALL, 3)).astype(np.float32)
        np.testing.assert_allclose(model.predict_on_batch({"rgb": x})[0], p_ep.predict_on_batch(eat.p_inputs(x)), atol=1e-5)
        with self.assertRaises(RuntimeError):                               # E1 built from ImageNet is not E1-EP
            ep.assert_initialisation("e1_ep", e1t.build_compiled_model(42, PRIOR, None, CW, mixed_precision=False, image_size=SMALL),
                                     42, adapted, PRIOR)
        with self.assertRaises(ValueError):
            ep.build_model("e1_ep", 42, adapted, CW, None, mixed_precision=False, image_size=SMALL)
        with self.assertRaises(ValueError):
            ep.build_model("e3_ep", 42, adapted, CW, PRIOR, mixed_precision=False, image_size=SMALL)

    def test_run_identity_records_arm_encoder_split_and_lambda(self):
        class Bundle:
            bundle = {"split_sha256": "s" * 64, "bundle_id": "b", "population_sha256": "p" * 64}
            fingerprint, stage4_sha256, stage4_generation = "f" * 64, "4" * 64, "g"
        adapted = fake_adapted()
        p = ep.run_mapping("p_ep", Bundle, 42, adapted, CW)
        e1 = ep.run_mapping("e1_ep", Bundle, 42, adapted, CW, PRIOR)
        e2m = ep.run_mapping("e2_ep", Bundle, 42, adapted, CW, PRIOR, {"fixed_points": 0})
        for m in (p, e1, e2m):
            self.assertEqual((m["adapted_encoder_sha256"], m["split_sha256"], m["batch_size"], m["checkpoint_selection"]),
                             ("c" * 64, "s" * 64, 2, ep.SELECTION))
        self.assertNotIn("lesion_loss_weight", p)
        self.assertEqual((e1["lesion_loss_weight"], e1["training_targets"], e1["lesion_target"]["source_channels"]),
                         (1.0, "aligned", ["MA:max", "HE:max", "EX:max", "SE:max"]))
        self.assertEqual((e2m["training_targets"], e2m["derangement"]), ("fixed derangement (e2_control)", {"fixed_points": 0}))
        self.assertEqual(len({at.config_hash(m) for m in (p, e1, e2m)}), 3)
        other = dict(adapted, sha256="d" * 64)
        self.assertNotEqual(at.config_hash(ep.run_mapping("p_ep", Bundle, 42, other, CW)), at.config_hash(p))


class RunTests(unittest.TestCase):
    """A synthetic adaptation (64 px) -> its pinned encoder -> one-epoch P-EP and E1-EP runs on the 5-image bundle."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        tmp = cls.tmp.name
        b = fx.build(os.path.join(tmp, "data"))
        cls.bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=b["roots"], expected_population=5)
        cls.grade_of = dict(fx.TRAIN + fx.VAL)
        cls.exp = os.path.join(tmp, "exp")
        frames = tea.FakeFrames()
        tea.small_adaptation(cls.exp, frames, tmp)
        cls.adaptation_dir = eat.run_dir_for(cls.exp, frames.manifest_sha256)
        cls.kw = dict(repo_dir=REPO, staging_root=os.path.join(tmp, "staging"), log=lambda *a: None, grade_of=cls.grade_of)

        def train(arm, run_dir, bundle, seed, adapted, class_weights, **kw):
            return ep.train_seed(arm, run_dir, bundle, seed, adapted, class_weights, max_epochs=1, mixed_precision=False, **kw)

        def evaluate(arm, run_dir, bundle, seed, adapted, class_weights, **kw):
            return ep.evaluate_run(arm, run_dir, bundle, seed, adapted, class_weights, mixed_precision=False, **kw)
        cls.train, cls.evaluate = staticmethod(train), staticmethod(evaluate)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_1_order_the_encoder_is_released_only_after_the_heldout_evaluation(self):
        with self.assertRaises(RuntimeError):                               # pinned, but the held-out test is not recorded
            ep.load_adapted_encoder(self.adaptation_dir, IMAGENET, image_size=SMALL)
        with self.assertRaises(RuntimeError):                               # a directory without a pinned checkpoint
            ep.load_adapted_encoder(os.path.join(self.tmp.name, "nothing"), IMAGENET, image_size=SMALL)
        _, frozen = eat.read_frozen(self.adaptation_dir)
        os.makedirs(os.path.join(self.adaptation_dir, eat.HELDOUT_DIR))
        with open(os.path.join(self.adaptation_dir, eat.HELDOUT_DIR, "metrics.json"), "w") as fh:
            json.dump({"checkpoint_sha256": "0" * 64, "metrics": {"qwk": 0.0}}, fh)
        with self.assertRaises(RuntimeError):                               # a held-out result of another checkpoint
            ep.load_adapted_encoder(self.adaptation_dir, IMAGENET, image_size=SMALL)
        with open(os.path.join(self.adaptation_dir, eat.HELDOUT_DIR, "metrics.json"), "w") as fh:
            json.dump({"checkpoint_sha256": frozen["sha256"], "metrics": {"qwk": 0.0}}, fh)
        with self.assertRaises(RuntimeError):
            ep.load_adapted_encoder(self.adaptation_dir, IMAGENET, expected_sha256="1" * 64, image_size=SMALL)
        adapted = ep.load_adapted_encoder(self.adaptation_dir, IMAGENET, expected_sha256=frozen["sha256"], image_size=SMALL)
        self.assertEqual((adapted["sha256"], len(adapted["arrays"])), (frozen["sha256"], len(IMAGENET)))
        self.assertNotEqual(adapted["digest"], adapted["imagenet_digest"])
        type(self).adapted = adapted

    def test_2_p_ep_and_e1_ep_runs(self):
        adapted, bundle = self.adapted, self.bundle
        with self.assertRaises(RuntimeError):                               # E1-EP before P-EP is complete
            ep.run_arm("e1_ep", self.exp, bundle, adapted, CW, lesion_prior=PRIOR, train_fn=self.train, evaluate_fn=self.evaluate, **self.kw)
        seeds = ep.SEEDS
        try:
            ep.SEEDS = (42,)                                                # one seed is enough for the mechanics
            p = ep.run_arm("p_ep", self.exp, bundle, adapted, CW, train_fn=self.train, evaluate_fn=self.evaluate, **self.kw)
            e1 = ep.run_arm("e1_ep", self.exp, bundle, adapted, CW, lesion_prior=PRIOR, train_fn=self.train,
                            evaluate_fn=self.evaluate, **self.kw)
            with self.assertRaises(RuntimeError):                           # E2-EP without the criterion record
                ep.run_arm("e2_ep", self.exp, bundle, adapted, CW, lesion_prior=PRIOR, partner=e2.derangement(bundle.train_ids),
                           train_fn=self.train, evaluate_fn=self.evaluate, **self.kw)
            gate = os.path.join(self.tmp.name, ep.ELIGIBILITY_NAME)          # a met criterion makes E2-EP eligible ...
            ep.write_e2_ep_eligibility(gate, {"variants": {"five_epoch": {"mean": {"e1_ep_minus_p_ep": {
                "per_seed": [0.02], "mean": 0.02, "ci": [0.005, 0.03], "positive_seeds": 1}}}}})   # one seed in this test
            started = []
            for phrase in (None, "yes"):                                     # ... but nothing starts without the phrase
                with self.assertRaisesRegex(RuntimeError, "not confirmed"):
                    ep.run_arm("e2_ep", self.exp, bundle, adapted, CW, lesion_prior=PRIOR, partner=e2.derangement(bundle.train_ids),
                               eligibility=gate, confirmation=phrase, train_fn=lambda *a, **k: started.append(a),
                               evaluate_fn=self.evaluate, **self.kw)
            self.assertEqual((started, ep.E2_EP_CONFIRMATION), ([], "TRAIN E2-EP ON APTOS"))
            with self.assertRaises(RuntimeError):
                ep.train_seed("e2_ep", os.path.join(self.tmp.name, "e2"), bundle, 42, adapted, CW, repo_dir=REPO,
                              staging_dir=os.path.join(self.tmp.name, "s"), lesion_prior=PRIOR,
                              partner=e2.derangement(bundle.train_ids), grade_of=self.grade_of, log=lambda *a: None)
            calls = []
            kept = ep.run_arm("p_ep", self.exp, bundle, adapted, CW, train_fn=lambda *a, **k: calls.append(a),
                              evaluate_fn=self.evaluate, **self.kw)
            self.assertEqual((calls, kept[42]["frozen"]), ([], p[42]["frozen"]))   # a finished seed is kept
        finally:
            ep.SEEDS = seeds
        for arm, result in (("p_ep", p[42]), ("e1_ep", e1[42])):
            run_dir = ep.run_dir_for(self.exp, arm, adapted["sha256"], 42)
            with open(os.path.join(run_dir, "initialization.json")) as fh:
                started = json.load(fh)
            self.assertEqual((started["arm"], started["encoder_is_adapted"], started["encoder_is_imagenet"], started["fresh_corn_head"],
                              started["adapted_encoder_sha256"], started["seed"]), (arm, True, False, True, adapted["sha256"], 42))
            cfg = result["config"]
            self.assertEqual((cfg["experiment"], cfg["arm"], cfg["split_sha256"], cfg["batch_size"], cfg["max_epochs"]),
                             (ep.EXPERIMENT, arm, bundle.bundle["split_sha256"], 2, 50))
            self.assertEqual(result["best"]["checkpoint"]["weights_sha256"], result["frozen"]["sha256"])
            self.assertEqual(result["frozen"]["identity"]["adapted_encoder_sha256"], adapted["sha256"])
            self.assertIn("qwk", result["best"]["metrics"])
            self.assertEqual(len(result["history"]), 1)
        self.assertEqual(e1[42]["config"]["lesion_loss_weight"], 1.0)
        self.assertIn("lesion_head_validation", e1[42]["best"])
        self.assertTrue(json.load(open(os.path.join(ep.run_dir_for(self.exp, "e1_ep", adapted["sha256"], 42),
                                                    "initialization.json")))["fresh_lesion_head"])
        with self.assertRaises(RuntimeError):                               # another encoder in the same run directory
            self.train("p_ep", ep.run_dir_for(self.exp, "p_ep", adapted["sha256"], 42), bundle, 42, dict(adapted, sha256="9" * 64),
                       CW, repo_dir=REPO, staging_dir=os.path.join(self.tmp.name, "s3"), grade_of=self.grade_of, log=lambda *a: None)

    def test_3_p_ep_batches_are_ps_frames_order_and_augmentation(self):
        import improved_training_data as itd
        bundle = self.bundle
        train_entries, val_entries = ep.entries(bundle, self.grade_of)
        self.assertEqual(([i for i, _ in train_entries], [i for i, _ in val_entries]), (list(bundle.train_ids), list(bundle.val_ids)))
        frames = ep.BundleFrames(bundle)
        x, y = eat.make_epoch_sequence(frames, train_entries, 1, 42, 2, True)[0]
        e1_x, (e1_y, _) = ed.make_epoch_sequence(bundle, train_entries, 1, 42, 2, augment=True)[0]
        np.testing.assert_array_equal(x["stage5_input"][..., :3], e1_x["rgb"])     # the frames E1 / P see, same order,
        np.testing.assert_array_equal(y, e1_y)                                      # same per-image augmentation
        self.assertEqual([g for _, g in itd.epoch_training_order(train_entries, 42, 1)][:2], list(y))


if __name__ == "__main__":
    unittest.main()
