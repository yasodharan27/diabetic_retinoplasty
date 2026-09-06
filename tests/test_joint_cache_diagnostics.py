"""
Tests for `joint_cache_diagnostics.py` -- the measurement-only instrumentation of the Phase 1
joint-training cache path.

Per `PROJECT_CODE.md`'s Implementation Rules this suite uses synthetic/temporary data only: the
same synthetic fundus images, synthetic (untrained) Stage 03 vessel model and synthetic
(untrained) Stage 04 Attention U-Net that `tests/test_joint_training.py` already builds, never a
real dataset or a real gitignored checkpoint. No Drive is mounted; the "persistent" cache is a
second temp directory, which is exactly what `persistent_cache_dir` is -- a directory path the
pipeline stats and reads. NO TRAINING IS RUN: no `fit()`, no epoch, no checkpoint.

The central obligation of a diagnostic is that it must not perturb what it measures. The first
test class below establishes that directly: it runs the real Phase 1 twice from identical
starting states, once instrumented and once not, and asserts the resulting cache files are
byte-for-byte identical -- so every number the other tests assert on describes the real pipeline,
not an instrumented variant of it.
"""

import csv
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

import joint_cache_diagnostics as jcd
import joint_training_dataset as jtd
import lesion_segmentation_model as lsm
import local_feature_extraction_dataset as lfed
import racaf
import torch
from vessel_segmentation_model import build_vessel_segmentation_model, load_state_dict_from_checkpoint

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_SIZE = 256  # large enough for Stage 03's FOV circle-fit to succeed reliably
CANONICAL_SIZE = jtd.STAGE5_IMAGE_SIZE  # (512, 512) -- NOT reducible: `racaf.prepare_stage4_input`
# resizes to this, and Stage 04 is a fixed `(512,512,4)` graph, so a smaller size makes the real
# code path raise rather than run. These tests therefore measure the real canonical resolution.


def _synthetic_fundus_image(size=IMAGE_SIZE, seed=0):
    """Identical recipe to test_joint_training.py's helper of the same name."""
    rng = np.random.RandomState(seed)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    yy, xx = np.ogrid[:size, :size]
    center = size // 2
    radius = int(size * 0.4)
    circle = (xx - center) ** 2 + (yy - center) ** 2 <= radius ** 2
    base = 140 + rng.randint(-20, 20, size=(size, size, 3))
    image[circle] = np.clip(base[circle], 60, 220).astype(np.uint8)
    return image


def _build_synthetic_vessel_model():
    model = build_vessel_segmentation_model()
    tmp_dir = tempfile.mkdtemp(prefix="synthetic_vessel_ckpt_")
    checkpoint_path = os.path.join(tmp_dir, "synthetic_checkpoint.pth")
    torch.save(
        {"model_state_dict": model.state_dict(), "optimizer_state_dict": {}, "stats": None},
        checkpoint_path,
    )
    loaded = load_state_dict_from_checkpoint(model, checkpoint_path, device="cpu")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return loaded


def _build_synthetic_frozen_stage4_model():
    """A real (untrained) Attention U-Net at Stage 04's REAL native input shape `(512,512,4)` --
    required since `racaf.prepare_stage4_input`/`tta_views` always resize to that fixed
    resolution. `trainable=False` reproduces `racaf.load_frozen_stage4_model()`'s post-condition
    without needing its real, gitignored `.keras` file."""
    model = lsm.build_attention_unet(input_shape=(512, 512, 4), base_filters=4)
    model.trainable = False
    return model


class _SyntheticTree:
    """A temp APTOS-shaped tree plus the four directories the diagnostic distinguishes: a local
    cache pair and a "persistent" cache pair standing in for Drive."""

    def __init__(self, pairs):
        self.root = tempfile.mkdtemp(prefix="aptos_jcd_")
        self.image_dir = os.path.join(self.root, "train_images")
        self.csv_path = os.path.join(self.root, "train.csv")
        self.cache_dir = os.path.join(self.root, "local_cache")
        self.racaf_cache_dir = os.path.join(self.root, "local_racaf")
        self.drive_cache_dir = os.path.join(self.root, "drive_cache")
        self.drive_racaf_cache_dir = os.path.join(self.root, "drive_racaf")
        self.processed_dir = os.path.join(self.root, "no_precomputed_stage02")
        for directory in (self.image_dir, self.cache_dir, self.racaf_cache_dir,
                          self.drive_cache_dir, self.drive_racaf_cache_dir):
            os.makedirs(directory, exist_ok=True)
        self.pairs = list(pairs)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id_code", "diagnosis"])
            for i, (id_code, diagnosis) in enumerate(self.pairs):
                writer.writerow([id_code, diagnosis])
                Image.fromarray(_synthetic_fundus_image(seed=i)).save(
                    os.path.join(self.image_dir, id_code + ".png"))

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)


class _DiagnosticTestBase(unittest.TestCase):
    """Shared fixture: one synthetic image, real (synthetic) frozen models built once per class."""

    PAIRS = [("jcd_a", 0), ("jcd_b", 2)]

    @classmethod
    def setUpClass(cls):
        cls.vessel_model = _build_synthetic_vessel_model()
        cls.stage4_model = _build_synthetic_frozen_stage4_model()

    def setUp(self):
        self.tree = _SyntheticTree(self.PAIRS)
        self.addCleanup(self.tree.cleanup)

    def run_phase1(self, entries, cache_dir=None, racaf_cache_dir=None,
                   persistent_cache_dir=None, persistent_racaf_cache_dir=None):
        """The real, uninstrumented Phase 1 -- used to build starting states for the diagnostic to
        then measure, exactly as a prior real run would have left them."""
        return jtd.precompute_joint_frozen_caches(
            entries,
            image_dir=self.tree.image_dir,
            cache_dir=cache_dir if cache_dir is not None else self.tree.cache_dir,
            racaf_cache_dir=racaf_cache_dir if racaf_cache_dir is not None else self.tree.racaf_cache_dir,
            persistent_cache_dir=persistent_cache_dir,
            persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            vessel_model=self.vessel_model,
            stage4_model=self.stage4_model,
            processed_dir=self.tree.processed_dir,
            image_size=CANONICAL_SIZE,
            progress_every=0,
        )

    def diagnose(self, **overrides):
        kwargs = dict(
            entries=self.tree.pairs,
            image_dir=self.tree.image_dir,
            cache_dir=self.tree.cache_dir,
            racaf_cache_dir=self.tree.racaf_cache_dir,
            persistent_cache_dir=self.tree.drive_cache_dir,
            persistent_racaf_cache_dir=self.tree.drive_racaf_cache_dir,
            processed_dir=self.tree.processed_dir,
            vessel_model=self.vessel_model,
            stage4_model=self.stage4_model,
            image_size=CANONICAL_SIZE,
            max_images=1,
        )
        kwargs.update(overrides)
        return jcd.run_diagnostic(**kwargs)


# =====================================================================
# The diagnostic must not perturb what it measures.
# =====================================================================

class InstrumentationIsNonPerturbingTests(_DiagnosticTestBase):
    def test_instrumented_phase1_writes_byte_identical_cache_files(self):
        """The whole diagnostic rests on this: run the REAL Phase 1 twice from identical empty
        starting states -- once plain, once under `_instrument` -- and the cache files it produces
        must be byte-for-byte identical. If instrumentation changed any computed value, this fails."""
        plain_cache = os.path.join(self.tree.root, "plain_cache")
        plain_racaf = os.path.join(self.tree.root, "plain_racaf")
        self.run_phase1(self.tree.pairs[:1], cache_dir=plain_cache, racaf_cache_dir=plain_racaf)

        recorder = jcd._Recorder(jcd._Roots([("local", self.tree.cache_dir, False)]))
        with jcd._instrument(recorder):
            self.run_phase1(self.tree.pairs[:1])

        plain_files = sorted(os.listdir(plain_cache))
        instrumented_files = sorted(os.listdir(self.tree.cache_dir))
        self.assertEqual(plain_files, instrumented_files)
        self.assertTrue(plain_files, "expected Phase 1 to have written cache files")
        for name in plain_files:
            with open(os.path.join(plain_cache, name), "rb") as handle:
                plain_bytes = handle.read()
            with open(os.path.join(self.tree.cache_dir, name), "rb") as handle:
                instrumented_bytes = handle.read()
            self.assertEqual(plain_bytes, instrumented_bytes,
                             "instrumented run produced different bytes for " + name)

        plain_npz = sorted(os.listdir(plain_racaf))
        self.assertEqual(plain_npz, sorted(os.listdir(self.tree.racaf_cache_dir)))
        for name in plain_npz:
            plain_reliability = np.load(os.path.join(plain_racaf, name))
            instrumented_reliability = np.load(os.path.join(self.tree.racaf_cache_dir, name))
            np.testing.assert_array_equal(plain_reliability["kappa"], instrumented_reliability["kappa"])
            np.testing.assert_array_equal(plain_reliability["r"], instrumented_reliability["r"])

    def test_every_patched_attribute_is_restored_afterward(self):
        originals = {
            ("jtd", "os"): jtd.os, ("jtd", "np"): jtd.np,
            ("jtd", "_resize_rgb_01"): jtd._resize_rgb_01,
            ("jtd", "predict_vessel_mask"): jtd.predict_vessel_mask,
            ("lfed", "_load_raw_bgr"): lfed._load_raw_bgr,
            ("lfed", "_resolve_processed_rgb"): lfed._resolve_processed_rgb,
            ("racaf", "prepare_stage4_input"): racaf.prepare_stage4_input,
            ("racaf", "tta_views"): racaf.tta_views,
            ("racaf", "compute_reliability"): racaf.compute_reliability,
        }
        recorder = jcd._Recorder(jcd._Roots([]))
        with jcd._instrument(recorder):
            self.assertIsNot(jtd.os, originals[("jtd", "os")], "expected os to be patched inside")
        modules = {"jtd": jtd, "lfed": lfed, "racaf": racaf}
        for (module_name, attribute), original in originals.items():
            self.assertIs(getattr(modules[module_name], attribute), original,
                          module_name + "." + attribute + " was not restored")

    def test_patched_attributes_are_restored_even_when_the_body_raises(self):
        recorder = jcd._Recorder(jcd._Roots([]))
        original_os = jtd.os
        with self.assertRaises(RuntimeError):
            with jcd._instrument(recorder):
                raise RuntimeError("boom")
        self.assertIs(jtd.os, original_os)

    def test_os_and_np_proxies_forward_unwrapped_attributes_untouched(self):
        recorder = jcd._Recorder(jcd._Roots([]))
        os_proxy = jcd._OsProxy(os, recorder)
        np_proxy = jcd._NpProxy(np, recorder)
        self.assertIs(os_proxy.sep, os.sep)
        self.assertIs(os_proxy.path.join, os.path.join)
        self.assertIs(np_proxy.concatenate, np.concatenate)
        self.assertIs(np_proxy.float32, np.float32)
        self.assertEqual(recorder.ops, [], "attribute forwarding must record nothing")


# =====================================================================
# Path/artifact classification.
# =====================================================================

class PathClassificationTests(unittest.TestCase):
    def test_each_real_cache_filename_maps_to_its_artifact(self):
        self.assertEqual(jcd._artifact_of_path(lfed._cache_path("/c", "abc", "vessel", (512, 512))),
                         "vessel")
        self.assertEqual(jcd._artifact_of_path(lfed._cache_path("/c", "abc", "lesion", (512, 512))),
                         "lesion")
        self.assertEqual(jcd._artifact_of_path(jtd._canonical_rgb_cache_path("abc", "/c", (512, 512))),
                         "rgb")
        self.assertEqual(jcd._artifact_of_path(racaf.reliability_cache_path("/c", "abc")),
                         "reliability")

    def test_an_unrelated_filename_maps_to_no_artifact(self):
        self.assertIsNone(jcd._artifact_of_path("/c/train.csv"))
        self.assertIsNone(jcd._artifact_of_path("/c/abc.png"))

    def test_longest_prefix_wins_so_a_nested_root_is_not_shadowed(self):
        roots = jcd._Roots([("outer", "/data", False), ("inner", "/data/drive", True)])
        self.assertEqual(roots.classify("/data/drive/x.npy"), ("inner", True))
        self.assertEqual(roots.classify("/data/x.npy"), ("outer", False))

    def test_a_path_under_no_registered_root_is_other_not_silently_local(self):
        roots = jcd._Roots([("local", "/data", False)])
        self.assertEqual(roots.classify("/somewhere/else/x.npy"), ("other", False))

    def test_a_none_root_is_skipped_rather_than_matching_everything(self):
        roots = jcd._Roots([("drive", None, True), ("local", "/data", False)])
        self.assertEqual(roots.classify("/data/x.npy"), ("local", False))
        self.assertEqual(roots.classify("/other/x.npy"), ("other", False))


class EntrySelectionTests(unittest.TestCase):
    ENTRIES = [("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4), ("f", 0)]

    def test_default_takes_the_first_max_images_entries_in_split_order(self):
        self.assertEqual(jcd.select_diagnostic_entries(self.ENTRIES, max_images=3),
                         [("a", 0), ("b", 1), ("c", 2)])

    def test_default_max_images_is_five(self):
        self.assertEqual(jcd.DEFAULT_MAX_IMAGES, 5)
        self.assertEqual(len(jcd.select_diagnostic_entries(self.ENTRIES)), 5)

    def test_explicit_ids_select_exactly_those_in_order_and_are_not_padded(self):
        selected = jcd.select_diagnostic_entries(self.ENTRIES, max_images=5,
                                                 explicit_ids=["e", "a"])
        self.assertEqual(selected, [("e", 4), ("a", 0)])

    def test_an_explicit_id_absent_from_the_split_is_still_selectable(self):
        """A known empty-FOV id from a real Phase 1 log must be measurable even if it is not in
        the entries list handed in."""
        selected = jcd.select_diagnostic_entries(self.ENTRIES, explicit_ids=["not_in_split"])
        self.assertEqual(selected, [("not_in_split", 0)])


# =====================================================================
# Per-image state classification, against the real pipeline.
# =====================================================================

class FreshMissTests(_DiagnosticTestBase):
    def test_a_completely_uncached_image_reports_every_artifact_as_computed(self):
        report = self.diagnose()
        record = report["pass1"][0]
        self.assertIsNone(record["error"])
        for artifact in jcd.ARTIFACTS:
            self.assertEqual(record["artifacts"][artifact]["action"], jcd.COMPUTED_FROM_MISS,
                             artifact + " should have been computed from a total miss")
            self.assertTrue(record["artifacts"][artifact]["compute_calls"])

    def test_a_fresh_miss_loads_the_raw_image_and_runs_stage02(self):
        record = self.diagnose()["pass1"][0]
        self.assertTrue(record["raw_image"]["exists"])
        self.assertTrue(record["raw_image"]["loaded"])
        self.assertEqual(record["raw_image"]["load_count"], 1)
        self.assertTrue(record["raw_image"]["stage02_ran"])

    def test_the_real_phase1_branch_counters_agree_with_the_classification(self):
        """Cross-check: the diagnostic's path-based verdict and `precompute_joint_frozen_caches`'s
        OWN `stats` dict are independent signals and must not disagree."""
        record = self.diagnose()["pass1"][0]
        self.assertEqual(record["phase1_stats"]["cached"], 1)
        self.assertEqual(record["phase1_stats"]["already_cached"], 0)
        self.assertEqual(record["phase1_stats"]["mirrored_from_persistent"], 0)

    def test_a_fresh_miss_writes_all_four_artifacts_locally_and_none_to_drive(self):
        record = self.diagnose()["pass1"][0]
        self.assertEqual(record["totals"]["drive_writes"], 0)
        for artifact in jcd.ARTIFACTS:
            self.assertEqual(record["artifacts"][artifact]["local_writes"], 1, artifact)


class PersistentHitTests(_DiagnosticTestBase):
    def test_a_fully_persistent_entry_is_mirrored_not_recomputed(self):
        """Populates the "Drive" cache first (as a prior run would have), then measures a fresh
        local cache against it."""
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        record = self.diagnose()["pass1"][0]
        for artifact in jcd.ARTIFACTS:
            measurement = record["artifacts"][artifact]
            self.assertEqual(measurement["action"], jcd.PERSISTENT_HIT_MIRRORED, artifact)
            self.assertTrue(measurement["persistent_before"], artifact)
            self.assertFalse(measurement["local_before"], artifact)
            self.assertGreaterEqual(measurement["drive_reads"], 1, artifact)
            self.assertGreaterEqual(measurement["local_writes"], 1, artifact)
            self.assertFalse(measurement["computed"], artifact)

    def test_a_fully_persistent_entry_never_touches_the_raw_image(self):
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        record = self.diagnose()["pass1"][0]
        self.assertFalse(record["raw_image"]["loaded"])
        self.assertEqual(record["raw_image"]["load_count"], 0)
        self.assertEqual(record["totals"]["compute_seconds"], 0.0)

    def test_a_legacy_persistent_entry_without_rgb_recomputes_only_rgb(self):
        """The exact shape of a Drive cache populated BEFORE the canonical-RGB cache kind existed:
        vessel/lesion/reliability are persistent hits, RGB alone must be regenerated. This is the
        state a real fresh runtime would find if RGB had never been flushed to Drive."""
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        legacy_rgb = jtd._canonical_rgb_cache_path("jcd_a", self.tree.drive_cache_dir, CANONICAL_SIZE)
        os.remove(legacy_rgb)

        record = self.diagnose()["pass1"][0]
        for artifact in ("vessel", "lesion", "reliability"):
            self.assertEqual(record["artifacts"][artifact]["action"], jcd.PERSISTENT_HIT_MIRRORED,
                             artifact)
            self.assertFalse(record["artifacts"][artifact]["computed"], artifact)
        rgb = record["artifacts"]["rgb"]
        self.assertEqual(rgb["action"], jcd.COMPUTED_FROM_MISS)
        self.assertFalse(rgb["persistent_before"])
        self.assertIn("_resize_rgb_01", rgb["compute_calls"])
        self.assertTrue(record["raw_image"]["loaded"],
                        "regenerating RGB requires the raw image, so it must show as loaded")

    def test_no_persistent_dirs_configured_reports_persistent_state_as_not_applicable(self):
        report = self.diagnose(persistent_cache_dir=None, persistent_racaf_cache_dir=None)
        record = report["pass1"][0]
        for artifact in jcd.ARTIFACTS:
            self.assertIsNone(record["artifacts"][artifact]["persistent_before"], artifact)
            self.assertIsNone(record["artifacts"][artifact]["persistent_path"], artifact)
            self.assertEqual(record["artifacts"][artifact]["drive_stats"], 0, artifact)


class EmptyFieldOfViewTests(_DiagnosticTestBase):
    """The real Phase 1 catches `EmptyFieldOfViewError` itself, logs it and moves on -- so such an
    image must be MEASURED, not treated as a crash, and must not be reported as a successful
    computation either. Forced via the same `predict_vessel_mask` side-effect pattern
    `tests/test_joint_training.py` already uses for this error."""

    def _diagnose_with_empty_fov(self):
        from unittest import mock
        with mock.patch("joint_training_dataset.predict_vessel_mask",
                        side_effect=jtd.EmptyFieldOfViewError("no fundus disk detected")):
            return self.diagnose()

    def test_an_empty_fov_image_is_reported_as_error_for_the_frozen_artifacts(self):
        record = self._diagnose_with_empty_fov()["pass1"][0]
        self.assertTrue(record["skipped_empty_fov"])
        self.assertIsNone(record["error"], "Phase 1 handles this itself -- it must not surface here")
        self.assertEqual(record["phase1_stats"]["skipped_empty_fov"], ["jcd_a"])
        for artifact in ("vessel", "lesion", "reliability"):
            self.assertEqual(record["artifacts"][artifact]["action"], jcd.ERROR, artifact)

    def test_an_empty_fov_image_still_gets_its_canonical_rgb_cached(self):
        """RGB does not depend on Stage 03's FOV detection, and the real code path caches it
        regardless -- the diagnostic must report that rather than lumping it in with the failure."""
        record = self._diagnose_with_empty_fov()["pass1"][0]
        self.assertEqual(record["artifacts"]["rgb"]["action"], jcd.COMPUTED_FROM_MISS)
        self.assertTrue(record["artifacts"]["rgb"]["local_after"])
        self.assertEqual(record["artifacts"]["rgb"]["local_writes"], 1)

    def test_an_empty_fov_image_does_not_abort_the_diagnostic(self):
        report = self._diagnose_with_empty_fov()
        self.assertEqual(len(report["pass1"]), 1)
        self.assertEqual(report["drive_write_violations"], [])


class SecondPassTests(_DiagnosticTestBase):
    def test_pass2_is_a_pure_local_hit_with_zero_drive_reads_after_a_fresh_miss_pass1(self):
        report = self.diagnose()
        self.assertEqual(len(report["pass2"]), 1)
        record = report["pass2"][0]
        for artifact in jcd.ARTIFACTS:
            self.assertEqual(record["artifacts"][artifact]["action"], jcd.LOCAL_HIT, artifact)
        totals = report["summary"]["pass_totals"]["pass2"]
        self.assertEqual(totals["drive_reads"], 0)
        self.assertEqual(totals["drive_writes"], 0)
        self.assertEqual(totals["compute_seconds"], 0.0)
        self.assertEqual(totals["raw_loads"], 0)

    def test_pass2_is_a_pure_local_hit_after_a_persistent_mirroring_pass1(self):
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        report = self.diagnose()
        for artifact in jcd.ARTIFACTS:
            self.assertEqual(report["pass1"][0]["artifacts"][artifact]["action"],
                             jcd.PERSISTENT_HIT_MIRRORED, artifact)
            self.assertEqual(report["pass2"][0]["artifacts"][artifact]["action"],
                             jcd.LOCAL_HIT, artifact)
        self.assertEqual(report["summary"]["pass_totals"]["pass2"]["drive_reads"], 0)

    def test_the_second_pass_can_be_switched_off(self):
        report = self.diagnose(run_second_pass=False)
        self.assertEqual(report["pass2"], [])


class DriveWriteTripwireTests(_DiagnosticTestBase):
    def test_no_pass_ever_writes_to_a_persistent_root(self):
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        report = self.diagnose()
        self.assertEqual(report["drive_write_violations"], [])

    def test_the_persistent_cache_files_are_untouched_by_a_diagnostic_run(self):
        """Content and mtime stability -- the diagnostic must be read-only against Drive."""
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)

        def snapshot():
            state = {}
            for directory in (self.tree.drive_cache_dir, self.tree.drive_racaf_cache_dir):
                for name in sorted(os.listdir(directory)):
                    path = os.path.join(directory, name)
                    with open(path, "rb") as handle:
                        state[path] = (handle.read(), os.stat(path).st_mtime_ns)
            return state

        before = snapshot()
        self.assertTrue(before, "expected the persistent stand-in to hold files")
        self.diagnose()
        self.assertEqual(before, snapshot())

    def test_probe_counts_are_reported_separately_from_pipeline_counts(self):
        report = self.diagnose()
        self.assertGreater(report["probe_stats"]["total"], 0)
        self.assertEqual(report["probe_stats"]["total"],
                         report["probe_stats"]["local"] + report["probe_stats"]["persistent"])


# =====================================================================
# Training-time path and numerical integrity.
# =====================================================================

class BuildJointSampleMeasurementTests(_DiagnosticTestBase):
    def test_the_training_time_path_touches_no_drive_path_once_the_cache_is_local(self):
        report = self.diagnose()
        self.assertEqual(len(report["build_sample"]), 1)
        record = report["build_sample"][0]
        self.assertIsNone(record["error"])
        self.assertEqual(record["drive_reads"], 0)
        self.assertEqual(record["drive_stats"], 0)
        self.assertEqual(record["drive_paths_touched"], [])
        self.assertFalse(record["raw_loaded"])
        self.assertEqual(record["compute_calls"], [])

    def test_the_training_time_measurement_can_be_switched_off(self):
        self.assertEqual(self.diagnose(measure_build_sample=False)["build_sample"], [])


class RGBNumericalIntegrityTests(_DiagnosticTestBase):
    def test_a_cached_rgb_file_is_bitwise_identical_to_a_fresh_generation(self):
        check = self.diagnose()["rgb_numerical_check"]
        self.assertTrue(check["compared"], check.get("error"))
        self.assertTrue(check["shapes_match"])
        self.assertTrue(check["dtypes_match"])
        self.assertEqual(check["max_abs_diff"], 0.0)
        self.assertTrue(check["exactly_equal"])
        self.assertTrue(check["bitwise_identical"])

    def test_a_persistently_mirrored_rgb_file_is_also_bitwise_identical_to_a_fresh_generation(self):
        """Covers the round trip that matters in the real runtime: Drive -> local mirror -> read."""
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        report = self.diagnose()
        self.assertEqual(report["pass1"][0]["artifacts"]["rgb"]["action"],
                         jcd.PERSISTENT_HIT_MIRRORED)
        check = report["rgb_numerical_check"]
        self.assertTrue(check["compared"], check.get("error"))
        self.assertTrue(check["bitwise_identical"])

    def test_the_comparison_reports_rather_than_raises_when_no_cache_file_exists(self):
        check = jcd.compare_cached_rgb_to_fresh(
            "jcd_a", self.tree.image_dir, os.path.join(self.tree.cache_dir, "absent.npy"),
            self.tree.processed_dir, CANONICAL_SIZE)
        self.assertFalse(check["compared"])
        self.assertIn("does not exist", check["error"])

    def test_the_comparison_never_writes_anything(self):
        self.diagnose()
        before = sorted(os.listdir(self.tree.cache_dir))
        paths = jcd.artifact_paths("jcd_a", self.tree.cache_dir, self.tree.racaf_cache_dir,
                                   CANONICAL_SIZE)
        jcd.compare_cached_rgb_to_fresh("jcd_a", self.tree.image_dir, paths["rgb"],
                                        self.tree.processed_dir, CANONICAL_SIZE)
        self.assertEqual(before, sorted(os.listdir(self.tree.cache_dir)))


# =====================================================================
# Cache-root inspection.
# =====================================================================

class CacheRootInspectionTests(_DiagnosticTestBase):
    def test_counts_are_reported_per_artifact_kind(self):
        self.run_phase1(self.tree.pairs)
        info = jcd.inspect_cache_root(self.tree.cache_dir)
        self.assertTrue(info["exists"])
        self.assertEqual(info["counts"]["vessel"], 2)
        self.assertEqual(info["counts"]["lesion"], 2)
        self.assertEqual(info["counts"]["rgb"], 2)
        self.assertEqual(info["counts"]["reliability"], 0)  # lives in the racaf dir
        self.assertEqual(info["total_files"], 6)
        self.assertFalse(info["truncated"])

    def test_a_root_missing_rgb_entries_reports_zero_rather_than_failing(self):
        """The decisive check for the real question: are RGB files actually on Drive?"""
        self.run_phase1(self.tree.pairs, cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        for name in os.listdir(self.tree.drive_cache_dir):
            if "_rgb_" in name:
                os.remove(os.path.join(self.tree.drive_cache_dir, name))
        info = jcd.inspect_cache_root(self.tree.drive_cache_dir)
        self.assertEqual(info["counts"]["rgb"], 0)
        self.assertEqual(info["counts"]["vessel"], 2)

    def test_a_nonexistent_root_is_reported_as_absent_not_raised(self):
        info = jcd.inspect_cache_root(os.path.join(self.tree.root, "definitely_absent"))
        self.assertFalse(info["exists"])
        self.assertIsNone(info["error"])

    def test_a_none_root_is_handled(self):
        info = jcd.inspect_cache_root(None)
        self.assertFalse(info["exists"])
        self.assertIsNone(info["abspath"])

    def test_listing_stops_at_max_entries_and_says_so(self):
        self.run_phase1(self.tree.pairs)
        info = jcd.inspect_cache_root(self.tree.cache_dir, max_entries=2)
        self.assertTrue(info["truncated"])
        self.assertEqual(info["total_files"], 2)

    def test_file_counting_can_be_skipped_entirely(self):
        self.run_phase1(self.tree.pairs)
        info = jcd.inspect_cache_root(self.tree.cache_dir, count_files=False)
        self.assertTrue(info["exists"])
        self.assertEqual(info["counts"], {})

    def test_size_is_reported_as_a_sampled_estimate_with_its_basis(self):
        self.run_phase1(self.tree.pairs)
        info = jcd.inspect_cache_root(self.tree.cache_dir)
        self.assertIsNotNone(info["estimated_bytes"])
        self.assertGreater(info["estimated_bytes"], 0)
        self.assertIn("vessel", info["estimate_basis"])
        self.assertIn("sampled_files", info["estimate_basis"]["vessel"])


# =====================================================================
# Report assembly and rendering.
# =====================================================================

class ReportTests(_DiagnosticTestBase):
    def test_expected_paths_are_reported_for_the_first_two_ids_with_all_four_artifacts(self):
        report = self.diagnose(max_images=2)
        self.assertEqual(list(report["expected_paths"]), ["jcd_a", "jcd_b"])
        for states in report["expected_paths"].values():
            self.assertEqual(sorted(states), sorted(jcd.ARTIFACTS))
            for state in states.values():
                self.assertTrue(os.path.isabs(state["local_path"]))
                self.assertTrue(os.path.isabs(state["persistent_path"]))

    def test_the_reported_rgb_paths_are_exactly_what_the_real_path_builder_produces(self):
        report = self.diagnose()
        state = report["expected_paths"]["jcd_a"]["rgb"]
        self.assertEqual(state["local_path"],
                         jtd._canonical_rgb_cache_path("jcd_a", self.tree.cache_dir, CANONICAL_SIZE))
        self.assertEqual(state["persistent_path"],
                         jtd._canonical_rgb_cache_path("jcd_a", self.tree.drive_cache_dir,
                                                       CANONICAL_SIZE))

    def test_summary_counts_match_the_per_image_records(self):
        report = self.diagnose(max_images=2)
        for artifact in jcd.ARTIFACTS:
            stats = report["summary"]["artifacts"][artifact]
            self.assertEqual(stats["computed"], 2)
            self.assertEqual(stats["local_hits"], 0)
            self.assertEqual(stats["persistent_hits"], 0)
        self.assertEqual(report["summary"]["pass_totals"]["pass1"]["images"], 2)

    def test_print_report_renders_without_raising_and_names_the_states(self):
        import io
        import contextlib
        report = self.diagnose()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            jcd.print_report(report)
        rendered = buffer.getvalue()
        self.assertIn("JOINT CACHE DIAGNOSTIC", rendered)
        self.assertIn("CLASSIFICATION", rendered)
        self.assertIn("PROVEN", rendered)
        self.assertIn("UNKNOWN", rendered)
        self.assertIn(jcd.COMPUTED_FROM_MISS, rendered)
        self.assertIn("INTEGRITY TRIPWIRE", rendered)
        self.assertIn("PASS -- zero writes", rendered)

    def test_the_experiment_dir_is_reported_read_only_and_never_created(self):
        """It is listed among the paths the report resolves, but the diagnostic must not create it
        -- creating an experiment directory would touch experiment semantics."""
        experiment_dir = os.path.join(self.tree.root, "experiments", "FinalClassification")
        report = self.diagnose(experiment_dir=experiment_dir)
        info = report["roots"]["experiment_dir"]
        self.assertEqual(info["abspath"], os.path.abspath(experiment_dir))
        self.assertFalse(info["exists"])
        self.assertFalse(os.path.exists(experiment_dir),
                         "the diagnostic must not create the experiment directory")

    def test_an_omitted_experiment_dir_renders_as_not_configured(self):
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            jcd.print_report(self.diagnose())
        self.assertIn("experiment_dir:", buffer.getvalue())
        self.assertIn("(not configured)", buffer.getvalue())

    def test_the_report_states_that_canonical_rgb_shares_the_frozen_cache_directory(self):
        """The exact confusion this diagnostic exists to rule out -- whether the notebook checks
        the same location the RGB artifacts were persisted to -- must be answered on the face of
        the report, not left to be inferred."""
        import io
        import contextlib
        self.run_phase1(self.tree.pairs[:1])
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            jcd.print_report(self.diagnose())
        rendered = buffer.getvalue()
        self.assertIn("canonical RGB cache:", rendered)
        self.assertIn("shares local_cache_dir", rendered)

    def test_print_report_distinguishes_an_unconfigured_persistent_dir_from_a_missing_one(self):
        """"Not configured" and "configured but absent" are different findings and must not render
        as the same line -- confusing them would send the next audit after the wrong thing."""
        import io
        import contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            jcd.print_report(self.diagnose(persistent_cache_dir=None,
                                           persistent_racaf_cache_dir=None))
        self.assertIn("not configured", buffer.getvalue())
        self.assertNotIn("does NOT exist", buffer.getvalue())

        absent = os.path.join(self.tree.root, "drive_that_is_not_mounted")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            jcd.print_report(self.diagnose(persistent_cache_dir=absent,
                                           persistent_racaf_cache_dir=absent))
        self.assertIn("does NOT exist", buffer.getvalue())

    def test_print_report_renders_a_persistent_hit_run_too(self):
        import io
        import contextlib
        self.run_phase1(self.tree.pairs[:1], cache_dir=self.tree.drive_cache_dir,
                        racaf_cache_dir=self.tree.drive_racaf_cache_dir)
        report = self.diagnose()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            jcd.print_report(report)
        rendered = buffer.getvalue()
        self.assertIn(jcd.PERSISTENT_HIT_MIRRORED, rendered)
        self.assertIn(jcd.LOCAL_HIT, rendered)


class NoPipelineDependencyTests(unittest.TestCase):
    def test_the_pipeline_modules_do_not_import_the_diagnostic(self):
        """The diagnostic must be strictly one-directional: it imports the pipeline, never the
        reverse, so nothing a real run executes can be affected by this module existing."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for module_name in ("joint_training_dataset.py", "joint_training_model.py", "racaf.py",
                            "local_feature_extraction_dataset.py", "training.py"):
            path = os.path.join(repo_root, module_name)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as handle:
                source = handle.read()
            self.assertNotIn("joint_cache_diagnostics", source,
                             module_name + " must not import the diagnostic module")


if __name__ == "__main__":
    unittest.main()
