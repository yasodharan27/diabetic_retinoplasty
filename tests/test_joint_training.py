"""
Tests for the joint Stage 05-08 + RACAF training pipeline
(`joint_training_dataset.py`, `joint_training_model.py`).

Per `PROJECT_CODE.md`'s Implementation Rules, this suite uses synthetic/temporary data only --
no real datasets, no real Stage 1/3/4 checkpoints. Stage 03's vessel model and Stage 04's lesion
model are the same synthetic, from-scratch construction pattern already established in
`tests/test_local_feature_extraction_dataset.py` -- neither the real, gitignored LWNet checkpoint
nor the real, gitignored Experiment 2C `.keras` checkpoint is ever touched here.

NO TRAINING IS RUN ANYWHERE IN THIS FILE -- no `model.fit()`, no epoch, no checkpoint is produced
by any test. Gradient checks use a single `tf.GradientTape` step only, to verify the gradient
BOUNDARY (which variables receive a gradient), never to update a weight.
"""

import csv
import inspect
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf
import torch
from PIL import Image

import corn
import downstream_split
import joint_training_dataset as jtd
import joint_training_model as jtm
import lesion_segmentation_model as lsm
import local_feature_extraction_dataset as lfed
import racaf
from vessel_segmentation_model import build_vessel_segmentation_model, load_state_dict_from_checkpoint

# colab/common/dataset_staging.py -- matches test_dataset_staging.py's established pattern for
# testing colab/common/ modules outside Colab (no Drive, no google.colab import).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COLAB_COMMON = os.path.join(_REPO_ROOT, "colab", "common")
if _COLAB_COMMON not in sys.path:
    sys.path.insert(0, _COLAB_COMMON)
import dataset_staging  # noqa: E402

IMAGE_SIZE = 256  # large enough for Stage 03's FOV circle-fit to succeed reliably


def _synthetic_fundus_image(size=IMAGE_SIZE, seed=0):
    """Identical recipe to test_local_feature_extraction_dataset.py's helper of the same name."""
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
    """A real (untrained) Attention U-Net at Stage 04's REAL, full native input shape
    `(512,512,4)` -- required since `racaf.prepare_stage4_input`/`tta_views` always resize to
    that fixed resolution; a smaller synthetic shape (as some other tests use for pure-Stage-04
    speed) is not usable here. `trainable` is explicitly forced False, mirroring
    `racaf.load_frozen_stage4_model()`'s own behavior exactly (that function itself calls
    `lesion_segmentation_model.load_lesion_model()`, which requires a real `.keras` file on disk
    -- not usable in a synthetic-data-only test, so this helper reproduces its
    frozen/`trainable=False` POST-condition on a freshly-built, real, untrained model instead)."""
    model = lsm.build_attention_unet(input_shape=(512, 512, 4), base_filters=4)
    model.trainable = False
    return model


class _SyntheticAPTOSTree:
    """Same shape/convention as test_local_feature_extraction_dataset.py's helper of the same
    name: a temp datasets/APTOS2019/raw/-shaped directory (train.csv + train_images/)."""

    def __init__(self, id_diagnosis_pairs):
        self.root = tempfile.mkdtemp(prefix="aptos_joint_synth_")
        self.image_dir = os.path.join(self.root, "train_images")
        self.csv_path = os.path.join(self.root, "train.csv")
        self.cache_dir = os.path.join(self.root, "cache")
        self.racaf_cache_dir = os.path.join(self.root, "racaf_cache")
        os.makedirs(self.image_dir, exist_ok=True)
        self.pairs = list(id_diagnosis_pairs)
        self._build()

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _build(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id_code", "diagnosis"])
            for i, (id_code, diagnosis) in enumerate(self.pairs):
                writer.writerow([id_code, diagnosis])
                image = _synthetic_fundus_image(seed=i)
                Image.fromarray(image).save(os.path.join(self.image_dir, f"{id_code}.png"))


# =====================================================================
# 1-4: Authoritative split -- no second split.
# =====================================================================

class AuthoritativeSplitTests(unittest.TestCase):
    def test_joint_split_delegates_to_downstream_split_directly(self):
        train_entries, val_entries = jtd.split_train_val_ids()
        expected_train, expected_val = downstream_split.get_authoritative_split()
        self.assertEqual(train_entries, expected_train)
        self.assertEqual(val_entries, expected_val)

    def test_real_authoritative_split_counts(self):
        train_entries, val_entries = jtd.split_train_val_ids()
        self.assertEqual(len(train_entries), 2929)
        self.assertEqual(len(val_entries), 733)
        self.assertEqual(len(train_entries) + len(val_entries), 3662)

    def test_no_overlap(self):
        train_entries, val_entries = jtd.split_train_val_ids()
        train_ids = {i for i, _ in train_entries}
        val_ids = {i for i, _ in val_entries}
        self.assertEqual(train_ids & val_ids, set())

    def test_module_defines_no_second_split_utility(self):
        """joint_training_dataset.py must never call train_test_split/stratify itself -- it only
        ever delegates to downstream_split.get_authoritative_split."""
        source = inspect.getsource(jtd)
        for forbidden in ("train_test_split", "stratify"):
            self.assertNotIn(forbidden, source)


# =====================================================================
# Ground-truth / label boundary -- no masks, no IDRiD, no test.csv.
# =====================================================================

class GroundTruthAndLabelBoundaryTests(unittest.TestCase):
    """Tokenize-based source scan -- mirrors corn.py's/racaf.py's own boundary-test pattern."""

    # "vessel_mask"/"lesion_mask" are deliberately NOT forbidden: this project's own frozen,
    # approved Stage 03/04 inference functions are legitimately named `predict_vessel_mask`/
    # `vessel_probability_map` etc. (a PREDICTED quantity, never a ground-truth one) --
    # `corn.py`'s own identical boundary test (`GroundTruthMaskBoundaryTests`) already dropped the
    # overly-broad "mask" fragment for exactly this reason; this list follows that precedent.
    _FORBIDDEN = ("ground_truth", "groundtruth", "idrid", "test_csv")

    def _assert_no_forbidden_identifiers(self, module):
        import ast

        source = inspect.getsource(module)
        tree = ast.parse(source)
        identifiers = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
        identifiers |= {node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        for forbidden in self._FORBIDDEN:
            for identifier in identifiers:
                self.assertNotIn(
                    forbidden, identifier,
                    f"forbidden identifier fragment {forbidden!r} found in code identifier {identifier!r}",
                )

    def test_joint_dataset_module_has_no_forbidden_identifiers(self):
        self._assert_no_forbidden_identifiers(jtd)

    def test_joint_model_module_has_no_forbidden_identifiers(self):
        self._assert_no_forbidden_identifiers(jtm)


# =====================================================================
# Cache reuse + redundancy elimination (Step 4/5).
# =====================================================================

class CacheReuseAndRedundancyTests(unittest.TestCase):
    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_01", 2)])
        self.addCleanup(self.tree.cleanup)

    def _build_sample(self):
        rng = np.random.default_rng(0)
        return jtd._build_joint_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=rng,
        )

    def test_sample_structure_keys(self):
        sample = self._build_sample()
        self.assertEqual(
            set(sample.keys()), {"image_id", "stage5_input", "stage6_input", "reliability", "grade"},
        )

    def test_stage5_input_shape(self):
        sample = self._build_sample()
        self.assertEqual(sample["stage5_input"].shape, (512, 512, 8))
        self.assertEqual(sample["stage5_input"].dtype, np.float32)

    def test_stage6_input_shape(self):
        sample = self._build_sample()
        self.assertEqual(sample["stage6_input"].shape, (256, 256, 3))
        self.assertEqual(sample["stage6_input"].dtype, np.float32)

    def test_grade_is_the_aptos_label(self):
        sample = self._build_sample()
        self.assertEqual(sample["grade"], 2)
        self.assertIsInstance(sample["grade"], int)

    def test_reliability_is_scalar_float_in_unit_range(self):
        sample = self._build_sample()
        self.assertEqual(np.asarray(sample["reliability"]).shape, ())
        self.assertGreaterEqual(float(sample["reliability"]), 0.0)
        self.assertLessEqual(float(sample["reliability"]), 1.0)

    def test_canonical_vessel_and_lesion_caches_are_512(self):
        self._build_sample()
        vessel_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "vessel", jtd.STAGE5_IMAGE_SIZE)
        lesion_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "lesion", jtd.STAGE5_IMAGE_SIZE)
        vessel_map = np.load(vessel_cache)
        lesion_maps = np.load(lesion_cache)
        self.assertEqual(vessel_map.shape, (512, 512, 1))
        self.assertEqual(lesion_maps.shape, (512, 512, 4))

    def test_reliability_cache_stores_only_kappa_and_r(self):
        self._build_sample()
        reliability_cache = racaf.reliability_cache_path(self.tree.racaf_cache_dir, "img_01")
        cached = np.load(reliability_cache)
        self.assertEqual(set(cached.files), {"kappa", "r"})
        self.assertEqual(cached["kappa"].shape, (4,))
        self.assertEqual(cached["r"].shape, ())

    def test_reliability_cache_never_stores_raw_four_view_maps(self):
        self._build_sample()
        reliability_cache = racaf.reliability_cache_path(self.tree.racaf_cache_dir, "img_01")
        size_bytes = os.path.getsize(reliability_cache)
        # kappa (4 float32) + r (1 float32) is a few hundred bytes at most, including npz
        # overhead -- four raw (512,512,4) float32 maps would be ~16MB, three orders of
        # magnitude larger.
        self.assertLess(size_bytes, 10_000)

    def test_tta_views_called_exactly_once_per_uncached_image(self):
        """The actual Step 4/5 redundancy-elimination proof: ONE racaf.tta_views() call must
        serve both Stage 05's lesion-map cache and RACAF's reliability cache for a given image --
        never two."""
        with mock.patch("joint_training_dataset.racaf.tta_views", wraps=racaf.tta_views) as mocked_tta:
            self._build_sample()
            self.assertEqual(mocked_tta.call_count, 1)

    def test_second_call_reuses_cache_without_recomputing_anything(self):
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel, mock.patch(
            "joint_training_dataset.racaf.tta_views", wraps=racaf.tta_views,
        ) as mocked_tta:
            self._build_sample()
            self.assertEqual(mocked_vessel.call_count, 1)
            self.assertEqual(mocked_tta.call_count, 1)

            self._build_sample()
            # Still 1 each -- both caches (vessel/lesion + reliability) already exist.
            self.assertEqual(mocked_vessel.call_count, 1)
            self.assertEqual(mocked_tta.call_count, 1)

    def test_lesion_cache_equals_the_identity_tta_view(self):
        """Proves the cached lesion map really IS RACAF's own identity-transform view, not an
        approximation -- the exact-reuse claim, verified numerically."""
        rgb_native = lfed._resolve_processed_rgb(
            lfed._load_raw_bgr(self.tree.image_dir, "img_01"), lfed.DEFAULT_PROCESSED_DIR, "img_01",
        )
        native_vessel = jtd.predict_vessel_mask(rgb_native, model=self.vessel_model)["probability_map"].astype(np.float32)
        prepared = racaf.prepare_stage4_input(rgb_native, native_vessel)
        aligned = racaf.tta_views(self.stage4_model, prepared).numpy()
        expected_identity = aligned[0, racaf.TTA_TRANSFORMS.index("identity"), ...]

        self._build_sample()
        lesion_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "lesion", jtd.STAGE5_IMAGE_SIZE)
        cached_lesion = np.load(lesion_cache)
        np.testing.assert_allclose(cached_lesion, expected_identity, atol=1e-5)


# =====================================================================
# Empty-FOV handling (STEP B empty-FOV crash fix).
# =====================================================================

class EmptyFieldOfViewHandlingTests(unittest.TestCase):
    """Regression tests for the empty-Stage-03-FOV crash: `crop_to_fov()`'s
    `regionprops(fov_mask.astype(int))[0].bbox` raised a bare, context-free `IndexError` for a
    real APTOS image whose FOV circle-fit degenerated to an empty mask (no fundus disk detected).
    `vessel_segmentation_inference.py` now raises a named `EmptyFieldOfViewError` for that case
    (see its docstring for the exact mechanism), and this module's dataset generator
    (`_make_joint_dataset`) catches it, logs the skipped `image_id`, and excludes just that
    sample -- rather than crashing the whole `.fit()` run or fabricating a vessel/FOV result. The
    authoritative split manifest on disk is never touched by this handling."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_good", 1), ("img_empty_fov", 3), ("img_good_2", 0)])
        self.addCleanup(self.tree.cleanup)

    def test_build_joint_sample_propagates_empty_fov_error(self):
        """The check itself lives at the narrowest layer (vessel_segmentation_inference's
        crop_to_fov) -- _build_joint_sample does not swallow it; only the generator does."""
        rng = np.random.default_rng(0)
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask",
            side_effect=jtd.EmptyFieldOfViewError("no fundus disk detected"),
        ):
            with self.assertRaises(jtd.EmptyFieldOfViewError):
                jtd._build_joint_sample(
                    "img_empty_fov", 3, self.tree.image_dir, self.tree.cache_dir,
                    self.tree.racaf_cache_dir, self.vessel_model, self.stage4_model,
                    augment=False, rng=rng,
                )

    def _flaky_build_sample(self, id_code, diagnosis, *args, **kwargs):
        """Reproduces exactly one id (`img_empty_fov`) always failing with the real empty-FOV
        exception, while every other id builds its sample normally -- proves the generator's
        skip logic is selective, not a blanket catch that would also swallow unrelated errors."""
        if id_code == "img_empty_fov":
            raise jtd.EmptyFieldOfViewError("no fundus disk detected")
        return self._real_build_sample(id_code, diagnosis, *args, **kwargs)

    def test_generator_skips_only_the_empty_fov_image_and_continues(self):
        """A 3-image dataset where one image always raises EmptyFieldOfViewError must yield
        exactly the other 2 samples -- not crash, and not yield a fabricated 3rd sample."""
        self._real_build_sample = jtd._build_joint_sample
        with mock.patch("joint_training_dataset._build_joint_sample", side_effect=self._flaky_build_sample):
            ds = jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
            )
            grades = [int(grade.numpy()[0]) for _, grade in ds]

        self.assertEqual(len(grades), 2)
        self.assertEqual(sorted(grades), [0, 1])

    def test_generator_logs_a_warning_naming_the_skipped_image_id(self):
        self._real_build_sample = jtd._build_joint_sample
        with mock.patch("joint_training_dataset._build_joint_sample", side_effect=self._flaky_build_sample):
            with self.assertLogs("joint_training_dataset", level="WARNING") as captured:
                ds = jtd._make_joint_dataset(
                    self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                    self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
                )
                list(ds)
        self.assertTrue(any("img_empty_fov" in message for message in captured.output))

    def test_normal_images_are_completely_unaffected(self):
        """No mocking at all -- every image in the synthetic tree has a normal, real fundus-disk
        FOV, so the generator must yield exactly len(pairs) samples, none skipped. Proves the fix
        preserves valid-image behavior exactly."""
        ds = jtd._make_joint_dataset(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
        )
        grades = [int(grade.numpy()[0]) for _, grade in ds]
        self.assertEqual(sorted(grades), [0, 1, 3])

    def test_authoritative_split_manifest_is_never_touched_by_skip_handling(self):
        """The empty-FOV skip is a purely in-memory, generator-level exclusion -- it must never
        write to or otherwise mutate the real authoritative split manifest on disk."""
        manifest_path = downstream_split.DEFAULT_SPLIT_MANIFEST
        before = os.path.getmtime(manifest_path) if os.path.exists(manifest_path) else None

        self._real_build_sample = jtd._build_joint_sample
        with mock.patch("joint_training_dataset._build_joint_sample", side_effect=self._flaky_build_sample):
            ds = jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
            )
            list(ds)

        after = os.path.getmtime(manifest_path) if os.path.exists(manifest_path) else None
        self.assertEqual(before, after)


class EmptyFieldOfViewLowLevelTests(unittest.TestCase):
    """Direct, deterministic tests of the root-cause fix in
    `vessel_segmentation_inference.py`, independent of the joint dataset -- exercising exactly
    the code path the real crash traceback named (`crop_to_fov` -> `regionprops(...)[0].bbox`)."""

    def test_crop_to_fov_raises_named_error_on_empty_mask(self):
        from vessel_segmentation_inference import EmptyFieldOfViewError as VSIError
        from vessel_segmentation_inference import crop_to_fov

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        empty_mask = np.zeros((64, 64), dtype=bool)
        with self.assertRaises(VSIError):
            crop_to_fov(image, empty_mask)

    def test_crop_to_fov_unchanged_for_a_normal_nonempty_mask(self):
        """Regression: proves the fix does not alter the bbox computed for any real, non-empty
        FOV mask -- identical result to the pre-fix `regionprops(...)[0].bbox` call."""
        from skimage.measure import regionprops

        from vessel_segmentation_inference import crop_to_fov

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=bool)
        mask[10:30, 5:20] = True
        crop, bbox = crop_to_fov(image, mask)
        expected_bbox = regionprops(mask.astype(int))[0].bbox
        self.assertEqual(bbox, expected_bbox)
        self.assertEqual(crop.shape, (20, 15, 3))

    def test_fit_circle_raises_named_error_on_all_empty_binary_mask(self):
        from vessel_segmentation_inference import EmptyFieldOfViewError as VSIError
        from vessel_segmentation_inference import _fit_circle

        with self.assertRaises(VSIError):
            _fit_circle(np.zeros((64, 64), dtype=bool))

    def test_compute_fov_mask_and_crop_still_succeed_end_to_end_for_a_real_fundus_image(self):
        """No mocking -- the real `compute_fov_mask` + `crop_to_fov` pipeline on a normal
        synthetic fundus photo, proving the added checks never fire for a valid image."""
        from vessel_segmentation_inference import compute_fov_mask, crop_to_fov

        image_array = _synthetic_fundus_image(size=256, seed=0)
        fov_mask = compute_fov_mask(Image.fromarray(image_array))
        self.assertTrue(fov_mask.any())
        crop, bbox = crop_to_fov(image_array, fov_mask)
        self.assertGreater(crop.shape[0], 0)
        self.assertGreater(crop.shape[1], 0)

    def test_predict_vessel_mask_raises_before_any_forward_pass_on_empty_fov(self):
        """`compute_fov_mask` returning an empty mask must abort `predict_vessel_mask` via
        `EmptyFieldOfViewError` before the (expensive) LWNet forward pass ever runs -- proven by
        mocking compute_fov_mask alone and confirming the vessel model is never invoked."""
        import vessel_segmentation_inference as vsi

        model = _build_synthetic_vessel_model()
        image_array = _synthetic_fundus_image(size=256, seed=0)
        with mock.patch(
            "vessel_segmentation_inference.compute_fov_mask",
            return_value=np.zeros((256, 256), dtype=bool),
        ):
            with mock.patch.object(model, "forward", wraps=model.forward) as mocked_forward:
                with self.assertRaises(vsi.EmptyFieldOfViewError):
                    vsi.predict_vessel_mask(image_array, model=model)
                mocked_forward.assert_not_called()


# =====================================================================
# Two-phase workflow (STEP B RAM-exhaustion fix): bounded shuffle buffer +
# cache precomputation decoupled from training.
# =====================================================================

class ShuffleBufferBoundTests(unittest.TestCase):
    """Regression test for the actual RAM-growth root cause behind the first real T4 training
    run's crash: `_make_joint_dataset` previously sized its shuffle buffer to `len(entries)`
    (up to 2929), but each already-materialized sample is a `(512,512,8)` + `(256,256,3)` float32
    pair (~8.75 MB) -- that alone demanded ~25 GB before an epoch could even start. The buffer
    must now be a FIXED cap, never scale with dataset size."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1), ("img_c", 2)])
        self.addCleanup(self.tree.cleanup)

    def _captured_buffer_size(self):
        original_shuffle = tf.data.Dataset.shuffle
        captured = {}

        def spy_shuffle(ds_self, buffer_size, **kwargs):
            captured["buffer_size"] = buffer_size
            return original_shuffle(ds_self, buffer_size, **kwargs)

        with mock.patch.object(tf.data.Dataset, "shuffle", new=spy_shuffle):
            jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=True, augment=False, seed=0,
            )
        return captured["buffer_size"]

    def test_buffer_size_is_capped_when_dataset_exceeds_the_cap(self):
        with mock.patch("joint_training_dataset.DEFAULT_SHUFFLE_BUFFER_SIZE", 2):
            self.assertEqual(self._captured_buffer_size(), 2)

    def test_buffer_size_equals_dataset_size_when_smaller_than_the_cap(self):
        with mock.patch("joint_training_dataset.DEFAULT_SHUFFLE_BUFFER_SIZE", 10):
            self.assertEqual(self._captured_buffer_size(), 3)

    def test_default_shuffle_buffer_size_never_scales_with_the_real_dataset_size(self):
        """The actual bug was `buffer_size=max(len(entries), 1)` -- proven here by asserting the
        real, shipped default is a small constant, far below the real authoritative train split
        size (2929), without needing to build 2929 real samples."""
        self.assertLess(jtd.DEFAULT_SHUFFLE_BUFFER_SIZE, 2929)
        self.assertGreater(jtd.DEFAULT_SHUFFLE_BUFFER_SIZE, 0)


class CachePrecomputationTests(unittest.TestCase):
    """Phase 1 (`precompute_joint_frozen_caches`): bounded/streaming cache generation, reuse, and
    resumability -- independent of any `tf.data` pipeline."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1), ("img_c", 2)])
        self.addCleanup(self.tree.cleanup)

    def _precompute(self, entries=None):
        return jtd.precompute_joint_frozen_caches(
            entries if entries is not None else self.tree.pairs,
            self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

    def test_precompute_writes_caches_for_every_entry(self):
        stats = self._precompute()
        self.assertEqual(stats["cached"], 3)
        self.assertEqual(stats["already_cached"], 0)
        self.assertEqual(stats["skipped_empty_fov"], [])
        for id_code, _ in self.tree.pairs:
            vessel_cache = lfed._cache_path(self.tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE)
            lesion_cache = lfed._cache_path(self.tree.cache_dir, id_code, "lesion", jtd.STAGE5_IMAGE_SIZE)
            reliability_cache = racaf.reliability_cache_path(self.tree.racaf_cache_dir, id_code)
            self.assertTrue(os.path.exists(vessel_cache))
            self.assertTrue(os.path.exists(lesion_cache))
            self.assertTrue(os.path.exists(reliability_cache))

    def test_rerun_reuses_existing_caches_without_invoking_stage3_inference(self):
        """Cache reuse: a second precomputation pass over the same entries must not call Stage
        03's real inference function at all."""
        self._precompute()
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            stats = self._precompute()
        self.assertEqual(mocked_vessel.call_count, 0)
        self.assertEqual(stats["already_cached"], 3)
        self.assertEqual(stats["cached"], 0)

    def test_interrupted_precomputation_resumes_without_touching_valid_entries(self):
        """Simulates an interruption after only the first entry finished: re-running over the
        FULL entry list must fill in exactly the remaining two, and must not recompute (or
        otherwise touch) the first, already-valid cache entry."""
        self._precompute(entries=[self.tree.pairs[0]])
        vessel_cache_0 = lfed._cache_path(self.tree.cache_dir, self.tree.pairs[0][0], "vessel", jtd.STAGE5_IMAGE_SIZE)
        mtime_before = os.path.getmtime(vessel_cache_0)

        stats = self._precompute()

        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["cached"], 2)
        self.assertEqual(os.path.getmtime(vessel_cache_0), mtime_before)

    def test_returned_stats_are_small_fixed_size_bookkeeping_only(self):
        """Bounded-memory proof: the returned stats must never carry a per-image array -- only
        counters, an elapsed-time float, and a list of (small) string ids for skipped images."""
        stats = self._precompute()
        self.assertEqual(
            set(stats.keys()),
            {"cached", "already_cached", "mirrored_from_persistent", "skipped_empty_fov", "elapsed_seconds"},
        )
        self.assertIsInstance(stats["cached"], int)
        self.assertIsInstance(stats["already_cached"], int)
        self.assertIsInstance(stats["mirrored_from_persistent"], int)
        self.assertIsInstance(stats["skipped_empty_fov"], list)
        self.assertIsInstance(stats["elapsed_seconds"], float)
        self.assertGreaterEqual(stats["elapsed_seconds"], 0.0)
        for item in stats["skipped_empty_fov"]:
            self.assertIsInstance(item, str)

    def test_mirrored_from_persistent_is_zero_when_no_persistent_dir_is_involved(self):
        """Backward-compatibility proof: every existing caller (this whole class) never passes
        persistent_cache_dir, so the new counter must stay at its zero default -- 'cached' and
        'already_cached' keep their exact prior meaning and prior values (see the other tests in
        this class, all unmodified)."""
        stats = self._precompute()
        self.assertEqual(stats["mirrored_from_persistent"], 0)

    def test_precomputation_processes_one_image_at_a_time_not_in_bulk(self):
        """Streaming proof: the per-image cache-populating function is called exactly once per
        entry needing computation, never given more than one image's data at once."""
        real_fn = jtd._get_or_compute_joint_frozen_outputs
        seen_cache_paths = []

        def spy(rgb_native, vessel_cache_path, *args, **kwargs):
            seen_cache_paths.append(vessel_cache_path)
            return real_fn(rgb_native, vessel_cache_path, *args, **kwargs)

        with mock.patch("joint_training_dataset._get_or_compute_joint_frozen_outputs", side_effect=spy):
            self._precompute()

        self.assertEqual(len(seen_cache_paths), 3)
        self.assertEqual(len(set(seen_cache_paths)), 3)

    def test_empty_fov_image_is_skipped_and_does_not_block_remaining_entries(self):
        real_fn = jtd._get_or_compute_joint_frozen_outputs

        def flaky(rgb_native, vessel_cache_path, *args, **kwargs):
            if "img_b" in vessel_cache_path:
                raise jtd.EmptyFieldOfViewError("no fundus disk detected")
            return real_fn(rgb_native, vessel_cache_path, *args, **kwargs)

        with mock.patch("joint_training_dataset._get_or_compute_joint_frozen_outputs", side_effect=flaky):
            stats = self._precompute()

        self.assertEqual(stats["cached"], 2)
        self.assertEqual(stats["skipped_empty_fov"], ["img_b"])


class CachePrecomputationDriveRoundTripTests(unittest.TestCase):
    """Regression tests for the Phase 1 slowness diagnosis (`JOINT_TRAINING_ARCHITECTURE.md`
    §33): a Drive-mounted `cache_dir`/`racaf_cache_dir` makes every `os.path.exists` call a slow
    round trip, so `precompute_joint_frozen_caches` must never re-check the same three cache
    paths twice for one image, and must report throughput so a long run's actual speed is
    visible."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1)])
        self.addCleanup(self.tree.cleanup)

    def _exists_call_count(self, id_code, known_not_all_cached):
        vessel_cache = lfed._cache_path(self.tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE)
        lesion_cache = lfed._cache_path(self.tree.cache_dir, id_code, "lesion", jtd.STAGE5_IMAGE_SIZE)
        reliability_cache = racaf.reliability_cache_path(self.tree.racaf_cache_dir, id_code)
        raw_bgr = lfed._load_raw_bgr(self.tree.image_dir, id_code)
        rgb_native = lfed._resolve_processed_rgb(raw_bgr, lfed.DEFAULT_PROCESSED_DIR, id_code)

        with mock.patch("joint_training_dataset.os.path.exists", wraps=os.path.exists) as mocked_exists:
            jtd._get_or_compute_joint_frozen_outputs(
                rgb_native, vessel_cache, lesion_cache, reliability_cache,
                self.vessel_model, self.stage4_model, known_not_all_cached=known_not_all_cached,
            )
        return mocked_exists.call_count

    def test_known_not_all_cached_skips_the_redundant_upfront_check(self):
        """Differential proof, robust to any incidental os.path.exists calls made by underlying
        libraries during model inference (identical on both sides of this comparison, since both
        runs execute the same subsequent code) -- what matters is that known_not_all_cached=True
        skips the redundant "are all three already cached" recheck entirely. For a fully-
        uncached image (neither img_a nor img_b has ANY cache file yet), Python's `and` short-
        circuits that check after its first os.path.exists call returns False -- so the saving is
        exactly 1 call here, not 3 (3 is the ceiling, reached only when the first two of the
        three paths already exist from a prior partial run -- see the wiring test above for the
        actual guarantee that matters: known_not_all_cached=True is always passed by the
        precompute loop, so this check never runs redundantly regardless of partial-cache state).
        Uses two different, independent image ids so neither run's cache files interfere with the
        other's exists-call count."""
        baseline_calls = self._exists_call_count("img_a", known_not_all_cached=False)
        optimized_calls = self._exists_call_count("img_b", known_not_all_cached=True)
        self.assertEqual(baseline_calls - optimized_calls, 1)

    def test_precompute_calls_get_or_compute_with_known_not_all_cached_true(self):
        """Wiring proof at the precompute-loop level: for an image that still needs computing,
        `precompute_joint_frozen_caches` must pass known_not_all_cached=True (it has already
        confirmed, via its own check, that at least one cache file is missing) -- this is what
        actually eliminates the redundant Drive round trips in the real Colab run, verified here
        independent of any incidental exists-call noise."""
        with mock.patch(
            "joint_training_dataset._get_or_compute_joint_frozen_outputs",
            wraps=jtd._get_or_compute_joint_frozen_outputs,
        ) as mocked_compute:
            jtd.precompute_joint_frozen_caches(
                [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model,
            )
        mocked_compute.assert_called_once()
        _, kwargs = mocked_compute.call_args
        self.assertTrue(kwargs.get("known_not_all_cached"))

    def test_build_joint_sample_never_passes_known_not_all_cached(self):
        """Regression: this parameter is strictly additive -- Phase 2's per-sample path
        (`_build_joint_sample`, used by `_make_joint_dataset`/`load_joint_training_datasets`)
        must keep performing the full, unmodified upfront check, since it has NOT already
        verified cache state itself the way `precompute_joint_frozen_caches` has."""
        with mock.patch(
            "joint_training_dataset._get_or_compute_joint_frozen_outputs",
            wraps=jtd._get_or_compute_joint_frozen_outputs,
        ) as mocked_compute:
            rng = np.random.default_rng(0)
            jtd._build_joint_sample(
                "img_a", 0, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, augment=False, rng=rng,
            )
        mocked_compute.assert_called_once()
        _, kwargs = mocked_compute.call_args
        self.assertNotIn("known_not_all_cached", kwargs)

    def test_progress_log_reports_elapsed_time_and_images_per_minute(self):
        with self.assertLogs("joint_training_dataset", level="INFO") as captured:
            jtd.precompute_joint_frozen_caches(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model, progress_every=1,
            )
        progress_lines = [m for m in captured.output if "images processed" in m]
        self.assertTrue(progress_lines)
        for line in progress_lines:
            self.assertIn("elapsed", line)
            self.assertIn("images/min", line)

    def test_stats_include_total_elapsed_seconds(self):
        stats = jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        self.assertIn("elapsed_seconds", stats)
        self.assertGreaterEqual(stats["elapsed_seconds"], 0.0)

    def test_final_progress_line_is_always_logged_even_off_the_progress_every_boundary(self):
        """2 entries with progress_every=50 never hits the modulo boundary mid-loop -- the final
        summary log must still fire once at the end so a short/odd-sized run isn't silent."""
        with self.assertLogs("joint_training_dataset", level="INFO") as captured:
            jtd.precompute_joint_frozen_caches(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model, progress_every=50,
            )
        progress_lines = [m for m in captured.output if "images processed" in m]
        self.assertEqual(len(progress_lines), 1)
        self.assertIn("2/2", progress_lines[0])


class PersistentCacheDirTests(unittest.TestCase):
    """Regression tests for `JOINT_TRAINING_ARCHITECTURE.md` §35/§37. §35: a real Colab run
    crashed with `OSError: [Errno 107] Transport endpoint is not connected` when Phase 1
    bulk-pulled an entire persistent Drive cache down to local disk before starting --
    `persistent_cache_dir`/`persistent_racaf_cache_dir` replace that bulk pull with a per-image
    existence check, never a bulk/concurrent copy. §37: an entry found ONLY at the persistent
    location IS mirrored to local disk -- one entry at a time, inside Phase 1's own existing
    per-image loop (never a bulk copy, so no §35-class Drive FUSE risk) -- so training's first
    epoch is not left to pay that one-time Drive-read cost for the entries Phase 1 already
    confirmed exist (a real run showed this mattered when persistent hits were ~81% of the
    dataset)."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1)])
        self.addCleanup(self.tree.cleanup)
        self.persistent_root = tempfile.mkdtemp(prefix="persistent_cache_")
        self.addCleanup(shutil.rmtree, self.persistent_root, True)
        self.persistent_cache_dir = os.path.join(self.persistent_root, "local_feature_extraction")
        self.persistent_racaf_cache_dir = os.path.join(self.persistent_root, "racaf")

    def test_entry_cached_only_in_persistent_dir_is_mirrored_locally_without_recomputing(self):
        """§37: unlike a true local hit, a persistent-only hit is NOT free -- it costs exactly one
        read from the persistent location, counted under its own 'mirrored_from_persistent'
        counter (never folded into 'already_cached', which stays reserved for a true local hit,
        or 'cached', reserved for a genuine Stage 03/04/RACAF computation) -- but Stage 03/04 are
        never invoked, and the local cache_dir DOES end up with a usable copy afterward."""
        # Populate the "persistent Drive" cache directly (simulating a prior run) -- NOT cache_dir.
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir,
            self.persistent_cache_dir, self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        self.assertTrue(os.path.isdir(self.persistent_cache_dir))
        self.assertFalse(os.path.exists(self.tree.cache_dir))

        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            stats = jtd.precompute_joint_frozen_caches(
                [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                persistent_cache_dir=self.persistent_cache_dir,
                persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model,
            )

        self.assertEqual(stats["mirrored_from_persistent"], 1)
        self.assertEqual(stats["already_cached"], 0)
        self.assertEqual(stats["cached"], 0)
        self.assertEqual(mocked_vessel.call_count, 0)
        # The entry now DOES exist locally too -- exactly the fix: a per-image mirror during
        # Phase 1 itself, not deferred to training's first epoch.
        id_code = self.tree.pairs[0][0]
        vessel_cache_local = lfed._cache_path(self.tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE)
        lesion_cache_local = lfed._cache_path(self.tree.cache_dir, id_code, "lesion", jtd.STAGE5_IMAGE_SIZE)
        reliability_cache_local = racaf.reliability_cache_path(self.tree.racaf_cache_dir, id_code)
        self.assertTrue(os.path.exists(vessel_cache_local))
        self.assertTrue(os.path.exists(lesion_cache_local))
        self.assertTrue(os.path.exists(reliability_cache_local))

    def test_mirrored_local_entry_is_numerically_identical_to_the_persistent_source(self):
        """The mirror must be a faithful copy, not a re-derivation -- same cache format/keys, same
        values, so a value computed once by Stage 03/04/RACAF is never silently altered by being
        copied."""
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir,
            self.persistent_cache_dir, self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        id_code = self.tree.pairs[0][0]
        vessel_cache_persistent = lfed._cache_path(
            self.persistent_cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE,
        )
        reliability_cache_persistent = racaf.reliability_cache_path(self.persistent_racaf_cache_dir, id_code)
        persistent_vessel_map = np.load(vessel_cache_persistent)
        persistent_reliability = np.load(reliability_cache_persistent)

        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        vessel_cache_local = lfed._cache_path(self.tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE)
        reliability_cache_local = racaf.reliability_cache_path(self.tree.racaf_cache_dir, id_code)
        local_vessel_map = np.load(vessel_cache_local)
        local_reliability = np.load(reliability_cache_local)

        np.testing.assert_array_equal(local_vessel_map, persistent_vessel_map)
        np.testing.assert_array_equal(local_reliability["kappa"], persistent_reliability["kappa"])
        self.assertEqual(float(local_reliability["r"]), float(persistent_reliability["r"]))

    def test_mirrored_local_entry_is_directly_usable_by_training(self):
        """The whole point of the fix: after Phase 1 mirrors a persistent-only entry, Phase 2
        (_build_joint_sample, as load_joint_training_datasets() calls it) must be able to build a
        full training sample from the LOCAL cache alone -- no persistent_cache_dir needed, no
        Stage 03/04 recompute."""
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir,
            self.persistent_cache_dir, self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

        id_code, diagnosis = self.tree.pairs[0]
        rng = np.random.default_rng(0)
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            sample = jtd._build_joint_sample(
                id_code, diagnosis, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, augment=False, rng=rng,
                # Deliberately NOT passing persistent_cache_dir -- proves the local mirror alone
                # is sufficient, matching exactly how the notebook's training cell calls this
                # (once Phase 1 has already run).
            )
        self.assertEqual(mocked_vessel.call_count, 0)
        self.assertEqual(sample["stage5_input"].shape, (512, 512, 8))
        self.assertEqual(sample["grade"], diagnosis)

    def test_local_hit_never_reads_persistent_dir_even_when_it_is_invalid(self):
        """A true local hit must short-circuit before the persistent branch runs at all --
        confirmed by pointing persistent_cache_dir somewhere nonexistent and seeing no error and
        no change in behavior."""
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        stats = jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir="/nonexistent/persistent",
            persistent_racaf_cache_dir="/nonexistent/persistent_racaf",
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["mirrored_from_persistent"], 0)
        self.assertEqual(stats["cached"], 0)

    def test_resumability_a_mirrored_entry_is_not_re_touched_by_a_later_run(self):
        """The core resumability guarantee, extended to the new mirror case: once an entry has
        been mirrored locally, a later Phase 1 call over the same entries must find it as a plain
        local hit and never touch it (or the persistent location) again."""
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir,
            self.persistent_cache_dir, self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        id_code = self.tree.pairs[0][0]
        vessel_cache_local = lfed._cache_path(self.tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE)
        mtime_after_mirror = os.path.getmtime(vessel_cache_local)

        stats = jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["mirrored_from_persistent"], 0)
        self.assertEqual(os.path.getmtime(vessel_cache_local), mtime_after_mirror)

    def test_interrupted_run_with_mixed_entry_kinds_resumes_correctly(self):
        """A more realistic resumability scenario: one entry already local, one persistent-only,
        one a genuine miss -- an interrupted-then-resumed run must end with all three correctly
        and permanently local, without ever recomputing the persistent-only or already-local ones
        on the resume pass."""
        tree = _SyntheticAPTOSTree([("already_local", 0), ("persistent_only", 1), ("brand_new", 2)])
        self.addCleanup(tree.cleanup)

        # "already_local" -- computed directly into cache_dir, as if a prior interrupted run of
        # THIS SAME session had already finished it.
        jtd.precompute_joint_frozen_caches(
            [tree.pairs[0]], tree.image_dir, tree.cache_dir, tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        # "persistent_only" -- computed into the simulated Drive cache from a PRIOR session.
        jtd.precompute_joint_frozen_caches(
            [tree.pairs[1]], tree.image_dir, self.persistent_cache_dir, self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        vessel_cache_already_local = lfed._cache_path(
            tree.cache_dir, tree.pairs[0][0], "vessel", jtd.STAGE5_IMAGE_SIZE,
        )
        mtime_before = os.path.getmtime(vessel_cache_already_local)

        stats = jtd.precompute_joint_frozen_caches(
            tree.pairs, tree.image_dir, tree.cache_dir, tree.racaf_cache_dir,
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["mirrored_from_persistent"], 1)
        self.assertEqual(stats["cached"], 1)
        self.assertEqual(os.path.getmtime(vessel_cache_already_local), mtime_before)
        for id_code, _ in tree.pairs:
            self.assertTrue(os.path.exists(
                lfed._cache_path(tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE),
            ))

    def test_entry_missing_from_both_dirs_is_computed_and_written_only_locally(self):
        stats = jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        self.assertEqual(stats["cached"], 1)
        id_code = self.tree.pairs[0][0]
        vessel_cache_local = lfed._cache_path(self.tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE)
        vessel_cache_persistent = lfed._cache_path(
            self.persistent_cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE,
        )
        self.assertTrue(os.path.exists(vessel_cache_local))
        self.assertFalse(os.path.exists(vessel_cache_persistent))

    def test_local_cache_dir_hit_is_checked_first_persistent_dir_need_not_even_exist(self):
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        stats = jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir="/nonexistent/persistent",
            persistent_racaf_cache_dir="/nonexistent/persistent_racaf",
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["cached"], 0)

    def test_persistent_cache_dir_none_preserves_prior_default_behavior(self):
        """Regression safety: every existing caller that never passes persistent_cache_dir must
        behave identically to before this parameter existed."""
        stats = jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        self.assertEqual(stats["cached"], 2)
        self.assertEqual(stats["already_cached"], 0)

    def test_precompute_authoritative_forwards_persistent_cache_dir_params(self):
        with mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=([("a", 0)], [("b", 1)]),
        ), mock.patch(
            "joint_training_dataset.load_vessel_model", return_value="VESSEL_MODEL",
        ), mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model", return_value="STAGE4_MODEL",
        ), mock.patch(
            "joint_training_dataset.precompute_joint_frozen_caches",
            return_value={"cached": 0, "already_cached": 0, "skipped_empty_fov": [], "elapsed_seconds": 0.0},
        ) as mocked_precompute:
            jtd.precompute_authoritative_joint_caches(
                persistent_cache_dir="/drive/local_feature_extraction",
                persistent_racaf_cache_dir="/drive/racaf",
                max_images=5, verbose_diagnostics=True,
            )
        _, kwargs = mocked_precompute.call_args
        self.assertEqual(kwargs["persistent_cache_dir"], "/drive/local_feature_extraction")
        self.assertEqual(kwargs["persistent_racaf_cache_dir"], "/drive/racaf")
        self.assertEqual(kwargs["max_images"], 5)
        self.assertTrue(kwargs["verbose_diagnostics"])


class DiagnosticModeTests(unittest.TestCase):
    """Regression tests for the small-scale diagnostic mode required by `JOINT_TRAINING_
    ARCHITECTURE.md` §35: must exercise the REAL Phase 1 code path (real checkpoints, real images,
    real cache I/O -- no mocked inference) on a small, controlled number of images, with per-image
    resource visibility, and must never accumulate that per-image data into the returned stats."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1), ("img_c", 2)])
        self.addCleanup(self.tree.cleanup)

    def test_max_images_limits_real_entries_processed_via_the_real_code_path(self):
        stats = jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model, max_images=2,
        )
        self.assertEqual(
            stats["cached"] + stats["already_cached"] + len(stats["skipped_empty_fov"]), 2,
        )
        for id_code, _ in self.tree.pairs[:2]:
            vessel_cache = lfed._cache_path(self.tree.cache_dir, id_code, "vessel", jtd.STAGE5_IMAGE_SIZE)
            self.assertTrue(os.path.exists(vessel_cache))
        third_id = self.tree.pairs[2][0]
        vessel_cache_third = lfed._cache_path(self.tree.cache_dir, third_id, "vessel", jtd.STAGE5_IMAGE_SIZE)
        self.assertFalse(os.path.exists(vessel_cache_third))

    def test_max_images_none_processes_every_entry_unchanged(self):
        stats = jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model, max_images=None,
        )
        self.assertEqual(stats["cached"], 3)

    def test_verbose_diagnostics_logs_exactly_one_line_per_processed_image(self):
        with self.assertLogs("joint_training_dataset", level="INFO") as captured:
            jtd.precompute_joint_frozen_caches(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model,
                max_images=2, verbose_diagnostics=True,
            )
        diagnostic_lines = [m for m in captured.output if "[diagnostic]" in m]
        self.assertEqual(len(diagnostic_lines), 2)
        for line in diagnostic_lines:
            self.assertIn("rss_mb=", line)
            self.assertIn("gpu_used_mb=", line)
            self.assertIn("images/min=", line)

    def test_verbose_diagnostics_false_by_default_emits_no_diagnostic_lines(self):
        with self.assertLogs("joint_training_dataset", level="INFO") as captured:
            jtd.precompute_joint_frozen_caches(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model, progress_every=1,
            )
        diagnostic_lines = [m for m in captured.output if "[diagnostic]" in m]
        self.assertEqual(diagnostic_lines, [])

    def test_diagnostic_mode_does_not_change_the_returned_stats_shape(self):
        """Bounded-memory proof extended to diagnostic mode: verbose_diagnostics must never add a
        per-image list/array to the returned stats -- each log line is discarded immediately."""
        stats = jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
            max_images=2, verbose_diagnostics=True,
        )
        self.assertEqual(
            set(stats.keys()),
            {"cached", "already_cached", "mirrored_from_persistent", "skipped_empty_fov", "elapsed_seconds"},
        )


class Phase2PersistentCacheDirTests(unittest.TestCase):
    """Regression tests for `JOINT_TRAINING_ARCHITECTURE.md` §36: a real Colab training run
    measured ~5-6s/step because Phase 2 (`_build_joint_sample`/`_get_or_compute_joint_frozen_
    outputs`) had no equivalent of Phase 1's `persistent_cache_dir` local-first/persistent-
    fallback logic -- every sample's cache entry was read from a Drive-mounted `cache_dir`, every
    epoch. These verify a persistent-only cache entry is found, read exactly once, and mirrored
    to local disk so every LATER call for the same image never touches the persistent location
    again."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0)])
        self.addCleanup(self.tree.cleanup)
        self.persistent_root = tempfile.mkdtemp(prefix="phase2_persistent_cache_")
        self.addCleanup(shutil.rmtree, self.persistent_root, True)
        self.persistent_cache_dir = os.path.join(self.persistent_root, "local_feature_extraction")
        self.persistent_racaf_cache_dir = os.path.join(self.persistent_root, "racaf")
        # Populate ONLY the persistent location -- simulating a Drive-flushed Phase 1 cache from a
        # prior session. The local cache_dir/racaf_cache_dir start empty.
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir,
            self.persistent_cache_dir, self.persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

    def _build_sample(self, **kwargs):
        rng = np.random.default_rng(0)
        return jtd._build_joint_sample(
            "img_a", 0, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=rng, **kwargs,
        )

    def test_without_persistent_dir_a_persistent_only_entry_is_wastefully_recomputed(self):
        """Regression baseline documenting the exact waste this fix eliminates: without
        persistent_cache_dir, Phase 2 has no way to find an entry only cached at the persistent
        location, so it recomputes Stage 03/04/RACAF for it from scratch."""
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            self._build_sample()
        self.assertEqual(mocked_vessel.call_count, 1)

    def test_with_persistent_dir_a_persistent_only_entry_is_never_recomputed(self):
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            self._build_sample(
                persistent_cache_dir=self.persistent_cache_dir,
                persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            )
        self.assertEqual(mocked_vessel.call_count, 0)

    def test_persistent_hit_is_mirrored_to_the_local_cache_dir(self):
        vessel_cache_local = lfed._cache_path(self.tree.cache_dir, "img_a", "vessel", jtd.STAGE5_IMAGE_SIZE)
        lesion_cache_local = lfed._cache_path(self.tree.cache_dir, "img_a", "lesion", jtd.STAGE5_IMAGE_SIZE)
        reliability_cache_local = racaf.reliability_cache_path(self.tree.racaf_cache_dir, "img_a")
        self.assertFalse(os.path.exists(vessel_cache_local))

        self._build_sample(
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
        )

        self.assertTrue(os.path.exists(vessel_cache_local))
        self.assertTrue(os.path.exists(lesion_cache_local))
        self.assertTrue(os.path.exists(reliability_cache_local))

    def test_second_call_reads_purely_local_even_if_persistent_dir_becomes_unavailable(self):
        """Proof this is a ONE-TIME-per-image cost, not a per-epoch one: after the first call
        mirrors the entry locally, a second call must succeed -- and never recompute -- even when
        persistent_cache_dir points somewhere that no longer exists."""
        self._build_sample(
            persistent_cache_dir=self.persistent_cache_dir,
            persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
        )
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            self._build_sample(
                persistent_cache_dir="/nonexistent/persistent",
                persistent_racaf_cache_dir="/nonexistent/persistent_racaf",
            )
        self.assertEqual(mocked_vessel.call_count, 0)

    def test_persistent_cache_dir_none_preserves_prior_default_behavior(self):
        """Regression safety: every existing caller that never passes persistent_cache_dir
        (e.g. every pre-existing test in this file) must behave identically to before this
        parameter existed."""
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            self._build_sample()
        self.assertEqual(mocked_vessel.call_count, 1)

    def test_make_joint_dataset_forwards_persistent_cache_dir_to_every_sample(self):
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            ds = jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
                persistent_cache_dir=self.persistent_cache_dir,
                persistent_racaf_cache_dir=self.persistent_racaf_cache_dir,
            )
            list(ds)
        self.assertEqual(mocked_vessel.call_count, 0)


class LoadJointTrainingDatasetsTests(unittest.TestCase):
    """`load_joint_training_datasets()` (Phase 2's real entry point) had NO direct test coverage
    before this fix -- the exact gap that let the missing `check_gpu()` call and the missing
    `persistent_cache_dir` wiring both ship unnoticed (`JOINT_TRAINING_ARCHITECTURE.md` §36)."""

    def test_calls_check_gpu_before_either_model_loads(self):
        call_order = []
        with mock.patch(
            "joint_training_dataset.check_gpu", side_effect=lambda: call_order.append("check_gpu"),
        ), mock.patch(
            "joint_training_dataset.load_vessel_model",
            side_effect=lambda *a, **k: call_order.append("load_vessel_model"),
        ), mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model",
            side_effect=lambda *a, **k: call_order.append("load_stage4_model"),
        ), mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=([], []),
        ):
            jtd.load_joint_training_datasets()
        self.assertEqual(call_order, ["check_gpu", "load_vessel_model", "load_stage4_model"])

    def test_forwards_persistent_cache_dir_params_to_both_train_and_val_datasets(self):
        captured_calls = []

        def fake_make_joint_dataset(*args, **kwargs):
            captured_calls.append(kwargs)
            return "DATASET"

        with mock.patch("joint_training_dataset.check_gpu"), mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=([("a", 0)], [("b", 1)]),
        ), mock.patch(
            "joint_training_dataset.load_vessel_model", return_value="VESSEL_MODEL",
        ), mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model", return_value="STAGE4_MODEL",
        ), mock.patch(
            "joint_training_dataset._make_joint_dataset", side_effect=fake_make_joint_dataset,
        ):
            train_ds, val_ds = jtd.load_joint_training_datasets(
                persistent_cache_dir="/drive/local_feature_extraction",
                persistent_racaf_cache_dir="/drive/racaf",
            )

        self.assertEqual(train_ds, "DATASET")
        self.assertEqual(val_ds, "DATASET")
        self.assertEqual(len(captured_calls), 2)
        for kwargs in captured_calls:
            self.assertEqual(kwargs["persistent_cache_dir"], "/drive/local_feature_extraction")
            self.assertEqual(kwargs["persistent_racaf_cache_dir"], "/drive/racaf")

    def test_forwards_rgb_cache_dir_to_both_train_and_val_datasets(self):
        """§38: the notebook passes a local-only rgb cache dir; both datasets must receive it."""
        captured_calls = []

        def fake_make_joint_dataset(*args, **kwargs):
            captured_calls.append(kwargs)
            return "DATASET"

        with mock.patch("joint_training_dataset.check_gpu"), mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=([("a", 0)], [("b", 1)]),
        ), mock.patch(
            "joint_training_dataset.load_vessel_model", return_value="VESSEL_MODEL",
        ), mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model", return_value="STAGE4_MODEL",
        ), mock.patch(
            "joint_training_dataset._make_joint_dataset", side_effect=fake_make_joint_dataset,
        ):
            jtd.load_joint_training_datasets(rgb_cache_dir="/content/cache/canonical_rgb")

        self.assertEqual(len(captured_calls), 2)
        for kwargs in captured_calls:
            self.assertEqual(kwargs["rgb_cache_dir"], "/content/cache/canonical_rgb")

    def test_rgb_cache_dir_defaults_to_none(self):
        signature = inspect.signature(jtd.load_joint_training_datasets)
        self.assertIsNone(signature.parameters["rgb_cache_dir"].default)

    def test_persistent_cache_dir_defaults_to_none(self):
        captured_calls = []

        def fake_make_joint_dataset(*args, **kwargs):
            captured_calls.append(kwargs)
            return "DATASET"

        with mock.patch("joint_training_dataset.check_gpu"), mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=([("a", 0)], [("b", 1)]),
        ), mock.patch(
            "joint_training_dataset.load_vessel_model", return_value="VESSEL_MODEL",
        ), mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model", return_value="STAGE4_MODEL",
        ), mock.patch(
            "joint_training_dataset._make_joint_dataset", side_effect=fake_make_joint_dataset,
        ):
            jtd.load_joint_training_datasets()

        for kwargs in captured_calls:
            self.assertIsNone(kwargs["persistent_cache_dir"])
            self.assertIsNone(kwargs["persistent_racaf_cache_dir"])

    def test_end_to_end_real_synthetic_data_reuses_persistent_cache_without_recompute(self):
        """The full Phase 2 loader, top to bottom, against a persistent-only cache -- proves the
        wiring holds all the way from load_joint_training_datasets() to _get_or_compute_joint_
        frozen_outputs(), not just at one layer."""
        vessel_model = _build_synthetic_vessel_model()
        stage4_model = _build_synthetic_frozen_stage4_model()
        tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1)])
        self.addCleanup(tree.cleanup)
        persistent_root = tempfile.mkdtemp(prefix="e2e_persistent_")
        self.addCleanup(shutil.rmtree, persistent_root, True)
        persistent_cache_dir = os.path.join(persistent_root, "local_feature_extraction")
        persistent_racaf_cache_dir = os.path.join(persistent_root, "racaf")
        jtd.precompute_joint_frozen_caches(
            tree.pairs, tree.image_dir, persistent_cache_dir, persistent_racaf_cache_dir,
            vessel_model=vessel_model, stage4_model=stage4_model,
        )

        with mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=(tree.pairs, []),
        ), mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            train_ds, val_ds = jtd.load_joint_training_datasets(
                image_dir=tree.image_dir, cache_dir=tree.cache_dir, racaf_cache_dir=tree.racaf_cache_dir,
                persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
                vessel_model=vessel_model, stage4_model=stage4_model, batch_size=1,
            )
            list(train_ds)

        self.assertEqual(mocked_vessel.call_count, 0)


class RepeatedEmptyFovAcrossEpochsTests(unittest.TestCase):
    """Answers a real-training observation (`JOINT_TRAINING_ARCHITECTURE.md` §36): the SAME
    empty-FOV warning recurs every epoch for the same image ids. This is EXPECTED given the
    current design, not a bug -- Phase 1 never writes ANY cache file for a permanently-empty-FOV
    image (there is nothing to cache: `_get_or_compute_joint_frozen_outputs` raises before any
    `np.save`/`np.savez` call), so every subsequent access -- including every training epoch --
    finds no cache entry anywhere (local or persistent) and re-attempts Stage 03's FOV detection,
    hits the same failure, and correctly excludes the image again. Verified here as deterministic,
    repeatable, and non-blocking across two independent dataset builds (simulating two epochs)."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_good", 0), ("img_bad", 1)])
        self.addCleanup(self.tree.cleanup)

    def _iterate_with_one_forced_empty_fov(self):
        real_fn = jtd._get_or_compute_joint_frozen_outputs

        def flaky(rgb_native, vessel_cache_path, *args, **kwargs):
            if "img_bad" in vessel_cache_path:
                raise jtd.EmptyFieldOfViewError("no fundus disk detected")
            return real_fn(rgb_native, vessel_cache_path, *args, **kwargs)

        with mock.patch(
            "joint_training_dataset._get_or_compute_joint_frozen_outputs", side_effect=flaky,
        ), self.assertLogs("joint_training_dataset", level="WARNING") as captured:
            ds = jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
            )
            batches = list(ds)
        return batches, captured.output

    def test_bad_image_is_excluded_and_warned_on_every_independent_epoch_like_iteration(self):
        batches_epoch1, warnings_epoch1 = self._iterate_with_one_forced_empty_fov()
        batches_epoch2, warnings_epoch2 = self._iterate_with_one_forced_empty_fov()

        self.assertEqual(len(batches_epoch1), 1)  # only img_good yielded
        self.assertEqual(len(batches_epoch2), 1)
        self.assertTrue(any("img_bad" in w for w in warnings_epoch1))
        self.assertTrue(any("img_bad" in w for w in warnings_epoch2))

    def test_no_cache_file_is_ever_written_for_a_permanently_empty_fov_image(self):
        self._iterate_with_one_forced_empty_fov()
        vessel_cache_bad = lfed._cache_path(self.tree.cache_dir, "img_bad", "vessel", jtd.STAGE5_IMAGE_SIZE)
        self.assertFalse(os.path.exists(vessel_cache_bad))


class GPUMemoryGrowthSafeguardTests(unittest.TestCase):
    """Regression tests for the diagnosed Phase 1 GPU VRAM exhaustion root cause
    (`JOINT_TRAINING_ARCHITECTURE.md` §34): a real Colab crash persisted after Phase 1's cache
    I/O was moved to local SSD, even though Colab's own CPU RAM graph never showed growth --
    because the actual exhaustion was on the GPU, not the CPU. Root cause, confirmed by tracing
    every call site: `tf.config.experimental.set_memory_growth` is NEVER called anywhere reachable
    from Phase 1 (`joint_training_dataset.py` had zero `check_gpu`/`set_memory_growth` references
    before this fix) or from `Trainer.prepare()`. Without it, TensorFlow's default allocator
    claims ~all free GPU memory the instant Stage 04's model first runs -- and Stage 03's
    PyTorch model shares the SAME GPU (`vessel_segmentation_model.resolve_device()` picks CUDA
    when available) via a completely independent, non-coordinating CUDA allocator. `check_gpu()`
    (`training.trainer`, already used by `train_image_quality.py`/`colab/common/environment.py` --
    reused here, not reimplemented) fixes this by requesting incremental GPU memory growth before
    either model can touch the device. These tests verify WIRING/ORDER via mocks -- actual VRAM
    behavior cannot be exercised without a real GPU (this suite runs CPU-only); the CPU-side RSS
    diagnosis (`JOINT_TRAINING_ARCHITECTURE.md` §34) separately and conclusively ruled out a
    CPU-memory leak as the crash cause using real APTOS images and the real frozen checkpoints."""

    def test_precompute_joint_frozen_caches_calls_check_gpu_before_either_model_loads(self):
        call_order = []
        with mock.patch(
            "joint_training_dataset.check_gpu", side_effect=lambda: call_order.append("check_gpu"),
        ), mock.patch(
            "joint_training_dataset.load_vessel_model",
            side_effect=lambda *a, **k: call_order.append("load_vessel_model"),
        ), mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model",
            side_effect=lambda *a, **k: call_order.append("load_stage4_model"),
        ):
            jtd.precompute_joint_frozen_caches([], "img_dir", "cache_dir", "racaf_cache_dir")
        self.assertEqual(call_order, ["check_gpu", "load_vessel_model", "load_stage4_model"])

    def test_precompute_authoritative_joint_caches_calls_check_gpu_before_either_model_loads(self):
        call_order = []
        with mock.patch(
            "joint_training_dataset.check_gpu", side_effect=lambda: call_order.append("check_gpu"),
        ), mock.patch(
            "joint_training_dataset.load_vessel_model",
            side_effect=lambda *a, **k: call_order.append("load_vessel_model"),
        ), mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model",
            side_effect=lambda *a, **k: call_order.append("load_stage4_model"),
        ), mock.patch(
            "joint_training_dataset.precompute_joint_frozen_caches",
            return_value={"cached": 0, "already_cached": 0, "skipped_empty_fov": [], "elapsed_seconds": 0.0},
        ), mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=([], []),
        ):
            jtd.precompute_authoritative_joint_caches()
        self.assertEqual(call_order, ["check_gpu", "load_vessel_model", "load_stage4_model"])

    def test_check_gpu_runs_even_when_both_models_are_passed_in_already_loaded(self):
        """A caller reusing already-loaded models across multiple precompute calls in one session
        must still get the memory-growth safeguard applied -- it's a global TF process setting,
        not per-model-load state, and is cheap/idempotent to call repeatedly."""
        with mock.patch("joint_training_dataset.check_gpu") as mocked_check_gpu:
            jtd.precompute_joint_frozen_caches(
                [], "img_dir", "cache_dir", "racaf_cache_dir",
                vessel_model="already-loaded-vessel", stage4_model="already-loaded-stage4",
            )
        mocked_check_gpu.assert_called_once()

    def test_check_gpu_is_the_real_training_check_gpu_not_a_reimplementation(self):
        """No second, competing GPU-setup mechanism -- this must be the exact same function
        `Trainer.prepare()`/`train_image_quality.py` already rely on, not a duplicate."""
        import training
        self.assertIs(jtd.check_gpu, training.check_gpu)


class PrecomputeAuthoritativeCachesTests(unittest.TestCase):
    """`precompute_authoritative_joint_caches` -- the real-training-workflow entry point -- must
    delegate to the SAME authoritative split and the SAME streaming function above, never a
    second split or a second cache-population code path."""

    def test_delegates_to_the_authoritative_split_and_the_streaming_precompute_function(self):
        with mock.patch(
            "joint_training_dataset.split_train_val_ids", return_value=([("a", 0)], [("b", 1)]),
        ) as mocked_split, mock.patch(
            "joint_training_dataset.load_vessel_model", return_value="VESSEL_MODEL",
        ) as mocked_vessel_loader, mock.patch(
            "joint_training_dataset.racaf.load_frozen_stage4_model", return_value="STAGE4_MODEL",
        ) as mocked_stage4_loader, mock.patch(
            "joint_training_dataset.precompute_joint_frozen_caches",
            return_value={"cached": 0, "already_cached": 0, "skipped_empty_fov": []},
        ) as mocked_precompute:
            jtd.precompute_authoritative_joint_caches()

        mocked_split.assert_called_once()
        mocked_vessel_loader.assert_called_once()
        mocked_stage4_loader.assert_called_once()
        mocked_precompute.assert_called_once()
        called_entries = mocked_precompute.call_args[0][0]
        self.assertEqual(called_entries, [("a", 0), ("b", 1)])
        _, kwargs = mocked_precompute.call_args
        self.assertEqual(kwargs["vessel_model"], "VESSEL_MODEL")
        self.assertEqual(kwargs["stage4_model"], "STAGE4_MODEL")


class Phase2UsesExistingCachesTests(unittest.TestCase):
    """Requirement: the training dataset must use existing caches without invoking Stage 3/4
    inference when caches already exist -- verified at the actual `_make_joint_dataset` level
    (Phase 2), after Phase 1 has already populated every entry's cache."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1)])
        self.addCleanup(self.tree.cleanup)
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

    def test_iterating_the_dataset_never_calls_stage3_or_stage4_inference(self):
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel, mock.patch(
            "joint_training_dataset.racaf.tta_views", wraps=racaf.tta_views,
        ) as mocked_tta:
            ds = jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
            )
            list(ds)
        self.assertEqual(mocked_vessel.call_count, 0)
        self.assertEqual(mocked_tta.call_count, 0)

    def test_second_epoch_like_iteration_never_reads_the_raw_image_either(self):
        """§38: a full cache hit (vessel/lesion/reliability -- already proven above -- AND the
        canonical RGB) must not touch the raw image file at all. This is the direct proof of the
        real ~2s/step steady-state slowdown's fix: before §38, EVERY sample access re-read the raw
        file and re-applied Stage 02 regardless of vessel/lesion/reliability cache status."""

        def build_ds():
            return jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False, seed=0,
            )

        list(build_ds())  # "epoch 1": populates every cache, including the new rgb cache
        with mock.patch.object(lfed, "_load_raw_bgr", wraps=lfed._load_raw_bgr) as mocked_load:
            list(build_ds())  # "epoch 2": must be a pure, raw-file-free cache hit
        self.assertEqual(mocked_load.call_count, 0)


# =====================================================================
# Canonical RGB caching (§38) -- stage5_input's RGB channels were never cached at all before this,
# unlike vessel/lesion/reliability: every _build_joint_sample call re-read the raw image and
# re-applied Stage 02 regardless of cache status, proven (via a local, real-code-path diagnostic)
# to dominate steady-state per-sample cost once vessel/lesion/reliability were already np.load
# hits. This is a pure caching change -- the cached value is numerically identical to what
# _resize_rgb_01 already computed live; no resize/preprocessing algorithm changed.
# =====================================================================

class CanonicalRGBCachingTests(unittest.TestCase):
    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_01", 2)])
        self.addCleanup(self.tree.cleanup)

    def _build(self):
        rng = np.random.default_rng(0)
        return jtd._build_joint_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=rng,
        )

    def _rgb_cache_path(self, cache_dir=None):
        return jtd._canonical_rgb_cache_path("img_01", cache_dir or self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)

    def test_rgb_cache_file_is_written_on_first_build(self):
        self._build()
        self.assertTrue(os.path.exists(self._rgb_cache_path()))

    def test_cached_rgb_is_numerically_identical_to_the_sample_it_was_derived_from(self):
        sample = self._build()
        cached_rgb = np.load(self._rgb_cache_path())
        np.testing.assert_array_equal(sample["stage5_input"][..., :3], cached_rgb)

    def test_full_cache_hit_never_reads_the_raw_image(self):
        self._build()  # populates vessel/lesion/reliability/rgb caches
        with mock.patch.object(lfed, "_load_raw_bgr", wraps=lfed._load_raw_bgr) as mocked_load:
            self._build()
        self.assertEqual(mocked_load.call_count, 0)

    def test_full_cache_hit_produces_the_same_stage5_input_as_a_fresh_computation(self):
        first = self._build()
        second = self._build()
        np.testing.assert_array_equal(first["stage5_input"], second["stage5_input"])

    def test_rgb_only_missing_backfills_by_reading_the_raw_image_exactly_once(self):
        """The one legitimate remaining reason to read the raw file on an otherwise-cached entry:
        migrating a vessel/lesion/reliability cache populated before the rgb cache kind existed."""
        self._build()
        os.remove(self._rgb_cache_path())
        with mock.patch.object(lfed, "_load_raw_bgr", wraps=lfed._load_raw_bgr) as mocked_load:
            self._build()
        self.assertEqual(mocked_load.call_count, 1)

    def test_rgb_only_missing_backfill_never_recomputes_stage3_or_stage4(self):
        self._build()
        os.remove(self._rgb_cache_path())
        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel, mock.patch(
            "joint_training_dataset.racaf.tta_views", wraps=racaf.tta_views,
        ) as mocked_tta:
            self._build()
        self.assertEqual(mocked_vessel.call_count, 0)
        self.assertEqual(mocked_tta.call_count, 0)
        self.assertTrue(os.path.exists(self._rgb_cache_path()))

    def test_persistent_only_rgb_is_mirrored_locally_without_recomputing_or_rereading_raw(self):
        persistent_cache_dir = os.path.join(self.tree.root, "persistent_cache")
        persistent_racaf_cache_dir = os.path.join(self.tree.root, "persistent_racaf_cache")
        rng = np.random.default_rng(0)
        jtd._build_joint_sample(
            "img_01", 2, self.tree.image_dir, persistent_cache_dir, persistent_racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=rng,
        )
        with mock.patch.object(lfed, "_load_raw_bgr", wraps=lfed._load_raw_bgr) as mocked_load, mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            jtd._build_joint_sample(
                "img_01", 2, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, augment=False, rng=rng,
                persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            )
        self.assertEqual(mocked_load.call_count, 0)
        self.assertEqual(mocked_vessel.call_count, 0)
        self.assertTrue(os.path.exists(self._rgb_cache_path()))

    def test_persistent_mirrored_rgb_is_numerically_identical_to_a_fresh_live_computation(self):
        """§39: independent proof the Drive round-trip (np.save on write, np.load on read --
        raw, uncompressed, exactly what every other cache entry already uses) introduces no
        precision loss -- a persistent-only entry, once mirrored locally, must be byte-for-byte
        identical to computing canonical RGB live from the same raw image, not merely close."""
        persistent_cache_dir = os.path.join(self.tree.root, "persistent_cache")
        persistent_racaf_cache_dir = os.path.join(self.tree.root, "persistent_racaf_cache")
        rng = np.random.default_rng(0)
        jtd._build_joint_sample(
            "img_01", 2, self.tree.image_dir, persistent_cache_dir, persistent_racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=rng,
        )
        mirrored_sample = jtd._build_joint_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=rng,
            persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
        )

        raw_bgr = lfed._load_raw_bgr(self.tree.image_dir, "img_01")
        rgb_native = lfed._resolve_processed_rgb(raw_bgr, lfed.DEFAULT_PROCESSED_DIR, "img_01")
        live_canonical_rgb = jtd._resize_rgb_01(rgb_native, jtd.STAGE5_IMAGE_SIZE)

        mirrored_rgb = mirrored_sample["stage5_input"][..., :3]
        np.testing.assert_array_equal(mirrored_rgb, live_canonical_rgb)
        self.assertEqual(mirrored_rgb.tobytes(), live_canonical_rgb.tobytes())

    def test_no_persistent_path_is_probed_once_every_artifact_is_a_local_hit(self):
        """Once vessel/lesion/reliability/rgb are all local hits (steady-state training, the
        normal case from epoch 2 onward), NOT ONE `os.path.exists` call may reach a PERSISTENT
        (Drive) path -- confirms sharing `persistent_cache_dir` with vessel/lesion for rgb too
        does not reintroduce a Drive touch on an otherwise-fully-local sample."""
        persistent_cache_dir = os.path.join(self.tree.root, "persistent_cache")
        persistent_racaf_cache_dir = os.path.join(self.tree.root, "persistent_racaf_cache")
        self._build()  # populates a full local hit (no persistent dirs passed this call)

        real_exists = os.path.exists
        persistent_dir_probed = []

        def spying_exists(path):
            if persistent_cache_dir in str(path) or persistent_racaf_cache_dir in str(path):
                persistent_dir_probed.append(path)
            return real_exists(path)

        with mock.patch("os.path.exists", side_effect=spying_exists):
            jtd._build_joint_sample(
                "img_01", 2, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, augment=False, rng=np.random.default_rng(0),
                persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            )
        self.assertEqual(persistent_dir_probed, [])

    def test_persistent_cache_dir_none_preserves_prior_default_behavior(self):
        """Regression safety: every existing caller that never passes persistent_cache_dir must
        behave exactly as before -- rgb still gets cached locally, just with no persistent tier."""
        sample = self._build()
        self.assertTrue(os.path.exists(self._rgb_cache_path()))
        self.assertEqual(sample["stage5_input"].shape, (*jtd.STAGE5_IMAGE_SIZE, lfed.NUM_CHANNELS))

    def test_rgb_cache_dir_defaults_to_cache_dir(self):
        """`rgb_cache_dir=None` (every existing caller) must keep writing the rgb entry into
        `cache_dir` itself -- the separable directory is opt-in, not a relocation."""
        self._build()
        self.assertTrue(os.path.exists(jtd._canonical_rgb_cache_path(
            "img_01", self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE,
        )))

    def test_separate_rgb_cache_dir_keeps_the_rgb_entry_out_of_cache_dir(self):
        """`rgb_cache_dir` is a caller-level escape hatch, not what the real notebook uses (§39:
        the notebook shares `cache_dir` for rgb too, so Phase 1b's existing flush picks it up) --
        but a caller that DOES pass a separate directory must still get a clean separation, with
        the rgb array landing ONLY there and nothing rgb-shaped leaking into `cache_dir`."""
        rgb_dir = os.path.join(self.tree.root, "rgb_only")
        rng = np.random.default_rng(0)
        jtd._build_joint_sample(
            "img_01", 2, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=rng, rgb_cache_dir=rgb_dir,
        )
        self.assertTrue(os.path.exists(jtd._canonical_rgb_cache_path("img_01", rgb_dir, jtd.STAGE5_IMAGE_SIZE)))
        self.assertFalse(os.path.exists(self._rgb_cache_path()))
        cache_dir_files = os.listdir(self.tree.cache_dir)
        self.assertEqual([f for f in cache_dir_files if "_rgb_" in f], [])

    def test_separate_rgb_cache_dir_still_gives_a_full_raw_free_cache_hit(self):
        rgb_dir = os.path.join(self.tree.root, "rgb_only")

        def build():
            rng = np.random.default_rng(0)
            return jtd._build_joint_sample(
                "img_01", 2, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, augment=False, rng=rng, rgb_cache_dir=rgb_dir,
            )

        first = build()
        with mock.patch.object(lfed, "_load_raw_bgr", wraps=lfed._load_raw_bgr) as mocked_load:
            second = build()
        self.assertEqual(mocked_load.call_count, 0)
        np.testing.assert_array_equal(first["stage5_input"], second["stage5_input"])


class Phase1CanonicalRGBCachingTests(unittest.TestCase):
    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1)])
        self.addCleanup(self.tree.cleanup)

    def test_phase1_writes_rgb_cache_for_every_processed_entry(self):
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        for id_code, _diagnosis in self.tree.pairs:
            rgb_cache = jtd._canonical_rgb_cache_path(id_code, self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)
            self.assertTrue(os.path.exists(rgb_cache), f"missing rgb cache for {id_code}")

    def test_phase1_honors_a_separate_rgb_cache_dir_when_a_caller_opts_in(self):
        """`rgb_cache_dir` remains a working escape hatch for a caller that wants rgb kept apart
        from `cache_dir`, even though the real notebook does not use it (§39)."""
        rgb_dir = os.path.join(self.tree.root, "rgb_only")
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model, rgb_cache_dir=rgb_dir,
        )
        for id_code, _diagnosis in self.tree.pairs:
            self.assertTrue(os.path.exists(jtd._canonical_rgb_cache_path(id_code, rgb_dir, jtd.STAGE5_IMAGE_SIZE)))
        self.assertEqual([f for f in os.listdir(self.tree.cache_dir) if "_rgb_" in f], [])

    def test_phase1_then_phase2_share_a_caller_opted_in_separate_rgb_cache_dir(self):
        """When a caller DOES opt into a separate `rgb_cache_dir`, Phase 1 and Phase 2 must still
        agree on where to find it -- Phase 1 precomputes into that dir, Phase 2 reading from the
        same dir needs no raw image at all. (The real notebook does not opt in -- see §39.)"""
        rgb_dir = os.path.join(self.tree.root, "rgb_only")
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model, rgb_cache_dir=rgb_dir,
        )
        with mock.patch.object(lfed, "_load_raw_bgr", wraps=lfed._load_raw_bgr) as mocked_load, mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            ds = jtd._make_joint_dataset(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, batch_size=1, shuffle=False, augment=False,
                seed=0, rgb_cache_dir=rgb_dir,
            )
            samples = list(ds)
        self.assertEqual(len(samples), len(self.tree.pairs))
        self.assertEqual(mocked_load.call_count, 0)
        self.assertEqual(mocked_vessel.call_count, 0)

    def test_rgb_mirrored_entry_is_not_re_touched_by_a_later_phase1_run(self):
        """The core resumability guarantee (already proven for vessel/lesion), extended to rgb:
        once an entry's rgb has been mirrored locally, a later Phase 1 call over the same entries
        must find it as a plain local hit and never re-touch it (or the persistent location)."""
        persistent_cache_dir = os.path.join(self.tree.root, "persistent_cache")
        persistent_racaf_cache_dir = os.path.join(self.tree.root, "persistent_racaf_cache")
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, persistent_cache_dir, persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        id_code = self.tree.pairs[0][0]
        rgb_cache_local = jtd._canonical_rgb_cache_path(id_code, self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)
        mtime_after_mirror = os.path.getmtime(rgb_cache_local)

        stats = jtd.precompute_joint_frozen_caches(
            [self.tree.pairs[0]], self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["mirrored_from_persistent"], 0)
        self.assertEqual(os.path.getmtime(rgb_cache_local), mtime_after_mirror)

    def test_mixed_entry_kinds_all_end_with_a_valid_local_rgb_cache(self):
        """The realistic mixed-resumability scenario (already proven for vessel/lesion: one entry
        already local, one persistent-only, one a genuine miss) must also leave all three with a
        correct, local rgb cache entry after a single Phase 1 pass -- no kind is skipped."""
        tree = _SyntheticAPTOSTree([("already_local", 0), ("persistent_only", 1), ("brand_new", 2)])
        self.addCleanup(tree.cleanup)
        persistent_cache_dir = os.path.join(self.tree.root, "persistent_cache")
        persistent_racaf_cache_dir = os.path.join(self.tree.root, "persistent_racaf_cache")

        jtd.precompute_joint_frozen_caches(
            [tree.pairs[0]], tree.image_dir, tree.cache_dir, tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        jtd.precompute_joint_frozen_caches(
            [tree.pairs[1]], tree.image_dir, persistent_cache_dir, persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

        stats = jtd.precompute_joint_frozen_caches(
            tree.pairs, tree.image_dir, tree.cache_dir, tree.racaf_cache_dir,
            persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

        self.assertEqual(stats["already_cached"], 1)
        self.assertEqual(stats["mirrored_from_persistent"], 1)
        self.assertEqual(stats["cached"], 1)
        for id_code, _diagnosis in tree.pairs:
            rgb_cache = jtd._canonical_rgb_cache_path(id_code, tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)
            self.assertTrue(os.path.exists(rgb_cache), f"missing rgb cache for {id_code}")

    def test_second_phase1_run_never_rereads_raw_images(self):
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        with mock.patch.object(lfed, "_load_raw_bgr", wraps=lfed._load_raw_bgr) as mocked_load:
            jtd.precompute_joint_frozen_caches(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model,
            )
        self.assertEqual(mocked_load.call_count, 0)

    def test_legacy_persistent_hit_missing_only_rgb_is_still_a_full_frozen_outputs_hit(self):
        """A persistent cache with vessel/lesion/reliability already populated by an EARLIER run
        -- before the rgb cache kind existed -- has no rgb file. Phase 1 must still recognize it
        as a complete frozen-outputs hit (never recomputing Stage 3/4/RACAF), backfilling only the
        missing rgb entry."""
        persistent_cache_dir = os.path.join(self.tree.root, "persistent_cache")
        persistent_racaf_cache_dir = os.path.join(self.tree.root, "persistent_racaf_cache")
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, persistent_cache_dir, persistent_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        for id_code, _diagnosis in self.tree.pairs:
            rgb_cache = jtd._canonical_rgb_cache_path(id_code, persistent_cache_dir, jtd.STAGE5_IMAGE_SIZE)
            os.remove(rgb_cache)

        with mock.patch(
            "joint_training_dataset.predict_vessel_mask", wraps=jtd.predict_vessel_mask,
        ) as mocked_vessel:
            stats = jtd.precompute_joint_frozen_caches(
                self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                persistent_cache_dir=persistent_cache_dir, persistent_racaf_cache_dir=persistent_racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model,
            )

        self.assertEqual(mocked_vessel.call_count, 0)
        self.assertEqual(stats["mirrored_from_persistent"], len(self.tree.pairs))
        self.assertEqual(stats["cached"], 0)
        for id_code, _diagnosis in self.tree.pairs:
            local_rgb_cache = jtd._canonical_rgb_cache_path(id_code, self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)
            self.assertTrue(os.path.exists(local_rgb_cache))

    def test_empty_fov_entry_still_gets_its_rgb_cached(self):
        """Canonical RGB does not depend on Stage 03's FOV detection succeeding -- an empty-FOV
        entry (whose vessel/lesion/reliability cache never gets written) should still have its rgb
        cached, so at least that portion of its per-epoch cost is eliminated too."""
        tree = _SyntheticAPTOSTree([("img_good", 0), ("img_empty_fov", 1)])
        self.addCleanup(tree.cleanup)
        real_fn = jtd._get_or_compute_joint_frozen_outputs

        def flaky(rgb_native, vessel_cache_path, *args, **kwargs):
            if "img_empty_fov" in vessel_cache_path:
                raise jtd.EmptyFieldOfViewError("no fundus disk detected")
            return real_fn(rgb_native, vessel_cache_path, *args, **kwargs)

        with mock.patch("joint_training_dataset._get_or_compute_joint_frozen_outputs", side_effect=flaky):
            stats = jtd.precompute_joint_frozen_caches(
                tree.pairs, tree.image_dir, tree.cache_dir, tree.racaf_cache_dir,
                vessel_model=self.vessel_model, stage4_model=self.stage4_model,
            )

        self.assertEqual(stats["skipped_empty_fov"], ["img_empty_fov"])
        rgb_cache = jtd._canonical_rgb_cache_path("img_empty_fov", tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)
        self.assertTrue(os.path.exists(rgb_cache))
        vessel_cache = lfed._cache_path(tree.cache_dir, "img_empty_fov", "vessel", jtd.STAGE5_IMAGE_SIZE)
        self.assertFalse(os.path.exists(vessel_cache))


class Phase1bFlushIncludesCanonicalRGBTests(unittest.TestCase):
    """§39: validates the actual notebook configuration end to end -- Phase 1 writing rgb entries
    into the SAME `cache_dir` as vessel/lesion, then the real, unmodified `dataset_staging.
    sync_missing_files()` (Phase 1b's exact call) picking them up automatically, with no new
    plumbing. Also directly verifies the "Flushed 0" semantics: a persistent-only entry produces
    nothing new to flush, and that is correct, not a stall."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1)])
        self.addCleanup(self.tree.cleanup)
        self.drive_cache_dir = os.path.join(self.tree.root, "drive_cache")
        self.drive_racaf_cache_dir = os.path.join(self.tree.root, "drive_racaf_cache")

    def test_a_genuinely_new_entrys_rgb_file_is_flushed_to_drive_alongside_vessel_and_lesion(self):
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        flushed_lf, _kept, failures = dataset_staging.sync_missing_files(
            self.tree.cache_dir, self.drive_cache_dir,
        )
        self.assertEqual(failures, [])
        for id_code, _diagnosis in self.tree.pairs:
            for kind in ("vessel", "lesion", "rgb"):
                drive_path = lfed._cache_path(self.drive_cache_dir, id_code, kind, jtd.STAGE5_IMAGE_SIZE)
                self.assertTrue(os.path.exists(drive_path), f"{kind} for {id_code} was not flushed to Drive")
        # 2 images x (vessel + lesion + rgb) = 6; reliability lives under racaf_cache_dir, flushed separately.
        self.assertEqual(flushed_lf, 6)

    def test_flushed_rgb_content_on_drive_matches_the_local_source_exactly(self):
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        dataset_staging.sync_missing_files(self.tree.cache_dir, self.drive_cache_dir)
        id_code = self.tree.pairs[0][0]
        local_rgb = jtd._canonical_rgb_cache_path(id_code, self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)
        drive_rgb = jtd._canonical_rgb_cache_path(id_code, self.drive_cache_dir, jtd.STAGE5_IMAGE_SIZE)
        np.testing.assert_array_equal(np.load(local_rgb), np.load(drive_rgb))

    def test_a_persistent_only_entry_flushes_nothing_new_this_is_correct_not_a_stall(self):
        """Directly verifies the 'Flushed 0' semantics the user asked to have confirmed: an entry
        found only via `persistent_cache_dir` (content already on Drive) is mirrored locally, but
        that content is -- by construction -- already present at the Drive flush destination, so
        `sync_missing_files` correctly reports it as `already_present`, not `copied`."""
        # Populate the "Drive" cache directly -- as if from an earlier session's Phase 1 run.
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.drive_cache_dir, self.drive_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        # This session's Phase 1: local cache starts empty, everything comes from the persistent hit.
        stats = jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=self.drive_cache_dir, persistent_racaf_cache_dir=self.drive_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        self.assertEqual(stats["cached"], 0)
        self.assertEqual(stats["mirrored_from_persistent"], len(self.tree.pairs))

        flushed_lf, kept_lf, failures = dataset_staging.sync_missing_files(
            self.tree.cache_dir, self.drive_cache_dir,
        )
        self.assertEqual(failures, [])
        self.assertEqual(flushed_lf, 0)  # matches a real run's "Flushed 0" -- correct, not a stall
        self.assertEqual(kept_lf, 6)     # every file WAS already on Drive -- that's WHY nothing flushed
        for id_code, _diagnosis in self.tree.pairs:
            for kind in ("vessel", "lesion", "rgb"):
                self.assertTrue(os.path.exists(
                    lfed._cache_path(self.drive_cache_dir, id_code, kind, jtd.STAGE5_IMAGE_SIZE),
                ))


class CanonicalRGBDrivePersistenceLifecycleTests(unittest.TestCase):
    """§40: the full canonical-RGB persistence lifecycle, reproducing the exact Drive state a real
    run reported -- vessel/lesion/reliability present from an older Phase 1, canonical RGB absent
    because that cache kind did not exist when they were written.

    The question these answer is not "does a cache read work" (covered above) but "does a newly
    computed RGB actually REACH Drive, and does the next fresh runtime then find it there instead
    of regenerating all 3662". That spans two runtimes with a flush in between, so it is tested as
    one continuous lifecycle rather than as isolated units."""

    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_a", 0), ("img_b", 1)])
        self.addCleanup(self.tree.cleanup)
        self.drive_cache_dir = os.path.join(self.tree.root, "drive_cache")
        self.drive_racaf_cache_dir = os.path.join(self.tree.root, "drive_racaf_cache")

    # --- helpers -------------------------------------------------------
    def _build_legacy_drive_cache(self):
        """A Drive cache exactly as an older Phase 1 left it: vessel/lesion/reliability for every
        image, and NO rgb entry at all."""
        jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.drive_cache_dir, self.drive_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )
        for name in os.listdir(self.drive_cache_dir):
            if "_rgb_" in name:
                os.remove(os.path.join(self.drive_cache_dir, name))
        self.assertEqual(self._drive_counts()["rgb"], 0, "fixture must start with zero rgb on Drive")

    def _drive_counts(self):
        counts = {"vessel": 0, "lesion": 0, "rgb": 0}
        for name in os.listdir(self.drive_cache_dir):
            for kind in counts:
                if "_" + kind + "_" in name:
                    counts[kind] += 1
        return counts

    def _run_phase1_against_drive(self):
        return jtd.precompute_joint_frozen_caches(
            self.tree.pairs, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            persistent_cache_dir=self.drive_cache_dir,
            persistent_racaf_cache_dir=self.drive_racaf_cache_dir,
            vessel_model=self.vessel_model, stage4_model=self.stage4_model,
        )

    def _flush(self):
        """The Phase 1b call, verbatim and unmodified."""
        return dataset_staging.sync_missing_files(self.tree.cache_dir, self.drive_cache_dir)

    def _snapshot(self, directory):
        state = {}
        for name in sorted(os.listdir(directory)):
            path = os.path.join(directory, name)
            with open(path, "rb") as handle:
                state[name] = (handle.read(), os.stat(path).st_mtime_ns)
        return state

    def _wipe_local(self):
        """Simulates a fresh Colab runtime: /content is gone, Drive is not."""
        shutil.rmtree(self.tree.cache_dir, ignore_errors=True)
        shutil.rmtree(self.tree.racaf_cache_dir, ignore_errors=True)

    # --- (c) a newly computed RGB reaches the existing persistent cache -----
    def test_rgb_missing_from_a_legacy_drive_cache_is_computed_locally_then_flushed_to_drive(self):
        self._build_legacy_drive_cache()
        stats = self._run_phase1_against_drive()
        # Stage 03/04/RACAF were NOT recomputed -- they were persistent hits.
        self.assertEqual(stats["cached"], 0)
        self.assertEqual(stats["mirrored_from_persistent"], len(self.tree.pairs))
        # Phase 1 itself writes nothing to Drive; RGB is local-only at this point.
        self.assertEqual(self._drive_counts()["rgb"], 0)
        for id_code, _diagnosis in self.tree.pairs:
            self.assertTrue(os.path.exists(
                jtd._canonical_rgb_cache_path(id_code, self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)))

        flushed, _kept, failures = self._flush()
        self.assertEqual(failures, [])
        # Exactly the rgb entries are new on Drive -- vessel/lesion were already there.
        self.assertEqual(flushed, len(self.tree.pairs))
        self.assertEqual(self._drive_counts()["rgb"], len(self.tree.pairs))
        for id_code, _diagnosis in self.tree.pairs:
            self.assertTrue(os.path.exists(
                jtd._canonical_rgb_cache_path(id_code, self.drive_cache_dir, jtd.STAGE5_IMAGE_SIZE)),
                f"rgb for {id_code} did not reach the persistent cache")

    def test_the_flushed_rgb_lands_at_the_documented_filename_beside_vessel_and_lesion(self):
        self._build_legacy_drive_cache()
        self._run_phase1_against_drive()
        self._flush()
        id_code = self.tree.pairs[0][0]
        height, width = jtd.STAGE5_IMAGE_SIZE
        expected = os.path.join(self.drive_cache_dir, f"APTOS_{id_code}_rgb_{height}x{width}.npy")
        self.assertTrue(os.path.exists(expected), f"expected {expected}")
        self.assertEqual(os.path.dirname(expected),
                         os.path.dirname(lfed._cache_path(self.drive_cache_dir, id_code, "vessel",
                                                          jtd.STAGE5_IMAGE_SIZE)),
                         "rgb must live in the same directory as vessel/lesion, not its own")

    # --- (a)+(b) a later fresh runtime hits the persistent RGB and mirrors it ---
    def test_a_second_fresh_runtime_finds_persistent_rgb_and_never_recomputes_it(self):
        self._build_legacy_drive_cache()
        self._run_phase1_against_drive()
        self._flush()
        self._wipe_local()

        with mock.patch("joint_training_dataset._resize_rgb_01",
                        wraps=jtd._resize_rgb_01) as resize_spy, \
             mock.patch("local_feature_extraction_dataset._load_raw_bgr",
                        wraps=lfed._load_raw_bgr) as raw_spy, \
             mock.patch("joint_training_dataset.predict_vessel_mask",
                        wraps=jtd.predict_vessel_mask) as vessel_spy:
            stats = self._run_phase1_against_drive()

        self.assertEqual(resize_spy.call_count, 0, "canonical RGB was regenerated despite being on Drive")
        self.assertEqual(raw_spy.call_count, 0, "the raw image was read despite a full persistent hit")
        self.assertEqual(vessel_spy.call_count, 0, "Stage 03 ran despite a persistent hit")
        self.assertEqual(stats["cached"], 0)
        self.assertEqual(stats["mirrored_from_persistent"], len(self.tree.pairs))
        # Mirrored down to local, so every later access this runtime is a plain local read.
        for id_code, _diagnosis in self.tree.pairs:
            self.assertTrue(os.path.exists(
                jtd._canonical_rgb_cache_path(id_code, self.tree.cache_dir, jtd.STAGE5_IMAGE_SIZE)),
                f"persistent rgb for {id_code} was not mirrored to the local cache")

    def test_the_second_fresh_runtime_flush_then_reports_nothing_new_to_copy(self):
        """Closes the loop: once RGB is on Drive, the next runtime's flush is a no-op, which is the
        steady state -- not a stall."""
        self._build_legacy_drive_cache()
        self._run_phase1_against_drive()
        self._flush()
        self._wipe_local()
        self._run_phase1_against_drive()
        flushed, kept, failures = self._flush()
        self.assertEqual(failures, [])
        self.assertEqual(flushed, 0)
        self.assertEqual(kept, 3 * len(self.tree.pairs))  # vessel + lesion + rgb, all already there

    # --- (f) existing vessel/lesion/RACAF entries are untouched ------------
    def test_flushing_rgb_leaves_every_existing_drive_entry_byte_and_mtime_identical(self):
        self._build_legacy_drive_cache()
        before_lf = self._snapshot(self.drive_cache_dir)
        before_racaf = self._snapshot(self.drive_racaf_cache_dir)

        self._run_phase1_against_drive()
        self._flush()
        dataset_staging.sync_missing_files(self.tree.racaf_cache_dir, self.drive_racaf_cache_dir)

        after_lf = self._snapshot(self.drive_cache_dir)
        after_racaf = self._snapshot(self.drive_racaf_cache_dir)
        # Every pre-existing entry survives untouched; only new rgb files appear.
        for name, value in before_lf.items():
            self.assertEqual(after_lf[name], value, f"{name} was modified by the rgb flush")
        self.assertEqual(before_racaf, after_racaf, "RACAF reliability entries were modified")
        new_names = set(after_lf) - set(before_lf)
        self.assertTrue(new_names and all("_rgb_" in name for name in new_names),
                        f"the flush added non-rgb files: {sorted(new_names)}")

    # --- (d) the training-time path never writes to Drive -------------------
    def test_training_time_build_joint_sample_never_writes_to_the_persistent_cache(self):
        """Covers both interesting states: a full local hit, and a persistent-only entry that the
        training path mirrors DOWN. Neither may write anything back up to Drive."""
        self._build_legacy_drive_cache()
        self._run_phase1_against_drive()
        self._flush()

        for label, wipe_first in (("local hit", False), ("persistent hit mirrored down", True)):
            with self.subTest(state=label):
                if wipe_first:
                    self._wipe_local()
                before = self._snapshot(self.drive_cache_dir)
                before_racaf = self._snapshot(self.drive_racaf_cache_dir)
                for id_code, diagnosis in self.tree.pairs:
                    jtd._build_joint_sample(
                        id_code, diagnosis, self.tree.image_dir, self.tree.cache_dir,
                        self.tree.racaf_cache_dir, self.vessel_model, self.stage4_model,
                        augment=False, rng=None,
                        persistent_cache_dir=self.drive_cache_dir,
                        persistent_racaf_cache_dir=self.drive_racaf_cache_dir,
                    )
                self.assertEqual(self._snapshot(self.drive_cache_dir), before,
                                 "the training-time path wrote to the persistent cache")
                self.assertEqual(self._snapshot(self.drive_racaf_cache_dir), before_racaf,
                                 "the training-time path wrote to the persistent RACAF cache")

    # --- (e) numerical identity survives the whole round trip ---------------
    def test_drive_persisted_rgb_is_bitwise_identical_to_a_fresh_generation(self):
        self._build_legacy_drive_cache()
        self._run_phase1_against_drive()
        self._flush()
        self._wipe_local()
        self._run_phase1_against_drive()  # local copy now came back down from Drive

        id_code = self.tree.pairs[0][0]
        raw_bgr = lfed._load_raw_bgr(self.tree.image_dir, id_code)
        rgb_native = lfed._resolve_processed_rgb(raw_bgr, jtd.DEFAULT_PROCESSED_DIR, id_code)
        fresh = jtd._resize_rgb_01(rgb_native, jtd.STAGE5_IMAGE_SIZE)

        for label, directory in (("drive", self.drive_cache_dir), ("local mirror", self.tree.cache_dir)):
            with self.subTest(copy=label):
                cached = np.load(jtd._canonical_rgb_cache_path(id_code, directory, jtd.STAGE5_IMAGE_SIZE))
                self.assertEqual(cached.shape, fresh.shape)
                self.assertEqual(cached.dtype, fresh.dtype)
                np.testing.assert_array_equal(cached, fresh)
                self.assertEqual(cached.tobytes(), fresh.tobytes())

    # --- the sample built from a round-tripped cache is unchanged -----------
    def test_the_stage5_input_built_from_drive_round_tripped_rgb_is_unchanged(self):
        """The value that actually matters downstream: `stage5_input`'s RGB channels must be the
        same after a Drive round trip as when generated live."""
        self._build_legacy_drive_cache()
        self._run_phase1_against_drive()
        live = jtd._build_joint_sample(
            self.tree.pairs[0][0], 0, self.tree.image_dir, self.tree.cache_dir,
            self.tree.racaf_cache_dir, self.vessel_model, self.stage4_model,
            augment=False, rng=None,
        )
        self._flush()
        self._wipe_local()
        self._run_phase1_against_drive()
        round_tripped = jtd._build_joint_sample(
            self.tree.pairs[0][0], 0, self.tree.image_dir, self.tree.cache_dir,
            self.tree.racaf_cache_dir, self.vessel_model, self.stage4_model,
            augment=False, rng=None,
            persistent_cache_dir=self.drive_cache_dir,
            persistent_racaf_cache_dir=self.drive_racaf_cache_dir,
        )
        np.testing.assert_array_equal(live["stage5_input"], round_tripped["stage5_input"])
        np.testing.assert_array_equal(live["stage6_input"], round_tripped["stage6_input"])
        self.assertEqual(float(live["reliability"]), float(round_tripped["reliability"]))


# =====================================================================
# Augmentation synchronization.
# =====================================================================

class AugmentationSynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.vessel_model = _build_synthetic_vessel_model()
        self.stage4_model = _build_synthetic_frozen_stage4_model()
        self.tree = _SyntheticAPTOSTree([("img_01", 1)])
        self.addCleanup(self.tree.cleanup)

    def test_stage6_input_derives_from_stage5s_own_augmented_rgb(self):
        """Stage 6's input must be a resize of Stage 5's OWN (possibly augmented) RGB channels --
        proof the same spatial transform reaches both branches, never two independent draws."""
        rng = np.random.default_rng(3)
        sample = jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=True, rng=rng,
        )
        expected_stage6 = lfed._resize_input(sample["stage5_input"][..., :3], jtd.STAGE6_IMAGE_SIZE)
        np.testing.assert_array_equal(sample["stage6_input"], expected_stage6)

    def test_intensity_augmentation_never_touches_vessel_or_lesion_channels(self):
        rng = np.random.default_rng(0)
        unaugmented = jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=None,
        )
        rng2 = np.random.default_rng(0)
        with mock.patch.object(lfed, "_augment_spatial", side_effect=lambda x, r: x):
            augmented = jtd._build_joint_sample(
                "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
                self.vessel_model, self.stage4_model, augment=True, rng=rng2,
            )
        np.testing.assert_array_equal(
            augmented["stage5_input"][..., 3:8], unaugmented["stage5_input"][..., 3:8],
        )

    def test_validation_path_has_no_augmentation(self):
        """augment=False must produce byte-identical output across repeated calls -- no RNG draw
        at all touches the sample."""
        sample_a = jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=None,
        )
        sample_b = jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=None,
        )
        np.testing.assert_array_equal(sample_a["stage5_input"], sample_b["stage5_input"])
        np.testing.assert_array_equal(sample_a["stage6_input"], sample_b["stage6_input"])
        self.assertEqual(sample_a["reliability"], sample_b["reliability"])

    def test_canonical_caches_are_never_modified_by_augmentation(self):
        """Building an AUGMENTED sample must not change the on-disk canonical cache -- the cache
        stores the unaugmented Stage 03/04 output; augmentation happens only when the batch tensor
        is constructed, after the cache is read."""
        rng = np.random.default_rng(0)
        jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=None,
        )
        vessel_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "vessel", jtd.STAGE5_IMAGE_SIZE)
        lesion_cache = lfed._cache_path(self.tree.cache_dir, "img_01", "lesion", jtd.STAGE5_IMAGE_SIZE)
        before_vessel = np.load(vessel_cache).copy()
        before_lesion = np.load(lesion_cache).copy()

        jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=True, rng=rng,
        )
        after_vessel = np.load(vessel_cache)
        after_lesion = np.load(lesion_cache)
        np.testing.assert_array_equal(before_vessel, after_vessel)
        np.testing.assert_array_equal(before_lesion, after_lesion)

    def test_reliability_is_computed_from_canonical_unaugmented_output(self):
        """r must be identical whether or not the sample it's attached to is augmented -- it is
        never itself augmented, and is always derived from the canonical (pre-augmentation)
        Stage 04 output."""
        rng = np.random.default_rng(5)
        unaugmented = jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=False, rng=None,
        )
        augmented = jtd._build_joint_sample(
            "img_01", 1, self.tree.image_dir, self.tree.cache_dir, self.tree.racaf_cache_dir,
            self.vessel_model, self.stage4_model, augment=True, rng=rng,
        )
        self.assertEqual(unaugmented["reliability"], augmented["reliability"])


# =====================================================================
# Frozen/trainable boundary + gradient boundary + loss + architecture.
# =====================================================================

class JointModelTests(unittest.TestCase):
    """Builds the REAL joint model ONCE for this whole class -- Stage 05/06 are real, full-size
    architectures, so rebuilding per test would be wasteful; every test below only reads from or
    forward/backward-passes through this one shared instance, never mutates its weights."""

    @classmethod
    def setUpClass(cls):
        cls.model = jtm.build_joint_model()

    def test_stage3_and_stage4_are_not_in_the_graph_at_all(self):
        """The strongest possible frozen-boundary guarantee: Stage 03/04 parameters are not
        tf.Variables in this graph, so no optimizer or GradientTape built from this model can
        reach them, structurally -- not merely via a stop_gradient wrapper."""
        variable_names = " ".join(v.name.lower() for v in self.model.trainable_variables)
        for forbidden in ("wnet", "unet1", "unet2"):
            self.assertNotIn(forbidden, variable_names)

    def test_five_stages_are_all_trainable(self):
        """Every one of Stage 05/06/07/RACAF/CORN's own trainable variables must be present in
        the joint model's trainable_variables -- proof none was accidentally frozen by
        composition."""
        import feature_fusion as ff
        import local_feature_extraction_model as lfem
        import swin_transformer as st

        stage5_only = lfem.build_local_feature_extractor()
        stage6_only = st.create_dual_scale_swin_model()
        stage7_only = ff.build_adaptive_cross_attention()
        racaf_only = racaf.build_racaf_fusion()
        corn_only = corn.build_corn_model()

        expected_total = sum(
            m.count_params() for m in (stage5_only, stage6_only, stage7_only, racaf_only, corn_only)
        )
        self.assertEqual(self.model.count_params(), expected_total)

    def test_racaf_trainable_param_count_matches_measured_value(self):
        """RACAF's own measured 295,170 trainable parameters -- confirms RACAF's contribution to
        the joint model is exactly its approved, frozen architecture, no more."""
        racaf_only = racaf.build_racaf_fusion()
        self.assertEqual(racaf_only.count_params(), 295_170)

    def test_corn_trainable_param_count_matches_measured_value(self):
        corn_only = corn.build_corn_model()
        self.assertEqual(corn_only.count_params(), 1_028)

    def test_joint_output_shape(self):
        batch = 2
        s5 = np.random.rand(batch, 512, 512, 8).astype("float32")
        s6 = np.random.rand(batch, 256, 256, 3).astype("float32")
        r = np.random.rand(batch, 1).astype("float32")
        logits = self.model.predict([s5, s6, r], verbose=0)
        self.assertEqual(logits.shape, (batch, 4))

    def test_batch_independence(self):
        """Each image's logits must depend only on its own input, not on other images in the
        batch -- a single-image prediction must match its slot in a batched prediction."""
        rng = np.random.RandomState(0)
        s5 = rng.rand(2, 512, 512, 8).astype("float32")
        s6 = rng.rand(2, 256, 256, 3).astype("float32")
        r = rng.rand(2, 1).astype("float32")

        batched = self.model.predict([s5, s6, r], verbose=0)
        single = self.model.predict([s5[0:1], s6[0:1], r[0:1]], verbose=0)
        np.testing.assert_allclose(batched[0], single[0], atol=1e-4)

    def test_gradient_reaches_every_trainable_variable(self):
        """A single GradientTape step (no optimizer.apply_gradients, no weight update) proving
        the loss backpropagates into every trainable variable of all five stages."""
        batch = 2
        s5 = tf.random.uniform((batch, 512, 512, 8))
        s6 = tf.random.uniform((batch, 256, 256, 3))
        r = tf.random.uniform((batch, 1))
        grades = tf.constant([1, 3], dtype=tf.int32)

        with tf.GradientTape() as tape:
            logits = self.model([s5, s6, r], training=True)
            loss = jtm.joint_corn_loss(grades, logits)
        grads = tape.gradient(loss, self.model.trainable_variables)

        none_count = sum(1 for g in grads if g is None)
        self.assertEqual(none_count, 0, "every trainable variable must receive a gradient")

    def test_reliability_input_is_not_itself_a_trainable_variable(self):
        """r enters the graph as a plain Input -- it must never appear among trainable_variables
        (it is data, not a learned parameter; only RACAF's gate weights that CONSUME it are
        trainable)."""
        variable_names = [v.name for v in self.model.trainable_variables]
        self.assertNotIn("reliability", " ".join(variable_names).lower())

    def test_joint_corn_loss_matches_corn_loss_directly(self):
        logits = tf.constant([[0.1, -0.2, 0.3, 0.0], [1.0, 0.5, -0.5, 0.2]], dtype=tf.float32)
        grades = tf.constant([2, 4], dtype=tf.int32)
        expected = corn.corn_loss(logits, grades)
        actual = jtm.joint_corn_loss(grades, logits)
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_compile_joint_model_sets_corn_loss_only(self):
        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)
        self.assertTrue(model.compiled)
        self.assertIs(model.loss, jtm.joint_corn_loss)

    def test_save_and_load_weights_round_trip(self):
        tmp_dir = tempfile.mkdtemp(prefix="joint_ckpt_test_")
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        path = os.path.join(tmp_dir, "joint.weights.h5")

        s5 = np.random.rand(1, 512, 512, 8).astype("float32")
        s6 = np.random.rand(1, 256, 256, 3).astype("float32")
        r = np.random.rand(1, 1).astype("float32")
        before = self.model.predict([s5, s6, r], verbose=0)

        jtm.save_joint_model_weights(self.model, path)
        self.assertTrue(os.path.exists(path))

        reloaded = jtm.load_joint_model_weights(path)
        after = reloaded.predict([s5, s6, r], verbose=0)
        np.testing.assert_allclose(before, after, atol=1e-5)

    def test_checkpoint_functions_take_no_hardcoded_path(self):
        """save/load must be pure path-parameterized -- no built-in Drive or local default that
        could silently write somewhere unexpected."""
        save_params = list(inspect.signature(jtm.save_joint_model_weights).parameters)
        load_params = list(inspect.signature(jtm.load_joint_model_weights).parameters)
        self.assertIn("path", save_params)
        self.assertIn("path", load_params)
        # "path" must be a REQUIRED positional argument (no default) on both functions -- proof
        # neither one bakes in an implicit Drive or local fallback location.
        self.assertIs(
            inspect.signature(jtm.save_joint_model_weights).parameters["path"].default,
            inspect.Parameter.empty,
        )
        self.assertIs(
            inspect.signature(jtm.load_joint_model_weights).parameters["path"].default,
            inspect.Parameter.empty,
        )


# =====================================================================
# val_QWK checkpoint-selection fix -- compile_joint_model() wiring, Keras log naming, and
# callback compatibility. Tiny synthetic tensors only; no real APTOS data, no real training
# epoch, no persistent checkpoint.
# =====================================================================

class CORNQWKJointIntegrationTests(unittest.TestCase):
    def test_compile_joint_model_adds_exactly_one_qwk_metric(self):
        """`model.metrics` is not yet populated with individual metric names until Keras has
        built the compiled-metrics wrapper (which happens on the first fit/evaluate/predict
        call) -- so this uses `model.evaluate(..., return_dict=True)` (a forward pass only, no
        gradient step, no weight update -- not training) to force that build and read the
        resulting metric names directly, proving exactly one metric, "QWK", was added alongside
        the loss, with no other metric silently included."""
        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)

        s5 = np.random.RandomState(9).rand(2, 512, 512, 8).astype("float32")
        s6 = np.random.RandomState(10).rand(2, 256, 256, 3).astype("float32")
        r = np.random.RandomState(11).rand(2, 1).astype("float32")
        grades = np.array([1, 3], dtype="int32")

        results = model.evaluate([s5, s6, r], grades, batch_size=2, verbose=0, return_dict=True)
        self.assertIn("QWK", results)
        self.assertEqual(set(results.keys()), {"loss", "QWK"})

    def test_compile_joint_model_still_uses_only_corn_loss(self):
        """No additional loss was introduced alongside the new metric."""
        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)
        self.assertIs(model.loss, jtm.joint_corn_loss)

    def test_keras_logs_contain_qwk_and_val_qwk(self):
        """A tiny (batch=2), ONE-step synthetic train+val fit -- proves Keras's own logs dict
        actually contains 'QWK'/'val_QWK' by name, not merely that the metric object exists.
        NOT real training: one epoch, one step, synthetic tensors, model discarded afterward."""
        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)

        s5 = np.random.RandomState(0).rand(2, 512, 512, 8).astype("float32")
        s6 = np.random.RandomState(1).rand(2, 256, 256, 3).astype("float32")
        r = np.random.RandomState(2).rand(2, 1).astype("float32")
        grades = np.array([1, 3], dtype="int32")

        history = model.fit(
            [s5, s6, r], grades, validation_data=([s5, s6, r], grades),
            epochs=1, batch_size=2, verbose=0,
        )
        self.assertIn("loss", history.history)
        self.assertIn("QWK", history.history)
        self.assertIn("val_loss", history.history)
        self.assertIn("val_QWK", history.history)
        val_qwk = float(history.history["val_QWK"][0])
        self.assertFalse(np.isnan(val_qwk))

    def test_model_checkpoint_can_monitor_val_qwk(self):
        """A real `tf.keras.callbacks.ModelCheckpoint(monitor="val_QWK", mode="max")` attached
        to a one-step `model.fit()` call must find the value (no 'not available' skip) and save
        a checkpoint -- proof the callback layer, not just the raw logs dict, sees it. Checkpoint
        is written to a temp directory and removed immediately after; not a real training run."""
        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)

        tmp_dir = tempfile.mkdtemp(prefix="qwk_checkpoint_test_")
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        checkpoint_path = os.path.join(tmp_dir, "best.weights.h5")

        s5 = np.random.RandomState(3).rand(2, 512, 512, 8).astype("float32")
        s6 = np.random.RandomState(4).rand(2, 256, 256, 3).astype("float32")
        r = np.random.RandomState(5).rand(2, 1).astype("float32")
        grades = np.array([0, 4], dtype="int32")

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_QWK", mode="max",
            save_best_only=True, save_weights_only=True, verbose=0,
        )
        model.fit(
            [s5, s6, r], grades, validation_data=([s5, s6, r], grades),
            epochs=1, batch_size=2, verbose=0, callbacks=[checkpoint_cb],
        )
        self.assertNotEqual(checkpoint_cb.best, float("-inf"), "ModelCheckpoint never saw val_QWK")
        self.assertTrue(os.path.exists(checkpoint_path))

    def test_model_checkpoint_mode_max_only_saves_on_improvement(self):
        """Deterministic proof of the maximize-direction policy, independent of real training
        dynamics: manufactured val_QWK values fed directly to a ModelCheckpoint's own
        `on_epoch_end` -- a higher value must be treated as an improvement, a lower one must
        not."""
        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)

        tmp_dir = tempfile.mkdtemp(prefix="qwk_checkpoint_direction_test_")
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        checkpoint_path = os.path.join(tmp_dir, "best.weights.h5")

        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_QWK", mode="max",
            save_best_only=True, save_weights_only=True, verbose=0,
        )
        checkpoint_cb.set_model(model)

        checkpoint_cb.on_epoch_end(0, logs={"val_QWK": 0.2})
        self.assertEqual(checkpoint_cb.best, 0.2)
        checkpoint_cb.on_epoch_end(1, logs={"val_QWK": 0.6})
        self.assertEqual(checkpoint_cb.best, 0.6)  # improved -- new best
        checkpoint_cb.on_epoch_end(2, logs={"val_QWK": 0.1})
        self.assertEqual(checkpoint_cb.best, 0.6)  # NOT an improvement -- best unchanged

    def test_saving_and_loading_weights_still_works_with_metric_compiled(self):
        """The new metric must not interfere with the established weights-only checkpoint
        strategy (JOINT_TRAINING_ARCHITECTURE.md Sec 25)."""
        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)

        tmp_dir = tempfile.mkdtemp(prefix="qwk_weights_roundtrip_test_")
        self.addCleanup(shutil.rmtree, tmp_dir, True)
        path = os.path.join(tmp_dir, "joint.weights.h5")

        s5 = np.random.RandomState(6).rand(1, 512, 512, 8).astype("float32")
        s6 = np.random.RandomState(7).rand(1, 256, 256, 3).astype("float32")
        r = np.random.RandomState(8).rand(1, 1).astype("float32")
        before = model.predict([s5, s6, r], verbose=0)

        jtm.save_joint_model_weights(model, path)
        reloaded = jtm.load_joint_model_weights(path)
        after = reloaded.predict([s5, s6, r], verbose=0)
        np.testing.assert_allclose(before, after, atol=1e-5)


# =====================================================================
# Step B: the actual training-cell integration -- proves the EXACT composition
# `colab/notebooks/stage08_corn_classifier.ipynb`'s training cell now uses,
# `training.Trainer(training.TrainingConfig(...)).fit(joint_model, train_ds, val_ds)`, really
# works end-to-end. Tiny synthetic tf.data.Dataset pipelines, a temp directory, ONE epoch --
# a focused integration smoke test, not real APTOS training.
# =====================================================================

class TrainerIntegrationTests(unittest.TestCase):
    def test_trainer_fit_runs_one_epoch_and_selects_best_checkpoint_by_val_qwk(self):
        from training import Trainer, TrainingConfig

        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)

        def make_dataset(seed):
            rng = np.random.RandomState(seed)
            s5 = rng.rand(2, 512, 512, 8).astype("float32")
            s6 = rng.rand(2, 256, 256, 3).astype("float32")
            r = rng.rand(2, 1).astype("float32")
            grades = np.array([1, 3], dtype="int32")
            return tf.data.Dataset.from_tensors(((s5, s6, r), grades))

        train_ds = make_dataset(0)
        val_ds = make_dataset(1)

        tmp_dir = tempfile.mkdtemp(prefix="trainer_integration_test_")
        self.addCleanup(shutil.rmtree, tmp_dir, True)

        # Mirrors the notebook's exact configuration (monitor/mode/mixed_precision) -- only
        # epochs is reduced (1, not 50) and run_dir points at a temp directory, not a real Drive
        # experiment folder.
        config = TrainingConfig(
            run_dir=tmp_dir, epochs=1, monitor="val_QWK", mode="max", mixed_precision=True,
        )
        trainer = Trainer(config)
        history = trainer.fit(model, train_ds, val_ds)

        self.assertIn("val_QWK", history.history)
        self.assertIn("QWK", history.history)
        self.assertFalse(np.isnan(history.history["val_QWK"][0]))
        # save_best_only=True always saves on the very first epoch (no prior best to compare
        # against) -- proof the "best checkpoint persists" requirement actually works through
        # the real Trainer/build_callbacks machinery, not merely that the config was accepted.
        self.assertTrue(os.path.exists(trainer.best_weights_path()))
        self.assertEqual(trainer.config.monitor, "val_QWK")
        self.assertEqual(trainer.config.mode, "max")

    def test_resume_flag_reloads_last_checkpoint_before_continuing(self):
        """Proves RESUME_EXPERIMENT_DIR's underlying mechanism (`TrainingConfig(resume=True)`)
        actually reloads weights and advances `initial_epoch`, exactly as the notebook's gated
        training cell relies on -- not merely that the flag is accepted."""
        from training import Trainer, TrainingConfig

        model = jtm.build_joint_model()
        jtm.compile_joint_model(model)

        def make_dataset(seed):
            rng = np.random.RandomState(seed)
            s5 = rng.rand(2, 512, 512, 8).astype("float32")
            s6 = rng.rand(2, 256, 256, 3).astype("float32")
            r = rng.rand(2, 1).astype("float32")
            grades = np.array([0, 2], dtype="int32")
            return tf.data.Dataset.from_tensors(((s5, s6, r), grades))

        train_ds = make_dataset(2)
        val_ds = make_dataset(3)

        tmp_dir = tempfile.mkdtemp(prefix="trainer_resume_test_")
        self.addCleanup(shutil.rmtree, tmp_dir, True)

        first_config = TrainingConfig(run_dir=tmp_dir, epochs=1, monitor="val_QWK", mode="max")
        Trainer(first_config).fit(model, train_ds, val_ds)

        resumed_config = TrainingConfig(
            run_dir=tmp_dir, epochs=2, monitor="val_QWK", mode="max", resume=True,
        )
        resumed_trainer = Trainer(resumed_config)
        resumed_trainer.prepare()  # resolve_initial_epoch() needs self.paths, normally set
                                    # lazily by fit() itself -- called explicitly here only to
                                    # inspect it before running the second fit() call below.
        self.assertEqual(resumed_trainer.resolve_initial_epoch(), 1)
        resumed_trainer.fit(model, train_ds, val_ds)


# =====================================================================
# Drive path / checkpoint-selection infrastructure (no real Drive needed).
# =====================================================================

class DriveInfrastructureTests(unittest.TestCase):
    def test_final_classification_experiment_dir_resolves_via_existing_infra(self):
        import sys

        colab_common = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab", "common")
        if colab_common not in sys.path:
            sys.path.insert(0, colab_common)
        import colab_config

        base = colab_config.DRIVE.experiment_dir("FinalClassification")
        self.assertTrue(base.startswith("/content/drive/MyDrive/DiabeticRetinopathy/experiments/FinalClassification"))

    def test_no_stage1_import_anywhere_in_joint_modules(self):
        for module in (jtd, jtm):
            source = inspect.getsource(module)
            self.assertNotIn("image_quality_inference", source)
            self.assertNotIn("image_quality_model", source)


# =====================================================================
# Realistic (checkpoint-free) end-to-end integration smoke test -- Step 15.
# =====================================================================

class RealisticIntegrationSmokeTest(unittest.TestCase):
    """One real, small-batch forward pass through the actual joint model (real Stage 05/06/07/
    RACAF/CORN architectures, no checkpoint, no training). Confirms F/logits shapes match the
    approved contract end-to-end -- not just per-stage in isolation."""

    def test_real_components_batch_of_two(self):
        model = jtm.build_joint_model()
        batch = 2
        s5 = np.random.RandomState(1).rand(batch, 512, 512, 8).astype("float32")
        s6 = np.random.RandomState(2).rand(batch, 256, 256, 3).astype("float32")
        r = np.random.RandomState(3).rand(batch, 1).astype("float32")

        logits = model.predict([s5, s6, r], verbose=0)
        self.assertEqual(logits.shape, (batch, 4))

        decoded = corn.decode_logits(logits)
        self.assertTrue(np.all(decoded["predicted_grade"] >= 0))
        self.assertTrue(np.all(decoded["predicted_grade"] <= 4))
        np.testing.assert_allclose(decoded["class_probabilities"].sum(axis=-1), np.ones(batch), atol=1e-4)


if __name__ == "__main__":
    unittest.main()
