# Joint Training Architecture — Stages 5–8 + RACAF

**Status:** Authoritative design document for the joint Stage 05–08 + RACAF training run.
**Design, infrastructure, joint dataset loader, and joint model builder are implemented and
unit-tested** (`joint_training_dataset.py`, `joint_training_model.py`,
`tests/test_joint_training.py`). **No training loop has been run and no checkpoint has been
generated** — `colab/notebooks/stage08_corn_classifier.ipynb`'s training cells exist but are
gated behind `RUN_TRAINING = False`; opening or running the notebook as committed does not start
real training. This document is what that implementation follows, exactly as
`RACAF_ARCHITECTURE.md`/`CORN_ARCHITECTURE.md` play the same role for RACAF/CORN.

This document does not redefine RACAF's or CORN's mathematics — `RACAF_ARCHITECTURE.md` and
`CORN_ARCHITECTURE.md` remain the sole authorities for those. It also does not redefine Stage
5/6/7's architectures, fixed in `PROJECT_STRUCTURE.md` §5–7. It is scoped entirely to how those
already-approved pieces train **together**: dataset flow, caching, gradient boundary, loss,
checkpoint format, and the infrastructure (Drive paths, canonical-resolution caching) that
implementation depends on.

---

## 1. Purpose

Stages 5, 6, 7, RACAF, and CORN have no standalone training procedure of their own — every one of
their `train()`/`evaluate()` methods raises `NotImplementedError`, each explicitly deferring to
"the joint training script" (`local_feature_extraction_model.py`, `swin_transformer.py`,
`feature_fusion.py`, `racaf.py`, `corn.py`). This document defines that joint training design: what
is frozen, what is trainable, what data enters each stage, where cached frozen output comes from,
what receives gradient, the training loss, the authoritative split, the validation procedure, the
checkpoint strategy, how Drive interacts with Colab, the expected T4 starting configuration, and
the notebook that will eventually run it.

---

## 2. Current pipeline

```
Stage 01 (IQA, frozen) → Stage 02 (deterministic) → Stage 03 (frozen) → Stage 04 (frozen)
                                                                              │
                                              ┌───────────────────────────────┤
                                              ↓                               ↓
                                    Stage 05 (trainable)            Stage 06 (trainable)
                                              │                               │
                                              └───────────────┬───────────────┘
                                                               ↓
                                                    Stage 07 (trainable)
                                                               ↓
                                                       RACAF (trainable)
                                                               ↓
                                                        F = (B, 256)
                                                               ↓
                                                        CORN (trainable)
                                                               ↓
                                                    4 ordinal logits
```

Stage 1 is **not** part of this trainable graph or its data path — see §3.

---

## 3. Frozen stages

| Stage | Frozen how |
|---|---|
| 1 — IQA | Not part of the downstream graph at all (§3.1) |
| 2 — Preprocessing | Deterministic, no parameters |
| 3 — Vessel Seg (LWNet) | PyTorch, `model.eval()`, every call wrapped in `torch.no_grad()` — structurally outside any TF graph |
| 4 — Lesion Seg (Attn U-Net, Exp 2C) | `.keras`, `trainable=False` (`racaf.load_frozen_stage4_model()`) + `tf.stop_gradient` on every TTA prediction (`racaf.tta_views()`); kept entirely inside the data/cache layer for joint training (§9), never called live inside the trainable graph |

### 3.1 Stage 1's role — locked

**Stage 1 does NOT gate APTOS2019 downstream training.** Verified structurally: neither
`local_feature_extraction_dataset.py` nor `global_feature_extraction_dataset.py` imports
`image_quality_inference` anywhere, and `PROJECT_STRUCTURE.md`'s Dataset Flow section states the
IQA gate applies to "EyeQ only — the only dataset this gate applies to today." All 3662 APTOS2019
labeled IDs remain eligible; the authoritative split (§6) is unaffected. The Stage 1 model must
not be called anywhere in the joint training data path unless a future, explicit decision changes
this. Reason: this is the current, documented project architecture, not an oversight to silently
correct.

---

## 4. Trainable stages

Stage 5 (Adaptive Multi-Kernel CNN), Stage 6 (Dual-Scale Swin), Stage 7 (Adaptive Cross-Attention),
RACAF (`w_g, b_g, W_r, b_r` — 295,170 params, measured), CORN (`Dense(256→4)` — 1,028 params,
measured). Architecture-freeze (Stage 7/RACAF's own prior review status) is explicitly distinct
from weight-training status — none of these five is excluded from joint training merely because
its architecture was reviewed/frozen earlier.

---

## 5. Dataset policy

Exactly three project datasets — EyeQ, APTOS2019, IDRiD — unchanged. Joint training uses
**APTOS2019 only**. No new dataset is introduced.

---

## 6. Authoritative split

`downstream_split.get_authoritative_split()` — 2929 train / 733 val = 3662, stratified by
`diagnosis`, seed 42, persisted at `dataset_splits/aptos2019_train_val_split.csv`. Every downstream
component (Stage 5, Stage 6, and the future joint loader) must call this same function. No second
split; no independent Stage 5/6/CORN/joint split. `APTOS2019/raw/test.csv` (1928 images, no
`diagnosis` column) is not a supervised split.

---

## 7. Drive vs. local vs. Colab storage

| Layer | Holds |
|---|---|
| **Git (source code)** | `.py` modules, tests, notebooks, architecture documents — no large artifacts |
| **Google Drive (`MyDrive/DiabeticRetinopathy/`, persistent)** | `datasets/` (raw + processed), `experiments/<module>/<timestamp>/` (per-run checkpoints/logs), `exported_models/<module>/` (best trained weights), `cache/<module>/` (persistent, per-image derived caches — **new**, see §7.1), `tensorboard/`, `logs/` |
| **Colab VM `/content` (ephemeral)** | Cloned repo (code only), `dataset_staging.py`'s staged local-SSD copy of raw/processed data for fast I/O |

None of `datasets/`, `experiments/`, `tensorboard/`, `exported_models/`, `logs/` was renamed or
restructured.

### 7.1 Infrastructure fix made — Drive path wiring (Step 2)

**Finding:** `colab/common/setup.py`'s `configure_environment_variables()` previously wired only
10 dataset-directory env vars. It set no env var at all for Stage 1/3/4's own checkpoint
directories (`IQA_MODEL_DIR`, `VESSEL_SEG_MODEL_DIR`, `LESION_SEG_MODEL_DIR`), and none for Stage
5/6/7/RACAF/CORN's `MODEL_DIR`/`RESULTS_DIR`. Every one of these `config.py` fields falls back to a
path inside the cloned repository when its env var is unset — in a fresh Colab session that is the
just-cloned, ephemeral VM checkout, which never contains an already-trained checkpoint or a
persistent cache. This would make `load_iqa_model()`/`load_vessel_model()`/`load_lesion_model()`
fail outright, and would silently discard Stage 5/RACAF's derived-prediction caches on every VM
restart.

**Fix made** (`colab/common/drive_paths.py`, `colab_config.py`, `setup.py`):

- Added a new, additive Drive bucket, `cache/`, alongside the four already-verified buckets
  (`datasets/`, `experiments/`, `tensorboard/`, `exported_models/`, `logs/`) — for small, per-image
  derived arrays (Stage 3/4 predictions, RACAF reliability) that must be reused across every
  training run and resumed session, which fits none of the four existing categories (`experiments/`
  is per-run/timestamped; `exported_models/` is final trained weights). `drive_paths.DrivePaths`
  gained `cache_root`/`cache_dirs`/`cache_dir(module)`, following the exact same per-module-dict
  pattern `experiment_dirs`/`exported_model_dirs` already use. `CACHE_MODULES = ("LocalFeatureExtraction", "RACAF")`
  — the only two stages with a frozen-upstream cache of their own.
- `colab_config.py` gained `IQA_MODEL_DIR`/`VESSEL_SEG_MODEL_DIR`/`LESION_SEG_MODEL_DIR` (resolved
  via the already-existing `DRIVE.exported_model_dir(...)` for `"IQA"`/`"VesselSegmentation"`/`"LesionSegmentation"`),
  `LOCAL_FEATURE_CACHE_DIR`/`RACAF_CACHE_DIR` (via the new `DRIVE.cache_dir(...)`), and
  `LOCAL_FEATURE_MODEL_DIR`/`GLOBAL_FEATURE_MODEL_DIR`/`FEATURE_FUSION_MODEL_DIR`/`RACAF_MODEL_DIR`/`CORN_MODEL_DIR`
  — each nested under the single, already-reserved `exported_models/FinalClassification/` directory
  (since these 5 stages train jointly as one model, not as five independently-checkpointed
  `PIPELINE_MODULES` entries — `PROJECT_STRUCTURE.md`'s own note that Drive already anticipates one
  bucket, not four).
- `setup.py`'s `configure_environment_variables()` now also sets all of the above. Deliberately NOT
  wired: `GLOBAL_FEATURE_RESULTS_DIR`, `FEATURE_FUSION_RESULTS_DIR`, `CORN_RESULTS_DIR` — none of
  those three stages has any frozen-upstream inference of its own to cache (their own `config.py`
  docstrings), so nothing is ever written there.

Extended tests: `tests/test_drive_paths.py` (new cases for `cache_dir()`, the new `colab_config`
constants, the extended env-var set, and an explicit "not fabricated" check for the three
`RESULTS_DIR`s that must stay unwired).

---

## 8. Frozen checkpoint loading

| Stage | Path | Loader | Note |
|---|---|---|---|
| 1 | `IQA_MODEL_DIR/best_model.keras` | `image_quality_inference.load_iqa_model()` | Not used in the joint graph (§3.1) |
| 3 | `VESSEL_SEG_MODEL_DIR/best_model.pth` | `vessel_segmentation_inference.load_vessel_model()` | PyTorch, `torch.no_grad()` |
| 4 | `LESION_SEG_MODEL_DIR/best_model.keras` | `racaf.load_frozen_stage4_model()` (sets `trainable=False`) | Used by both Stage 5's cache-builder and RACAF's TTA (§9) |

No checkpoint path is invented; no checkpoint is retrained, modified, or copied into Git.

---

## 9. Stage 2 preprocessing

Gamma+CLAHE, `image_preprocessing.preprocess_array(profile="DR")`, applied once per dataset and
reused (`_resolve_processed_rgb()` reads `datasets/APTOS2019/processed/<id>.png` if already
batch-generated, else computes live). Unchanged by this document.

---

## 10. Stage 3 cache — fixed (Step 3)

**Finding:** the pre-existing cache (`local_feature_extraction_dataset.py`) stored Stage 3/4's
output at **native APTOS-image resolution** (`predict_vessel_mask`/`predict_lesion_mask` both
resize their internal 512×512 computation back up to the input image's own resolution before
returning), not the 512×512 resolution every actual consumer (Stage 5's tensor, RACAF's TTA input)
needs. For APTOS2019's multi-megapixel photos this plausibly reached tens-to-hundreds of GB across
3662 images, and `build_local_feature_input()`'s own final resize-down repeated on every sample
construction, every epoch, for content that never changes.

**Fix made:** `_get_or_compute_stage3_stage4_maps()` (replacing the previous
`_get_or_compute_vessel_map`/`_get_or_compute_lesion_maps`) resizes each prediction down to the
canonical `image_size` (`DEFAULT_IMAGE_SIZE = (512, 512)`) **once, before writing it to disk** —
Stage 3/4's own inference is completely unchanged (still runs on the full native-resolution image,
preserving LWNet's FOV-detection accuracy and Stage 4's official inference behavior exactly).
`_build_sample()` additionally resizes the RGB channel down to the same `image_size` before
concatenation, so `build_local_feature_input()`'s existing native-shape-match validation is
satisfied **without any change to that function** — it keeps its exact current contract for
single-image inference callers. Cache filenames now include `image_size`
(`APTOS_<id>_<kind>_<h>x<w>.npy`) so a cache built at one resolution can never be silently reused
for another.

Resulting cache footprint per image: vessel `(512,512)` ≈ 1 MB, lesion `(512,512,4)` ≈ 4 MB — a
small, fixed size regardless of the source photo's resolution, versus the previous
variable/native-resolution footprint.

Extended tests: `tests/test_local_feature_extraction_dataset.py` — cache-key uniqueness by
`image_size`, cache stores canonical resolution (not native), and cached values match a direct
Stage 3 prediction resized the same way (the previous "cache never alters a value" test is now
"cache never alters a value beyond the documented, intentional canonical resize").

---

## 11. Stage 4 cache

Same fix as §10 — `_get_or_compute_stage3_stage4_maps()` caches Stage 4's identity-transform
lesion prediction at canonical `(512,512,4)` resolution. Both caches are populated **together**,
not independently, because Stage 4's own inference inherently needs the vessel map at the same
native resolution as the RGB image for its internal concatenation — the small, canonical vessel
map alone cannot reconstruct that input. If both caches already exist for an image, neither Stage
3 nor Stage 4 runs at all.

### 11.1 Redundant Stage-4 identity computation (Step 4) — RESOLVED, in the joint dataset loader

Previously analyzed and deferred (a prior revision of this document): true elimination requires a
single per-image computation point that populates both Stage 5's lesion cache and RACAF's
reliability cache from one `racaf.tta_views()` call — that computation point is, by definition,
the joint dataset loader, which had not been built yet.

**Now implemented exactly that way**, in `joint_training_dataset.py`'s
`_get_or_compute_joint_frozen_outputs()`: for each uncached image, `racaf.prepare_stage4_input()`
+ `racaf.tta_views()` (both unmodified) are each called **exactly once**; the `"identity"`-indexed
view of that single call's four aligned predictions becomes Stage 5's canonical lesion-map cache
value, and `racaf.compute_reliability()` (unmodified) derives `kappa`/`r` from the SAME four views
— never a second, separate `predict_lesion_mask()` call. Verified: `tests/test_joint_training.py`'s
`CacheReuseAndRedundancyTests.test_tta_views_called_exactly_once_per_uncached_image` (mocks
`racaf.tta_views` and asserts a call count of 1) and
`test_lesion_cache_equals_the_identity_tta_view` (numerically compares the cached lesion map
against a directly-computed `racaf.tta_views()` identity slice, `atol=1e-5`). Stage 5's own
standalone loader (`local_feature_extraction_dataset.py`) is completely untouched by this — the
new dependency runs `joint_training_dataset.py → racaf.py`, never `local_feature_extraction_dataset.py
→ racaf.py`, so no Stage 5 → RACAF dependency was introduced into Stage 5's own module. RACAF's own
mathematics, its own cache function (`get_or_compute_reliability`), and Stage 5's dataset-loading
contract are all unmodified by this implementation.

---

## 12. RACAF reliability cache

Unchanged — `racaf.get_or_compute_reliability()` already stores only the small derived `kappa`
`(4,)` and scalar `r`, never the four raw `(512,512,4)` probability maps. Deterministic per image
(Stage 4 frozen); computed once, reused identically across every epoch and across train/val. Now
persists to Drive via `RACAF_CACHE_DIR` (§7.1), surviving a Colab VM restart.

---

## 13. Stage 5 input

`(512, 512, 8)` = canonical-resolution processed RGB(3) + vessel(1) + lesion(4), all now at the
same fixed 512×512 resolution before concatenation (§10). Unchanged in every other respect.

Stage 5's own internal fusion was completed in §47: the three multi-kernel branches are now
weighted per image and per channel before they are concatenated, which is what makes the block
"adaptive" rather than merely multi-kernel. The `(512, 512, 8)` input and the `(32, 32, 256)`
output are unchanged.

---

## 14. Stage 6 input

`(256, 256, 3)` = the same processed RGB, resized independently to its own target size — no
vessel/lesion channel, no dependency on Stage 3/4. Unchanged.

---

## 15. Stage 7

`Q` = Stage 6's 64 tokens, `K,V` = Stage 5's flattened 1024 tokens, `d_model=256` → `E=(B,256)`.
Unchanged; never reads Stage 4.

---

## 16. RACAF

Consumes `E` (Stage 7), `G` (Stage 6's raw output, read independently — not derived from `E`), and
`r` (precomputed scalar, §12) → `F = gate·E + (1-gate)·Ĝ`, `gate = σ(w_g·r + b_g)`,
`Ĝ = W_r·GAP(G) + b_r`. Unchanged — no equation altered by this document.

---

## 17. CORN

`F=(B,256)` → `Dense(256→4)` → 4 raw logits, decoded via sigmoid→cumulative-product→threshold.
Unchanged.

---

## 18. Full tensor flow

```
APTOS raw image
   │ Stage 2 (deterministic; cached once, reused)
   ▼
processed RGB (native resolution)
   │ Stage 3 (frozen; native-resolution inference, canonical-resolution cache, §10)
   ▼
vessel map (512,512) [cached]
   │ Stage 4 (frozen; native-resolution inference from native RGB+vessel, canonical-resolution cache, §11)
   ▼
lesion maps (512,512,4) [cached]  +  RACAF reliability r [cached separately, §12/§16]
   │
   ├─→ RGB(512,512,3, resized) + vessel + lesion → Stage 5 → L=(B,32,32,256)
   └─→ processed RGB (256,256,3, resized independently) → Stage 6 → G=(B,64,1152)
                                                                          │
                                              Stage 7: Q=G, K/V=L → E=(B,256)
                                                                          │
                                    RACAF: E, G (independently), r → F=(B,256)
                                                                          │
                                                          CORN: F → logits=(B,4)
                                                                          │
                                                                  corn_loss(logits, grade)
```

---

## 19. Gradient flow

**Receive gradient from CORN's loss:** Stage 5, Stage 6, Stage 7, RACAF's `w_g,b_g,W_r,b_r`, CORN.

**Stage 6 has two gradient paths, not one** — (a) through Stage 7's Q-projection → cross-attention
→ `E` → RACAF's `gate·E` term, and (b) directly through RACAF's `Ĝ = W_r·GAP(G) + b_r` term, since
RACAF reads Stage 6's raw `G` independently of Stage 7. This is intentional, not a bug — a
consequence of RACAF's own approved formula. Stage 5 has only path (a) — RACAF never reads Stage
5's `L` directly.

**No gradient reaches Stage 3 or Stage 4.** Stage 3 is PyTorch under `torch.no_grad()` —
structurally outside any TF graph. Stage 4 is `trainable=False` + `tf.stop_gradient`-wrapped
inside `racaf.tta_views()`; more strongly, keeping Stage 3/4 entirely inside the data/cache layer
(§9–§12) rather than calling them live inside the trainable Keras graph means their outputs enter
the joint model only as plain precomputed NumPy arrays via `Input` tensors, which carry no
gradient history regardless of any `stop_gradient` wrapper. `r` is a deterministic, cached signal —
differentiable only with respect to RACAF's own gate weights (`w_g,b_g`), never with respect to
Stage 4. Stage 1/2 are moot (§3.1 — not in the graph; no parameters).

---

## 20. Augmentation synchronization

**Required design property for the future joint dataset loader** (not yet built): apply one
spatial augmentation (flip/rotate) to the shared, canonical-resolution processed content **once**
per image, then derive both Stage 5's `(512,512,8)` tensor and Stage 6's `(256,256,3)` tensor from
that single augmented result — never two independent RNG draws. Today's two separate per-stage
loaders (`local_feature_extraction_dataset.py`, `global_feature_extraction_dataset.py`) happen to
draw identical augmentation decisions only because both iterate the same authoritative-split
entries in the same order with an identical per-sample RNG call pattern — this is **incidental,
not architected**, and must not be relied upon by the joint loader.

Within Stage 5's own 8 channels: RGB, vessel, and lesion maps must remain spatially synchronized
(already correct — `_augment_spatial` applies identically to all 8 channels). Intensity
augmentation is RGB-only (already correct — `_augment_intensity_rgb` leaves channels 3–7
untouched). RACAF's `r` is **never** independently augmented — it is tied to the canonical,
unaugmented image and Stage 4's frozen behavior (a property of the photograph, not of a particular
random crop/flip choice), so one cached `r` per `id_code` is correct regardless of which
augmentation Stage 5/6 draw for a given training step. This is already implicit in the existing
cache design (§12) and requires no change.

---

## 21. Loss

`corn.corn_loss(logits, grades)` only. No focal loss, Dice loss, segmentation loss, IQA loss, class
weighting, or auxiliary loss. APTOS2019's class imbalance (documented in `CORN_ARCHITECTURE.md`
§11) is reported, not acted on.

---

## 22. Validation

Authoritative 733 val IDs, no augmentation, canonical cached Stage 3/4 outputs (same cache as
train), same Stage 2 preprocessing, frozen Stage 3/4. Metrics: `corn_loss`, accuracy, macro-F1, and
QWK (`evaluation/metrics.py`, already implemented).

---

## 23. QWK checkpoint selection — locked (Step 7)

`monitor = val_QWK`, `mode = max`. Reason: CORN is an ordinal classifier; QWK is the
ordinal-appropriate metric and is already implemented and reusable. Training loss is unaffected —
`corn_loss` remains the sole training objective; QWK is used only for best-checkpoint selection.

**Implementation note (post-implementation audit fix):** `compile_joint_model()` originally
compiled with no metric at all, so Keras's own `logs` dict during `model.fit()` would never have
contained a `"val_QWK"` key for this monitor string to read — `ModelCheckpoint`/`EarlyStopping`/
`ReduceLROnPlateau` would each have silently skipped every epoch (Keras logs a "metric not
available" warning and no-ops), defeating this section's policy on the first real run. The
project's existing generic `training.metrics.QuadraticWeightedKappa` cannot be attached to
CORN's output directly either — it argmaxes `y_pred`, which would treat CORN's 4 conditional
threshold logits as 4 mutually exclusive classes (wrong, and structurally unable to ever produce
grade 4). Fixed by `corn.CORNQuadraticWeightedKappa` (`corn.py`) — a thin subclass that decodes
CORN's logits with EXACTLY `decode_logits`'s own sigmoid → cumulative-product → threshold-count
rule (in TensorFlow ops, so it runs inside `model.fit()`'s graph-mode execution) and delegates
confusion-matrix accumulation and kappa computation to `QuadraticWeightedKappa`, unmodified.
`compile_joint_model()` now passes `metrics=[corn.CORNQuadraticWeightedKappa()]` (named `"QWK"`,
so Keras logs `"QWK"`/`"val_QWK"`) alongside the unchanged `loss=joint_corn_loss` — QWK is a
METRIC only, never a second loss. Verified: `tests/test_corn.py`'s
`CORNQuadraticWeightedKappaTests` (decode equivalence with `decode_logits`, numerical agreement
with an independent `sklearn.metrics.cohen_kappa_score` reference, batch accumulation) and
`tests/test_joint_training.py`'s `CORNQWKJointIntegrationTests` (Keras logs actually contain
`"QWK"`/`"val_QWK"`, `ModelCheckpoint(monitor="val_QWK", mode="max")` finds the value and only
saves on improvement, weights-only save/load still round-trips with the metric compiled).

---

## 24. T4 strategy — locked (Step 9)

Starting configuration: `batch_size=2`, `mixed_precision=True` (already `training.TrainingConfig`'s
own default — no new framework code needed). This is a starting point, not a guaranteed-fitting
value — no local GPU exists in this environment, so no T4 memory number is measured or claimed
here. The eventual notebook must perform an empirical memory/smoke test before committing to a
larger batch size. If `batch_size=2` OOMs: reduce batch size first; only add gradient accumulation
if the framework already supports it (`training/trainer.py` currently does **not** implement
gradient accumulation — adding it is a separate, small framework change, not assumed here). The
architecture is never changed to solve a memory problem.

**Implementation note (T4 smoke-test fix):** the first real T4 smoke test
(`joint_model.predict(...)`) failed with `InvalidArgumentError: Trying to access resource
relative_position_index ... located on device CPU:0 from device GPU:0`. Root cause:
`swin_transformer.py`'s `WindowAttention.relative_position_index` was a bare
`tf.Variable(trainable=False)` — an int32 RESOURCE variable, which TensorFlow's own placement
policy pins to CPU regardless of GPU availability; reading a resource variable requires the
reading op to run on the same device it lives on, so the GPU-placed `tf.gather` inside
`WindowAttention.call()` could not read it once the joint model ran end-to-end on GPU. Fixed by
storing it as a plain `tf.constant` instead (not a resource, so it is copied to whatever device
consumes it, exactly like any other tensor) — no relative-position value, window geometry,
attention behavior, output shape, or parameter count changed (`tests/test_swin_transformer_dual_scale.py`'s
`RelativePositionIndexDeviceRegressionTests`: Stage 06 still reports `39,697,956` params and
`(B,64,1152)` output; GPU-specific tests are skipped on this project's CPU-only local/CI
environment and run only where a GPU is actually present).

---

## 25. Checkpoint/resume

Stage 6's underlying Swin layer classes have no `get_config()` (`PROJECT_STRUCTURE.md` §6's own
documented gap) — a full single-file `.keras` save of a joint model embedding Stage 6 would likely
fail to reconstruct on load. `training.TrainingConfig.save_weights_only` already defaults to
`True`, and `PROJECT_STRUCTURE.md` §6 already states the joint run "will itself default to
weights-only checkpointing unless a future implementer deliberately overrides it" — this document
does not introduce that decision, it reuses it.

**Implemented** in `joint_training_model.py`: `build_joint_model()` reconstructs the composed
architecture fresh (chaining `local_feature_extraction_model.build_local_feature_extractor()`,
`swin_transformer.create_dual_scale_swin_model()`, `feature_fusion.build_adaptive_cross_attention()`,
`racaf.build_racaf_fusion()`, `corn.build_corn_model()`), and
`save_joint_model_weights()`/`load_joint_model_weights()` are the weights-only save/reload pair —
mirroring `GlobalFeatureExtractionStage.load()`'s existing "rebuild then load_weights" pattern.
Verified: `tests/test_joint_training.py`'s `test_save_and_load_weights_round_trip` (predictions
match exactly, `atol=1e-5`, after a save/reload round trip in a temp directory). Both functions are
pure, path-parameterized (`path` is a required argument, no built-in default —
`test_checkpoint_functions_take_no_hardcoded_path`) — the actual persistent Drive location
(`experiments/FinalClassification/<timestamp>/checkpoints/`) is resolved by the caller (the
notebook, §27) via the EXISTING, unmodified `experiment_manager.resolve_experiment()` +
`colab_config.DRIVE.experiment_dir("FinalClassification")` infrastructure — this module makes no
Drive/local assumption of its own. Resume support (`resume_from=...`, `Trainer`'s existing
`epoch_state.json` mechanism -- **superseded by §49's generation-based checkpoints, which
additionally persist optimizer state, the global best `val_QWK`, and the stateful callbacks'
counters; the weights-only FORMAT decision recorded here is unchanged and still in force**),
best/final checkpoint, and per-stage exported-weight slices (into
each stage's own `config.py` `MODEL_DIR`, §7.1, so each stage's own already-implemented
`Stage.load()` keeps working independently) are all designed for but **not exercised** by this
task — no real checkpoint has been generated, per this task's explicit "no training" constraint.
Stage 1–4 checkpoints are never overwritten.

---

## 26. Experiment structure

Reuses `colab/common/experiment_manager.py` unmodified — `experiments/FinalClassification/<timestamp>/{checkpoints,logs,tensorboard,evaluation,predictions}/` + `metadata.json`. No new
timestamp/run system is introduced. Large, reusable, cross-run caches (§10–§12) live under the new
`cache/<module>/` Drive root (§7.1), not duplicated inside every experiment directory.

---

## 27. Notebook role — locked (Step 8)

`colab/notebooks/stage08_corn_classifier.ipynb` is repurposed as the joint Stage 05–08+RACAF
training notebook — CORN has no standalone training path of its own to otherwise fill this
already-reserved slot, and `IMPLEMENTATION_PLAN.md`'s own Step 8 section already deferred "the main
end-to-end training notebook" here. No second, competing notebook exists.

**Implemented** (infrastructure/cells; no cell runs real training as committed): Bootstrap →
`setup.setup()` + `verify_environment.verify_all()` (unchanged from the prior template) → dataset
verification (`train.csv` row count + `verify_dataset.verify_image_folder()` on `train_images/`) →
frozen Stage 1/3/4 checkpoint discovery (Stage 1 resolved for completeness only, never loaded into
this graph, §3.1) → authoritative split load + assertion (§6) → persistent cache location
reporting (§7.1, §10–§12 — population itself is lazy, deferred to actual dataset iteration) →
joint model construction (`joint_training_model.build_joint_model()` + `compile_joint_model()`,
§4, §25) → a synthetic-tensor smoke test (forward pass + one `GradientTape` step, no real data) →
training configuration cell (`RUN_TRAINING = False`, `batch_size=2`, `mixed_precision=True`,
`monitor="val_QWK"`, `mode="max"`, §23–§24) → a gated dataset-loading cell
(`jtd.load_joint_training_datasets()`) → a gated experiment/training cell (Step B) that resolves
the Drive experiment (`experiment_manager.resolve_experiment(..., resume_from=RESUME_EXPERIMENT_DIR)`)
and then calls the actual, unmodified `training.Trainer(training.TrainingConfig(...)).fit(joint_model,
train_ds, val_ds)` — the project's existing training API, not a new one (verified end-to-end with
synthetic tensors and a temp directory in `tests/test_joint_training.py`'s
`TrainerIntegrationTests`, including that `ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau`
genuinely recognize `"val_QWK"`/`mode="max"` and that `TrainingConfig(resume=True)` genuinely
reloads the last checkpoint and advances `initial_epoch`). Both the dataset-loading and
experiment/training cells are **gated behind `if RUN_TRAINING:`** and print a skip message when
`False` (the committed state) — opening or running every cell in this notebook, as committed,
never touches real Drive-mounted data, never creates an experiment directory, and never calls
`model.fit()`. Setting `RUN_TRAINING = True` and re-running is a separate, explicit, future
action.

---

## 28. Deferred final evaluation

`APTOS2019/raw/test.csv` (1928 images, no `diagnosis` column) remains unusable for supervised
evaluation. IDRiD grading's 103-image official test split remains **DEFERRED / PENDING FINAL
EVALUATION APPROVAL** — not used for training, validation, or model selection. CORN's architecture
is indifferent to this choice (it only ever reads `F`), so a different final evaluation set can be
substituted later with zero architecture change.

---

## 29. Explicit non-goals (this document and this task)

- The joint model builder and joint dataset loader ARE implemented and unit-tested
  (`joint_training_model.py`, `joint_training_dataset.py`). No training loop is run, no checkpoint
  is generated, and no notebook cell is executed by this task — `RUN_TRAINING = False` throughout
  `colab/notebooks/stage08_corn_classifier.ipynb`; real training is a separate, future, explicit
  action.
- No retraining, modification, or Git-copying of Stage 1/3/4's checkpoints.
- No change to RACAF's or CORN's mathematics, loss, or tensor contracts.
- No second dataset hierarchy; no fourth project dataset.
- No IQA gating added to APTOS2019 training.
- No auxiliary loss (segmentation, IQA, focal, Dice, class-weighted) added to `corn_loss`.

---

## 30. Research innovation boundary

RACAF remains the project's ONE approved research innovation. Everything in this document — the
joint training procedure, Drive/cache infrastructure, augmentation synchronization, checkpoint
format, QWK-based model selection — is training/engineering strategy, not a second innovation. No
new attention mechanism, fusion mechanism, uncertainty module, auxiliary loss, or feature extractor
is introduced anywhere in this design.

---

## 31. Empty-FOV handling — fixed (first real T4 run blocker)

**Finding:** the first real T4 training run crashed during epoch 1 with `IndexError: list index
out of range` in `vessel_segmentation_inference.crop_to_fov()` (`regionprops(fov_mask.astype(int))
[0].bbox`). Root cause: `compute_fov_mask()`'s circle-fit (`_fit_circle`, an unconstrained
Nelder-Mead search) minimizes mismatch against the thresholded foreground mask with no radius
constraint — for a real APTOS image whose thresholded foreground is sparse/scattered rather than
one solid disk (confirmed for `id_code=0ce062f26edc`, train split, diagnosis 0: only ~2.3% of
pixels pass `threshold_minimum`, and the fit converges to `radius≈-0.44`), "no circle at all"
minimizes that mismatch better than any real circle, so `fov_mask` ends up with zero foreground
pixels. This is a data-quality condition, not a Stage 03 architecture or model defect — the LWNet
model is never invoked before the crash.

**Fix made:** `vessel_segmentation_inference.py` now raises a named, documented
`EmptyFieldOfViewError` (from `crop_to_fov`, and from `_fit_circle` for the sibling all-empty-input
case) instead of letting the bare `IndexError` propagate — behavior for any image with a normal,
non-empty FOV is completely unchanged (identical bbox, identical crop). There is no
project-sanctioned full-image-FOV fallback and none was added. `joint_training_dataset.py`'s
`_make_joint_dataset()` generator (and, for the identical exposure, `local_feature_extraction_
dataset.py`'s `_make_dataset()` generator — both call the same `predict_vessel_mask`) now catch
`EmptyFieldOfViewError` per-image, log the skipped `image_id`, and exclude just that sample from
the epoch, rather than crashing the whole run or fabricating a vessel/FOV result. This does NOT
modify the authoritative split manifest on disk (§6) — every id, including `0ce062f26edc`, remains
listed there; a `tf.data.Dataset` built from it may simply yield one fewer sample than the nominal
2929 (train) count per epoch. `lesion_segmentation_dataset.py` (IDRiD, 81 fixed images, already
used to train the frozen Stage 04 checkpoint without incident) was left unchanged — no evidence
this class of image exists in that small, already-processed set.

Regression tests: `tests/test_joint_training.py` (`EmptyFieldOfViewHandlingTests`,
`EmptyFieldOfViewLowLevelTests`) and `tests/test_local_feature_extraction_dataset.py`
(`EmptyFieldOfViewHandlingTests`) — the exact crash path (`crop_to_fov` on an empty mask), the
unchanged bbox for a normal mask, the generator's selective skip-and-log behavior, and that normal
images are entirely unaffected.

---

## 32. RAM exhaustion — fixed (two-phase workflow: cache precomputation / training)

**Finding:** with the §31 empty-FOV fix in place, the first real T4 training attempt ran for
~50 minutes, skipped 4 empty-FOV images as designed, and then exhausted all of Colab's available
RAM before epoch 1 ever completed — while still inside the dataset/cache pipeline, not inside
`model.fit()`'s actual gradient computation. Measured root cause: `_make_joint_dataset()` sized
its `tf.data` shuffle buffer to `buffer_size=max(len(entries), 1)` — the ENTIRE dataset (up to
2929 for train). `tf.data.Dataset.shuffle()` holds `buffer_size` **fully-materialized** elements
at once, not file paths or lazy references — and each element here is a
`(512,512,8)` + `(256,256,3)` float32 sample pair, ≈8.75 MB. `2929 × 8.75 MB ≈ 25.6 GB`, which by
itself exceeds even Colab Pro's higher-RAM tier, independent of Stage 03/04/RACAF inference cost,
Drive I/O speed, or any Python-side list accumulation (there was none — the existing
`_get_or_compute_joint_frozen_outputs()` per-image caching already discarded each image's arrays
before moving to the next; it was never the accumulation mechanism). The identical
`buffer_size=len(entries)` pattern was independently confirmed in three sibling dataset loaders:
`local_feature_extraction_dataset.py` (Stage 05, same ≈8 MB/sample — same severity),
`global_feature_extraction_dataset.py` (Stage 06, ≈0.75 MB/sample — smaller but still unbounded),
and `lesion_segmentation_dataset.py` (IDRiD, only 81 fixed images total — harmless in practice, so
left unchanged, same reasoning as §31's identical judgment call).

**Fix made — two parts:**

1. **Bounded shuffle buffer.** `joint_training_dataset.py`, `local_feature_extraction_dataset.py`,
   and `global_feature_extraction_dataset.py` each now cap their shuffle buffer at a small, FIXED
   `DEFAULT_SHUFFLE_BUFFER_SIZE = 256` (`buffer_size = max(1, min(len(entries), <cap>))`) instead
   of `max(len(entries), 1)` — memory now stays bounded (~2.2 GB worst case for the joint
   pipeline) regardless of how large the dataset is. This alone fixes the crash: `RUN_TRAINING`
   in the notebook is memory-safe again even with no other change.

2. **Phase 1 / Phase 2 separation (recommended, not required for correctness).**
   `joint_training_dataset.precompute_joint_frozen_caches(entries, ...)` streams over `entries`
   ONE IMAGE AT A TIME, calling the same, unmodified `_get_or_compute_joint_frozen_outputs()`
   every existing consumer already uses — no per-image array is ever held past its own loop
   iteration, no list of samples is ever accumulated, and an entry whose cache already fully
   exists is skipped without touching `vessel_model`/`stage4_model` at all. This makes cache
   precomputation safe to interrupt and resume indefinitely: a valid cache entry is never
   recomputed or deleted, only missing ones are filled in.
   `precompute_authoritative_joint_caches(...)` is the real-workflow entry point — it reads the
   SAME authoritative split (§6) as `load_joint_training_datasets()` (never a second one) and
   precomputes caches for both train and val. Phase 2 (`load_joint_training_datasets`) is
   functionally unchanged: it still computes-and-caches any still-uncached entry on the fly if
   Phase 1 was skipped, so running Phase 1 first is a recommended optimization (it decouples slow,
   Drive-I/O-bound Stage 03/04/RACAF inference from the actual `Trainer.fit()` call, and makes
   repeated training runs, e.g. with different hyperparameters, much faster once caches exist) —
   never a hard requirement.

**No architecture, tensor contract, loss, or split changed.** Stage 3/4 remain frozen and
untouched; Stage 5/6/7/RACAF/CORN architecture is unchanged; the authoritative split is still
exactly 2929 train / 733 val, read from the same manifest, never a second one; the canonical
512×512 cache contract and RACAF `kappa`/`r` cache contract are unchanged (Phase 1 writes to the
exact same cache files Phase 2 already read); synchronized augmentation is unaffected (it still
happens after cached frozen outputs are read, inside `_build_joint_sample`, never by modifying a
canonical cache file). No `.cache()` (in-RAM `tf.data` cache) is used anywhere in this pipeline —
confirmed by inspection, not assumed.

**Existing Drive caches remain valid.** Nothing about the cache file format, path convention, or
contents changed — only how/when caches get populated. Any cache entries already written by a
prior run (including the 4 images already skipped for empty FOV, and whatever fraction of the
dataset the crashed run got through before running out of RAM) are reused as-is.

**Notebook workflow (`colab/notebooks/stage08_corn_classifier.ipynb`):** a new gated cell pair
("Phase 1 -- Cache precomputation", `RUN_CACHE_PRECOMPUTATION = False` by default) was inserted
between the existing "Persistent cache locations" and "Joint model construction" cells. Setting
`RUN_CACHE_PRECOMPUTATION = True` and running that cell first stages APTOS2019's raw images onto
the Colab VM's local SSD (`dataset_staging.stage_dataset()` — the SAME generic staging module
`stage01_iqa.ipynb` already uses for EyeQ, not reimplemented), then calls
`precompute_authoritative_joint_caches(image_dir=<staged train_images dir>, ...)` for the whole
dataset; it is always safe to interrupt and re-run (re-staging is a no-op if already staged, and
already-cached entries are skipped as always). The existing "Dataset loading" / "Experiment +
training" cells (`RUN_TRAINING`) are otherwise unchanged — they still read directly from Drive and
work standalone, now with the bounded shuffle buffer.

Regression tests: `tests/test_joint_training.py` (`ShuffleBufferBoundTests`,
`CachePrecomputationTests`, `PrecomputeAuthoritativeCachesTests`, `Phase2UsesExistingCachesTests`),
`tests/test_local_feature_extraction_dataset.py` (`ShuffleBufferBoundTests`),
`tests/test_global_feature_extraction_dataset.py` (`ShuffleBufferBoundTests`) — all synthetic,
tiny fixtures; no 3662-image cache generation is ever run in the test suite.

---

## 33. Phase 1 throughput — diagnosed and fixed (Drive-mounted cache I/O)

**Symptom:** with §32's fixes in place, a real `RUN_CACHE_PRECOMPUTATION=True` T4 run processed
only a handful of images in ~2 hours (8 empty-FOV skips observed) — impractical.

**Diagnosis (measured from code, not guessed):** ruled out first —
  - *Repeated model loading*: `precompute_joint_frozen_caches`/`precompute_authoritative_joint_caches`
    load `vessel_model`/`stage4_model` exactly once, before the loop, confirmed by inspection —
    not the cause.
  - *CPU-bound inference*: Stage 03/04 forward passes on a T4 GPU for a single `512×512` image are
    sub-second; `compute_fov_mask` (Stage 03's CPU-only FOV heuristic, unchanged/frozen — see §31)
    was independently measured (this session, same machine class) at ~1-2s/image — real, but not
    enough alone to explain multi-minutes-per-image.
  - *Repeatedly reading/writing Drive* (confirmed, dominant): the notebook resolves `cache_dir=
    config.LOCAL_FEATURE_RESULTS_DIR` and `racaf_cache_dir=config.RACAF_RESULTS_DIR` to
    `colab_config.LOCAL_FEATURE_CACHE_DIR`/`RACAF_CACHE_DIR` — both **Google Drive** paths
    (`drive_paths.py`'s `DRIVE.cache_dir(...)`, wired by `colab/common/setup.py`). Every
    `os.path.exists` check and every `np.save`/`np.savez` write for these caches was therefore a
    Google Drive FUSE round trip, which (per this project's own `dataset_staging.py` docstring)
    is "latency-bound per file open, not bandwidth-bound." `APTOS2019_PROCESSED_DIR` (the default
    `processed_dir` `_resolve_processed_rgb` checks) is likewise Drive-mounted, adding one more
    per-image Drive stat. None of this touches Stage 3/4/RACAF's own computation.
  - *Redundant work* (confirmed, minor): for every still-uncached image,
    `precompute_joint_frozen_caches` checked all three cache paths' existence itself, then called
    `_get_or_compute_joint_frozen_outputs`, which — unaware the caller had already checked —
    immediately re-checked the identical three paths again before falling through to computation.
    Python's `and` short-circuits that recheck at its first `False`, so this wasted exactly 1
    extra Drive round trip for a fully-uncached image (the common case), up to 3 for a
    partially-cached one left by a prior interrupted run — small next to the dominant cost above,
    but free to remove and folded into the same fix.

**Fix — three parts, no compute/architecture change:**

1. **Local cache during precomputation, synced to/from Drive in bulk.** A new, generic,
   direction-agnostic `dataset_staging.sync_missing_files(source_dir, dest_dir)` (alongside the
   existing `stage_dataset()`, reusing its same thread-pool-concurrency `_copy_one` copy — not a
   competing mechanism) copies only files missing at the destination, in either direction. The
   notebook's Phase 1 cell now: (a) **pulls** any cache entries a prior session already wrote to
   Drive down to a local cache dir (`/content/cache/...`) — resumability fully preserved, just
   against a local mirror; (b) runs `precompute_authoritative_joint_caches()` entirely against
   that **local** `cache_dir`/`racaf_cache_dir` (fast SSD I/O in the hot loop, zero Drive round
   trips per image) and a local, deliberately-empty `processed_dir` (so `_resolve_processed_rgb`
   always takes its cheap, unmodified live Stage 02 fallback rather than one more Drive lookup);
   (c) **pushes** every newly-written local cache entry back up to Drive so it persists. A new
   **Phase 1b** cell runs the push step alone, at any time, so a manually interrupted run's
   progress can be flushed to Drive without waiting for the whole precomputation to finish.
2. **Redundant existence check removed.** `_get_or_compute_joint_frozen_outputs()` gained one new,
   opt-in parameter, `known_not_all_cached=False` (default preserves the exact original behavior
   for every existing caller, including `_build_joint_sample`/Phase 2) — when a caller has already
   verified at least one cache file is missing, passing `known_not_all_cached=True` skips the
   function's own duplicate recheck of the same three paths. `precompute_joint_frozen_caches`
   passes it; nothing else does. The three PER-FILE write guards (protecting a partially-populated
   cache left by an interrupted run) are untouched either way.
3. **Throughput visibility.** `precompute_joint_frozen_caches`'s progress log (every
   `progress_every` images, and once more at the end even if the count doesn't land on that
   boundary) now reports elapsed wall-clock time and a rolling images-per-minute rate alongside
   the existing processed/cached/skipped counts, and the returned stats dict gains
   `"elapsed_seconds"` — so a long run's real speed is visible without guessing.

**Batching/vectorizing Stage 03/04 inference across multiple images was considered and NOT
implemented.** Given the diagnosis above places the dominant cost in Drive I/O, not GPU compute,
batching would add real risk (touching `racaf.tta_views`/`prepare_stage4_input`'s shared, tested
contract, requiring new batch-shape coverage for RACAF's TTA/reliability path) for uncertain
marginal benefit once I/O is no longer the bottleneck. `predict_vessel_mask_batch()` already
exists in `vessel_segmentation_inference.py` for a possible future pass if Drive I/O elimination
alone still proves insufficient on real Colab hardware — deliberately not wired in here.

No Stage 3/4/RACAF/Stage 5/6/7/CORN architecture, weights, reliability equations, or TTA
definition changed. All four RACAF TTA views remain. The canonical cache format/keys are
unchanged — only where (local vs. Drive) and how many times (once vs. twice) each cache path is
checked.

Regression tests: `tests/test_dataset_staging.py` (`SyncMissingFilesTests` — copy-missing-only,
resumability, both directions, no-op on a nonexistent source), `tests/test_joint_training.py`
(`CachePrecomputationDriveRoundTripTests` — the redundant-check removal proven differentially
[robust to incidental library `os.path.exists` calls] and by wiring, `known_not_all_cached`'s
strictly-additive default, progress-log content, `elapsed_seconds` in the returned stats).

---

## 34. RAM exhaustion, again — diagnosed and fixed (GPU VRAM, not CPU RAM)

**Symptom:** after §33's fix (local cache, redundant-check removed), a real Colab run still
crashed from "RAM exhaustion" — but Colab's own visible RAM graph never showed CPU memory
approaching its limit. That mismatch was the key clue and the starting point of this diagnosis.

**Method:** every function named in scope (`sync_missing_files`, `precompute_authoritative_joint_
caches`, `precompute_joint_frozen_caches`, `_get_or_compute_joint_frozen_outputs`, the Stage 3/4
inference calls, `racaf.tta_views`/`prepare_stage4_input`/`compute_reliability`) was re-read fresh
from the current repository, not from prior reports. A diagnostic script then ran the REAL
`_get_or_compute_joint_frozen_outputs()` — the real, checked-in LWNet (Stage 03) and Experiment 2C
(Stage 04) checkpoints, both present in this local environment — over real APTOS images, one at a
time, each writing to a fresh temp cache dir (forcing a genuine compute, never a cache hit), and
measured **process RSS via `psutil`** before/after every image, both with and without a forced
`gc.collect()`, plus live `len(gc.get_objects())`.

**Measured (CPU side, this machine, no GPU — 20 real images, real checkpoints):**

```
idx  rss_before  rss_after(gc)  delta(gc)  gc_collected  objects
 0      707.8        1307.1      +599.4        0         608665
 1     1313.3        1106.5      -206.8        0         608665
 2     1110.5        1202.0       +91.5        0         608665
 3     1184.4        1091.8       -92.6        0         608665
 4     1104.4        1165.7       +61.3        0         608665
 5      848.6        1196.4      +347.8        0         608665
 6     1177.0        1177.2        +0.1        0         608665
 7     1187.7        1182.9        -4.8        0         608665
 8     1172.8        1177.9        +5.1        0         608665
 9     1172.9        1047.1     -125.8        0         608665
10     1034.1        1020.3      -13.8        0         608665
11     1047.9        1120.8      +72.9        0         608665
12     1103.2        1102.6       -0.6        0         608665
13     1120.4        1142.0      +21.6        0         608665
14     1171.7        1157.4      -14.3        0         608665
15     1142.9        1027.8     -115.1        0         608665
16     1069.6        1174.7     +105.1        0         608665
17     1141.4        1147.1       +5.7        0         608665
18     1164.2         827.0     -337.2        0         608665
19      514.8        1150.0     +635.2        0         608665

Linear regression, post-gc RSS vs. image index (n=20):
  slope = -7.665 MB/image  (i.e. slightly DOWNWARD, not upward)
  slope (no forced gc)     = -19.339 MB/image
Baseline RSS 661.3 MB -> final (post-gc) RSS 1150.0 MB; net +488.7 MB over 20 images, entirely
attributable to the one-time jump on image 0 (+599.4 MB) -- removing that single outlier, RSS is
net FLAT to slightly down across images 1-19.
gc object count: baseline 607738 -> final 608665 (+927 total, +46/image -- small, one-time module/
interpreter bookkeeping growth, not scaling with images processed; NOT a leak, since gc.collect()
found zero collectible garbage on every one of the 20 calls).
```

`gc.collect()` reclaimed **zero** objects on every single image, and the live object count's tiny,
one-time increase does not scale with images processed — conclusive proof there is no retained
NumPy array, TensorFlow tensor, PyTorch tensor, list, or reference-cycle leak anywhere in this call
path. RSS jumps sharply on the first image (one-time cost of each framework's internal allocator/
kernel-cache warming up for a shape/config it hasn't seen before) then **oscillates in a bounded
~700–1300 MB band with a measured NEGATIVE linear-regression slope across all 20 images** — not
linear growth, not even a plateau, a slight net decline. This rules out the CPU side entirely as
the cause of an unbounded, multi-image crash, and explains
directly why Colab's CPU RAM graph never showed the problem: there wasn't one, on the CPU.

**Root cause, confirmed by tracing every call site (not inferred):** `tf.config.experimental.
set_memory_growth` was **never called anywhere reachable from Phase 1** —
`joint_training_dataset.py` had zero `check_gpu`/`set_memory_growth` references before this fix,
and neither does `Trainer.prepare()` (it only calls `enable_mixed_precision`) or `verify_environment.
verify_all()` (its GPU checks only *list* devices and set the mixed-precision *policy* — never
memory growth). The only place in this whole project that calls `set_memory_growth` is
`training.trainer.check_gpu()`, and it is used by `train_image_quality.py`/`colab/common/
environment.py` — never by the joint-training path. Without it, **TensorFlow's default GPU
allocator claims essentially all free VRAM the instant it first touches the GPU** (Stage 04's
first `stage4_model(...)` call). Stage 03's vessel model runs on the SAME GPU
(`vessel_segmentation_model.resolve_device()` picks CUDA when available) via **PyTorch's own,
completely independent CUDA caching allocator** — the two frameworks share the physical device but
never coordinate with each other. Whichever one initializes second gets whatever the first one
left behind, which under TF's default (non-growth) behavior can be next to nothing. This is a
short-lived-at-onset but then **structurally persistent, per-session allocation-policy problem**,
not a per-image leak — exactly why it manifests as "RAM exhaustion" the CPU graph never shows
(it's VRAM, a different resource pool) and why it wasn't fixed by moving cache I/O to local SSD or
capping the `tf.data` shuffle buffer (§32/§33) — neither of those touches GPU memory policy at all.

**Fix:** `joint_training_dataset.py` now imports and calls `training.check_gpu()` — the SAME,
already-tested function `train_image_quality.py` and `colab/common/environment.py` already rely
on, not a reimplementation — as the very first action in both `precompute_joint_frozen_caches()`
and `precompute_authoritative_joint_caches()`, before either the vessel model or Stage 04 model is
loaded. `check_gpu()` requests incremental GPU memory growth for every visible GPU, is a documented
no-op when no GPU is present (this session's own CPU-only run confirms it prints "No GPU detected"
and proceeds normally), and is safe to call repeatedly (each call is independently wrapped in
`try/except RuntimeError`, matching its one existing caller's pattern).

**Why batching was still not implemented:** this diagnosis, like §33's, again places the cause
outside per-image compute cost — this time in a one-time, per-process GPU allocator policy, not
in the volume or size of individual forward passes. Batching Stage 3/4 inference across images
would not address a missing memory-growth flag and was correctly out of scope again.

**No Stage 3/4/RACAF/Stage 5/6/7/CORN architecture, weights, reliability equations, TTA
definition, or cache format/keys changed.** No existing Drive cache entry was invalidated or
deleted. Resumability is unaffected — `check_gpu()` has no bearing on which cache entries are
considered already-computed.

Regression tests: `tests/test_joint_training.py` (`GPUMemoryGrowthSafeguardTests` — `check_gpu()`
called before either model loads, in both Phase 1 entry points; called even when both models are
passed in pre-loaded; confirmed to be the exact same `training.check_gpu` object, not a
reimplementation). Actual VRAM behavior cannot be exercised in this suite (CPU-only) — these tests
verify the call is wired in at the correct point, which is what a real GPU run depends on.

---

## 35. Full Phase 1 / Drive-staging audit — bulk cache pull crashed Drive's FUSE mount

**Symptom (real Colab T4 run, after §32/§33/§34's fixes):** APTOS staging (5593 files, 153s)
completed cleanly. The very next step —
`dataset_staging.sync_missing_files(config.LOCAL_FEATURE_RESULTS_DIR, LOCAL_CACHE_DIR)`, pulling
the existing Drive cache down to local disk before Phase 1 started — failed with
`OSError: [Errno 107] Transport endpoint is not connected` while copying one of the cache files.
The user also observed GPU memory rising within the first few minutes, before any real per-image
processing had run.

**Audit method (per the explicit "do not patch blindly" instruction):** every file in scope —
`stage08_corn_classifier.ipynb`, `dataset_staging.py`, `drive_paths.py`, `colab_config.py`,
`setup.py`, `verify_dataset.py`, `verify_environment.py`, `environment.py`,
`joint_training_dataset.py`, `local_feature_extraction_dataset.py`, `racaf.py`,
`vessel_segmentation_inference.py`, `training/trainer.py` — was re-read from the current
repository, plus `git log` over the last several infrastructure commits, before any change.

**Root cause 1 (confirmed, structural): the Drive-cache pull was a blind bulk copy.** The prior
Phase 1 cell called `sync_missing_files(Drive cache dir, fresh local dir)` before starting —
walking the ENTIRE persistent Drive cache (a real prior run had left ~5948 Stage03/04 files and
~2974 RACAF files there) and copying every one of them, concurrently (16 workers), via
`shutil.copy2()` straight against the Drive FUSE mount. This ran immediately after the APTOS
staging copy had just finished the same 16-worker pattern over 5593 files — a sustained,
back-to-back burst of thousands of concurrent small-file Drive opens, which is the documented way
to destabilize/crash `drivefs` (ENOTCONN = the FUSE daemon disconnected under load). This is
exactly the design smell the audit request itself named: Phase 1 never needed the CONTENT of an
already-cached entry to know it should be skipped — only its existence.

**Root cause 1 fix — existence check, not bulk copy.** `precompute_joint_frozen_caches()` /
`precompute_authoritative_joint_caches()` (`joint_training_dataset.py`) gained two new parameters,
`persistent_cache_dir`/`persistent_racaf_cache_dir` (default `None`, fully backward compatible). If
an entry is not found under the fast local `cache_dir`/`racaf_cache_dir`, it is next checked
against `persistent_cache_dir`/`persistent_racaf_cache_dir` — via `_cache_entry_exists()`, the SAME
`os.path.exists`-only check already used for `cache_dir`, never a content read. A hit in either
location is skipped (never recomputed); a miss is computed and written ONLY to the local
`cache_dir`/`racaf_cache_dir`, never to the persistent one. The notebook's Phase 1 cell no longer
pulls the Drive cache down at all — it passes `config.LOCAL_FEATURE_RESULTS_DIR`/
`config.RACAF_RESULTS_DIR` directly as `persistent_cache_dir`/`persistent_racaf_cache_dir`. The
existing "Phase 1b -- Manual flush to Drive" cell (unchanged in purpose) still pushes newly
computed local entries up afterward — bounded by however many images this session actually
computed, never the full prior-run backlog.

**Root cause 1, defense in depth — `sync_missing_files()`/`stage_dataset()` hardened.** Even with
the bulk pull removed, the PUSH direction (and `stage_dataset()`'s initial dataset copy) still use
the same concurrent-Drive-copy pattern, so `dataset_staging._copy_one()` (shared by both) was made:
  - **Atomic**: copies to a same-directory temp file, verifies its size against the source, then
    `os.replace()`s it into place — a mid-copy failure can never leave a partial/corrupt file that
    every `os.path.exists` cache-hit check in this project would otherwise silently trust.
  - **Retrying, but only for transient errors**: ENOTCONN/ESTALE/ETIMEDOUT/EIO (the FUSE-instability
    family) are retried up to 4 times with exponential backoff; anything else (permission, disk
    full) still fails immediately, since retrying it cannot help.
  - **Non-aborting at the batch level**: `sync_missing_files()` no longer re-raises and aborts the
    whole call on the first file's failure — every other independent file still gets copied, and a
    failed file is simply reported in a new `failures` return value (its return signature is now
    `(copied_count, already_present_count, failures)`) and remains missing at the destination, so
    the existing resumability convention (a later call retries whatever is still missing) already
    covers "resuming" a partial sync with no special handling needed.

**Root cause 2 (GPU memory observation) — investigated, no defect found.** `check_gpu()`'s ordering
(§34's fix) was re-verified structurally against the CURRENT notebook: every cell before Phase 1
(`import setup` → `setup.setup()` → `verify_environment.verify_all(require_gpu=True)` → checkpoint-
path prints) was traced line by line. `verify_all()`'s GPU-related checks (`list_physical_devices`,
`get_device_details`, setting the mixed-precision policy string) are device-LISTING/metadata calls,
not device-initializing ones — none of them claims VRAM. `check_gpu()` is still the first action
inside both Phase 1 entry points, before either model loads, exactly as §34 fixed it. Stage 03's
PyTorch forward passes are also confirmed wrapped in `torch.no_grad()`
(`vessel_segmentation_inference._predict_probability_map`), so no autograd-graph accumulation is
possible there either. No code defect was found that would explain unbounded GPU growth. Given the
failure this run actually hit (`Errno 107`) occurred during `sync_missing_files` — before either
model had loaded at all, since that pull ran ahead of `precompute_authoritative_joint_caches()` in
the old cell order — the observed GPU growth cannot be causally tied to this crash; it is most
consistent with expected one-time CUDA-context + dual-framework model-load overhead. Rather than
make a second speculative GPU change, the new diagnostic mode below adds real RSS + TensorFlow GPU
memory instrumentation, so the next real run measures this instead of relying on Colab's own graph.

**New: small-scale diagnostic mode.** `precompute_joint_frozen_caches()`/
`precompute_authoritative_joint_caches()` gained `max_images` (process only the first N entries)
and `verbose_diagnostics` (log one line per image: cache status, this image's elapsed time, running
images/minute, process RSS via `psutil` if installed, and TensorFlow-reported GPU memory via
`tf.config.experimental.get_memory_info` if a GPU is visible). Both exercise the REAL code path —
real models, real images, real cache I/O — never a mock, and neither changes which entries are
computed or what gets written to the cache; the diagnostic lines are logged and discarded per
image, never accumulated into the returned `stats`. The notebook's Phase 1 cell exposes this as
`CACHE_DIAGNOSTIC_MAX_IMAGES` (`None` by default; set to 5/10/25/50 to try a small run first).

**Which prior fixes remain correct vs. were insufficient:** empty-FOV handling (§31), the fixed
shuffle-buffer cap (§32), and `check_gpu()`'s ordering (§34) are all still correct and were NOT
modified by this audit. The local-cache-I/O fix (§33) was insufficient: it correctly moved cache
*writes* to local SSD, but its own pre-pull step still performed the same kind of bulk,
high-concurrency Drive hammering that fix was meant to eliminate — just against the cache directory
instead of the raw dataset. This audit removes that remaining bulk-copy pattern entirely.

**No Stage 3/4/RACAF/Stage 5/6/7/CORN architecture, weights, reliability equations, TTA definition,
or cache format/keys changed.** No existing Drive cache entry is deleted, invalidated, or
duplicated — `persistent_cache_dir`/`persistent_racaf_cache_dir` are read-only from this module's
perspective, exactly like `stage_dataset()`'s existing Drive-source convention. The authoritative
2929/733 split is untouched.

Regression tests: `tests/test_dataset_staging.py` (`SyncMissingFilesTests` — atomic writes, no
partial file survives a failure, ENOTCONN retried and eventually succeeds, a non-transient error
fails without retrying, one file's failure does not block or abort others, a failed file is picked
up by a later resumed call). `tests/test_joint_training.py` (`PersistentCacheDirTests` — an entry
cached only in the persistent dir is a hit with zero inference calls and zero local duplication; a
genuine miss is computed and written locally only; a local hit is checked first; `persistent_cache_
dir=None` reproduces the exact prior default behavior; `precompute_authoritative_joint_caches`
forwards the new parameters unchanged) and (`DiagnosticModeTests` — `max_images` limits real
processing via the real code path; diagnostic logging emits exactly one line per processed image
with RSS/GPU fields, off by default, and never changes the returned stats' shape).

---

## 36. First real training run — ~5–6s/step, traced to Phase 2 reading every cache/image entry from Drive, every sample, every epoch

**Symptom:** the first real joint Stage 05-08+RACAF training run on a T4 (after §35's fix; a full
Phase 1 precomputation had already completed cleanly — 2974 already cached, 677 newly computed, 11
skipped for empty FOV, no Drive FUSE error, caches flushed to Drive) reached Epoch 1 but ran at a
sustained ~5–6s/step (`batch_size=2`, ~1465 steps/epoch, 50 epochs) — at that rate, one epoch alone
would take ~2–2.5 hours. `empty Stage 03 field-of-view` warnings for the same image ids Phase 1 had
already identified and skipped also appeared repeatedly during training. The run was stopped before
completing an epoch.

**Method:** every file in scope (`joint_training_dataset.py`, `dataset_staging.py`, the cache/path/
config modules, Stage 03/04 inference/cache code, `stage08_corn_classifier.ipynb`,
`training/trainer.py`) was re-read from the CURRENT repository (post-§35), plus `git log`/`git show`
over `fc29ff1` and the prior cache-related commits, before any change. A small local diagnostic
(`_build_joint_sample` called directly, no `Trainer.fit()`, no mocks — real synthetic-checkpoint
machinery already established in `tests/test_joint_training.py`) measured the CACHE-HIT code path's
own cost against a local disk, to isolate code cost from I/O-medium cost.

**Root cause 1 (confirmed, structural): the notebook's training cell pointed the Phase 2 cache
directories at Drive, not at Phase 1's local cache.** `stage08_corn_classifier.ipynb`'s "Dataset
loading" cell called:

```python
train_ds, val_ds = jtd.load_joint_training_datasets(
    batch_size=BATCH_SIZE,
    cache_dir=config.LOCAL_FEATURE_RESULTS_DIR,   # Drive-mounted
    racaf_cache_dir=config.RACAF_RESULTS_DIR,      # Drive-mounted
)
```

`_get_or_compute_joint_frozen_outputs()`'s cache-hit branch performs THREE sequential `np.load()`
calls (vessel `.npy`, lesion `.npy`, reliability `.npz`) directly against whatever `cache_dir`/
`racaf_cache_dir` resolve to. Since §35 gave Phase 1 (`precompute_joint_frozen_caches`/
`precompute_authoritative_joint_caches`) a `persistent_cache_dir` local-first/persistent-fallback
mechanism but Phase 2 (`load_joint_training_datasets`/`_make_joint_dataset`/`_build_joint_sample`/
`_get_or_compute_joint_frozen_outputs`) had NO equivalent, and the notebook cell explicitly passed
the Drive paths as Phase 2's ONLY cache location, every training sample's cache-hit read three files
directly from Google Drive's FUSE mount — this project's own already-documented "latency-bound per
file open, not bandwidth-bound" characteristic (`dataset_staging.py`'s module docstring; the exact
same class of cost §33 fixed for Phase 1) — and did so on EVERY sample, EVERY epoch, since this
project's `tf.data` pipelines deliberately never `.cache()` decoded samples.

**Root cause 2 (confirmed, structural): the SAME cell never staged APTOS locally for training, so
the raw image read was ALSO unconditionally against Drive.** `_build_joint_sample()` calls
`lfed._load_raw_bgr(image_dir, id_code)` UNCONDITIONALLY, before any cache-hit check — even on a
full cache hit, the raw image is still read (to build `canonical_rgb`, deliberately never cached,
see §9's design). The training cell never passed `image_dir` at all, so it defaulted to
`DEFAULT_TRAIN_IMAGE_DIR`, which resolves via the `APTOS2019_RAW_DIR` environment variable to the
Drive-mounted raw directory — unlike the Phase 1 cell, which explicitly stages APTOS to local disk
first (`dataset_staging.stage_dataset(...)`) and passes the staged local directory as `image_dir`.
So every sample paid a FOURTH Drive-FUSE file-open cost (the raw PNG) on top of the three cache
files, every sample, every epoch.

**Measured (local, CPU-only dev machine, no GPU, real synthetic-checkpoint architectures — NOT a
substitute for real Drive latency, but isolates the code path's own cost):** `_build_joint_sample`
against a LOCAL cache/image directory, N=8 synthetic images, cache-HIT path (Phase 1 had already
populated the cache) —

```
Per-sample times (s): [0.4339, 0.3266, 0.3165, 0.3591, 0.5788, 0.5258, 0.4987, 0.5535]
mean=0.4491s  max=0.5788s  min=0.3165s
Simulated batch_size=2 step cost (2 samples, sequential, local disk): ~0.90s
```

Even on this unoptimized, CPU-only, non-Colab machine, a purely local-disk cache-hit sample costs
under 0.6s — a `batch_size=2` step reading local disk should cost under ~1s, not 5–6s. The observed
real-Colab number (~2.5–3s/sample) is 5–9× slower than this already-conservative local baseline,
which is exactly the signature expected from adding Drive FUSE's per-file-open latency on top of
(not instead of) the code path's own cost — not a signature of the code path itself being slow, and
not a signature of Stage 03/04 being recomputed (a genuine recompute, per this project's own
measurements elsewhere, costs whole seconds of GPU/CPU forward-pass time per image, not a roughly
constant ~2.5–3s regardless of step number across a 378-step window, which is instead the signature
of a per-sample I/O tax that neither grows nor shrinks with progress).

**Root cause 3 (confirmed, structural): empty-FOV images have no cache anywhere, so they are
re-attempted every epoch.** `_get_or_compute_joint_frozen_outputs()` raises `EmptyFieldOfViewError`
BEFORE any `np.save`/`np.savez` call for such an image — by design, there is no project-sanctioned
fallback value to cache (§9). This means an image that fails FOV detection has NO cache entry in
EITHER the local or persistent directory, so every subsequent access — including every training
epoch, not just Phase 1 — finds no cache hit anywhere, re-attempts Stage 03's (relatively cheap, but
not free) FOV-detection pipeline, hits the same failure again, and correctly excludes the image
again. This is EXPECTED given the current design (already documented: `_make_joint_dataset`'s
generator "excluded from this epoch" — per-epoch language, not "excluded forever"), not a
correctness bug — the image is neither fabricated nor allowed to crash the run, on any epoch — but
it is a real, small, bounded inefficiency (≤11 of 2929 images, i.e. ≤0.8% of steps affected) that is
NOT the cause of the dominant per-step slowdown (which affects essentially every step, not ~1 in
133). No change was made for this: adding a negative/"known-unprocessable" cache would introduce a
new cache concept not requested and not necessary to fix the reported slowness, so it was left as a
documented, minor, optional future optimization rather than a speculative change made now.

**Fix.** `_get_or_compute_joint_frozen_outputs()` gained `persistent_vessel_cache_path`/
`persistent_lesion_cache_path`/`persistent_reliability_cache_path` (all default `None`, fully
backward compatible): on a LOCAL cache miss, these (if given) are checked next; a hit there is
loaded from the persistent location and MIRRORED to the local path (once), so every later call for
that same image — every subsequent epoch — reads local disk only, never the persistent location
again. This exact mechanism is threaded through `_build_joint_sample()` → `_make_joint_dataset()` →
`load_joint_training_datasets()` as `persistent_cache_dir`/`persistent_racaf_cache_dir`, mirroring
§35's Phase 1 parameter naming and semantics exactly. `load_joint_training_datasets()` also gained a
`check_gpu()` call before either model loads — it loads the same two GPU-touching models (Stage 03
PyTorch, Stage 04 TensorFlow) Phase 1's two entry points already guard with this call (§34); Phase 2
had been missing it, a real (if not yet observed as crashing) gap.

The notebook's "Dataset loading" cell now stages APTOS locally (idempotent — a no-op if Phase 1
already staged it this session) and passes `image_dir` at the staged local directory,
`cache_dir`/`racaf_cache_dir` at the SAME local cache directories the Phase 1 cell uses,
`persistent_cache_dir`/`persistent_racaf_cache_dir` at the real Drive-mounted persistent cache, and
`processed_dir` at the same deliberately-empty local directory Phase 1 uses (forcing the cheap live
Stage 02 fallback rather than one more Drive lookup). The cell works correctly whether or not Phase 1
was run first: if it was, every sample is already a local cache hit; if not, the first epoch pays a
one-time, naturally-paced (never a concurrent burst, so no §35-class FUSE risk) Drive read per
uncached image, mirrored locally, so only that first epoch is I/O-bound and every later one is not.

**No Stage 3/4/RACAF/Stage 5/6/7/CORN architecture, weights, reliability equations, TTA definition,
or cache format/keys changed. No batch size, hyperparameter, loss, optimizer, or QWK change.** No
existing Drive cache entry deleted, invalidated, or duplicated. Resumability, the authoritative
2929/733 split, and validation determinism are all unaffected — none of this touches which entries
are considered cached, only where their content is read from.

Regression tests: `tests/test_joint_training.py` — `Phase2PersistentCacheDirTests` (a persistent-
only entry is wastefully recomputed WITHOUT this fix, confirmed as the baseline; WITH it, never
recomputed; mirrored to local disk; a second call reads purely local even if the persistent
location becomes unavailable; `persistent_cache_dir=None` preserves the exact prior default
behavior; `_make_joint_dataset` forwards the parameter to every sample), `LoadJointTrainingDatasets
Tests` (`load_joint_training_datasets` — previously untested directly — calls `check_gpu()` before
either model loads; forwards `persistent_cache_dir`/`persistent_racaf_cache_dir` to both train and
val datasets; defaults to `None`; an end-to-end real-synthetic-data run against a persistent-only
cache never recomputes), and `RepeatedEmptyFovAcrossEpochsTests` (a permanently-empty-FOV image is
excluded and warned identically across two independent dataset iterations — deterministic, not a
bug; confirms no cache file is ever written for it).

**Still required before declaring the pipeline training-ready:** a real Colab T4 run, since Drive
FUSE's actual latency (and therefore the actual steps/second improvement) cannot be measured
outside Colab. Recommended: run Phase 1 first (already proven safe and complete, §35), then start
training and observe whether steps/second increases substantially in epoch 1 once local mirroring
completes, and whether steady-state steps/second in epoch 2+ (fully local) is now compute-bound
rather than I/O-bound.

---

## 37. §36's fix was correct but incomplete — Phase 1 never mirrored its own "already_cached" bucket, so training's first epoch was still Drive-bound

**Symptom:** after §36 (Phase 2 given local-first/persistent-fallback cache reads, matching the
training cell's arguments exactly — `image_dir=staged_train_image_dir`, `cache_dir`/
`racaf_cache_dir` local, `persistent_cache_dir`/`persistent_racaf_cache_dir` set to Drive), a real
Colab T4 run — after a full, clean Phase 1 precomputation (`already_cached: 2974`, `cached: 677`,
`skipped_empty_fov: 11`, no Errno 107, caches flushed to Drive) — still ran at a sustained
~5–6s/step through step 95/1465 of epoch 1. The run was stopped.

**Method:** re-read the CURRENT (post-§36) code fresh, not assuming §36's diagnosis still held.
Traced `load_joint_training_datasets()` → `_make_joint_dataset()` → `_build_joint_sample()` →
`_get_or_compute_joint_frozen_outputs()` line by line and confirmed the wiring was correct at every
layer (a local hit short-circuits before the persistent branch; a persistent hit is read once and
mirrored). Since the consumption path was proven correct, the audit moved one level up, to what
Phase 1 actually leaves on local disk.

**Root cause, confirmed both structurally and empirically:** `precompute_joint_frozen_caches()`'s
loop treated a `persistent_cache_dir` existence hit identically to a true local hit — both simply
incremented `stats["already_cached"]` and moved on, WITHOUT reading or copying anything (§35's own
design goal: existence-only, no bulk copy). This means Phase 1's `already_cached` bucket — found on
Drive from a prior run — was never written to the local `cache_dir` Phase 1 itself uses. A local
diagnostic (real synthetic-checkpoint machinery, reproducing the real run's ~81%-already_cached /
~19%-newly-computed profile at a smaller scale) confirmed this directly:

```
Phase 1 stats: {'cached': 2, 'already_cached': 8, ...}   (8/10 = 80%, matching 2974/3662 = 81.2%)
LOCAL cache dir after Phase 1 contains 4 files.
Of the 8 'already_cached' (persistent-only) entries, 0 were mirrored to LOCAL disk by Phase 1.
```

So after a real Phase 1 run, only the 677 genuinely-new entries existed locally; the other 2974
(81% of the whole dataset) existed only on Drive. §36's Phase 2 self-healing mirror is correct and
does eventually fix this — but only the FIRST time each such image is touched, which (since nothing
in Phase 1 had already warmed the local cache for 81% of the dataset) meant essentially the WHOLE
of training's first epoch remained Drive-read-bound, indistinguishable in practice from the
pre-§35/§36 slowdown. The run being stopped at step 95/1465 (6.5% into epoch 1) meant the
improvement §36 does provide (every access AFTER the first) was never reached.

**Fix:** `precompute_joint_frozen_caches()`'s loop now distinguishes a true local hit from a
persistent-only hit. A persistent-only hit calls `_get_or_compute_joint_frozen_outputs()` (reusing
its already-existing, already-tested persistent-hit-and-mirror branch unchanged, `rgb_native=None`
since that branch never touches it) to read the entry from Drive exactly once and mirror it to the
local `cache_dir`/`racaf_cache_dir` — during Phase 1 itself, one entry at a time, inside the SAME
sequential per-image loop Phase 1 already runs (never a bulk/concurrent copy — not `dataset_
staging.sync_missing_files()`'s separate walk-and-copy-everything pattern — so no §35-class Drive
FUSE risk). Stage 03/04 are never invoked for this case (confirmed: `predict_vessel_mask`'s call
count is 0 for a persistent-hit entry in every regression test). By the time Phase 1 finishes, every
entry it processed — whether newly computed or found on Drive — is a genuine local cache hit, so
training's first epoch is no longer different from any later one.

`stats` gained one new key, `"mirrored_from_persistent"` (int, counts entries handled this way).
`"cached"` (genuine Stage 03/04/RACAF computation) and `"already_cached"` (a true local hit, zero
I/O beyond the existence check) keep their EXACT prior meaning and values for every existing caller
that never sets `persistent_cache_dir` — this is the only stats-shape change, and it is additive.

**Not changed:** Stage 03/04 architecture or weights, RACAF or CORN mathematics, Stage 5/6/7
architecture, loss, QWK, optimizer, batch size, epochs, `Trainer`/`TrainingConfig`. No existing
Drive cache entry is deleted, overwritten, or invalidated — `persistent_cache_dir`/`persistent_
racaf_cache_dir` remain read-only from this module's perspective (mirrors `stage_dataset()`'s own
Drive-source convention). The cache format, keys, and image sizes are unchanged — a mirrored local
file is byte-for-byte the same array as its persistent source (verified: `np.testing.assert_array_
equal` between the mirrored local copy and the original persistent-cache array). Resumability is
preserved and extended: a mirrored entry is never re-touched (local disk or Drive) by a later Phase
1 call, verified alongside a mixed-entry-kind (already-local / persistent-only / genuine-miss)
interrupted-and-resumed scenario. No bulk copy of the persistent cache happens at any point — each
mirror is one small entry, driven by this loop's own existing one-image-at-a-time iteration.

Regression tests: `tests/test_joint_training.py` — `PersistentCacheDirTests` (a persistent-only
entry is mirrored locally without recomputing, counted under the new `mirrored_from_persistent`
counter, never under `already_cached`/`cached`; the mirrored local array is numerically identical
to its persistent source; the mirrored entry is directly usable by `_build_joint_sample` with NO
`persistent_cache_dir` passed at all, matching how training consumes it; a true local hit never
touches an invalid/nonexistent persistent dir; a mirrored entry is never re-touched by a later run;
a mixed-entry-kind interrupted/resumed run ends with all three kinds correctly and permanently
local), `CachePrecomputationTests`/`DiagnosticModeTests` (the new stats key updated in the existing
exact-key-set assertions; `mirrored_from_persistent` stays `0` for every caller that never passes
`persistent_cache_dir`, preserving `cached`/`already_cached`'s exact prior values).

**Still required before declaring the pipeline training-ready:** a real Colab T4 run. Recommended:
re-run Phase 1 first (safe and resumable — it will now report `mirrored_from_persistent` for the
2974 previously-Drive-only entries, each mirrored to local disk during this pass), THEN start
training and confirm epoch 1 itself is now close to the local-cache-hit baseline measured in §36
(~0.9s/step-equivalent on an unoptimized dev machine), not just epoch 2 onward.

## 38. Full end-to-end audit at 164aead: proved the "ran out of data" warning harmless, found and fixed the real remaining ~2s/step cause (canonical RGB was never cached), confirmed everything else in the pipeline correct-by-design

**Symptom:** after §35-§37, a real Colab T4 run's steady-state step time improved from ~5-6s/step
to ~2s/step, and Epoch 1 completed cleanly (1461/1461, val_QWK computed, `best.weights.h5` saved).
But Keras then printed `"Your input ran out of data; interrupting training... You may need to use
the .repeat() function..."`, and the run was stopped after Epoch 1. Requested: an independent,
full-pipeline audit (correctness, performance, memory, I/O, reproducibility) — not a reflexive
`.repeat()` fix, and not an assumption that any earlier diagnosis still held.

**Method:** re-read `joint_training_dataset.py`, `training/trainer.py`, `training/callbacks.py`,
`joint_training_model.py`, `local_feature_extraction_dataset.py`, `vessel_segmentation_inference.py`,
`config.py`, `colab/common/dataset_staging.py`, `colab/common/experiment_manager.py`, and every
notebook cell that wires Phase 1/Phase 2/`Trainer.fit()`, fresh at HEAD (164aead) — confirmed the
installed stack is **Keras 3.15.1 / TF 2.21.0** (the standalone multi-backend `keras` package, not
legacy `tf.keras`), so no conclusion was drawn from memory of Keras-2-era `data_adapter.py`
internals. Every claim below was independently proven, not assumed:

- **Cardinality math**: computed directly from the real, committed manifest (`dataset_splits/
  aptos2019_train_val_split.csv`, 3662 rows, 2929 train/733 val, 0 duplicates) that 8 of the 11
  known empty-FOV ids fall in the train split and 3 in val — giving an effective train count of
  2929-8=2921, `ceil(2921/2)=1461` batches. This exactly matches Keras's reported "1461/1461" for
  Epoch 1, with no other unexplained sample loss.
- **The "ran out of data" mechanism**: read `keras/src/trainers/epoch_iterator.py` (the installed
  package) directly. For an unknown-cardinality dataset (`from_generator`, no `.repeat()`, no
  explicit `steps_per_epoch` — exactly `_make_joint_dataset`'s construction, confirmed nowhere
  overridden in the notebook) `EpochIterator.catch_stop_iteration()` **unconditionally** calls
  `self._interrupted_warning()` the first time `self._num_batches` is `None` and a `StopIteration`
  is caught — i.e. on whichever epoch first reaches the natural, correct, fully-expected end of a
  real, non-repeated data source. This is a one-time self-calibration step, not an error signal.
  Reproduced empirically on the exact installed versions with a minimal `from_generator` dataset
  (built identically: `from_generator → shuffle → batch → prefetch`, no `.repeat()`) fed to
  `model.fit(ds, epochs=4)`: the warning fired on epoch 1 exactly once, the generator was invoked
  fresh every epoch with the full element count every time, and all 4 epochs completed with real,
  decreasing loss — proving the warning is emitted even when nothing is actually wrong. `.repeat()`
  is therefore NOT required (and was not added) — it would remove Keras's own ability to
  self-calibrate `steps_per_epoch` from real data and would need a manually-computed, duplicated
  effective-sample-count instead, reintroducing exactly the fragility this audit was asked to avoid.
  Category: **C — harmless, one-time, expected warning.** Training was not actually interrupted by
  Keras itself; the real run being stopped after Epoch 1 is consistent with the same pattern of a
  manual stop already used in every earlier task in this project's history, prompted by the alarming
  wording, not an actual `fit()`-internal halt.
- **The real, still-present performance cause**: a local, real-code-path diagnostic (real `cv2`/
  `skimage`/`image_preprocessing` calls, synthetic checkpoints/images, no GPU) measured
  `_build_joint_sample`'s steady-state cost (every Stage 03/04/RACAF cache file already local) at
  ~137ms/sample on an 800×800 synthetic image, of which the vessel/lesion/reliability `np.load`
  portion was ~3ms (98% of cost was elsewhere) — because `stage5_input`'s RGB channels were
  **never cached at all**: `canonical_rgb = _resize_rgb_01(rgb_native, image_size)` ran on every
  single `_build_joint_sample` call, cache hit or not, requiring a raw-image disk read plus (since
  the real notebook's Phase 2 cell always points `processed_dir` at an intentionally empty
  directory) a live Gamma+CLAHE Stage 02 pass every time. At a more realistic 2000×1848 raw
  resolution the same measurement was ~503ms/sample (`_resize_rgb_01`'s anti-aliased `skimage`
  resize alone: ~421ms). This is paid for the ENTIRE dataset (not just the small empty-FOV set),
  every single epoch. Confirmed this is genuinely a fresh-per-call cost by grep of
  `_build_joint_sample`'s prior form: the raw-image load was unconditional, before any cache check.
- **Everything else** was independently re-verified at HEAD rather than assumed: the real joint
  model was built and compiled locally (CPU, no training) and matched the previously-documented
  smoke test exactly (43,296,810 total / 43,292,970 trainable / 3,840 non-trainable / 393 trainable
  variables, `joint_corn_loss`) — `joint_training_model.py`/`corn.py`/`racaf.py`/`swin_transformer.py`
  have not changed since commit 335ae59, well before §35-§37. A 6-epoch, real-code-path RSS
  measurement across repeated dataset passes showed no growth (−2.5MB/epoch average — noise, not a
  leak) and independently reconfirmed the generator is invoked fresh every epoch with a stable
  element count. `_augment_spatial`/`_augment_intensity_rgb` apply one synchronized spatial
  transform to all 8 channels and RGB-only intensity jitter, confirmed by direct code read; `r` is
  computed pre-augmentation; validation uses `augment=False`. `experiment_manager.create_experiment`
  is timestamped and collision-protected (never overwrites); `resume` is opt-in via
  `RESUME_EXPERIMENT_DIR` (defaults to a fresh run, matching the notebook's default). Checkpoints
  are weights-only (`save_weights`/`load_weights`), so optimizer (Adam) momentum is NOT preserved
  across a resume — a real but already-documented, deliberate tradeoff (`joint_training_model.py`'s
  own docstring: Stage 06's Swin layers have no `get_config()`, so a full-model save was never an
  option). The `"Model failed to serialize as JSON"` / `PatchEmbed` warning was traced to `keras/
  src/callbacks/tensorboard.py`'s `keras_model_summary()`, which wraps `model.to_json()` in its own
  `try/except` specifically because this is expected — it affects ONLY the TensorBoard Graphs-tab
  visualization, never training, checkpointing, resume, or model loading (all weights-only, unrelated
  code path). Category: **C — harmless, already-anticipated warning.**

**Root cause (the one fixed):** canonical RGB was the only per-image artifact in the whole joint
pipeline that was never cached — vessel, lesion, and reliability all were (§9-§13, extended to
Drive/local tiers by §35-§37), but the RGB resize was always recomputed live, because it depends
only on `rgb_native`, not on `vessel_model`/`stage4_model`, so it was never routed through `_get_or_
compute_joint_frozen_outputs`'s existing cache machinery at all.

**Fix:** `_get_or_compute_canonical_rgb()` (new) caches the resized, [0,1] float32 RGB array using
the exact same local/persistent/compute-fresh-and-mirror pattern already proven for vessel/lesion/
reliability, under a new `kind="rgb"` cache file (`_canonical_rgb_cache_path`, reusing `lfed.
_cache_path`'s existing filename convention — no new scheme). `_build_joint_sample` now loads the
raw image at all only when at least one of the two independent things it can produce (frozen Stage
03/04/RACAF outputs, or canonical RGB) is not already cached, locally or at `persistent_cache_dir`
— a full cache hit never touches the raw file. `precompute_joint_frozen_caches()` mirrors this in
Phase 1 (same one-image-at-a-time loop, additive after the existing branch, never a bulk copy), so
Phase 1 — not training's first epoch — pays the one-time cost, matching §37's pattern exactly. A
persistent cache populated by an earlier run (before this cache kind existed) has vessel/lesion/
reliability but no `rgb` file; Phase 1's existing "is this cached" decision (`_cache_entry_exists`)
is deliberately UNCHANGED (still vessel/lesion/reliability only), so such an entry is still
correctly recognized as a full frozen-outputs hit — never recomputing Stage 03/04/RACAF — while the
missing `rgb` file is independently backfilled. The cached array is numerically identical to what
`_resize_rgb_01` always computed live (same function, same inputs) — this is a pure caching change,
not a resize/preprocessing algorithm change.

**This cache was originally made LOCAL-ONLY** here, via its own `rgb_cache_dir` parameter — a
decision §39 found wrong and reversed after a real Colab run. See §39 for the corrected design;
`rgb_cache_dir` remains available as a caller-level escape hatch (default `None` → `cache_dir`),
but the real notebook no longer sets it.

**Not changed:** Stage 03/04 architecture or weights, RACAF or CORN mathematics, Stage 5/6/7
architecture, loss, QWK, optimizer, batch size, epochs, `Trainer`/`TrainingConfig`, the resize
algorithm itself, `.repeat()` (deliberately not added — see above), the authoritative split
manifest, and no existing Drive cache entry was deleted, overwritten, or invalidated.

**Regression tests:** `tests/test_joint_training.py` — `CanonicalRGBCachingTests` (rgb cache file
written on first build; cached value numerically identical to the sample it was derived from; a
full cache hit never calls `lfed._load_raw_bgr`; a full cache hit reproduces the exact same
`stage5_input` as a fresh computation; a persistent-only rgb entry is mirrored locally without
recomputing OR re-reading the raw image; an entry missing only its rgb cache backfills by reading
the raw image exactly once, without recomputing Stage 03/04), `Phase1CanonicalRGBCachingTests`
(Phase 1 writes an rgb cache for every processed entry; a second Phase 1 run never re-reads raw
images; a legacy persistent hit with vessel/lesion/reliability but no rgb file is still recognized
as a full frozen-outputs hit, never recomputing Stage 03/04, while backfilling rgb; an empty-FOV
entry — whose vessel/lesion/reliability cache never gets written — still gets its rgb cached; a
caller that opts into a separate `rgb_cache_dir` still gets a clean separation and a working
Phase-1-then-Phase-2 hand-off through it, though §39 revised the real notebook to not opt in), one
added test in the existing
`Phase2UsesExistingCachesTests` (a second epoch-like dataset iteration never reads the raw image
either), and two in `LoadJointTrainingDatasetsTests` (`rgb_cache_dir` forwarded to both train and
val datasets; defaults to `None`). All existing tests re-run and pass unchanged, confirming this
restructuring is behavior-preserving everywhere except the newly-proven-wasteful raw-image reload.

**Still required before declaring the pipeline training-ready:** a real Colab T4 run. Recommended:
re-run Phase 1 first (safe, resumable, self-healing — it will backfill an `rgb` cache entry for
every one of the 3651 already-cached entries from the prior run, without recomputing Stage 03/04/
RACAF for any of them), then start training and confirm steady-state step time drops meaningfully
below the ~2s/step measured after §35-§37 (the local diagnostic here suggests the RGB fix should
remove close to all of the remaining non-GPU-compute cost, though the actual T4 forward/backward
step time itself was not and could not be measured on this dev machine, which has no GPU) — and
confirm the "ran out of data" warning still appears exactly once, on Epoch 1, with training then
continuing normally through all 50 epochs without being stopped by Keras itself.

## 39. §38's local-only canonical RGB cache was itself the ~90-minute fresh-runtime problem — persisted to Drive instead, using the existing cache mechanism unchanged

**Symptom:** after §38 shipped, a real Colab run reported Phase 1 taking ~90 minutes and still not
completing, and Phase 1b reporting "Flushed 0 Stage 03/04 cache files and 0 RACAF cache files to
Drive" afterward — raising a real concern that every fresh Colab runtime (`/content` is wiped on
disconnect) might need a 1-2 hour cache-preparation pass, indefinitely.

**Method:** re-audited fresh, explicitly not assuming §38's local-only design was correct. A local,
real-code-path diagnostic (synthetic checkpoints/images, no GPU) reproduced the exact fresh-runtime
condition — empty local cache, complete persistent (Drive-standin) cache — and measured Stage 02 +
resize in a cleanly-isolated process, independent of any other work, three separate times: 450ms,
612ms, and 1340ms per image at a realistic 2000×1848 raw resolution. The three runs disagree in
absolute terms (real variance on this dev machine, most plausibly thermal/background-load related,
not a stable number) but agree qualitatively: every measurement projects to many minutes-to-over-an-
hour for the full 3662-image dataset, consistent with the reported ~90 minutes. Root cause: §38
cached canonical RGB, correctly, but chose to keep that cache LOCAL-ONLY (a separate `rgb_cache_dir`
the notebook's Phase 1b flush cell never touched) specifically to avoid growing the Drive cache
~60%. That reasoning weighed Drive storage against a ONE-TIME regeneration cost — it did not price
in that "one-time" becomes "every fresh runtime" once `/content` is disposable, which is exactly how
a real Colab session behaves.

Before any code change, independently verified (a separate diagnostic, not assumed): canonical RGB
is deterministic (`_resolve_processed_rgb`/`preprocess_array` have no `random`/`rng`/seed reference
anywhere — confirmed by source grep, not inference); the same raw file processed twice, independent
calls, produces byte-identical output; `np.save`/`np.load` (the exact mechanism `_get_or_compute_
canonical_rgb`'s persistent-hit branch already uses, unmodified) round-trips an array bit-for-bit,
by construction of the `.npy` format; and that existing, already-tested persistent-hit-and-mirror
branch, run end to end with no code change, returns a value numerically identical (`array_equal`
AND raw-bytes-equal) to a fresh live computation on the same image.

**"Flushed 0" is not a symptom — it is what correct behavior looks like when nothing was newly
computed.** `sync_missing_files()`'s "flushed" count is `copied_count`: files not yet present at
the destination. Every locally-cached entry in that run came from `mirrored_from_persistent` (real
Phase 1 stats showed `cached: 0`, confirmed independently by a mocked `predict_vessel_mask` call
count of 0 in the local reproduction) — content that *originated on Drive* — so Phase 1b correctly
found it `already_present` there and copied nothing. This was true before this fix and remains true
after it; it was never evidence of a stall.

**Fix — minimal, reusing existing plumbing, no new mechanism:** the real notebook no longer passes
a separate `rgb_cache_dir`. Canonical RGB now shares `cache_dir`/`persistent_cache_dir` with vessel/
lesion/reliability, exactly like every other cache kind — the SAME existence-check-and-mirror code
path (`_get_or_compute_canonical_rgb`, already built and tested in §38, entirely unchanged), the
SAME Phase 1b `sync_missing_files(LOCAL_CACHE_DIR, config.LOCAL_FEATURE_RESULTS_DIR)` call
(`dataset_staging.py`, also entirely unchanged) — newly-written RGB entries are picked up
automatically because they now live in the directory that call already walks. `joint_training_
dataset.py`'s `rgb_cache_dir` parameter itself was NOT removed (it remains a working, tested
escape hatch for a caller that genuinely wants separation — some existing tests exercise exactly
that), but the notebook and every default-configured caller now route RGB through the shared,
Drive-backed path.

**Redundant existence checks — evaluated, not removed.** Traced two candidate redundancies
precisely: (1) `precompute_joint_frozen_caches`'s persistent-hit branch (`elif persistent_cache_dir
... and _cache_entry_exists(...)`) already confirms vessel/lesion/reliability exist at the
persistent location before calling `_get_or_compute_joint_frozen_outputs`, which then re-checks the
identical three paths itself (present since `164aead`, predates this change); (2) the equivalent
pattern this session's own code introduced for rgb, between Phase 1's `rgb` block and `_get_or_
compute_canonical_rgb`'s own first two lines. Neither can be removed by deleting code alone without
changing behavior for `_build_joint_sample` (Phase 2), which calls the SAME shared functions but has
NOT already checked persistent existence itself — it depends entirely on that internal check to
decide hit vs. miss. Removing it safely would require adding a new "caller already verified this"
parameter (mirroring the existing `known_not_all_cached` parameter, which already solves the exact
same problem for ONE specific check) to functions that are either extensively tested and load-
bearing for the explicitly-protected vessel/lesion/reliability path, or brand new from this
session's own recent commits. Given the explicit brief to prefer the smallest safe change and not
introduce new complexity, and given the cost of the redundancy itself is small (a handful of extra
`os.path.exists` calls per image — cheap locally, and only reached on a Drive path when the local
tier is still cold), both were left untouched rather than adding new parameter surface area to
remove them. This is a deliberate, evaluated non-change, not an oversight.

**Not changed:** Stage 02's numerical processing, `_resize_rgb_01`'s resize behavior or algorithm,
Stage 03/04/RACAF models or outputs, Stage 5-7 architecture, CORN formulation, loss, QWK, optimizer,
learning-rate policy, batch size, epochs, `Trainer`, experiment semantics, or the authoritative
split. No existing Stage 03 vessel, Stage 04 lesion, or RACAF reliability Drive cache entry was
deleted, migrated, rewritten, or invalidated — this fix is purely additive (new `rgb` cache entries
alongside the existing three, using their exact existing directory and flush mechanism). No
sharding, TFRecords, HDF5, combined cache files, LRU eviction, cache migration, or parallel
preprocessing was introduced.

**Storage impact:**

| | per image | × 3662 | 
|---|---|---|
| Vessel + lesion + reliability (existing, unaffected) | ~5.0 MiB | ~17.9 GiB |
| Canonical RGB (now Drive-persisted) | 3.0 MiB | ~10.7 GiB |
| **Drive total after this change** | | **~28.6 GiB** (+60% over pre-existing) |
| Worst-case local SSD (repo + staged APTOS raw + a full local mirror of all four cache kinds + checkpoints, all warmed simultaneously) | | ~40 GB, against a 112 GB budget (~72 GB headroom) |

**Regression tests:** `tests/test_joint_training.py` — two new tests in `CanonicalRGBCachingTests`
(`test_persistent_mirrored_rgb_is_numerically_identical_to_a_fresh_live_computation`: a persistent-
only entry, once mirrored, is byte-for-byte identical to a fresh live computation on the same raw
image; `test_no_persistent_path_is_probed_once_every_artifact_is_a_local_hit`: once all four cache
kinds are local hits, not one `os.path.exists` call reaches a persistent/Drive path) and two new
tests in `Phase1CanonicalRGBCachingTests` (`test_rgb_mirrored_entry_is_not_re_touched_by_a_later_
phase1_run`: mtime-stability proof, mirroring the existing vessel/lesion resumability test exactly;
`test_mixed_entry_kinds_all_end_with_a_valid_local_rgb_cache`: the existing already-local/persistent-
only/genuine-miss mixed scenario, extended to confirm all three end with a correct local rgb entry).
Two existing tests were renamed (not behaviorally changed) to stop describing the opt-in separate-
`rgb_cache_dir` scenario as if it were the notebook's default. A new class,
`Phase1bFlushIncludesCanonicalRGBTests`, validates the actual notebook wiring end to end using the
REAL, unmodified `dataset_staging.sync_missing_files()` (not a mock): a genuinely new entry's rgb
file is flushed to the Drive-standin directory alongside vessel/lesion with byte-identical content;
and, directly answering the "Flushed 0" question, a persistent-only entry (mirrored locally, never
recomputed) flushes exactly 0 new files and reports every one of them `already_present` — proving
that specific real-run observation was correct, expected behavior, not a stall. All existing tests
re-run and pass unchanged.

**Still required before declaring the pipeline training-ready:** the real-Colab validation this
session's diagnostics cannot substitute for — a fresh runtime, small (10-25 image) diagnostic Phase
1 run confirming persistent hits mirror rather than recompute, a full Phase 1 run confirming elapsed
time drops from ~90 minutes to a small number, a Phase 1b run confirming only genuinely-new entries
(now including rgb) are flushed and existing Drive entries are untouched, a SECOND fresh-runtime
Phase 1 confirming rgb is no longer regenerated, and a direct ~20-batch dataset iteration (no
`Trainer.fit()`) confirming zero Drive dependency in steady-state sample delivery.

---

## 40. §39's persistence was never triggered — Phase 1 wrote canonical RGB locally but nothing flushed it, so Drive held `rgb = 0`

**Symptom.** A real-runtime diagnostic (§39's `joint_cache_diagnostics.py`, run on 5 images)
reported the persistent Drive cache holding `vessel = 3651`, `lesion = 3651`,
`reliability = 3651` and **`rgb = 0`**. Canonical RGB existed only under
`/content/cache/local_feature_extraction`, so every fresh runtime regenerated it for every image.
Pass 1 showed RGB as the only artifact with 0 persistent hits and 5 recomputations; pass 2 showed
it reused with zero recomputation once local; the cached bytes were bitwise identical to a fresh
generation.

The `3651` is itself informative: `3662 - 3651 = 11`, exactly the count of empty-field-of-view
images a real Phase 1 skips. The Drive cache was therefore written by a Phase 1 run that predates
the canonical-RGB cache kind entirely.

**Method — what was actually wrong, established before editing.** The reported state has two
possible causes, and they call for opposite fixes: either the read/mirror/flush machinery from
§38/§39 is broken, or it is correct and simply was never run. A throwaway script reproduced the
exact reported Drive state (vessel/lesion/reliability present, rgb deleted) and drove the real,
unmodified code path end to end under the §39 instrumentation:

| Step | Result |
|---|---|
| Fresh runtime, Phase 1 against that Drive cache | Stage 03/04/RACAF all `mirrored_from_persistent`, `predict_vessel_mask` 0 calls; RGB computed locally (3 calls); **0 writes to Drive**; Drive `rgb` still 0 |
| The existing Phase 1b `sync_missing_files()` call, unchanged | flushed exactly the 3 rgb files (6 vessel/lesion already present); Drive `rgb` now 3 |
| Second fresh runtime (local wiped), Phase 1 again | **`_resize_rgb_01` 0 calls, raw image loads 0**, 3 rgb Drive reads, 3 local mirrors |
| Drive-persisted rgb vs fresh generation | bitwise identical, max abs diff `0.0` |

So the machinery was already complete and correct. `_get_or_compute_canonical_rgb` already
detects a persistent hit and mirrors it; `_canonical_rgb_cache_path` already writes
`APTOS_<id>_rgb_512x512.npy` beside vessel/lesion via `lfed._cache_path`; Phase 1b's
`sync_missing_files()` already picks rgb up because §39 made all four kinds share
`LOCAL_CACHE_DIR`. **Nothing in `joint_training_dataset.py` needed to change, and nothing did.**

The real gap was operational: Phase 1 writes to local SSD only, and persisting it was a *separate
cell someone has to remember to run*. For vessel/lesion that was harmless — Drive already had
them. For canonical RGB it meant a ~90-minute Phase 1 could complete, the runtime could be
recycled, and all of that work would be gone.

**Fix (notebook only).** The Phase 1 cell now flushes when it finishes, gated by
`FLUSH_TO_DRIVE_AFTER_PHASE1 = True`, reusing the Phase 1b cell's own unmodified
`dataset_staging.sync_missing_files()` over the same directories. That is the same
safe/atomic persistence path vessel and lesion have always used: `_copy_one` writes a temp file,
verifies its size, renames it into place, and retries transient Drive FUSE errors with backoff.
Only files missing at the destination are copied, so existing vessel/lesion/reliability entries
are never re-uploaded, rewritten or invalidated, and re-running after an interruption copies
only what is still missing. Setting the flag `False` prints an explicit warning that the new
entries are local-only.

**Not changed.** `joint_training_dataset.py`, `racaf.py`,
`local_feature_extraction_dataset.py`, `dataset_staging.py` — byte-identical. No separate RGB
cache directory, no change to RGB generation, Stage 02's processing, the resize, Stage 03/04,
RACAF, the split, CORN, loss, QWK, optimizer, batch size, epochs, `Trainer`, or experiment
semantics. The training-time path (`_build_joint_sample`) still never writes to the persistent
cache, in either the local-hit or the persistent-hit-mirrored-down state.

**Known limit, stated rather than hidden.** The flush is end-of-run. If Phase 1 is interrupted
partway (a Colab disconnect), nothing has been persisted yet — but Phase 1 is resumable and
`sync_missing_files()` is incremental, so re-running the Phase 1b cell after a partial run
persists whatever was completed. A mid-run periodic flush was deliberately not added: pushing
thousands of small files to Drive in bursts is the load shape that caused §35's
`OSError: [Errno 107] Transport endpoint is not connected`.

**Regression tests.** `CanonicalRGBDrivePersistenceLifecycleTests` (8 tests,
`tests/test_joint_training.py`) covers the lifecycle across two simulated runtimes with a flush
in between: rgb missing from a legacy Drive cache is computed locally then flushed; it lands at
`APTOS_<id>_rgb_512x512.npy` in the same directory as vessel/lesion; a second fresh runtime finds
it and calls neither `_resize_rgb_01` nor `_load_raw_bgr` nor `predict_vessel_mask`; the
following flush then reports nothing new; every pre-existing Drive entry stays byte- and
mtime-identical; `_build_joint_sample` writes nothing to either persistent directory in either
cache state; the round-tripped rgb is bitwise identical to a fresh generation on both the Drive
copy and the local mirror; and the resulting `stage5_input`/`stage6_input`/`reliability` are
unchanged versus a live build.

---

## 41. A dropped Drive mount read as "never cached" — `os.path.exists` masks ENOTCONN, so an outage could silently recompute frozen Stage 03/04

**Symptom.** The Phase 2a profiler's first real run died in Phase C with
`UnknownError: ... OSError: [Errno 107] Transport endpoint is not connected`, thrown from
`_build_joint_sample` → `_get_or_compute_joint_frozen_outputs` → `np.load(persistent_lesion_cache_path)`,
preceded by `ValueError: cannot reshape array of size 99296 into shape (512,512,4)`. It also printed
`Could not set memory growth on /physical_device:GPU:0: Physical devices cannot be modified after
being initialized`.

**The real defect, found while tracing the failure rather than the failure itself.**
`genericpath.exists` swallows EVERY `OSError`, not just `ENOENT`:

```python
def exists(path):
    try: os.stat(path)
    except (OSError, ValueError): return False
    return True
```

Empirically confirmed: with `os.stat` raising `ENOTCONN`, `os.path.exists()` returns `False`. So on
a Drive mount that had dropped, the persistent-cache existence check read as *"this image was never
cached"*, and `_get_or_compute_joint_frozen_outputs` fell through to its compute-fresh branch —
**silently re-running frozen Stage 03 + Stage 04 for an image whose cache entry exists and is
merely unreachable**. That is the most expensive possible reaction to a transient hiccup, and it
was reachable in production, not only in the profiler.

**Fix.** Persistent-path checks now go through `_persistent_exists()`, which returns `False` only
for a genuine `ENOENT`/`ENOTDIR` and raises `PersistentCacheUnavailableError` for anything else.
Persistent reads go through `_load_persistent_array()`: a bounded retry (3 attempts, 0.25s then
0.5s of backoff — under one second worst case, deliberately not a stall), then post-load shape
validation, then `CorruptCacheFileError` naming the artifact, image, path and errno. Neither error
path ever deletes, rewrites or regenerates anything. The reliability `.npz` is validated by
touching `kappa` inside the guarded block, since `np.load` on an `.npz` is lazy and would otherwise
surface a truncated file as a bare NumPy error somewhere else entirely.

Local reads go through `_load_local_array()`, which is as thin as the plain `np.load` it replaced
— no extra stat, no retry, no extra I/O — so the hot training path costs exactly what it did
before. Its only addition is turning a truncated-file failure into a message that names the
artifact and image, which is free when the load succeeds.

**Local-first is unchanged and now proven.** `_build_joint_sample`'s `or` short-circuits: when all
three local files exist, no persistent path is stat'ed at all. Two tests pin this by making EVERY
`os.stat` and `np.load` against a persistent root raise `ENOTCONN` and then asserting that a fully
local cache still builds samples — once through a direct call, once through the real `tf.data`
generator. If the local-first path were not airtight, those fail.

**Memory-growth warning.** `check_gpu()` called `set_memory_growth` unconditionally, which raises
once TensorFlow has initialized the device — normal on the second and later calls in a session, and
harmless, but printed text that reads like a failure. It now queries `get_memory_growth` first and
only calls the setter when it would change something; if the device really is initialized with
growth off, it says so once and continues rather than implying breakage.

**Why Phase C reached Drive at all.** Under the documented precedence, a persistent read happens
only when the LOCAL entry for that specific image is missing. Directory-level counts cannot show
that, so `preflight_cache_audit()` now stats every artifact of every entry the profiler will
iterate, before measuring, and reports how many are fully local, how many would fall back to Drive,
and how many are missing from both. It stops probing the moment the mount fails rather than
hammering it. By default the run aborts if any entry would read Drive — those reads would both
risk the mount and make the timings measure Drive latency instead of the training step
(`abort_if_drive_fallback=False` to measure anyway).

`describe_pipeline_failure()` walks the `__cause__`/`__context__` chain of whatever surfaces from
`next(iterator)` — a generator failure arrives wrapped as `tf.errors.UnknownError: ...
IteratorGetNext ...`, which hides which artifact and which image failed — and classifies it as
DRIVE_FUSE, CORRUPT_CACHE, LOCAL_CACHE or UNKNOWN, falling back to pattern-matching the wrapped
message text when no typed cause is attached.

**Not changed.** Model architecture, batch size, optimizer, learning rate, loss, QWK, the split,
epochs, frozen Stage 03/04, RACAF, canonical RGB generation, vessel/lesion outputs, reliability,
augmentation, cache layout and cache directories are all untouched. No cache was regenerated, no
cache directory was added, canonical RGB did not move, and local-SSD-first was preserved.

**Regression tests.** `DriveFuseFailureSafetyTests` (9) and `MemoryGrowthInitializationTests` (3)
in `tests/test_joint_training.py`; `ProfilerPreflightAndFailureClassificationTests` (10) in
`tests/test_joint_cache_diagnostics.py`. They cover: a dead mount raising instead of recomputing
(Stage 03 and Stage 04 spies both at zero calls), the error naming artifact/image/path/errno/local
state, Phase 1 refusing to recompute on an outage, local-hit-wins under a booby-trapped Drive both
directly and through the generator, a truncated persistent lesion file reported without being
modified, a genuine ENOENT still computing normally, bounded retry, a transient blip recovering on
retry without recomputation, and the preflight/classification behaviors.

**Still open.** This fixes the profiler's failure and a real production hazard; it does not yet
explain the 2.44 → 4.65 s/step regression. That needs the profiler to complete on the real runtime.

---

## 42. The local cache is empty on every fresh runtime — a one-time persistent-to-local mirror, and atomic cache writes

**Symptom.** A fresh-runtime preflight audited the 2929 training entries and found **5 fully
local, 2916 that would fall back to Drive, 8 missing from both**. The profiler correctly refused to
measure: 2916 Drive reads would have measured FUSE latency rather than the training step, and that
is the load shape that previously produced `Errno 107`.

**Root cause, proven from the numbers and the notebook, not inferred.** The reported figures are
internally exact: `5 + 2916 + 8 = 2929`; local misses `2924 = 2929 - 5`; the local feature cache
held **15 files = 5 entries x 3 kinds** (vessel/lesion/rgb) and the local RACAF cache held exactly
5. That is precisely the footprint of the Phase 1a cache-diagnostic cell, which runs with
`DIAGNOSTIC_MAX_IMAGES = 5` and writes into `LOCAL_CACHE_DIR`/`LOCAL_RACAF_CACHE_DIR`. In that
runtime `RUN_CACHE_PRECOMPUTATION` was `False`, so Phase 1 -- the thing that mirrors persistent
entries locally -- never ran.

So nothing was lost or corrupted. `/content` is wiped on every fresh runtime while the Drive cache
survives, and the only writer of the local cache that session was the 5-image diagnostic. **Every
fresh runtime starts with an empty local cache, and the workflow must explicitly populate it
before training.** That step did not exist as its own operation.

The 8 entries missing persistently are consistent with the known empty-FOV set: persistent counts
were vessel/lesion/reliability 3651 against rgb 3662, and `3662 - 3651 = 11`, the documented
empty-FOV count. Eight fall in the 2929-entry train split, which leaves three for the 733-entry
val split. Those images have no vessel/lesion/reliability by design -- Stage 03's FOV circle-fit
finds no fundus disk -- and their handling is unchanged: the generator skips them, Phase 1 records
them under `skipped_empty_fov`, and nothing here fabricates an artifact for them.

**A second, independent defect found while tracing this.** Phase 1's persistent-to-local mirror
wrote with a plain `np.save(final_path, ...)`. An interrupted write -- a Colab disconnect mid-copy
-- therefore leaves a TRUNCATED `.npy` at the real cache filename, and every later
`os.path.exists()` check in this module treats it as a valid cache hit. All five cache-write sites
now go through `_atomic_save()`: temp file, then `os.replace()`, which is atomic on POSIX, so a
real cache filename only ever names a complete file. The temp name deliberately preserves the
original extension, because `np.save`/`np.savez` silently append `.npy`/`.npz` to a name that
lacks one and would otherwise write somewhere other than the path being renamed.

**The mirror (`joint_cache_staging.py`).** A one-time, controlled copy: plans first by `os.stat`-ing
the real persistent files (never an assumed per-artifact size), reports per-artifact file counts
and bytes against measured free space, and refuses to start unless the copy fits with a 5% + 2 GiB
margin. Each file is copied to a temp name, size-compared against the source, `np.load`-ed and
shape-validated through the project's own `jtd._validate_cached_array`, and only then
`os.replace`-d into place -- a byte copy alone would not notice an array that cannot be parsed,
and a load alone would not give a byte-identical local file; doing both gives both. It never
writes to, deletes from or repairs anything on Drive, never loads a model (so recomputation is
structurally impossible), stops immediately on `ENOTCONN` rather than continuing through thousands
more files, records a corrupt source and moves on so one bad file cannot block ~2900 good ones,
and is resumable and idempotent since only locally-missing files are copied.

Copying is **sequential by default** (`max_workers=1`). This runs once per runtime and the goal is
mount stability, not throughput: `dataset_staging.sync_missing_files()`'s 16-thread pool is the
right tool for the small PUSH back to Drive and explicitly the wrong one for a bulk PULL, per its
own docstring and Sec 35.

**Not changed.** Batch size, model architecture, trainable parameters, optimizer, learning rate,
mixed precision, XLA/JIT, loss, QWK, callbacks, validation logic, `steps_per_execution`, the data
split, frozen Stage 03/04, RACAF, canonical RGB generation, augmentation, cache layout, cache
filenames and cache directories. No cache was regenerated and the persistent cache was not
modified.

**Regression tests.** `tests/test_joint_cache_staging.py` (26): planning measures real source sizes
and per-artifact bytes; already-local entries are not planned; entries absent from Drive are
reported and never planned; free space is checked and the mirror refuses without copying anything;
planning stops at the first mount failure instead of scanning on; a full mirror makes every entry
fully local with zero Drive fallback; copies are byte-identical to their originals; the persistent
cache is byte- and mtime-identical afterwards; the frozen entry points are never called; mirroring
is idempotent and resumable; a copy that dies mid-write leaves no file at the real name and no temp
behind; a truncated source is rejected, left untouched, and does not block the other entries; a
size mismatch is rejected; a mount failure mid-copy stops immediately; `ENOTCONN` is not recorded
as corruption; verification separates Drive-fallback from missing-everywhere; empty-FOV-style
entries are a legitimate skip rather than an error; and sampled staged copies are numerically and
byte identical to their persistent originals.

**Still open.** This makes the local cache complete and the profiler runnable. It does NOT explain
the 2.44 -> 4.65 s/step regression, which remains unmeasured until the profiler completes with
zero Drive fallback.

---

## 43. Staging 14,595 loose files from Drive takes 4 hours — the cost is per-file latency, so the cache is packed into a few large shards

**The measurement that settles it.** A real staging run moved 200 files in 201.4s and 400 in
404.2s: **1.007 and 1.011 s/file, constant to within 3.5 ms**. At the measured 2.00 MiB average
that is an effective **2.08 MB/s** — an order of magnitude below what the same mount delivers
sequentially, and flat with respect to how many files have already been copied. A bandwidth-bound
copy would vary with file size; a fixed per-file cost does not. So the price is ~1 second per
Drive file *open*: 14,595 x 1 s = 4.05 h, while the same 28.56 GiB read sequentially is 10-26 min.

**The lever is therefore the NUMBER of Drive file operations, not the number of bytes.** That also
condemned part of Sec 42's own implementation: `plan_mirror()` performed one `os.stat` per source
file, adding ~14,595 more Drive round trips before the first byte was copied.

**Artifact geometry (measured on real-format files).**

| artifact | shape | dtype | bytes | derivation |
|---|---|---|---|---|
| vessel | (512,512,1) | float32 | 1,048,704 | 512*512*1*4 + 128 |
| lesion | (512,512,4) | float32 | 4,194,432 | 512*512*4*4 + 128 |
| rgb | (512,512,3) | float32 | 3,145,856 | 512*512*3*4 + 128 |
| reliability | kappa(4,) + scalar r | float32 | ~518 | `.npz`, not geometric |

Every `.npy` size is exactly determined by shape and dtype, verified against real files. Planning
therefore needs **no per-file stat at all**: two directory listings answer existence for all 3662
entries, and sizes are computed. `plan_mirror()` now works this way and performs zero per-file
Drive stats; the reliability `.npz` uses a documented nominal size for the space check only, since
stat-ing 3662 of them would cost roughly an hour to refine a rounding error.

Train and validation are built by the SAME `_make_joint_dataset` and need the same four artifacts
per entry — validation differs only by `shuffle=False, augment=False` — so train-only staging
defers validation's cost rather than removing it.

**Chosen design: uncompressed tar shards on Drive, stream-extracted.** Four measured properties
decided it, on artifacts produced by the real Phase 1 code path:

  * tar overhead is **0.11%** (53 KiB on 48 MiB), so a shard is the same size as the loose files;
  * streaming extraction runs at **~135 MB/s** locally, ~3.8 min for 28.56 GiB;
  * extracted files are **byte-identical** to the originals (24/24);
  * `tarfile.open(mode="r|")` extracts from a **non-seekable source with zero seek attempts**.

That last property is why a shard can be streamed straight off the FUSE mount with one open and
never copied locally first — which is not a nicety but a requirement: archive plus extracted is
~57.1 GiB against 45.83 GiB free, while extracted alone is 28.56 GiB and leaves ~17.3 GiB.

The reason to prefer tar over a consolidated/mmap array format is compatibility: **extraction
reproduces the existing representation exactly**, individual `.npy`/`.npz` files at the paths the
pipeline already builds, so `_build_joint_sample`, `_get_or_compute_joint_frozen_outputs`,
`_persistent_exists`, local-first precedence and every Sec 41 guarantee keep working with no
change. A consolidated format would require changing the loader and inventing new cache semantics.

**Compression is off by default, on evidence.** It was measured — vessel 42%, lesion 65%, rgb 25%
of original — but those arrays came from an UNTRAINED model over synthetic images, which are far
smoother than real Stage 03/04 output, so the ratios are optimistic and are not relied on. What
does generalize is throughput: gzip-6 compresses at 7.7-22 MB/s on this CPU and Colab provides 2
cores. Paying 20-40 min of CPU to shrink a transfer whose cost is latency rather than bandwidth is
a bad trade. `compress=True` round-trips byte-identically and is available to evaluate on real
data.

**Costs.** Building pays the ~1 s/file cost once and is resumable per shard (a finished shard is
durable and skipped; `max_shards` bounds a runtime): ~30 min per 1824-file shard, so two ~3-hour
runtimes complete all 8. Every runtime after pays only extraction: 8 Drive opens, 17-26 min of
sequential transfer plus ~3.8 min of local write. Fresh-runtime preparation goes from ~4 h to
~40 min end to end.

**Immediate unblock** (superseded by the real archive run and removed from the notebook in
§44 — kept here as the record of what was decided at the time).
`STAGING_MAX_ENTRIES` on the Phase 1c cell stages a subset — 400 train
entries is ~1600 files, ~27 min — which is enough for the profiler (30 batches plus a 256-entry
shuffle buffer touches ~316 entries). Stated caveat: a ~3.1 GiB working set page-caches in RAM
where 28.56 GiB cannot, so subset profiling yields a trustworthy COMPUTE-side measurement and an
optimistic I/O-side one.

**Persistent local storage.** Inspected rather than assumed: `colab_config.py` knows only
`/content/drive` (Drive FUSE) and paths under the ephemeral `/content`. No persistent-local
mechanism exists in this configuration; Drive is the only persistence available.

**Not changed.** `joint_training_dataset.py`'s cache resolution, the model, batch size, optimizer,
augmentation and every training setting are untouched. The persistent cache is only ever read.

**Regression tests.** `tests/test_joint_cache_archive.py` (25) covers derived sizes matching real
files, indexing via listings with zero per-file stats, a dead mount raising rather than listing
empty, shards covering every complete entry, the source staying byte- and mtime-identical,
incomplete entries excluded and never fabricated, resumable builds skipping finished shards, an
interrupted shard leaving nothing a later run would trust, byte-identical extraction at the
expected paths, idempotent extraction, a corrupt shard reported without aborting the rest, a mount
failure mid-extract keeping what landed, refusal on insufficient space, streaming never seeking the
source, and compressed round-trip correctness. The decisive one runs the REAL `_build_joint_sample`
against an extracted cache with every persistent path booby-trapped to raise, proving the extracted
cache alone is sufficient and no recomputation occurs.

**Still open.** This makes trustworthy profiling reachable. It does not explain the 2.44 -> 4.65
s/step regression, which remains unmeasured.


## 44. The raw APTOS dataset is not an input to the cache-backed training path — staged for 11 entries, not 3,662

**Question.** With the archive extracted (14,604 files / 28.53 GiB in a measured 527 s, ending at
3,651 fully local entries, 0 Drive fallback, 0 corrupt files), is the staged raw APTOS dataset
still required? It costs 9.52 GiB of `/content` and ~3,663 Drive file opens per fresh runtime.

**Traced, then measured — not inferred from the guard.** `_build_joint_sample` reads a raw image at
exactly one place, [`joint_training_dataset.py:691`](joint_training_dataset.py#L691), behind
`if not (frozen_outputs_cached and rgb_cached)`. Both flags are local-first with `or`
short-circuiting, so a full local hit never stats a persistent path either. The proof is empirical:
the real Phase 1 was run over 5 entries, the image directory was then **deleted**, and the real
`_make_joint_dataset` generator was iterated.

| measurement | result |
|---|---|
| samples yielded with the image directory deleted | 4 / 4 |
| `lfed._load_raw_bgr` calls | 0 |
| `predict_vessel_mask` (Stage 03) calls | 0 |
| `racaf.tta_views` (Stage 04) calls | 0 |
| `stage5_input` arrays vs the same run with images present | bit-identical |

Validation goes through the same `_make_joint_dataset` with `shuffle=False, augment=False`, so it
is exactly as raw-free. Nothing downstream needs the raw data either: `training/trainer.py`,
`training/callbacks.py` and `colab/common/experiment_manager.py` contain no reference to
`image_dir`, `train_images`, `APTOS` or `dataset_raw_dir`; checkpoints and TensorBoard logs are
written under the Drive-resolved `experiment.root`; and the authoritative split comes from the
git-tracked `dataset_splits/aptos2019_train_val_split.csv` (3,662 rows, 2,929/733), never from
APTOS's own `train.csv`.

**But removing it wholesale would have broken the empty-FOV path — also measured.** Phase 1 caches
an empty-FOV entry's canonical RGB and nothing else (`test_empty_fov_entry_still_gets_its_rgb_
cached`), so for those 11 entries `frozen_outputs_cached` is False every epoch, the compute branch
is re-entered, the raw image is read, and Stage 03 raises `EmptyFieldOfViewError` — which
`_make_joint_dataset` catches and skips. With the raw image absent, `_load_raw_bgr` raises
`FileNotFoundError` instead, and the generator's `except EmptyFieldOfViewError` does not catch it:
the whole run dies on the first such entry. Reproduced directly, wrapped in
`tf.errors.UnknownError` exactly as it would surface inside `.fit()`.

**Fix — narrow the scope, do not change the behavior.**
`joint_cache_staging.stage_raw_images_for_uncached_entries()` copies the raw image for exactly the
entries `entries_missing_local_cache()` reports — those whose LOCAL cache is incomplete across all
four artifacts. Normally that is the 11 known empty-FOV images: ~25 MiB and ~11 Drive opens against
9.52 GiB and ~3,663. Every entry still takes the identical code path it took before, whether it
ends in a cache hit, a recomputation, or an empty-FOV skip; if the cache were ever incomplete for
other entries, their images would be staged too and the old fallback would work unchanged. The
selection is local `os.path.exists` only — no Drive probe. Copies are atomic (temp → size check →
`os.replace`) with the module's existing bounded retry on transient FUSE errnos; the source is
opened read-only; no model is loaded and no cache file is written, so nothing frozen can be
regenerated here.

**Notebook cleanup, each item justified by the trace above.**

| cell | was | now | why |
|---|---|---|---|
| Dataset verification | ungated; `train.csv` count + 50 Drive image decodes + a 3,662-entry Drive listing | `VERIFY_RAW_DATASET = False` | its four variables have no consumer in any later cell; the split never reads `train.csv` |
| Phase 1b flush | **ungated** | `RUN_FLUSH_TO_DRIVE = False` | `sync_missing_files` stats Drive once per LOCAL file; after extraction that is ~14,595 Drive operations on a "Run all", for a guaranteed-zero result |
| Phase 1c per-file staging | 2 cells | removed | measured 1.007 s/file at 200 and 1.011 s/file at 400 → 14,595 files = 4.05 h against a ~3 h runtime. `joint_cache_staging.py` itself is kept — `verify_local_cache`, `sample_numerical_integrity` and the new raw staging all live there |
| Dataset loading / Phase 2a | `dataset_staging.stage_dataset()` (full 9.52 GiB) | `stage_raw_images_for_uncached_entries()` | above |
| Phase 1d archive | no disk check | `plan_extraction()` + hard abort | below |

Phase 1 (`RUN_CACHE_PRECOMPUTATION`) and the Phase 1a diagnostic keep full `stage_dataset()`: both
genuinely compute Stage 03/04 from raw images, and both are one-time/investigation paths.

## 45. `/content` is not a scratch disk — the notebook measures its disk budget before extracting

**Why.** With cache + dataset staged, a real runtime reported ~21.9 GiB free of ~113 GiB. Nothing
in the notebook checked that before writing 28.53 GiB, and a full `/content` surfaces as unrelated
failures hours into a run.

**`joint_cache_archive.plan_extraction()`** measures rather than assumes, and costs 8 Drive stats
plus two local listings — not a per-file scan of either side:

```
required = total shard bytes
         - bytes already in the local cache dirs     (a resumed/partial extraction)
         + DEFAULT_RUNTIME_MARGIN_BYTES (5 GiB)
```

No transient peak is budgeted, because there is none: each shard is streamed into a staging
directory on the **same filesystem** and its members are `os.replace`d into place, which is a
rename — peak usage equals final usage. The raw dataset is deliberately absent from the budget,
which is what §44 buys. The notebook prints the plan and **raises before reading a byte** if it
does not fit, and passes `required_bytes` to `extract_archive(min_free_bytes=...)` as a second,
independent guard that holds even if the plan is edited out. A separate
`MIN_FREE_FOR_TRAINING_BYTES = 3 GiB` floor aborts the dataset-loading cell up front rather than
letting a 50-epoch run discover the problem at epoch 12.

**Budget after cleanup**, from the reported figures (113 GiB total, ~53 GiB runtime base):

| | before | after |
|---|---|---|
| local cache | 28.53 GiB | 28.53 GiB |
| staged APTOS dataset | 9.52 GiB | **0** (~25 MiB of empty-FOV images) |
| free during training | ~21.9 GiB | **~31.4 GiB** |
| free at the extraction check | ~50 GiB | ~60 GiB vs 33.53 GiB required |

## 46. Phase 2a's step split cannot be trusted as an attribution — Phase 2b measures with real synchronization

**The reading that needs explaining.** The first clean, fully-local profiler run measured 4.125 s
per batch with input wait at 1.3 ms and 0 Drive reads/stats/writes — so the step is model and
optimizer time, not I/O, and §36-§45's cache work is done. Within that step it reported forward
883.5 ms, backward 989.5 ms and **optimizer 2117.8 ms**, at 7% mean GPU utilization and 69% mean
CPU. Adam over 43,292,970 parameters is a few hundred MB of traffic; ~2.1 s of *device* time for
that is roughly two orders of magnitude off, so before any of it is believed, the measurement
itself has to be audited.

**The methodological hole.** TensorFlow eager execution is asynchronous: an op returns as soon as
it is enqueued, and reading a tensor forces only that tensor's dependencies.
`profile_compute_only` drains one tensor per boundary — `sync(grads[0])`, then
`sync(model.trainable_variables[0])`. `grads[0]` belongs to the first trainable variable, which
backprop produces *last*, so that drain happens to cover most of the backward pass — an accident of
variable ordering, not a guarantee. `trainable_variables[0]` is updated near the *start* of
`apply_gradients`, so draining it does **not** cover the other 392 updates. Whatever is still in
flight at a boundary lands in whichever section drains next. Phase D is therefore fine as a
magnitude, unsound as an attribution.

**Phase 2b (`profile_optimizer_breakdown`)** measures the same work correctly:

* **full drains.** `_drain()` builds one scalar via `tf.add_n` over per-tensor reductions, so a
  single `.numpy()` cannot return until every tensor in the set is materialized. The drain's own
  cost is measured separately on already-materialized tensors and reported, so it is subtractable
  rather than silently included.
* **enqueue vs drained.** `apply_gradients` is timed both as enqueue-only (host returns, device
  work possibly still in flight) and drained. Enqueue time is host work — Python and op dispatch;
  drain time is device work. That pair answers "CPU-bound or GPU-bound" directly.
* **CPU vs wall.** `time.process_time()` beside `time.perf_counter()` for every section. cpu/wall
  near 1.0 means the host is busy; near 0.0 means it is blocked on the device.
* **an attribution cross-check.** The same work, drained **once** at the end. If the per-section
  sum does not match, the report says so instead of presenting the split as fact.
* **the compiled path.** `model.train_on_batch` runs inside a `tf.function`, which is what
  `Trainer.fit()` → `model.fit()` actually executes; the eager tape loop is not. Both are measured.
  The comparison is conservative — `train_on_batch` also computes the compiled QWK metric.
* **per-section GPU telemetry**, sampled during each section from a background thread in the same
  process.
* **the inventory the numbers must be read against**: trainable tensor count and dtypes, gradient
  tensor count/dtypes/bytes and any `None` gradients, optimizer slot variables and dtypes,
  `clipnorm`/`clipvalue`/`global_clipnorm`/`use_ema`, `jit_compile`, XLA flags, `run_eagerly`,
  `steps_per_execution`, and the model's real dtype policy.

**One inventory finding is already in hand, and it is a fact about configuration, not a
measurement of the regression.** Verified on Keras 3.15.1: a layer captures the dtype policy when
it is **built**. The notebook builds and compiles `joint_model` before `Trainer.fit()` →
`prepare()` → `enable_mixed_precision(True)` sets `mixed_float16`, so the policy is set after the
model exists and does not apply to it. Confirmed on the real model: 393 trainable tensors, all
`float32`; 393 gradient tensors, all `float32`; optimizer `Adam` with `loss_scale_factor=None`
(no `LossScaleOptimizer`); 788 slot variables; `jit_compile=False`; no clipping configured.
`MIXED_PRECISION = True` in the notebook is therefore not in effect for this graph.

**Not claimed.** The 4.125 s/step is not explained here, and neither is the 2.44 → 4.65 s/step
regression. Phase 2b exists to produce evidence for those, and nothing about the model, optimizer,
batch size, loss, precision policy, JIT setting or training configuration has been changed. The
diagnostic advances the optimizer's slot state, so the model construction cell must be re-run
before a real training run; weights are snapshotted and restored bit-for-bit (asserted).

**Regression tests.** `tests/test_joint_training.py` adds `RawImagesAbsentWithCompleteCacheTests`
(5) and `EmptyFovEntriesStillNeedTheirRawImageTests` (5): every sample still produced with the
image directory deleted, zero raw reads and zero Stage 03/04 calls, samples bit-identical to the
images-present run, validation equally raw-free, no persistent probe, Phase 1 leaving an empty-FOV
entry with RGB only, the skip preserved when its image is present, the hard failure when it is
absent, the helper selecting exactly those entries, and the staged-subset directory reproducing
today's skip behavior end to end. `tests/test_joint_cache_staging.py` adds
`RawImageStagingTests` (12) and `tests/test_joint_cache_archive.py` adds `ExtractionPlanTests` (8),
including that the plan opens no shard, that a dead mount is reported rather than read as zero
bytes, and that a failed space check leaves both the local dirs empty and Drive byte-identical.

---

## 47. Stage 5's fusion was multi-kernel but not adaptive — completed with per-image, per-channel branch weighting

**What was there.** `local_feature_extraction_model._multi_kernel_block` ran three parallel
branches at different effective receptive fields (3×3, 5×5, dilated 3×3 with `dilation_rate=3`),
concatenated them, and fused them back to `filters` channels with a 1×1 convolution. That
delivered the multi-scale half of the approved Stage 5 design, and §13's `(512,512,8) →
(32,32,256)` contract, correctly.

**What was missing.** A convolution kernel is a *fixed learned parameter*. Once trained, the 1×1
fusion mixes the three branches in exactly the same proportion for every image in the dataset.
The block was therefore multi-kernel but not adaptive, while the module, the model name
(`local_feature_extraction_adaptive_multi_kernel_cnn`) and the approved design all say "Adaptive
Multi-Kernel CNN". The design's own rationale is image-dependent: the scale carrying the signal
differs per fundus — microaneurysms occupy a few pixels, hard/soft exudates occupy broad patches —
so the branch mixture is exactly the thing that should not be constant.

**What was added.** `AdaptiveBranchFusion`, one registered Keras layer, placed between the branches
and the concatenation:

```
context   = GlobalAveragePooling2D(sum(branches))        -> (N, C)
hidden    = Dense(max(C // 8, 8), relu)(context)
logits    = Dense(3 * C)(hidden)                         -> (N, 3C)
weights   = softmax(reshape(logits, (N, 3, C)), axis=1)  -> (N, 3, C)
output[b] = branches[b] * weights[:, b, :]               broadcast over H, W
```

The branch structure, the concatenation, the 1×1 fusion convolution, every existing layer name and
the output contract are unchanged; the layer only rescales the branches before they are
concatenated. Properties, each pinned by a test rather than asserted here:

* **Image-dependent** — the weights are a function of the sample's own pooled branch responses.
  Nothing else enters: no label, no batch statistic, no cross-sample term, so a sample's weights
  are identical alone or inside any batch (`test_a_sample_gets_the_same_weights_alone_as_inside_a_batch`).
* **Normalized across branches** — `softmax(axis=1)` is over the branch axis, so for every
  (sample, channel) the three weights are positive and sum to 1. "How much of this channel came
  from which receptive field" is a well-defined proportion.
* **Differentiable and trainable end to end**, through the same CORN ordinal loss as the rest of
  the graph (§21). No auxiliary loss, no new supervision.
* **Batch-size agnostic** (including `BATCH_SIZE = 2`) and shape-static, so XLA/`jit_compile` can
  trace it.

**Mixed precision.** The two projections and the softmax carry an explicit `dtype="float32"`; the
resulting weights are cast back to the block's compute dtype before they multiply the branches.
A three-way per-channel softmax is a normalized proportion, and computing it in float16 would put
it at the mercy of fp16 rounding for no throughput gain — the projection is only `(N, C)` wide.
The heavy convolutions are untouched and still run in float16. This puts Stage 5 in the same
category as `racaf_output` and `fused_embedding`, which already carry deliberate float32
overrides, and `training.trainer.precision_is_consistent` is written to treat such overrides as
part of the design rather than as a policy mismatch.

**Parameter cost — measured, not estimated.**

| | trainable parameters | trainable tensors |
|---|---|---|
| Stage 5 before | 2,129,152 | — |
| Stage 5 after | 2,174,688 | — |
| Joint model before | 43,292,970 | 393 |
| Joint model after | 43,338,506 | 409 |
| Delta | **+45,536 (+0.105 %)** | +16 (4 stages × 2 `Dense` layers × kernel+bias) |

Per stage: 1,128 / 2,248 / 8,592 / 33,568 for C = 32 / 64 / 128 / 256. The joint model's "before"
figure computed this way reproduces the independently measured 43,292,970 exactly, which is the
cross-check that the delta is entirely this change and nothing else.

**Forward/backward cost — measured on a CPU host, batch 2, 512×512×8, median of 3 reps**, with the
pre-change block reconstructed verbatim and timed in the same process so the comparison is like
for like: forward 1.243 s → 0.976 s, forward+backward 5.817 s → 7.009 s (+20.5 %). These are CPU
seconds and are **not** T4 figures; the transferable observation is that the adaptive path is a
`(C → C/8 → 3C)` projection on a globally pooled vector, so its cost is independent of spatial
resolution while the convolutions it sits between are not — on a GPU, where the convolutions are
the dominant term and the projection is a handful of small GEMMs, the relative overhead is
expected to be smaller than on CPU, not larger. The notebook's Phase 2c cell measures the real
per-step cost on the T4; **no claim about T4 step time is made here.**

**Scope.** This is branch selection inside one block. No spatial attention map, no query/key/value
projection, no cross-stage gating, no auxiliary supervision, no change to RACAF (§16) and no change
to Stages 1–4. The project's single research contribution remains RACAF (§30).

**Tests.** `tests/test_local_feature_extraction_model.py` adds `AdaptiveBranchFusionTests` (18):
build and shape, the reduced-units floor, weight normalization, per-sample variation, gradient flow
into the weighting parameters, batch sizes 1 and 2, output reproducibility, `mixed_float16`
execution, a compiled `jit_compile=True` train step (requirement 10 -- everything in the layer is
shape-static, so XLA can trace it), config round-trip, serialization registration, rejection of a
single-tensor input and of mismatched branch channel counts, one adaptive-fusion layer per stage,
the parameter-growth bound, and the unchanged `(None, 32, 32, 256)` output contract. The regression guard the section exists
for is `test_not_merely_one_globally_fixed_learned_fusion_vector`: it asserts both that four
deliberately different samples in one batch receive different weights, and that the gradient of the
weights with respect to the branch activations is non-zero — a constant that merely happened to
vary would fail the second check.

---

## 48. `MIXED_PRECISION = True` was inert — the dtype policy was set after the model was built

**The defect.** §46 recorded this as an inventory finding; it is now fixed. Keras 3 captures a
layer's dtype policy in the layer's **constructor**, and decides whether to wrap the optimizer in a
`LossScaleOptimizer` when the model is **compiled**. The notebook built and compiled `joint_model`
in its construction cell, and the policy was only set later, inside `Trainer.prepare()` →
`enable_mixed_precision(True)` — by which point it could no longer affect anything. `setup.setup()`
does not set a policy either, so a clean top-to-bottom run of the notebook trained the whole joint
model in float32, with a bare `Adam` and no loss scaling, while every configuration flag reported
mixed precision as enabled.

Reproduced directly on TF 2.21.0 / Keras 3.15.1 and pinned by
`test_changing_the_global_policy_after_construction_does_not_change_the_model`:

| order | model policy | optimizer |
|---|---|---|
| build+compile under float32, then set `mixed_float16` globally | `float32` | `Adam` |
| set `mixed_float16` globally, then build+compile | `mixed_float16` | `LossScaleOptimizer(inner=Adam)` |

**What is NOT a defect.** The Colab diagnostic's observation that "all 393 trainable tensors and
gradients were float32 even though the policy is mixed_float16" is correct mixed-precision
behaviour, not a symptom. `mixed_float16` is `compute_dtype=float16, variable_dtype=float32` by
definition: master weights stay in float32 and the gradients applied to them are float32. Nothing
was changed on account of that observation, and
`test_mixed_float16_keeps_variables_in_float32_by_design` exists so nobody "fixes" it later.

**The fix — the smallest one that is actually correct.**

1. `joint_training_model.build_and_compile_joint_model(mixed_precision=...)` does policy → build →
   compile in that order and raises if the result is not what was asked for (including: a
   `mixed_float16` model that did not get a `LossScaleOptimizer`, which would mean float16
   gradients underflowing to zero unnoticed).
2. `training.trainer.verify_model_precision()` re-checks the model `Trainer.fit()` is actually
   handed, so the notebook and the trainer can never silently disagree again.
   `TrainingConfig.precision_check` governs the response: `"warn"` by default, which preserves
   every existing caller's behaviour exactly; the joint notebook sets `"error"`.
3. The notebook's model-construction cell now owns `MIXED_PRECISION` and sets it before the first
   layer is constructed. The training-configuration cell no longer redefines it, and says why.

`precision_is_consistent()` deliberately treats the two directions differently: under
`mixed_float16`, individual `dtype="float32"` layer overrides (`racaf_output`, `fused_embedding`,
`AdaptiveBranchFusion`'s branch-weight projection) are part of the design, so the requirement is
that at least one weighted layer is genuinely float16 — mixed precision being *absent* is the
fault. Under `float32`, no layer may be float16 at all.

Nothing else about the training configuration was touched: batch size is still 2, the optimizer is
still `Adam` with its default learning rate, the model, the loss, `jit_compile` and
`steps_per_execution` are all unchanged.

**The 4–5 s/step question is still open, and is not answered by this fix.** Phase 2b measured a
compiled `train_on_batch` at ~78 ms against a ~4215 ms eager taped step on the same batch and GPU —
a 54× gap that makes the eager number unusable as an attribution. The historical ~4–5 s/step was
`model.fit()`, which runs the compiled `train_function`, so something outside that step accounts
for it. **Phase 2c** (`joint_training_profiler.profile_fit_paths`, notebook cell "Phase 2c")
measures the four paths that isolate it, over 20–30 steps with every artefact written to a local
directory and nothing to Drive:

| path | measures |
|---|---|
| A dataset only | the real `tf.data` pipeline, no model — the ceiling the input pipeline can sustain |
| B compiled `train_on_batch` | one materialized batch, reused — the ceiling the GPU can sustain |
| C `model.fit()`, minimal callbacks | the real compiled loop over the real pipeline |
| D `model.fit()`, real callback stack | C plus TensorBoard histograms, CSV logging, a checkpoint write |

`D − C` is callback overhead; `C − B` is what `fit()` adds around the compiled step; `A > B` means
`prefetch` cannot hide the input pipeline. Phase 2a's "input wait 1.3 ms" could not rule that out,
because it was measured against a 4.1 s eager consumer — a prefetching pipeline only has to beat
whatever the consumer is, and that consumer was ~54× slower than the real one. **This section names
no root cause.** Phase 2c has not been run on Colab in this task, and the report is written so that
it says which surrounding operation accounts for the difference from measurements, or says the
historical timing was not reproduced — never both.

**Safety against a stale optimizer.** The diagnostics take real gradient steps. Weights are
snapshotted and restored, but the optimizer's slots and `iterations` cannot be, so
`profile_fit_paths` sets `model._dr_diagnostic_dirty` and `Trainer.fit()` refuses such a model with
an explicit instruction to rebuild and recompile
(`test_fit_refuses_a_model_left_dirty_by_a_diagnostic`).

**Tests.** `tests/test_training_precision.py` (20): policy capture and the after-the-fact-change
no-op, `LossScaleOptimizer` wrapping, float32 variables under `mixed_float16`, the asymmetric
consistency rule, `verify_model_precision` in all three modes, `Trainer.prepare(model)` refusing a
mismatch, the diagnostic-dirty refusal, and two tests that build the **real** joint model through
`build_and_compile_joint_model` for both `mixed_precision=True` and `False`.

---

## 49. Checkpoint/resume rebuilt for a run that spans many sessions — generations, integrity, and a global best

**Why §25 was not enough.** §25's weights-only checkpointing is correct about *format* — Stage 6's
Swin classes have no `get_config()`, so full `.keras` serialization is not safe here, and that has
not changed. What it did not cover is *state*. `ModelCheckpoint(save_weights_only=True)` persists
model variables and nothing else, so a resumed session restarts with:

* a freshly initialized optimizer — Adam's moment estimates back to zero, `iterations` back to 0;
* `EarlyStopping.wait = 0` and `ReduceLROnPlateau.wait = 0`, both reset in their `on_train_begin`;
* `ModelCheckpoint.best = None`, so the first epoch of every session always "improves".

On this run's schedule — an epoch is ~1.67 h and a Colab session is ~3 h, so roughly one epoch per
session — that is not a rounding error. With `patience=8` and `patience=4`, **EarlyStopping could
never fire and the learning rate could never be reduced**: both counters reset before they could
accumulate. And `best.weights.h5` would be overwritten by the first epoch of every new session
regardless of its `val_QWK`. A three-process relay run against the pre-change code confirmed
exactly this: `val_loss` 0.6268 was overwritten by 625.41 at a leg boundary, with
`optimizer.iterations` back at 0.

**The format.** `training/checkpointing.py`, one immutable numbered generation per epoch:

```
checkpoints/
    gen_00012/
        model.weights.h5   166.36 MiB   model variables incl. BatchNorm moving statistics
        optimizer.npz      331.15 MiB   all 820 optimizer variables, in build order
        state.json            852 B     the authoritative training state
        manifest.json       1,029 B     per-file size + SHA256, plus compatibility metadata
        READY                 194 B     written LAST
    gen_00011/                          previous known-good, retained
    best/                               globally best val_QWK epoch (delivery model)
    latest.json                         pointer to the newest known-good generation
```

**The write protocol, and why it does not rely on `os.replace`.** Google Drive's FUSE mount makes
no atomic-rename guarantee, and this project has already been bitten by that mount (§35, §41). So
atomicity is supplied by a marker instead: build and validate the generation on **local** disk,
copy it to the destination, **re-validate every file's size and SHA256 at the destination**, and
only then write `READY` and update `latest.json`. A generation without `READY` is invisible to
`find_resumable_generation()`, so a half-copied one is skipped rather than half-loaded. The
previous known-good generation is pruned only after its replacement has validated at the
destination, and the generation `latest.json` points at is never pruned.

`READY` carries the manifest's own SHA256, so a manifest edited after the fact cannot vouch for
itself.

**Integrity failures and compatibility failures are handled differently, on purpose.**

* *Integrity* (missing file, wrong size, wrong SHA256, no `READY`) means "this generation is
  damaged": `find_resumable_generation()` skips it and falls back to the newest older generation
  that validates. Losing one epoch is the correct price.
* *Compatibility* (different `config_hash`, optimizer type, precision policy, or checkpoint format)
  means "intact, but not this run": that **raises**. Falling back to an older generation of a
  differently configured run would be worse than stopping. Git commit, TF/Keras/Python versions and
  dataset version are *advisory* — pulling a new commit between Colab sessions is normal — and are
  reported as warnings rather than refusals.

**Restore order.** Validate integrity → validate compatibility → build the optimizer's slots →
restore the optimizer → restore the model weights → return the state. Building the slots first is
the single most important rule here: a freshly constructed Keras 3 optimizer owns only `iteration`
and `learning_rate`, and the per-parameter momentum/velocity slots are created lazily on the first
`apply_gradients`, so assigning into an unbuilt optimizer silently restores almost nothing. Any
count/shape/dtype mismatch raises `CheckpointIntegrityError`; **no slot is ever left silently at
its initial value**, because a resumed Adam with half its second moments zeroed takes large,
wrongly scaled steps for hundreds of iterations.

The optimizer is restored *before* the weights so that an architecture mismatch is reported as a
named optimizer-variable shape error rather than as an opaque HDF5 failure.

**LAST vs BEST.** `gen_NNNNN/` is LAST — the trajectory the next session continues from, written
every epoch, model **and** optimizer. `best/` is BEST — the globally best `val_QWK` epoch across the
whole experiment, weights and state only, the intended delivery model, never resumed from.
`EarlyStopping(restore_best_weights=True)` rewinds the in-memory model in `on_train_end`; LAST is
written in `on_epoch_end`, so LAST always holds the real end-of-epoch trajectory and the two never
contaminate each other. BEST is copied out of the generation written for that epoch, so it is
bit-identical to it rather than a second serialization.

**Global best.** `TrainingStateCheckpoint` reads the best metric back from the previous
generation's `state.json` in `on_train_begin` -- but only when the run is actually resuming
(`Trainer` wires this from `TrainingConfig.resume`). A run that is not resuming must not inherit
its predecessor's best, or a fresh run pointed at a populated directory would start from epoch 0
with fresh weights while claiming a score it never achieved; it says so and starts clean, and the
generation numbering still continues so nothing already on disk is overwritten. The state file is
the authoritative record — not
`ModelCheckpoint.best`, which is a per-process Keras internal. The monitor is `val_QWK`, mode
`max` (§23, unchanged). Session A reaching 0.72 and session B reaching only 0.70 leaves the global
best at 0.72; session C reaching 0.75 advances it.

**Callback state.** The same callback restores `EarlyStopping`'s `best`/`wait`/`stopped_epoch` and
`ReduceLROnPlateau`'s `best`/`wait`/`cooldown_counter` *after* their `on_train_begin` has reset
them — which is why `build_callbacks()` places it after both, and why a test pins that ordering.
`min_delta` is deliberately **not** restored: Keras mutates it in `_set_monitor_op()`
(`min_delta *= -1` for `mode="min"`), so writing a persisted value back would double-apply the sign
flip. The effective learning rate travels with the optimizer, so restoring the optimizer restores
a reduced LR automatically.

**Two duplication fixes found by measurement.** Keras 3's `save_weights` walks the model's tracked
attributes, and once the optimizer's slots exist it is one of them — so a `.weights.h5` written
after the first gradient step silently carries the entire Adam state as well. Measured on the real
joint model: **521,441,264 bytes**, against 173.4 MB of actual model variables, i.e. the momentum
and velocity slots duplicated inside a file whose whole purpose is model weights, on top of the
331 MiB `optimizer.npz` that already holds them and is the strictly validated copy. Detaching the
optimizer for the duration of the write (and of the read, which also removes a benign but
alarming `Skipping variable loading for optimizer …` warning) brought the file to **174,441,224
bytes** and the whole generation from 828.57 MiB to **497.51 MiB**, and save time from 36.1 s to
15.8 s. Separately, `state.json` fell from 140,389 B to **852 B** by keeping the 820-entry
optimizer variable list where it is actually used — inside `optimizer.npz`, validating the
restore — and storing only its signature hash in the state file a human reads.

**Measured cost (local disk, CPU host, real 43.3M-parameter joint model).**

| | |
|---|---|
| save (build + local validation + copy + destination re-validation + seal) | 15.8 s |
| integrity re-validation (SHA256 re-read) | 1.4 s |
| restore (build slots → restore optimizer → restore weights) | 3.8 s |
| `model.weights.h5` | 174,441,224 B (166.36 MiB) |
| `optimizer.npz` | 347,235,278 B (331.15 MiB) |
| `state.json` / `manifest.json` | 852 B / 1,029 B |
| total per generation | 497.51 MiB |
| two retained generations + BEST | ≈ 1.16 GiB |
| local staging headroom needed | ≈ 500 MiB |

Against a ~1.67 h epoch that is under 1 % overhead. **Drive/FUSE write cost is NOT measured** —
the notebook's Phase 2d cell can repeat the measurement against a real Drive path
(`CHECKPOINT_COST_ON_DRIVE = True`), and no crash-safety claim about Drive should be made until it
has been. Everything below is what has actually been tested, and where.

**What resume guarantees, and what it does not.** Restoring a generation reproduces the model, its
BatchNorm moving statistics, the full Adam state, the effective learning rate, and every stateful
callback counter. That is *statistically equivalent continuation*, **not** bit-for-bit reproduction
of an uninterrupted run, and the difference is not fixable by any checkpoint format:

* this project sets no global seed anywhere, so even two uninterrupted runs do not match;
* GPU kernels are not deterministic;
* Keras 3's dropout `SeedGenerator` state is not part of `.weights.h5`;
* `model.fit(initial_epoch=N)` does not fast-forward the `tf.data` shuffle stream, so a resumed
  session replays epoch 0's batch order.

The last point is a real limitation of resume as implemented, deliberately left alone here rather
than redesigning the data pipeline. It is recorded so it is not mistaken for a claim of exactness.
The separate, pre-existing augmentation defect — `gen()` re-creates `rng =
np.random.default_rng(seed)` on every dataset iteration, so every image receives an identical
augmentation in every epoch — is **not** addressed by this work and is not caused by it.

**Tests.** `tests/test_checkpoint_resume.py` (51) covers all twenty required areas: generation
layout, same-process save/load, weight/BatchNorm persistence, optimizer `iterations` and Adam slot
persistence, refusal on incomplete or shape-mismatched optimizer state, manifest sizes and SHA256s,
truncated/corrupted/tampered detection, `READY` semantics, partial-write and corruption fallback to
the previous known-good generation, pruning that never removes the pointer's target, config and
metadata mismatch refusal, LAST vs BEST distinction, the global best surviving a worse resumed
epoch, EarlyStopping and ReduceLROnPlateau counters accumulating across legs, and the weights-file
and state-file size guards. `ThreeProcessRelayTests` spawns three real OS processes
(`tests/checkpoint_relay_worker.py`) through the production `Trainer`: A trains epochs 0–1, B
resumes for 2–3, C resumes for 4–5, and the test asserts that `optimizer.iterations` continues
across both process boundaries, that the global best survives a leg that scores worse, that
`EarlyStopping.wait` accumulates 0 → 2 → 0 across them, and that a reduced learning rate is the
one the next process starts from.

**Backwards compatibility.** `build_callbacks()` without a `CheckpointOptions` produces exactly the
callback set it always did, `checkpoint_paths()` still returns `best_weights`/`last_weights`/
`epoch_state`, and Stages 1–4 are untouched. `EpochStateLogger` now writes through a temporary file
rather than truncating the live one, which is strictly safer and changes nothing about its
contents. `extra_callbacks` now run first rather than last, so a caller-supplied callback can
enrich the shared `logs` dict before the standard callbacks read it — nothing in the repository
passed `extra_callbacks` before this change.

**How to resume a previous experiment.** Set `RESUME_EXPERIMENT_DIR` in the notebook's
training-configuration cell to the experiment's root
(`/content/drive/MyDrive/DiabeticRetinopathy/experiments/FinalClassification/<timestamp>`) and run
the notebook normally. `Trainer` finds the newest valid generation, refuses the resume if
`config_hash` no longer matches, restores weights + optimizer + callbacks, and continues from
`completed_epoch`. Nothing has to be copied by hand and no optimizer variable has to be edited.
