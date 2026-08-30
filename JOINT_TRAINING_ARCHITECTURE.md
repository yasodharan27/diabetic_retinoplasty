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
`epoch_state.json` mechanism), best/final checkpoint, and per-stage exported-weight slices (into
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
