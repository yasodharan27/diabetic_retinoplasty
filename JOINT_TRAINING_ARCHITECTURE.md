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

**This cache is deliberately LOCAL-ONLY**, via its own `rgb_cache_dir` parameter (default `None` →
`cache_dir`, so every existing caller and test is unaffected; the notebook points it at
`/content/cache/canonical_rgb`). The reason is a storage/benefit asymmetry that only became visible
once the array size was computed exactly:

| | per image | × 3662 images |
|---|---|---|
| vessel (512×512×1 f32) | 1.0 MiB | |
| lesion (512×512×4 f32) | 4.0 MiB | |
| **existing Drive cache** | **~5.0 MiB** | **17.9 GiB** |
| **new canonical RGB (512×512×3 f32)** | **3.0 MiB** | **10.7 GiB** |

The notebook's Phase 1b flush cell syncs a whole directory (`sync_missing_files(LOCAL_CACHE_DIR,
…)`), so leaving the rgb entries in `cache_dir` would have silently grown the Drive cache ~60%, from
17.9 GiB to 28.6 GiB — plus the corresponding bulk-write FUSE exposure this project has already been
burned by twice (§33, §35). That cost buys almost nothing: unlike vessel/lesion/reliability (a
frozen Stage 03/04 forward pass, seconds per image, genuinely worth persisting across sessions),
canonical RGB is regenerable from the already-locally-staged raw image in ~90–500ms — roughly what
reading the same 3 MiB back over Drive FUSE costs anyway. So it is rebuilt locally, once per
session, by Phase 1 (the phase explicitly designed to be the slow, resumable, run-once-before-
training one), and never written to Drive. `persistent_cache_dir` is still consulted for an rgb
entry if one happens to exist there — harmless, and correct if a caller ever does choose to persist
it.

**Not changed:** Stage 03/04 architecture or weights, RACAF or CORN mathematics, Stage 5/6/7
architecture, loss, QWK, optimizer, batch size, epochs, `Trainer`/`TrainingConfig`, the resize
algorithm itself, `.repeat()` (deliberately not added — see above), the authoritative split
manifest, and no existing Drive cache entry was deleted, overwritten, or invalidated. Drive cache
SIZE is also unchanged: the one new cache kind is local-only by construction (above), so Phase 1b
still flushes exactly the same vessel/lesion/reliability files it did before.

**Regression tests:** `tests/test_joint_training.py` — `CanonicalRGBCachingTests` (rgb cache file
written on first build; cached value numerically identical to the sample it was derived from; a
full cache hit never calls `lfed._load_raw_bgr`; a full cache hit reproduces the exact same
`stage5_input` as a fresh computation; a persistent-only rgb entry is mirrored locally without
recomputing OR re-reading the raw image; an entry missing only its rgb cache backfills by reading
the raw image exactly once, without recomputing Stage 03/04), `Phase1CanonicalRGBCachingTests`
(Phase 1 writes an rgb cache for every processed entry; a second Phase 1 run never re-reads raw
images; a legacy persistent hit with vessel/lesion/reliability but no rgb file is still recognized
as a full frozen-outputs hit, never recomputing Stage 03/04, while backfilling rgb; an empty-FOV
entry — whose vessel/lesion/reliability cache never gets written — still gets its rgb cached; Phase
1 honors a separate local-only `rgb_cache_dir`, leaving `cache_dir` — the directory that IS flushed
to Drive — with no `_rgb_` files at all; Phase 1 → Phase 2 end-to-end through that same separate
dir needs neither a raw image nor Stage 03/04), one added test in the existing
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
