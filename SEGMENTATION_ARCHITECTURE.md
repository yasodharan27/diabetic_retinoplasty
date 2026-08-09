# Segmentation Stage — Architecture Design Document

**Status:** Frozen architecture specification for pipeline stages 3 (Vessel Segmentation) and 4 (Lesion Segmentation), plus the interface/shape contract connecting them to stages 5–8 (Local Feature Extraction → Global Feature Extraction → Feature Fusion → Ordinal Classification). **Vessel Segmentation uses a pretrained, externally-sourced model (LWNet) for inference only — it is not trained within this project.** **Lesion Segmentation is trained within this project**, on IDRiD's segmentation subset. An earlier design trained Vessel Segmentation within this project on DRIVE + CHASE_DB1 instead; that design was itself adopted, then superseded again by the current pretrained-LWNet design, and is retained only in the Appendix (Design History) for context.

**Sources consulted:** `PROJECT_CODE.md`, `IMPLEMENTATION_PLAN.md`, the `datasets/` directory tree, the existing `pipeline/`, `training/`, `evaluation/`, `config.py` code, and the papers in `research_papers/`.

Design alternatives that were considered and set aside are recorded in the [Appendix: Design History](#appendix-design-history), not in the sections below — the sections below describe only the current, frozen architecture.

---

## 1. Dataset Assignment

### 1.1 Datasets on disk

| Path | Contents |
|---|---|
| `datasets/APTOS2019/raw/{train_images,test_images}` | Fundus images, image-level DR grade labels |
| `datasets/EyePACS/raw/{train,test}` + `trainLabels.csv` | Fundus images, image-level DR grade labels — historical only; used once to reconstruct EyeQ, not used by this pipeline (§1.5) |
| `datasets/EyeQ/raw/{train,test}` | Fundus images, image-level quality labels |
| `datasets/IDRiD/grading/raw/` | Images + image-level DR/DME grade labels |
| `datasets/IDRiD/localization/raw/` | Images + Optic Disc / Fovea center point coordinates (not pixel masks) |
| `datasets/IDRiD/segmentation/raw/` | Reserved for the IDRiD Segmentation subset (Microaneurysm, Haemorrhage, Hard Exudate, Soft Exudate, Optic Disc masks); not yet populated |

DRIVE and CHASE_DB1 are **not** project datasets. They were added to an earlier, now-superseded design specifically to train Vessel Segmentation within this project; that design has been revisited (see [Appendix A.1](#a1-vessel-segmentation-model-source)). Vessel Segmentation now runs a pretrained checkpoint (LWNet, §1.2/§2) that was trained on DRIVE by its original authors, entirely outside this project — this repository never downloads, stages, or trains against DRIVE or CHASE_DB1 itself.

### 1.2 Vessel Segmentation

Vessel Segmentation runs a **pretrained LWNet model** ("The Little W-Net That Could," Galdrán et al., MIT-licensed — see §8) for **inference only**. It is not trained within this project. The vendored checkpoint (`experiments/wnet_drive/model_checkpoint.pth` in the upstream `lwnet` repository, a `wnet` architecture with `in_c=3, n_classes=1`) was trained by LWNet's original authors on the DRIVE dataset; this project consumes that trained checkpoint directly and never re-trains, fine-tunes, or downloads DRIVE/CHASE_DB1 itself.

This design was adopted deliberately, reversing the previously-frozen "Baseline U-Net trained within this project on DRIVE + CHASE_DB1" design (see [Appendix A.1](#a1-vessel-segmentation-model-source) for the full history, including why that trainable design was itself adopted and later revisited again).

**Known distribution-shift caveat:** LWNet's authors trained and tuned it (including its 0.4196 default binarizing threshold) against DRIVE images normalized their own way — not against Stage 02's Gamma+CLAHE output. Running Stage 03 on Stage 02's processed images is therefore inference on an out-of-distribution input relative to LWNet's original training data. This is a known, documented limitation, not a fabricated equivalence claim — no benchmark exists yet quantifying the effect (§7.1 note).

### 1.3 Lesion Segmentation

Lesion Segmentation (Attention U-Net) trains on `datasets/IDRiD/segmentation` (per `PROJECT_CODE.md`'s Datasets table, corroborated by `dr_gan++.pdf` and the systematic review's IDRiD citation).

`datasets/IDRiD/segmentation/raw/` is currently empty. Per `PROJECT_CODE.md`'s Dataset Handling rule (no automatic downloads; always use datasets already inside `datasets/`), the IDRiD Segmentation subset's files (Original Images + the Groundtruth mask folders) must be placed there before training code is run.

### 1.4 Training vs. inference usage per dataset

- **Vessel Segmentation** is not trained on any dataset in this project — it runs a pretrained external checkpoint. It runs **inference** over every dataset it touches in the pipeline — EyeQ (if ever needed), APTOS2019, and IDRiD (including `datasets/IDRiD/segmentation` itself, to produce the vessel-mask input channel Lesion Segmentation's training requires).
- **Lesion Segmentation** is trained on `datasets/IDRiD/segmentation`. It runs **inference** (not training) on APTOS 2019, since it ships only image-level DR grade labels, not masks.

### 1.5 EyePACS (historical only)

EyePACS was used once, historically, to reconstruct the official EyeQ dataset (see `PROJECT_CODE.md`'s Datasets section). The reconstructed EyeQ dataset is the dataset actually used for Image Quality Assessment. EyePACS itself is **not** part of the Segmentation stage, or any other part of the implemented pipeline — it is not used for training, preprocessing, or inference here, and is not required to reproduce or run this repository.

### 1.6 Datasets never used for segmentation

**EyeQ** is scoped to Image Quality Assessment only (`PROJECT_CODE.md`'s Datasets table). It carries no lesion/vessel annotation and is not part of the Segmentation stage's input.

### 1.7 Summary

| Dataset | Role |
|---|---|
| IDRiD (`segmentation` subset) | Trains Lesion Segmentation (Stage 4). Also passes through Vessel Segmentation's pretrained model for inference (vessel-mask input channel). |
| APTOS 2019 | Inference-only for both Vessel Segmentation and Lesion Segmentation |
| EyeQ | Not used for segmentation |
| EyePACS | Not part of this project (§1.5) — historical only, used once to reconstruct EyeQ |

DRIVE and CHASE_DB1 are not project datasets (§1.1) — Vessel Segmentation is a pretrained checkpoint, not a model this project trains.

Every image in `datasets/IDRiD/segmentation` also passes through Vessel Segmentation inference, since Lesion Segmentation's training input requires the vessel mask (§3.1).

---

## 2. Vessel Segmentation

| Property | Specification |
|---|---|
| **Model** | Pretrained LWNet (`wnet`, `in_c=3, n_classes=1`) — external, inference only (`PROJECT_CODE.md`'s Models table) |
| **Training** | None within this project — the vendored checkpoint was trained by LWNet's original authors on DRIVE (§1.2) |
| **Input** | Preprocessed RGB fundus image, single image per forward pass (batchable); see §2.1 |
| **Output** | Per-pixel vessel probability map, 1 channel |
| **Output resolution** | Matches input resolution — LWNet internally resizes to its own working resolution and resizes its prediction back before returning; see §2.2 |
| **Saved model format** | PyTorch `.pth` — fixed, since this is a vendored external checkpoint, not an open framework decision (see §6). This is a departure from Image Quality Assessment's TensorFlow/`.keras` convention (`image_quality_model.py` / `image_quality_inference.py`), and from the framework-agnostic placeholder the previous, now-superseded design used. |
| **Inference API** | `pipeline.SegmentationStage` contract (§5) — `predict`/`predict_batch`/`load` only; see §5 |

### 2.1 Input channel count

The canonical preprocessed image used throughout this pipeline (Vessel Segmentation, Lesion Segmentation, Local/Global Feature Extraction, and elsewhere "the preprocessed image" is referenced) is **RGB, 3 channels** — Stage 02's frozen output format (`PROJECT_CODE.md`; RGB in, RGB out, Gamma + CLAHE only).

LWNet's own input layer is already built for 3-channel RGB (`in_c=3`, matching the DRIVE fundus images it was trained on), so no channel-count adapter is needed here either — coincidentally, not because this project trained it that way. See [Appendix A.2](#a2-input-channel-count-for-vesselleision-segmentation) for the earlier single-channel design this supersedes, unaffected by the pretrained/trained reversal in A.1.

### 2.2 Input/output resolution

LWNet's own inference pipeline (`predict_one_image.py` in the upstream repository) resizes the field-of-view-cropped input to a fixed working resolution (512×512 for the vendored DRIVE checkpoint, per its `config.cfg`'s `im_size`), runs the forward pass there, then resizes the resulting probability map back to the original crop size before compositing it into a full-size output — so Stage 03's *externally visible* input/output resolution still matches Stage 02's native, unresized image, even though the model computes internally at 512×512. This is fixed by the vendored checkpoint, not a project-level training-time choice. Lesion Segmentation's output resolution (§3.2) matches Vessel Segmentation's *output* (the full-size composited mask), so the two masks still stack for Local Feature Extraction without a resize step.

### 2.3 Output

Single-channel probability map, `(H, W, 1)`, values in `[0, 1]`, produced by a sigmoid over LWNet's raw logits. A binary vessel mask is obtained by thresholding — LWNet's authors tuned `0.4196` as the optimal threshold on DRIVE's own validation split; this project treats that as a starting default, not a value re-derived on its own data, per the distribution-shift caveat in §1.2.

---

## 3. Lesion Segmentation

Lesion Segmentation (Attention U-Net) is trained within this project, on `datasets/IDRiD/segmentation`. Unlike Vessel Segmentation (§2), which is a pretrained, externally-sourced model run for inference only, Lesion Segmentation has a full training workflow of its own within this project.

| Property | Specification |
|---|---|
| **Model** | Attention U-Net, trained on `datasets/IDRiD/segmentation` (`PROJECT_CODE.md` Models table: "Lesion Segmentation \| Attention U-Net") |
| **Input** | Preprocessed RGB fundus image + the Vessel Segmentation mask (§2), as an auxiliary channel — see §3.1 |
| **Output** | Per-pixel lesion probability map, 4 channels (Microaneurysm, Haemorrhage, Hard Exudate, Soft Exudate) |
| **Output resolution** | Same `H×W` as Vessel Segmentation's output (§2.2), so the two masks stack without a resize |
| **Saved model format** | `.keras`, matching this repo's existing TensorFlow model-loading convention (`image_quality_model.py` / `image_quality_inference.py`). Unlike Vessel Segmentation (§2.1), Lesion Segmentation's architecture (Attention U-Net) and framework are fixed, not left open — see §6. |
| **Inference API** | `pipeline.SegmentationStage` contract (§5) |

### 3.1 Vessel mask as an input to Lesion Segmentation

Vessel information feeds into Lesion Segmentation as an input channel, not merely as an earlier pipeline stage. This is supported by `research_papers/2307.16622v1.pdf`:

> *"Figure 7 illustrates that the removal of blood vessels in the process of segmenting microaneurysms and hemorrhages leads to a reduction in the number of false positives."*
> — `2307.16622v1.pdf` ("Detecting diabetic retinopathy severity through fundus images using an ensemble of classifiers"), §IV.B

Vessels and hemorrhages/microaneurysms are both dark, roughly circular-to-linear structures on a red background, and are a common source of confusion for lesion detectors.

The vessel mask is **concatenated** with the preprocessed image — a **4-channel input to the Attention U-Net (3-channel RGB image + 1-channel vessel mask)**, rather than used to multiplicatively mask the image. Concatenation lets the network learn how much to trust the vessel signal per lesion type, and preserves the original image content where the vessel mask (produced by Stage 03's pretrained LWNet model) is imperfect.

### 3.2 Output resolution

Matches Vessel Segmentation's output resolution (§2.2), so the two masks stack for Local Feature Extraction without a resize step.

### 3.3 Output channel semantics

4 channels, one per lesion category (Microaneurysm, Haemorrhage, Hard Exudate, Soft Exudate), matching IDRiD's four lesion mask folders (`dr_gan++.pdf`: "IDRID (which contain four lesions: microaneurysms, hemorrhages, hard exudates and soft exudates)"). This is a multi-label output — channels are not mutually exclusive, since a pixel can in principle belong to more than one lesion category.

### 3.4 Optic Disc

IDRiD's segmentation subset also ships an Optic Disc mask. It is **not** included in Lesion Segmentation's output: `PROJECT_CODE.md`'s Models table names four lesion categories for this stage, and `dr_gan++.pdf` itself treats Optic Disc as a separate model output from the lesion masks, not a joint one. The OD mask remains available on disk if a future pipeline stage needs it.

---

## 4. Data Flow: Image → CORN Classifier

```
Image (preprocessed)
    │  shape: (H_seg, W_seg, 3) -- RGB, per Stage 02's frozen pipeline (§2.1)
    ▼
Vessel Segmentation  ── Pretrained LWNet (external, inference only,
    not trained within this project, §1.2/§2)
    │  shape: (H_seg, W_seg, 1)
    ▼
Lesion Segmentation  ── Attention U-Net (trained within this project, on
    datasets/IDRiD/segmentation, §1.3/§3 -- the second of two segmentation
    models this project trains, alongside Vessel Segmentation above)
    (consumes: preprocessed RGB image + vessel mask, 4 channels in, §3.1)
    │  shape: (H_seg, W_seg, 4)   [4 output lesion-probability channels]
    ▼
[merge] Local Feature Extraction consumes: preprocessed RGB image + vessel
    mask + lesion mask, concatenated into a single (H_seg, W_seg, 8) tensor
    (3 RGB + 1 vessel + 4 lesion)
    ▼
Local Feature Extraction  ── Adaptive Multi-Kernel CNN
    │  shape: (H_local, W_local, C_local) -- Local Feature Extraction's own
    │  design document, not yet written, fixes this shape and its internal
    │  fusion architecture; out of scope here.
    ▼
Global Feature Extraction  ── Dual-Scale Swin Transformer
    (consumes the preprocessed RGB image directly; parallel to, not sequential
    with, Local Feature Extraction. Any resizing this stage needs is internal
    to it -- Stage 02 does not resize, and no fixed resolution is documented
    here; the final input resolution is configurable and will be selected
    during implementation based on memory and model performance.)
    │  shape: not fixed by this document -- see above.
    ▼
Adaptive Cross-Attention  (Feature Fusion)
    │  shape: reduces to a flat feature vector for the CORN head (fixed by
    │  CORN's own formulation). Exact dimensionality and fusion mechanism
    │  belong to Feature Fusion's own future design document.
    ▼
CORN Classifier
    │  shape: (num_classes - 1) cumulative logits = 4 logits for the 5
    │  APTOS 2019 DR severity grades (0-4), per CORN's standard
    │  rank-consistent ordinal formulation.
```

Only `Image → Vessel Segmentation → Lesion Segmentation` and the hand-off into Local Feature Extraction are formally specified by this document. The Local/Global/Fusion/CORN shapes are shown for traceability but belong to those stages' own future design documents.

---

## 5. Model Interfaces

Both segmentation stages conform to `pipeline.SegmentationStage` (already implemented in this repo: `pipeline/trainable.py`, `pipeline/inference.py`, `pipeline/segmentation.py`), which combines `TrainableStage` + `InferenceStage`:

```python
class VesselSegmentationStage(SegmentationStage):
    # Pretrained LWNet, inference only (§1.2/§2) -- not trained within this
    # project. `load()`/`predict()`/`predict_batch()` are fully functional;
    # `train()`/`save()` have no real training workflow behind them, since
    # this stage never trains its own weights (see note below).
    def train(self, train_data, val_data=None, **kwargs): ...
    def evaluate(self, eval_data, **kwargs): ...
    def save(self, path: str) -> str: ...
    def load(self, path: str) -> "VesselSegmentationStage": ...
    def predict(self, input_data) -> Any: ...          # -> (H, W, 1) probability map
    def predict_batch(self, inputs) -> list[Any]: ...  # -> list of (H, W, 1) maps

class LesionSegmentationStage(SegmentationStage):
    # Trained within this project on datasets/IDRiD/segmentation (§1.3/§3) --
    # train()/evaluate() are fully functional for this stage.
    def train(self, train_data, val_data=None, **kwargs): ...
    def evaluate(self, eval_data, **kwargs): ...
    def save(self, path: str) -> str: ...
    def load(self, path: str) -> "LesionSegmentationStage": ...
    def predict(self, input_data) -> Any: ...          # -> (H, W, 4) probability map
    def predict_batch(self, inputs) -> list[Any]: ...  # -> list of (H, W, 4) maps
```

No method bodies are defined here — this is interface specification only; the comments above describe intended behavior, not implementation.

`pipeline.SegmentationStage` is an `abc.ABC` combining `TrainableStage` + `InferenceStage`; both stages implement the same four-plus-two method contract, but the two stages are not symmetric in what those methods actually do. The orchestrator (§7.1) still calls `load()`/`predict()`/`predict_batch()` identically for both, exactly because that abstraction is what lets it not care which stage is pretrained vs. project-trained — but `train()`/`evaluate()`/`save()` are only meaningfully implemented for Lesion Segmentation. For Vessel Segmentation, `train()`/`save()` have no real backing workflow (there is no training data or training loop for this stage in this project) — the concrete implementation decision (e.g. raising `NotImplementedError`, or being a documented no-op) is left to the code that implements this class, not fixed by this design document. `evaluate()` may optionally run a benchmark of the vendored checkpoint's predictions against a labeled test set if one is ever added, but this is not required for the pipeline to function and no such benchmark exists today.

`evaluate()`'s internals for Lesion Segmentation use Dice/IoU as the evaluation metrics, rather than `evaluation.Evaluator`, which is classification-oriented per its own docstring — concretely `training.metrics.dice_coefficient` / `training.metrics.iou_score` (already implemented, exported via `training.build_metrics("segmentation")`), since Lesion Segmentation is fixed to TensorFlow.

`train_data`/`val_data` for Lesion Segmentation yield `(image, mask)` pairs rather than `(image, label)` pairs — a departure from every classification module in this repo (IQA, and eventually the CORN classifier). No dataset-loader module exists yet for `datasets/IDRiD/segmentation`; it is the first module in the repo that will need one. Vessel Segmentation needs no such dataset loader, since it never trains within this project.

---

## 6. Model Storage Layout

Following this repo's existing `config.py` convention for the Image Quality Assessment module (`EyeQPaths`: `models/image_quality_assessment/`, `results/image_quality_assessment/`):

```
models/
  vessel_segmentation/
    best_model.pth             # vendored pretrained LWNet checkpoint (from
                                # experiments/wnet_drive/model_checkpoint.pth
                                # in the upstream lwnet repository) -- not an
                                # output this project produces, an input this
                                # project consumes. No training_run/ directory:
                                # nothing trains here.
    config.cfg                 # LWNet's own architecture/inference config,
                                # vendored alongside the checkpoint
  lesion_segmentation/
    best_model.keras          # TensorFlow -- this project's own trained output
    training_run/             # checkpoints/, logs/ -- per training.TrainingConfig

results/
  lesion_segmentation/        # evaluation_report.json, dice/IoU plots
```

Vessel Segmentation is a special case in this table, unlike every other trainable module: it has no `training_run/` and no `results/` directory, because nothing about it is trained or evaluated within this project — `models/vessel_segmentation/` holds a vendored artifact, not a training output. Its file format is fixed to PyTorch's `.pth`, not left open, since adopting a specific external checkpoint (rather than training our own) settles the framework question by construction. `training.Trainer`, `training.get_loss(...)`, and `training.build_metrics("segmentation")` (TensorFlow/Keras-based; confirmed directly in `training/trainer.py`, `training/losses.py`, `training/metrics.py`) do not apply to Vessel Segmentation at all. `experiment_manager.py` and `dataset_staging.py` remain framework-agnostic and continue to apply to every other trainable module. Lesion Segmentation's architecture (Attention U-Net) and `.keras` format remain fixed, consistent with the rest of this TensorFlow-based repository.

Corresponding environment variables (to be added to `.env_sample` / `config.py` at implementation time, not by this document): `VESSEL_SEG_MODEL_DIR` (defaults to `models/vessel_segmentation/`, holding the vendored checkpoint), `LESION_SEG_MODEL_DIR`, `LESION_SEG_RESULTS_DIR`, mirroring `IQA_MODEL_DIR`/`IQA_RESULTS_DIR`. No `VESSEL_SEG_RESULTS_DIR` is needed unless a future benchmark evaluation is added.

`config.py`'s existing generic helpers, `dataset_raw_dir(name)` / `dataset_processed_dir(name)`, resolve to `datasets/<name>/raw` / `datasets/<name>/processed` — a flat one-level convention, unaffected by Vessel Segmentation's pretrained-model reversal since it needs no dataset directory of its own. IDRiD in this repo is subdivided by task (`grading/`, `localization/`, `segmentation/`), which the generic helper cannot address as-is; the intended direction is to extend those helpers with an optional `subtask` parameter (e.g. `dataset_raw_dir("IDRiD", subtask="segmentation")`), backward-compatible with every existing call site. This document does not modify `config.py`; it records the direction a future implementation should take.

---

## 7. Training and Inference Workflow

### 7.0 Training workflow

- **Vessel Segmentation has no training workflow in this project.** There is no dataset loader, no training loop, and no Colab training notebook for this stage — `models/vessel_segmentation/best_model.pth` (§6) is vendored directly from the upstream `lwnet` repository's `experiments/wnet_drive/model_checkpoint.pth`, trained by LWNet's original authors on DRIVE, outside this project entirely.
- **Lesion Segmentation has its own full training workflow**: a dataset loader for `datasets/IDRiD/segmentation` (image + 4-channel lesion mask pairs, §5) feeds `training.Trainer`, in its own dedicated Colab notebook, mirroring `train_image_quality.py`'s overall structure (build model → configure a trainer → fit → export best weights).
- **Interaction between the two:** training Lesion Segmentation requires running Vessel Segmentation's pretrained model in inference mode over every IDRiD/segmentation training image first, to produce the vessel-mask input channel (§3.1). This is a checkpoint-availability dependency, not a training-order dependency: Vessel Segmentation never trains, so Lesion Segmentation only needs the vendored checkpoint to already be in place (`models/vessel_segmentation/best_model.pth`) before its own training run starts, rather than waiting for another training job to finish.

### 7.1 Inference workflow

Per `PROJECT_CODE.md`'s Deployment Requirement ("The final project must integrate all inference modules into a single end-to-end pipeline capable of accepting one retinal fundus image and producing the complete diabetic retinopathy analysis"), the end-to-end orchestrator:

1. Loads every stage's model once, at process start — mirroring `image_quality_inference.py`'s existing `model=None` reuse pattern. `vessel_stage.load(VESSEL_SEG_MODEL_DIR/best_model.pth)` and `lesion_stage.load(LESION_SEG_MODEL_DIR/best_model.keras)` use the identical `pipeline.SegmentationStage.load()` method (§5), even though one loads a vendored pretrained artifact and the other loads this project's own trained output — `load()`'s contract is the file path and the returned stage object, not the artifact's provenance or framework, so the orchestrator does not need to special-case either stage.
2. For a single incoming fundus image: runs the IQA gate → Preprocessing → `vessel_stage.predict(preprocessed_image)` → `lesion_stage.predict(preprocessed_image, vessel_mask)` → hands both masks (plus the preprocessed image) to Local Feature Extraction, continuing down the chain in §4.
3. Needs only the `pipeline.SegmentationStage` contract (`load`/`predict`/`predict_batch`) — no knowledge of Vessel U-Net vs. Attention U-Net internals, per the model-agnostic orchestration goal `pipeline/__init__.py`'s own docstring states.
4. For batch/offline use (e.g. generating masks over APTOS 2019, §1.4), uses the same stages' `predict_batch()`, following the batching pattern already established by `image_quality_inference.py`'s `predict_quality_batch`.

The orchestrator itself is a composed list of `pipeline.*Stage` objects run in sequence, rather than one monolithic script — this reuses the `pipeline/` package already built in this repo for exactly this purpose (per its own `__init__.py` docstring), and keeps each stage swappable/testable in isolation.

---

## 8. Source Index

- **Project documents:** `PROJECT_CODE.md`, `IMPLEMENTATION_PLAN.md`, `PROJECT_STRUCTURE.md`.
- **External model source:** [`lwnet`](https://github.com/agaldran/lwnet) (Adrián Galdrán et al., "The Little W-Net That Could: State-of-the-Art Retinal Vessel Segmentation with Minimalistic Models," [arXiv:2009.01907](https://arxiv.org/abs/2009.01907), Sep. 2020), MIT License. Inspected in full under `external/lwnet/` (temporary, local-only copy) before integration; the vendored checkpoint is `experiments/wnet_drive/model_checkpoint.pth` (the `wnet` architecture, trained on DRIVE by the paper's own authors — not retrained here).
- **Papers cited:**
  - `research_papers/dr_gan++.pdf` — Zhou et al., "DR-GAN: Conditional Generative Adversarial Network for Fine-Grained Lesion Synthesis on Diabetic Retinopathy Images." Used for: U-Net for vessel/lesion/OD segmentation; DRIVE as a standard vessel-mask dataset (cited precedent only — DRIVE is not a project dataset, see §1.1/§1.2); IDRiD's four lesion categories.
  - `research_papers/2307.16622v1.pdf` — Popescu, Groza, Damian, "Detecting diabetic retinopathy severity through fundus images using an ensemble of classifiers." Used for: vessel-removal-before-lesion-segmentation false-positive reduction; U-Net for per-lesion segmentation.
  - `research_papers/A_Systematic_Review_on_Fundus_Image-Based_Diabetic_Retinopathy_Detection_and_Grading_Current_Status_and_Future_Directions.pdf` — Ikram et al. Used for: standard vessel-segmentation datasets (DRIVE, STARE, CHASE-DB1, HRF — cited precedent only, none are project datasets); green-channel-extraction rationale (historical — see [Appendix A.2](#a2-input-channel-count-for-vesselleision-segmentation), no longer the canonical design); IDRiD/Porwal citation; U-Net lesion-segmentation precedents.
- **Repository code referenced (not modified):** `config.py`, `pipeline/` (all four modules), `training/` (`trainer.py`, `metrics.py`, `losses.py`), `evaluation/evaluator.py`, `image_preprocessing.py`, `image_quality_model.py`, `image_quality_inference.py`, `train_image_quality.py`, `swin_transformer.py`, `.env_sample`, and the `datasets/` directory tree.
- Adaptive Multi-Kernel CNN, Dual-Scale Swin Transformer, Adaptive Cross-Attention, and CORN are named in `PROJECT_CODE.md`'s Models table; no paper in `research_papers/` describes their internals, so those four stages rest on `PROJECT_CODE.md` alone among the sources available to this document.

---

## Appendix: Design History

This appendix records alternatives that were considered, and in two cases actively adopted and later superseded, before arriving at the architecture in §§1–7. It is the **only** place this document (or any other governing document) records reasoning behind decisions that are no longer current — nothing in §§1–7 above should be read as reflecting these superseded designs.

### A.1 Vessel Segmentation model source

This decision has been revisited twice. Both earlier designs are retained here in chronological order; **neither should be read as describing the current architecture** (§1.2/§2).

**Design 1 (superseded) — generic pretrained, inference-only U-Net.** The original version of this document used an unspecified pretrained, externally-sourced U-Net for Vessel Segmentation, run for inference only, with no training workflow of its own. That design was adopted specifically because no dataset then approved for this project (EyeQ, APTOS 2019, IDRiD) shipped pixel-level vessel masks, and adding a new dataset for that sole purpose was, at the time, considered out of scope.

Before settling on that pretrained approach, two other alternatives were considered and set aside at the time:

- **Adding a new vessel-mask dataset** (e.g. DRIVE, the smallest and most-cited public option per `dr_gan++.pdf` and the systematic review) to train Vessel Segmentation within this project. This would have produced real pixel-level ground truth and matched `dr_gan++.pdf`'s own precedent, but required adding a dataset outside the then-approved list.
- **Deriving weak/pseudo vessel labels** from an approved dataset (e.g. APTOS 2019) via a classical vesselness filter (Frangi, top-hat), used as noisy training targets. This stayed within the approved dataset list, but risked training against unverified, filter-generated ground truth — in tension with `PROJECT_CODE.md`'s rule against fabricating evaluation results.

**Design 2 (superseded) — Baseline U-Net trained within this project on DRIVE + CHASE_DB1.** Design 1 was revisited: DRIVE and CHASE_DB1 were added as approved project datasets specifically to make Vessel Segmentation trainable within this project, producing real pixel-level ground truth and matching `dr_gan++.pdf`'s own precedent. The model was deliberately named "Baseline U-Net" rather than "Standard U-Net" so that its exact architecture could evolve without a rename. Model storage was left framework-agnostic (`best_model[.ext]`) pending that decision. This design was implemented across every governing document, but no code was written against it before it was revisited again.

**Design 3 (current) — pretrained LWNet, inference only.** Design 2 was revisited and reversed: Vessel Segmentation now runs [`lwnet`](https://github.com/agaldran/lwnet)'s vendored `wnet_drive` checkpoint (MIT-licensed, ~70k parameters, competitive published DRIVE performance — see §8), trained by its original authors, for inference only. This was an intentional, explicit decision, not a rediscovery of Design 1: DRIVE and CHASE_DB1 are removed from the project's dataset list again (§1.1) — not because vessel-mask datasets are unavailable (Design 2 already proved they could be added), but because using an already-trained, published, minimalistic model avoids retraining a segmentation network from scratch for a stage the wider pipeline only consumes as an input signal. See §1.2/§2 for the current design, including the distribution-shift caveat it introduces (Stage 02's Gamma+CLAHE output was not part of LWNet's own training distribution) that neither Design 1 nor Design 2 needed to document.

### A.2 Input channel count for Vessel/Lesion Segmentation

**Superseded.** An earlier version of this document used a single-channel, green-channel-derived canonical image (rather than RGB) as the input every downstream stage consumed, on the grounds that `PROJECT_CODE.md`'s Preprocessing section at the time named Green Channel Extraction as a required pipeline step, and that green-channel isolation is a well-established technique for concentrating vessel/lesion contrast in fundus imaging (per the systematic review cited in §8).

A 3-channel RGB input (matching `image_preprocessing.py`'s CLAHE+Gamma output) was considered as an alternative at the time and set aside, since single-channel was the then-required design.

**This alternative was revisited and adopted.** Stage 02's frozen output is now RGB (`PROJECT_CODE.md`), and Green Channel Extraction is explicitly excluded from Stage 02. See §2.1 for the current design and its rationale (no channel-count adapter needed, since LWNet's own input layer already expects 3-channel RGB). This section is retained for historical context on why the original design differed, and is unaffected by A.1's pretrained/trained history — the RGB-vs-single-channel decision and the pretrained-vs-trained decision are independent of each other.

### A.3 Vessel-mask integration mechanism

Multiplicative masking (zeroing vessel pixels in the image before Lesion Segmentation) was considered as an alternative to channel-concatenation (§3.1). It was set aside because it destroys image information irreversibly wherever the vessel mask has a false positive, whereas concatenation lets the trainable Attention U-Net learn how much to trust the vessel signal. This decision is unaffected by the RGB/single-channel change in A.2 — only the channel *count* changed (2→4), not the concatenation mechanism itself.

### A.4 Optic Disc channel

Including IDRiD's Optic Disc mask as a 5th output channel of Lesion Segmentation was considered, since it ships in the same dataset subset at no extra data cost. It was set aside because `PROJECT_CODE.md`'s Models table and `dr_gan++.pdf` both treat Optic Disc as separate from the four lesion categories.

### A.5 Config-layer path helpers

Extending `config.py`'s generic `dataset_raw_dir`/`dataset_processed_dir` helpers with an optional `subtask` parameter was preferred over adding a hardcoded `IDRiDPaths` dataclass (mirroring `EyeQPaths`), since IDRiD already needs three parallel path pairs (grading/localization/segmentation) and a generic helper avoids that duplication. DRIVE and CHASE_DB1, added later, reinforce the case for the generic-helper direction: both resolve cleanly through the existing `dataset_raw_dir(name)`/`dataset_processed_dir(name)` helpers with no subtask nesting needed at all, requiring zero new config-layer code beyond what already exists.

### A.6 Inference orchestration style

A composed list of `pipeline.*Stage` objects, run by a small orchestrator, was preferred over a monolithic script mirroring `train_hybrid_model.py`'s style, since the `pipeline/` package already exists in this repository specifically to let stages be composed without the orchestrator needing to know which are pretrained vs. project-trained. This reasoning is unaffected by A.1's back-and-forth history — it is exactly what makes Design 2 → Design 3's reversal (trained Baseline U-Net → pretrained LWNet) a documentation-and-model-storage change rather than an orchestrator change: `pipeline.SegmentationStage.load()`/`predict()`/`predict_batch()` (§5, §7.1) work identically regardless of which design is current.
