# Segmentation Stage — Architecture Design Document

**Status:** Frozen architecture specification for pipeline stages 3 (Vessel Segmentation) and 4 (Lesion Segmentation), plus the interface/shape contract connecting them to stages 5–8 (Local Feature Extraction → Global Feature Extraction → Feature Fusion → Ordinal Classification). **Both Vessel Segmentation and Lesion Segmentation are trained within this project** — Vessel Segmentation on DRIVE + CHASE_DB1, Lesion Segmentation on IDRiD's segmentation subset. An earlier design used a pretrained, inference-only model for Vessel Segmentation instead; that design is superseded and retained only in the Appendix (Design History) for context.

**Sources consulted:** `PROJECT_CODE.md`, `IMPLEMENTATION_PLAN.md`, the `datasets/` directory tree, the existing `pipeline/`, `training/`, `evaluation/`, `config.py` code, and the papers in `research_papers/`.

Design alternatives that were considered and set aside are recorded in the [Appendix: Design History](#appendix-design-history), not in the sections below — the sections below describe only the current, frozen architecture.

---

## 1. Dataset Assignment

### 1.1 Datasets on disk (and to be added)

| Path | Contents |
|---|---|
| `datasets/APTOS2019/raw/{train_images,test_images}` | Fundus images, image-level DR grade labels |
| `datasets/EyePACS/raw/{train,test}` + `trainLabels.csv` | Fundus images, image-level DR grade labels — historical only; used once to reconstruct EyeQ, not used by this pipeline (§1.5) |
| `datasets/EyeQ/raw/{train,test}` | Fundus images, image-level quality labels |
| `datasets/DRIVE/raw/` | Fundus images + pixel-level vessel masks (`1st_manual`). New, approved specifically for Vessel Segmentation training. Not yet placed on disk. |
| `datasets/CHASE_DB1/raw/` | Fundus images + pixel-level vessel masks. New, approved specifically for Vessel Segmentation training. Not yet placed on disk. |
| `datasets/IDRiD/grading/raw/` | Images + image-level DR/DME grade labels |
| `datasets/IDRiD/localization/raw/` | Images + Optic Disc / Fovea center point coordinates (not pixel masks) |
| `datasets/IDRiD/segmentation/raw/` | Reserved for the IDRiD Segmentation subset (Microaneurysm, Haemorrhage, Hard Exudate, Soft Exudate, Optic Disc masks); not yet populated |

Of the datasets already on disk (EyeQ, APTOS2019, IDRiD), none ship pixel-level retinal vessel masks — this is why DRIVE and CHASE_DB1 were added specifically for Stage 3.

### 1.2 Vessel Segmentation

Vessel Segmentation trains a **Baseline U-Net** within this project, on **DRIVE + CHASE_DB1**. Both datasets are run through Stage 02's own RGB → Gamma → CLAHE pipeline before training, so the model trains on the same input distribution it sees at inference time on EyeQ/APTOS/IDRiD-derived images — this project does not train it on the raw, unprocessed DRIVE/CHASE_DB1 originals.

The model is named "Baseline U-Net" rather than "Standard U-Net" deliberately — the exact architecture may evolve during implementation, and this document (and every other document referencing it) should not need renaming when it does.

An earlier design used an externally pretrained, inference-only U-Net instead, specifically because no dataset then approved for this project shipped vessel masks. That constraint no longer applies now that DRIVE and CHASE_DB1 are approved datasets. See [Appendix A.1](#a1-vessel-segmentation-model-source) for the full history of that earlier decision and why it was revisited.

### 1.3 Lesion Segmentation

Lesion Segmentation (Attention U-Net) trains on `datasets/IDRiD/segmentation` (per `PROJECT_CODE.md`'s Datasets table, corroborated by `dr_gan++.pdf` and the systematic review's IDRiD citation).

`datasets/IDRiD/segmentation/raw/` is currently empty. Per `PROJECT_CODE.md`'s Dataset Handling rule (no automatic downloads; always use datasets already inside `datasets/`), the IDRiD Segmentation subset's files (Original Images + the Groundtruth mask folders) must be placed there before training code is run.

### 1.4 Training vs. inference usage per dataset

- **Vessel Segmentation** is trained on DRIVE + CHASE_DB1 (train/val/test splits within those two datasets). It additionally runs **inference** over every other dataset it touches in the pipeline — EyeQ (if ever needed), APTOS2019, and IDRiD (including `datasets/IDRiD/segmentation` itself, to produce the vessel-mask input channel Lesion Segmentation's training requires) — none of which are used to train it.
- **Lesion Segmentation** is trained on `datasets/IDRiD/segmentation`. It runs **inference** (not training) on APTOS 2019, since it ships only image-level DR grade labels, not masks.

### 1.5 EyePACS (historical only)

EyePACS was used once, historically, to reconstruct the official EyeQ dataset (see `PROJECT_CODE.md`'s Datasets section). The reconstructed EyeQ dataset is the dataset actually used for Image Quality Assessment. EyePACS itself is **not** part of the Segmentation stage, or any other part of the implemented pipeline — it is not used for training, preprocessing, or inference here, and is not required to reproduce or run this repository.

### 1.6 Datasets never used for segmentation

**EyeQ** is scoped to Image Quality Assessment only (`PROJECT_CODE.md`'s Datasets table). It carries no lesion/vessel annotation and is not part of the Segmentation stage's input.

### 1.7 Summary

| Dataset | Role |
|---|---|
| DRIVE | Trains Vessel Segmentation (Stage 3), alongside CHASE_DB1 |
| CHASE_DB1 | Trains Vessel Segmentation (Stage 3), alongside DRIVE |
| IDRiD (`segmentation` subset) | Trains Lesion Segmentation (Stage 4) |
| APTOS 2019 | Inference-only for both Vessel Segmentation and Lesion Segmentation |
| EyeQ | Not used for segmentation |
| EyePACS | Not part of this project (§1.5) — historical only, used once to reconstruct EyeQ |

Every image in `datasets/IDRiD/segmentation` also passes through Vessel Segmentation inference, since Lesion Segmentation's training input requires the vessel mask (§3.1).

---

## 2. Vessel Segmentation

| Property | Specification |
|---|---|
| **Model** | Baseline U-Net, trained within this project (`PROJECT_CODE.md`'s Models table) |
| **Training** | DRIVE + CHASE_DB1, run through Stage 02's own preprocessing before training (§1.2) |
| **Input** | Preprocessed RGB fundus image, single image per forward pass (batchable); see §2.1 |
| **Output** | Per-pixel vessel probability map, 1 channel |
| **Output resolution** | Matches input resolution (U-Net is fully convolutional); see §2.2 |
| **Saved model format** | Framework-agnostic (see §6) — not assumed to be TensorFlow's `.keras` convention, unlike Image Quality Assessment (`image_quality_model.py` / `image_quality_inference.py`). The exact format depends on the final framework decision for this stage's model. |
| **Inference API** | `pipeline.SegmentationStage` contract (§5) |

### 2.1 Input channel count

The canonical preprocessed image used throughout this pipeline (Vessel Segmentation, Lesion Segmentation, Local/Global Feature Extraction, and elsewhere "the preprocessed image" is referenced) is **RGB, 3 channels** — Stage 02's frozen output format (`PROJECT_CODE.md`; RGB in, RGB out, Gamma + CLAHE only).

Because Vessel Segmentation is trained within this project directly on Stage 02's RGB output, no channel-count adapter is needed for this stage: the model's own input layer is simply built for 3 channels from the start. This is a direct consequence of training the model ourselves rather than adapting to an externally fixed pretrained checkpoint's own input convention — see [Appendix A.2](#a2-input-channel-count-for-vesselleision-segmentation) for the earlier single-channel design this supersedes.

### 2.2 Input/output resolution

Vessel Segmentation's input/output resolution is a training-time hyperparameter for this project's own Baseline U-Net, not fixed by an external checkpoint. It is decoupled from the 224×224 convention used elsewhere in this pipeline (EfficientNetB0, the Swin blocks, the GAN, IQA), since that convention was set for whole-image classification and risks losing fine vessel/microaneurysm detail. U-Net is fully convolutional, so this stage can train and run at a resolution chosen independently of the classification branch. The specific value will be selected during implementation, based on Colab memory constraints and preservation of thin-vessel detail — not documented as a fixed number here. Lesion Segmentation's training resolution (§3.2) is chosen to match whatever Vessel Segmentation ultimately uses, so the two masks stack for Local Feature Extraction without a resize step.

### 2.3 Output

Single-channel probability map, `(H, W, 1)`, values in `[0, 1]`. A binary vessel mask is obtained by thresholding (e.g. 0.5) as a downstream consumption step, not part of the model's own output.

---

## 3. Lesion Segmentation

Lesion Segmentation (Attention U-Net) is the second segmentation model trained within this project, alongside Vessel Segmentation (§2) — both are now trained within this project; neither is pretrained/inference-only.

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

The vessel mask is **concatenated** with the preprocessed image — a **4-channel input to the Attention U-Net (3-channel RGB image + 1-channel vessel mask)**, rather than used to multiplicatively mask the image. Concatenation lets the network learn how much to trust the vessel signal per lesion type, and preserves the original image content where the vessel mask (produced by this project's own Vessel Segmentation model) is imperfect.

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
Vessel Segmentation  ── Baseline U-Net (trained within this project on
    DRIVE + CHASE_DB1, §1.2/§2)
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
    # Trained within this project on DRIVE + CHASE_DB1 (§1.2/§2) --
    # train()/evaluate() are fully functional for this stage, the same as
    # LesionSegmentationStage below.
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

`pipeline.SegmentationStage` is an `abc.ABC` combining `TrainableStage` + `InferenceStage`. Both stages now conform to the same contract with every method fully functional; the orchestrator (§7.1) calls `load()`/`predict()`/`predict_batch()` identically for both, and no longer needs to special-case either as pretrained-only.

`evaluate()`'s internals for both stages use Dice/IoU as the evaluation metrics, rather than `evaluation.Evaluator`, which is classification-oriented per its own docstring. For Lesion Segmentation (fixed to TensorFlow), this concretely means `training.metrics.dice_coefficient` / `training.metrics.iou_score` (already implemented, exported via `training.build_metrics("segmentation")`). For Vessel Segmentation, the same Dice/IoU *metrics* are used conceptually, but the concrete implementation depends on its still-open framework choice (§6) -- `training.metrics` applies directly only if that choice is TensorFlow.

`train_data`/`val_data` for **both** Vessel Segmentation and Lesion Segmentation yield `(image, mask)` pairs rather than `(image, label)` pairs — a departure from every classification module in this repo (IQA, and eventually the CORN classifier). No dataset-loader module exists yet for either `datasets/DRIVE` + `datasets/CHASE_DB1` or `datasets/IDRiD/segmentation`; these are the first two modules in the repo that will need one.

---

## 6. Model Storage Layout

Following this repo's existing `config.py` convention for the Image Quality Assessment module (`EyeQPaths`: `models/image_quality_assessment/`, `results/image_quality_assessment/`):

```
models/
  vessel_segmentation/
    best_model[.ext]          # this project's own trained output; extension is
                                # framework-dependent (e.g. best_model for a
                                # framework-neutral reference, or best_model.pt
                                # if implemented in PyTorch) -- deliberately not
                                # fixed to .keras here, see note below
    training_run/              # checkpoints/, logs/ -- per training.TrainingConfig,
                                # if the final framework is TensorFlow; an
                                # equivalent structure otherwise
  lesion_segmentation/
    best_model.keras          # TensorFlow -- Lesion Segmentation's framework is
                                # fixed, unlike Vessel Segmentation's (see below)
    training_run/             # checkpoints/, logs/ -- per training.TrainingConfig

results/
  vessel_segmentation/        # evaluation_report.json, dice/IoU plots
  lesion_segmentation/        # evaluation_report.json, dice/IoU plots
```

Both segmentation modules follow the identical layout *shape* — Vessel Segmentation is no longer a special case with no `training_run/` or `results/` directory, since it is a genuinely trained-and-evaluated module like every other one in this table. Its exact file **extension** is deliberately left open: `PROJECT_CODE.md`'s Models table already documents Vessel Segmentation as "Baseline U-Net" rather than "Standard U-Net" specifically because its exact architecture may evolve, and the documentation should not force TensorFlow (or any framework) on that decision. `training.Trainer`, `training.get_loss(...)`, and `training.build_metrics("segmentation")` are TensorFlow/Keras-based (confirmed directly in `training/trainer.py`, `training/losses.py`, `training/metrics.py`) and apply to Vessel Segmentation's training only if TensorFlow ends up being the framework chosen for it; if not, an equivalent training loop is implemented instead, without changing this document's `predict()`/`SegmentationStage` contract. `experiment_manager.py` and `dataset_staging.py` are already framework-agnostic (pure filesystem/metadata logic, no TensorFlow import) and apply regardless of which framework is ultimately chosen. Lesion Segmentation's framework is not left open the same way — its architecture (Attention U-Net) and `.keras` format are fixed, consistent with the rest of this TensorFlow-based repository.

Corresponding environment variables (to be added to `.env_sample` / `config.py` at implementation time, not by this document): `VESSEL_SEG_MODEL_DIR` (defaults to `models/vessel_segmentation/`), `VESSEL_SEG_RESULTS_DIR` (defaults to `results/vessel_segmentation/`), `LESION_SEG_MODEL_DIR`, `LESION_SEG_RESULTS_DIR`, mirroring `IQA_MODEL_DIR`/`IQA_RESULTS_DIR`.

`config.py`'s existing generic helpers, `dataset_raw_dir(name)` / `dataset_processed_dir(name)`, resolve to `datasets/<name>/raw` / `datasets/<name>/processed` — a flat one-level convention. This covers DRIVE and CHASE_DB1 directly, with no changes needed. IDRiD in this repo is subdivided by task (`grading/`, `localization/`, `segmentation/`), which the generic helper cannot address as-is; the intended direction is to extend those helpers with an optional `subtask` parameter (e.g. `dataset_raw_dir("IDRiD", subtask="segmentation")`), backward-compatible with every existing call site. This document does not modify `config.py`; it records the direction a future implementation should take.

---

## 7. Training and Inference Workflow

### 7.0 Training workflow

- **Vessel Segmentation has its own full training workflow**, the same *shape* as every other trainable module in this repo: a dataset loader for `datasets/DRIVE` + `datasets/CHASE_DB1` (image + single-channel vessel-mask pairs, both datasets run through Stage 02's preprocessing first, §1.2) feeds a training loop, mirroring `train_image_quality.py`'s overall structure (build model → configure a trainer → fit → export best weights), in a dedicated Colab notebook per `PROJECT_CODE.md`'s Colab-only training policy. Whether that training loop is literally `training.Trainer` depends on the framework decision noted in §6 -- `training.Trainer` is TensorFlow/Keras-specific, so it applies directly only if Vessel Segmentation is implemented in TensorFlow; an equivalent loop is used otherwise, without changing anything about this stage's `predict()`/`SegmentationStage` contract or its dataset/evaluation structure.
- **Lesion Segmentation has its own full training workflow**, structured identically: a dataset loader for `datasets/IDRiD/segmentation` (image + 4-channel lesion mask pairs, §5) feeds `training.Trainer` the same way, in its own dedicated Colab notebook.
- **Interaction between the two at training time:** training Lesion Segmentation requires running Vessel Segmentation's **trained** model in inference mode over every IDRiD/segmentation training image first, to produce the vessel-mask input channel (§3.1). This means **Vessel Segmentation's training must complete before Lesion Segmentation training can begin** — a real sequencing dependency, not merely a conceptual one, now that Vessel Segmentation is itself trained within this project rather than always available from an external checkpoint. Vessel Segmentation's trained model participates in Lesion Segmentation's training pipeline as a fixed, frozen feature source, without itself being updated by that training run.

### 7.1 Inference workflow

Per `PROJECT_CODE.md`'s Deployment Requirement ("The final project must integrate all inference modules into a single end-to-end pipeline capable of accepting one retinal fundus image and producing the complete diabetic retinopathy analysis"), the end-to-end orchestrator:

1. Loads every stage's model once, at process start — mirroring `image_quality_inference.py`'s existing `model=None` reuse pattern. `vessel_stage.load(VESSEL_SEG_MODEL_DIR/best_model[.ext])` and `lesion_stage.load(LESION_SEG_MODEL_DIR/best_model.keras)` use the identical `pipeline.SegmentationStage.load()` method (§5) — both now load this project's own trained output, with no distinction between the two at the orchestrator level. `load()`'s contract is the file path and the returned stage object, not a specific file extension, so Vessel Segmentation's still-undecided format (§6) does not affect the orchestrator.
2. For a single incoming fundus image: runs the IQA gate → Preprocessing → `vessel_stage.predict(preprocessed_image)` → `lesion_stage.predict(preprocessed_image, vessel_mask)` → hands both masks (plus the preprocessed image) to Local Feature Extraction, continuing down the chain in §4.
3. Needs only the `pipeline.SegmentationStage` contract (`load`/`predict`/`predict_batch`) — no knowledge of Vessel U-Net vs. Attention U-Net internals, per the model-agnostic orchestration goal `pipeline/__init__.py`'s own docstring states.
4. For batch/offline use (e.g. generating masks over APTOS 2019, §1.4), uses the same stages' `predict_batch()`, following the batching pattern already established by `image_quality_inference.py`'s `predict_quality_batch`.

The orchestrator itself is a composed list of `pipeline.*Stage` objects run in sequence, rather than one monolithic script — this reuses the `pipeline/` package already built in this repo for exactly this purpose (per its own `__init__.py` docstring), and keeps each stage swappable/testable in isolation.

---

## 8. Source Index

- **Project documents:** `PROJECT_CODE.md`, `IMPLEMENTATION_PLAN.md`, `PROJECT_STRUCTURE.md`.
- **Papers cited:**
  - `research_papers/dr_gan++.pdf` — Zhou et al., "DR-GAN: Conditional Generative Adversarial Network for Fine-Grained Lesion Synthesis on Diabetic Retinopathy Images." Used for: U-Net for vessel/lesion/OD segmentation; DRIVE for vessel masks (now an actively used project dataset, not merely a cited precedent — see §1.2); IDRiD's four lesion categories.
  - `research_papers/2307.16622v1.pdf` — Popescu, Groza, Damian, "Detecting diabetic retinopathy severity through fundus images using an ensemble of classifiers." Used for: vessel-removal-before-lesion-segmentation false-positive reduction; U-Net for per-lesion segmentation.
  - `research_papers/A_Systematic_Review_on_Fundus_Image-Based_Diabetic_Retinopathy_Detection_and_Grading_Current_Status_and_Future_Directions.pdf` — Ikram et al. Used for: standard vessel-segmentation datasets (DRIVE, STARE, CHASE-DB1, HRF — DRIVE and CHASE_DB1 are now this project's own approved training datasets, per this citation's own precedent); green-channel-extraction rationale (historical — see [Appendix A.2](#a2-input-channel-count-for-vesselleision-segmentation), no longer the canonical design); IDRiD/Porwal citation; U-Net lesion-segmentation precedents.
- **Repository code referenced (not modified):** `config.py`, `pipeline/` (all four modules), `training/` (`trainer.py`, `metrics.py`, `losses.py`), `evaluation/evaluator.py`, `image_preprocessing.py`, `image_quality_model.py`, `image_quality_inference.py`, `train_image_quality.py`, `swin_transformer.py`, `.env_sample`, and the `datasets/` directory tree.
- Adaptive Multi-Kernel CNN, Dual-Scale Swin Transformer, Adaptive Cross-Attention, and CORN are named in `PROJECT_CODE.md`'s Models table; no paper in `research_papers/` describes their internals, so those four stages rest on `PROJECT_CODE.md` alone among the sources available to this document.

---

## Appendix: Design History

This appendix records alternatives that were considered, and in two cases actively adopted and later superseded, before arriving at the architecture in §§1–7. It is the **only** place this document (or any other governing document) records reasoning behind decisions that are no longer current — nothing in §§1–7 above should be read as reflecting these superseded designs.

### A.1 Vessel Segmentation model source

**Superseded.** An earlier version of this document used a pretrained, externally-sourced U-Net for Vessel Segmentation, run for inference only, with no training workflow of its own. That design was adopted specifically because no dataset then approved for this project (EyeQ, APTOS 2019, IDRiD) shipped pixel-level vessel masks, and adding a new dataset for that sole purpose was, at the time, considered out of scope.

Before settling on that pretrained approach, two other alternatives were considered and set aside at the time:

- **Adding a new vessel-mask dataset** (e.g. DRIVE, the smallest and most-cited public option per `dr_gan++.pdf` and the systematic review) to train Vessel Segmentation within this project. This would have produced real pixel-level ground truth and matched `dr_gan++.pdf`'s own precedent, but required adding a dataset outside the then-approved list.
- **Deriving weak/pseudo vessel labels** from an approved dataset (e.g. APTOS 2019) via a classical vesselness filter (Frangi, top-hat), used as noisy training targets. This stayed within the approved dataset list, but risked training against unverified, filter-generated ground truth — in tension with `PROJECT_CODE.md`'s rule against fabricating evaluation results.

**This alternative was revisited and adopted.** DRIVE (and CHASE_DB1) are now approved project datasets, used specifically to train Vessel Segmentation within this project. See §1.2/§2 for the current, superseding design. This section is retained for historical context on why the original design differed, and should not be read as describing the current architecture.

### A.2 Input channel count for Vessel/Lesion Segmentation

**Superseded.** An earlier version of this document used a single-channel, green-channel-derived canonical image (rather than RGB) as the input every downstream stage consumed, on the grounds that `PROJECT_CODE.md`'s Preprocessing section at the time named Green Channel Extraction as a required pipeline step, and that green-channel isolation is a well-established technique for concentrating vessel/lesion contrast in fundus imaging (per the systematic review cited in §8).

A 3-channel RGB input (matching `image_preprocessing.py`'s CLAHE+Gamma output) was considered as an alternative at the time and set aside, since single-channel was the then-required design.

**This alternative was revisited and adopted.** Stage 02's frozen output is now RGB (`PROJECT_CODE.md`), and Green Channel Extraction is explicitly excluded from Stage 02. See §2.1 for the current, superseding design and its rationale (no channel-count adapter needed once Vessel Segmentation is trained directly on Stage 02's own RGB output). This section is retained for historical context on why the original design differed.

### A.3 Vessel-mask integration mechanism

Multiplicative masking (zeroing vessel pixels in the image before Lesion Segmentation) was considered as an alternative to channel-concatenation (§3.1). It was set aside because it destroys image information irreversibly wherever the vessel mask has a false positive, whereas concatenation lets the trainable Attention U-Net learn how much to trust the vessel signal. This decision is unaffected by the RGB/single-channel change in A.2 — only the channel *count* changed (2→4), not the concatenation mechanism itself.

### A.4 Optic Disc channel

Including IDRiD's Optic Disc mask as a 5th output channel of Lesion Segmentation was considered, since it ships in the same dataset subset at no extra data cost. It was set aside because `PROJECT_CODE.md`'s Models table and `dr_gan++.pdf` both treat Optic Disc as separate from the four lesion categories.

### A.5 Config-layer path helpers

Extending `config.py`'s generic `dataset_raw_dir`/`dataset_processed_dir` helpers with an optional `subtask` parameter was preferred over adding a hardcoded `IDRiDPaths` dataclass (mirroring `EyeQPaths`), since IDRiD already needs three parallel path pairs (grading/localization/segmentation) and a generic helper avoids that duplication. DRIVE and CHASE_DB1, added later, reinforce the case for the generic-helper direction: both resolve cleanly through the existing `dataset_raw_dir(name)`/`dataset_processed_dir(name)` helpers with no subtask nesting needed at all, requiring zero new config-layer code beyond what already exists.

### A.6 Inference orchestration style

A composed list of `pipeline.*Stage` objects, run by a small orchestrator, was preferred over a monolithic script mirroring `train_hybrid_model.py`'s style, since the `pipeline/` package already exists in this repository specifically to let stages be composed without the orchestrator needing to know which are pretrained vs. project-trained. This reasoning is unaffected by A.1's reversal — if anything, it is strengthened: with both segmentation stages now project-trained, the "orchestrator doesn't need to know which is pretrained" distinction from the original design no longer even arises, since neither stage is pretrained.
