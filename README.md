# Diabetic Retinopathy Detection

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.9%2B-FF6F00?style=flat-square&logo=tensorflow)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-AI-9cf?style=flat-square&logo=pytorch)](https://github.com/yourusername/diabetic-retinopathy-detection)

A hybrid deep learning framework for automated diabetic retinopathy detection with uncertainty quantification and explainability.

![Diabetic Retinopathy System](assets/research_summary_dashboard.png)

> **Repository status:** this repository contains two things side by side -- the **original
> baseline** (documented in the sections below: APTOS-only, Ben Graham preprocessing, the
> EfficientNet+Swin hybrid model, GAN augmentation, MC Dropout, Grad-CAM) and the **actively
> developed target architecture**, an 11-stage pipeline being built out incrementally per
> `PROJECT_CODE.md`. See [Repository Status & Target Architecture](#-repository-status--target-architecture)
> below for what's real today versus planned, before relying on anything described further down.

## 📋 Table of Contents

- [Repository Status & Target Architecture](#-repository-status--target-architecture)
  - [Master Pipeline](#master-pipeline)
  - [Dataset Summary](#dataset-summary)
  - [Folder Structure](#folder-structure)
  - [Development Workflow](#development-workflow)
  - [Training Workflow](#training-workflow)
  - [Experiment Workflow](#experiment-workflow)
  - [Current Implementation Status](#current-implementation-status)
  - [Future Roadmap](#future-roadmap)
- [Baseline Overview](#-overview)
- [Baseline Features](#-features)
- [Baseline Model Architecture](#-model-architecture)
- [Baseline Dataset](#-dataset)
- [Installation](#-installation)
- [Baseline Usage](#-usage)
- [Baseline Results](#-results)
- [Future Work (baseline)](#-future-work)
- [References](#-references)
- [Contributing](#-contributing)

---

## 🏛️ Repository Status & Target Architecture

> **Status (updated 2026-10-08).** Stages 1–4 are built and frozen (Stage 4 as the v2 SE-ResNet-101 U-Net).
> The original Stage 5–8 design below (multi-kernel CNN, Swin, cross-attention, RACAF) was implemented and
> trained; RACAF closed without a supported benefit. The reference grader is now **P** (ImageNet ConvNeXt-Tiny +
> CORN, RGB; APTOS validation QWK 0.917) and the current research model is **E1** (P + an auxiliary lesion head
> trained on Stage-4 targets; QWK 0.918, comparable to P, with a demonstrated change in the representation).
> The authoritative summary of results and the approved plan is `PROJECT_CODE.md`, section "Current Project State
> and Plan"; the full evidence is `docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md`.

The target architecture is an 11-stage, end-to-end diabetic retinopathy pipeline -- full detail
in `PROJECT_CODE.md` (rules and target design), `PROJECT_STRUCTURE.md` (master architectural
reference: every folder's purpose, every stage's input/output/dataset/status, output locations,
and the project rules), and `SEGMENTATION_ARCHITECTURE.md` (Vessel/Lesion Segmentation design,
including tensor contracts). This section is a summary; those documents are authoritative.

### Master Pipeline

This is the repository's single, canonical end-to-end architecture diagram — every other document
references this one rather than repeating a diverging copy. It shows the **original design**; the pipeline as
built is given directly after it.

```
 [1] Image Quality Assessment  --(Good/Usable only)-->  [2] Image Preprocessing
                                                                  |
                                                                  v
                                          [3] Vessel Segmentation (pretrained LWNet, inference only)
                                                                  |
                                                                  v
                                          [4] Lesion Segmentation (trained on IDRiD)
                                                                  |
                        +-----------------------------------------+
                        v                                         v
        [5] Local Feature Extraction              [6] Global Feature Extraction
           (Adaptive Multi-Kernel CNN)               (Dual-Scale Swin Transformer)
                        |                                         |
                        +-----------------> [7] Feature Fusion <--+
                                          (Adaptive Cross-Attention)
                                                    |
                                                    v
                                    [8] CORN Ordinal Classification
                                        (trained on APTOS 2019)
                        +---------------------------+---------------------------+
                        v                                                       v
        [9] Uncertainty Estimation                                [10] Explainability
           (Monte Carlo Dropout)                          (Grad-CAM++, SHAP, Attention Rollout)
                        |                                                       |
                        +---------------------------+---------------------------+
                                                    v
                                          [11] Evaluation
                                (end-to-end, held-out test set, real metrics only)
```

Every stage depends only on the stage(s) immediately before it — no stage bypasses another. See `PROJECT_STRUCTURE.md`'s "Stage Dependencies" section for the explicit dependency chain.

**Pipeline as built (2026-10-08):**

```
 [1] IQA (EfficientNetB0, frozen; not in the grading graph)
 [2] Preprocessing: gamma + CLAHE, once, native size  -->  full-frame resize to 512 x 512 RGB
        |                                   |
        |                                   +--> [3] LWNet vessel map (frozen)      } teachers / inputs of the
        |                                   +--> [4] Stage-4 v2 lesion maps (frozen) } segmentation-based models
        v
 Graders (CORN ordinal decoding):
   P   ConvNeXt-Tiny on RGB                                              reference baseline
   E1  P + 1x1 lesion head on the final feature map, Stage-4 targets     current research model
   E2  E1 with shuffled lesion targets                                   control
   (earlier lines, completed: original Stage 5-8 +/- RACAF, PL, Architecture 1, pathology grader, fixed fusions)
 [9]-[11] uncertainty, explainability, final evaluation: not started for the current models
```

### Dataset Summary

| Dataset | Used by | Status |
|---|---|---|
| **EyeQ** | Stage 1 (Image Quality Assessment) — used only for Stage 1 | Local copy present, verified (`colab/common/verify_dataset.py`) |
| **APTOS 2019** | All grading models (fixed split 2,921 / 730); also the pre-refactor baseline below | Local copy present |
| **IDRiD** | Segmentation set: Stage 4 training and its one-time test gate. Grading test set: evaluation only | Local copy present. Grading test used once (frozen pipeline); two further pre-declared batches allowed |
| **TJDR** | Stage 4 v2 training only | Local copy present, verified |
| **EyePACS** | Reconstruction of EyeQ; approved 2026-10-08 as a controlled factor for grading (supervised adaptation, held-out labelled test subset) | Local copy present under `datasets/EyePACS`; not yet preprocessed or used for grading |
| **DDR** | Planned ground-truth for the lesion probe (757 expert-annotated images) | Approved, not yet downloaded |

Stage 3 (Vessel Segmentation) needs no project dataset: it integrates a pretrained, externally-sourced model (LWNet) for inference only, not trained within this project. DRIVE and CHASE_DB1 were approved under an earlier, since-superseded design that trained Vessel Segmentation within this project; neither is a project dataset today. See `PROJECT_STRUCTURE.md`'s Dataset Organization for current usage per dataset, and `SEGMENTATION_ARCHITECTURE.md`'s design-history appendix for why this reversal happened.

### Folder Structure

```
diabetic_retinoplasty/
├── *.py                    # flat top-level pipeline modules (no src/ layout)
├── training/                # reusable training framework (Trainer, callbacks, losses, metrics)
├── evaluation/               # reusable evaluation framework (Evaluator, metrics, visualization)
├── pipeline/                  # ABC contracts for future trainable/inference stages
├── datasets/                   # EyeQ/, APTOS2019/, IDRiD/ -- raw/ is read-only, never
│                                 modified. Vessel Segmentation uses a vendored pretrained
│                                 checkpoint, not a dataset here.
├── colab/                        # official Colab training infrastructure
│   ├── common/                    # setup, verification, experiment management (shared by every stage)
│   └── notebooks/                  # stage01_iqa.ipynb (implemented) + stage02-11 (templates)
├── tests/                          # pytest unit tests, synthetic/temporary data only
├── docs/                            # operational runbooks (e.g. FIRST_TRAINING_CHECKLIST.md)
├── research_papers/                  # background reading
├── PROJECT_CODE.md                    # target architecture + development rules (canonical)
├── PROJECT_STRUCTURE.md                # master architectural reference (this summary's source)
├── IMPLEMENTATION_PLAN.md               # baseline-vs-target gap analysis
└── SEGMENTATION_ARCHITECTURE.md          # Vessel/Lesion Segmentation design
```

### Development Workflow

Per `PROJECT_CODE.md`: understand the existing implementation, explain why it needs to change,
propose a plan, wait for approval, implement one module at a time, verify integration, stop.
Local development happens in VS Code (+ Claude Code for structured implementation work); all real
model training happens in Google Colab. See `PROJECT_STRUCTURE.md`'s Local Development Workflow
section for how these tools interact.

### Training Workflow

1. Implement/verify a stage's dataset loader, model, and training entry point locally (small
   real-data smoke tests only -- no local GPU, no fabricated results).
2. Open the matching `colab/notebooks/stageNN_*.ipynb` in Google Colab.
3. Run it top to bottom: Setup -> Verification -> Dataset Loading -> Model Creation -> Training ->
   Evaluation -> Export (see `colab/README.md`).
4. Review results locally; commit the exported model deliberately (never automatically).

### Experiment Workflow

Every Colab training run gets its own isolated, timestamped folder on Google Drive under
`experiments/<Module>/<timestamp>/` (`checkpoints/`, `logs/`, `tensorboard/`, `evaluation/`,
`predictions/`, `metadata.json`) -- never overwritten, resumable by pointing a later run at the
same folder. See `colab/README.md`'s "How experiments are organized".

### Current Implementation Status

| Stage | Status |
|---|---|
| 1. Image Quality Assessment | **Completed -- Verified -- Baseline Established.** Trained end-to-end in Google Colab; see Stage 1 Baseline Results below. |
| 2. Image Preprocessing | **Complete, frozen.** RGB → Gamma Correction → CLAHE only (`image_preprocessing.py`); applied once per dataset. |
| 3. Vessel Segmentation | **Complete, frozen.** Pretrained LWNet, inference only. |
| 4. Lesion Segmentation | **Complete, frozen (v2).** SE-ResNet-101 U-Net for MA / HE / EX / SE, trained on IDRiD + TJDR; one-time IDRiD gate passed. The earlier Attention U-Net is superseded. |
| 5–8. Grading | **Original design implemented, trained and closed** (RACAF not supported). **Current graders:** P (QWK 0.917), E1 (0.918), E2 control (0.914) on APTOS validation; frozen dual-branch pipeline evaluated once on IDRiD (QWK 0.629; P 0.642). |
| 9–11. Uncertainty, explainability, evaluation | Not started for the current models. |

### Stage 1 Baseline Results

Held-out EyeQ test split (16,249 images, never used in training or validation), from the
completed and verified Stage 1 run (`colab/notebooks/stage01_iqa.ipynb`, experiment
`2026-08-05_09-11-28`):

| Metric | Value |
|---|---|
| Accuracy | 88.05% |
| F1 Score | 86.12% |
| AUC | 96.48% |
| Quadratic Weighted Kappa (QWK) | 0.8987 |

EfficientNetB0 (ImageNet-pretrained) with mixed precision (`mixed_float16`); best validation
checkpoint occurred at Epoch 2, `ReduceLROnPlateau` reduced the learning rate twice, and
`EarlyStopping` (patience 8) stopped training at Epoch 10 and restored the Epoch 2 weights -- the
exported model is that Epoch 2 checkpoint, not the final epoch's. Full detail, including exact
output locations on Google Drive, is recorded in `colab/notebooks/stage01_iqa.ipynb`'s Section 7
and `docs/FIRST_TRAINING_CHECKLIST.md`'s completed-run record. No metrics beyond the four above
are claimed as part of this baseline.

### Future Roadmap

The approved next phase (2026-10-08), in order: IDRiD batch 1 (E1 and E2 on the locked protocol); DDR
download and a ground-truth lesion probe on P / E1 / E2; one supervised EyePACS adaptation run; P-EP and E1-EP on
APTOS; a conditional shuffled control; one-shot evaluations of the adapted models. Details and rules are in
`PROJECT_CODE.md` ("Current Project State and Plan") and research record §65. `IMPLEMENTATION_PLAN.md` describes
the original build order and is kept as history.

---

## 🔍 Overview

Diabetic Retinopathy (DR) is a diabetes complication that affects the eyes and can lead to blindness if left untreated. Early detection is crucial for effective treatment, but manual screening by ophthalmologists is time-consuming and subject to variability.

This project implements a hybrid deep learning approach that:

1. **Accurately classifies** retinal images into 5 severity levels of DR
2. **Quantifies uncertainty** in predictions using Bayesian methods
3. **Explains decisions** through gradient-based visualization techniques
4. **Addresses class imbalance** through focal loss and weighting strategies

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📊 **Preprocessing Pipeline** | Ben Graham's technique with green channel extraction, CLAHE, and denoising |
| 🧠 **Hybrid Architecture** | EfficientNetB0 + Swin Transformer for improved feature representation |
| 🔍 **Bayesian Uncertainty** | Monte Carlo Dropout for confidence estimation and uncertainty quantification |
| 👁️ **Explainable AI** | Grad-CAM visualizations showing which retinal regions influence decisions |
| ⚖️ **Class Imbalance Handling** | Focal Loss and class weighting techniques to handle unbalanced datasets |

## 🏗️ Model Architecture

![Architecture](assets/hybrid_model_architecture.png)

Our hybrid architecture combines:
- **EfficientNetB0**: Pre-trained CNN for efficient feature extraction
- **Swin Transformer**: Attention-based refinement of features with hierarchical window partitioning
- **Monte Carlo Dropout**: Bayesian approximation for uncertainty estimation
- **Grad-CAM**: Class activation mapping for model explainability

## 📊 Dataset

The model is trained and evaluated on the [APTOS 2019 Diabetic Retinopathy Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) dataset, which contains retinal fundus photographs labeled with DR severity levels:

| Class | Severity Level | Description | Visual Signs |
|-------|---------------|-------------|--------------|
| 0 | No DR | No signs of diabetic retinopathy | Normal retina |
| 1 | Mild | Microaneurysms only | Small red dots |
| 2 | Moderate | More than microaneurysms but less than severe | Red lesions, hard exudates |
| 3 | Severe | Extensive hemorrhages and venous beading | Cotton wool spots, venous beading |
| 4 | Proliferative | Abnormal blood vessel growth and potential retinal detachment | Neovascularization, preretinal hemorrhage |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/romilagarwal/diabetic_retinoplasty.git
cd diabetic_retinopathy

# Create and activate virtual environment
python -m venv env
source env/bin/activate  
# On Windows use: env\Scripts\activate

# Install dependencies from requirements
pip install -r [requirements.txt]
```
Additionally, Graphviz must be installed on your system for model visualization.


## 🚀 Usage

1. Data Preprocessing
```bash
python pre_process_with_dataset_download.py
```
<details> <summary><b>Preprocessing Details</b></summary>
This script:

Downloads the dataset (if not already present)
Applies Ben Graham's preprocessing technique with green channel extraction
Enhances images with CLAHE (Contrast Limited Adaptive Histogram Equalization)
Applies denoising filters
Resizes images to 224×224
Organizes processed images into class folders
</details>

2. Base Model Training
```bash
python efficientnet_model.py
```
Trains a baseline EfficientNetB0 model with transfer learning from ImageNet weights.

3. Hybrid Model Training
```bash
python train_hybrid_model.py
```
<details> <summary><b>Training Parameters</b></summary>
The hybrid model training uses:

 · Focal loss for class imbalance
 · Mixed precision for memory efficiency
 · Class weighting for balanced learning
 · Learning rate scheduling
 · Early stopping to prevent overfitting
</details>

4. Model Evaluation
```bash
# Test the base model
python testing_efficientnet_model.py

# Test the hybrid model
python test_hybrid_model.py
```

5.Bayesian Uncertainty Estimation
```bash
python bayesian_inference.py
```
<details> <summary><b>Uncertainty Metrics</b></summary>
The Bayesian component performs:

 · Monte Carlo Dropout inference with multiple forward passes
 · Confidence score calculation
 · Uncertainty estimation (standard deviation of predictions)
 · Predictive entropy calculation
 · Reliability diagram generation
</details>

6.Explainable AI Visualizations
```bash
python explainable_ai.py
```
Generates Generates Grad-CAM visualizations highlighting regions that influence the model's decisions.

7. Generate Visualizations for Publication
```bash
python generate_all_visualizations.py
```
Creates comprehensive visualizations for research papers or presentations.

> **Note:** Steps 1–7 above describe the **pre-refactor baseline** scripts, unchanged and still present at the repository root. They are unrelated to the target 11-stage pipeline's Stage 02 (`image_preprocessing.py`, RGB → Gamma → CLAHE only, no green-channel extraction, Ben Graham, or denoising) -- see [Repository Status & Target Architecture](#-repository-status--target-architecture) above for which one actually reflects the current, frozen architecture.

## 📈 Results

> **Note:** The table below matches the placeholder/sample values hardcoded as fallback data in `Visualization_Scripts/create_publication_tables.py` (used when no real `model_comparison.csv` is present) rather than a verified `classification_report` run on held-out test data. Treat these numbers as illustrative until they are regenerated from an actual evaluation run.

Performance Metrics
|Model        | No DR  | Mild	| Moderate | Severe | Proliferative | Average |
|-------------|--------|--------|----------|--------|---------------|---------|
|EfficientNet | 0.76   | 0.70	|  0.72	   |  0.65	|     0.63	    |   0.69  |
|Hybrid Model |	0.82   | 0.75	|  0.79	   |  0.73	|     0.71	    |   0.76  |

Key Improvements
· +7% Average F1 Score improvement over baseline EfficientNet
· Better Generalization across all DR severity classes
· Enhanced Performance on minority classes (Severe and Proliferative)
· Reduced Uncertainty in predictions compared to baseline

## 👁️ Visualizations

<h3>Grad-CAM Explainability</h3>
<img alt="Grad-CAM" src="assets\explainability_summary.png">

<h3>Uncertainty Analysis</h3>

<img alt="Uncertainty" src="assets\uncertainty_processed.png">


## 🔮 Future Work

1. DR-GAN++: Implementation of Generative Adversarial Networks for synthetic data generation to further address class imbalance
2. Ensemble Methods: Combining multiple models for improved performance
3. Clinical Integration: Development of a user-friendly interface for clinical use
4. Mobile Deployment: Optimization for edge devices to enable screening in remote areas
Multimodal Learning: Integrating patient metadata with retinal images


## 📚 References

1. Huang, G., Liu, Z., Van Der Maaten, L., & Weinberger, K. Q. (2017). Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition, 4700-4708.

2. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. Proceedings of the IEEE/CVF International Conference on Computer Vision, 10012-10022.

3. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian approximation: Representing model uncertainty in deep learning. International conference on machine learning, 1050-1059.

4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE international conference on computer vision, 618-626.

5. Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. Proceedings of the IEEE international conference on computer vision, 2980-2988.


## 👥 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
