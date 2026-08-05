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
  - [Pipeline Diagram](#pipeline-diagram)
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

The target architecture is an 11-stage, end-to-end diabetic retinopathy pipeline -- full detail
in `PROJECT_CODE.md` (rules and target design) and `PROJECT_STRUCTURE.md` (master architectural
reference: every folder's purpose, every stage's input/output/dataset/status, output locations,
and the project rules). This section is a summary; those two files are authoritative.

### Pipeline Diagram

```
 [1] Image Quality Assessment  --(Good/Usable only)-->  [2] Image Preprocessing
                                                                  |
                                                                  v
                                          [3] Vessel Segmentation (pretrained, inference-only)
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
                                                    |
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

### Dataset Summary

| Dataset | Used by | Status |
|---|---|---|
| **EyeQ** | Stage 1 (Image Quality Assessment) | Local copy present, verified (`colab/common/verify_dataset.py`) |
| **APTOS 2019** | Stage 8 (CORN Classification); also the pre-refactor baseline below | Local copy present |
| **IDRiD** | Stage 4 (Lesion Segmentation) | Local copy present, not yet consumed by any implemented stage |
| **EyePACS** | Historical only | Used once, outside this repository, to reconstruct EyeQ. Not present under `datasets/`; not required to run anything here. |

See `PROJECT_STRUCTURE.md`'s Dataset Organization for current vs. future usage per dataset.

### Folder Structure

```
diabetic_retinoplasty/
├── *.py                    # flat top-level pipeline modules (no src/ layout)
├── training/                # reusable training framework (Trainer, callbacks, losses, metrics)
├── evaluation/               # reusable evaluation framework (Evaluator, metrics, visualization)
├── pipeline/                  # ABC contracts for future trainable/inference stages
├── datasets/                   # EyeQ/, APTOS2019/, IDRiD/ -- raw/ is read-only, never modified
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
| 1. Image Quality Assessment | **Implemented**, not yet trained for real -- see `docs/FIRST_TRAINING_CHECKLIST.md` |
| 2. Image Preprocessing | Transform logic exists (`image_preprocessing.py`); not yet wired into a Colab notebook |
| 3-11 | Not implemented (design for 3-4 exists in `SEGMENTATION_ARCHITECTURE.md`; template notebooks exist under `colab/notebooks/`) |

### Future Roadmap

Implement Stages 2-11 one at a time, per `PROJECT_CODE.md`'s "one module at a time, wait for
approval" rule -- see `IMPLEMENTATION_PLAN.md` for the detailed gap analysis driving this order.

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
