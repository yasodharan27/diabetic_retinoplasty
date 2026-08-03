
## Project Overview

This repository is the baseline implementation for my diabetic retinopathy detection project.

The objective is to extend this repository into the architecture proposed in my 0th Review presentation. Do not rewrite the repository from scratch. Preserve existing functionality and improve it incrementally.

All model training will be performed in Google Colab.

---

## Existing Baseline

The current repository already implements:

- Image preprocessing
- EfficientNet-based classification
- Swin Transformer hybrid model
- Monte Carlo Dropout
- Grad-CAM
- Training and testing scripts

Reuse these components wherever appropriate instead of replacing them unnecessarily.

---

## Target Pipeline

1. Image Quality Assessment
2. Image Preprocessing
3. Vessel Segmentation
4. Lesion Segmentation
5. Local Feature Extraction
6. Global Feature Extraction
7. Feature Fusion
8. Ordinal Classification
9. Uncertainty Estimation
10. Explainability
11. Evaluation

---

## Preprocessing

The preprocessing pipeline should include:

- Image Quality Assessment (EfficientNetB0)
- CLAHE
- Gamma Correction
- Green Channel Extraction
- Ben Graham Preprocessing
- Median Denoising
- Image Resizing
- Data Augmentation

Reuse existing preprocessing wherever possible and extend it only when necessary.

---

## Models

| Module | Model |
|---------|-------|
| Image Quality Assessment | EfficientNetB0 |
| Vessel Segmentation | U-Net |
| Lesion Segmentation | Attention U-Net |
| Local Feature Extraction | Adaptive Multi-Kernel CNN |
| Global Feature Extraction | Dual-Scale Swin Transformer |
| Feature Fusion | Adaptive Cross-Attention |
| Classification | CORN Ordinal Classification |
| Uncertainty | Monte Carlo Dropout |
| Explainability | Grad-CAM++, SHAP, Attention Rollout |

---

## Datasets

Datasets

- EyeQ
  - Image Quality Assessment
  - Generated from EyePACS using the official EyeQ generation repository.

- APTOS 2019
  - Diabetic Retinopathy Classification

- EyePACS
  - Additional training and fine-tuning
  - Source dataset for EyeQ generation

- IDRiD
  - Vessel Segmentation
  - Lesion Segmentation

---

## Development Rules

- Understand the existing implementation before modifying it.
- Implement one module at a time.
- Preserve existing functionality.
- Do not rewrite the repository.
- Do not modify unrelated files.
- Keep the project modular.
- Explain the implementation plan before writing code.
- Wait for approval before moving to the next module.

---

## Training

All deep learning model training must be performed using Google Colab.

The local repository should only contain:

- source code
- notebooks
- trained weights
- evaluation scripts

Datasets must remain inside the datasets directory and should not be committed.

For every trainable model:

- Create a separate Colab notebook.
- Save the best model.
- Export trained weights.
- Integrate the trained model back into the repository.

---

## Coding Standards

- Write clean and modular code.
- Reuse existing code wherever possible.
- Avoid duplicate implementations.
- Add comments only where necessary.
- Do not generate placeholder implementations.
- Do not fabricate evaluation metrics or results.

## Development Workflow

Before implementing any module:

1. Explain the existing implementation.
2. Explain why it needs to be changed.
3. Propose the implementation plan.
4. Wait for approval.
5. Implement the module.
6. Verify integration.
7. Stop and wait for the next task.

## Dataset Handling

Do not automatically download datasets.

Always use datasets available inside:

datasets/

Before implementing any module, inspect the available dataset structure and adapt the data loader accordingly.

Do not assume fixed folder names.

---------

## Existing Repository

Before implementing any module:

- Inspect the current implementation.
- Reuse existing functionality whenever possible.
- Extend instead of replacing.
- Avoid duplicate implementations.

--------

## Training Strategy

Each trainable module must have:

- Separate Google Colab notebook
- Independent training
- Saved best weights
- Easy integration into the main repository

------ 

## Dataset Policy

Never modify the original datasets inside `datasets/*/raw`.

All preprocessing outputs must be written to the corresponding `processed` folder.

The original datasets must always remain untouched.

-----
## Deployment Requirement

Every trainable module must provide two components:

1. A training implementation used to train and export the model.
2. An inference implementation that loads the exported model and exposes a reusable prediction interface.

The final project must integrate all inference modules into a single end-to-end pipeline capable of accepting one retinal fundus image and producing the complete diabetic retinopathy analysis.

No module should exist only for training.
-----
Compare the existing preprocessing pipeline with the target preprocessing pipeline and retain only the useful components. Add the missing preprocessing steps (such as CLAHE and Gamma Correction) only if they are not already implemented. Avoid duplicate preprocessing operations