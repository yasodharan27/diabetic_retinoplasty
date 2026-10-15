# Diabetic Retinopathy Grading from Fundus Photographs

A staged pipeline that grades diabetic retinopathy (ICDR grades 0–4) from a single colour fundus photograph,
together with an image-quality check, vessel and lesion maps, a seed-ensemble uncertainty estimate and
explanation maps. The grader is a ConvNeXt-Tiny with a CORN ordinal head, trained jointly on APTOS 2019 and
EyePACS and tested externally on DDR and IDRiD.

This repository is the full research workspace: every model, experiment, protocol and result of the project,
including the lines of work that were tested and not adopted. The complete chronological evidence is the
research record, [`docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md`](docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md).

> Research prototype. It is not a medical device and its output is not a diagnosis.

## Contents
- [Pipeline](#pipeline)
- [Results](#results)
- [Datasets](#datasets)
- [Repository layout](#repository-layout)
- [Running it](#running-it)
- [What was tested and not adopted](#what-was-tested-and-not-adopted)
- [Limitations](#limitations)
- [References](#references)

## Pipeline

```
 fundus photograph
   |
 [1] Image quality      EfficientNet-B0 trained on EyeQ (Good / Usable / Reject)
 [2] Preprocessing      gamma 1.2, then CLAHE (clip 2.0, 8 x 8 tiles) on the LAB lightness channel;
                        fundus bounding-box crop; resize to 512 x 512
   |------------------------------.------------------------------.
 [3] Vessel map                [4] Lesion maps                    |
     pretrained LWNet              SE-ResNet-101 U-Net:            |
     (inference only)              microaneurysms, haemorrhages,  |
                                   hard and soft exudates         |
     (shown as supporting outputs; the grade does not use them)   |
                                                                  v
 [5] Grader             ConvNeXt-Tiny (ImageNet) + CORN ordinal head, three seeds (42, 123, 2026)
 [6] Uncertainty        three-seed ensemble entropy and disagreement; test-time augmentation (8 flips/rotations)
 [7] Explanations       Grad-CAM, Integrated Gradients, Occlusion; superpixel Kernel SHAP for figures
```

**Grader training.** Joint APTOS 2019 + EyePACS training with two EyePACS images per APTOS image in every
epoch (2,921 + 5,842), batch 16, AdamW (learning rate 1e-4, weight decay 0.05), class-weighted CORN loss,
early stopping, and checkpoint selection on the mean of the APTOS-validation QWK (730 images) and the
EyePACS-validation QWK (3,512 images). Two input variants were trained with three seeds each: the full frame
resized to 512 (**J**) and the fundus bounding box cropped first (**J-crop**). The decoded grade is the number
of CORN cumulative probabilities above 0.5.

**Final model.** J-crop, chosen after the external re-test below (see [Limitations](#limitations)).

## Results

### Development (validation sets, mean of three seeds)

| Model | APTOS validation QWK | EyePACS validation QWK |
|---|---|---|
| J (full frame) | 0.913 | 0.817 |
| **J-crop** | **0.916** | **0.827** |

### External test

Quadratic weighted kappa, mean of three seeds; differences with 95 % intervals from a paired, grade-stratified
bootstrap (2,000 resamples). P is the earlier APTOS-only ConvNeXt-Tiny + CORN grader; P-EP is P adapted to EyePACS.

| Model | DDR (3,540 images) | IDRiD (100 images) |
|---|---|---|
| P | 0.509 | 0.642 |
| P-EP | 0.593 | 0.738 |
| J | 0.800 | 0.703 |
| **J-crop** | **0.811** | **0.788** |
| J − P | +0.290 (+0.270 … +0.311) | +0.062 (−0.004 … +0.129) |
| J-crop − P | +0.301 (+0.281 … +0.323) | +0.146 (+0.078 … +0.223) |
| J-crop − J | +0.011 (+0.003 … +0.019) | +0.084 (+0.037 … +0.141) |

On DDR, referable-DR sensitivity at the fixed decoding rises from 0.31 (P) to 0.70–0.71 at specificity
0.985–0.988, and the calibration error falls from 0.35 to 0.06–0.07.

### Uncertainty

The three-seed ensemble's entropy separates wrong from correct grades with AUROC 0.77–0.87 on the APTOS,
EyePACS and DDR sets (0.69 on IDRiD), and referring the 20 % least certain images raises the QWK of the rest
above what random referral gives on every set.

### Explanations

Agreement of each explanation map with the expert lesion masks of 60 DDR lesion-segmentation training images
(none in the DDR grading test list); lesions cover a median of 0.2–0.3 % of the pixels.

| J-crop | Pointing game | Top-10 % lesion precision | AUROC |
|---|---|---|---|
| Grad-CAM | 0.07 | 0.024 | 0.82 |
| Integrated Gradients | 0.40 | 0.039 | 0.94 |
| Occlusion (20 images) | 0.05 | 0.015 | 0.82 |
| uniform map | 0.008 | 0.008 | 0.50 |

### Stage 1 (image quality)

Held-out EyeQ test split, 16,249 images: accuracy 88.05 %, F1 86.12 %, AUC 96.48 %, QWK 0.8987.

## Datasets

| Dataset | Used for |
|---|---|
| [EyeQ](https://github.com/HzFu/EyeQ) | Stage 1 training and test |
| [APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection) | grader training and validation (fixed split 2,921 / 730) |
| [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) | grader training and validation; a labelled held-out subset is kept unused |
| [IDRiD](https://idrid.grand-challenge.org/) | Stage 4 training (lesion-segmentation training images, 44 / 10 split); external grading test |
| [TJDR](https://github.com/NekoPii/TJDR) | Stage 4 training |
| [DDR](https://github.com/nkicsl/DDR-dataset) | external grading test; lesion masks for the explanation check |

Images are not included in this repository; the fixed splits are in [`dataset_splits/`](dataset_splits/).

## Repository layout

```
*.py                         pipeline modules, flat at the top level
  image_quality_*.py           Stage 1
  image_preprocessing.py       Stage 2
  geometry_diagnostic.py,
  crop_frames.py               fundus bounding-box crop
  vessel_segmentation_*.py     Stage 3 (LWNet)
  stage4_v2*.py                Stage 4 lesion model and training
  pl_convnext.py, corn.py,
  weighted_corn.py             grader and ordinal loss
  joint_train.py               joint APTOS + EyePACS training (J and J-crop)
  final_external_eval.py       DDR / IDRiD external re-test
  uncertainty_eval.py          ensemble uncertainty, selective prediction, test-time augmentation
  explain.py, gradcam.py       explanation maps and the localisation check
  demo_app.py, demo_pipeline.py  local web demo and one-image summary
*_protocol*.json             frozen evaluation protocols (paths and checksums of every model used)
colab/                       Colab setup (common/) and the notebooks that ran every experiment
dataset_splits/              fixed train / validation / test splits
docs/experiments/            the research record and per-experiment reports
research/                    architecture audits and analysis scripts
tests/                       CPU unit tests
```

## Running it

All training and the external evaluations ran on Google Colab (T4 GPU) from the notebooks in
[`colab/notebooks/`](colab/notebooks/), reading datasets and writing outputs to Google Drive:

| Step | Notebook |
|---|---|
| Stage 1 | `stage01_iqa.ipynb` |
| Stage 3 | `stage03_vessel_segmentation.ipynb` |
| Stage 4 | `stage4_v2_training_cache.ipynb`, `stage4_v2_training.ipynb` |
| Grader (J, J-crop) | `joint_aptos_eyepacs.ipynb`, `joint_aptos_eyepacs_crop.ipynb` |
| External re-test and uncertainty | `final_external_eval.ipynb` |
| Explanations | `explanations.ipynb` |

Local web demo (CPU), with the trained models placed under `trained_models/demo/`:

```bash
pip install -r requirements.txt
python demo_app.py          # then open http://localhost:8501
```

Upload a fundus photograph or pick a demo image; each stage is shown as it finishes. Tests: `python -m pytest tests`.

## What was tested and not adopted

Each of these was implemented, trained and evaluated under a fixed protocol, and is documented with its
results in the research record:

- the original Stage 5–8 design (adaptive multi-kernel CNN, dual-scale Swin transformer, cross-attention
  fusion) with the reliability-aware fusion gate RACAF, which showed no supported benefit;
- feeding or supervising the grader with the Stage 3 / 4 maps (pathology-grader fusions, an auxiliary lesion
  head), which gave no grading gain over the RGB grader;
- selective-prediction and operating-point rules on the earlier graders, and a generic vessel route for
  proliferative DR;
- Monte Carlo dropout, which does not apply to this grader (it has no dropout layers); the seed ensemble is
  used instead.

## Limitations

- **J-crop was chosen after the external re-test.** Neither variant was declared final in advance, so J-crop's
  DDR and IDRiD results are those of the variant that looked better on them and are optimistic; they are not an
  unbiased estimate for a model selected beforehand.
- DDR and IDRiD had been used in earlier evaluations of this project before this re-test.
- DDR has no patient identifiers, and its image source is confounded with grade; IDRiD's test set is small
  (100 images), so its intervals are wide.
- Overlap of an explanation map with lesions shows where the grade score is sensitive; it does not show that
  the grader detects lesions.
- Nothing here is a clinical-utility claim.

## References

- Liu, Z. et al. A ConvNet for the 2020s. CVPR 2022 (ConvNeXt).
- Shi, X., Cao, W., Raschka, S. Deep neural networks for rank-consistent ordinal regression based on conditional
  probabilities. Pattern Analysis and Applications 26, 2023 (CORN).
- Tan, M., Le, Q. EfficientNet: Rethinking model scaling for convolutional neural networks. ICML 2019.
- Galdran, A. et al. State-of-the-art retinal vessel segmentation with minimalistic models. Scientific Reports 12, 2022 (LWNet).
- Hu, J., Shen, L., Sun, G. Squeeze-and-excitation networks. CVPR 2018 (SE-ResNet).
- Ronneberger, O., Fischer, P., Brox, T. U-Net. MICCAI 2015.
- Lakshminarayanan, B., Pritzel, A., Blundell, C. Simple and scalable predictive uncertainty estimation using deep
  ensembles. NeurIPS 2017.
- Selvaraju, R. R. et al. Grad-CAM. ICCV 2017.
- Sundararajan, M., Taly, A., Yan, Q. Axiomatic attribution for deep networks. ICML 2017 (Integrated Gradients).
- Zeiler, M. D., Fergus, R. Visualizing and understanding convolutional networks. ECCV 2014 (occlusion).
- Lundberg, S. M., Lee, S.-I. A unified approach to interpreting model predictions. NeurIPS 2017 (SHAP).
- Fu, H. et al. Evaluation of retinal image quality assessment networks in different color-spaces. MICCAI 2019 (EyeQ).
- Porwal, P. et al. IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge. Medical Image Analysis 59, 2020.
- Li, T. et al. Diagnostic assessment of deep learning algorithms for diabetic retinopathy screening.
  Information Sciences 501, 2019 (DDR).
