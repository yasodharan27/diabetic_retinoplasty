# Sources

Retrieved or verified during this study (2026-09-23) unless marked *(standard reference, not
re-fetched)*. Evidence weight is noted where it matters for the design.

## Project evidence (read-only, this repository)

- `docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md` — research record §1–§12 (C1, gate
  init, C3, D6, PDR/vessel diagnostics, CORN threshold, nested threshold, trade-off, score
  comparison, §12 pre-registration).
- `docs/experiments/Multiseed_Improved_Training_Report.md` — six-run verdict; §9 severe-grade;
  §12.2–12.5 gate forcing, lesion/vessel permutation, Stage 04 audit.
- `docs/experiments/RACAF_Duplicate_Contamination_Audit.md` — 41 train/val duplicates.
- `docs/experiments/IDRiD_External_Evaluation_Report.md` — protocol only; results pending.
- `SEGMENTATION_ARCHITECTURE.md` §3.5 — Stage 04 Dice on IDRiD test.
- Code: `joint_training_model.py`, `no_racaf_model.py`, `local_feature_extraction_model.py`,
  `swin_transformer.py`, `feature_fusion.py`, `corn.py`, `weighted_corn.py`,
  `joint_training_dataset.py`, `local_feature_extraction_dataset.py`,
  `improved_training_data.py`, `multiseed_runs.py`,
  `colab/notebooks/grade4_aux_loss_experiment.ipynb`.

## Clinical grounding

- Esmaeilkhanian H. et al. (2023). *The relationship of diabetic retinopathy severity scales with
  frequency and surface area of diabetic retinopathy lesions.* Graefe's Arch Clin Exp Ophthalmol.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10587246/ — **key external fact:** lesion counts/areas
  fall from severe NPDR to PDR (HE 40.96 → 6.23, MA 27.52 → 0.97, EX 48.64 → 32.47). UWF images.
- Krause J. et al. (2018). *Grader variability and the importance of reference standards for
  evaluating machine learning models for diabetic retinopathy.* Ophthalmology.
  https://ar5iv.labs.arxiv.org/html/1710.01711 — missed NV, MA-vs-HE and PRP-scar disagreements.
- INSPIRED Study Report 1. *Intergrader agreement … ultra-widefield imaging.*
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12293693/ — severe NPDR has the lowest pre-consensus agreement.
- Wilkinson C.P. et al. (2003). *Proposed international clinical diabetic retinopathy and diabetic
  macular edema disease severity scales.* Ophthalmology 110(9). *(standard reference)*
- ETDRS Research Group (1991). *Fundus photographic risk factors for progression of diabetic
  retinopathy. ETDRS Report 12.* Ophthalmology 98. *(standard reference)*

## PDR-relevant annotated datasets

- Zhou Y. et al. (2021). *A Benchmark for Studying Diabetic Retinopathy: Segmentation, Grading, and
  Transferability* (FGADR). IEEE TMI 40(3):818–828. https://arxiv.org/pdf/2008.09772 ·
  https://csyizhou.github.io/FGADR/ — IRMA and NV masks.
- Wei Q. et al. (2020). *Learn to Segment Retinal Lesions and Beyond* (Retinal-Lesions). ICPR 2020.
  https://github.com/WeiQijie/retinal-lesions — NV, fibrous proliferation, preretinal/vitreous
  haemorrhage; 1,593 images (DR3 99, DR4 62).

## Ordinal learning

- Shi X., Cao W., Raschka S. (2023). *Deep neural networks for rank-consistent ordinal regression
  based on conditional probabilities* (CORN). Pattern Analysis and Applications.
  https://arxiv.org/abs/2111.08851 · https://link.springer.com/article/10.1007/s10044-023-01181-9
- Cao W., Mirjalili V., Raschka S. (2020). *Rank consistent ordinal regression for neural networks*
  (CORAL). Pattern Recognition Letters. *(standard reference)*
- Díaz R., Marathe A. (2019). *Soft Labels for Ordinal Regression* (SORD). CVPR.
  https://openaccess.thecvf.com/content_CVPR_2019/html/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.html
- Polat G. et al. (2025). *Class Distance Weighted Cross Entropy Loss for Classification of Disease
  Severity.* Expert Systems with Applications. https://arxiv.org/abs/2412.01246
- Shaik N.S. et al. (2025). *Ordinal Label-Distribution Learning with Constrained Asymmetric Priors
  for Imbalanced Retinal Grading.* NeurIPS 2025 Workshop (GenAI for Health).
  https://arxiv.org/abs/2509.26146 — workshop-level evidence.
- Hashmi A. (2026). *Robust Diabetic Retinopathy Grading Using Dual-Resolution Attention-Based Deep
  Learning with Ordinal Regression.* https://arxiv.org/html/2604.17341v1 — CORAL on APTOS; no
  per-grade 3/4 results.

## Lesion-aware DR grading

- Wang Z. et al. (2017). *Zoom-in-Net: Deep Mining Lesions for Diabetic Retinopathy Detection.*
  MICCAI. https://arxiv.org/pdf/1706.04372
- Li X. et al. (2020). *CANet: Cross-disease Attention Network for Joint DR and DME Grading.* IEEE
  TMI. https://arxiv.org/pdf/1911.01376
- Dai L. et al. (2021). *A deep learning system for detecting diabetic retinopathy across the
  disease spectrum* (DeepDR). Nature Communications.
  https://www.nature.com/articles/s41467-021-23458-5
- Huang Y. et al. (2021). *Lesion-Based Contrastive Learning for Diabetic Retinopathy Grading from
  Fundus Images.* MICCAI. https://github.com/YijinHuang/Lesion-based-Contrastive-Learning
- Cheng S. et al. (2023). *Lesion-aware Contrastive Learning for Diabetic Retinopathy Diagnosis.*
  MICCAI. https://conferences.miccai.org/2023/papers/383-Paper1429.html
- Xia X. et al. (2024). *Lesion-aware network for diabetic retinopathy diagnosis.* Int J Imaging
  Syst Technol. https://arxiv.org/abs/2408.07264
- Tusfiqur H.M. et al. (2022). *DRG-Net: Interactive Joint Learning of Multi-lesion Segmentation and
  Classification for Diabetic Retinopathy Grading.* https://arxiv.org/abs/2212.14615
- Saleem S. et al. (2026). *Clinical-guided deep learning framework for diabetic retinopathy …*
  Scientific Reports. https://pmc.ncbi.nlm.nih.gov/articles/PMC13443102/ — **low credibility for
  3v4:** reports 5,590 "APTOS 2019" images (3,662 labelled exist) on a random 80/20 split.
- Wu Z. et al. (2020). *Coarse-to-fine classification for diabetic retinopathy grading using
  convolutional neural network* (CF-DRNet). Artificial Intelligence in Medicine.
  https://www.sciencedirect.com/science/article/abs/pii/S0933365720301354 (abstract page returned
  403; citation from search index)

## Aggregation / cardinality / pooling

- Zhang S., Xie L. (2020). *Improving Attention Mechanism in Graph Neural Networks via Cardinality
  Preservation.* IJCAI. https://www.ijcai.org/proceedings/2020/194 — softmax attention discards
  cardinality.
- Xu K. et al. (2019). *How Powerful are Graph Neural Networks?* ICLR — mean/max aggregators cannot
  distinguish multisets. *(standard reference)*
- Zaheer M. et al. (2017). *Deep Sets.* NeurIPS. *(standard reference)*
- Ilse M., Tomczak J., Welling M. (2018). *Attention-based Deep Multiple Instance Learning.* ICML.
  https://proceedings.mlr.press/v80/ilse18a.html
- Lee J. et al. (2019). *Set Transformer.* ICML. http://proceedings.mlr.press/v97/lee19d/lee19d.pdf

## Imbalance / objectives (context for candidate B)

- Cui Y. et al. (2019). *Class-Balanced Loss Based on Effective Number of Samples.* CVPR. *(standard reference)*
- Lin T.-Y. et al. (2017). *Focal Loss for Dense Object Detection.* ICCV. *(standard reference)*
- Cao K. et al. (2019). *Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss.*
  NeurIPS. *(standard reference)*
- Menon A.K. et al. (2021). *Long-tail learning via logit adjustment.* ICLR. *(standard reference)*
- Kang B. et al. (2020). *Decoupling Representation and Classifier for Long-Tailed Recognition.*
  ICLR. *(standard reference)*

## Normalisation / backbones (context)

- Wu Y., He K. (2018). *Group Normalization.* ECCV. https://arxiv.org/abs/1803.08494 — at batch 2,
  GN 23.0 % vs BN 31.9 % error (ResNet-101, ImageNet).
- Zhou Y. et al. (2023). *A foundation model for generalizable disease detection from retinal
  images* (RETFound). Nature. https://www.nature.com/articles/s41586-023-06555-x — APTOS AUROC 0.943.

## Searched, not used as evidence

- Sälzer M. et al. (2025/ICLR 2026). *The Counting Power of Transformers.*
  https://arxiv.org/abs/2505.11199 — concerns expressivity of counting properties in sequence
  transformers, not the aggregation argument used here.
- APTOS label-noise / laser-scar prevalence: no source found that quantifies PRP scars in APTOS
  grade 4. H5 therefore stays a hypothesis, to be checked by the P0.5 visual audit.
