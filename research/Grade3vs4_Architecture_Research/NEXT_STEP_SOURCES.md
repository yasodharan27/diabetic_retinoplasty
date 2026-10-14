# Sources — next-step study (2026-09-26)

"(verified)" = the source page, record or abstract was opened or retrieved during this study.
"(standard reference)" = a well-known reference cited from its standard bibliographic record, not
re-fetched. Earlier sources for this folder are in SOURCES.md.

## Project evidence (read-only)

- `docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md` — research record §1–§14.1
  (including the 2026-09-25 caveat on §14.1).
- `docs/experiments/Multiseed_Improved_Training_Report.md` — six-run verdict, §9 severe grades,
  §12.2–12.5.
- `docs/experiments/RACAF_Duplicate_Contamination_Audit.md` — the 41 byte-identical train/val pairs
  (size prefilter → MD5 → full byte compare); re-tabulated here for grade agreement.
- `docs/experiments/IDRiD_External_Evaluation_Report.md` — pre-registered, not run; IDRiD test split
  34/5/32/19/13; Stage-04 overlap policy.
- `research/Grade3vs4_Architecture_Research/` — earlier design documents and the Phase-0 script.
- Code: `corn.py`, `weighted_corn.py`, `task_weighted_corn.py`, `multiseed_runs.py`,
  `colab/notebooks/grade4_aux_loss_experiment.ipynb`,
  `colab/notebooks/task3_weighting_placebo_experiment.ipynb`.
- Drive artifacts cited: `experiments/Grade3vs4_Phase0/2026-09-23_18-09-08`,
  `experiments/Grade4AuxLoss/grade4_aux_loss_2026_09`.

## Ordinal regression — methods and theory

- Shi X., Cao W., Raschka S. (2023). *Deep neural networks for rank-consistent ordinal regression
  based on conditional probabilities* (CORN). Pattern Analysis and Applications.
  https://arxiv.org/abs/2111.08851 · https://link.springer.com/article/10.1007/s10044-023-01181-9 (verified)
- Cao W., Mirjalili V., Raschka S. (2020). *Rank consistent ordinal regression for neural networks
  with application to age estimation* (CORAL). Pattern Recognition Letters 140. (standard reference)
- Tutz G. (2022). *Ordinal regression: A review and a taxonomy of models.* WIREs Computational
  Statistics. https://wires.onlinelibrary.wiley.com/doi/full/10.1002/wics.1545 (verified)
- Tutz G. (1991). *Sequential models in categorical regression.* Computational Statistics & Data
  Analysis 11. (standard reference; cited as the origin of the sequential / continuation-ratio model)
- McCullagh P. (1980). *Regression models for ordinal data.* JRSS-B 42(2). (standard reference)
- Gutiérrez P.A., Pérez-Ortiz M., Sánchez-Monedero J., Fernández-Navarro F., Hervás-Martínez C. (2016).
  *Ordinal regression methods: survey and experimental study.* IEEE TKDE 28(1):127–146.
  doi:10.1109/TKDE.2015.2457911 (verified)
- Vargas V.M., Gutiérrez P.A., Hervás-Martínez C. (2020). *Cumulative link models for deep ordinal
  classification.* Neurocomputing 401:48–58. https://arxiv.org/abs/1905.13392 (verified)
- Frank E., Hall M. (2001). *A simple approach to ordinal classification.* ECML, LNCS 2167.
  https://link.springer.com/chapter/10.1007/3-540-44795-4_13 (verified)
- Niu Z., Zhou M., Wang L., Gao X., Hua G. (2016). *Ordinal regression with multiple output CNN for
  age estimation.* CVPR.
  https://openaccess.thecvf.com/content_cvpr_2016/html/Niu_Ordinal_Regression_With_CVPR_2016_paper.html (verified)
- Beckham C., Pal C. (2017). *Unimodal probability distributions for deep ordinal classification.*
  ICML, PMLR 70. https://proceedings.mlr.press/v70/beckham17a.html (verified)
- Díaz R., Marathe A. (2019). *Soft labels for ordinal regression.* CVPR. (verified in earlier study)
- de la Torre J., Puig D., Valls A. (2018). *Weighted kappa loss function for multi-class
  classification of ordinal data in deep learning.* Pattern Recognition Letters 105:144–154.
  https://www.sciencedirect.com/science/article/abs/pii/S0167865517301666 (verified)
- Polat G. et al. (2025). *Class distance weighted cross entropy loss for classification of disease
  severity.* ESWA. https://arxiv.org/abs/2412.01246 (verified in earlier study)

## Classifier vs representation; imbalance; calibration

- Kang B. et al. (2020). *Decoupling representation and classifier for long-tailed recognition.*
  ICLR. https://arxiv.org/abs/1910.09217 (verified)
- Menon A.K. et al. (2021). *Long-tail learning via logit adjustment.* ICLR. (standard reference)

## Label noise and reference standards

- Krause J. et al. (2018). *Grader variability and the importance of reference standards for
  evaluating machine learning models for diabetic retinopathy.* Ophthalmology.
  https://arxiv.org/abs/1710.01711 (verified)
- *Intergrader agreement on qualitative and quantitative assessment of DR severity using
  ultra-widefield imaging: INSPIRED Study Report 1.* https://pmc.ncbi.nlm.nih.gov/articles/PMC12293693/ (verified in earlier study)
- Ruamviboonsuk P. et al. (2019). *Deep learning versus human graders for classifying diabetic
  retinopathy severity in a nationwide screening program.* npj Digital Medicine 2:25.
  doi:10.1038/s41746-019-0099-8 (verified)
- Northcutt C.G., Jiang L., Chuang I.L. (2021). *Confident learning: estimating uncertainty in dataset
  labels.* JAIR 70:1373–1411. doi:10.1613/jair.1.12125 (verified)
- Hsu J. et al. (2020). *Improving medical annotation quality to decrease labeling burden using
  stratified noisy cross-validation.* ACM CHIL (glaucoma, not DR). https://arxiv.org/abs/2009.10858 (verified)

## Clinical grounding for PDR

- Esmaeilkhanian H. et al. (2023). *The relationship of diabetic retinopathy severity scales with
  frequency and surface area of diabetic retinopathy lesions.* Graefe's Arch Clin Exp Ophthalmol.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10587246/ (verified in earlier study)
- Wilkinson C.P. et al. (2003). *Proposed international clinical DR and DME disease severity
  scales.* Ophthalmology 110(9). (standard reference)
- *Classification of diabetic retinopathy: past, present and future.* Frontiers in Endocrinology
  (2022). https://pmc.ncbi.nlm.nih.gov/articles/PMC9800497/ (verified: search record)
- *Neovascularization detection and localization in fundus images using deep learning.* Sensors 21
  (2021). doi:10.3390/s21165327 (verified: search record)

## Datasets

- Porwal P. et al. (2018). *Indian Diabetic Retinopathy Image Dataset (IDRiD).* Data.
  https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid (verified)
- Li T. et al. (2019). *Diagnostic assessment of deep learning algorithms for diabetic retinopathy
  screening* (DDR). Information Sciences 501:511–522. https://github.com/nkicsl/DDR-dataset (verified)
- Zhou Y. et al. (2021). *A benchmark for studying diabetic retinopathy: segmentation, grading, and
  transferability* (FGADR). IEEE TMI 40(3):818–828. https://arxiv.org/pdf/2008.09772 (verified in earlier study)
- Wei Q. et al. (2020). *Learn to segment retinal lesions and beyond* (Retinal-Lesions). ICPR.
  https://github.com/WeiQijie/retinal-lesions (verified in earlier study)
- Pachade S. et al. (2021). *Retinal Fundus Multi-Disease Image Dataset (RFMiD).* Data 6(2):14.
  https://www.mdpi.com/2306-5729/6/2/14 (verified)

## Hierarchical / multi-task DR and external validation

- Wu Z. et al. (2020). *Coarse-to-fine classification for diabetic retinopathy grading using CNN*
  (CF-DRNet). Artificial Intelligence in Medicine. (search record; publisher page returned 403)
- Dai L. et al. (2021). *A deep learning system for detecting diabetic retinopathy across the disease
  spectrum.* Nature Communications. https://www.nature.com/articles/s41467-021-23458-5 (verified)
- Gulshan V. et al. (2016). *Development and validation of a deep learning algorithm for detection of
  diabetic retinopathy in retinal fundus photographs.* JAMA 316(22). (standard reference)
- Ting D.S.W. et al. (2017). *Development and validation of a deep learning system for DR and related
  eye diseases using retinal images from multiethnic populations with diabetes.* JAMA 318(22).
  https://jamanetwork.com/journals/jama/fullarticle/2665775 (verified)
- Zhou Y. et al. (2023). *A foundation model for generalizable disease detection from retinal images*
  (RETFound). Nature. https://www.nature.com/articles/s41586-023-06555-x (verified in earlier study)
- Hashmi A. (2026). *Robust DR grading using dual-resolution attention-based deep learning with
  ordinal regression.* arXiv preprint (not peer-reviewed). https://arxiv.org/html/2604.17341v1 (verified in earlier study)

## Searched, not used as evidence

- An OpenReview paper on calibration of ordinal regression networks: the page could not be retrieved
  (verification wall), so it is not cited.
- No source was found that quantifies laser-scar (PRP) prevalence in APTOS grade 4. The laser-scar
  confound stays a hypothesis, checked visually in Step LN §5.
