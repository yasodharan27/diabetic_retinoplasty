# Literature Review — next step for the Grade-3/Grade-4 bottleneck

Focus: what published evidence says about the mechanisms now in play. Citations are in
NEXT_STEP_SOURCES.md; "(verified)" there means the source was opened or its record retrieved during
this study. **Established** = shown in a peer-reviewed or primary source; **inference** = my
reading, applied to this project.

---

## 1. What CORN is, and how it relates to the classical ordinal families

- **CORN** (Shi, Cao & Raschka, *Pattern Analysis and Applications* 2023) trains K−1 binary tasks,
  task k on the conditional subset `y ≥ k`, and recovers marginals by the chain rule. Rank
  consistency comes from the product, not from weight sharing. The authors report CORN outperforming
  CORAL, attributing it to dropping CORAL's weight-sharing constraint. *Established.*
- The classical statistics literature calls this construction the **sequential or continuation-ratio
  model** (Tutz 1991). Tutz's 2022 review places it alongside cumulative (proportional-odds) and
  adjacent-category models as the three main ordinal families. Continuation-ratio logits model
  `P(y = k | y ≥ k)`, and the likelihood factorises into binary logistic terms. *Established.*
  **Inference:** CORN is a continuation-ratio model with category-specific slopes on a learned
  representation. `corn_loss` is its negative log-likelihood (NEXT_STEP_AUDIT §3).
- **Cumulative-link models** project each case onto one latent axis with ordered cut-points
  (McCullagh 1980; deep version: Vargas, Gutiérrez & Hervás-Martínez, *Neurocomputing* 2020). CORAL
  (Cao, Mirjalili & Raschka, *Pattern Recognition Letters* 2020) similarly shares one weight vector
  across thresholds. *Established.* **Inference:** both assume a single ordered direction, which
  is ill-suited when lesion burden rises to grade 3 and falls at grade 4.
- **Independent binary decompositions** (Frank & Hall, ECML 2001; Niu et al., CVPR 2016 OR-CNN)
  learn each P(y > k) from all data, without chaining; rank consistency is not guaranteed.
  *Established.*
- **Survey evidence:** Gutiérrez et al. (*IEEE TKDE* 2016) benchmark ordinal methods and find no
  family uniformly best. The choice should follow the data's structure. *Established.*

**Consequence.** Nothing in this literature says CORN is defective. The literature says the
conditional (continuation-ratio) quantity is the natural object for "how likely is the next grade
given at least this grade". That is exactly `p_cond_3` for 3-vs-4.

## 2. Ordinal vs nominal, and distance-aware losses, for DR

- Ordinal formulations are widely used for DR because errors should be near-misses. Kappa-based
  losses (de la Torre, Puig & Valls, *Pattern Recognition Letters* 2018), soft ordinal labels (Díaz &
  Marathe, CVPR 2019) and class-distance-weighted CE (Polat et al., *ESWA* 2025) all penalise
  **distant** errors more than adjacent ones. *Established.*
- **Inference:** these methods make a 3↔4 error *cheaper* relative to distant errors, which is the
  opposite of what this bottleneck needs. None of the DR papers found reports a severe-NPDR-vs-PDR
  improvement from switching between ordinal and nominal formulations. Evidence that a different
  output family fixes *this* failure: **none found**.
- Unimodal output constraints (Beckham & Pal, ICML 2017) regularise the shape of the predicted
  distribution. They do not relate to the observed failure.

## 3. Hierarchical and multi-task DR models

- Coarse-to-fine DR grading (CF-DRNet, Wu et al., *Artificial Intelligence in Medicine* 2020) and
  multi-task lesion + grading systems (DeepDR, Dai et al., *Nature Communications* 2021; joint
  segmentation-grading frameworks) are established. *Established.*
- **Inference:** CORN already is a conditional hierarchy. Hierarchical designs add value only when
  the upper branch gets different *information*, which is what the PDR-specific-lesion literature
  (§5) is about.

## 4. Representation vs classifier: decoupling

- Kang et al. (ICLR 2020) show that, in long-tailed recognition, representations learned with
  standard sampling are often good enough, and **re-training only the classifier** (cRT) on frozen
  features recovers much of the rare-class performance. *Established* (natural images).
- **Inference:** §13's probe pattern (E 0.71 > jointly trained head 0.64) is the long-tail signature
  Kang et al. describe. A decoupled CORN-head re-estimation on frozen E is the direct, cheap test
  (Step cRT). Whether it transfers to a 5-class ordinal medical task with 390 boundary examples is
  unproven.

## 5. PDR-specific signs and the non-monotonic burden

- Lesion counts and areas rise to severe NPDR and **fall** from severe NPDR to PDR. Haemorrhage
  medians fall 40.96 → 6.23, MA 27.52 → 0.97 (Esmaeilkhanian et al., *Graefe's Archive* 2023; 1,520
  UWF eyes). *Established.*
- PDR is defined by neovascularisation (NVD/NVE) and/or vitreous or preretinal haemorrhage, and
  severe NPDR by the 4-2-1 rule (ICDR, Wilkinson et al. 2003; ETDRS Report 12). *Established.*
- Deep NV detection on fundus photographs exists but is small-scale and hard, because NV is small
  and irregular (e.g. *Sensors* 2021, NV detection/localisation). Public pixel-level NV/IRMA
  annotation exists in FGADR (Zhou et al., *IEEE TMI* 2021) and Retinal-Lesions (Wei et al., ICPR
  2020; only 62 DR4 images). *Established.*
- **Inference:** a PDR-feature pathway is the only intervention aimed at the *cause* suggested by
  mechanism 8. It is also the most expensive, most domain-shifted, and hardest to attribute. It is
  justified only once the failing PDR images are shown to be genuine, untreated PDR (Step LN).

## 6. Label noise and reference standards in DR grading

- Grader variability is substantial. Missed NV, MA-vs-haemorrhage confusion and PRP-scar
  interpretation are named sources of disagreement. Adjudicated consensus measurably changes
  algorithm evaluation (AUC 0.942 → 0.986 for moderate+ DR against adjudicated labels) (Krause et
  al., *Ophthalmology* 2018). *Established.*
- In a UWF grading study, severe NPDR had the lowest pre-consensus intergrader agreement among ICDR
  categories (INSPIRED Report 1). *Established.*
- In a nationwide screening programme, adjudicated reference grades by retinal specialists were
  needed to compare algorithms and human graders (Ruamviboonsuk et al., *npj Digital Medicine* 2019).
  *Established.*
- Methods to **estimate** (not change) label noise include confident learning (Northcutt, Jiang &
  Chuang, *JAIR* 2021), which estimates the joint distribution of given vs latent labels from
  out-of-sample predicted probabilities, and stratified noisy cross-validation for medical annotation
  quality (Hsu et al., ACM CHIL 2020, glaucoma). *Established.*
- **Inference:** APTOS labels come without adjudication. This project's own byte-identical
  duplicates disagree in 8/41 pairs overall and 6/12 pairs touching grade 3 or 4. Label noise is
  therefore a first-class candidate explanation for the 3-vs-4 plateau and for part of the PDR ≤ 2
  failure, and can be estimated read-only (Step LN).

## 7. External validation

- Large DR systems were validated on independent populations (Gulshan et al., *JAMA* 2016; Ting et
  al., *JAMA* 2017), and cross-dataset performance drops are common (e.g. APTOS-trained QWK 0.88 →
  Messidor-2 0.68 in a 2026 CORAL study). *Established.*
- RETFound (Zhou et al., *Nature* 2023) reports APTOS AUROC 0.943 and lower cross-dataset AUROCs
  (IDRiD 0.822 when fine-tuned on APTOS). *Established.*
- Candidate external grading sets: IDRiD (516 images; Porwal et al., *Data* 2018), DDR (13,673
  images, 7 graders with consensus; Li et al., *Information Sciences* 2019), Messidor-2.
  **Inference:** the IDRiD **test** split alone has 19 grade-3 and 13 grade-4 images (verified on
  Drive in this project), too few for a decisive 3-vs-4 contrast (SE ≈ 0.1). Adding the IDRiD
  **training** grading split, after excluding any overlap with the 81 Stage-04 training images,
  roughly quintuples it. DDR is the largest adjudicated option and requires no annotation.

## 8. Calibration and decision rules for ordinal models

- Threshold/decision-rule adjustments move along an existing ranking curve. This project already
  showed that for grade 4 (§8–§10: recall up, grade-3 recall −61 %). Logit adjustment (Menon et al.,
  ICLR 2021) is the principled version of what the nested threshold did empirically.
  *Established* in-project. **Inference:** no literature found supports a decision-rule change as a
  fix for a *ranking* deficit at an adjacent boundary.

---

## What the literature adds, in one paragraph

The literature does not point to a better ordinal family for this failure. CORN is the most flexible
of the standard families for a non-monotonic sign, and its conditional output is the principled 3-vs-4
estimand. The literature strongly supports two things this project has **not** yet done: estimate
label noise at the severe end before interpreting small AUROC differences, and confirm any
APTOS-validation finding on an external, adjudicated set. It supports decoupled classifier
re-estimation as a cheap test of head under-use. It supports a PDR-feature pathway only as a
later, data-heavy step whose premise (genuine untreated PDR among the failures) is currently unverified.
