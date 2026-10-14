# Deep Research — Grade 3 (severe NPDR) vs Grade 4 (PDR)

Scope: literature synthesis plus the project's own saved evidence, for the unresolved question

> How can the existing lesion information be represented and optimised so the model learns the
> Grade-3/Grade-4 distinction without sacrificing the overall ordinal task?

The companion documents are ARCHITECTURE_AUDIT.md (code and information flow),
CANDIDATE_ARCHITECTURES.md (A–H), RECOMMENDED_DESIGN.md (the protocol) and SOURCES.md (the
references, with URLs).
Nothing here was trained or run. The running 12-run auxiliary-loss experiment was not read,
touched or waited on; no §13 results existed locally when this audit ran.

---

## 1. Project evidence, re-audited (not taken on trust)

Each claim was checked against the saved reports (`docs/experiments/*.md`) and, where possible, the
code. Evidence labels: ESTABLISHED, HYPOTHESIS, CLOSED, OPEN.

### 1.1 ESTABLISHED

| # | Finding | Evidence |
|---|---|---|
| E1 | RACAF vs NO-RACAF: INCONCLUSIVE (paired Δ −0.0072, 1/3 seeds positive, every CI spans 0) | Multiseed report §8 |
| E2 | C1 and C3 landed inside the noise band (C1 decomposition terms ±0.0025; C3 +0.0018 despite residual ratio 0.073) | record §1, §3 |
| E3 | RACAF's `w_g` was never learned in a way that matters; reliability scalar contributes 0.002–0.017 QWK | record §2; Multiseed §12.3 |
| E4 | D6: reliability does not predict grading error (0/6 runs above 0.5; incremental +0.0009) | record §4 |
| E5 | Trained models depend heavily on the lesion channels (lesion permutation costs 0.63–0.74 QWK); vessel permutation costs 6–10 % as much, mainly grade-0 recall | Multiseed §12.3–12.4 |
| E6 | Generic vessel representation: grade 4 vs 3 AUROC 0.572 (CI includes 0.5); increment over lesion −0.028 | record §7 |
| E7 | `p_gt_3` ranks grade 4 vs rest well (mean AUROC 0.8815, spread 0.031) but grade 4 vs 3 near chance (0.5979) | record §8 |
| E8 | Lowering the grade-4 threshold recovers grade-4 recall out-of-sample (0.190 → 0.601, optimism 0.017) … | record §9 |
| E9 | … at the cost of grade-3 recall 0.419 → 0.162 (−61 %) and grade-4 precision 0.503 → 0.353; only 30 % of new PDR calls correct; QWK nearly blind to it (a 3↔4 error weighs 1/16 of a 0↔4 error) | record §10 |
| E10 | Grade 4 vs 3 (n = 97): lesion statistics 0.670 > `p_gt_3` direct 0.598; lesion adds +0.128 to `p_gt_3` (6/6), `p_gt_3` adds +0.013 to lesion; margin-based confidence 0.491 | record §11 |
| E11 | Stage 04 is weak (IDRiD test Dice EX 0.357, HE 0.127, SE 0.024, MA 0.017) and has no NV/IRMA/VB/VH channel | SEGMENTATION_ARCHITECTURE.md §3.5 |
| E12 | Stage 06 (92 % of parameters) is a randomly initialised Swin that sees **RGB only** at 256×256; lesion maps reach E only through Stage 07's softmax cross-attention | code (ARCHITECTURE_AUDIT.md §1) |

### 1.2 HYPOTHESIS (plausible, not tested)

| # | Hypothesis | How it could be tested |
|---|---|---|
| H1 | Stage 07's softmax attention preserves lesion *presence* but discards lesion *amount* (cardinality), and amount is what carries the 3-vs-4 signal in these four channels | Linear probes on frozen E vs GAP(L) (P0.3); a count-preserving bypass (recommended design) |
| H2 | The marginal decode `p_gt_3 = p_gt_2·σ(z_3)` penalises PDR images with lower burden, so the conditional `σ(z_3)` ranks 3-vs-4 better than the marginal | Read-only on saved `p_gt_2`, `p_gt_3` (P0.1) |
| H3 | Insufficient objective pressure on the 3/4 boundary (6.5 %/10.4 % of loss pairs; QWK-blind selection) | **Running now** (§12 aux-loss, pre-registered) |
| H4 | BatchNorm at batch size 2 blurs image-level burden in Stage 05 | Probe on GAP(L) vs lesion stats (P0.3); GroupNorm variant (not recommended now: global confound) |
| H5 | Part of the lesion "PDR" signal is laser-scar (PRP) detection by the EX channel | Visual audit of the top-EX grade-4 images |
| H6 | APTOS 3/4 label noise caps achievable AUROC well below 1 | Not testable without re-grading; recorded as an unknown ceiling |
| H7 | The in-validation 0.670 probe transfers from the training split | Train-fitted probe (P0.2) — **never run; the whole lesion direction rests on it** |

### 1.3 CLOSED (do not revisit without genuinely new evidence)

- RACAF fusion; reliability fusion of any form; C1; C3 (seeds 123/2026 never to be run); shuffled-κ
  fusion; D6 reliability selective prediction.
- The generic-vessel PDR route (E6).
- **Threshold/decision-rule calibration as a representation fix.** It is a real, bounded
  calibration result (E8–E9), not a solution to 3-vs-4, and is not proposed again here.

### 1.4 OPEN

- O1: Is the lesion-derived 3-vs-4 signal learnable from the training split (H7)?
- O2: Where between input and `p_gt_3` is it lost: Stage 05, Stage 07 aggregation, head/objective,
  or the marginal decode (H1, H2, H4)?
- O3: Does objective pressure alone recover it (H3, §12 pending)?
- O4: Can a count-preserving lesion pathway recover it without harming the ordinal task? This is
  the recommended experiment.
- O5: PDR-specific lesion information (NV, IRMA, VH, fibrous proliferation) is not in the pipeline
  at all. That needs new annotations or public datasets (FGADR, Retinal-Lesions); it is a long-horizon question.

---

## 2. Clinical grounding for 3 vs 4

- **Definitions.** ICDR (Wilkinson et al., 2003): severe NPDR follows the 4-2-1 rule (haemorrhages in
  4 quadrants, venous beading in ≥2, IRMA in ≥1) with no PDR signs; PDR = neovascularisation
  and/or vitreous or preretinal haemorrhage. The quadrant rule descends from ETDRS Report 12 (1991).
- **Lesion burden is non-monotonic across 3 → 4.** In 1,520 UWF eyes (6 centres), lesion counts
  and areas rise up to severe NPDR and **decrease from severe NPDR to PDR**: haemorrhage median
  count 40.96 → 6.23, MA 27.52 → 0.97, hard exudate 48.64 → 32.47, all p < 0.05 (Esmaeilkhanian
  et al., 2023). This is the single most important external fact for this design. It says the
  available four channels can distinguish 3 from 4 mainly through **amount** and with a
  **reversed direction** relative to grades 0→3. (Caveat: UWF imaging and a different population
  from APTOS. The direction must be verified on this project's own cached maps, P0.1.)
- **Grader variability.** Missed NV, MA-vs-haemorrhage confusion and PRP-scar interpretation are
  named sources of disagreement at the severe end (Krause et al., 2018). In a UWF study, severe
  NPDR had the lowest intergrader agreement of the ICDR categories before consensus training
  (INSPIRED Report 1). APTOS provides no adjudication, so the label ceiling for 3-vs-4 is unknown.
- **PDR-defining lesions are annotatable** in public data: FGADR Seg-set (1,842 images, includes
  IRMA and NV masks; Zhou et al., TMI 2021) and Retinal-Lesions (1,593 images; NV, fibrous
  proliferation, preretinal and vitreous haemorrhage; only 62 DR4 images; Wei et al., ICPR 2020).
  These are what a clinically faithful PDR pathway would need (candidate H in
  CANDIDATE_ARCHITECTURES.md).

---

## 3. Literature by topic — what each serious method actually offers here

Each entry covers: exact method; task and dataset; labels; architecture and loss; whether it needs
new annotations or a Stage 04 retrain; whether it can use our frozen maps; relevance to 3-vs-4. "Our
maps?" means it could consume the frozen `(512,512,4)` MA/HE/EX/SE probabilities.

### 3.1 Ordinal learning

| Method | Essentials | New labels? | Our maps? | Addresses 3v4? | Notes |
|---|---|---|---|---|---|
| **CORN** (Shi, Cao & Raschka, PAA 2023) | K−1 conditional binary tasks, task k trained on {y ≥ k}; marginal via chain rule; no weight-sharing constraint (unlike CORAL) | No | n/a (head) | Indirectly: task 3 *is* a dedicated 3-vs-4 classifier on grade ≥3 samples | Already implemented; it is already a hierarchical conditional decomposition (relevant to candidate G) |
| **CORAL** (Cao, Mirjalili & Raschka, PRL 2020) | Shared weights, per-threshold biases; rank-consistent | No | n/a | No: shared weight direction cannot reverse sign between thresholds, which is **worse** for a non-monotonic burden signal | Argument against switching to CORAL |
| **SORD** (Díaz & Marathe, CVPR 2019) | Soft labels from a distance metric, cross-entropy | No | n/a | Weakly: distance-weighted soft labels *reduce* the penalty on 3↔4 | Wrong direction for this problem |
| **CDW-CE** (Polat et al., ESWA 2025) | CE with a penalty growing with class distance | No | n/a | No: penalises far errors; adjacent 3↔4 stays cheap | Same issue as QWK |
| **Ordinal label distribution with asymmetric priors** (Shaik et al., NeurIPS'25 workshop) | WAE with asymmetric class priors + direction-aware ordinal loss for imbalanced retinal grading | No | n/a | Claims better minority grades; no per-grade 3/4 numbers in the abstract | Workshop-level evidence; heavy (generative prior); high confound |

**Synthesis.** Ordinal losses in the literature mostly make adjacent-grade errors *cheaper* than
distant ones. That is exactly the property that leaves 3↔4 under-pressured (E9). None of them is
a mechanism for extracting burden. The asymmetric/boundary-weighted variants are the relevant
family (candidate B), and the running §12 experiment already tests the core objective-pressure
hypothesis.

### 3.2 Lesion-aware DR grading

| Method | Essentials | New labels? | Our maps? | Evidence | 3v4? |
|---|---|---|---|---|---|
| **Zoom-in-Net** (Wang et al., MICCAI 2017) | Image-level supervision; attention maps pick suspicious patches, high-res patch re-examination | No | No (learns own) | EyePACS/Messidor gains | Not reported |
| **CANet** (Li et al., TMI 2020) | Disease-specific + cross-disease attention, DR & DME jointly | No | No | IDRiD/Messidor | Not reported |
| **DeepDR** (Dai et al., Nat Commun 2021) | Multi-task: IQA, lesion-aware sub-network (MA, CWS, EX, HE on 14,901 annotated images), grading | Yes (large) | Conceptually yes | 666k-image real-world system; good across the spectrum | Lesion-aware transfer helped grading; no NV channel either |
| **Lesion-based contrastive learning** (Huang et al., MICCAI 2021) | Lesion detector trained on IDRiD; lesion patches as contrastive views; then grading | IDRiD boxes (existing) | Yes (patches from our maps) | EyePACS | Representation-level; needs large unlabelled pretraining |
| **Lesion-aware contrastive learning** (Cheng et al., MICCAI 2023) | Detector-defined positives/negatives, hard-negative distillation | IDRiD (existing) | Yes | EyePACS 88.0 % ACC / 86.8 κ | Not per-grade |
| **Lesion-aware network (LAM + FPM)** (Xia et al., IJIST 2024) | Lesion-aware module + feature-preserve module; joint segmentation | Pixel lesion labels | Partially | Screening AUC 0.967 | Screening, not 3v4 |
| **DRG-Net** (Tusfiqur et al., 2022) | Joint multi-lesion segmentation + grading with lesion↔classification attention; IDRiD/FGADR | Pixel labels | Partially | Reported SOTA on IDRiD/FGADR | Not isolated |
| **Clinical-guided lesion-aware attention** (Saleem et al., Sci Rep 2026) | EfficientNet-B4 + spatial/channel attention, adversarial augmentation, MC uncertainty | No | No | Claims G3 F1 0.718 → 0.847, G4 0.731 → 0.879 on APTOS | **Low credibility for this question:** reports "5,590 APTOS images" (APTOS 2019 has 3,662 labelled), random 80/20 split, and APTOS has known train/val duplicates. Not usable as evidence |

**Synthesis.** Lesion-aware grading is well established. Almost every method injects lesion
information as **attention** or **multi-task supervision** over spatial features. This project
already does the analogous thing (lesion maps as input channels plus cross-attention). No paper
found isolates 3-vs-4, and none reports a controlled test of *burden* vs *presence*. Novelty
claims should therefore be avoided. The defensible contribution is a controlled mechanistic test
inside this pipeline, not a new lesion-aware architecture.

### 3.3 Aggregation, cardinality and pooling (the mechanism behind H1)

- **Softmax attention discards cardinality.** Attention aggregators cannot distinguish multisets
  that differ only in multiplicity, because weights are normalised to sum to 1 (Zhang & Xie,
  IJCAI 2020, "cardinality-preserved attention"). Mean and max aggregators have the same
  limitation (Xu et al., ICLR 2019, GIN). Sum pooling over a set is the universal choice (Zaheer
  et al., Deep Sets, NeurIPS 2017). Attention-MIL pooling (Ilse et al., ICML 2018) is also a
  normalised weighted average. The Set Transformer (Lee et al., ICML 2019) pools with learned
  seed queries (PMA), which is again normalised attention.
- **Implication.** A GAP over a *fixed* grid (constant N) is proportional to a sum and preserves
  amount. A softmax cross-attention read-out does not. Stage 07 is the latter.
  Cardinality-preserving fixes in the literature add an unnormalised (sum or count) channel next to
  the attention output. That is exactly the shape of the recommended pathway.

### 3.4 Hierarchical / coarse-to-fine DR grading

- **CF-DRNet** (Wu et al., Artif Intell Med 2020): coarse No-DR/DR network with attention gates,
  then a fine 4-class network. **CORN is already a conditional hierarchy** (task 3 = 3-vs-4
  given y ≥ 3), so a hierarchy adds nothing unless it also adds a *different representation*
  for the top branch. With 390 training images for that branch, a separate backbone would overfit.

### 3.5 Minority/severe grades — auxiliary objectives and rebalancing (context for B)

- Class-balanced reweighting (Cui et al., CVPR 2019), focal loss (Lin et al., ICCV 2017), LDAM
  (Cao et al., NeurIPS 2019), logit adjustment (Menon et al., ICLR 2021) and decoupled
  representation/classifier training (Kang et al., ICLR 2020) are the standard tools. They change
  *how much* the boundary is weighted, and they make no specific claim about *what* representation
  carries the signal.
- Pairwise/ranking objectives between adjacent grades need both grades in a batch. At the frozen
  batch size of 2 that happens in ~12 batches per epoch (ARCHITECTURE_AUDIT.md L7). They are
  infeasible without a memory bank, which would be a larger protocol change.

### 3.6 Normalisation at tiny batch sizes (context for H4)

- BN error rises steeply at small batch; at batch 2, GN beat BN by 8.9 points on ImageNet
  ResNet-101 (Wu & He, ECCV 2018). Relevant to Stage 05, but swapping BN would change every grade
  and every run. That is a global confound, so it is out of scope for a 3-vs-4 mechanistic study.

### 3.7 Foundation backbones (context only)

- RETFound (Zhou et al., Nature 2023): MAE pretraining on 1.6 M retinal images; APTOS AUROC 0.943
  after fine-tuning. A pretrained backbone would likely improve everything. It would replace 92 %
  of the parameters and so confound any 3-vs-4 attribution. It is not a 3-vs-4 mechanism, so it is
  kept out of this study (recorded in CANDIDATE_ARCHITECTURES.md as H2, not shortlisted).

### 3.8 Calibration / decision rules (supporting context only)

- Already characterised in-project (E8–E9). Logit adjustment (Menon et al.) is the principled
  version of what the nested threshold did empirically. It stays a calibration layer, not a
  representation fix, and is not proposed.

---

## 4. Synthesis: separating the five layers the task asks us to separate

| Layer | Current evidence | What remains unknown |
|---|---|---|
| **Information availability** (is 3v4 info in the input?) | Yes, in-validation (E10). Clinical data explain *why*: burden falls at PDR | Whether it is learnable from the 390 training images (H7) |
| **Representation quality** (does E carry it?) | Indirect only: `p_gt_3` is poor, but nobody has probed E | Probe E vs GAP(L) (P0.3) |
| **Training objective** (is there pressure on the boundary?) | Weak pressure by construction (L7) | §12 is testing it now |
| **Ordinal decision rule** (does the marginal product hurt?) | The fixed 0.5 rule hurts recall (E7–E8); the marginal-vs-conditional question is untested | P0.1, read-only |
| **Calibration** | Thresholds move along a poor curve (E9) | Nothing further needed |

The distinguishing feature of this problem is that **amount** matters and **direction reverses**
at 3→4. The current aggregation is structurally blind to amount (H1), and the current decode can
penalise reversed direction (H2). Those two points, not "more attention" or "more capacity", are
where a mechanistic architecture change should aim. The recommended design (RECOMMENDED_DESIGN.md)
therefore:

1. first **localises** the loss read-only (P0.1–P0.4), costing no training;
2. then tests, with an information-destroyed, capacity-identical control, whether a
   **count-preserving lesion-burden pathway** into the CORN head recovers 3-vs-4 without harming
   the ordinal task.
