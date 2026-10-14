# Post-S5 Research Direction Gate (2026-09-30)

- **Evidence base:** research record §§19–27.1 and the earlier project literature audits
  (`NEXT_STEP_LITERATURE_REVIEW.md`, `SOURCES.md`), plus a fresh web search on the one remaining problem.
- **Scope:** no experiments were run for this gate. Nothing was trained or implemented.

---

## 1. Established findings (Stages 5–8)

1. **P is the strong baseline, and the Stage 3/4 priors add nothing to it.**
   - An ImageNet-pretrained ConvNeXt-Tiny on RGB (P) far exceeds the random-init Stage 5–8 pipeline.
   - Early fusion of frozen Stage 3/4 priors adds < 0.01 AUROC (§22).
2. **No readout bottleneck (§23).** Global pooling does not discard linearly recoverable grade
   information: `pooled_final` is the best linear readout, and spatial, multi-scale and early-stage
   readouts add nothing.
3. **Grade 3 vs 4 information is present but saturated (§24).**
   - It is linearly recoverable at about 0.85 from epoch 2 and never improves on validation.
   - Adjacent-grade difficulty rises smoothly with severity; 3 vs 4 is not uniquely hard at a matched
     training size.
   - The train-to-validation gap grows with severity.
4. **Grade 4 does not continue the NPDR severity axis of the representation (§24).**
5. **CORN decoding is faithful (§25).** The decoded 3-vs-4 loss comes entirely from the
   P(Y ≥ 3) judgement of a grade-4 subset, and the frozen representation reproduces that judgement.
6. **The recurrent grade-4 failures are a stable image set (§26).**
   - The same images fail across seeds, checkpoints and two unrelated architectures, always toward
     grades 1–2.
   - They have about 2.5× lower segmented haemorrhage/exudate burden than recognised grade-4 images,
     with normal image-quality statistics.
   - Documented label conflicts are sparse, but label coverage is only about 12 %.
7. **Burden matching is inconclusive (§27, §27.1).** After matching on burden, label-free ImageNet
   features separate grade 4 from grade 2 only weakly (AUROC 0.72, p = 0.016). The fine-tuned P
   features are contaminated by the training labels.

## 2. Closed directions

- **Architecture and fusion:** RACAF and reliability-fusion variants; D6 selective prediction; the
  generic-vessel PDR route; P/PL early fusion of Stage 3/4 priors.
- **Head and pathway:** C1 two-route head; G0 pathway rebalancing.
- **Readout and objective:** pooling/readout redesign (spatial, multi-scale, MIL / lesion-query
  pooling motivated by "GAP loses information"); replacing CORN decoding or the objective.
- **Diagnostics:** APTOS-internal grade-4 failure diagnostics (closed by the user; reopen only with
  new external evidence).

## 3. The remaining problem

A reproducible subset of PDR-labelled images has low NPDR-type lesion burden, and every model places
it with moderate NPDR.

APTOS cannot say whether these images carry PDR-defining evidence that the models fail to recognise,
or whether their labels are unreliable. Two reasons:
- APTOS has no annotations for PDR-defining signs: neovascularisation (NV), preretinal/vitreous
  haemorrhage (PRH/VH), fibrous proliferation, or treatment marks.
- Its labels are unadjudicated.

The Stage-4 channels (MA/HE/EX/SE) cannot represent PDR-defining signs by design.

Clinical fact (verified in the EyePACS grading protocol): VH/PRH counts as PDR even when no NV or
fibrous proliferation is seen. PDR is therefore **not** "more NPDR lesions". Lesion counts fall from
severe NPDR to PDR (Esmaeilkhanian 2023, already in `SOURCES.md`).

## 4. Viable research questions (at most three)

### RQ1 — Do the low-burden PDR failures carry annotated PDR-defining signs on independently graded data?
- **Question:** on an external dataset with expert grades and PDR-sign annotations, is
  grading-model failure on PDR concentrated in low-NPDR-burden images, and are those images enriched
  for annotated NV / proliferative membrane / laser marks (FGADR), or PRH/VH / fibrous proliferation
  (Retinal-Lesions), compared with recognised PDR?
- **Literature status:**
  - NV detection exists but is small-scale and weak (FGADR NV PR-AUC about 0.34).
  - Laser-photocoagulation detection is established (AUC 0.95–0.98).
  - Lesion-aware grading is established.
  - "Hidden stratification" subgroup-failure analysis is established methodology (Oakden-Rayner 2020;
    MICCAI 2025).
  - **Not found in the search:** a characterisation of the low-NPDR-burden PDR stratum by PDR-sign
    annotations, linked to grading-model failure. This is **underexplored as an analysis**, not a new
    method.
- **Data:**
  - FGADR Seg-set: 1,842 images; 287 grade 4 / 647 grade 3; three-ophthalmologist grades; pixel NV and
    IRMA masks; image-level laser-mark and proliferative-membrane labels, which must be verified on
    receipt. Available under a signed research-use agreement; the Grade-set is unreleased.
  - Retinal-Lesions: 62 DR4 / 99 DR3; masks include NV, fibrous proliferation, PRH and VH. Free on
    request.
- **Supervision needed:** none. This is frozen inference (Stages 2–4 and P) on external images, plus
  the datasets' own annotations.
- **Feasibility:** high once access is granted. It needs a Colab T4 for inference; nothing is trained
  or tuned.

### RQ2 — Is the fixed APTOS recurrent-failure set a label-reliability problem?
- **Literature status:** established. Grader variability, missed NV and PRP interpretation are known
  (Krause 2018). APTOS same-image label conflicts are about 24–29 % (§15; an external 2026 report
  gives 29 %). Answering this is data curation, not research novelty, though it is decisive for this
  project.
- **Data:** blinded adjudication by a retinal specialist of the fixed sets from §26: B (17),
  controls D (29) and burden-matched grade 2.
- **Feasibility:** depends entirely on clinician access; about 60–80 images.

### RQ3 — Conditional on RQ1 being positive: does sign-level PDR evidence help grading on that stratum?
- **Literature status:** incremental. Lesion-aware and multi-task grading, and NV/laser detectors,
  exist. It would be application-level unless RQ1 shows a specific, measurable stratum those methods
  miss.
- **Data:** training on FGADR/Retinal-Lesions annotations. It carries domain shift, and a small
  number of PDR-sign examples.
- **Feasibility:** premature. **Not to be designed until RQ1 has returned.**

## 5. Single next investigation

**RQ1, external replication on the FGADR Seg-set** (Retinal-Lesions as the confirmatory second
source), read-only.

- **Rules:**
  - obtain the data under its licence;
  - run frozen Stage 2–4 and frozen P (3 seeds, BEST) inference with no tuning;
  - before looking at any result, pre-register a within-dataset test.
- **Test 1:** among expert-graded grade-4 images, is P's failure (decoded ≤ 2) associated with low
  Stage-4 NPDR burden?
- **Test 2:** are failed PDR images enriched, relative to recognised PDR, for annotated PDR-specific
  signs (NV / proliferative membrane / laser mark) versus none?
- **Why this one:** it separates "APTOS label artefact" (the pattern does not replicate on
  better-graded data) from "a PDR-evidence stratum the models miss" (it replicates, and the failures
  carry annotated PDR signs).
- **No hidden action:** DDR stays reserved (MA/HE/EX/SE masks only, with an unusual 236:913
  grade-3:grade-4 ratio) and IDRiD stays single-use. Nothing is trained.
- **Blocking action for the user:** sign and send the FGADR research-use agreement (and optionally
  the Retinal-Lesions request form).
