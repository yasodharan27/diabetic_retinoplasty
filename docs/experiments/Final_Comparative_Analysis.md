# Final comparative analysis: P, E1, E2 and the EyePACS-adapted arms P-EP, E1-EP, E2-EP

Consolidated on 2026-10-11 from the research record (`RACAF_Gate_Initialization_And_C1_Control.md`, §60 – §79) and the stored result files. Nothing was rerun for this document: every number below is a recorded result, and the section it comes from is given. Where this summary and the record differ, the record is authoritative (two small errata in the record are listed in section 9).

## 1. What was compared

All models are the same RGB grader — ConvNeXt-Tiny, global average pooling, a CORN ordinal head — trained on APTOS 2019 (2,921 training / 730 validation images) with one protocol and three seeds (42, 123, 2026). They differ in two factors only.

| Arm | Encoder it starts from | Extra training signal |
|---|---|---|
| **P** | ImageNet | none |
| **E1** | ImageNet | a 1×1 lesion head on the final feature map, trained on each image's own Stage-4 lesion map |
| **E2** (control) | ImageNet | the same head, trained on lesion maps taken from other images |
| **P-EP** | one encoder first trained on EyePACS grades | none |
| **E1-EP** | the same EyePACS-adapted encoder | as E1 |
| **E2-EP** (control) | the same EyePACS-adapted encoder | as E2 |

The EyePACS adaptation was a single supervised run of P on 31,610 EyePACS images (seed 42), selected on 3,512 held-out EyePACS validation images. All nine "-EP" models start from that one checkpoint with a freshly initialised grading head.

Data and their roles:

| Dataset | Role | Images |
|---|---|---|
| APTOS 2019 | training and validation of every arm; validation also selects the checkpoint | 2,921 / 730 |
| EyePACS (Kaggle) | adaptation only; held-out test for the adaptation model | 31,610 train, 3,512 validation, 16,249 test |
| DDR (lesion subset) | expert lesion masks for the representation probe | 383 / 148 / 225 |
| IDRiD (grading test set) | external grading evaluation | 100 primary, 103 sensitivity |

## 2. How to read the evidence

Three kinds of statement appear below and are kept apart:

- **Pre-declared.** A criterion or a primary comparison written into the record before the result existed. These are the only results called "supported" or "criterion met".
- **Descriptive.** A comparison computed with the same statistics but with no criterion fixed in advance. It is reported with its interval and is not a test.
- **Interpretation.** An explanation offered for a result and not tested.

Intervals are 95 % paired bootstrap intervals over images for the mean over the three seeds (2,000 resamples, seed 20260927). They cover image sampling only, not the variation between training runs beyond the three seeds, and not the variation of the single EyePACS adaptation.

## 3. Grading performance on APTOS validation (in distribution)

Quadratic weighted kappa (QWK) on the 730 validation images (§63, §63.4, §74, §75, §77):

| Arm | Seed 42 | Seed 123 | Seed 2026 | Mean | SD |
|---|---|---|---|---|---|
| P | 0.9184 | 0.9148 | 0.9176 | 0.9170 | 0.0019 |
| E1 | 0.9203 | 0.9134 | 0.9202 | 0.9180 | 0.0039 |
| E2 | 0.9139 | 0.9100 | 0.9180 | 0.9140 | — |
| P-EP | 0.9198 | 0.9223 | 0.9229 | 0.9216 | 0.0017 |
| E1-EP | 0.9244 | 0.9235 | 0.9274 | 0.9251 | 0.0020 |
| E2-EP | 0.9309 | 0.9218 | 0.9228 | 0.9251 | 0.0050 |

Matched-seed paired differences in QWK:

| Comparison | Mean difference | 95 % interval | Per seed | Status |
|---|---|---|---|---|
| E1 − P | +0.0010 | −0.0073 to +0.0095 | +0.002 / −0.001 / +0.003 | no difference shown |
| P-EP − P | +0.0047 | −0.0062 to +0.0147 | +0.001 / +0.008 / +0.005 | no difference shown |
| E1-EP − P-EP | +0.0034 | −0.0049 to +0.0125 | +0.005 / +0.001 / +0.005 | no difference shown |
| E2-EP − P-EP | +0.0035 | −0.0045 to +0.0119 | +0.011 / −0.001 / −0.000 | no difference shown |
| E2-EP − E1-EP | +0.0001 | −0.0077 to +0.0077 | +0.006 / −0.002 / −0.005 | no difference shown |
| E1-EP − P | +0.0081 | −0.0025 to +0.0191 | +0.006 / +0.009 / +0.010 | no difference shown |

What this shows:

- Every arm lies between 0.914 and 0.925, and **no interval for a QWK difference excludes zero**. The ordering P ≈ E1 < P-EP < E1-EP ≈ E2-EP is consistent across seeds but too small to resolve with 730 images and three seeds.
- The shuffled control E2-EP matches E1-EP exactly, so the small edge of E1-EP over P-EP cannot be attributed to lesion supervision.
- Pre-set guardrails against the same-seed baseline were breached in three places: P-EP's grade-3 recall fell below tolerance against P in seeds 42 and 123 (mean recall 0.55 → 0.45 on 39 images); E1-EP's false-urgent rate exceeded tolerance in seed 2026.
- APTOS validation is also the set that selects each checkpoint. It is not an untouched test, and APTOS has no labelled test set.

## 4. The EyePACS adaptation model on its own data

The single adaptation run stopped after 18 epochs; the checkpoint of the sixth epoch was selected and pinned (§72, §73).

| | EyePACS validation (selection set) | EyePACS held-out test |
|---|---|---|
| Images | 3,512 | 16,249 |
| QWK | 0.8274 | 0.8083 |
| AUROC, grade ≥ 3 / ≥ 4 | 0.980 / 0.977 | 0.973 / 0.979 |
| Recall, grades 0 – 4 | 0.96 / 0.32 / 0.62 / 0.52 / 0.59 | 0.96 / 0.38 / 0.56 / 0.52 / 0.53 |

The held-out test was scored once, after the checkpoint was pinned, and was read by no later step. It is in-domain performance of the adaptation model, not an external test and not a result for P-EP.

## 5. External transfer: IDRiD

IDRiD's grading test set, 100-image primary set (grades 34 / 5 / 29 / 19 / 13; three images that duplicate Stage-4 training images excluded). Batch 1 evaluated E1 and E2 (§67); batch 2 evaluated the three adapted arms (§79). P's predictions are those stored by the first IDRiD run (§61).

| Arm | QWK, seeds 42 / 123 / 2026 | Mean ± SD | AUROC ≥ 3 | AUROC ≥ 4 | Recall, grade 3 | Recall, grade 4 |
|---|---|---|---|---|---|---|
| P | 0.538 / 0.702 / 0.685 | 0.642 ± 0.090 | 0.893 | 0.861 | 0.30 | 0.21 |
| E1 | — | 0.576 ± 0.033 | 0.912 | 0.929 | 0.23 | 0.26 |
| E2 | — | 0.646 ± 0.061 | 0.916 | 0.918 | 0.32 | 0.15 |
| P-EP | 0.783 / 0.699 / 0.733 | 0.738 ± 0.042 | 0.926 | 0.946 | 0.25 | 0.26 |
| E1-EP | 0.663 / 0.756 / 0.692 | 0.704 ± 0.048 | 0.926 | 0.956 | 0.23 | 0.44 |
| E2-EP | 0.712 / 0.663 / 0.625 | 0.667 ± 0.044 | 0.906 | 0.929 | 0.19 | 0.33 |

Paired differences in QWK on the primary set:

| Comparison | Role | Mean difference | 95 % interval | Per seed | Reading |
|---|---|---|---|---|---|
| E1 − P (batch 1) | primary | −0.066 | −0.144 to −0.005 | +0.070 / −0.126 / −0.143 | lower |
| E2 − P (batch 1) | primary | +0.004 | −0.045 to +0.054 | +0.060 / +0.012 / −0.059 | comparable |
| **P-EP − P** (batch 2) | primary | **+0.097** | **+0.040 to +0.162** | +0.245 / −0.004 / +0.048 | higher |
| E1-EP − P-EP (batch 2) | primary | −0.035 | −0.090 to +0.015 | −0.120 / +0.058 / −0.041 | comparable |
| E2-EP − P-EP (batch 2) | descriptive | −0.072 | −0.124 to −0.027 | −0.071 / −0.036 / −0.108 | lower |
| E1-EP − E2-EP (batch 2) | descriptive | +0.037 | −0.026 to +0.095 | −0.049 / +0.093 / +0.067 | comparable |

The 103-image sensitivity analysis gives the same reading for every comparison (P-EP − P: +0.100, +0.044 to +0.164).

What this shows:

- **P-EP has a higher QWK than P on this set**, and the threshold-free measure agrees (mean AUROC over the four cuts +0.039, +0.019 to +0.064). This is the only place in the project where the EyePACS adaptation shows a difference.
- It is small-sample evidence with large seed variation. The QWK gain is +0.245 in seed 42, where P was at its worst (0.538), and −0.004 and +0.048 in the other two seeds. The gain in decoded grades comes from grades 0 and 1 (grade-0 recall 0.51 → 0.75) together with a loss on grade 2 (0.72 → 0.53). It requires replication on an independent external set before it can be called a general effect.
- **Neither lesion-supervised arm shows a grading advantage over P-EP.** E1-EP is comparable; E2-EP is lower in decoded grades. Without adaptation, E1 was lower than P.
- **Severe grades are not reliably classified by any arm.** Grade-3 recall is 0.19 – 0.32 and grade-4 recall 0.15 – 0.44, while AUROC ≥ 3 and ≥ 4 are 0.86 – 0.96: the models rank severe eyes well and the fixed decoding rule (cumulative probability above 0.5) assigns their exact grade poorly. The rule was not changed.
- Absolute performance on IDRiD (best mean QWK 0.74) is far below APTOS validation (0.92).

## 6. Lesion-representation probes

A fresh 1×1 probe is trained on each frozen encoder's final 16×16×768 feature map to predict, per cell, whether a lesion of each type is present. The score is the mean cell AUROC over four lesion types. Two probe lengths were declared in advance: five epochs (primary) and trained to a validation stopping rule ("converged", robustness).

### 6.1 Against Stage-4 pseudo-labels on APTOS (§63.2, §63.4)

| | P | E1 | E2 |
|---|---|---|---|
| Mean cell AUROC, five-epoch, per seed | 0.897 / 0.906 / 0.912 | 0.976 / 0.970 / 0.975 | 0.905 / 0.918 / 0.912 |

E1 − P +0.0685 (0.0659 to 0.0715, 3 / 3 seeds): pre-declared criterion met. E2 − E1 −0.0618 (−0.0644 to −0.0594): the shuffled control keeps about a tenth of the gain. The targets here are the same Stage-4 maps E1 was trained to predict.

### 6.2 Against expert lesion masks on DDR (§70, §76, §77)

Test split, 225 images neither Stage 4 nor any grader was trained on.

| Probe | P | E1 | E2 | P-EP | E1-EP | E2-EP |
|---|---|---|---|---|---|---|
| Five-epoch (primary) | 0.762 | 0.852 | 0.853 | 0.653 | 0.851 | 0.834 |
| Converged (robustness) | 0.899 | 0.929 | 0.905 | 0.878 | 0.929 | 0.912 |

| Comparison | Probe | Mean difference | 95 % interval | Positive seeds | Status |
|---|---|---|---|---|---|
| **E1 − P** | five-epoch | +0.090 | +0.075 to +0.104 | 3 / 3 | pre-declared criterion met |
| E1 − P | converged | +0.030 | +0.020 to +0.039 | 3 / 3 | agrees |
| E2 − P | five-epoch | +0.091 | +0.074 to +0.107 | 3 / 3 | control also higher |
| E2 − P | converged | +0.006 | +0.0002 to +0.012 | 3 / 3 | control nearly level with P |
| E1 − E2 | five-epoch | −0.001 | −0.016 to +0.014 | 2 / 3 | no separation |
| E1 − E2 | converged | +0.024 | +0.013 to +0.032 | 3 / 3 | separation |
| **E1-EP − P-EP** | five-epoch | +0.198 | +0.180 to +0.216 | 3 / 3 | pre-declared criterion met |
| E1-EP − P-EP | converged | +0.051 | +0.040 to +0.062 | 3 / 3 | agrees |
| E2-EP − P-EP | five-epoch | +0.181 | +0.159 to +0.202 | 3 / 3 | control also higher (descriptive) |
| E2-EP − P-EP | converged | +0.034 | +0.023 to +0.046 | 3 / 3 | control also higher (descriptive) |
| E1-EP − E2-EP | five-epoch | +0.018 | +0.003 to +0.032 | 2 / 3 | descriptive; one seed reversed |
| E1-EP − E2-EP | converged | +0.017 | +0.009 to +0.024 | 3 / 3 | descriptive |
| P-EP − P | converged | −0.021 | −0.032 to −0.011 | 1 / 3 | descriptive |
| E1-EP − E1 | converged | −0.0001 | −0.007 to +0.006 | 1 / 3 | descriptive |

What this shows:

- **A representation effect is supported.** With or without EyePACS adaptation, the encoder trained with the lesion head carries more linearly readable information about expert-annotated lesions than the encoder trained without it. Both pre-declared criteria were met, in every seed, and the converged probes agree.
- **Much of that effect is not specific to image-aligned supervision.** A head trained on lesion maps from other images reproduces all of the five-epoch gain without adaptation and about nine tenths of it with adaptation. At convergence the control reproduces about a fifth of the gain without adaptation and about two thirds with it.
- **A smaller part is specific.** At convergence the aligned arm is above the shuffled control in all three seeds in both settings (+0.024 and +0.017, both intervals above zero). On the primary five-epoch probe this separation is absent without adaptation and inconsistent across seeds with it. No criterion was pre-declared for these contrasts.
- **The honest size of the effect is the converged one** (+0.03 to +0.05), not the five-epoch one. P-type encoders' probes are far from converged after five epochs, so the five-epoch gap mixes information content with how quickly a linear probe fits. That reading is an interpretation supported by the validation-loss curves, not a separate test.
- **EyePACS adaptation alone did not make lesions more readable**; P-EP is slightly below P at convergence. Lesion supervision brings the adapted and the non-adapted encoder to the same level (0.929).
- The probe measures what can be read from frozen features. It does not show that the grading head uses that information, and it is not a lesion-segmentation result.

## 7. Research questions and their evidence status

| # | Question | Evidence | Status |
|---|---|---|---|
| 1 | Does lesion supervision (E1) change APTOS grading relative to P? | E1 − P +0.001 (−0.007 to +0.010) | **Unsupported** as an improvement; no difference shown |
| 2 | Does E1's representation encode Stage-4 lesion maps better than P's? | APTOS probe +0.069 (0.066 to 0.072), 3 / 3; pre-declared | **Supported** |
| 3 | Does E1's representation hold more information about expert lesions than P's? | DDR five-epoch +0.090 (0.075 to 0.104), 3 / 3; pre-declared; converged +0.030 | **Supported** |
| 4 | Is that gain specific to image-aligned targets (E1 against E2)? | five-epoch −0.001 (through zero); converged +0.024 (0.013 to 0.032) | **Inconclusive**: variants disagree; mostly non-specific on the primary probe |
| 5 | Does EyePACS adaptation improve APTOS grading (P-EP against P)? | +0.005 (−0.006 to +0.015); grade-3 recall guardrail breached in 2 / 3 seeds | **Inconclusive**: no difference shown |
| 6 | Does EyePACS adaptation improve external grading (P-EP against P, IDRiD)? | +0.097 (0.040 to 0.162), primary; AUROC +0.039 (0.019 to 0.064); per seed +0.245 / −0.004 / +0.048 | **Supported on this sample only**; needs independent replication |
| 7 | After adaptation, does lesion supervision still change the representation (E1-EP against P-EP)? | DDR five-epoch +0.198 (0.180 to 0.216), 3 / 3; pre-declared; converged +0.051 | **Supported** |
| 8 | After adaptation, is that gain specific to image-aligned targets (E1-EP against E2-EP)? | converged +0.017 (0.009 to 0.024), 3 / 3; five-epoch +0.018, 2 / 3; control recovers about two thirds; descriptive | **Partly supported, descriptive**: a small specific part, a larger non-specific part |
| 9 | Does adaptation alone make lesion information more readable (P-EP against P)? | converged −0.021 (−0.032 to −0.011); descriptive | **Unsupported**; slightly lower |
| 10 | Does lesion supervision improve grading, in or out of distribution? | APTOS: E1-EP − P-EP +0.003 (through zero), matched by E2-EP. IDRiD: E1-EP − P-EP −0.035 (through zero); E1 − P −0.066 (below zero) | **Unsupported** |
| 11 | Are severe grades classified reliably on external data? | IDRiD grade-3 recall 0.19 – 0.32, grade-4 recall 0.15 – 0.44, all arms | **Unsupported** |
| 12 | Is clinical utility demonstrated? | not evaluated | **Unsupported**; no claim is made |

## 8. Conclusions

### Research conclusion

On APTOS, every variant of this RGB grader reaches a validation QWK of 0.91 – 0.93, and none of the changes studied — a lesion-supervision head, a shuffled-target control, or pre-adaptation on EyePACS — produces a grading difference that the data can resolve. Supervised pre-adaptation on EyePACS is associated with better external grading on IDRiD (QWK 0.642 → 0.738), which is the one grading difference in the project whose interval excludes zero; it rests on 100 images, is dominated by one of three seeds, and was obtained on a test set the project had already used, so it stands as a result to be replicated, not as an established improvement. Lesion supervision changes what the encoder represents — expert-annotated lesions are more linearly readable from it — but most of that change is also produced by a control trained on mismatched lesion maps, a smaller part is consistently attributable to matching the maps to the image, and none of it translates into better grading on APTOS or IDRiD.

### Strongest defensible contribution

A controlled, pre-declared, matched-seed evaluation showing that:

1. auxiliary lesion supervision measurably changes a DR grader's representation, verified against independent expert lesion masks (DDR) and not only against the pseudo-labels it was trained on; and
2. a shuffled-target control accounts for most of that change, so that "image-aligned lesion supervision" is responsible for only a modest, convergence-dependent part of it; and
3. the representation change does not improve grading.

The value of the work is the separation of these three statements. Reporting the first without the second and third would overstate what lesion supervision does. The EyePACS transfer result is a secondary, promising observation.

### What is not claimed

- That lesion supervision improves DR grading.
- That EyePACS adaptation improves grading in general, or on APTOS.
- That the probe gain is specific to image-aligned supervision.
- That any arm classifies severe DR reliably, or that any result bears on clinical use.
- Equivalence or non-inferiority between any two arms: no margin was pre-specified and none was tested.
- Novelty of the architecture: the grader, the auxiliary-head design and the pre-adaptation are standard components.

### Main limitations

1. **Small and reused external test.** IDRiD's primary set has 100 images, 19 of grade 3 and 13 of grade 4. It was used three times in the project (§61, §67, §79). It is now closed: no further model is to be evaluated on it.
2. **One adaptation run.** All nine adapted models start from a single EyePACS checkpoint. The variance of the adaptation itself is unmeasured, and the smaller seed spread of the adapted arms is not evidence of robustness.
3. **Three seeds.** Seed-to-seed spread is large on IDRiD (P: SD 0.09) and intervals cover image resampling only.
4. **Selection on APTOS validation.** Every APTOS checkpoint was selected on the set it is reported on; APTOS has no labelled test set.
5. **Teacher-domain exposure.** Stage 4, the source of the lesion targets, was trained on IDRiD-domain segmentation images. This indirect exposure affects E1, E2, E1-EP and E2-EP on IDRiD and not P or P-EP.
6. **Probe scope.** One source (DDR), annotated images of grades 1 – 4 only, an image-level split, coarse 16×16 targets, a linear probe. The non-specific part of the gain may reflect how a 1×1 head conditions the feature map and not lesion information; this was not tested.
7. **Decoding.** The fixed 0.5 rule is conservative for severe grades on external data. It was deliberately left unchanged, so the decoded-grade results understate what the scores could support with a calibrated rule.
8. **Mixed runtimes.** P's IDRiD predictions were produced under an earlier TensorFlow version than the adapted arms'.

### Future work, in priority order

1. **Independent external replication of the P-EP result.** A labelled grading set that the project has never used, evaluated once under a protocol fixed in advance, with P and P-EP as the primary comparison. This is the single most valuable next step, because the only supported grading difference depends on it. IDRiD cannot serve this purpose.
2. **Measure the variance of the adaptation.** Repeat the EyePACS adaptation with additional seeds, then repeat P-EP from each, so that the IDRiD-scale comparison includes adaptation variance.
3. **Separate information from conditioning in the probe.** Analyses on the existing frozen encoders, with no new grader: a probe form that removes optimisation speed (for example a regularised closed-form linear read-out, or feature standardisation before probing), to test the interpretation in limitation 6.
4. **Severe-grade decoding.** A decision rule for grades 3 and 4 fixed on data that is neither APTOS validation nor an external test, then evaluated once on a new external set. This must be pre-registered; tuning thresholds on a test set would repeat the problem it is meant to solve.
5. **Patient-level uncertainty.** Where patient identifiers exist (EyePACS), resample patients instead of images.
6. **Broader lesion evidence.** A second expert-annotated lesion set, and annotated healthy images, to check the probe result beyond DDR.
7. **Clinical evaluation** is out of scope of the present evidence and would need its own design: referable-DR operating points, image-quality handling and prospective data.

## 9. Record integrity

The record sections §60 – §79, the three locked evaluation protocols and the stored result files were cross-checked for this document (see §80 of the record for the method). Two errata were found in the record and are corrected there by an erratum note, the original text being left in place:

- §77, APTOS table: E2-EP's mean QWK is 0.9251 (0.92515), not 0.9252. The difference from E1-EP, +0.0001, is as stated.
- §79, reading point 3: E1-EP assigns grade 4 correctly in 17 of 39 decisions over the three seeds against P-EP's 10, that is seven more, not six. The recall values 0.44 and 0.26 are as stated.

Sections §69.5 – §69.8 appear after §71 in the file because the record is append-only and they were written later; their numbering reflects their topic, not their position.

## 10. Where each result is recorded

| Result | Record section | Stored files |
|---|---|---|
| P, external reference | §60 – §61 | `experiments/IDRiDGrading/idrid_grading_f93c1237eb81_…` |
| E1 on APTOS; APTOS probe | §62 – §63.2 | `experiments/E1MultiTask/…` |
| E2 on APTOS; its probe | §63.3 – §63.4 | `experiments/E2ShuffledTargets/…` |
| IDRiD batch 1 (E1, E2) | §65 – §67 | `experiments/IDRiDGrading/idrid_batch1_e1e2_656dde8932eb_…` |
| DDR probe (P, E1, E2) | §68, §70 | `experiments/DDRProbe/ddr_ground_truth_probe_v1` |
| EyePACS protocol, caches, adaptation, held-out test | §69 – §69.8, §72, §73 | `experiments/EyePACSAdaptation/p_eyepacs_c5f7e41501b9_seed42` |
| P-EP, E1-EP, E2-EP on APTOS | §74, §75, §77 | `experiments/EyePACSAdaptedAPTOS/…` |
| DDR probe (P-EP, E1-EP, E2-EP) | §71, §76, §77 | `experiments/DDRProbe/ddr_ground_truth_probe_ep_v1`, `…_with_e2_ep` |
| IDRiD batch 2 (P-EP, E1-EP, E2-EP) | §78, §79 | `experiments/IDRiDGrading/idrid_batch2_ep_5ea937c4edb8_…` |
