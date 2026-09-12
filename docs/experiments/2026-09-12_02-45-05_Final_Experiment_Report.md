# Final Experiment Report — Joint Stage 05–08 + RACAF CORN Classifier

**Experiment ID: `2026-09-12_02-45-05`**

Status: **COMPLETE and FROZEN.** No part of this experiment — architecture, hyperparameters,
dataset split, checkpoints, baseline, cache, RACAF or CORN — was modified in producing this
report. Every figure below was read from the saved artifacts and the executed notebook, or
recomputed from the saved checkpoints by the read-only diagnostic chain `[D2]`–`[D13]`.

---

## 1. Experiment identification

| Item | Value |
|---|---|
| Experiment ID | `2026-09-12_02-45-05` |
| Experiment directory | `/content/drive/MyDrive/DiabeticRetinopathy/experiments/FinalClassification/2026-09-12_02-45-05` |
| Final archived model | `/content/drive/MyDrive/DiabeticRetinopathy/exported_models/FinalClassification/2026-09-12_02-45-05_BEST` |
| Notebook | [colab/notebooks/stage08_corn_classifier.ipynb](../../colab/notebooks/stage08_corn_classifier.ipynb) |
| Configuration hash | `3f549e1638d9409f7862f1e799bd7052` |
| Split manifest SHA-256 | `f512a7a086ac7f53e1a5ef8b49a703ae0c5e70db37891253640c91306d965837` |
| Git commit recorded in the checkpoints | `87d56719726c82004b3561d8e14f101ab8e85964` |
| BEST model weights SHA-256 | `7b780c230e4ce83ae9e9b1c0bdb9ae73c707b262e5a9c759748343f9417b8b9a` |
| Dataset version tag | `aptos2019-joint-cache-v1` |
| Diagnostic report generated | 2026-09-12T12:00:51 |
| Runtime | Python 3.13.15, TensorFlow 2.20.0, CUDA 12.5.1, single Tesla T4, `mixed_float16` |

Machine-readable artifacts live in the experiment's `evaluation/` directory
(`final_diagnostic_report.json`, `final_diagnostic_report.md`, `evaluation_manifest.json`,
`best_metrics.json`, `confusion_matrix.csv`, `ordinal_error_analysis.json`,
`per_sample_corn_predictions.csv`, `calibration_diagnostics.json`, `best_vs_last.json`,
`artifact_verification.json`, the eight `cases_*.csv` lists, and `plots/`). This report
summarises them and does not reproduce the large per-sample tables.

A note on provenance: the checkpoints record commit `87d5671`, the commit the training run was
launched from. The read-only diagnostic chain `[D2]`–`[D13]` was added afterwards in commit
`5c0251e` and operates only on the saved checkpoints, so it could be applied to the finished run
without altering it.

---

## 2. Research objective

Train the joint Stage 05–08 graph with RACAF end-to-end on the project's APTOS 2019 split, as an
ordinal five-grade diabetic-retinopathy classifier using the CORN formulation, and establish
whether the configuration materially outperforms the recorded baseline-0 run under a comparison
rule fixed in advance.

The single deliberate change from baseline-0 (`2026-09-11_12-18-34`) was the initial learning
rate, 1e-3 → 1e-4, together with an EarlyStopping patience of 12 instead of 8 so that two
learning-rate reductions each get a full window. The optimizer type, architecture, split,
preprocessing and loss are unchanged. This experiment is therefore a **learning-rate and
patience** comparison against baseline-0 — it is not an ablation of any architectural component.

---

## 3. Experimental architecture

Model identifier `joint_stage05_08_racaf`. Parameter counts as verified at construction and again
by `[D10]`:

| Quantity | Value |
|---|---|
| Trainable parameters | 43,338,506 |
| All weights incl. BatchNorm statistics | 43,342,346 |
| Non-trainable parameters | 3,840 |
| Trainable tensors | 409 |

| Component | Output shape | Parameters |
|---|---|---|
| `stage5_input` | (None, 512, 512, 8) | 0 |
| `stage6_input` | (None, 256, 256, 3) | 0 |
| `reliability` input | (None, 1) | 0 |
| Stage 05 local feature extraction + AdaptiveBranchFusion | (None, 32, 32, 256) | 2,174,688 |
| Stage 06 global feature extraction, dual-scale Swin | (None, 64, 1152) | 39,697,956 |
| Stage 07 adaptive cross-attention fusion | (None, 256) | 1,173,504 |
| RACAF fusion | (None, 256) | 295,170 |
| CORN head | (None, 4) | 1,028 |

Stages 03 (vessel segmentation) and 04 (lesion segmentation) are **frozen and outside this
trainable graph**: their outputs enter through the cache layer as pre-computed artifacts. Their
checkpoints were resolved and existence-checked at session start
(`VesselSegmentation/best_model.pth`, `LesionSegmentation/best_model.keras`) but never loaded into
this graph or written to. The Stage 1 IQA checkpoint was resolved for path completeness only and
is not part of this graph.

Precision policy is established before any layer is constructed, so `mixed_float16` is genuinely
in force; the optimizer is `LossScaleOptimizer(inner=Adam)`. All seven model-construction checks
passed, including "optimizer never stepped (iterations 0)" at build time.

---

## 4. Dataset and validation protocol

Source: APTOS 2019, through the project's committed split manifest
`dataset_splits/aptos2019_train_val_split.csv` (seed 42, validation fraction 0.2). The split was
verified exactly at session start — size, per-grade counts, disjointness, and SHA-256.

| Split | Total | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 |
|---|---|---|---|---|---|---|
| Train | 2,929 | 1,444 | 296 | 799 | 154 | 236 |
| Validation (manifest) | 733 | 361 | 74 | 200 | 39 | 59 |
| Validation (evaluated) | 730 | 361 | 74 | 198 | 39 | 58 |

**Exact evaluated population.** Three validation images are excluded:
`262ad704319c`, `26453eb7e989`, `3a122851e526`. Stage 03 detects no fundus disk in these (empty
field of view), so the frozen FOV-crop step cannot process them and the generator skips them.
These are three of eleven such images across the whole split; `[6]` confirmed none is recoverable
from Drive. The exclusions are **reported, never silently dropped** — `[D2]` asserts that the
evaluated set plus the reported skips accounts for the entire validation split (730 + 3 = 733),
and `[D8]` asserts that no skipped image appears in any image panel.

Every metric in this report therefore describes **730 images**, with grade-2 support 198 and
grade-4 support 58 rather than the manifest's 200 and 59.

Validation batches: 365 at batch size 2, matching expectation exactly, with augmentation and
shuffling both off on the validation branch.

**This is a validation split, not a test set.** See §18 and §19.

---

## 5. Preprocessing and data pipeline

Training and evaluation read a pre-computed cache rather than the raw dataset. The cache holds
four artifacts per image — `rgb` (Stage 02-processed canonical RGB), `vessel`, `lesion` and
`reliability` — 3,651 of each, i.e. every split entry except the eleven empty-FOV ones. The
8-channel Stage 05 input is assembled from the canonical RGB together with the frozen Stage 03 and
Stage 04 maps; Stage 06 receives a 256×256 RGB view; RACAF receives the scalar reliability value.

Cache verification at session start:

| Check | Result |
|---|---|
| Total split entries | 3,662 |
| Fully local | 3,651 |
| Drive fallback required | 0 |
| Missing everywhere (known empty-FOV) | 11 |
| Corrupt / zero-byte local files | 0 / 0 |
| Local cache size | 28.53 GiB (8 archive shards) |

The evaluation session rebuilt the validation pipeline through the **same production call** used
for training (`load_joint_training_datasets`), using only its validation half. Because validation
is unshuffled and unaugmented, these are exactly the batches Keras validated on during training.
Eleven raw images were staged locally solely so the generator's graceful empty-FOV skip does not
become a `FileNotFoundError`; this is read-only with respect to Drive and writes no cache file.

---

## 6. Training configuration

| Setting | Value |
|---|---|
| Batch size | 2 |
| Epoch cap | 50 |
| Monitored metric | `val_QWK`, mode `max` |
| Optimizer | Adam, wrapped in `LossScaleOptimizer` |
| Initial learning rate | 1e-4 |
| Loss | `joint_corn_loss` (CORN) |
| Mixed precision | `mixed_float16` |
| EarlyStopping | patience 12, `restore_best_weights=True` |
| ReduceLROnPlateau | patience 4, factor 0.5, min_lr 1e-6 |

All of these are pinned by config hash `3f549e1638d9409f7862f1e799bd7052`, which `[R1]` confirmed
is recorded identically in `metadata.json` and in every surviving checkpoint (`gen_00029`,
`gen_00030`, `best`). A resume whose hash differs is refused rather than silently continued.

Optimizer step accounting: 43,810 recorded iterations against 1,461 train batches × 30 epochs =
43,830, i.e. 20 steps (0.05%) skipped by the loss scaler — within the <1% tolerance.

---

## 7. CORN formulation and ordinal prediction

The head emits four logits `z = (z₀, z₁, z₂, z₃)`, each the logit of a **conditional**
probability `P(Y > k | Y ≥ k)`. By the chain rule the cumulative threshold probabilities are

```
P(Y > k) = ∏(i=0..k) σ(z_i)
```

and the production decode is the cumulative-threshold count:

```
grade = count( P(Y > k) > 0.5 )
```

`[D5]` verified, on all 730 samples, that:

- `P(Y > k)` is non-increasing in `k` (it is a cumulative product of values in (0,1));
- the finite-difference reconstruction `P(Y=0) = 1 − p₀`, `P(Y=k) = p_{k−1} − p_k`,
  `P(Y=4) = p₃` is non-negative and sums to 1 (max |sum − 1| = 4.54 × 10⁻⁸);
- the grade recomputed from `P(Y>k)` equals the production decode exactly.

That verified reconstruction is what makes the Brier score and ECE in §13 defensible rather than
invented. `[D3]` separately confirmed that `corn.decode_logits` reproduces the training metric's
own decode, and that the project's QWK formula and scikit-learn's quadratic kappa agree to better
than 1e-6.

**Decode versus argmax.** The argmax of the reconstructed class distribution differs from the CORN
decode on 38 of 730 samples (5.21%). This is expected and is not a defect: the decoder *is* the
cumulative-threshold rule, and the two need not agree. Both ECE conventions are reported below for
completeness. No change to the decoder is recommended or implied.

---

## 8. RACAF integration

RACAF (reliability-aware conditional adaptive fusion) sits between Stage 07 and the CORN head,
contributing 295,170 trainable parameters. It consumes the Stage 07 fused representation, the
Stage 06 global representation, and the scalar per-image reliability input, and produces the
256-dimensional representation the CORN head reads. It was trainable throughout, inside the joint
graph, and its parameters are part of the archived delivery model.

**What this experiment establishes about RACAF: that the final joint model incorporating RACAF
achieved the validation performance reported below.** No RACAF ablation — with/without, or
reliability-ablated — was run in this experiment. Nothing here isolates or quantifies RACAF's
individual contribution, and no such claim is made. A controlled ablation is listed in §21.

---

## 9. Training trajectory and checkpoint selection

Training ran 30 epochs of a 50-epoch cap and was stopped by EarlyStopping. `[R1]` independently
replayed a patience-12 EarlyStopping over the logged `val_QWK` series and confirmed it stops at
exactly epoch 30, matching the recorded `stopped_epoch`.

| Epoch | train loss | train QWK | val_loss | val_QWK | lr |
|---|---|---|---|---|---|
| 1 | 0.6038 | 0.4657 | 0.5374 | 0.6559 | 1e-4 |
| 2 | 0.3974 | 0.6798 | 0.5874 | 0.6099 | 1e-4 |
| 3 | 0.3590 | 0.7225 | 0.3835 | 0.7210 | 1e-4 |
| 4 | 0.3256 | 0.7424 | 0.5488 | 0.5956 | 1e-4 |
| 5 | 0.3231 | 0.7482 | 0.4674 | 0.6315 | 1e-4 |
| 6 | 0.3231 | 0.7542 | 0.4588 | 0.6073 | 1e-4 |
| 7 | 0.3029 | 0.7661 | 0.6032 | 0.3439 | 1e-4 |
| 8 | 0.2715 | 0.7944 | 0.5255 | 0.4662 | 5e-5 |
| 9 | 0.2679 | 0.8020 | 0.3689 | 0.7093 | 5e-5 |
| 10 | 0.2654 | 0.8107 | 0.3973 | 0.6652 | 5e-5 |
| 11 | 0.2598 | 0.8184 | 0.3803 | 0.7039 | 5e-5 |
| 12 | 0.2403 | 0.8287 | 0.3604 | 0.6916 | 2.5e-5 |
| 13 | 0.2379 | 0.8373 | 0.3154 | 0.7607 | 2.5e-5 |
| 14 | 0.2348 | 0.8331 | 0.3165 | 0.7504 | 2.5e-5 |
| 15 | 0.2345 | 0.8336 | 0.3324 | 0.7278 | 2.5e-5 |
| 16 | 0.2298 | 0.8363 | 0.3308 | 0.7524 | 2.5e-5 |
| 17 | 0.2273 | 0.8361 | 0.3198 | 0.7563 | 2.5e-5 |
| **18** | **0.2214** | **0.8475** | **0.3024** | **0.7633** | **1.25e-5** |
| 19 | 0.2123 | 0.8585 | 0.3208 | 0.7245 | 1.25e-5 |
| 20 | 0.2132 | 0.8551 | 0.3404 | 0.7284 | 1.25e-5 |
| 21 | 0.2090 | 0.8566 | 0.3242 | 0.7506 | 1.25e-5 |
| 22 | 0.2115 | 0.8524 | 0.3547 | 0.7142 | 1.25e-5 |
| 23 | 0.1999 | 0.8622 | 0.3468 | 0.7202 | 6.25e-6 |
| 24 | 0.2030 | 0.8651 | 0.3636 | 0.6952 | 6.25e-6 |
| 25 | 0.1964 | 0.8638 | 0.3246 | 0.7467 | 6.25e-6 |
| 26 | 0.1970 | 0.8684 | 0.3440 | 0.7365 | 6.25e-6 |
| 27 | 0.1968 | 0.8666 | 0.3302 | 0.7391 | 3.12e-6 |
| 28 | 0.1944 | 0.8706 | 0.3403 | 0.7123 | 3.12e-6 |
| 29 | 0.1940 | 0.8709 | 0.3336 | 0.7209 | 3.12e-6 |
| 30 | 0.1891 | 0.8706 | 0.3550 | 0.7192 | 3.12e-6 |

Six ReduceLROnPlateau halvings occurred (1e-4 → 5e-5 → 2.5e-5 → 1.25e-5 → 6.25e-6 → 3.12e-6, with
1.5625e-6 carried into the final state). `[R1]` verified that the learning rate starts at 1e-4 and
changes *only* by those halvings, that all 30 epochs are logged with no NaN or infinity, and that
no epoch was superseded by a re-run.

**Checkpoint selection.** `best/` is the first epoch attaining the maximum logged `val_QWK`, which
`[R1]` confirmed is epoch 18 at 0.7632750272750854 — matching the log exactly. Training loss kept
improving to epoch 30 (0.1891, train QWK 0.8706) while validation QWK peaked at epoch 18 and
declined; the gap between a train QWK of 0.87 and a validation QWK of 0.72–0.76 over the later
epochs is consistent with the model continuing to fit the training split past the point of
validation benefit.

---

## 10. Final BEST model performance

BEST = `checkpoints/best`, epoch 18, evaluated on the 730-image validation population.

| Metric | Value |
|---|---|
| Quadratic weighted kappa (QWK) | **0.763275** |
| QWK (scikit-learn, cross-check) | 0.763275 |
| Loss | 0.302438 |
| Accuracy | 0.738356 |
| Balanced accuracy (= macro recall) | 0.489899 |
| Mean absolute ordinal error | 0.380822 |
| Macro F1 | 0.484207 |
| Weighted F1 | 0.718715 |
| Macro precision | 0.522063 |
| Weighted precision | 0.719528 |
| Weighted recall | 0.738356 |
| QWK bootstrap 95% CI | [0.7230, 0.8023] |

The bootstrap interval is 2,000 resamples of the 730 evaluated images under a fixed RNG seed
(20260911), so it is reproducible.

**Recorded versus recomputed.** The recorded `val_QWK` is 0.7632750272750854 and the recomputed
value is 0.763275; the recorded `val_loss` is 0.30243971943855286 and the recomputed value is
0.30243766. These differences (≈3 × 10⁻⁸ and ≈2 × 10⁻⁶) are numerical precision, not disagreement:
the recorded figures are the float32 Keras metric and per-batch loss aggregation, while the
diagnostics recompute in float64 from the confusion matrix and a sample-weighted sum. Both agree
well inside the pre-set tolerances (<0.01 for QWK, <1e-3 for loss). The recorded values stand as
the experiment's own record; no recorded value was replaced.

The headline figure named in the task brief as "val_loss ≈ 0.302442" is not what the artifacts
hold — the recorded value is 0.30243972 and the recomputed value is 0.30243766, both of which
round to **0.302438**, which is what `[D12]` reports. This is a presentation difference at the
sixth decimal, documented here rather than silently adjusted.

Note that balanced accuracy (0.4899) is far below accuracy (0.7384). The headline QWK and accuracy
are carried substantially by grade 0, which is half the validation population. §11 and §12 are the
honest description of this model's behaviour.

---

## 11. Comprehensive per-class performance

| Grade | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 361 | 364 | 341 | 23 | 20 | 0.9368 | 0.9446 | 0.9407 |
| 1 | 74 | 70 | 32 | 38 | 42 | 0.4571 | 0.4324 | 0.4444 |
| 2 | 198 | 246 | 153 | 93 | 45 | 0.6220 | 0.7727 | 0.6892 |
| 3 | 39 | 39 | 9 | 30 | 30 | 0.2308 | 0.2308 | 0.2308 |
| 4 | 58 | 11 | 4 | 7 | 54 | 0.3636 | **0.0690** | **0.1159** |

Prediction histogram `[364, 70, 246, 39, 11]` against true `[361, 74, 198, 39, 58]`.

Grade 0 is strongly recognised (recall 0.945, F1 0.941). Grade 2 is over-predicted — 246 predicted
against 198 present — and is the dominant predicted class among diseased cases. Grade 4 is by a
wide margin the weakest class: the model emitted only 11 grade-4 predictions for 58 grade-4
images, recovering 4 of them (recall 0.069). Grade 3 is predicted exactly 39 times for 39 true
cases, but only 9 of those land on the right images, so precision and recall coincide at 0.231 by
arithmetic coincidence rather than by calibration.

---

## 12. Confusion matrix and ordinal error analysis

Rows are true grade, columns predicted grade:

| | pred 0 | pred 1 | pred 2 | pred 3 | pred 4 | total |
|---|---|---|---|---|---|---|
| **true 0** | 341 | 6 | 6 | 8 | 0 | 361 |
| **true 1** | 13 | 32 | 28 | 1 | 0 | 74 |
| **true 2** | 6 | 27 | 153 | 10 | 2 | 198 |
| **true 3** | 1 | 0 | 24 | 9 | 5 | 39 |
| **true 4** | 3 | 5 | 35 | 11 | 4 | 58 |
| **total** | 364 | 70 | 246 | 39 | 11 | 730 |

Ordinal error distance |true − predicted|:

| Distance | Count | % of evaluated | % of errors |
|---|---|---|---|
| 0 | 539 | 73.84% | — |
| 1 | 124 | 16.99% | 64.92% |
| 2 | 50 | 6.85% | 26.18% |
| 3 | 14 | 1.92% | 7.33% |
| 4 | 3 | 0.41% | 1.57% |

191 errors in total: **124 adjacent (64.9% of all errors)** and **67 two or more grades away**.
Maximum observed ordinal error distance is **4**. Mean error distance is 0.3808, identical to the
MAE, as `[D4]` asserts.

**Dominant error structure.** The confusion matrix supports the following description:

1. **Grade 0 is strongly recognised.** 341 of 361 correct; its 20 errors are scattered thinly
   (6 → grade 1, 6 → grade 2, 8 → grade 3, none → grade 4).
2. **Grade 2 is the dominant predicted class among diseased cases.** It absorbs 28 of 74 grade-1
   images, 24 of 39 grade-3 images and 35 of 58 grade-4 images, giving it 93 false positives
   against 153 true positives.
3. **Grades 3 and 4 are substantially compressed toward grade 2.** 24/39 (61.5%) of grade-3 cases
   and 35/58 (60.3%) of grade-4 cases are predicted as grade 2. The upper end of the ordinal scale
   is systematically pulled down toward the middle.
4. **Grade 4 is the weakest class.** Beyond the 35 predicted as grade 2, a further 11 are predicted
   as grade 3, 5 as grade 1 and 3 as grade 0 — only 4 of 58 are recovered.
5. **Adjacent errors are the majority of errors** (64.9%), which is what the ordinal formulation is
   designed to encourage, but the 67 non-adjacent errors are concentrated in exactly the
   compression described above.

These are statements about model behaviour as recorded in the confusion matrix. The cause is not
established by this experiment; class support is very uneven (only 39 grade-3 and 58 grade-4
validation images, and 154/236 in training), and no controlled experiment separating imbalance,
representation capacity, or the known augmentation defect was run.

---

## 13. CORN probability and calibration analysis

**Threshold-level calibration** — predicted `P(Y>k)` against the observed rate of `true > k`:

| Threshold k | mean P(Y>k) | empirical rate | gap | n(true > k) |
|---|---|---|---|---|
| 0 | 0.5014 | 0.5055 | −0.0040 | 369 |
| 1 | 0.3896 | 0.4041 | −0.0145 | 295 |
| 2 | 0.1144 | 0.1329 | −0.0185 | 97 |
| 3 | 0.0677 | 0.0795 | −0.0118 | 58 |

Every threshold is slightly **under**-predicted, and the largest absolute gap is at k = 2 — the
same threshold whose decisions separate grade 2 from grades 3 and 4. In aggregate the gaps are
small (≤0.019); their consequence is concentrated where the compression in §12 occurs.

**Multiclass calibration**, from the reconstruction verified in §7:

| Quantity | Value |
|---|---|
| Brier score (multiclass, 0 = perfect) | 0.3420 |
| ECE, CORN decode convention | 0.0524 |
| ECE, standard argmax convention | 0.0520 |

The two ECE conventions differ only where argmax and the decode disagree (38 samples).

**Confidence, margin and entropy.** Confidence is the probability assigned to the decoded grade;
margin is `min_k |P(Y>k) − 0.5|`, i.e. how close the sample is to flipping an ordinal decision.

| Group | n | Confidence (mean / median) | Margin (mean / median) | Entropy, nats (mean) |
|---|---|---|---|---|
| All | 730 | 0.7899 / 0.8338 | 0.3366 / 0.3760 | 0.5356 |
| Correct | 539 | 0.8620 / 0.9729 | 0.3878 / 0.4729 | 0.3919 |
| Incorrect | 191 | 0.5865 / 0.5884 | 0.1919 / 0.1757 | 0.9410 |

Correct predictions are markedly more confident, wider-margin and lower-entropy than incorrect
ones — the model's uncertainty signal is informative. Maximum observed entropy is 1.4305 nats
against 1.6094 for a uniform distribution over five grades.

Confidence by **predicted** grade is revealing: 0.9481 (grade 0), 0.6031 (1), 0.6928 (2), **0.3005
(3)**, 0.6515 (4). A grade-3 prediction carries very low probability mass on grade 3 itself,
consistent with grade 3 sitting in a narrow band between two thresholds. Confidence by **true**
grade declines monotonically with severity: 0.9306, 0.6684, 0.6752, 0.5969, 0.5907.

Margin by error distance — 0.3878 (correct), 0.1909 (d=1), 0.2108 (d=2), 0.1346 (d=3), 0.1836
(d=4) — shows that the large-distance errors are not uniformly low-margin; some distant errors are
made with a comfortable margin, which §14 illustrates.

---

## 14. Visual error analysis

`[D8]` produced 30 deterministic image panels covering correct cases per grade, misclassified cases
per grade, **every one of the 17 observed off-diagonal confusion pairs**, and the largest-error,
highest-confidence-incorrect and lowest-confidence-correct extremes. Selection is by explicit sort
with `image_id` as final tie-break — nothing sampled, nothing hand-picked. Every selected image
rendered from the cached canonical RGB the model actually consumed; none was unavailable or
substituted.

The following are **visual observations of image appearance**, separated from inference. No image
is diagnosed clinically, and no image is claimed to be mislabelled.

**Correctly classified cases.** Grade-0 panels show evenly illuminated fundus images with a clearly
visible disc and macula and no conspicuous bright or dark focal features at this scale; all four
carry confidence 1.000 and margin 0.500. Grade-2 correct cases (confidence ≈0.91) show conspicuous
clustered yellow-white deposits. Grade-3 correct cases are notably *less* confident (0.443–0.556)
despite being correct, and are visually busy — extensive bright deposits together with darker
focal marks, one with punctate marks distributed across the periphery. Grade-4 correct cases
(0.532–0.759) are visually heterogeneous, including large dark regions and acquisition artifacts.

**Misclassified cases.** The grade-0 → grade-3 panel is striking: all four images are hazy,
low-contrast and bluish-grey, and the model over-predicts severity on them. This is consistent
with the model responding to global image quality rather than to focal findings, though this
experiment does not establish that. Conversely, the grade-1 → grade-0 and grade-2 → grade-0 panels
show images that appear comparatively clean and well-exposed at this resolution, where the model
appears to underestimate severity with high confidence.

**Every confusion direction.** All 17 directions were inspected. The two largest —
**grade 4 → grade 2 (35 images)** and **grade 3 → grade 2 (24 images)** — look alike in an
important respect: the images plainly show bright exudate-like deposits and darker focal marks, and
the model does register disease (`P(Y>0)` ≈ 1.00, `P(Y>1)` ≈ 0.89–0.99), but `P(Y>2)` collapses to
roughly 0.06–0.18. The model is confidently deciding "at least grade 2" and then confidently
declining to go further. This is the probabilistic signature of the compression described in §12.
The smaller directions are consistent with this: grade 2 → grade 4 (2 images) shows very extensive
bright deposits, and grade 3 → grade 4 (5 images) shows large pale lesions — cases where the model
pushes past threshold 3 on visually severe-appearing images.

**Largest ordinal errors (distance 4).** All three are true grade 4 predicted grade 0. Two are
visually challenging — one washed-out and low-contrast, one with a pronounced blue-green colour
cast and surface artifact — and the third appears relatively featureless at this scale. These are
the cases where the model failed to cross even the first threshold (`P(Y>0)` = 0.255, 0.320, 0.375).

**Highest-confidence incorrect.** All four are true grade 1 or 2 predicted grade 0 at confidence
0.986–0.999 with near-maximal margins. Visually these are clean, well-illuminated images in which
any findings are subtle at this resolution. They are the most consequential failure mode for a
confidence-gated use of this model: the model is not uncertain here, it is wrong.

**Lowest-confidence correct.** Confidence 0.166–0.244, comprising true grade-3 and grade-1 images
decided correctly but only barely — `P(Y>2)` of 0.53 and 0.64 in the grade-3 cases, i.e. just over
the threshold. Being correct here is fragile.

**Ambiguous / borderline.** The most ambiguous sample in the whole set has margin 0.000635
(`P(Y>2)` = 0.499 for a true grade-4 image predicted grade 2). Nine samples sit within 0.006 of a
threshold. Under `mixed_float16`, decisions this close to a boundary are numerically fragile — see
§19.

---

## 15. BEST vs LAST analysis

BEST = `checkpoints/best`, epoch 18 (delivery model). LAST = `checkpoints/gen_00030`, epoch 30
(trajectory / resume checkpoint). Both evaluated on the identical 730-image population.

| Metric | BEST | LAST | BEST − LAST |
|---|---|---|---|
| loss | 0.302438 | 0.355001 | −0.052563 |
| **QWK** | **0.763275** | 0.719239 | **+0.044036** |
| accuracy | 0.738356 | 0.709589 | +0.028767 |
| balanced accuracy | 0.489899 | 0.509338 | −0.019439 |
| MAE | 0.380822 | 0.436986 | −0.056164 |
| macro F1 | 0.484207 | 0.498639 | −0.014432 |
| weighted F1 | 0.718715 | 0.706191 | +0.012524 |

Per-class recall:

| | grade 0 | grade 1 | grade 2 | grade 3 | grade 4 |
|---|---|---|---|---|---|
| BEST | 0.9446 | 0.4324 | 0.7727 | 0.2308 | 0.0690 |
| LAST | 0.8892 | 0.5405 | 0.6970 | 0.2821 | 0.1379 |

Prediction histograms: BEST `[364, 70, 246, 39, 11]`, LAST `[345, 104, 217, 46, 18]`, true
`[361, 74, 198, 39, 58]`.

**BEST is the delivery model** because `val_QWK` is the pre-declared selection metric and BEST
attains its global maximum. It is also better on loss, accuracy, MAE and weighted F1.

**LAST is not "bad" — it simply did not maximise the selected metric, and it trades differently.**
LAST is meaningfully better on the minority grades: higher recall on grades 1, 3 and 4 (0.1379 vs
0.0690 on grade 4, i.e. 8 recovered instead of 4), a better-spread prediction histogram, higher
balanced accuracy (+0.019) and higher macro F1 (+0.014). It pays for this with 40 fewer correct
grade-0 images and a worse QWK, loss and MAE. In other words, LAST is less severity-compressed but
less accurate overall.

This is a genuine trade-off, and it is worth stating plainly: had the selection metric been
balanced accuracy or macro F1 rather than QWK, epoch 30 would have been preferred over epoch 18.
The selection rule was fixed in advance and was not changed after seeing this; LAST remains
un-promoted and is retained only as the resume trajectory. Its existence is recorded in the
archive's `provenance.json` as a path reference only.

---

## 16. Baseline comparison

The comparison rule was registered in the LR = 1e-4 plan (section G) **before** this run started
and was applied unchanged.

Baseline-0: experiment `2026-09-11_12-18-34`, best `val_QWK` **0.17963171** at epoch 3. Its value
on disk was re-read and asserted equal to the pre-registered figure before any comparison was made.

| Quantity | LR 1e-4 (this run) | baseline-0 | Pre-registered rule |
|---|---|---|---|
| BEST_p8 val_QWK (**primary**) | 0.7210 (epoch 3) | 0.1796 (epoch 3) | ≥ 0.2796 ⇒ materially better |
| BEST_full val_QWK | 0.7633 (epoch 18) | 0.1796 | < 0.2296 ⇒ not materially better |
| BEST val_loss (recomputed) | 0.3024 | prior reference 0.6548 (here 0.6519) | ≤ 0.6048 |
| Distinct grades predicted | 5 | — | ≥ 3 |
| QWK bootstrap 95% CI | [0.7230, 0.8023] | — | lower bound > 0.1796 confirms |
| Epochs with val_QWK < 0.01 | 0 | 7 (epochs 1, 2, 4, 5, 6, 7, 8) | secondary |

**Pre-registered verdict: MATERIALLY BETTER than baseline-0** — confirmed, bootstrap lower bound
0.7230 > 0.1796.

The primary quantity is `BEST_p8` (0.7210), the best `val_QWK` a patience-8 EarlyStopping would
have reached; it is reported because baseline-0 ran with patience 8, so this keeps the comparison
like-for-like rather than crediting the longer schedule. A patience-8 run would have stopped at
epoch 11. `BEST_full` (0.7633) is the value actually attained under this run's patience-12
schedule. Both clear the threshold by a wide margin, and the two thresholds (0.2796 and 0.2296)
were never approached from either side, so the verdict does not hinge on which is used.

**What this establishes:** under the predefined comparison, on this project's own APTOS 2019
validation split, this run materially outperformed baseline-0. **What it does not establish:**
anything about state-of-the-art performance, performance relative to any external method, or
performance on any data outside this split. Baseline-0 was read only; `[R4]` and `[D13]` both
verified it is byte-for-byte unchanged.

---

## 17. Artifact, checkpoint and reproducibility verification

**80 automated checks passed in total** — 18 technical-validity checks in `[R1]`–`[R4]` and 62
diagnostic checks in `[D2]`–`[D13]`. No check failed.

Checkpoint and configuration integrity (`[R1]`, `[R4]`, `[D10]`):

- checkpoint integrity: no failure; `gen_00029` and `gen_00030` valid, `latest.json` → `gen_00030`,
  `best/` valid from `gen_00018`;
- config hash `3f549e1638d9409f7862f1e799bd7052` identical in `metadata.json`, `gen_00029`,
  `gen_00030` and `best`;
- `metadata.json` records the production configuration exactly;
- epochs logged are 1..30 = LAST's completed epochs, no NaN/inf, no superseded rows;
- learning rate starts at 1e-4 and changes only by ReduceLROnPlateau halvings;
- optimizer iterations 43,810 of 43,830 (20 loss-scale skips, 0.05%);
- `best/` is the first epoch attaining the maximum logged `val_QWK`;
- EarlyStopping stopped where a patience-12 replay of the log stops;
- BEST weights validate (READY marker, manifest, sizes, SHA-256) and match their manifest SHA-256;
- trainable parameter count is the production 43,338,506; output shape is (batch, 4) CORN logits;
  all five grades are reachable through the production decoder; inference output is finite;
- optimizer type, precision policy and git commit metadata preserved in the checkpoint.

**Final model archive.** `[D11]` archived the delivery model only after `[D10]` passed all 48 of
its checks; it built the archive locally, checksummed it, copied it, re-verified every checksum at
the destination, and only then wrote the `ARCHIVE_READY` marker. It refuses an existing
destination and never overwrites or deletes.

`exported_models/FinalClassification/2026-09-12_02-45-05_BEST`:

| File | Bytes |
|---|---|
| `model.weights.h5` | 174,441,224 |
| `aptos2019_train_val_split.csv` | 75,460 |
| `provenance.json` | 3,865 |
| `evaluation_manifest.json` | 1,956 |
| `checkpoint_state.json` | 1,582 |
| `checkpoint_manifest.json` | 1,009 |
| `ARCHIVE_READY` | 867 |
| `experiment_metadata.json` | 459 |

**Independent reload verification (`[D13]`).** The production architecture was rebuilt from
scratch and **only the archived weights** were loaded into it — proving the archive alone
reproduces the delivery model:

- archived weights reproduce the verified BEST val_QWK: 0.76327503 vs 0.76327500 (|diff| < 1e-6);
- archived weights reproduce the verified BEST val_loss: 0.30243766 vs 0.30243766 (|diff| < 1e-4);
- the archived model reproduces the **per-sample predictions** exactly;
- archived weights SHA-256 = `7b780c…7b8b9a`, matching the verified BEST and the READY marker;
- every required provenance file present; no unexpected or temporary file in the archive;
- the original experiment `checkpoints/` is **byte-for-byte unchanged** (fingerprinted before any
  diagnostic ran, re-checked after archival);
- **baseline-0 unchanged**; the authoritative split unchanged; the cache inventory unchanged since
  `[6]` (3,651 fully local, 11 missing).

The original BEST checkpoint is preserved in place; the archive is a verified copy, not a move.
The archive is to be treated as **immutable**.

**Preserved notebook state.** The notebook is committed in the state in which it was executed, with
its cell outputs and the 30 `[D8]` image panels intact, so the evidence behind this report is
inspectable without re-running anything. Consequently the `[S]` session-configuration cell is
committed as it was run — `RUN_POST_RUN_EVALUATION = True` and `POST_RUN_EXPERIMENT_DIR` pointing
at this experiment — rather than reset to the inert defaults. `RUN_TRAINING` remains `False`, and
every `[D2]`–`[D13]` cell is read-only apart from `[D11]`, which refuses to write over the archive
that now exists. **Any later session must reset `[S]` for its own role before running the
notebook.** Execution counts 1–27 show `[P2]` and `[T]` were never executed in this session, which
is correct for a post-run evaluation; `[P3]`, the end-of-session safety check, was also not
executed — its checks are a subset of what `[D13]` verified and re-verified.

---

## 18. Main findings

1. Training completed under its own stopping rule: EarlyStopping halted the run at epoch 30 of a
   50-epoch cap, and an independent replay of the logged metric confirms that stopping point.
2. The BEST checkpoint is epoch 18, with **validation QWK 0.763275** (bootstrap 95% CI
   [0.7230, 0.8023]), accuracy 0.7384, loss 0.3024, over 730 evaluated validation images.
3. Under the pre-registered rule, the run is **materially better than baseline-0** (0.1796). The
   conclusion is robust: the bootstrap lower bound exceeds the baseline by a wide margin, and both
   the patience-matched (`BEST_p8` = 0.7210) and full (`BEST_full` = 0.7633) variants clear their
   thresholds.
4. Aggregate performance is carried substantially by grade 0 (half the validation population,
   recall 0.945). Balanced accuracy is 0.4899 and macro F1 is 0.4842 — far below the headline
   accuracy of 0.7384.
5. The dominant error structure is **severity compression toward grade 2**: 61.5% of grade-3 and
   60.3% of grade-4 images are predicted as grade 2, and grade 4 attains recall 0.069 (4 of 58).
   Adjacent errors are 64.9% of all errors, but 67 errors are two or more grades away and the
   maximum ordinal distance is 4.
6. The CORN formulation behaves correctly and verifiably: `P(Y>k)` is monotone on every sample,
   the finite-difference reconstruction is a proper distribution, and the production decode is
   reproduced exactly from the cumulative probabilities. Threshold calibration is close (gaps
   ≤0.019, all slightly under-predicting), with the largest gap at the k = 2 threshold that governs
   the observed compression.
7. The model's confidence is informative — correct predictions average 0.862 confidence against
   0.587 for incorrect ones — but a small set of high-confidence errors exists (true grade 1–2
   predicted grade 0 at ≥0.986).
8. BEST and LAST trade off differently: LAST (epoch 30) is better on balanced accuracy, macro F1
   and minority-grade recall, while BEST is better on the selected metric QWK, and on loss,
   accuracy and MAE. Selection followed the pre-declared rule.
9. All 80 verification checks passed; the archived model independently reproduces the BEST metrics
   and per-sample predictions from its weights alone; the experiment, baseline-0, the split and the
   cache are all verified unchanged.

**Scope of these findings.** All performance figures are **validation** performance on a single
project-defined split. Because the delivery checkpoint was *selected* on this same split by
maximising `val_QWK`, these figures are **model-selection performance** and are optimistically
biased for it. They are **not** an unbiased estimate of **generalization** performance, and they
are **not** test performance. The contribution of **RACAF** specifically is not measured here: the
statement supported by the evidence is that the final joint model *incorporating* RACAF achieved
the reported validation performance. No clinical effectiveness of any kind is claimed or supported.

---

## 19. Limitations

1. **One project-defined APTOS 2019 validation split.** All results come from a single split of a
   single public dataset, generated by this project's own manifest.
2. **No external or held-out test evidence.** Nothing here speaks to performance on other cameras,
   sites, populations or acquisition conditions.
3. **Validation-based model selection.** The delivery checkpoint was chosen by maximising `val_QWK`
   on the same 730 images the metrics are reported on, so the reported figures are optimistic.
4. **Class imbalance.** Training support is 1,444 / 296 / 799 / 154 / 236 across grades 0–4; the
   validation population is half grade 0.
5. **Small support for the rarer grades.** Only **39 grade-3** and **58 grade-4** validation images
   were evaluated. Per-class metrics for these grades rest on few samples — grade 4's recall of
   0.069 is 4 images out of 58, and a handful of decisions would move it appreciably.
6. **Three empty-FOV exclusions.** `262ad704319c`, `26453eb7e989`, `3a122851e526` are excluded
   because Stage 03 finds no fundus disk. Metrics describe the remaining 730 images; the model's
   behaviour on such images is untested rather than good.
7. **`mixed_float16` numerical boundary sensitivity.** Validation ran in mixed precision; logits
   within roughly 0.1% of a decode boundary can flip. The most ambiguous sample sits 0.000635 from
   a threshold, and one borderline flip moves QWK by ≈0.003 on 730 images — which is why the
   recorded-vs-recomputed tolerance is 0.01 rather than exact equality.
8. **Known augmentation defect.** As documented in `JOINT_TRAINING_ARCHITECTURE.md` §49, `gen()`
   re-creates `rng = np.random.default_rng(seed)` on every dataset iteration, so every image
   receives an identical augmentation in every epoch. This defect pre-dates this experiment, was
   present during it, and was deliberately not corrected here (correcting it would have changed the
   frozen configuration).
9. **No RACAF ablation in this experiment.** No component-level contribution is isolated.
10. **No clinical claim.** Nothing in this report supports any statement about clinical utility,
    safety or effectiveness.

---

## 20. Conclusions

The joint Stage 05–08 + RACAF CORN classifier trained to completion under a verified, reproducible
configuration and produced a delivery model at epoch 18 with a validation QWK of 0.763275 on 730
APTOS 2019 validation images. Under a comparison rule fixed before the run, it is materially better
than baseline-0 (0.1796), and the margin is large enough that the conclusion does not depend on the
choice between the patience-matched and full variants of the metric.

The model's behaviour is well characterised rather than merely scored. It recognises grade 0
reliably, concentrates its diseased predictions on grade 2, and systematically compresses grades 3
and 4 toward grade 2 — leaving grade 4 with a recall of 0.069. Most errors are adjacent, but 67 of
191 are two or more grades away. Its probability outputs are internally valid and reasonably
calibrated at the threshold level, and its confidence separates correct from incorrect predictions,
with a small but real set of confident errors.

These are validation results on a split that also drove model selection. They establish that this
configuration works and that it substantially outperforms baseline-0 under the registered rule.
They do not establish generalization, they do not isolate RACAF's contribution, and they support no
clinical claim. The archived model at
`exported_models/FinalClassification/2026-09-12_02-45-05_BEST` is verified, reproducible from its
weights alone, and immutable.

---

## 21. Future experiments

Kept deliberately separate from the frozen result above. None of these was performed.

1. **Independent repeat with a different random seed** — to separate the learning-rate effect from
   run-to-run variation, and to put an uncertainty band on the epoch-18 result.
2. **Class-imbalance investigation** — the compression of grades 3 and 4 toward grade 2 is the
   clearest deficiency; loss weighting, resampling or threshold adjustment are the obvious
   candidates, evaluated against the same registered rule.
3. **Correction of the known augmentation defect** (`JOINT_TRAINING_ARCHITECTURE.md` §49) before
   further tuning, so that augmentation actually varies across epochs.
4. **Controlled RACAF ablation** — with/without RACAF and with reliability ablated, holding
   everything else fixed, to measure the component's contribution rather than assume it.
5. **Genuinely held-out or external test evaluation** — required before any performance claim that
   goes beyond this validation split.

A repeat run would also allow the BEST/LAST trade-off in §15 to be examined properly: whether
epoch 30's better minority-grade recall is a reproducible property of the later trajectory or an
artifact of this single run.
