# RACAF Ablation / NO-RACAF Control — Experiment 1

> **STATUS: COMPLETE.** Pre-registered 2026-09-13; run `2026-09-13_04-11-07` finished (50-epoch
> cap) and passed every post-run check (`[R1]`–`[R4]` all PASS, `[D2]`–`[D13]` 62/62 PASS).
> Sections 1–8 and 10–12 are the pre-registration and are kept as written; §9, §10's outcome and
> §14 are filled **only** from the executed notebook's saved outputs. Full post-run analysis:
> [RACAF_vs_NO_RACAF_Analysis.md](RACAF_vs_NO_RACAF_Analysis.md).
>
> **Headline:** under the controlled conditions, removing RACAF changed BEST val_QWK from
> **0.7632750** to **0.8461052** (pre-registered ΔQWK = with − without = **−0.0828302**). The
> control did better on the primary metric — from **one run per arm**, with worse validation loss
> and calibration. A subsequent exact per-image audit
> ([RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md)) confirmed the
> 41 train/validation duplicates do not explain it (§12 item 9).

> **This run is a RACAF ablation / NO-RACAF control. It is NOT the delivery model.**
> The delivery model remains the frozen reference
> `exported_models/FinalClassification/2026-09-12_02-45-05_BEST`.

---

## 1. Purpose

Measure the contribution of RACAF to the joint Stage 05–08 classifier by running the finalized
experiment again with RACAF removed and **nothing else changed**, then comparing the two under
identical model-selection rules.

The question this answers: *how much does RACAF contribute to the final joint model?* The
[final experiment report](2026-09-12_02-45-05_Final_Experiment_Report.md) explicitly could not
answer it — §8 of that report states that no RACAF ablation existed, so no component-level claim
was made. This experiment is the controlled test that was listed there as future work item 4.

**No outcome is assumed.** RACAF may help, do nothing, or hurt. All three are valid results and
will be reported as found. QWK is the pre-registered primary metric, but secondary metrics —
especially minority-grade recall, where the reference model is weakest — will be reported
whether or not they favour RACAF. No metric will be selected after the fact.

## 2. Reference experiment (frozen, read-only)

| Item | Value |
|---|---|
| Experiment ID | `2026-09-12_02-45-05` |
| Experiment directory | `experiments/FinalClassification/2026-09-12_02-45-05` |
| Archived BEST model | `exported_models/FinalClassification/2026-09-12_02-45-05_BEST` |
| Model | `joint_stage05_08_racaf` |
| Config hash | `3f549e1638d9409f7862f1e799bd7052` |
| Split SHA-256 | `f512a7a086ac7f53e1a5ef8b49a703ae0c5e70db37891253640c91306d965837` |
| BEST epoch | 18 |
| **BEST val_QWK (recorded)** | **0.7632750272750854** |
| **BEST val_loss (recorded)** | **0.30243971943855286** |
| LAST epoch | 30 (EarlyStopping) |
| BEST weights SHA-256 | `7b780c230e4ce83ae9e9b1c0bdb9ae73c707b262e5a9c759748343f9417b8b9a` |

The reference experiment, its checkpoints and its archive are **immutable**. The ablation notebook
refuses to train into, evaluate or archive over either path — see §11.

## 3. Ablation hypothesis

RACAF fuses Stage 07's cross-attention embedding `E` with a reliability-gated Global readout:

```
gate  = sigmoid(w_g · r + b_g)
G_hat = W_r · GAP(G) + b_r
F     = gate · E + (1 − gate) · G_hat
```

If that reliability-gated blend contributes, removing it should degrade validation performance.
If it does not, performance should be unchanged within run-to-run variation. If the gate is
harmful — for example by diluting `E` on images whose reliability estimate is poorly calibrated —
removing it may improve performance. The experiment distinguishes these three cases only to the
extent that a **single** run can, which is limited (§12).

## 4. Exact controlled difference

```
reference (frozen)  Stage 07 → E ──( RACAF: gate·E + (1−gate)·G_hat )──→ F ──→ CORN → (B, 4)
ablation  (this)    Stage 07 → E ───────────────────────────────────────────→ CORN → (B, 4)
```

One line of graph construction is removed. Nothing replaces it: no substitute fusion, no extra
attention, no compensating parameter.

Two consequences follow **by definition of the module being ablated**, and are not additional
changes:

1. RACAF was the **only** consumer of the per-image reliability scalar `r`, so the model no longer
   sees reliability at all. The `reliability` input remains *declared* so the dataset pipeline is
   byte-for-byte unchanged.

   Because Keras' Functional API rejects a declared Input that does not reach the output
   (`` `inputs` not connected to `outputs` ``), reliability is connected through a single
   **parameter-free** layer, `InertReliabilityConnection`, which adds zeros derived from the
   reliability tensor: `logits + cast(zeros_like(r[:, :1]))`. The graph edge is real, so the model
   is constructible, but the added value is exact zero — `x + 0.0` is exact in IEEE 754 — so the
   logits are bit-for-bit unchanged and the prediction cannot depend on reliability. Verified on
   the real model: `r = 0.0` and `r = 1.0` give **bit-identical logits, max |diff| = 0.0**, with
   **0 trainable parameters and 0 trainable tensors** added, so the delta against the reference
   remains exactly RACAF.
2. RACAF's `GAP(G) → Dense` readout was the **only** second, direct Stage 06 → classifier path.
   Global information now reaches CORN solely through Stage 07's cross-attention.

The reliability *computation and cache* are untouched: Stage 04 TTA, `racaf.compute_reliability`,
`LOCAL_RACAF_CACHE_DIR` and `config.RACAF_RESULTS_DIR` all still run and are still read by the
dataset. Only RACAF's **trainable fusion module** is removed from the model graph.

No repository `.py` file was modified. The ablation notebook composes the same unmodified
`build_*()` functions in the same order as `joint_training_model.build_joint_model()`, minus
RACAF, and compiles with the unmodified `jtm.compile_joint_model()`.

## 5. Architecture comparison

| Component | reference | ablation |
|---|---|---|
| Stage 05 local features + AdaptiveBranchFusion | present, trainable | **unchanged** |
| Stage 06 dual-scale Swin | present, trainable | **unchanged** |
| Stage 07 adaptive cross-attention | present, trainable | **unchanged** |
| RACAF fusion | present, trainable | **REMOVED** |
| CORN head | present, trainable | **unchanged** |
| Stage 03 / 04 | frozen, outside the graph | **unchanged** |
| Inputs | `stage5 (512,512,8)`, `stage6 (256,256,3)`, `reliability (1,)` | identical (reliability inert) |
| Output | `(None, 4)` CORN logits, float16 under `mixed_float16` | identical |

## 6. Parameter comparison (measured, not estimated)

| | trainable params | tensors | `count_params()` | non-trainable |
|---|---|---|---|---|
| `joint_stage05_08_racaf` | 43,338,506 | 409 | 43,342,346 | 3,840 |
| `joint_stage05_08_no_racaf` | **43,043,336** | **405** | 43,047,176 | 3,840 |
| difference | **295,170** | **4** | 295,170 | 0 |
| RACAF alone (measured) | 295,170 | 4 | — | — |

Per-layer, every surviving component is identical to the parameter: Stage 05 2,170,848 ·
Stage 06 39,697,956 · Stage 07 1,173,504 · CORN 1,028.

Verified by building both graphs in **isolated processes** (so Keras's global layer-name counter
starts fresh in each) and diffing their trainable-variable inventories: exactly **4** variables
exist only in the reference — `reliability_gate/kernel (1,1)`, `reliability_gate/bias (1,)`,
`global_projection/kernel (1152,256)`, `global_projection/bias (256,)` — **0** exist only in the
ablation, and 387 are shared. The only missing layer is `racaf_fusion`. The difference is
explainable solely by RACAF.

## 7. Dataset, split and seed parity

Identical to the reference in every respect: APTOS 2019, the committed manifest
`dataset_splits/aptos2019_train_val_split.csv`, 2,929 train / 733 validation, the same cache, the
same preprocessing, the same augmentation setting, validation unshuffled and unaugmented.

**Seed parity — verified, not assumed.** The reference run's seeds were determined by inspecting
the repository and the frozen experiment's own metadata rather than by assumption:

| Seed | Reference | Ablation |
|---|---|---|
| `downstream_split.DEFAULT_SEED` | 42 | 42 |
| `joint_training_dataset.DEFAULT_SEED` | 42 | 42 |
| validation fraction | 0.2 | 0.2 |
| dataset shuffle seed | 42 (same constant) | 42 |
| augmentation RNG seed | 42 (same constant) | 42 |
| **global weight-init seed** | **none set** | **none set** |

`[7]` asserts all of these and **raises loudly** if any differs, before any model is built. `[5]`
independently pins the resulting partition by SHA-256, which is stronger evidence than seed
equality alone: it proves the identical partition, not merely the same seed value.

**The reference run did not seed weight initialisation.** `tf.keras.utils.set_random_seed()` is
called nowhere in the reference notebook's production path or in `training/` — only in `tests/`
and in the notebook's `[D1]` overfit diagnostic (seed 1234), which never trains a production
model. The ablation therefore **deliberately does not introduce one**: adding a global seed would
be a *second* difference from the reference, not a control. The consequence is recorded honestly
as a limitation in §12 rather than hidden.

## 8. Training configuration

Every value is the reference's, unchanged. Nothing is retuned — this is a controlled ablation,
not an optimization experiment.

| Setting | Value |
|---|---|
| Batch size | 2 |
| Epoch cap | 50 (EarlyStopping decides the end) |
| Optimizer | Adam, wrapped in `LossScaleOptimizer` |
| Learning rate | 1e-4 |
| Precision | `mixed_float16` |
| Loss / decoder | CORN (`joint_corn_loss` / `corn.decode_logits`) |
| Monitor | `val_QWK`, mode `max` |
| EarlyStopping | patience 12, `restore_best_weights=True` |
| ReduceLROnPlateau | patience 4, factor 0.5, min_lr 1e-6 |

**Configuration identity.** The config mapping differs from the reference in **exactly one of
thirteen keys** — `model` — because the model identifier is part of the hash:

| | config hash |
|---|---|
| reference | `3f549e1638d9409f7862f1e799bd7052` |
| **ablation** | **`adf02bb3f8aa6ab93843036bad6b339e`** |

Both were computed with the repository's own `training.checkpointing.config_hash`, and the
reference mapping was **self-validated** by reproducing the frozen hash exactly before the
ablation hash was trusted. Because the hashes differ, neither experiment can resume or be
mistaken for the other.

## 9. Results

Every figure below is read from the executed notebook's saved outputs (cell in brackets). Nothing
is taken from the reference or from expectation.

### 9.1 Run identity and integrity

| Item | Value |
|---|---|
| Ablation experiment ID | `2026-09-13_04-11-07` |
| Experiment directory | `experiments/FinalClassification/2026-09-13_04-11-07` |
| Archive | `exported_models/FinalClassification/2026-09-13_04-11-07_NO_RACAF_BEST` (`[D11]`) |
| Archive SHA-256 | `61619656c0d66995f3e6668cdb30d21b080513faad4354f4913ad98fe5d2e773` |
| Git commit recorded in checkpoints | `f9dbc52647ffddd0aecdf92aacd9330b2d967623` |
| Config hash (metadata + every checkpoint) | `adf02bb3f8aa6ab93843036bad6b339e` (`[R1]`) |
| Runtime | Python 3.13.15, TensorFlow 2.20.0, Tesla T4, CUDA 12.5.1, `mixed_float16` (`[2]`) |
| Epochs completed / stop reason | **50 / the 50-epoch cap** (EarlyStopping wait 9/12 at the end) |
| Optimizer iterations | 73,020 of 73,050 (30 steps skipped by the loss scaler) |
| LR schedule | 1e-4 → 5e-5 (ep 8) → 2.5e-5 (ep 13–36) → 1.25e-5 (37) → 6.25e-6 (41) → 3.12e-6 (46) → 1.56e-6 (50) |
| Pre-run audit `[7]` | 26/26 PASS — seed parity, 43,043,336 params / 405 tensors, no RACAF layer or variable, reliability inert (max \|diff\| = 0.0) |
| Technical validity | `[R1]`–`[R4]` all PASS; `[D2]`–`[D13]` **62/62** PASS |
| Reconstruction + exact-prediction reproduction | **PASS** — archived weights reproduce BEST val_QWK (0.84610522 vs 0.84610524), val_loss (0.62148319) and every per-sample prediction (`[D13]`) |

### 9.2 Headline metrics

| Item | Value |
|---|---|
| **BEST epoch** | **41** (`best/` from `gen_00041`) |
| **BEST val_QWK** | **0.8461052179336548** (recomputed 0.846105; scikit-learn 0.846105) |
| **BEST val_loss** | **0.6215262413024902** recorded; 0.621483 recomputed |
| QWK bootstrap 95% CI | [0.8175, 0.8732] (`[R4]`, 2000 resamples) |
| LAST epoch / val_QWK / val_loss | 50 / 0.8318959474563599 / 0.7943733334541321 |
| Accuracy / balanced accuracy / MAE | 0.763014 / 0.603883 / 0.310959 |
| Macro / weighted F1 | 0.585919 / 0.763182 |
| Macro / weighted precision | 0.600733 / 0.775544 |
| Brier / ECE (decode) / ECE (argmax) | 0.402815 / 0.165210 / 0.164859 |
| Evaluated / excluded | 730 / 3 (`262ad704319c`, `26453eb7e989`, `3a122851e526`, empty field of view) |
| Ground truth / predicted histogram | [361, 74, 198, 39, 58] / [354, 93, 193, 63, 27] — all five grades predicted |

### 9.3 Per-class (BEST)

| Grade | Support | Predicted | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0 | 361 | 354 | 341 | 13 | 20 | 0.9633 | 0.9446 | 0.9538 |
| 1 | 74 | 93 | 45 | 48 | 29 | 0.4839 | 0.6081 | 0.5389 |
| 2 | 198 | 193 | 136 | 57 | 62 | 0.7047 | 0.6869 | 0.6957 |
| 3 | 39 | 63 | 21 | 42 | 18 | 0.3333 | 0.5385 | 0.4118 |
| 4 | 58 | 27 | 14 | 13 | 44 | 0.5185 | 0.2414 | 0.3294 |

### 9.4 Confusion matrix and ordinal errors (BEST)

| | pred 0 | pred 1 | pred 2 | pred 3 | pred 4 |
|---|---|---|---|---|---|
| **true 0** | 341 | 15 | 3 | 2 | 0 |
| **true 1** | 9 | 45 | 19 | 1 | 0 |
| **true 2** | 4 | 27 | 136 | 22 | 9 |
| **true 3** | 0 | 1 | 13 | 21 | 4 |
| **true 4** | 0 | 5 | 22 | 17 | 14 |

Error distance 0/1/2/3/4: **557 / 126 / 40 / 7 / 0**. 173 errors: 126 adjacent (72.8%), 47 two or
more grades away; maximum distance 3.

### 9.5 Training history (selected epochs, `[R1]`)

| Epoch | train loss | train QWK | val_loss | val_QWK |
|---|---|---|---|---|
| 1 | 0.6032 | 0.4699 | 0.4550 | 0.5780 |
| 8 | 0.2589 | 0.7934 | 0.3571 | 0.7649 |
| 13 | 0.2125 | 0.8552 | **0.3277** (minimum) | 0.7829 |
| 18 | 0.1819 | 0.8767 | 0.3582 | 0.7901 |
| 28 | 0.1210 | 0.9261 | 0.4161 | 0.8340 |
| 30 | 0.0942 | 0.9507 | 0.4576 | 0.8337 |
| **41** | 0.0251 | 0.9928 | 0.6215 | **0.8461** (BEST) |
| 50 | 0.0118 | 0.9968 | 0.7944 | 0.8319 (LAST) |

The complete 50-epoch log is in `[R1]`. val_QWK kept rising while val_loss rose from its epoch-13
minimum — the model became increasingly overconfident on the errors it still made (mean confidence
0.9258; 0.8341 on incorrect predictions).

### 9.6 Execution deviations from the pre-registration

1. **`[P2]` launch-gate constants were corrected during the run.** The committed notebook's `[P2]`
   still checked the reference's `409` tensors / `43,338,506` parameters and would have refused to
   launch; the executed notebook checks `405` / `43,043,336`. Gate constants only — no model, data
   or training behaviour changed. This was an omission in the committed ablation notebook.
2. **Session-local weight restore.** The final training session printed "Restoring model weights
   from the end of the best epoch: 43" (Keras EarlyStopping's in-session best, not the global best,
   epoch 41). It affected only the in-memory model after `fit()`; all evaluation reloaded
   `checkpoints/best` from disk with SHA-256 verification (`[R2]`), and `[D13]` reproduced every
   prediction from the archive. **No result is affected.**
3. **Training length differed from the reference** (50 epochs vs the reference's 30, both under the
   identical callback configuration). The advantage does not depend on the extra epochs: the control
   exceeded the reference's best at epoch 8 and reached 0.8340 by epoch 28.
4. **`[R4]`'s printed verdict ("INCONCLUSIVE") is not the Experiment 1 comparison.** `[R4]` applies the
   inherited learning-rate-vs-baseline-0 rule; it reads INCONCLUSIVE because BEST val_loss 0.6215
   exceeds that rule's 0.6048 loss threshold. The pre-registered RACAF comparison is §10.

## 10. Pre-registered comparison

Fixed **before** the ablation is run, so the comparison cannot be chosen after seeing the result.

- **Primary:** `ΔQWK = QWK_with_RACAF − QWK_without_RACAF`, both the **BEST** checkpoint of each
  run, selected by the same rule (global maximum `val_QWK`), on the **identical** evaluated
  population (733 manifest entries, 3 known empty-FOV exclusions, 730 evaluated).
  `QWK_with_RACAF = 0.7632750272750854`.
- **Never** BEST of one run against LAST of the other.
- **Secondary, all reported regardless of direction:** val_loss, accuracy, balanced accuracy, MAE,
  macro/weighted F1, per-class recall (**grades 3 and 4 specifically** — the reference's weakest,
  at 0.2308 and 0.0690), adjacent vs non-adjacent error counts, prediction histogram, Brier, ECE.
- **Interpretation wording:** "under the controlled experimental conditions, removing RACAF
  changed \<metric\> from X to Y." No causal claim beyond this ablation; no clinical claim; no
  generalization claim.

### 10.1 Outcome against the pre-registered comparison

The rule above is unchanged. Both values are BEST checkpoints selected by global maximum `val_QWK`
on the identical 730-image population.

**Primary:** `ΔQWK = QWK_with_RACAF − QWK_without_RACAF = 0.7632750272750854 − 0.8461052179336548 =
−0.0828302` (the model without RACAF is 10.85% higher). Recorded bootstrap 95% CIs do not overlap:
with RACAF [0.7230, 0.8023], without RACAF [0.8175, 0.8732].

**Secondary** (with RACAF → without RACAF; all reported regardless of direction):

| Metric | With RACAF | Without RACAF | Favours |
|---|---|---|---|
| val_loss (recorded) | 0.3024397 | 0.6215262 | with RACAF |
| Accuracy | 0.738356 | 0.763014 | without |
| Balanced accuracy | 0.489899 | 0.603883 | without |
| MAE | 0.380822 | 0.310959 | without |
| Macro F1 | 0.484207 | 0.585919 | without |
| Weighted F1 | 0.718715 | 0.763182 | without |
| **Grade 3 recall** | 0.2308 | **0.5385** | without |
| **Grade 4 recall** | 0.0690 | **0.2414** | without |
| Grade 1 / 2 recall | 0.4324 / 0.7727 | 0.6081 / 0.6869 | without / with |
| Adjacent / non-adjacent errors | 124 / 67 | 126 / 47 | without (non-adjacent) |
| Prediction histogram | [364, 70, 246, 39, 11] | [354, 93, 193, 63, 27] | — (true [361, 74, 198, 39, 58]) |
| Brier | 0.342009 | 0.402815 | with RACAF |
| ECE (decode) | 0.052427 | 0.165210 | with RACAF |

**In the pre-registered wording:** under the controlled experimental conditions, removing RACAF
changed BEST val_QWK from 0.7633 to 0.8461, balanced accuracy from 0.4899 to 0.6039, grade-4 recall
from 0.0690 to 0.2414, validation loss from 0.3024 to 0.6215, and ECE from 0.052 to 0.165.

## 11. Frozen-reference protection

| Guard | Where | Refuses |
|---|---|---|
| `is_baseline_0()` | `[3]` | baseline-0, as before |
| `is_frozen_reference()` | `[3]` | the frozen experiment **and** its BEST archive, as written and fully resolved |
| training guard | `[T]` | training into the frozen reference |
| evaluation guard | `[R1]` | evaluating the frozen reference |
| archival guard | `[D11]` | archiving into the frozen artifact; also refuses any pre-existing destination |
| config hash | `[P2]` | resuming the reference (hashes differ) |

The ablation writes only to its **own** new timestamped experiment directory and its **own**
archive suffixed `_NO_RACAF_BEST`.

## 12. Limitations

1. **Single run, unseeded initialisation.** The reference did not seed weight initialisation, so
   the two runs start from different random weights. A single ablation run therefore **cannot
   separate RACAF's effect from run-to-run initialisation variance.** A ΔQWK smaller than
   that variance is not evidence of anything. Quantifying it needs repeated runs, which this
   experiment does not perform.
2. **One validation split, which also drives model selection.** As in the reference, reported
   figures are model-selection performance on a single APTOS 2019 split — not an unbiased estimate
   of generalization, and not test performance.
3. **Removing RACAF removes two things at once** — the reliability gate and the direct Stage 06
   readout — because they are one module. This experiment cannot attribute any observed change to
   one rather than the other. Separating them would need a further ablation.
4. **Three empty-FOV exclusions** (`262ad704319c`, `26453eb7e989`, `3a122851e526`), exactly as in
   the reference.
5. **Small support for the rarer grades** — 39 grade-3 and 58 grade-4 validation images — so
   per-class deltas for those grades will be noisy.
6. **The known augmentation defect** (`JOINT_TRAINING_ARCHITECTURE.md` §49) is present in both
   runs. It is held constant, not fixed, so the comparison stays controlled.
7. **`mixed_float16` boundary sensitivity**, identical in both runs.
8. **No clinical claim** is supported by this experiment.

*Added after the run — findings that bound how the result can be read (the pre-registered items
above are unchanged):*

9. **Train/validation duplicate contamination, identical in both runs — since resolved exactly.** 41
   of the 730 evaluated validation images (5.62%; true grades 0/1/2/3/4 = 5/6/21/1/8) are
   byte-identical copies of training images, 9 with a conflicting training label. A subsequent
   read-only audit of both experiments' per-image prediction CSVs
   ([RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md)) computed the
   exact duplicate-excluded comparison: ΔQWK moves from +0.0828302 (n=730) to +0.0828904 (n=689,
   paired bootstrap 95% CI [+0.0453, +0.1232]) — a shift of +0.00006. **The control's advantage does
   not depend on, and is not inflated by, the duplicates.**
10. **Limitation 1 is binding for this result.** The two runs followed visibly different trajectories
    (30 vs 50 epochs; LR held at 2.5e-5 for 24 epochs in the control vs 6 in the reference). The gap is
    large, but its share due to initialisation variance is unknown.
11. **The control overfit heavily** (training loss 0.0118, val_loss tripled from its minimum). Its
    better QWK coexists with markedly worse probability calibration.
12. **A paired test has since been computed from the saved per-image CSVs.** The exact paired
    bootstrap of ΔQWK (5000 resamples, seed 20260913, same resampled indices for both models) gives a
    95% interval of **[+0.0460, +0.1202]** in the control's favour, P(Δ ≤ 0) = 0.0000 — narrower than
    the earlier conservative unpaired approximation ([+0.0355, +0.1331]), as expected once the
    correlation between the two models' per-image correctness is accounted for. It still covers
    evaluation-set sampling only, not run-to-run variance. An exact McNemar test on correctness alone
    (not QWK) gives p = 0.1203 — not significant; the QWK advantage is driven more by the size of
    ordinal errors than by a significant swing in exact-grade correctness. Full detail in
    [RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md).

## 13. How to run (runbook)

*Executed as written for run `2026-09-13_04-11-07`, with the `[P2]` constant correction in §9.6.
Kept for reproducibility.*

Requires a Colab **T4 GPU** runtime with Google Drive mounted and the 28.53 GiB cache archive —
roughly one epoch per ~1.67 h, and the reference needed 30 epochs across multiple sessions.

1. **Training, first session** — fresh runtime. In `[S]`: `RUN_TRAINING = True`,
   `RESUME_EXPERIMENT_DIR = None`. Run `[S]`→`[8]`, then `[9] [P2] [T] [P3]`.
   `[7]` prints the seed-parity block and the RACAF parameter audit and refuses to continue if
   either fails; `[P2]` is the launch gate; `[P3]` prints the exact resume line.
2. **Later sessions** — fresh runtime, `RESUME_EXPERIMENT_DIR = "<root printed by [P3]>"`, same
   cells. Interrupt only once `Epoch N+1/50` has appeared, never while checkpoints are being
   written.
3. **Post-run** — fresh runtime, `RUN_POST_RUN_EVALUATION = True`,
   `POST_RUN_EXPERIMENT_DIR = "<root>"`. Run `[S]`→`[8]`, then `[R1]`–`[R4]`, `[D2]`–`[D13]`,
   then `[P3]`. Budget four validation passes.
4. **Fill in this report** from `evaluation/final_diagnostic_report.json` and the `[D2]`–`[D13]`
   outputs, then complete §9, §14 and the ΔQWK comparison.

## 14. Conclusion

**Under this experiment, the NO-RACAF control performed better than the finalized RACAF model on
the pre-registered primary metric.** Pre-registered ΔQWK (with − without) = **−0.0828302**: BEST
val_QWK 0.7632750 with RACAF against 0.8461052 without.

**What supports it.** The gap is large and is not a single noisy peak: the control exceeded the
reference's best at epoch 8, led in 26 of 30 matched epochs, averaged 0.8291 over epochs 31–50, and
its LAST checkpoint (0.8319) also beats the reference's BEST. The gain is concentrated where the
reference was weakest. Severe cases graded ≤ 2 fell from 68 to 41; grade 3 recall rose from 0.2308 to
0.5385 and grade 4 from 0.0690 to 0.2414; errors of three or more grades fell from 17 to 7. About
three quarters of the reduction in QWK's weighted disagreement came from true grades 3 and 4.

**What goes against it or limits it.**
- **Loss and calibration favour RACAF:** val_loss 0.3024 vs 0.6215; ECE 0.052 vs 0.165; Brier 0.342
  vs 0.403. The control overfit and became overconfident.
- **Some compression was traded for over-grading:** true grades 0–2 predicted as 3–4 rose from 21 to
  34, and grade-2 recall fell from 0.7727 to 0.6869.
- **One run per arm, unseeded initialisation** (limitation 1), with visibly different trajectories.
- **Duplicates have since been excluded exactly** (limitation 9) and the advantage is unchanged
  (ΔQWK +0.0829 on 689 images, 95% CI excluding zero) — this is no longer an open concern.

**What can be claimed:** under the controlled experimental conditions, removing RACAF raised BEST
val_QWK by 0.0828 and improved severe-grade recognition, while worsening validation loss and
calibration. **RACAF's contribution to discrimination is not supported by this experiment.**

**What cannot be claimed:** that RACAF is harmful in general, that the NO-RACAF control generalizes
better, that the gap is purely architectural rather than partly run-to-run variance, or anything
clinical.

**Delivery model unchanged.** This run is a control. The delivery model remains
`2026-09-12_02-45-05_BEST`; any change would be a separate, explicit decision.

**Next steps**, detailed in [RACAF_vs_NO_RACAF_Analysis.md](RACAF_vs_NO_RACAF_Analysis.md) §14:
1. ~~Exact paired and duplicate-excluded comparison~~ — done; see
   [RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md). The advantage
   is unchanged by exact duplicate exclusion (ΔQWK +0.0829 on 689 images) and by an exact paired
   bootstrap (95% CI [+0.0460, +0.1202]).
2. A pre-registered multi-seed repeat of RACAF vs NO-RACAF before any architectural claim — now the
   primary open item.
3. IDRiD external evaluation of the finalized model, reporting RACAF as not demonstrated superior.
4. No redesign or retuning of RACAF on the basis of this validation result.

The finalized RACAF experiment `2026-09-12_02-45-05` and its report remain unmodified by this
experiment.
