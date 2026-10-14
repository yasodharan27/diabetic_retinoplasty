# Multi-Seed RACAF vs NO-RACAF — Improved Training (Pre-Registration)

> **STATUS: COMPLETE.** All six runs finished and were evaluated; experiment id
> `improved_multiseed_2026_09`. Sections 1–5 and 13 were fixed before any run started.
> Sections 6–12 and 14 are filled in **only** from the six runs' saved, executed outputs
> (`experiments/ImprovedTraining/improved_multiseed_2026_09/…`), exactly as
> `RACAF_Ablation_Experiment_1_Report.md` was filled in from its own executed notebook.
>
> **Primary pre-registered verdict: INCONCLUSIVE** (1/3 seeds positive under the sign-consistency
> rule). Every number below was independently recomputed from the persisted per-sample prediction
> CSVs by the post-hoc analysis script and agreed with each run's saved `metrics_*.json` to within
> 1e-6; the recomputed BEST QWK also agreed with the originally printed values to within 1e-4.

> **This experiment is a separate, additive undertaking.** It does not modify, retrain, or
> supersede the finalized RACAF experiment (`2026-09-12_02-45-05`), the finalized NO-RACAF
> ablation (`2026-09-13_04-11-07`), or either of their reports. The delivery model remains
> `exported_models/FinalClassification/2026-09-12_02-45-05_BEST` unless a separate, explicit
> decision changes that after this experiment and a subsequent IDRiD evaluation are both complete.

---

## 1. Protocol

**Question:** how do RACAF and NO-RACAF compare under a corrected training protocol, across
multiple random seeds? This experiment does **not** ask whether any one of the three protocol
changes (A/C/D below) individually helps — see §13's limitations.

Six runs, one experiment: {RACAF, NO-RACAF} × seed ∈ {42, 123, 2026}, all sharing the identical
APTOS2019 split, cache, batch size (2), maximum epoch count (50, early stopping active), monitor
(`val_QWK`, mode `max`), and every other setting the pre-registration below fixes. The **only**
intended difference between RACAF and NO-RACAF, at a matched seed, is RACAF's presence.

Full machine-readable pre-registration: `experiments/ImprovedTraining/<experiment_id>/PREREGISTRATION.json`
(written once, hash-verified on every later read; see `multiseed_runs.write_preregistration()` /
`load_and_verify_preregistration()`).

## 2. Changes A–D (the corrected protocol)

| | Change | What it fixes / why | What it does NOT change |
|---|---|---|---|
| A | Augmentation RNG fix | The pre-existing defect (`JOINT_TRAINING_ARCHITECTURE.md` Sec 49): `gen()` re-seeded on every epoch, so every image received an identical augmentation every epoch. `improved_training_data.py` makes each image's augmentation and each epoch's training order a pure function of `(run_seed, epoch, image_id)` — genuinely stochastic across epochs, exactly reproducible for a given seed and epoch, and resume-safe by construction (rebuilding an epoch's dataset from scratch always reproduces it exactly). | The augmentation policy itself: still exactly `local_feature_extraction_dataset._augment_spatial()` + `_augment_intensity_rgb()`, unmodified, applied in the same order to the same 8-channel tensor. |
| B | Multi-seed | Three run seeds (42, 123, 2026) per arm, six runs total, matched pairwise by seed. `keras.utils.set_random_seed(run_seed)` immediately before model construction. | The split seed, always 42, never a run seed (`multiseed_runs.verify_split()` hardcodes it — no call signature accepts a run seed in its place). |
| C | Class-imbalance handling | A dedicated, CORN-compatible weighted loss (`weighted_corn.py`): square-root inverse-frequency class weights, applied to the loss NUMERATOR only, with the denominator kept as the UNWEIGHTED included-pair count — so unit weights reproduce `corn.corn_loss` exactly. **Not** Keras `class_weight=`, which is mathematically incompatible with CORN's already-pooled loss (see the audit). | CORN's architecture, decode rule, and QWK metric — all unchanged. |
| D | Regularisation | AdamW (`learning_rate=1e-4`, `weight_decay=0.05`), decoupled decay excluded from every bias and BatchNorm/LayerNorm gamma/beta, applied **identically** to both arms — RACAF's own gate/projection kernels are **not** given a decay exception (a deliberate, pre-registered decision; see §14 for the disclosed consequence). | No new dropout, drop-path, label smoothing, or architecture change; ReduceLROnPlateau is kept as-is (patience 4, factor 0.5, min_lr 1e-6). |

Class weights (grades 0–4, from the committed training-split counts `[1444, 296, 799, 154, 236]`,
power 0.5): **[0.6929, 1.5304, 0.9315, 2.1217, 1.7139]** (`weighted_corn.PREREGISTERED_CLASS_WEIGHTS`).

## 3. Dataset / split

Identical to every prior experiment in this project: APTOS2019, the committed
`dataset_splits/aptos2019_train_val_split.csv`, split seed 42, 2929 train / 733 validation.
Split file SHA-256 (LF-normalised): `bc80fd450340b09307fbd80a1b00553e70e34d64a3cdf94635162b6c1e99aca5`.
Exact per-epoch training/validation yield (after empty-field-of-view exclusion, derived from the
real, already-warm cache — never guessed) is pinned once in `experiment_manifest.json` at
experiment creation, together with the excluded empty-field-of-view ids, and re-verified at the
start of every training/evaluation session.

**Data path (cache-first, as in the finalized runs).** Training and evaluation read only the
extracted Stage 02/03/04/RACAF cache: the Drive `cache_archive/` shards are extracted to local SSD
once per runtime and shared by both arms and all three seeds. No raw image is read and no
Stage 02–04 inference runs during training or resume. Augmentation (A) is applied to the cached
8-channel Stage 05 tensor — the same representation the finalized runs augmented. The
empty-field-of-view images, which have no cached representation, are excluded up front rather
than rediscovered by Stage 03 every epoch; the evaluated population is unchanged. A genuinely
missing cache entry would be generated once, by the existing Phase 1 generator, and persisted to
Drive (`improved_training_data.complete_local_cache()`).

## 4. Seeds

`SPLIT_SEED = 42` (fixed, dataset partition only) is entirely separate from `RUN_SEED ∈ {42, 123,
2026}` (weight initialisation, training order, augmentation). `run_config_hash()` includes the
run seed; a resume with a different seed is refused, not silently accepted.

## 5. Checkpoint / resume design

Full detail: `multiseed_runs.py`'s module docstring. Summary: `training.checkpointing`'s existing
generation-based LAST checkpoint (model + full optimiser state, one JSON training-state record)
is reused **unmodified**, invoked once per epoch (`model.fit(epochs=e+1, initial_epoch=e)`) rather
than once per multi-epoch run — with `TrainingConfig(resume=True)` fixed for every call, which is
what lets `EarlyStopping`/`ReduceLROnPlateau`'s persisted counters carry over correctly both
within one Colab session and across a brand-new runtime, using the exact same code path either
way. Additions on top: a two-slot BEST (`best_a`/`best_b` + `best.json`, never deleting the active
slot before the replacement is fully written and validated), a `stop_decision.json` sidecar so a
resumed session never trains one epoch past a sealed EarlyStopping/epoch-cap decision, one
immutable `run_manifest.json` per run, and a lightweight heartbeat lock so two runtimes cannot
train the same run at once without an explicit override.

**Resume compatibility: training-behaviour fingerprint, not git commit.** The experiment may span
several repository commits. A run's persisted checkpoints are resumed only under the same
*training behaviour*: `multiseed_runs.training_behavior_fingerprint()` hashes (canonical JSON,
SHA-256) the run configuration (arm, seed, split, class weights, loss, optimiser, weight decay and
its exclusions, LR schedule, batch size, epoch cap, early stopping, monitor, precision), the exact
train/validation membership, and the normalised source (comments and docstrings removed) of the
code that determines fitting — the architecture and matched initialisation, weighted CORN loss and
QWK metric, callbacks, and the cache-first sample construction, augmentation and epoch ordering
(`TRAINING_BEHAVIOR_SOURCES`). Infrastructure, documentation, notebook, cache-staging and
evaluation code are excluded. The active fingerprint, its components, and every git-commit or
restart event are recorded per run in `training_behavior.json`. Same fingerprint → resume, whatever
the commit. Changed (or missing) fingerprint on an unfinished run with saved state → its
checkpoints, BEST, history, logs and evaluation are moved to `superseded/restart_NNN/` (never
deleted) and the run restarts from epoch 0 with no old weights, optimiser state or history.
Completed runs are never restarted.

## 6. Per-run results

From each run's `evaluation/metrics_{best,last}.json` and `checkpoints/{best,latest}.json`.
All six runs reached `COMPLETED`; every run stopped by **early stopping**, none by the 50-epoch
cap. No run recorded a `restarted_from_epoch_0` event, so each trajectory below is a single,
uninterrupted training history. Evaluated population: **730 images** for every run (the pinned
validation set minus the three fixed empty-field-of-view exclusions).

BEST epoch is `checkpoints/best.json`'s `epoch` (0-based, `TrainingState.best_epoch`); LAST epoch
is `checkpoints/latest.json`'s `completed_epoch` (1-based, a count of finished epochs). The two
are on different conventions and are not directly comparable — see §13.8.

| Run | Status | BEST epoch (0-based) | BEST val_QWK | LAST epoch (1-based) | LAST val_QWK | Epochs trained | Stop reason | Restarted? |
|---|---|---|---|---|---|---|---|---|
| RACAF seed 42 | COMPLETED | 23 | **0.829753** | 36 | 0.811372 | 36 | early_stopping | no |
| RACAF seed 123 | COMPLETED | 14 | **0.795929** | 27 | 0.753978 | 27 | early_stopping | no |
| RACAF seed 2026 | COMPLETED | 14 | **0.816031** | 27 | 0.761316 | 27 | early_stopping | no |
| NO_RACAF seed 42 | COMPLETED | 23 | **0.829674** | 36 | 0.791865 | 36 | early_stopping | no |
| NO_RACAF seed 123 | COMPLETED | 14 | **0.806841** | 27 | 0.791470 | 27 | early_stopping | no |
| NO_RACAF seed 2026 | COMPLETED | 14 | **0.826873** | 27 | 0.817774 | 27 | early_stopping | no |

At each seed the two arms trained for the same number of epochs and selected BEST at the same
epoch index — expected under the matched-initialisation, matched-schedule protocol, not a sign of
coupling between the arms.

**BEST vs LAST.** BEST exceeded LAST in all six runs (+0.0184, +0.0420, +0.0547 for RACAF at
seeds 42/123/2026; +0.0378, +0.0154, +0.0091 for NO_RACAF), confirming the monitored-checkpoint
rule is doing real work and that both arms decline after their best epoch. No conclusion is drawn
from the fact that RACAF's BEST−LAST gap is larger at two of three seeds; the selection rule is
unchanged and identical for both arms.

## 7. Across-seed results

BEST `val_QWK`, three seeds per arm (n=3; descriptive only — see §13.2).

| Arm | Mean | SD | Median | Min | Max | Range |
|---|---|---|---|---|---|---|
| RACAF | 0.8139 | 0.01701 | 0.8160 | 0.7959 | 0.8298 | 0.0338 |
| NO_RACAF | 0.8211 | 0.01245 | 0.8269 | 0.8068 | 0.8297 | 0.0228 |

The two arms' ranges overlap substantially. RACAF's across-seed spread is the wider of the two,
but with three seeds this is a description of these six runs, not an estimate of seed-sensitivity.

## 8. Paired RACAF vs NO-RACAF comparison

Δ = QWK_RACAF − QWK_NO_RACAF, computed on the **identical** validation images at each seed. The
post-hoc analysis verified per seed that both arms evaluated exactly the same 730 `image_id`s with
identical ground-truth grades, with no duplicated, extra or missing ids (`identical_membership =
True` at all three seeds). Bootstrap: 2000 paired resamples, percentile method, `default_rng(20260913)`.

| Seed | Δ (BEST val_QWK) | 95% paired bootstrap CI | Sign |
|---|---|---|---|
| 42 | **+0.0001** | [−0.0213, +0.0192] | + |
| 123 | **−0.0109** | [−0.0364, +0.0139] | − |
| 2026 | **−0.0108** | [−0.0308, +0.0074] | − |

- **Mean paired Δ = −0.0072** (SD across the three seed-level Δ = 0.0063; n=3, descriptive only —
  this is *not* an inferential interval and does not increase the effective sample size).
- **Positive Δ in 1 of 3 seeds.**
- **Every per-seed CI contains zero.**

**Pre-registered decision rule → INCONCLUSIVE.** Sign consistency is not achieved in either
direction, and no per-seed interval excludes zero.

**Sample-level paired agreement (secondary).** Per-seed 2×2 correctness transitions on the same
730 images, with an exact (binomial) McNemar test:

| Seed | Both correct | RACAF only | NO_RACAF only | Both wrong | McNemar exact p |
|---|---|---|---|---|---|
| 42 | 518 | 27 | 44 | 141 | 0.0568 |
| 123 | 510 | 29 | 37 | 154 | 0.3891 |
| 2026 | 447 | 58 | 55 | 170 | 0.8509 |

No seed reaches p < 0.05. Seed 42 is the closest and leans toward NO_RACAF; seeds 123 and 2026 are
far from significance. **A non-significant McNemar result does not establish that the two arms are
equivalent** — it establishes only that this experiment did not detect a difference.

**Effect size (secondary).** Standardised paired difference in absolute grade error, within each
seed: d = +0.0609 (seed 42), +0.0368 (seed 123), −0.0058 (seed 2026) — negligible at every seed,
on a scale where 0.2 is conventionally "small". This is sample-level uncertainty *within* a seed and
is not pooled across seeds.

## 9. Severe-grade analysis

Secondary metric, BEST checkpoint, reported in both directions. "Severe error" is defined here as
|predicted − true| ≥ 3 grades; the project's existing ≥2-grade convention is reported alongside it
in the post-hoc `error_severity.csv`.

| Run | MAE | Grade-3 recall | Grade-4 recall | Severe-error rate (≥3) |
|---|---|---|---|---|
| RACAF seed 42 | 0.3356 | 0.333 | 0.241 | 0.0096 |
| NO_RACAF seed 42 | 0.3110 | 0.385 | 0.103 | 0.0082 |
| RACAF seed 123 | 0.3603 | 0.359 | 0.086 | 0.0123 |
| NO_RACAF seed 123 | 0.3438 | 0.410 | 0.052 | 0.0110 |
| RACAF seed 2026 | 0.3932 | 0.513 | 0.328 | 0.0082 |
| NO_RACAF seed 2026 | 0.3959 | 0.513 | 0.328 | 0.0068 |

Grade-4 recall is the weakest metric in the experiment for both arms at two of three seeds (0.05 –
0.33), and it is also the most seed-sensitive quantity reported anywhere in this report — it varies
by a factor of ~4 across seeds *within* the same arm (RACAF: 0.086 → 0.328; NO_RACAF: 0.052 →
0.328). At seed 2026 the two arms produce identical grade-3 and grade-4 recall. No arm is
characterised as better on this basis: the between-seed variation exceeds the between-arm
variation, which is the honest reading.

## 10. Duplicate-excluded analysis

Secondary. Uses the existing pinned 41-image train/validation duplicate list from
[RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md) (parsed from that
report at analysis time, not rediscovered). 41 images removed at each seed; 689 remain.

| Seed | Δ (all 730) | Δ (689, duplicates excluded) | Shift |
|---|---|---|---|
| 42 | +0.0001 | **−0.0029** | −0.0030 |
| 123 | −0.0109 | **−0.0158** | −0.0049 |
| 2026 | −0.0108 | **−0.0100** | +0.0008 |

**Note, reported because the protocol requires reporting secondary results in whichever direction
they fall:** with the pinned duplicates removed, Δ is negative at *all three* seeds — the single
positive seed (42) flips sign. Under the pre-registered rule this changes nothing: §8's primary
comparison is defined on the full evaluated population, and the verdict there remains INCONCLUSIVE.
It is recorded here so that the duplicate-excluded view is not quietly more favourable to RACAF
than the primary one — it is less.

## 11. Calibration

ECE and Brier were recomputed per run from the persisted `class_prob_*` /
`predicted_class_probability` columns (the same formulation
`multiseed_runs.summarize_predictions()` uses: multiclass Brier; 10 equal-width bins over top-1
confidence, `(lo, hi]`), and agreed with each run's saved `metrics_*.json` to within 1e-6. Per-run
values and reliability-diagram bins are persisted in the post-hoc analysis directory
(`per_run_metrics.csv`, `calibration.csv`, `figures/calibration_plots.png`).

*Per-run ECE/Brier values are not transcribed into this table yet — they were written to the
post-hoc artifacts but not printed to the console in the analysis run of 2026-09-18. Transcribe
them here from `per_run_metrics.csv` rather than recomputing or estimating them.*

As pre-registered, calibration is **shifted for both arms** by Change C's class-prior reweighting:
a weighted objective deliberately moves the predicted distribution away from the empirical prior,
so an ECE change here is an expected consequence of the protocol, not a defect, and is not evidence
for or against RACAF. It was not used to select any checkpoint or to decide the primary comparison.

## 12. Training dynamics

All six runs terminated by early stopping; none reached the 50-epoch cap. Train-minus-validation
QWK gap at the final epoch:

| Run | Epochs | Stop reason | Final train−val QWK gap |
|---|---|---|---|
| RACAF seed 42 | 36 | early_stopping | +0.0763 |
| NO_RACAF seed 42 | 36 | early_stopping | +0.1028 |
| RACAF seed 123 | 27 | early_stopping | +0.1115 |
| NO_RACAF seed 123 | 27 | early_stopping | +0.0824 |
| RACAF seed 2026 | 27 | early_stopping | +0.1123 |
| NO_RACAF seed 2026 | 27 | early_stopping | +0.0498 |

Every run ends with training QWK above validation QWK by 0.05–0.11 — consistent, mild overfitting
past the selected epoch, of similar magnitude in both arms and with no arm systematically higher
across all three seeds. No run showed divergence, loss explosion or a stalled learning rate.
Per-epoch curves: `figures/training_validation_qwk_curves.png` in the post-hoc analysis directory.

Gate statistics were never written to `history/epoch_*.json` during training, but they were
recovered afterwards, read-only, from the persisted artifacts — see §12.1.

### 12.1 RACAF reliability and gate diagnostics (recovered post-hoc, read-only)

Recovered from the Drive RACAF cache (`APTOS_<id>_racaf_reliability.npz`, n = 3651 images) and
from the `reliability_gate` Dense(1) kernel/bias in each RACAF run's active BEST checkpoint
(`checkpoints/best_<slot>/model.weights.h5`). No model was retrained and no run directory was
written to.

**Reliability input `r`** (the only quantity the gate consumes): mean 0.8631, SD 0.04251,
min 0.6427, p1 0.7549, median 0.8652, p99 0.9525, max 1.0000; 240 distinct values at 3 dp.
`r` therefore **does vary across images** — it is not a constant.

**Per-class reliability `κ`:**

| Class | Mean | SD | Fraction exactly 1.0 | Stage 04 test Dice |
|---|---|---|---|---|
| MA | 1.0000 | **0.00000** | **1.000** | 0.0165 |
| HE | 0.7911 | 0.07684 | 0.004 | 0.1273 |
| EX | 0.7565 | 0.13387 | 0.059 | 0.3574 |
| SE | 0.8420 | 0.15014 | 0.331 | 0.0244 |

**κ[MA] is a hard constant 1.0 for all 3651 images.** Stage 04 predicts no microaneurysm
foreground anywhere, so `compute_reliability()` takes its `mask.any() == False` branch, sets
δ = 0 and therefore κ = 1.0 — the class the segmenter is blind to is scored as *maximally
reliable*. κ[SE] takes the same branch on 33.1% of images. This is a defect in the reliability
definition, independent of the primary comparison's outcome.

Note also the compression: per-class κ carries SD up to 0.150, but the burden-weighted scalar `r`
the model actually receives has SD 0.0425 — roughly a 3× loss of dynamic range before the gate.

**Learned gate** (`gate = sigmoid(w_g · r + b_g)`, 2 trainable parameters), evaluated over the
p1–p99 span of the observed `r`:

| Run | `w_g` | `b_g` | gate at r=p1 | gate at r=p99 | Swing | Gate at median r |
|---|---|---|---|---|---|---|
| RACAF seed 42 | **+1.5607** | 0.0160 | 0.7675 | 0.8180 | **0.0505** | 0.797 |
| RACAF seed 123 | **−1.5565** | 0.0197 | 0.2395 | 0.1880 | **0.0515** | 0.210 |
| RACAF seed 2026 | **+0.5254** | 0.0225 | 0.6033 | 0.6278 | **0.0246** | 0.617 |

Three observations, all directly from these numbers:

1. **Per-image modulation is negligible.** Across the 1st–99th percentile of `r`, the gate moves
   by only 0.025–0.052. RACAF's adaptive mechanism changes the fusion mix by at most ~5 percentage
   points between its most- and least-reliable images.
2. **The learned sign is not consistent across seeds.** `w_g` is +1.56, −1.56 and +0.53 — seed 123
   learned the *opposite* reliability→trust relationship to the other two, at nearly mirrored
   magnitude. A genuine, learnable relationship between reliability and how much to trust the
   segmentation-dependent branch would be expected to reproduce its direction across seeds. It
   does not.
3. **What the gate actually learned was a fixed blend level, and that level varies enormously
   across seeds.** At median `r` the gate sits at 0.797, 0.210 and 0.617 for seeds 42/123/2026 —
   i.e. the fused embedding is 80% / 21% / 62% cross-attention output `E` and the remainder global
   readout `Ĝ`. The corresponding BEST QWKs are 0.8298, 0.7959 and 0.8160: a 3.8× swing in the
   blend ratio with only 0.034 between the outcomes, inside the across-seed spread of either arm
   (§7). **This across-seed insensitivity must not be read as the gate being a low-leverage
   parameter** — §12.2 shows it is the opposite within any single trained model. The two
   observations are reconciled by co-adaptation: the downstream CORN head trains against whatever
   blend level emerged in that run, so different runs reach different, internally self-consistent
   operating points of similar quality.

### 12.2 Gate-forcing ablation (post-hoc, read-only, no retraining)

Each RACAF BEST checkpoint was loaded into memory, its gate kernel/bias (identified by shape
(1,1)/(1,)) temporarily overwritten to force `gate ≡ 1` (kernel 0, bias +20) and `gate ≡ 0`
(kernel 0, bias −20), evaluated on the identical 730 validation images in identical order, and the
original gate weights restored and verified bit-for-bit after every seed. No checkpoint, run
directory or manifest was written.

**Setup validation:** the as-learned condition reproduced each run's persisted BEST QWK — 0.7959
and 0.8160 exactly at seeds 123/2026, and 0.8291 vs the persisted 0.829753 at seed 42 (a single
borderline sample flipping, attributable to forward-pass numerics; grade-3 recall moved 13/39 →
12/39). The evaluation path is therefore confirmed equivalent to the one that produced §6.

| Seed | as-learned QWK | gate≡1 QWK | gate≡0 QWK | Δ(learned−g1) | Δ(g1−g0) | MAE g0 | acc g0 |
|---|---|---|---|---|---|---|---|
| 42 | 0.8291 | 0.7784 | **0.0000** | +0.0506 | +0.7784 | 1.1219 | 0.4945 |
| 123 | 0.7959 | 0.6917 | **0.0000** | +0.1042 | +0.6917 | 1.3027 | 0.2712 |
| 2026 | 0.8160 | 0.7920 | **0.0000** | +0.0240 | +0.7920 | 1.3027 | 0.2712 |

**`gate ≡ 0` produces a constant prediction in all three runs** — every one of the 730 images
assigned grade 0 (seed 42) or grade 2 (seeds 123, 2026), giving QWK exactly 0, balanced accuracy
exactly 0.2, and an accuracy equal to that grade's prevalence. This is a collapse, not a
performance measurement.

**That collapse is demonstrably an extrapolation artifact, not evidence that `Ĝ` lacks
information.** Seed 123's learned gate is 0.210, i.e. its fused embedding is already ~79% `Ĝ`, and
it achieves QWK 0.7959 in that configuration. A representation that carries 79% of the input to a
head scoring 0.796 cannot itself be uninformative. Moving that same model from 79% `Ĝ` to 100% `Ĝ`
nonetheless drives it to a constant output — which is the signature of the CORN head being pushed
outside the input distribution it co-adapted to, not of `Ĝ` being noise. The symmetric reading
applies to `E`.

**Consequence for interpretation:** neither forced gate condition isolates the lesion-derived
pathway. `E` is the Adaptive Cross-Attention output and already contains global content, so
`gate ≡ 1` retaining QWK 0.69–0.79 does not attribute that performance to lesion-derived features;
and `gate ≡ 0` is uninterpretable for the reason above. **This ablation therefore does not measure
the lesion-derived branch's predictive contribution**, in either direction. What it does establish
is that the gate is a high-leverage parameter within a trained model, and that the fusion point is
sensitive to perturbation well before the extremes are reached.

### 12.3 Lesion-package permutation (post-hoc, read-only, no retraining)

Each RACAF BEST checkpoint was evaluated on the identical 730 validation images under three
conditions: **original**; **permuted lesion+reliability** (channels 4:8 of `stage5_input` — all four
lesion maps together — plus the cached reliability scalar, taken as one package from another
validation image); and **permuted lesion, original reliability**. RGB, vessel and `stage6_input`
were never altered; donors were validation images only; the donor assignment was a single Sattolo
derangement (seed 20260918) shared by all three seeds, verified to have no fixed point. The
original condition reproduced each persisted BEST QWK (drift 0.000000, 0.000000, 0.000661).

| Seed | original | permuted (lesion+rel) | Δ | 95% CI on Δ | permuted (lesion only) | Δ |
|---|---|---|---|---|---|---|
| 42 | 0.8291 | 0.1416 | **+0.6874** | [+0.6113, +0.7637] | 0.1387 | +0.6904 |
| 123 | 0.7959 | 0.0604 | **+0.7355** | [+0.6618, +0.8106] | 0.0778 | +0.7181 |
| 2026 | 0.8160 | 0.1869 | **+0.6292** | [+0.5579, +0.7041] | 0.1889 | +0.6272 |

MAE rises from 0.337/0.360/0.393 to ~1.11–1.18 in every seed. Paired per-sample: 416–443 of 730
predictions change, worsening outnumbering improvement roughly 8:1 (e.g. seed 42: 366 worse, 45
better; 295 correct→incorrect vs 36 incorrect→correct). Every bootstrap CI excludes zero by a wide
margin, and the direction is identical in all three seeds.

Unlike the gate≡0 collapse (§12.2), the permuted models do **not** emit a constant: predictions
remain spread across all five grades (seed 42 predicted-class counts 285/193/168/64/20). The
degradation is a loss of discriminative accuracy, not a degenerate output.

**The dissociation that matters.** Moving the reliability scalar with the lesion package versus
leaving the destination image's own reliability changes QWK by only 0.0029 / 0.0174 / 0.0020. The
lesion *maps* account for essentially the entire effect; the reliability *scalar* — RACAF's actual
mechanism — accounts for ~0.002–0.017. This is a direct inference-time measurement, and it agrees
with the gate's ±0.025–0.052 swing (§12.1) and with the six-run null (§8).

**What this does and does not license.** It establishes that these trained models depend heavily on
the lesion channels being image-specific. It does **not** establish that the dependence is on
*lesion* information as such: Stage 04 is a CNN applied to the image, so its output is an
informative image-derived transform regardless of its segmentation quality (mean Dice 0.1314), and
permutation destroys alignment with the image for any such channel. Distinguishing "lesion
information is diagnostic" from "any misaligned image-derived channel breaks a co-adapted model"
requires the same permutation applied to the **vessel** channel as a control; that has not been run.
Permutation importance also evaluates off the joint data manifold — each component is in
distribution, their combination is not.

### 12.4 Vessel-permutation control (post-hoc, read-only, no retraining)

The matched control §12.3 called for. Identical validation set, ordering, derangement (Sattolo,
seed 20260918) and bootstrap convention (seed 20260913, 2000 resamples); the only difference is
that `stage5_input[..., 3:4]` (vessel) is reassigned instead of `[..., 4:8]` (lesion). RGB, lesion
channels, reliability and `stage6_input` untouched. The original condition again reproduced each
persisted BEST QWK (drift 0.000000 / 0.000000 / 0.000661).

| Seed | original | vessel-permuted | ΔQWK | 95% CI | lesion ΔQWK (§12.3) | vessel / lesion |
|---|---|---|---|---|---|---|
| 42 | 0.8291 | 0.7867 | **+0.0424** | [+0.0182, +0.0661] | +0.6874 | **0.062** |
| 123 | 0.7959 | 0.7211 | **+0.0748** | [+0.0472, +0.1045] | +0.7355 | **0.102** |
| 2026 | 0.8160 | 0.7564 | **+0.0597** | [+0.0350, +0.0850] | +0.6292 | **0.095** |

Paired mean |error| change +0.093 / +0.096 / +0.107 (lesion: +0.775 / +0.823 / +0.718). Predictions
changed 98–158 of 730 (lesion: 416–443), with worse:better ≈ 3:1 (lesion ≈ 8:1).

**Reading.** Vessel permutation costs 6–10% of what lesion permutation costs, consistently at every
seed. Had the §12.3 degradation been driven by the generic effect of misaligning *any*
image-derived channel in a co-adapted model, the vessel control — also a frozen-model output, also
a channel of the same tensor, disarranged by the same derangement — would have been comparable. It
is an order of magnitude smaller. **This substantially weakens the generic-coherence explanation
and strengthens the interpretation that the lesion channels carry image-specific predictive
information.**

Two honest qualifications. First, the vessel effect is **not zero** — every CI excludes zero — so a
real generic-coherence component exists, of order 0.04–0.07 QWK. Second, lesion permutation
disarranges **four** channels versus vessel's **one**, so part of the gap is simply more of the
input being scrambled; the ~10–16× damage ratio against a 4× perturbation ratio is superlinear and
argues for lesion-specificity, but this is a crude accounting, not a rigorous decomposition.

Per-grade, the two interventions differ in character: vessel permutation mainly costs **grade-0
recall** (0.931→0.809, 0.906→0.842, 0.828→0.734) while leaving grades 3–4 roughly unchanged or
slightly improved, i.e. it makes healthy eyes look diseased; lesion permutation degrades
discrimination across all grades.

**What this still does not establish.** It does not show that Stage 04's *segmentation quality* is
the operative variable. Stage 04's output could be informative as a learned image transform whose
value does not scale with Dice; nothing measured here relates grading performance to segmentation
accuracy. A mask-degradation sensitivity sweep on the existing checkpoints would address that and
remains unrun. Permutation importance also still evaluates off the joint data manifold.

**Placed beside the other two diagnostics**, the contributions at inference time order as: lesion
maps ≈ 0.63–0.74 QWK; vessel map ≈ 0.04–0.07; RACAF's reliability scalar ≈ 0.002–0.017 (§12.3). The
representation RACAF gates is informative; the summary statistic RACAF computes from it is not.

### 12.5 Stage 04 quality audit (read-only) — closes the quality question for this population

A provenance audit was run to determine whether per-image Stage 04 segmentation **quality** can be
measured for the exact evaluated population. It inspected the Drive project tree (bounded walk,
GT-keyword directories), the APTOS 2019 dataset tree, the IDRiD segmentation ground truth, the
IDRiD grading labels and the persistent caches. No model was loaded and nothing was modified.

| Item | Finding | Provenance class |
|---|---|---|
| Validation population | 733 split entries → **730** with all four cached artifacts (3 absent = the pinned empty-FOV exclusions) | — |
| APTOS 2019 tree | **No** mask/annotation directory anywhere; raw and processed contain images only | GROUND TRUTH = image-level DR grade only |
| IDRiD segmentation subset | **81** distinct annotated ids (`IDRiD_01…IDRiD_81`), five folders per split (MA/HE/EX/SE/OD); SE has fewer files (26 train / 14 test) per IDRiD's absent-class convention | GROUND TRUTH (lesion, pixel-level) |
| Join: APTOS-val × IDRiD lesion GT | **intersection = 0**; and **0** filename-level hits across *every* GT-keyword directory on the Drive project | — |
| Stage 04 lesion arrays for the 730 | **730/730 available** | MODEL PREDICTION — never ground truth |
| RACAF reliability for the 730 | **730/730 available** | DERIVED CACHE ARTIFACT — from the prediction |
| Per-image Dice/IoU for the 730 | — | **UNAVAILABLE** |

Two keyword matches were false positives and are recorded as such: `exported_models/VesselSegmentation`
and `exported_models/LesionSegmentation` contain model weights and matched on directory name only.

**Verdict.** Per-image Stage 04 Dice/IoU **cannot be computed for the 730 APTOS validation images**,
because no lesion ground truth exists for them — not by a read-only analysis, and not by any means
short of acquiring pixel annotations APTOS does not ship. **The question "does Stage 04 segmentation
quality drive RACAF behaviour?" is therefore unanswerable on this experiment's population**, and no
proxy is substituted for it. Prediction-derived per-image quantities (lesion burden, entropy, κ, r)
remain joinable to RACAF errors entirely read-only, but they describe the prediction's own
properties, never its correctness, and must not be reported as segmentation quality.

**Recorded as unresolved, not as a finding:** the audit also returned 0 overlap between IDRiD's 81
lesion-GT ids and the 413 ids carrying a DR grade. That comparison used exact string matching, and
the two IDRiD subsets use different id formats (`IDRiD_NN` versus a longer zero-padded form), so it
does **not** establish that the two sets are genuinely disjoint. Whether IDRiD could serve as an
alternative population for a quality-versus-grading analysis is therefore still open; note that it
would in any case require frozen Stage 02/03/04 inference, since the cache is APTOS-keyed.

## 13. Limitations

1. **Bundled protocol change.** Changes A, C and D are applied together. This experiment cannot
   isolate which one (if any) drives an observed difference from the finalized single runs.
2. **Three seeds is limited statistical power.** A paired difference across 3 seeds has 2 degrees
   of freedom; the pre-registered decision rule (sign consistency + per-seed bootstrap CIs) is
   deliberately conservative rather than a formal significance test.
3. **Validation-based selection is optimistic for both arms equally.** BEST is chosen by maximum
   `val_QWK` on the same 730 images used to report it, for both arms and all three seeds alike.
4. **AdamW's weight decay applies to RACAF's own gate and projection kernels**, with no exception
   — a deliberate, pre-registered choice (§2), disclosed rather than hidden, but a further
   confound if RACAF's behaviour changes noticeably between this experiment and the finalized run.
5. **Cross-arm initialisation parity is exact only for the shared Stage 05/06/07/CORN
   construction order**, not a claim of bit-identical GPU training trajectories thereafter (see
   the audit's Reproducibility section: GPU kernels are not deterministic, and dropout's Keras 3
   `SeedGenerator` state is not part of a resumed checkpoint).
6. **This experiment ends at APTOS validation.** IDRiD external evaluation is a separate,
   subsequent step using the unmodified `colab/notebooks/idrid_external_evaluation.ipynb`, applied
   to all six BEST checkpoints, never a hand-picked seed.
7. **The 41 train/validation duplicate images remain in training** (the split is unchanged); the
   duplicate-excluded analysis (§10) is secondary, using the existing pinned list.
8. **BEST and LAST epochs are recorded on different conventions** (`best.json`'s `epoch` is
   0-based; `latest.json`'s `completed_epoch` is 1-based). §6 reports both raw rather than
   silently normalising one to the other.
9. **No RACAF gate or reliability diagnostics were persisted during training** (§12). The
   experiment can therefore say *that* RACAF did not separate from NO-RACAF, but not, from the
   saved artifacts alone, *why* — see §14.

## 14. Scientific conclusion

**Primary, pre-registered: INCONCLUSIVE.** Across three matched seeds, the paired difference in
BEST validation QWK (RACAF − NO-RACAF, same 730 images, verified identical membership) was
+0.0001, −0.0109 and −0.0108 — positive in 1 of 3 seeds, with every per-seed 95% paired bootstrap
CI containing zero. The pre-registered sign-consistency rule is not satisfied in either direction.
**This experiment does not support the claim that RACAF improves DR grading, and it does not
establish that RACAF harms it either.** Mean paired Δ = −0.0072 is reported descriptively only; with
three seeds it is not an inferential quantity, and it is not used to declare a winner.

Everything the protocol required to make that verdict trustworthy did hold: all six runs completed
under a verified-identical split (SHA-256 match), identical pre-registered class weights, identical
train/validation population digests across all six runs, configurations differing only in the
intended `model` field, no silent behaviour change or restart in any run, and independent
recomputation of every metric from the raw per-sample predictions agreeing with the saved values.
The result is a clean null, not a compromised experiment.

**Secondary results all point the same way or nowhere.** No per-seed McNemar test reaches p < 0.05
(closest: seed 42, p = 0.057, leaning NO-RACAF). Per-seed standardised effect sizes on absolute
grade error are negligible (|d| ≤ 0.06). With the pinned duplicates excluded, Δ is negative at all
three seeds (§10). Severe-grade behaviour varies more *between seeds within an arm* than between
arms (§9). Nothing in the secondary analyses rescues a positive reading, and — per the protocol —
nothing in them is permitted to override the primary verdict either.

**Why the arms did not separate — resolved by the §12.1 diagnostics.** The read-only recovery of
RACAF's reliability inputs and learned gate parameters answers this directly, and rules out the
explanation that would have been most convenient to assume.

*Not because the reliability signal is constant.* `r` varies across images (SD 0.0425, 240 distinct
values, range 0.6427–1.0000). The hypothesis that RACAF was fed a constant is **refuted**.

*Because the gate barely converts that variation into fusion change, and does not agree with
itself across seeds.* The gate moves by only 0.025–0.052 across the p1–p99 span of `r`, and its
learned sign is inconsistent (`w_g` = +1.56, −1.56, +0.53). What each run actually learned was a
**fixed blend level** — 0.797, 0.210, 0.617 at median `r` — with per-image modulation of a few
percentage points around it. RACAF therefore behaved, in all three runs, as an approximately static
re-weighting of `E` against `Ĝ`, which is a function NO-RACAF's architecture can already express.
That is a complete and sufficient explanation of Δ ≈ 0.

*And the blend level is not transferable between runs, though it is decisive within one.* Across
the three seeds the fixed blend ranged from 21% to 80% cross-attention output — a 3.8× swing —
while BEST QWK moved only 0.034, inside either arm's across-seed spread. An earlier draft of this
report read that as the mixing weight being a low-leverage parameter; **§12.2's gate-forcing
ablation shows that reading was wrong and it has been corrected.** Within any single trained model
the gate is extremely high-leverage: forcing it to 0 collapses all three models to a constant
prediction (QWK exactly 0). The reconciliation is co-adaptation — the CORN head trains against
whichever blend level emerged in its own run, so different runs settle at different,
internally-consistent operating points of similar quality, while any given run cannot tolerate
being moved off its own. What remains true, and is what actually bounds RACAF, is the *per-image*
modulation around that operating point: ±0.025–0.052 (§12.1).

*Independently, the reliability definition contains a defect.* κ[MA] = 1.0 for 100% of images and
κ[SE] = 1.0 for 33.1%, via the empty-foreground branch that assigns maximum reliability to a class
with no predicted foreground (§12.1). One quarter of the reliability vector is a constant, and the
burden-weighted collapse to a scalar discards roughly 3× of the available per-class dynamic range.
These would need fixing before any future claim rests on this reliability formulation, regardless
of what is decided about RACAF itself.

**What this still does not establish.** It does not show that reliability-aware fusion is a bad
idea in general — only that *this* instantiation, gating *this* fusion point, with *this*
reliability definition and *this* frozen Stage 04, produced no measurable benefit and had little
mechanical room to. **Whether the lesion-derived branch carries distinctive predictive information
remains open**, and the gate-forcing ablation run for §12.2 did *not* answer it: `gate ≡ 1` yields
`E`, which already contains global content via cross-attention, and `gate ≡ 0` drives the CORN head
off-distribution into a constant output. Neither condition isolates local from global. Answering
that question requires an intervention at the *input* to the local pathway rather than at the
fusion weight — and any decision about retraining Stage 04 should wait for it.

**Status of the delivery model is unchanged.** Per the header of this report, this experiment does
not supersede the finalized runs or change the delivery model. Stage 04 remains frozen; nothing in
this experiment licenses retraining it.
