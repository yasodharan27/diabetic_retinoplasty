# Multi-Seed RACAF vs NO-RACAF — Improved Training (Pre-Registration)

> **STATUS: PRE-REGISTERED, NOT YET RUN.** No training has occurred under this protocol.
> Sections 1–5 and 13 are fixed before any run starts. Sections 6–12 and 14 are filled in
> **only** from the six runs' saved, executed outputs (`experiments/ImprovedTraining/<experiment_id>/…`),
> exactly as `RACAF_Ablation_Experiment_1_Report.md` was filled in from its own executed notebook.

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
experiment creation and re-verified, cheaply, at the start of every session.

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

## 6. Per-run results

*Filled in from each run's `evaluation/metrics_{best,last}.json` and `history/epoch_*.json` once
that run is COMPLETED.*

| Run | Status | BEST epoch | BEST val_QWK | LAST epoch | LAST val_QWK | Epochs trained | Resumed? |
|---|---|---|---|---|---|---|---|
| RACAF seed 42 | *pending* | | | | | | |
| RACAF seed 123 | *pending* | | | | | | |
| RACAF seed 2026 | *pending* | | | | | | |
| NO_RACAF seed 42 | *pending* | | | | | | |
| NO_RACAF seed 123 | *pending* | | | | | | |
| NO_RACAF seed 2026 | *pending* | | | | | | |

## 7. Across-seed results

*Mean / SD / median / min / max of BEST val_QWK per arm, once all three seeds of that arm are
COMPLETED — see the notebook's `[X]` cell.*

## 8. Paired RACAF vs NO-RACAF comparison

*Per-seed Δ = QWK_RACAF − QWK_NO_RACAF on the identical validation images, mean Δ, sign
consistency, and the paired per-image bootstrap CI for each seed — from `[X]`.*

## 9. Severe-grade analysis

*Grade 3/4 recall, errors of ≥2 grades, per-seed and pooled, both arms — secondary metric,
reported regardless of which direction it points.*

## 10. Duplicate-excluded analysis

*Using the existing pinned 41-image train/validation duplicate list
([RACAF_Duplicate_Contamination_Audit.md](RACAF_Duplicate_Contamination_Audit.md)), not
rediscovered — secondary metric.*

## 11. Calibration

*ECE and Brier score, both arms, clearly labelled as shifted by Change C's class-prior
reweighting — informative, never used to select a checkpoint or decide the primary comparison.*

## 12. Training dynamics

*LR history, early-stopping epoch, training-vs-validation-loss trajectory, RACAF gate diagnostics
(mean gate value, `w_g`, `b_g`) per seed — from `history/epoch_*.json`.*

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

## 14. Scientific conclusion

*Written only after all six runs are COMPLETED, in the pre-registered wording, honestly reporting
whichever arm the pre-registered decision rule (§1, `PREREGISTRATION.json`'s
`primary_comparison_rule`) actually supports — including if NO-RACAF remains better, or if the
result is INCONCLUSIVE. Not written in advance, and not adjusted after seeing the result beyond
correcting a demonstrated code defect.*
