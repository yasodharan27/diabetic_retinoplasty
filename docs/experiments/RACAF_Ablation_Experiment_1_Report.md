# RACAF Ablation / NO-RACAF Control — Experiment 1

> **STATUS: PRE-REGISTERED, NOT YET RUN.**
> The notebook, architecture, configuration identity and seed parity below are implemented and
> audited. **No training has been performed and no results exist yet.** Every results section is
> explicitly empty and must be filled from the run's own artifacts — never estimated, never
> inferred from the reference. Training requires a Colab T4 session with Google Drive mounted
> (see §13); it cannot be run on a machine without a GPU or without the 28.53 GiB Drive cache.

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
   byte-for-byte unchanged; it is verifiably inert (identical output for `r = 0.0` and `r = 1.0`).
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

## 9. Results — **PENDING, NOT YET RUN**

Nothing in this section may be filled in from the reference or from expectation. Every figure
comes from the ablation's own artifacts once `[T]` and `[R1]`–`[D13]` have run.

| Item | Value |
|---|---|
| Ablation experiment ID | _pending — assigned by `create_experiment()` when `[T]` runs_ |
| Experiment directory | _pending_ |
| Archive | _pending —_ `exported_models/FinalClassification/<id>_NO_RACAF_BEST` |
| Epochs completed / stop reason | _pending_ |
| BEST epoch | _pending_ |
| BEST val_QWK | _pending_ |
| BEST val_loss | _pending_ |
| LAST epoch / val_QWK / val_loss | _pending_ |
| Accuracy, balanced accuracy, MAE | _pending_ |
| Macro / weighted F1 | _pending_ |
| Per-class precision / recall / F1 | _pending_ |
| Confusion matrix | _pending_ |
| Ordinal error distances | _pending_ |
| Brier / ECE | _pending_ |
| Archive SHA-256 | _pending_ |
| Reconstruction + exact-prediction reproduction | _pending_ |

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

## 13. How to run (runbook)

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

## 14. Conclusion — **PENDING**

To be written once §9 is filled from the run's artifacts. It must state the direction and size of
ΔQWK, report the secondary metrics regardless of whether they favour RACAF, and qualify the whole
result by limitation 1 (single run, unseeded initialisation).

The finalized RACAF experiment `2026-09-12_02-45-05` and its report remain unmodified by this
experiment.
