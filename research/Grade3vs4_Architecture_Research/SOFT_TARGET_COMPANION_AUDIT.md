# Companion mechanism for soft ordinal CORN targets — audit (2026-10-01)

This is a design audit only: nothing was implemented or trained.
- **Objective:** as recorded in §33 — improve overall DR grading; practical; testable in days; novelty not
  required.
- **Evidence base:** research record §§1–34, `COMBINATION_AUDIT.md`, P's pre-registration, and a literature
  search.

## P's measured limitations (unchanged by §34)

- **L1 — memorisation and overfitting.** Training QWK 0.99 vs validation 0.91; validation loss rises after
  epochs 5–8; LAST < BEST in 6/6 runs; late features degrade.
- **L2 — label noise.** 24 % of byte-identical duplicate pairs disagree.
- **L4 — checkpoint instability.** BEST epochs range from 3 to 19, and selection on validation QWK is itself
  a noise source.
- **L3 (representation prior)** is weakened by §34: DINOv2 is substitution, not complementarity.

## P's actual recipe (PREREGISTRATION.json) — what is missing

- AdamW, learning rate 1e-4 constant with reduce-on-plateau, weight decay 0.05.
- **Batch size 2.**
- **Stochastic depth 0.0.**
- **No weight EMA / averaging.**
- **No warm-up or layer-wise LR decay.**
- Augmentation: flips + rot90 + ±10 % brightness/contrast.
- Early stopping on validation QWK (patience 12).
- Class-weighted CORN.

Several standard fine-tuning regularisers are absent, and none of them has been tested here.

## Candidates (5)

| # | Mechanism | Limitation addressed | Evidence (DR / general) | Tested here? | Complements soft ordinal targets? | Data | Implementation / Colab cost | Overfitting risk | Clean ablation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Weight EMA (shadow averaged weights), evaluated at a fixed final epoch** | L1, **L4**, L2 | EMA/SWA give wider optima and better generalisation (Izmailov 2018). EMA improves generalisation, **label-noise robustness**, calibration, and reduces reliance on early stopping (arXiv 2411.18704). Used in DR pipelines (SwinV2 + SWA + EMA); model soups help medical classification (skin-cancer soups) | **No.** SWA/TTA was only set aside "as a contribution" | **Yes — orthogonal.** Weight space vs target space; routinely combined in modern recipes | None | **Negligible compute.** Keras optimisers support `use_ema`. About 1 day to wire EMA weights into the custom checkpoint/eval loop | Reduces it | EMA is a *shadow*: every run yields both raw-BEST and EMA-final models from the same trajectory — a within-run paired comparison |
| 2 | **SAM (sharpness-aware minimisation)** | L1, L2 | Generalisation (Foret 2021). Robustness to label noise (SAM analyses 2024; NCSAM 2026). A fundus noisy-label framework uses SAM | No | Yes — optimiser level | None | **About 2× compute per step** (≈ 6–10 T4-h per run). Custom train step with mixed precision and loss scaling: about 1–2 days | Reduces it | P vs P+SAM, same seeds |
| 3 | **Stochastic depth 0.1–0.2** (± layer-wise LR decay, warm-up/cosine) | L1 | Part of ConvNeXt's own fine-tuning recipe (Liu 2022), and critical on small sets; generic, with little DR-specific evidence | No (P used 0.0) | Yes | None | One constructor argument; no extra compute | Reduces it (mild underfit risk) | P vs P+drop-path |
| 4 | **Ordinal-aware mixup (C-Mixup / adjacent-grade mixing into soft CORN targets)** | L1, L2 | Mixup is common in DR; standard mixup **distorts ordinal structure** (DisMix 2026). C-Mixup mixes nearby labels | No | **Partially overlapping** — it is another target-softening mechanism | None | About 1 day; with batch 2, pairing has to be across batches | Reduces it; blended lesions are unrealistic | P+soft vs P+soft+mixup |
| 5 | **Noisy-label training (ELR / co-teaching / FM-guided down-weighting)** | L2 | Established (co-teaching, DivideMix, ELR; fundus QMix; CUFIT, NeurIPS 2024) | No (C9 listed, never run) | **Overlaps** soft targets (same limitation) | None | Co-teaching 2×; DivideMix complex; 2–4 days | Risks discarding genuinely hard PDR (§26) | Hard to isolate |

**B — replacing or improving Stage 3 or Stage 4: not recommended.**
- P does not consume Stage 3/4 outputs.
- Every route that feeds them into P is closed (early fusion §22, RACAF, vessel route, lesion pathway).
- The information they carry — lesion burden — is already linearly recoverable from P (§§24–27).
- A better lesion model would matter only if a lesion route were reopened, which the evidence does not
  support.

## The two strongest

1. **Weight EMA.** It is the only candidate that directly targets **checkpoint instability (L4)** as well
   as overfitting (L1) and label noise (L2). It has essentially zero compute cost, is orthogonal to soft
   targets, and allows a within-run paired ablation.
2. **SAM.** It has the strongest label-noise and generalisation evidence of the optimiser-level
   mechanisms, and is orthogonal to soft targets. It costs about 2× compute and more implementation.

(Stochastic depth is the cheapest runner-up; it could be added later as its own single-variable arm.)

## One next experiment (to be pre-registered before any run)

**A 2 × 2 factorial of soft ordinal CORN targets × weight EMA, using EMA as a shadow.**

- **Runs:** 3 seeds of P re-run with EMA tracking, and 3 seeds of P + soft targets with EMA tracking. That
  is 6 runs, about 18–30 T4-h. The recipe is otherwise identical to §22.
- **Arms per run:** EMA does not feed back into training, so each run yields a raw model (BEST, and a
  fixed final epoch) and an EMA model. That gives all four arms:
  - CORN raw;
  - CORN + EMA;
  - soft raw;
  - soft + EMA.
- **Reproducibility check:** the CORN-raw arm can be compared with the existing §22 P runs.
- **To fix before running:**
  - the EMA momentum;
  - the EMA evaluation rule (a fixed final epoch — no validation-based selection);
  - the soft-target σ;
  - the primary endpoint and guardrails (as in §33: mean CORN cumulative-threshold AUROC; QWK,
    grade-3 recall, and grade 4 vs 0–2).
- **Outcome:** the main effects of each mechanism and their interaction, from one set of runs.
