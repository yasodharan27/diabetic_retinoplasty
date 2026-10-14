# Decision Tree — the pending task-3 weighting vs mass-matched placebo experiment

**Outcome unknown at the time of writing.** Each branch below was reasoned before any result, so
that the response to the result is fixed in advance and cannot be shaped by it.

Recap: B = frozen NO_RACAF; T = CORN task 3 × 2.0; P = task 1 × 1.3792 (equal added loss mass);
seeds 42/123/2026. Primary endpoint: AUROC of `p_cond_3 = σ(z₃)`, grade 4 vs 3, APTOS validation
(n = 97). Primary contrast T − P. Verdict rules as frozen in the experiment's `PREREGISTRATION.json`.

One step is **outcome-independent** and should run whatever the verdict: **Step LN, the read-only
label-noise and ceiling audit** (NEXT_STEP_RECOMMENDATION.md). Every branch below is interpreted
alongside it, because a noise ceiling near 0.7 would change how any 3-vs-4 effect should be read.

---

## A. SUPPORTIVE (specific, improved, preserved)

**Demonstrated.**
- On the APTOS validation grade-3/4 images, at β₃ = 2 and three seeds, up-weighting CORN task 3
  raises the model's own 3-vs-4 posterior `p_cond_3` by ≥ 0.03, beyond an equal-mass perturbation of
  another task, without breaching QWK / MAE / grade-3 / grade-4-vs-rest guardrails.
- The head's use of E responds to boundary-specific pressure. This supports mechanism 2/3 (head
  optimisation / loss weighting) for the *conditional* 3-vs-4 boundary.

**Not demonstrated.**
- Generalisation: the 97 validation images have been used throughout §8–§14.
- Any change to predicted grades that matters clinically. The decoded grade still requires
  `p_gt_2 > 0.5` first, so the **≤ 2 PDR failure (≈ 55 % of PDR) is untouched by construction**.
- That β = 2 is a good value (one level tested), or that the effect is not a noise-limited artefact
  (see Step LN).

**Adopt task-3 weighting?** Not yet as a production change. It is a validated *mechanism*, not a
validated *model*.

**Keep β = 2 fixed?** Yes. Any other β would need its own pre-registered experiment and is not
justified before external confirmation; a sweep would be tuning on the reused validation set.

**Next: external evaluation on IDRiD (Step EXT)** — inference only on the nine frozen T/P/B
checkpoints, primary contrast T − P on `p_cond_3`, single use of IDRiD.

**Decoder changes still necessary?** Only if a 3-vs-4 decision is ever deployed. For research,
reporting `p_cond_3` as the 3-vs-4 score is sufficient. No decoder change is proposed.

**After external validation.** If EXT confirms, the remaining problem is the ≤ 2 PDR failure, which
task-3 weighting cannot touch. The next question is then *which* PDR images fail and why. That is
answered by Step LN (label disagreement vs genuine PDR) before any PDR-feature work. **No further
training is recommended merely because the result is positive.**

---

## B. NONSPECIFIC (T beats B, but not P)

**Reading.** Perturbing the loss by the same mass on an unrelated task lifts `p_cond_3` as much as
task-3 weighting does. This is the specificity failure the placebo was designed to detect.

**§14's 9/9 `p_cond_3` gain** is then most plausibly the same non-specific effect. It fits the RACAF
observation (+0.075 over the same baselines): the three frozen NO_RACAF runs sit low in the
`p_cond_3` distribution, and *any* retraining tends to land higher.

**Weakened or closed.**
- "Boundary-specific loss pressure improves the conditional head": **closed** for task weighting
  and for the marginal aux loss.
- The head-optimisation reading of the E-probe gap (mechanism 2) is **weakened**. The gap (0.71 vs
  0.64) may reflect the same between-run variance.

**Abandon task weighting?** Yes, for this question.

**Strongest remaining mechanism.** Label noise (7) and missing PDR features (8), which are not yet
distinguished. The estimand finding (5) stands regardless.

**Next: Step VAR, a read-only `p_cond_3` variance and power audit** over all saved models (B ×3,
aux ×9, T ×3, P ×3, RACAF ×3). It quantifies the between-run spread of `p_cond_3` and the minimum
detectable effect with three seeds. Any future 3-vs-4 experiment needs this, and it costs no training.

---

## C. NOT_SUPPORTIVE (T does not improve over B)

**Close training-objective manipulation for this boundary?** Yes. Marginal aux (§14) and
conditional task weighting (E14) have both failed on their pre-registered endpoints. Loss re-weighting
is closed unless a genuinely new reason appears.

**Does the evidence then point to the head / decomposition / architecture?** Partly. The
decomposition is a proper likelihood (NEXT_STEP_AUDIT §3); replacing CORN is not indicated. The one
untested head-level question is whether the *joint training procedure* leaves the head
under-fitted. BEST is selected by val_QWK and the head co-adapts with a 43M backbone at batch 2. A
decoupled head re-estimation on frozen E tests that directly and cheaply.

**Independent / non-chained grade-4 pathway?** Not justified. 4-vs-rest ranking is already 0.88; the
aux loss already added 4-vs-rest pressure without effect; a non-chained head mostly re-packages
the closed threshold route.

**Is the representation already sufficient?** Linearly, E carries ≈ 0.71 (post-hoc). Whether that
is usable by a CORN head, and whether it survives label noise, are exactly Steps cRT and LN.

**New head on frozen E?** Yes, as a **decoupled re-estimation (Step cRT)**: Kang et al. 2020's
classifier re-training applied to CORN, no backbone training. Primary: `p_cond_3` 4-vs-3.

**Hierarchical / nominal / other ordinal alternatives?** Currently unjustified (NEXT_STEP_AUDIT §3):
none specifically addresses a failure in which the tasks-0–2 fit of PDR images is poor.

---

## D. TRADE_OFF (specific and improved, but a guardrail fails)

**Reject, reframe, evaluate externally or abandon?**
- **Reject as a general-model change.** A pre-registered guardrail failed; it is not waived after
  the fact.
- **Do not reframe post hoc** as a "boundary-specific model". That would be endpoint switching.
- A boundary-specific use case (e.g. a secondary 3-vs-4 score only for images already called ≥ 3)
  may be *proposed* as a new pre-registered question, but only after the trade-off is shown to
  generalise.
- **Next: Step EXT** (IDRiD, inference only) with the *same* guardrails. The question is whether
  the trade-off is real out of sample or an artefact of the reused validation set. If the guardrail
  failure replicates externally, **abandon** task weighting. If only the specific gain replicates,
  record a TRADE_OFF confirmed externally and stop; do not tune β to escape the trade-off.

---

## E. INCONCLUSIVE

**Minimum additional evidence.** The honest question is whether *any* feasible experiment on APTOS
could resolve an effect of this size. That depends on:

1. the between-run variance of `p_cond_3` (from Step VAR);
2. the label-noise ceiling (from Step LN).

If three more seeds could not reach adequate power for the observed effect, adding seeds is wasted
compute and the branch should close.

**Next: Step VAR** (read-only, all saved models). It produces a pre-registered power calculation.
Only if that shows a feasible design (e.g. ≤ 6 additional runs for ≥ 80 % power at the observed
mean effect, with the Step-LN ceiling leaving room) may a confirmatory re-run be proposed, and then
as a *new* pre-registered experiment, not an extension. Otherwise: **close task weighting as
unresolved**. **No hyperparameter search. No automatic extra seeds.**

---

## Summary table (selected step per outcome)

| Pending outcome | Next step | Type | Training? |
|---|---|---|---|
| SUPPORTIVE | **Step EXT** — IDRiD external evaluation of frozen T/P/B checkpoints, T − P on `p_cond_3` | external evaluation | no |
| NONSPECIFIC | **Step VAR** — read-only `p_cond_3` variance / power audit; close task weighting | read-only audit | no |
| NOT_SUPPORTIVE | **Step cRT** — decoupled CORN-head re-estimation on frozen E (B checkpoints) vs the jointly trained head | small experiment (head only, no backbone) | head only |
| TRADE_OFF | **Step EXT** with the same guardrails; abandon if the guardrail failure replicates | external evaluation | no |
| INCONCLUSIVE | **Step VAR** → pre-registered power decision; close unless feasible | read-only audit | no |
| *all outcomes* | **Step LN** — read-only label-noise and ceiling audit | read-only audit | no |

Exact protocols are in NEXT_STEP_RECOMMENDATION.md.
