# Candidate Architectures for Grade 3 vs Grade 4

Baseline for every comparison: **NO_RACAF, matched init, weighted CORN, frozen protocol
`improved-multiseed-v1`**, 43,043,336 trainable parameters, frozen runs at seeds 42/123/2026
(BEST `val_QWK` 0.8297 / 0.8068 / 0.8269; grade 4 vs 3 `p_gt_3` AUROC 0.5199 / 0.5836 / 0.6286).
Tensor names follow ARCHITECTURE_AUDIT.md §1: `X5` = stage5_input (B,512,512,8), `P = X5[...,4:8]`
(lesion probabilities MA, HE, EX, SE), `L` = (B,32,32,256), `G` = (B,64,1152), `E` = (B,256),
`z` = CORN logits (B,4).

No numerical scores are used. Comparison is factual.

---

## A. Count-preserving lesion-burden pathway (LBP) into the CORN head

- **Problem solved.** The architecture has no path that carries lesion *amount* to the head:
  Stage 07's softmax attention is cardinality-blind (H1), and the Swin queries see RGB only.
- **Motivating evidence.** E10: 10 image-level burden statistics beat `p_gt_3` by 0.072 and add
  +0.128 to it. Clinical data show burden falls from severe NPDR to PDR (DEEP_RESEARCH.md §2).
  E5: the models already use lesion channels heavily, but for presence-type discrimination.
- **Information used.** `P` only, as fixed image-level statistics (the §11 variable
  definitions), precomputed from the cached canonical maps. Global statistics are exactly
  invariant to the flip/rot90 augmentation.
- **Information currently discarded that it restores.** Image-level lesion amount per channel.
- **Tensors.** `s ∈ ℝ^{B×10}` → transform `log10(s+1e-4)` → z-score with **training-split**
  constants → `Dense(10→32, GELU)` → `Dense(32→32)` = `u` → `Dense(32→4, no bias, zero-init)` =
  `Δz`; `z' = CORN(E) + Δz`.
- **Parameters.** +1,536 (352 + 1,056 + 128): **+0.0036 %**.
- **Cache regeneration.** None. It needs a derived per-image statistics table (read from existing
  lesion caches) and an additive data module that yields it as a 4th input.
- **New labels / Stage 04 change / split change.** None / none / none.
- **Training cost.** Same as a baseline run (the statistics are free at train time).
- **Overfitting risk.** Low from parameters. Moderate from signal: task 3 has only 390 training
  images, so the head can learn a fragile burden rule.
- **Leakage risk.** Standardisation constants must come from the training split only. The
  §11 probe was fitted *inside* validation, so none of its coefficients may be reused.
- **Confounding risk.** The laser-scar confound (H5) passes straight through the EX statistics.
  Any gain could come from parameters or optimisation perturbation, which is handled by the
  **capacity-identical permuted-statistics control**.
- **Literature support.** Cardinality-preserving aggregation (Zhang & Xie 2020; Xu et al.
  2019; Zaheer et al. 2017). Burden reversal at PDR (Esmaeilkhanian et al. 2023). Hybrid
  handcrafted+deep features are common in DR, so this is **not novel as an architecture**.
- **Novelty risk.** High if framed as a new architecture. Low if framed honestly as a controlled
  test of the presence-vs-amount mechanism.
- **Comparison to the 43M model.** Zero-init residual, so at step 0 the logits are bit-identical to
  the baseline, with matched init. Primary contrasts are LBP vs baseline and LBP vs the
  permuted-statistics control.
- **Falsifier.** If LBP does not beat the permuted control on grade 4 vs 3 `p_gt_3` AUROC by the
  pre-registered margin, then explicit burden exposure does not recover the signal end-to-end.

## B. Boundary-aware / asymmetric ordinal objective

- **Problem.** Weak gradient pressure on the 3/4 boundary (L7). CORN task 3 gets 6.5 % of pairs,
  and 3↔4 is "cheap" under QWK-style thinking.
- **Evidence.** E9–E10 plus the analytical L7 numbers. The **§12 experiment is already the
  direct test** (marginal-BCE auxiliary, λ ∈ {0.05, 0.10, 0.20}).
- **Information.** Labels only. Nothing is added to the representation path.
- **Variants.** (i) up-weight CORN task 3 only (conditional `z_3`, no conflict with tasks 0–2);
  (ii) marginal BCE (running); (iii) pairwise 3-vs-4 ranking, which is **infeasible at batch 2**
  (~12 mixed batches per epoch) without a memory bank.
- **Parameters / cache / labels / Stage 04 / split.** 0 / none / none / none / none.
- **Risks.** Grade-3 collapse (the §10 lesson). The marginal-BCE gradient conflicts with CORN
  tasks 0–2 on grade-3 samples (ARCHITECTURE_AUDIT.md L7 note). No new information enters, so it
  can only reweight what E already encodes.
- **Literature.** Class-balanced losses, LDAM and logit adjustment are general tools. No 3v4-specific evidence.
- **Falsifier.** §12's own pre-registered rule.
- **Position.** Do not launch any further B variant until §13 is recorded. If §13 is TRADE_OFF,
  variant (i) is the principled follow-up. If NOT_SUPPORTIVE, objective pressure is not the
  bottleneck and representation is (A/E).

## C. Lesion-conditioned ordinal head (task-specific, lesion-gated CORN)

- **Problem.** Allow each CORN task, especially task 3, to use lesion burden with its own
  sign.
- **Form.** `z_k = w_kᵀE + v_kᵀ s + b_k`, or a FiLM-style gate `z_k = w_kᵀ(E ⊙ (1+γ_k(s)))`.
- **Relation to A.** The linear form is A with an identity MLP. The FiLM form lets burden modulate
  *which* E features each task reads. That is more expressive and more parameters
  (γ: 10→256 per task, about 11k). It has the same information and the same controls.
- **Risks.** FiLM on 390 task-3 samples overfits easily. Multiplicative gating destabilised the
  RACAF-era fusion point (Multiseed §12.2 shows high leverage at fusion points).
- **Position.** Folded into A as its minimal, additive form. FiLM is not pursued first.

## D. Multi-scale lesion representation (lesion pyramid encoder)

- **Problem claimed.** Lesions of different sizes need different receptive fields.
- **Evidence against it being the bottleneck.** Stage 05 is already multi-kernel (3×3, 5×5,
  dilated 3×3) over four scales with per-image adaptive branch weights. E10's winning features
  are scale-free global means and fractions, and nothing indicates a scale deficit.
- **Parameters.** ~0.5–2 M for a separate lesion-only pyramid. Cost ~+10–20 % per epoch.
- **Risks.** Adds capacity without a mechanism aimed at 3v4. It still needs a pooling choice, and if
  that pooling is attention it reintroduces H1.
- **Position.** Not shortlisted.

## E. Spatial lesion-distribution representation (rotation-invariant quadrant/ring statistics)

- **Problem.** The 4-2-1 rule is about *where* haemorrhages are (how many quadrants), not only
  how much.
- **Information discarded now.** The distribution of burden across the retina.
- **Tensors.** Split the FOV at the image centre into 4 quadrants and 3 radial rings. Per channel,
  compute burden per region, then **sort the four quadrant values** (flip/rot90 permutes quadrants,
  so sorted values are augmentation-invariant) and add "number of quadrants with HE fraction > t".
  About 4 × (4 + 3 + 1) = 32 features → MLP → residual logits, as in A. About +2 k parameters.
- **Constraints.** There is no optic-disc or fovea localisation for APTOS, so the quadrants are image-centred, not
  anatomical. Field of view varies by camera. The HE channel (the 4-2-1 lesion) has Dice 0.127,
  so a spatial rule applied to a weak map is noisy. Venous beading and IRMA, the other two limbs
  of 4-2-1, have no channel.
- **Cost / cache / labels / Stage 04 / split.** Same as A: a derived table only.
- **Literature.** ETDRS Report 12 and ICDR define the rule. Quadrant-based lesion algorithms exist,
  but no controlled 3v4 evidence was found.
- **Falsifier.** No gain over A with A's global burden already present.
- **Position.** Shortlisted **as A's follow-up arm**. It is a strict superset of A's information, so
  it only makes sense once A establishes whether amount alone helps. Otherwise a gain could not be
  attributed to spatial distribution.

## F. Additional lesion ↔ global cross-attention module

- **Problem claimed.** Let a lesion token set and the global Swin tokens interact more directly.
- **Evidence.** Stage 07 *already is* a global-queries-local cross-attention whose K/V include
  the lesion channels. RACAF (a fusion-point module) was CLOSED on evidence.
- **Mechanistic objection.** A second softmax cross-attention inherits the same cardinality
  blindness (H1) that is the leading explanation of the loss. It would add capacity (~0.3–1.2 M)
  at the exact kind of fusion point where the project has repeatedly measured null effects.
- **Position.** **Rejected** (see shortlist).

## G. Hierarchical ordinal decision structure (dedicated 3-vs-4 expert)

- **Problem claimed.** Separate the rare, clinically distinct top boundary.
- **Evidence against novelty here.** CORN task 3 *is* a conditional 3-vs-4 classifier trained
  only on y ≥ 3. A hierarchy changes nothing unless the top branch gets its own representation.
  With 390 training images, a separate backbone would overfit badly, and a separate head on E
  is just CORN again.
- **Position.** Not shortlisted. Its one useful idea, reading the *conditional* `σ(z_3)` for 3-vs-4,
  is a decode question tested read-only in P0.1.

## H. PDR-specific lesion channels (NV / IRMA / VH / preretinal / fibrous proliferation)

- **Problem solved.** The pipeline has no channel for any PDR-defining lesion (E11). This is the
  *clinically faithful* fix.
- **Information.** New segmentation outputs from a model trained on FGADR (IRMA, NV) and/or
  Retinal-Lesions (NV, fibrous proliferation, preretinal and vitreous haemorrhage).
- **Tensors.** New channels `P' ∈ ℝ^{512×512×k}` (k = 2–5) appended to `X5`, or burden
  statistics of `P'` via A's pathway.
- **Cost.** A new segmentation model and training on external data. Cache regeneration for 3,662
  images. Stage 05's first convolution changes shape (breaking matched init) unless routed through
  A's pathway only.
- **Labels.** Public pixel labels exist, with no new manual annotation, but NV is tiny and rare
  (Retinal-Lesions has only 62 DR4 images). Domain shift runs IDRiD/FGADR/Retinal-Lesions → APTOS.
- **Risks.** Weak NV segmentation is likely (IDRiD-trained HE Dice is already 0.127). There is no APTOS
  NV ground truth to validate against. Many moving parts, so attributing any gain is hard.
- **Position.** Shortlisted as the **long-horizon** direction. It is justified only if A shows that
  lesion-derived evidence, once exposed, *does* move 3v4. That would indicate the head can use
  better lesion evidence when given it.

### H2 (considered, not shortlisted): pretrained foundation backbone (RETFound-style)
It would probably improve all grades, but it replaces 92 % of the parameters. That is a global
confound, not a 3v4 mechanism.

---

## Shortlist (factual comparison, no scores)

| | A — LBP | E — spatial distribution | H — PDR lesion channels |
|---|---|---|---|
| Mechanism aimed at | amount lost by attention; direction reversal at 3→4 | 4-2-1 quadrant rule | missing PDR-defining lesions |
| Project evidence | direct (E10: exactly these statistics beat the model) | indirect (no spatial variable ever tested) | indirect (E11: channels absent) |
| External evidence | burden falls at PDR; cardinality-preserving aggregation | ETDRS/ICDR definition | ICDR definition; public NV/IRMA datasets |
| New params | +1,536 | about +2 k | new segmentation model plus channels |
| Cache / labels / Stage 04 | none / none / none | none / none / none | regenerate / public external / new model |
| Can isolate causality | yes: capacity-identical permuted control, zero-init parity | only on top of A | poorly |
| Cost | 3 new runs + 3 control runs | +3 runs | weeks |

- **Three strongest:** A, E, H.
- **Recommended for implementation: A (count-preserving lesion-burden pathway)**, gated by the
  read-only Phase 0 in RECOMMENDED_DESIGN.md.
  - It targets the mechanism the audit identifies (amount lost at aggregation; direction reversal).
  - It uses exactly the information already shown to beat the model.
  - It needs no new data, labels or caches.
  - It adds 0.0036 % parameters.
  - It is the only candidate that supports a **capacity-identical, information-destroyed
    control**, so "the architecture works" and "the model gained parameters" can be separated
    cleanly.
- **Low-risk baseline/control:** a *post-hoc stacked head*. Fit logistic regression on the frozen
  baseline's logits `z` plus the 10 lesion statistics, **on the training split**, and evaluate on
  validation (P0.4). No network training at all. End-to-end A must beat this to claim
  anything beyond stacking.
- **Rejected: F (another lesion↔global cross-attention).** It repeats the softmax aggregation that
  is the leading suspect for the loss. It adds capacity at a fusion point where RACAF, C1 and C3
  all produced null effects. No project evidence says the missing ingredient is *more
  interaction*.
