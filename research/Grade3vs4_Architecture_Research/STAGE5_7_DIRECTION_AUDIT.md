# Stage 5–7 representation / fusion — research-direction audit (2026-09-27)

**Status: DESIGN AUDIT ONLY.**
- No code, notebook, checkpoint, cache, label, split, Stage 3/4 or Stage 9–11 component was
  modified, and nothing was trained.
- This file records the audit given in the working session. The corrections requested on
  2026-09-27 are applied, with the scientific findings preserved.
- Research record: §20 (C1 results) and §20.1 (C1 interpretation correction).

---

## 1. What Stages 5–7 extract, transform, aggregate and discard (from the code)

| Stage | Input | What it extracts | What it discards |
|---|---|---|---|
| **5**, adaptive multi-kernel CNN (`local_feature_extraction_model.py`) | 512×512×8: RGB, vessel, MA/HE/EX/SE lesion maps | 1,024 local tokens × 256 channels (32×32 grid) | Absolute lesion counts and areas are only implicitly retained (four 2×2 max-pool stages keep peaks, not totals). Randomly initialised on 2,921 images. |
| **6**, dual-scale Swin (`swin_transformer.py`) | **RGB only, 256×256**, half Stage 5's resolution | 64 tokens × 1,152 channels (8×8 grid) | Fine detail at 256 px. Randomly initialised. The only pathway that never sees the lesion maps. |
| **7**, adaptive cross-attention (`feature_fusion.py`) | Queries = Swin tokens; keys/values = CNN tokens | One softmax cross-attention layer, residual, FFN, then **global average pooling over the 64 query tokens** → E (256) | Softmax weights sum to 1, so absolute lesion extent is normalised away. Average pooling removes spatial distribution. Where attention looks is set by the Stage-6 branch. |

## 2–3. Evidence on where information is lost

**Evidence consistent with the pathway-dominance hypothesis**
- **Stage 5 → E (grade 3 vs 4, research record §13).** Probe AUROC falls from 0.758 to 0.710. The
  drop meets the 0.05 margin in only 1/3 seeds: a modest loss at Stage 7, not established.
- **The RGB pathway is weak.** The Stage-6 probe scores 0.585 (grade 3 vs 4), the lowest probe.
- **The model depends heavily on the lesion maps.** Shuffling them between images collapses QWK from
  about 0.82 to 0.06–0.19; shuffling the vessel map costs about 0.05 (multiseed report §12.3–12.4).
- **C1 (§20).** Changing the Stage-8 output structure did not recover the grade-4 under-triage. The
  dedicated PDR score q, fitted on E, stays below 0.30 in every seed for 16 of the 21 persistent
  failures.
- **Visual review (§15.2, AI-assisted, unconfirmed).** The failures look like preretinal-type
  haemorrhage, laser-scar-like spots and haze. None of these is a Stage-04 lesion class.

**What C1 does and does not show.** C1 showed that modifying the Stage 8 output structure did not
recover the observed Grade-4 under-triage from the frozen Stage 5–7 representation. This motivates
investigation of the upstream representation pathway but does not establish that Stage 5–7 is
definitively the bottleneck.

**What RACAF does and does not show.** RACAF added a direct pooled Swin-to-E pathway (Ĝ) and was null
across six matched runs (C1/C3 controls also null). That is evidence against **that specific tested
RGB/Swin-to-E fusion intervention**. It is not proof that RGB evidence cannot exist upstream, and not
proof that representation fusion is not responsible.

**Competing explanations, both consistent with the evidence above**
1. **Pathway dominance (modality imbalance).** The lesion-map pathway dominates joint training and
   under-trains the RGB pathway. PDR signs outside the four lesion classes therefore never enter E.
2. **Insufficient training signal or data in the RGB branch.** The RGB branch is randomly
   initialised, sees 256 px, and has only 236 grade-4 training images (a small, heterogeneous subset
   with the §15.2 appearances). It cannot learn those signs, with or without a dominant lesion
   pathway.

The existing evidence does not distinguish (1) from (2). **Gate G0 (§6) cannot distinguish them
either.** It can only rule the pathway-dominance hypothesis out, or leave it plausible.

## 4. Literature: what is already published (established work, not our contribution)

| Mechanism | Published examples | Implication |
|---|---|---|
| Image + lesion-segmentation dual path with lesion-guided attention fusion | MSGDA-Net (2024, https://pubmed.ncbi.nlm.nih.gov/39504782/); lesion-guided attention dual path (https://link.springer.com/article/10.1007/s11042-026-21344-0) | Our architecture family itself; no novelty there |
| Lesion tokens / lesion-aware transformers | Lesion-Aware Transformers, CVPR 2021; LLNet 2026 (https://pubmed.ncbi.nlm.nih.gov/41961568/) | Not novel |
| Learnable-query attention pooling replacing GAP | Lesion-query attention pooling (LQAP), 2026 (https://arxiv.org/pdf/2602.10315) | Not novel |
| Anatomy-anchored pooling (optic disc / fovea frame) | PARAGraph 2026 (https://arxiv.org/pdf/2608.08368) | Not novel; needs disc/fovea localisation |
| Count/cardinality-preserving aggregation | Additive MIL, cardinality-preserving attention; attention-MIL revisited 2024 (https://arxiv.org/html/2408.09449v1) | Established |
| Rebalancing a dominant vs a weak modality; modality dropout | OGM-GE (https://pubmed.ncbi.nlm.nih.gov/39321012/), PMR, AGM; BalanceBenchmark 2025 (https://arxiv.org/pdf/2502.10816); balanced multimodal learning 2025 (https://arxiv.org/pdf/2509.02281); DG gradient modulation 2026 (https://arxiv.org/pdf/2603.14175); ModDrop | **An established methodological family.** In the searches made, no work was found that treats segmentation-derived lesion maps as a dominant modality suppressing the raw-image pathway in DR grading. The search was not exhaustive and is **not a novelty claim**. |

## 5. Candidates

| | A: Pooling preserving lesion "mass" in Stage 7 | B: Learnable lesion queries | C: Quadrant / anatomy pooling | D: Separating lesion burden from other image content | E: Pathway rebalancing (lesion-map vs RGB) |
|---|---|---|---|---|---|
| Hypothesis | Softmax + averaging lose lesion extent | GAP dilutes lesions | The 4-2-1 rule needs quadrant information | PDR evidence is burden-independent | The lesion-map pathway suppresses RGB learning (explanation 1 above) |
| Evidence for it | Phase-0 loss in 1/3 seeds; Architecture A gate inconclusive | None specific | None; it targets grade 3, not PDR | Lesion burden falls at PDR (C1: grade-3 median P(≥3) 0.67 vs grade-4 0.57) | The evidence listed under explanation 1 |
| Status in the literature | Established | **Published** | **Published** | Related to disentangled multimodal fusion | **Established family.** Only its specific application to lesion-map dominance here could be differentiated, and that needs further literature checking **and** a successful mechanism gate. |
| Main risk | **Counter-productive:** preserving burden reinforces "more lesions = more severe", which pushes low-burden PDR down | — | Needs a new localiser | Cannot create information absent upstream | Explanation 2 may be the true one |
| Cost | 6 retrains | — | New model | 6 retrains | 6 retrains (about 24–30 T4-hours) |
| Verdict | Reject | Reject | Reject | Weak | **Only remaining conditional direction** |

All candidates keep Stages 3/4 frozen and CORN unchanged.

## 6. Candidate E: gate G0 and the specification that becomes eligible if it passes

### Gate G0 (read-only; pre-specified before it is run)

**Procedure**
- From each of the three frozen NO_RACAF BEST backbones (seeds 42/123/2026), extract for all cached
  training and validation images (2,921 / 730):
  - pooled Stage-5 features (mean over the 32×32 grid, 256-d);
  - pooled Stage-6 features (mean over 64 tokens, 1,152-d);
  - the Stage-7 output E (256-d).
- Pooling is identical to the Phase-0 probes. The extraction is run in one pass per seed, with the
  C1 parity and non-mutation checks.
- Fit the **identical C1 PDR-route fitter** to each representation: the C1 H2 PDR task, i.e.
  - all cached training images, target grade 4;
  - `weighted_corn` class weights, training-only standardisation;
  - L2 = 1e-4, L-BFGS from zero.

  Score validation once, as q.
- **Evaluation population:** validation grade 4 vs grades 0–2 (58 vs 633). This is the C1 endpoint
  population.
- **CIs:** the C1 stratified bootstrap (2,000 resamples, seed 20260927).

**Conditions**
- **G0-A, RGB weakness.** In each seed, Stage-6 AUROC ≤ Stage-5 AUROC − 0.05.
- **G0-B, persistent-21 evidence absence.** For each representation and seed, the median q of the 21
  persistent grade-4 failures (§15) is **not above** the median q of validation grade-2 images.

**Verdict (fixed before running)**

| Verdict | Condition |
|---|---|
| **PASS** | G0-A holds in 3/3 seeds **and** G0-B holds for every representation in every seed |
| **FAIL** | G0-A holds in ≤ 1/3 seeds (no RGB weakness to rebalance), **or** some representation places the persistent-21 above the grade-2 median in 3/3 seeds (the evidence is consistently present upstream, so "absence" is false) |
| **INCONCLUSIVE** | anything else (e.g. G0-A in 2/3 seeds, or G0-B violated in 1–2 seeds) |

**Interpretation limits.**
- G0 does not prove that Stage 5–7 is the bottleneck, and cannot separate explanation 1 from
  explanation 2.
- **PASS** means only that the pathway-dominance hypothesis remains plausible and testable.

### If G0 passes: specification eligible for a separate decision (not approved)

- **Intervention.** During training only, with fixed probability p = 0.5 per sample (the ModDrop
  convention; fixed, not tuned), zero Stage 5's vessel and lesion channels (3–7). Architecture and
  parameter count are unchanged (43,043,336); inference uses the full input.
- **Control.** A matched NO_RACAF retrain at seeds 42/123/2026, not the frozen baselines, which sit
  at the low end of the retrained distribution (§17). Six runs in total.
- **Primary endpoint.** Per seed, Δ AUROC of P(grade ≥ 3), grade 4 vs 0–2 (the C1 endpoint).
- **Mechanism endpoints.** Stage-6 probe AUROC; performance with the lesion channels zeroed;
  persistent-21 decoded ≤ 2; later, DDR grade 3 vs 4 (external, secondary).
- **Guardrails.** QWK ≥ −0.02, grade-3 recall ≥ −0.10, false-urgent rate ≤ +0.02.
- **Decision rule.**
  - **SUPPORTIVE:** mean Δ ≥ +0.03, 3/3 seeds, guardrails hold, and the Stage-6 probe improves in
    3/3 seeds (the mechanism must move).
  - **NOT_SUPPORTIVE:** mean Δ < +0.01, or Δ ≤ 0 in ≥ 2 seeds.
  - Otherwise **INCONCLUSIVE**.
- **Schedule if approved.** Implementation 1 day, training 2–3 days, analysis 0.5 day. With Stages
  9–11 (5–7 days) and documentation (about 4 days), this fits 3 weeks only with no slack, and does
  not fit 2 weeks.

## 7. Overall conclusion (conditional)

- **No Stage 5–7 innovation is currently justified for implementation.** Candidates A–D are
  published, weakly supported, or (A) likely counter-productive for PDR.
- **Candidate E (pathway rebalancing) is the only remaining conditional direction.** It is not
  described as a novel innovation: pathway/modality rebalancing and modality dropout are an
  established family. Only its specific application to lesion-map dominance in this DR architecture
  could provide a differentiated contribution, and that requires further literature verification
  and a successful mechanism gate.
- **G0 must be completed before deciding** whether E deserves implementation.
  - **If G0 fails:** Candidate E is closed, and the project moves on to the remaining stages
    (9–11 and documentation).
  - **If G0 passes:** E becomes **eligible for a separate implementation and pre-registration
    decision**. It is not automatically approved.
  - **If G0 is inconclusive:** the evidence does not justify implementation, and the project moves
    on.

## Current Decision / Next Action

**Run G0 only.** No ModDrop implementation or training until G0 passes. G0 is read-only (about 2
T4-hours plus CPU) and writes no artifact. Its stdout is recorded in the research record, and this
section is then updated with the outcome.

### Outcome (2026-09-28): G0 FAIL (research record §21)

**G0-A held in 3/3 seeds.** Stage 6 RGB AUROC is 0.70–0.75, against 0.88–0.90 for pooled Stage 5, for
grade 4 vs 0–2; every CI on the difference excludes 0.

**G0-B was violated in all 9 representation × seed cells.** The persistent-21 median q sits slightly
above the grade-2 median in every representation, with 11–14 of the 21 above it.

**Under the rule fixed in §6, the verdict is FAIL.** Accordingly:
- **Candidate E is closed.** No ModDrop / pathway-rebalancing implementation or training.
- **No Stage 5–7 innovation direction remains open.** A–D were already rejected above.
- **The next step is the user's decision.** DDR stays reserved for later external evaluation.

**Notes, fixed after the result and not changing it:**
- The grade-2-median bar proved lenient: it detects slight elevation, not recognition as PDR.
- Pooled Stage 5 and E are equal on this endpoint, so Stage 7 aggregation loses no
  grade-4-vs-(0–2) information.
