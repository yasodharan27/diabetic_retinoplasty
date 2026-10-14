# Final research direction — decision record (2026-09-27)

**Status: DESIGN DECISION ONLY.** Nothing is implemented, pre-registered, trained or downloaded.
Before any fit, the protocol below must be frozen in its own `PREREGISTRATION.json`, and it must be
approved. Deadline context: 2–3 weeks remain for the whole project.

---

## 1. Project state this decision rests on

**Pipeline status**
- Stages 1–8 are built.
- The backbone the evidence supports is **NO_RACAF**. RACAF, C1, C3, D6 and the generic-vessel route
  are all closed (record §1–§7).
- Stages 9, 10 and 11 are **template notebooks only** (`colab/notebooks/stage09/10/11_*.ipynb`).
  Together with the documentation, they need about 9–10 working days whatever innovation is chosen.
- MC dropout is feasible: the Stage-7 fusion block has dropout 0.1 (`feature_fusion.py`).

**Policy conflicts (user decision required)**
- `PROJECT_CODE.md` still names RACAF as "the single approved research innovation". The spec must be
  revised deliberately.
- `PROJECT_CODE.md` permits only EyeQ, APTOS and IDRiD. Any direction needing RFMiD,
  Retinal-Lesions, Fundus10K or DDR breaks that policy, on top of the acquisition time.

**The weighted clue**
- It is a structural fact about CORN, not the §15.2 visual impressions.
  - `corn.py:151`: task *k* is trained only on images with grade ≥ *k*.
  - A PDR image can therefore only be called 4 by passing tasks 0–2.
  - Every grade-4 label trains tasks 0–2 as "more severe than moderate".
- Supporting findings:
  - p_gt_2 ranks grade 4 **below** grade 3 (AUROC 0.40–0.50 in every run; §13.2, §14.1).
  - 21/58 PDR images have low p_gt_2 (0.03–0.47) and high p_cond_3 (0.60–0.92), and are decoded ≤ 2 by
    all three seeds (§15).
  - About 32/58 PDR images are decoded ≤ 2 at BEST (§16.1).
  - Thresholding recovers PDR only by collapsing grade-3 recall by 61 % (§10).
- Clinical grounding:
  - ICDR defines PDR **disjunctively** (neovascularisation or vitreous/preretinal haemorrhage),
    independent of NPDR lesion counts.
  - Lesion counts and areas **fall** from severe NPDR to PDR (Esmaeilkhanian 2023).
  - NHS screening grades treated PDR as a separate category (R3S).
- A single monotone chain is the wrong shape for that rule. The hypothesis does **not** depend on
  §15.2 being correct: §15.2 is used only for descriptive failure analysis.

---

## 2. Candidates considered (condensed A–S)

| | **C1 Two-route head** (SELECTED) | C2 Chain-discordance flag | C3 PDR-sign concept channels | C4 Zero-shot retinal vision-language model | C5 Grade-conditional conformal sets | C6 Image-quality vs severity confound |
|---|---|---|---|---|---|---|
| A. Question | Does forcing PDR through CORN's NPDR chain cause PDR under-triage? | Does chain disagreement flag missed PDR? | Do laser-scar and preretinal/vitreous-haemorrhage channels fix PDR → ≤ 2? | Can zero-shot concept scores supply PDR signs? | Can prediction sets guarantee grade-4 coverage? | Does the Stage-1 quality gate remove PDR? |
| B. Hypothesis | Chain structure causes part of the gating; a two-route head on the *same* features reduces it | High p_cond_3 with low p_gt_2 ranks missed PDR above other ≤ 2 calls | Missing channels cause the failures | The model's concepts carry PDR-sign information | Mondrian (per-grade) coverage restores grade-4 coverage | Reject rate rises with grade |
| C. Motivation | §10, §13.2, §15, §16.1 | §15 signature | §15.2 (AI-assisted, unconfirmed) | §15.2 | Stage-9 requirement | §6 `vessel_max` inversion; §15.2 haze |
| D. Novelty | Diagnosis of chain gating, a structure-matched head, and a representation-controlled test | Modest, CORN-specific | Low (NV/laser detectors exist) | Low | **Already done** on APTOS → IDRiD (Hasan 2026) | Low (vitreous haemorrhage making images ungradable is known) |
| E. Closest work | CORN; hurdle models; CF-DRNet; ORDER-DR | DR\|GRADUATE; conformal DR | NV-assisted PDR detection; laser screening | FLAIR | Hasan 2026; Dey 2023 | ICDR / ETDRS ungradable categories |
| F–H. Data | Existing labels; IDRiD is local; no external data | Saved predictions + IDRiD | RFMiD / Retinal-Lesions: **not permitted, must be acquired** | Weights only | Existing | Existing |
| I–J. Change / complexity | New 1,028-param head on frozen E; low | None; very low | New segmenter/classifier; high | New dependency; medium | Stage-9 code; low | Stage-1 inference; low |
| K. Runs | 6 convex head fits; no backbone training | 0 | Several | 0 | 0 | 0 |
| L. Decisive control | Refitted CORN head on the **same E**, same parameter count | Confidence baselines | Capacity-matched, permuted channel | — | Plain split conformal | — |
| M. Endpoint | AUROC of P(grade ≥ 3), grade 4 vs 0–2 (58 vs 635); IDRiD once | PDR among ≤ 2 calls, IDRiD | 58 PDR | 58 PDR | Grade-4 coverage | Reject rate by grade |
| N. Leakage | IDRiD/Stage-04 copies (§18) | Pattern found on APTOS validation, so it must be confirmed externally | IDRiD overlap unknown | **FLAIR pretrained on APTOS, IDRiD, DDR, RFMiD: fatal** | — | — |
| O. Statistics | Single-AUROC SE ≈ 0.03 (half of 3-vs-4); training noise ≈ 0 (deterministic fits) | About 20 missed PDR in IDRiD: very noisy. p_cond_3 is **off-support** on grades 0–2, so the flag may fire on healthy eyes | 58 PDR, no concept ground truth on APTOS | Prompt-dependent | 13–19 PDR per external set | Descriptive |
| P. Time | 4–5 days | 2 days | 8–12 days plus acquisition risk | 3–4 days | 2 days | 1 day |
| S. Thesis role | **Primary contribution** | Fallback (Stage 9) | Future work | Rejected | Stage-9 option, not the innovation | Stage-11 safety analysis |

**Also rejected**
- RETFound backbone swap: no hypothesis behind it.
- Checkpoint selection by a clinical metric: a protocol noise fix, not a contribution.
- More seeds (§17).
- Anything dependent on DDR.

---

## 3. Selected direction — ICDR-structured two-route ordinal head (PDR off the CORN chain)

1. **Contribution (one sentence).** Sequential ordinal models such as CORN force a disjunctively
   defined top grade through the severity chain. This makes PDR with low intraretinal burden look
   like moderate DR, and an output structure that matches the ICDR rule removes that failure on an
   identical representation.

2. **Hypothesis.** Part of PDR under-triage is caused by the output structure itself: grade-4 labels
   supervise the NPDR tasks 0–2, and grade 4 is reachable only through the chain. A head that models
   the probabilities below will rank PDR above grades 0–2 better on the same frozen E:
   - P(4) = q(E), directly;
   - P(k) = (1 − q)·P_NPDR(k) for k ≤ 3.

3. **Why the previous experiments lead here.**
   - The signal is in E (§13).
   - Losses at the head/objective level were non-specific (§14, §16).
   - Thresholding moved along a poor trade-off curve (§10).
   - The failing PDR images show the chain-gating signature (§15).
   - The 3-vs-4 endpoint is unresolvable at feasible cost (§17). This direction therefore moves to the
     58-vs-635 referral endpoint, which §17 identified as having a lower noise floor.

4. **Relation to the earlier "non-chained grade-4 head" (NEXT_STEP_RECOMMENDATION C6).**
   - This is a descendant of C6 with a different mechanism: grade-4 supervision is *removed* from
     the NPDR chain, and the decode follows the ICDR order.
   - It uses a different, threshold-free endpoint.
   - The new evidence that justifies revisiting it is §15, §15.2, §16.1 and §17.

5. **Distinctness from existing DR work.**
   - Hierarchical DR models (e.g. CF-DRNet) split at the bottom of the scale (DR vs no DR).
   - NV-based PDR detection needs pixel-level NV labels.
   - ORDER-DR fuses two branches but has no PDR route.
   - Conformal DR work changes the output sets, not the model structure.
   - No paper was found that removes PDR supervision from an ordinal NPDR chain or tests chain gating
     under representation control. It is **potentially novel, not verified exhaustively**: the search
     was broad but not systematic.

6. **Closest papers and the difference.**
   - CORN (Shi et al. 2023): we identify a failure mode for a disjunctively defined top class and
     change the structure.
   - Hurdle / two-part models: an established statistical idea, *adapted* here with the hurdle at the
     top grade and defined by the ICDR rule.
   - ORDER-DR (2026): fuses branches for referable risk; no structural PDR route.
   - DR|GRADUATE (2020): R4 was the least-accomplished grade. Its reported association of laser marks
     with R4 comes from a search snippet and is unverified.

7. **Data.** Existing APTOS labels and split; IDRiD grading (local, single use).

8. **Architecture change.** Only the Stage-8 head: a new module next to CORN.
   - PDR route: Dense(256→1).
   - NPDR CORN chain over grades 0–3: Dense(256→3).
   - Total **1,028 parameters**, exactly equal to CORN's Dense(256→4).
   - Decode in ICDR order: grade 4 if q > 0.5, otherwise CORN decode over grades 0–3.

9. **Minimal experiment matrix** (3 frozen NO_RACAF BEST backbones × 3 heads):

   | Head | What it is | Role |
   |---|---|---|
   | H0 | Original jointly trained CORN head | Reference only |
   | H1 | CORN head refitted on frozen training E | Control |
   | H2 | Two-route head, same E, same fitter | Treatment |

   - Both refitted heads are L2-regularised logistic regressions (convex, deterministic).
   - The same sqrt-inverse-frequency class weights and one fixed L2 value are used for both.
   - Nothing is tuned.

10. **Controls.**
    - H1 controls for refitting at all. This addresses the pattern that every retrained model beat the
      low B baselines (§17).
    - The parameter count and the representation are identical.
    - H2 − H1 therefore isolates only two things: whether grade-4 labels train the chain, and where
      the PDR decision sits.

11. **Primary endpoint.** Paired Δ AUROC (H2 − H1) of the model's own P(grade ≥ 3), grade 4 vs grades
    0–2, on APTOS validation, scored once.

12. **Secondary endpoints.**
    - AUROC of each arm's PDR score (H1 p_gt_3 vs H2 q), grade 4 vs 0–2.
    - Count of PDR decoded ≤ 2, including the 21 §15 images.
    - Grade-3 recall; false urgent calls on grades 0–2; QWK; MAE.
    - AUROC of P(grade ≥ 3), grade 3 vs 0–2.
    - Failure analysis by §15.2 appearance group (descriptive).
    - 3-vs-4 reported descriptively only.

13. **Leakage controls.**
    - Heads are fitted on training E only.
    - IDRiD is run in a single pass, after excluding:
      - Stage-04 exact and pixel copies, including IDRiD_088/089/091;
      - conflicting within-IDRiD duplicates (§18).
    - No threshold, fitting constant or checkpoint is chosen on IDRiD.

14. **Statistical decision rule** (to be pre-registered before any fit).

    | Verdict | Condition |
    |---|---|
    | SUPPORTIVE | mean Δ ≥ +0.03, Δ > 0 in 3/3 seeds, all guardrails hold |
    | TRADE_OFF | primary met, a guardrail fails |
    | NOT_SUPPORTIVE | mean Δ < +0.01, or Δ ≤ 0 in ≥ 2/3 seeds |
    | INCONCLUSIVE | otherwise |

    - Guardrails (H2 vs H1):
      - grade-3 recall ≥ −0.10;
      - grade-3-vs-(0–2) AUROC ≥ −0.03;
      - QWK ≥ −0.02;
      - false urgent calls on grades 0–2 ≤ +0.02.
    - IDRiD: CONFIRMED with the same bar; DIRECTIONAL if 3/3 seeds are positive but below the bar;
      NOT CONFIRMED otherwise.
    - **Claiming the contribution requires APTOS SUPPORTIVE and IDRiD at least DIRECTIONAL.**
    - **Step 0 gate** (read-only, saved B per-sample predictions). Proceed only if:
      - the baseline AUROC of p_gt_2 (grade 4 vs 0–2) is ≤ 0.95, i.e. there is room to improve;
      - the projected minimum detectable effect is ≤ 0.05.
      - Otherwise go to the fallback.

15. **Expected compute.**
    - About 1–2 T4-hours to extract E for all grades (the Phase-0 extraction covered grades 3/4 only).
    - CPU minutes for the fits.
    - About 2 T4-hours for the IDRiD pass (the existing IDRiD evaluation notebook pipeline can be reused).

16. **Implementation time: 4–5 days.**
    - Step 0 + pre-registration: 1 day.
    - Head module + tests: 1 day.
    - E extraction + fits + APTOS evaluation: 1 day.
    - IDRiD pass + failure analysis: 1 day.
    - Record entry: 0.5–1 day.

17. **Validation/documentation time.**
    - Stage 9 (MC dropout on the chosen head): ~2.5 days.
    - Stage 10 (Grad-CAM++, plus the route probability q as an explanation): ~2.5 days.
    - Stage 11 (reuses the IDRiD pass): ~2 days.
    - Thesis write-up: 3–4 days.

18. **What can go wrong.**
    - E from training images is over-fitted by the backbone, so fits may be over-confident. Both arms
      share this: the contrast stays fair, but absolute levels may be off.
    - H2 may simply call more grade-4s. The false-urgent and grade-3 guardrails catch this.
    - There may be no headroom (Step 0 catches this).
    - Gating may live in E, not in the head. The result would then be NOT_SUPPORTIVE, which is itself
      informative: it points to the representation (C3) as future work.

19. **Fallback.**
    - If Step 0 fails: C2, the chain-discordance flag. It is pre-registered on APTOS, confirmed once on
      IDRiD, and becomes a Stage-9 component with no training.
    - If H2 is NOT_SUPPORTIVE: the controlled negative, together with the §13/§17 diagnostic
      programme, becomes the documented finding.
    - Either way the deployed pipeline is complete. Pre-register H2 on the seed-42 backbone if the claim
      holds, and H0 otherwise.

20. **Completion within the deadline.**
    - **3 weeks:** yes. About 5 innovation days, 7 days for Stages 9–11 and 4 for documentation, i.e.
      ~16 working days plus a 2-day buffer.
    - **2 weeks:** only with descoping. Cut Stage 10 to Grad-CAM++ (drop SHAP and attention rollout)
      and compress the documentation.

---

## 4. Frozen, cited, forbidden, defensible

**Stays frozen**
- Stages 1–7, including Stage-04 and LWNet.
- The three NO_RACAF BEST weights; caches; split.
- All existing results and reports.

Only a new Stage-8 head is added. CORN stays as the control.

**Results to cite as motivation**
- Research record §8–10, §11/§13.2, §14.1, §15/§15.2, §16/§16.1, §17, §18.
- Multiseed report §12.3–12.4 (the lesion channels dominate the grading).

**Claims we must NOT make**
- That the 3-vs-4 boundary improved.
- That the failures are treated PDR, or that label noise is ruled out.
- Clinical readiness, or state of the art.
- That the result generalises beyond this frozen representation.
- That IDRiD is fully independent (it is the same source as the Stage-04 training images).

**What makes it defensible**
- Pre-registration.
- Identical representation and parameter count in both arms.
- Deterministic fits, which remove the training noise that sank §14/§16.
- A threshold-free primary endpoint on a larger population.
- An adjacent-grade guardrail (the §10 lesson).
- A single external confirmation.
- A clinical rationale (ICDR definition, R3S) that exists independently of our data.

---

## 5. Sources (literature review, 2026-09-27)

- Shi X., Cao W., Raschka S. (2023). CORN. https://arxiv.org/abs/2111.08851
- Esmaeilkhanian H. et al. (2023). Lesion frequency/area vs DR severity. https://pmc.ncbi.nlm.nih.gov/articles/PMC10587246/
- NHS DESP grading definitions (R3A / R3S). https://xerte.cardiff.ac.uk/USER-FILES/8618-sopkp-Nottingham/media/Grading_definitions_for_referrable_disease_2017_new_110117.pdf
- ORDER-DR (2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13477890/
- Hasan, Nayeem, Alghamdi (2026). Ordinal conformal prediction for DR (APTOS → IDRiD, Messidor-2). https://pmc.ncbi.nlm.nih.gov/articles/PMC13512306/
- Dey P. et al. (2023). Conformal prediction sets for ordinal classification. NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2023/file/029f699912bf3db747fe110948cc6169-Paper-Conference.pdf
- Silva-Rodríguez J. et al. FLAIR (pretraining list includes APTOS, IDRiD, OIA-DDR, RFMiD). https://arxiv.org/abs/2308.07898 · https://github.com/jusiro/FLAIR
- Araújo T. et al. (2020). DR|GRADUATE. MedIA 63. https://arxiv.org/pdf/1910.11777 (laser-mark statement from search snippet, unverified)
- Segmentation-assisted FCN for PDR (NV segmentation). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9821182/
- Autonomous screening for laser photocoagulation (BJO). https://pmc.ncbi.nlm.nih.gov/articles/PMC11137462/
- Uncertainty-aware ordinal cross-dataset DR (2026). https://arxiv.org/html/2602.10315v1
- Dual-branch DR network. https://arxiv.org/html/2308.09945v2
- Park, Moon, Jung (2025). Lesion-centred vs severity-based DR. https://pmc.ncbi.nlm.nih.gov/articles/PMC12191321
- Kang J., Kottas A. Structured mixture of continuation-ratio logits (statistical model). https://arxiv.org/abs/2211.04034
- Concept explanation methods for DR (2024). https://arxiv.org/html/2410.03188v1
- DL DR grading meta-analysis (2026). https://pmc.ncbi.nlm.nih.gov/articles/PMC13414145/
- Ultrawide-field vs two-field imaging, ungradable eyes. https://pmc.ncbi.nlm.nih.gov/articles/PMC13067663/
- Earlier sources for this folder: `SOURCES.md`, `NEXT_STEP_SOURCES.md`.

---
---

# Addendum A — Final re-audit after DDR was confirmed obtainable (2026-09-27)

**Status: DESIGN DECISION ONLY.** Nothing is downloaded, trained or modified. This addendum
supersedes §1–§5 above wherever the two differ; §1–§5 are left unchanged.

**New fact.** The user confirmed that the official OIA-DDR distribution is obtainable. It holds
13,673 images; grades 0/1/2/3/4/ungradable = 6,266 / 630 / 4,477 / 236 / 913 / 1,151. DDR is **not yet
downloaded**.

**Headline.** C1 is kept, in its **Stage-8-only** form. Stages 5–7 are **not** changed. DDR changes
C1's role, not the selection: C1 goes from a result confirmed on IDRiD to one that can be confirmed
with statistical power on DDR. The **novelty claim is narrowed**, because the output structure
itself is an established sequential/hurdle variant.

## A.1 What was re-checked

- Stage 7 as implemented (`feature_fusion.py`): Swin tokens are the queries, CNN tokens are the
  keys/values, softmax MHA is followed by global pooling to give E.
- CORN as implemented: `corn.py:151` masks task *k* to grades ≥ *k*.
- §13.3 locates the loss:
  - the signal is present in Stage 05 (probe AUROC 0.76) and in E (0.71);
  - the loss is downstream of E, in 3/3 seeds;
  - a Stage-7 aggregation loss appears in only 1/3 seeds;
  - Stage 06 is the weakest source (0.585).
- **No experiment shows a representation-level PDR failure.**

## A.2 Novelty threats (searched specifically)

| Work | Separates PDR from NPDR? | Top-grade hurdle / two routes? | Same probability structure? | Clinically disjunctive? |
|---|---|---|---|---|
| Sequential / continuation-ratio models (Tutz); hurdle models | n/a | **Yes: split trees and hurdles are established** | **Yes: C1 is a known sequential-model variant** | No |
| Ord2Seq (ICCV 2023) | No | Recursive binary splits (dichotomic tree) | Close relative | No |
| CORN | No | No (forward chain) | This is the baseline | No |
| D-ODR (2026) | No | No; enforces *monotone* progression, the opposite assumption | No | No |
| Coarse-to-fine / two-stage DR; dual-branch DR; ORDER-DR | Bottom split (DR vs no DR) or branch fusion | No | No | No |
| Neuro-symbolic DR (Urooj et al. 2025) | Rule features; PDR route not specified | No | No | Partly (the rules come from guidelines) |
| PARAGraph (2026) | No; relies on global context when NV is absent | No | No | No |
| NV-segmentation PDR detection | Yes | n/a | No | Yes, but needs NV pixel labels |

**Novelty verdict.**
- The **structure is not novel.**
- Not found published: the combination of
  - the chain-gating diagnosis in a sequential ordinal DR model;
  - removing grade-4 supervision from the NPDR chain on clinical grounds (ICDR disjunctive PDR,
    NHS R3S, lesion burden falling at PDR);
  - a representation-controlled test with external confirmation.
- Classification: **adaptation of an established technique + a new mechanism-driven question and
  test.** It is defensible only if it is claimed in exactly those terms, never as a new architecture
  or ordinal family.

## A.3 Candidates (23 criteria, condensed)

| # | C1-A: two-route head on frozen E | C1-C: two-route head, end-to-end | PDR route reading Stage-6 G | Arch A: burden pathway into Stage 7 | PDR-sign channels (Stage 4/5) | Pretrained or DDR-trained representation | Chain-discordance flag |
|---|---|---|---|---|---|---|---|
| 1–2 Question / hypothesis | The chain structure causes PDR gating | The same, plus E being shaped by chain supervision | PDR evidence lives outside the lesion-dominated E | Softmax pooling loses lesion burden | PDR signs have no channel | A better representation fixes PDR | Chain disagreement flags missed PDR |
| 3 Evidence | §10, §13.2, §15 (21 gated), §16.1 | Same | §6 `vessel_max`, §15.2 | Phase-0 gate INCONCLUSIVE; aggregation loss 1/3 seeds | §15.2 (AI-assisted only) | none specific | §15 |
| 4 Change | New Stage-8 head (1,028 params) | Stage 8 + retrain Stages 5–8 | Stages 7/8 | Stage 7 | Stages 4/5 + new model | Stage 6 or all | None (Stage 9) |
| 5 Frozen | Stages 1–7 and the backbones | Stages 1–4 | 1–6 | 1–6 | 1–3 | 1–4 | All |
| 6–8 Novelty vs closest work | Adaptation + new question (A.2) | Same claim, more confounded | Low | Low (MIL / cardinality work) | Low (NV/laser detectors exist) | None (backbone swap) | Modest |
| 9–12 Data / APTOS / IDRiD / DDR | APTOS enough; IDRiD usable; **DDR strengthens it greatly** | Same | Same | Same | **Needs RFMiD / Retinal-Lesions; DDR's masks are MA/HE/EX/SE only** | DDR would become training data, breaking the external rule | APTOS + external |
| 13 Leakage | Low (heads fitted on training E only) | Low | Low | Low | High (new sources) | FLAIR-type models were pretrained on APTOS/IDRiD | The pattern was found on APTOS validation |
| 14–15 Cost / experiments | 6 convex fits, about 2 GPU-h for E extraction | 6 retrains, about 1.5–2 days | 6 retrains | 6 retrains | New dataset + model | Large | 0 |
| 16 Decisive ablation | vs CORN head refit on the same E | vs a matched CORN retrain (affected by the "baselines sit low" confound, §17) | vs E-only route | vs a same-size control | Permuted channel | — | vs confidence baselines |
| 17–18 Endpoint | AUROC of P(grade ≥ 3), 4 vs 0–2 | Same | Same | Same | 58 PDR | QWK | Flag AUROC |
| 19 Failure mode | Gating lives in E | Checkpoint-selection noise (±0.14, §16.1) | G is the weakest signal (0.585) | Premise already weakened | Unvalidated concepts | Not a hypothesis | Off-support p_cond_3 on grades 0–2 |
| 20–22 Days / fits 2 wk / fits 3 wk | 4–5 / yes / yes | 7–9 / no / marginal | 7–9 / no / marginal | 7–9 / no / marginal | 12+ / no / no | 10+ / no / no | 2 / yes / yes |
| 23 Defensible contribution | **Yes** | Yes, but confounded and late | Weak | Weak | Not in time | No (rejected class) | Fallback |

Also rejected: Stage-7 attention/fusion variants (generic attention); another reliability gate,
loss weight or threshold trick; multi-source training for domain generalisation (would make DDR
training data).

**Level comparison.**

| Level | Assessment |
|---|---|
| A (Stage 8 only) | Isolates exactly the demonstrated mechanism; deterministic, capacity-matched, cheap. |
| B (Stages 5–7) | No demonstrated representation failure to target; each option lacks evidence, needs unavailable labels, or is a rejected class of change. |
| C (both) | Tests more, but reintroduces the retraining confound and checkpoint noise that sank §14/§16, at 2–3× the time. |

**Level A gives the most scientific value per day.** A limitation stays attached: a negative Level-A
result does not rule out Level C.

## A.4 The nine critical questions on C1

1. **Is the hypothesis sound?** Yes.
   - ICDR defines PDR by NV or vitreous/preretinal haemorrhage, regardless of NPDR lesion count.
   - Lesion burden falls from severe NPDR to PDR (Esmaeilkhanian 2023).
   - In CORN, grade-4 labels train tasks 0–2 as "more than moderate", and task 3 never sees grades 0–2.
2. **Is it novel enough?** Only as stated in A.2.
3. **Has an equivalent PDR hierarchy or hurdle been published?** The statistical structure, yes. In DR
   with this rationale and test, none was found; the search was broad but not exhaustive.
4. **Is Stage 8 alone adequate?** It tests the decoding and head-level supervision parts of the
   mechanism, not whether E was shaped by chain supervision. It is a conservative test: a positive
   result is sufficient, a negative result is not conclusive.
5. **Is there headroom?** Likely, but unverified: 21/58 PDR are consistently gated, and about 32/58
   are decoded ≤ 2 at BEST. Step 0 measures it before anything is built.
6. **Can the primary endpoint use grade 4 vs 0–2?** Yes. Hanley–McNeil SE of a single AUROC at 0.88:
   - APTOS (58 vs 635): ≈ 0.030;
   - IDRiD (53 vs ~340): ≈ 0.03;
   - DDR official test split (~275 vs ~3,400, pro-rata estimate, unverified): ≈ 0.014.
7. **Can DDR confirm cleanly?** Yes, if it passes the audit (different country, cameras and graders).
   Caveats:
   - the ungradable class may contain PDR obscured by vitreous haemorrhage, so excluding it biases the
     PDR population;
   - DDR has unusually few grade-3 images (236 vs 913 grade-4), which suggests different grading
     conventions;
   - Stage 04 was trained on IDRiD, so lesion maps degrade under the camera shift. Both arms are
     affected equally, so the paired contrast holds.
8. **Would a Stage 5–7 change be more meaningful?** Not on current evidence and not within the deadline.
9. **Smallest decisive experiment:** Step 0 (read-only), 6 deterministic head fits (H1, H2 × 3 frozen
   backbones), and one external pass.

## A.5 Final decision

1. **Selected innovation.** An ICDR-ordered two-route ordinal head (Stage 8), with PDR taken off the
   CORN NPDR chain, tested on a frozen representation and confirmed externally.
2. **Contribution.** CORN's chain structure makes PDR with low intraretinal burden look like moderate
   DR ("chain gating"). An output head ordered as ICDR orders it (PDR first, then NPDR severity)
   reduces that under-triage on an identical representation, and the effect is confirmed externally.
3. **Hypothesis.** H2 separates grade 4 from grades 0–2 better than H1 on the same E. The two heads
   differ only in:
   - whether grade-4 labels supervise NPDR tasks 0–2;
   - whether the PDR decision sits outside the chain.
4. **Why the experiments lead here.**
   - Signal present in E (§13).
   - Loss downstream of E (§13.3).
   - p_gt_2 inversion (§13.2, §14.1) and the gated PDR images (§15, §16.1).
   - Thresholding only trades one error for another (§10).
   - Objective manipulations were non-specific (§14, §16).
   - The 3-vs-4 endpoint is unresolvable (§17).
5. **Novelty.** A.2: an adaptation of an established technique plus a new mechanism-driven question
   and test, not a new model family.
6. **Stages that change.** Only Stage 8. Stages 5, 6 and 7 do not change.
7. **Architecture.**
   - PDR route: Dense(256→1) → q.
   - NPDR CORN chain over grades 0–3: Dense(256→3).
   - 1,028 parameters, the same as CORN's Dense(256→4).
   - P(4) = q; P(k) = (1 − q)·P_NPDR(k) for k ≤ 3.
   - Decode: grade 4 if q > 0.5, otherwise CORN decode over grades 0–3.
8. **Frozen.**
   - Stages 1–7 and the three NO_RACAF BEST backbones (seeds 42/123/2026).
   - Caches, split, and all previous results.
9. **Training data.** Training E from the APTOS training split, 2,929 images.
   - NPDR chain: grades 0–3 (2,693 images).
   - PDR route: all 2,929.
   - L2-regularised logistic fits, one fixed L2 value, sqrt-inverse-frequency class weights, no tuning.
     H1 is fitted identically.
10. **Evaluation datasets.**
    - APTOS validation (730): development decision.
    - External confirmation:
      - Scenario A: DDR official test split (gradable only), with IDRiD as a secondary external set;
      - Scenario B: IDRiD grading train+test, with the §18 exclusions.
11. **Role of APTOS.** Fitting on the training split; one pre-registered primary decision on the
    validation split, which is not untouched and is stated as such.
12. **Role of IDRiD.** One pass; confirmatory only in scenario B. It is not independent of Stage 04.
13. **Role of DDR (if acquired).**
    - External evaluation only, once.
    - Never used to fit, select a threshold, pick a checkpoint or tune.
    - It also hosts powered **mechanism replications** that do not depend on C1 succeeding:
      - is p_gt_2 inverted on 3 vs 4;
      - does p_cond_3 beat p_gt_3;
      - what is the gating rate among PDR images.
    - The DDR 3-vs-4 contrast (~71 vs ~275 in the test split, estimate) is a secondary endpoint.
14. **Leakage controls.**
    - DDR:
      - SHA-256 plus the calibrated pixel-thumbnail method (§18), against APTOS (3,662), IDRiD (597)
        and DDR's internal duplicates;
      - confirm ICDR coding; exclude grade 5 (ungradable);
      - smoke-test Stages 2–4 on a sample and record the field-of-view failure rate;
      - describe domain shift: H0 per-grade performance, the Stage-1 quality distribution, and
        lesion-map statistics against APTOS.
    - IDRiD: the §18 exclusions (Stage-04 copies including IDRiD_088/089/091; conflicting duplicates).
    - **DDR deadline: if the audit has not passed by the end of Day 2, switch automatically to
      scenario B.**
15. **Experiment matrix.** 3 backbones × {H0 original head (reference), H1 CORN refit (control), H2
    two-route head (treatment)}, scored on APTOS validation and one external set.
16. **Controls and ablations.**
    - H1 controls for refitting and capacity (identical parameter count and identical E).
    - H0 is reported for continuity.
    - No further ablations.
17. **Primary endpoint.** Paired Δ (H2 − H1) in the AUROC of each model's own P(grade ≥ 3), grade 4 vs
    grades 0–2.
18. **Secondary endpoints.**
    - AUROC of the PDR score (q vs p_gt_3), grade 4 vs 0–2.
    - PDR decoded ≤ 2, including the §15 set of 21 images.
    - Grade-3 recall; false urgent calls on grades 0–2.
    - AUROC of P(grade ≥ 3), grade 3 vs 0–2.
    - QWK; MAE.
    - DDR 3 vs 4 (secondary only).
    - Failure analysis by §15.2 appearance group (descriptive).
    - Mechanism replications.
19. **Decision rule** (pre-register before Step 0 results are seen).

    | Set | Verdict | Condition |
    |---|---|---|
    | APTOS | SUPPORTIVE | mean Δ ≥ +0.03, Δ > 0 in 3/3 seeds, guardrails hold |
    | APTOS | NOT_SUPPORTIVE | mean Δ < +0.01, or Δ ≤ 0 in ≥ 2/3 seeds |
    | APTOS | TRADE_OFF / INCONCLUSIVE | as in §3 above |
    | DDR | CONFIRMED | mean Δ ≥ +0.02, 3/3 positive, paired-bootstrap 95 % CI of the mean excludes 0 |
    | IDRiD | CONFIRMED | mean Δ ≥ +0.03, 3/3 positive |
    | IDRiD | DIRECTIONAL | 3/3 positive, below the bar |

    - Guardrails (H2 vs H1):
      - grade-3 recall ≥ −0.10;
      - grade-3-vs-(0–2) AUROC ≥ −0.03;
      - QWK ≥ −0.02;
      - false urgent calls on grades 0–2 ≤ +0.02.
    - **The claim requires APTOS SUPPORTIVE plus external confirmation.** In scenario B, IDRiD
      DIRECTIONAL is accepted, with the claim stated as weaker.
    - **Step 0 gate.** Proceed only if the baseline AUROC of p_gt_2 (grade 4 vs 0–2) is ≤ 0.95 and the
      projected MDE is ≤ 0.05; otherwise go to the fallback.
20. **Compute** (estimates; throughput not measured).
    - E extraction on APTOS: ~1–2 T4-h.
    - Head fits: CPU minutes.
    - IDRiD pass: ~1–2 T4-h.
    - DDR test pass: ~3–5 T4-h, plus downloading and staging (the Hugging Face mirror lists 15.9 GB;
      the official size is unverified).
21. **Calendar.**

    | Day | Work |
    |---|---|
    | 1 | Step 0, pre-registration, head module + tests; DDR download and audit in parallel |
    | 2 | APTOS E extraction, fits, primary decision; DDR audit deadline |
    | 3 | External pass (IDRiD always; DDR if the audit passed) |
    | 4 | Failure analysis, mechanism replication, record entry |

    Then Stage 9 (2–3 days), Stage 10 (1–2 days), Stage 11 (2 days, reusing the external passes) and
    documentation.
22. **2-week version.**
    - Innovation: 4 days (DDR test split only if the audit passes by Day 2).
    - Stage 9: MC dropout only, 2 days.
    - Stage 10: Grad-CAM++ only, 1 day.
    - Stage 11: 1.5 days.
    - Documentation: ~2 days.
23. **3-week version.**
    - Innovation: 5 days, including the full DDR audit and domain-shift description.
    - Stage 9 with calibration and referral analysis: 3 days.
    - Stage 10: 2 days.
    - Stage 11: 2 days.
    - Documentation: 5 days.
    - Buffer: 2 days.
    - No extra experiments. **C1-C (end-to-end training) stays future work in both versions.**
24. **If DDR fails.** Scenario B, confirmed on IDRiD. The contribution holds with a weaker external
    claim. No waiting on DDR beyond Day 2.
25. **Negative result.**
    - APTOS NOT_SUPPORTIVE, or no external confirmation.
    - Conclusion: "output structure is not the limiting factor on this representation; the PDR
      failure is representational." Still a clean, documented finding, especially with the DDR
      mechanism replications.
    - If Step 0 fails: fallback to the chain-discordance flag (Stage 9, no training).
26. **Claims we can make (if the rule passes).**
    - On a fixed representation, an ICDR-ordered two-route head reduces PDR under-triage relative to a
      capacity-matched CORN head, confirmed on DDR or (scenario B) directionally on IDRiD.
    - The CORN chain-gating and estimand findings replicate externally, if the DDR secondaries show it.
27. **Claims we must not make.**
    - A new ordinal model family.
    - State of the art, or clinical readiness.
    - A 3-vs-4 improvement, except as a DDR secondary result.
    - That the failures are treated PDR, or that label noise is ruled out.
    - That end-to-end training would behave the same way.
    - That DDR PDR is representative (ungradable images were excluded).
    - That IDRiD is independent of Stage 04.

**User decisions before implementation.**
- Revise `PROJECT_CODE.md`: it still names RACAF as the one innovation, and DDR must be added as an
  evaluation-only dataset.
- Approve the pre-registration.

## A.6 Additional sources (re-audit, 2026-09-27)

- Ord2Seq (ICCV 2023). https://arxiv.org/abs/2307.09004
- Directed Ordinal Diffusion Regularization (2026). https://arxiv.org/abs/2602.21942
- Urooj M. et al. (2025). Single Domain Generalization in DR: a neuro-symbolic approach. https://arxiv.org/abs/2509.02918
- PARAGraph (2026). https://pith.science/paper/2608.08368
- Biswas et al. (2025). Interpreting DNNs in DR grading vs human decision criteria. Life. https://pmc.ncbi.nlm.nih.gov/articles/PMC12472159/
- DDR paper record (Li T. et al. 2019, Information Sciences 501). https://www.researchgate.net/publication/333636771_Diagnostic_Assessment_of_Deep_Learning_Algorithms_for_Diabetic_Retinopathy_Screening
- DDR Hugging Face mirror (card counts inconsistent with the official figures; do not rely on it). https://huggingface.co/datasets/ctmedtech/DDR-dataset
- GDRNet (MICCAI 2023). https://link.springer.com/chapter/10.1007/978-3-031-43904-9_42 · https://github.com/chehx/DGDR
- CLIP-DR (MICCAI 2024). https://papers.miccai.org/miccai-2024/141-Paper1493.html
- Robust DR grading with ordinal regression (2026). https://arxiv.org/html/2604.17341v1
