# Full Stage 3–8 pipeline architecture audit (2026-10-01)

This is a design audit only: nothing was implemented or trained.

**Binding requirement (user, 2026-10-01).**
- Stage 3 (vessel) and Stage 4 (lesion) stay in the end-to-end pipeline, and their outputs are consumed
  by the classification stages.
- RGB stays available.
- Stages 3–7 may be retrained or replaced.
- Target: overall 5-class grading.

## 1. Project evidence that constrains the design

| Fact | Source | Consequence |
|---|---|---|
| **Stage 4 is weak.** IDRiD test Dice: EX 0.357, HE 0.127, SE 0.024, MA 0.017 (mean 0.131). Attention U-Net trained from scratch on 54 IDRiD images | `SEGMENTATION_ARCHITECTURE.md` §3.5 | The lesion stream carries poor lesion evidence. **The frozen Stage 4 is a genuine upstream limit.** |
| Stage 3 = pretrained LWNet (DRIVE), inference only; established quality; domain-shift caveat | `SEGMENTATION_ARCHITECTURE.md` §1.2 | No evidence that vessel *quality* limits grading |
| The old pipeline depended almost entirely on the lesion stream: permuting it cost ΔQWK 0.63–0.74; permuting the vessel map cost 0.04–0.07, mostly grade-0 recall | Multiseed §12.3–12.4 | Lesion maps are used heavily when the RGB path is weak; vessel information is modest |
| Random-init multi-kernel CNN + random-init Swin + cross-attention (Stages 5–7) is far below a pretrained RGB CNN: P beats NO_RACAF by QWK +0.096, AUROC +0.050 | §22 | **Replace Stages 5–7 with a pretrained backbone** |
| Stage 3/4 maps concatenated into P's *stem* (PL, zero-init extra kernels): QWK +0.0007, AUROC +0.0076; grade-3 recall −0.22 (mostly one seed) | §22 | Stem concatenation adds almost nothing; the fusion point matters |
| Lesion burden is already linearly recoverable from P's features (§§24–27); no pooling bottleneck (§23); CORN decoding is faithful (§25) | §§23–27 | Priors must add *spatially specific* evidence, not burden totals; keep CORN |
| P overfits (training QWK 0.99 vs validation 0.91); checkpoint instability; 24 % label disagreement | §§22, 24, 15 | Regularisation (EMA, soft targets) is justified *separately* from the architecture |
| P recipe: batch 2, no stochastic depth, no EMA, constant LR 1e-4, mild augmentation | P pre-registration | — |

Closed and not re-proposed: RACAF / reliability gating, stem concatenation (PL), the old random-init
Stage 7 cross-attention, G0 / modality rebalancing, 1024 px, RETFound, DINOv2 complementarity, and
grade-3/4-only mechanisms.

## 2. Literature that bears on the design

- **Lesion segmentation quality is far above ours when using pretrained encoders plus more masks.**
  - Public pixel-level lesion sets: IDRiD 81, DDR 757, FGADR 1,842, Retinal-Lesions 1,593 (also NV,
    preretinal and vitreous haemorrhage), MAPLES-DR 200.
  - An ImageNet SE-ResNet-101 U-Net was the best configuration in a cross-dataset study, and combining
    datasets plus weight averaging improves generalisation (Playout & Cheriet 2024).
  - State of the art reaches IDRiD AUPR about 0.89 for EX, and MA AUPR about 0.53 / SE about 0.73
    (DB U-Net).
- **Segmentation-guided grading is established.** Examples:
  - lesion attention from segmentation into the classifier (Zhou et al., CVPR 2019);
  - joint multi-lesion segmentation and grading (DRG-Net 2022; AAAI 2020 multi-task);
  - multi-lesion segmentation-guided attention (MSGDA-Net 2024);
  - lesion-map + image cross-feature fusion with pretrained Swin-T, **+2.4 % accuracy** over image-only
    (FGADR-trained lesion maps, EyePACS referable; arXiv 2411.03618);
  - vessel-gated multi-scale fusion (VesselMetaKAN 2026).
- **Typical reported gains are +1–3 %** accuracy/κ, mostly with stronger lesion models than ours and
  often weak baselines. Expect modest gains over a strong pretrained CNN.

## 3. Three complete Stage 3–8 architectures

### Architecture 1 — Pretrained RGB backbone with multi-scale gated prior injection (**recommended first**)

- **Stage 2:** unchanged; RGB at 512 px.
- **Stage 3:** LWNet, **frozen, retained** → vessel probability map (512², 1 channel).
- **Stage 4:** lesion segmentation → 4 probability maps (MA, HE, EX, SE).
  - First experiment: the *current* frozen model.
  - Later: **retrained** (Architecture 1b, §5).
- **Stage 5 (new) — prior encoder:** a small CNN (about 1–2 M params) on the 5 prior channels. It
  produces feature maps at the ConvNeXt stage resolutions (128², 64², 32², 16²). Lesion and vessel
  channels are *learned* into features, not merely concatenated.
- **Stage 6 (new) — RGB encoder:** ImageNet ConvNeXt-Tiny (the P backbone, proven). RGB keeps its
  pretrained path intact.
- **Stage 7 (new) — fusion.** At each ConvNeXt stage output:
  F_s ← F_s + γ_s · Conv1×1(Prior_s)  and  F_s ← F_s · (1 + α_s · σ(Conv(Prior_s))).
  - γ_s and α_s are **zero-initialised**, so the untrained model is exactly P.
  - The fusion is **spatially aligned before any pooling**: lesion location is not destroyed.
  - Then global average pooling → LN.
- **Stage 8:** CORN, retained.
- **Fate of each stream:**
  - RGB: the pretrained trunk.
  - Vessel map: the prior encoder, then spatial modulation.
  - Each lesion map: the prior encoder, then modulation of the stage whose resolution matches its
    lesion scale (MA/HE at the high-resolution stages).
- **Cost:** +1–3 M params; about +10–20 % compute. On a T4 about 3.5–4.5 h per run (P ≈ 3.5 h).
- **Why this is not PL:** priors enter at feature level, at four scales, through a separate learned
  encoder and gates. PL pushed them through pretrained RGB stem filters with zero-init kernels.
- **Why this is not RACAF:** there is no reliability scalar and no gating of a random-init stream.
- **Why this is not the old Stage 7:** no random-init encoders, no token cross-attention, no early GAP.
- **Evidence:** lesion attention (CVPR 2019), MSGDA-Net, vessel-gated multi-scale fusion, mask-guided
  attention.
- **Complementary to soft targets + EMA?** Yes: architecture/information vs target/weight
  regularisation.
- **Implementation:** about 1–2 days (reuses P's builder, caches, trainer and CORN).
- **Clean ablation:** zero-init means γ = α = 0 is P. Compare with the existing P runs (identical
  recipe). Add an inference-time prior-permutation check to prove the priors are used.

### Architecture 2 — Dual pretrained encoders + mid/late cross-attention

- **Stages 2–4:** as Architecture 1.
- **Stage 5:** a second pretrained encoder (e.g. ImageNet ResNet-18, or the same ConvNeXt-T) on the
  prior maps, rendered as a 3-channel image or with an adapted stem.
- **Stage 6:** ImageNet ConvNeXt-T on RGB.
- **Stage 7:** cross-attention at the 16² stage — RGB tokens as queries, prior tokens as keys/values —
  plus a residual; then pooling.
- **Stage 8:** CORN.
- **Evidence:** the closest literature analogue (+2.4 % accuracy; arXiv 2411.03618).
- **Cost:** about 1.6–2× P (≈ 6–8 T4-h per run at batch 2).
- **Risk:** higher overfitting risk; it *resembles* the closed Stage 7, though with pretrained
  encoders.
- **Implementation:** about 2–3 days.
- **Ablation:** vs P and vs Architecture 1.

### Architecture 3 — Joint multi-task: shared pretrained encoder with lesion/vessel heads feeding the grader

- **Stage 2:** as before.
- **Stages 3/4:** become **heads** on a shared ConvNeXt/U-Net encoder.
  - Lesion head: supervised by public lesion masks (IDRiD train + Retinal-Lesions ± MAPLES-DR).
  - Vessel head: distilled from LWNet pseudo-labels on APTOS (Stage 3 knowledge retained).
- **Stages 5–7:** predicted maps and decoder features modulate the grading features (as in
  Architecture 1).
- **Stage 8:** CORN.
- **Evidence:** Zhou CVPR 2019; DRG-Net; AAAI 2020 multi-task.
- **Benefit:** lesion supervision also *regularises* the grading encoder, addressing overfitting.
- **Costs:**
  - mixed-dataset sampling (grade-only APTOS + mask-only lesion sets), loss balancing, 3–5 days of
    engineering;
  - about 1.3–1.5× P compute;
  - the most moving parts, and the hardest to attribute.

## 4. Comparison

| | Arch 1: gated prior injection | Arch 2: dual encoders + cross-attention | Arch 3: joint multi-task |
|---|---|---|---|
| Stage 3 | LWNet frozen | LWNet frozen | LWNet distilled into a head |
| Stage 4 | current → retrained (1b) | current → retrained | a head trained on public masks |
| Uses spatial lesion location | yes (4 scales) | yes (16² tokens) | yes |
| Extra compute vs P | +10–20 % | +60–100 % | +30–50 % |
| Implementation | 1–2 days | 2–3 days | 3–5 days |
| Overfitting risk | low (zero-init, small) | medium–high | medium (regularising) |
| Attribution / ablation | cleanest (starts ≡ P) | clean | hard |
| Closeness to closed routes | distinct | nearest to the old Stage 7 | distinct |
| Expected gain | modest; depends on Stage-4 quality | modest; higher variance | potentially largest; highest risk |

## 5. Recommendation and experiment sequence (one at a time; each pre-registered)

**Test Architecture 1 first.**

- **E1 — Architecture 1 with the *current* frozen Stage 3/4 (existing caches), 3 seeds, P's recipe.**
  - EMA is tracked as a shadow: free, and it does not change the raw weights.
  - Comparator: the existing P runs (§22).
  - Primary endpoint: overall grading (mean CORN threshold AUROC, or QWK — fix one before running).
  - Guardrails: QWK, grade-3 recall, false-urgent rate.
  - Plus an inference-time prior-permutation usage check.
  - This answers: does feature-level injection extract more from the existing priors than stem
    concatenation did?
- **E2 — Stage-4 upgrade, segmentation only, no grading.**
  - Retrain the lesion model with an ImageNet-pretrained encoder U-Net on IDRiD train (54) +
    Retinal-Lesions (1,593; free on request; EyePACS-sourced, so no APTOS overlap) ± MAPLES-DR (200).
  - Keep the 4-channel contract. Optional extra channels: preretinal/vitreous haemorrhage from
    Retinal-Lesions.
  - Gate on the **27-image IDRiD test set** against the current mean Dice 0.131, with a pre-set
    improvement bar.
  - Exclude DDR (reserved external) and FGADR (agreement pending).
- **E3 — Architecture 1 with the upgraded Stage 4** (regenerate the APTOS lesion caches, single pass,
  about 30 min), 3 seeds. Compare with E1 and P.
- **E4 — soft ordinal CORN targets on the best of E1/E3.** EMA results come from the shadows already
  recorded; a 2 × 2 falls out.
- **E5 — final frozen evaluation:** IDRiD grading test, single use, reported once.

**Soft targets + EMA in the first experiment?**
- **EMA: yes, as a shadow.** It is free and does not confound the raw arm.
- **Soft targets: no.** Add them only after the architecture is validated (E4), so the architecture
  effect is attributable.
- **CORN** remains appropriate: decoding is faithful (§25), and soft targets fit as an expected CORN NLL.

## 6. What to retain

Stage 2 preprocessing; **Stage 3 LWNet (frozen)**; the Stage 4 *role* and 4-channel output contract;
the CORN head and decoder; the APTOS split and caches for E1; P's pretrained ConvNeXt-T trunk.

## 7. What to replace

- **Stage 4 weights** (from-scratch Attention U-Net → pretrained-encoder U-Net on more public masks).
- **Stage 5** random-init multi-kernel CNN → a small prior encoder.
- **Stage 6** random-init Swin → pretrained ConvNeXt-T on RGB.
- **Stage 7** cross-attention and RACAF reliability → zero-init multi-scale gated injection.
- The TTA reliability computation is no longer needed.
- `PROJECT_CODE.md` / `SEGMENTATION_ARCHITECTURE.md` "frozen" statements need updating once decisions
  are made.

## 8. Minimum experiments to a defensible final pipeline

E1 → E2 → E3 → E4 → E5: four grading experiments (3 seeds each) and one segmentation training.
Roughly 45–60 T4-hours in total. Each step is gated on the previous one.

**Not claimable from this audit:** that the priors *will* improve over P. Our PL evidence says their
information is small at current Stage-4 quality. The plan is ordered to find that out cheaply (E1)
before paying for the Stage-4 upgrade.
