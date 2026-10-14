# Candidate #2 — complementary pretrained branch for P: research only (2026-09-30)

No experiment was run for this document. P is the fixed baseline: an ImageNet-1k supervised ConvNeXt-Tiny
on RGB at 512 px, a CNN.

## Candidates

| # | Model | Weights / licence | Pretraining data | Different prior vs P | DR / retinal evidence | Leakage vs APTOS / IDRiD / DDR | T4 feasibility |
|---|---|---|---|---|---|---|---|
| 1 | **RETFound-DINOv2 (MEH)** — ViT-L/14 | HF `YukunZhou/RETFound_dinov2_meh`; **CC BY-NC 4.0**; **gated** (contact-info form) | Part of **AlzEye** (Moorfields, UK; private) | Retinal-domain, self-distillation SSL, ViT | Specialist RETFound-DINOv2 beats generalist DINOv2/v3 on ocular tasks, including linear probing (arXiv 2509.03421); DR 5-class accuracy 71.15 %, AUROC 0.932 (reported) | **None expected:** private UK data, no public DR sets. Confirm the card's data statement at download | Inference only; ViT-L at 224–448 in fp16 fits easily |
| 2 | **DINOv2 ViT-L/14 (or B/14)** — generalist control | **Apache-2.0**; ungated (torch.hub / HF) | LVD-142M natural images | Self-distillation SSL, ViT, natural domain | Frozen DINOv2 DR QWK 0.882 vs 0.904 fine-tuned (IOVS abstract); fine-tuned DINOv2-L APTOS AUROC 0.952 vs RETFound 0.944 | Negligible: web-curated natural images; a stray fundus image cannot be fully excluded | Inference only; easy |
| 3 | **RETFound-MAE** (natureCFP / meh) — ViT-L/16 | CC BY-NC 4.0; gated | MEH-MIDAS + public (EyePACS per FusionFM) | Retinal-domain, MAE | Strong **fine-tuned**; frozen collapses (QWK 0.672 vs 0.908) because MAE features are poor for linear read-out | EyePACS is not APTOS; the list of public components must be verified | Easy |
| 4 | **RETFound-Green** — ViT-S at 392 px with registers | GitHub (licence unverified) | 75 k images: **AIROGS, DDR (all), ODIR-2019** | Retinal token-reconstruction SSL | "More significant wins" than other retinal FMs across six datasets including IDRiD | **DDR leakage.** It would disqualify any later DDR external evaluation. APTOS and IDRiD are clean | Very cheap |
| 5 | **MedSigLIP-448** — 400 M vision tower | HAI-DEF terms; gated | Medical image-text, including **EyePACS** fundus plus other undisclosed ophthalmology data | Language-supervised medical prior | Limited DR-specific frozen evidence | **Unverifiable:** the full ophthalmology dataset list is not public | Feasible at 448 px |

**Excluded for leakage or availability:**
- FLAIR, RET-CLIP, RetiZero, EyeCLIP, UrFound — their pretraining pools include public DR sets (FLAIR
  explicitly includes APTOS, IDRiD and DDR).
- DINORET — self-supervised adaptation on EyePACS + **DDR** + AIROGS; weight release unclear.
- VisionFM — dataset composition and weight access not verified.

## What the literature establishes

- **Fusing foundation models is only lightly studied.**
  - FusionFM (2025) fused four eye-specific foundation models (RETFound, VisionFM, RetiZero, DINORET) with
    mixture-of-experts gating. Gains were modest and task-dependent (glaucoma +4 % AUC). **No ImageNet CNN
    baseline was included; APTOS, IDRiD and DDR were not used.**
  - CNN + ViT ensembles for DR grading are common, but at application level, with weak controls.
- **Domain foundation models vs compact ImageNet models:**
  - compact ImageNet models are near-optimal for most retinal tasks;
  - the specialist RETFound earned its cost **only for fine-grained DR grading under class imbalance**
    (arXiv 2511.22001);
  - specialist RETFound-DINOv2 still beats DINOv2/v3 generalists (arXiv 2509.03421).
- **Not found:** a leakage-controlled measurement of whether retinal-domain SSL features carry DR
  information that is *complementary to a strong ImageNet CNN*, as opposed to substituting for it.

## Research value

- A P + pretrained-branch model is an **application/integration** at the method level.
- The defensible question is empirical: *does retinal-domain SSL add severity information beyond a strong
  ImageNet-supervised CNN, and is any gain due to the retinal domain or just to a different SSL/ViT prior?*
- Pairing a retinal-domain model with a generalist control of the same objective family (#1 vs #2) is
  what makes the question answerable.

## Cleanest frozen complementarity test (pre-specify before running)

1. **Stage A — label-free on both sides, APTOS training split (2,921), no validation:**
   - Features:
     - the 512 px ImageNet ConvNeXt-Tiny features from the Candidate #1 run (P's label-free initialisation)
       — reused, not re-extracted;
     - frozen #1 and #2 embeddings from the same canonical Stage-2 RGB, at each model's pretraining
       resolution, fixed in advance.
   - Probes: the S5/HR probe and CV on ConvNeXt, FM, and [ConvNeXt ⊕ FM].
   - Primary Δ = AUROC(concatenation) − AUROC(ConvNeXt) for grade ≥ 3 vs ≤ 2, with a paired bootstrap, a
     score-swap permutation test and a label-permutation null.
   - Domain effect: Δ(#1) − Δ(#2).
   - Adjacent pairs, including 3 vs 4, reported descriptively.
2. **Stage B — only if Stage A passes:**
   - fit the FM probe on training features (clean);
   - fix an equal-weight logit average with P's *saved* validation P(Y ≥ 3);
   - compare with P alone once on validation (paired bootstrap);
   - no weights are tuned.

   P's own training features are never used — S5 showed they are label-contaminated.
3. **Confounds to report:** input resolution (224–448 px for the FMs vs 512 px for ConvNeXt), and
   architecture (ViT vs CNN). The #2 control absorbs the "different SSL/ViT prior" part.

## Single best candidate for the next frozen test

**RETFound-DINOv2 (MEH), run together with DINOv2 ViT-L/14 as the mandatory generalist control.**

This is not chosen for its scores. It is the only candidate that is at once:
- retinal-domain;
- trained with an objective whose frozen features are usable (MAE's are not);
- pretrained on private data with **no overlap with APTOS, IDRiD or DDR**;
- available under an academic licence.

**Blocking:** the user must accept the gated Hugging Face access for RETFound-DINOv2 (CC BY-NC 4.0,
non-commercial). DINOv2 is Apache-2.0 and ungated.
