# Stage 5–8 complementary-branch audit on top of P (2026-09-30)

This is a design audit only: nothing was trained or implemented.

- **Baseline, fixed:** P, an ImageNet ConvNeXt-Tiny on RGB at 512 px, with a CORN head.
- **Evidence used:**
  - §§22–28 of the research record;
  - the earlier literature audits (`STAGE5_7_DIRECTION_AUDIT.md`, `NEXT_STEP_LITERATURE_REVIEW.md`, `SOURCES.md`);
  - fresh searches for the mechanism families those audits did not cover.

## What any branch has to beat (the evidence)

| Finding | What it means for a branch |
|---|---|
| P's pooled representation is the best linear read-out of its own map; spatial/multi-scale read-outs of *that map* add nothing (S1) | A branch must bring **new input information**, not re-read P's features |
| Stage 3/4 priors fed into P's stem add < 0.01 (PL). RACAF, reliability gating, C1, G0 and the generic-vessel route are closed | Lesion-prior fusion, in any wiring, starts from a negative prior |
| P overfits heavily: training QWK 0.99 vs validation 0.91; validation loss rises after epochs 5–8; grade-3 recall collapses late (S2) | Extra parameters risk more overfitting; a branch must be low-capacity or frozen |
| 3-vs-4 is saturated near 0.85 and consistent with a data/label limit; decoding is faithful (S2, S3) | No branch should promise a fix for grade 3 vs 4 |
| Label-free features separate burden-matched grade 4 from grade 2 only weakly (0.72); fine-tuned features are label-contaminated on the training split (S5) | **Any screening diagnostic must use label-free frozen features, or held-out data** |

## Candidates

| # | Mechanism | One-line hypothesis | Information P lacks | Literature status | Inputs / supervision | Stage 3/4 support | Clean ablation vs P | Colab cost | Closed-direction risk |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **High-resolution local-detail branch** (native-resolution tiles, or disc/macula crops at about 1024–1536 px, frozen or lightly tuned encoder, late fusion) | Downsampling to 512 px removes fine structures (MA, small NV, IRMA) that carry severe-grade information | **Physically absent from P's input:** APTOS originals reach about 3,200 px, so detail is lost at 512 px | Resolution and dual-resolution DR models are **established**; evidence on the benefit above about 512–640 px is mixed. **Application-level**, but "is severe-grade information lost at 512 px?" is directly testable | Grade labels only; raw images are on Drive | Not needed. Stage 3 vessel convergence could locate the disc if disc crops are used | **Yes** — a single-variable change (resolution / extra crop branch) | Frozen diagnostic: cheap. Training at 1024: about 4× P compute per run | Low — different from S1, which re-read P's 512 map and never added input detail |
| 2 | **Complementary pretraining branch** (a frozen retinal-SSL or DINOv2-type encoder, late fusion) | Retina-specific or different self-supervised pretraining encodes severity cues that ImageNet-supervised ConvNeXt features miss | A different feature prior. Whether it holds *complementary* information is unknown | **Established / application-level:** RETFound and DINOv2 are well benchmarked; DINOv2-large ≈ RETFound on APTOS (AUROC 0.952 vs 0.944); frozen RETFound degrades strongly (QWK 0.67 vs 0.91 fine-tuned). Fusion and ensembling are standard | Grade labels; public weights (licence check needed) | No | **Yes** — P vs P + frozen branch (fixed late fusion) | Frozen feature extraction is cheap on a T4 (ViT-L inference) | Low. Earlier audits rejected a *backbone swap* "with no hypothesis"; a complementarity test is different |
| 3 | **Anatomy-registered lesion distribution** (Stage-4 HE/MA/EX/SE counts per retinal quadrant / ETDRS field, relative to the disc and fovea) | Severe NPDR is defined by quadrant-level spread (the 4-2-1 rule), which a translation-invariant GAP model does not encode explicitly | Explicit anatomical position of lesions | Anatomy-anchored pooling is **published** (PARAGraph 2026); lesion-quadrant features are classic. **Incremental** | Disc/fovea localisation (heuristic or trained on IDRiD localisation labels) plus existing Stage-4 maps | **Yes**, fully from caches | Yes (late fusion of low-dimensional features) | Cheap (features are computed on CPU) | **Medium:** S1's image-frame 2×2 grid added nothing for 3 vs 0–2. Lesion priors are closed as a route. Stage-4 quality ceiling: the 10-variable lesion statistics are weak |
| 4 | **Vessel-geometry descriptors** (tortuosity, fractal dimension, density, calibre, beading proxies from Stage 3) | Global vascular geometry changes with severity and is not explicitly summarised by P | Explicit geometric summaries; P may already capture them implicitly | Associations are **established** as biomarkers; DL-derived vascular morphology exists. **Incremental** for grading | Existing Stage-3 maps; no artery/vein separation | Yes | Yes | Cheap | **Medium-high:** the vessel channel contributed about 0 in the old pipeline (permutation ΔQWK 0.04–0.07); vessel priors in PL were null; the generic-vessel PDR route is closed |
| 5 | **Image-quality-aware conditioning** (Stage-1 IQA state gates or weights the prediction) | Grading errors concentrate in low-quality images; conditioning on quality improves robustness | Explicit quality state | **Established:** quality-guided DR grading (QGDR 2026; quality-aware co-embedding, CVPR 2023) | EyeQ-trained IQA model (exists) | Stage 1 exists; no APTOS IQA outputs are cached | Yes | Cheap | **Medium:** reliability-gated fusion (RACAF) is closed, and S4 found no quality link to the recurrent failures |
| 6 | **Frequency/texture branch** (wavelet / FFT / LBP) | Spectral texture cues complement spatial CNN features | Probably little — pretrained CNNs capture texture | **Established:** wavelet-CNN DR, multi-frequency attention, dual-path wavelet GNNs (2024–2026) | Grade labels | No | Yes | Cheap | Low novelty; weak prior of complementarity |
| 7 | **Consistency / ensemble reliability** (multi-seed ensemble, TTA consistency, SWA/EMA) | Averaging and consistency reduce P's variance and overfitting | Not new information — variance reduction | **Established** | None extra | No | Yes | Moderate (ensembles multiply inference) | Selective prediction (D6) is closed. This belongs in the **baseline**, not as a research contribution |

**Excluded and not re-proposed:** reliability-weighted or gated fusion of lesion priors (RACAF, C3);
learnable lesion queries / MIL pooling (S1 plus published work); the two-route or ordinal-objective
head (C1, S3); PDR-sign supervision (the FGADR direction, set aside by the user).

## Candidates worth deeper investigation (2–3)

1. **#1 High-resolution local-detail branch.** It is the only candidate that adds information *physically
   removed* from P's input, and it has a clean single-variable ablation. The label-free test: do frozen
   ImageNet ConvNeXt features at about 1024 px (or disc/macula crops) carry more grade information than
   the same frozen features at 512 px, under training-split CV and a permutation null? **Novelty:
   application-level.** The mechanism is established; the value lies in the controlled evidence for
   severe grades.
2. **#2 Complementary pretraining branch.** A cheap, label-free complementarity test: frozen retinal-SSL
   or DINOv2 features alone, and fixed-weight late fusion with P's *validation* outputs. No fitting on
   P's training features, which S5 showed are contaminated. **Novelty: application-level.** Worth
   screening only because it is cheap and cleanly ablatable.
3. **#3 Anatomy-registered lesion distribution — conditional.** It is clinically grounded for grade 3
   (the 4-2-1 rule) and computable entirely from existing caches, but S1's partial null and the Stage-4
   quality ceiling make it the weakest of the three. **Novelty: incremental** (anatomy pooling is
   published).

**Honest limit.** None of the candidates is a novel mechanism. A defensible research contribution would
come only from a **rigorous complementarity result** — information shown to be *absent from P and
recoverable by the branch* under label-free, leakage-controlled diagnostics — not from the module itself.
Stop here; no implementation.
