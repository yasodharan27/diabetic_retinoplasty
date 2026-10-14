# Architecture Audit — where the Grade-3/Grade-4 signal can be lost

Read-only audit of the repository at commit `b25da0f` (plus uncommitted docs). Nothing was trained,
run or modified. Parameter counts below were **measured** by building each stage on CPU
(`build_*()` → `trainable_variables`), not estimated.

Evidence labels used throughout: **ESTABLISHED** (demonstrated by a saved experiment or by the
code itself), **HYPOTHESIS** (plausible, untested), **CLOSED**, **OPEN**.

---

## 1. Trained graph actually used by the baseline (NO_RACAF, matched init)

`multiseed_runs.build_arm_model("NO_RACAF", seed)` → `no_racaf_model.build_no_racaf_joint_model_matched_init()`.

```
stage5_input (B,512,512,8)  = [RGB 0-2 | vessel 3 | lesion 4-7 (MA, HE, EX, SE)]   float32
    └─ _StopGradientBoundary
    └─ 4 × { multi-kernel block: Conv3x3 / Conv5x5 / dilated(3) Conv3x3, each Conv→BN→ReLU
             → AdaptiveBranchFusion (per-sample, per-channel softmax over 3 branches)
             → concat → Conv1x1 → BN → ReLU  → MaxPool2x2 }       filters 32, 64, 128, 256
    └─ L = local_features (B,32,32,256)  [1024 tokens, stride 16 px]           2,170,848 params

stage6_input (B,256,256,3)  = RGB only (resized from the *augmented* stage5 RGB)
    └─ Swin branch A: patch 4, embed 96, depths (2,2,6,2), heads (3,6,12,24), window 8 → 8×8×768
    └─ Swin branch B: patch 8, embed 96, depths (2,2,6),   heads (3,6,12),    window 8 → 8×8×384
    └─ concat → reshape → G = global_features (B,64,1152)                      39,697,956 params
       RANDOM INITIALISATION — no pretrained weights (swin_transformer.py:649)

Stage 07 (feature_fusion.build_adaptive_cross_attention)                     1,173,504 params
    L → reshape (1024,256) → LN → Dense(256) → + factorised 32×32 pos-emb   = K, V
    G → LN → Dense(1152→256) → + factorised 8×8 pos-emb                     = Q (64 tokens)
    A = MHA(8 heads × 32; Q=global, K=V=local; attention dropout 0.1)
    h = Q + A ;  h = h + FFN(LN(h))  [256→1024 GELU→256, dropout 0.1]
    E = GAP over the 64 query tokens → (B,256) float32

CORN (corn.build_corn_model)                                                     1,028 params
    z = Dense(256→4)(E)          raw logits, float16 under mixed_float16
    InertReliabilityConnection   (reliability input declared, exactly zero influence)

TOTAL trainable: 43,043,336  (405 tensors)  — measured, matches no_racaf_model.EXPECTED_*
```

Stage 06 alone is **92.2 %** of all trainable parameters. Stages 03/04 are not in the graph at all:
their outputs arrive as cached NumPy arrays.

### Decode (corn.decode_logits / CORNQuadraticWeightedKappa)

```
p_cond_k = σ(z_k)            = P(y>k | y≥k)
p_cum_k  = Π_{j≤k} σ(z_j)    = P(y>k)
grade    = Σ_k [p_cum_k > 0.5]
p_gt_3   = p_cum_3 = σ(z_0)σ(z_1)σ(z_2)σ(z_3) = P(y=4)       ← the score every §8–§12 analysis uses
```

---

## 2. Training configuration (multiseed_runs.py, frozen protocol `improved-multiseed-v1`)

| Item | Value |
|---|---|
| Split | `dataset_splits/aptos2019_train_val_split.csv`, sha256 `bc80fd45…`, split seed 42 |
| Train counts (0–4) | 1444 / 296 / 799 / **154** / **236** (2,929) |
| Val counts (0–4) | 361 / 74 / 200 / **39** / **59** (733; 730 after 3 pinned empty-FOV exclusions → 58 grade 4) |
| Train/val duplicates | 41 known pairs (RACAF_Duplicate_Contamination_Audit.md) |
| Loss | `weighted_corn`: sqrt-inverse-frequency weights (0.6929, 1.5304, 0.9315, 2.1217, 1.7139), numerator-weighted, unweighted denominator |
| Optimizer | AdamW, lr 1e-4, wd 0.05, decay excluded for names containing bias/gamma/beta |
| Schedule | ReduceLROnPlateau(val_QWK, ×0.5, patience 4, min 1e-6); EarlyStopping patience 12; max 50 epochs |
| Batch size | **2** |
| Precision | mixed_float16, LossScaleOptimizer asserted at build |
| Checkpoint selection | BEST = max `val_QWK` (two-slot BEST, stop_decision.json, behaviour fingerprint) |
| Augmentation | synchronised flip-H, flip-V, rot90 on all 8 channels; brightness/contrast ±10 % on RGB only (lesion/vessel never jittered); per-(run_seed, epoch, image) RNG |
| Data path | cache-only (`improved_training_data.py`); lesion maps = identity TTA view of frozen Stage 04 |

Across-seed BEST `val_QWK`: NO_RACAF 0.8297 / 0.8068 / 0.8269 (mean 0.8211, SD 0.0125). Train−val
QWK gap at the final epoch 0.05–0.11 in every run (mild, consistent overfitting).

---

## 3. Upstream inputs — what the lesion channels can and cannot carry

| Fact | Status | Source |
|---|---|---|
| Stage 04 = Attention U-Net trained on the 54-image IDRiD segmentation train set, frozen (Experiment 2C) | ESTABLISHED | SEGMENTATION_ARCHITECTURE.md §3.5 |
| IDRiD test Dice: EX 0.357, **HE 0.127**, SE 0.024, MA 0.017 | ESTABLISHED | same |
| No channel for IRMA, venous beading, NVD/NVE, preretinal/vitreous haemorrhage, fibrous proliferation, laser scars | ESTABLISHED (by construction) | lesion_segmentation_dataset.LESION_CLASSES |
| No lesion ground truth exists for APTOS, so per-image Stage 04 quality is unmeasurable on this population | ESTABLISHED | Multiseed report §12.5 |
| The trained models depend heavily on lesion channels (lesion permutation: QWK 0.83 → 0.06–0.19); vessel permutation costs 6–10 % as much | ESTABLISHED | Multiseed report §12.3–12.4 |
| 10 image-level lesion statistics (4× mean prob, 4× foreground fraction >0.5, total mean, total fraction) give OOF AUROC **0.670** on grade 4 vs 3 (n = 97) | ESTABLISHED (in-validation CV only) | record §11 |
| That 0.670 transfers from a probe **fitted on the training split** | **OPEN — never tested** | — |

Clinical context (literature, see DEEP_RESEARCH.md §2): lesion counts and areas rise up to severe
NPDR and **fall** from severe NPDR to PDR (Esmaeilkhanian et al., 2023). PDR-defining lesions are
absent from these four channels. So the usable 3-vs-4 signal in this pipeline is most plausibly a
**burden** (amount) signal, possibly with a direction reversal relative to grades 0→3.

---

## 4. Information-flow trace and candidate loss points

The question is where an image-level **amount** of lesion (burden) disappears between the input
channels 4–7 and `p_gt_3`.

### L1 — Stage 04 output quality (upstream ceiling)
Weak segmentation (HE Dice 0.127) bounds what any downstream stage can use. **But** the §11 probe
shows that crude global statistics of these same weak maps already beat the model, so L1 is not
what separates the probe from the model. Status: ESTABLISHED as a ceiling; not the explanation of
the probe-vs-model gap.

### L2 — Early fusion into a random-init CNN with BatchNorm at batch size 2 (Stage 05)
Lesion probabilities (small, sparse values) are concatenated with RGB in [0,1] and mixed by the
first random-init convolution. Every conv branch is followed by BatchNorm; with batch size 2 the
per-channel batch statistics are computed over two images, so during training an image's absolute
per-channel level is normalised *relative to its batch partner*, while inference uses running
averages. BN error grows sharply at batch size 2 (Wu & He, 2018). Status: **HYPOTHESIS** — could
blur image-level burden; not measured.

### L3 — Four 2×2 max-pools (Stage 05)
Max-pooling keeps local presence and discards local density inside each 2×2 window. Over four
stages each output token summarises a 16×16-pixel cell. Density across tokens is still
recoverable. Status: HYPOTHESIS, judged minor.

### L4 — Softmax cross-attention aggregation (Stage 07) — **the strongest structural candidate**
Each of the 64 query outputs is `Σ_j softmax(q·k_j) v_j`, a **convex combination** of local
values. If a query attends to lesion tokens, its output is (approximately) the *typical* lesion
value, and it is the same whether 5 or 500 tokens carry lesion. Softmax attention discards
multiset cardinality (Zhang & Xie, IJCAI 2020; same limitation for mean aggregators, Xu et al.,
ICLR 2019). Additionally:
- the queries come from the **RGB-only** Swin (G), so *which* local tokens are read is decided
  by RGB content, not by lesion content;
- the residual `h = Q + A` means E carries the RGB global stream by construction; the lesion
  stream enters only through A;
- the final GAP is over a fixed set of 64 query tokens, so it does not itself destroy burden.

Consequence: the architecture is structurally well suited to lesion **presence** (what separates
grades 0→1→2→3) and structurally poor at lesion **amount** (plausibly what separates 3 from 4 when
PDR lesions have no channel). Status: **HYPOTHESIS** (mechanistic, specific, testable read-only —
see RECOMMENDED_DESIGN.md Phase 0, probe P0.3).

### L5 — CORN head: one linear map shared by four tasks
`z = W E + b`, W ∈ ℝ^{4×256}. Each task has its own weight row, so a direction reversal (lower
burden ⇒ PDR) is linearly expressible *if* E encodes burden linearly. Rank consistency is kept by
the product decode, not by weight sharing. Status: ESTABLISHED (code).

### L6 — The marginal product decode `p_gt_3 = p_gt_2 · σ(z_3)`
If PDR images have *lower* lesion burden than severe-NPDR images, then `p_gt_2` (driven mainly
by burden-like evidence learned on grades 0→3) will tend to be *lower* for grade 4 than for
grade 3, and multiplies into the marginal score. The marginal `p_gt_3` could then rank 3-vs-4 worse
than the conditional `σ(z_3)` alone. Status: **HYPOTHESIS**, testable **read-only** from the saved
per-sample `p_gt_2`, `p_gt_3` (RECOMMENDED_DESIGN.md P0.1). This matters: if the conditional
ranks well, the loss is at the decode, not in the representation.

### L7 — Objective pressure on the 3/4 boundary
- CORN task 3 (the conditional 3-vs-4 task) sees only grade-3/4 training samples: 390 of 2,929
  images; **6.5 %** of included (sample, task) pairs unweighted, **10.4 %** after class weighting
  (computed from the pinned counts and weights).
- At batch size 2, a batch contains at least one grade-3/4 image with probability 0.249 (~364 of
  1,465 batches/epoch) and **both** a grade-3 and a grade-4 image with probability 0.0085 (~12
  batches/epoch). **Any in-batch pairwise ranking loss between grades 3 and 4 is therefore
  infeasible under the frozen protocol** without a memory bank.
- Checkpoint selection by `val_QWK` weights a 3↔4 error at 1/16 of a 0↔4 error (record §10), so
  the selected checkpoint is not chosen for 3-vs-4 discrimination.
Status: HYPOTHESIS, **currently under direct test** (§12 aux-loss experiment, running).

**Note for interpreting §13.** The §12 auxiliary BCE acts on the *marginal* `p_gt_3`. For a
grade-3 sample its gradient lowers the product, and so pushes `σ(z_0), σ(z_1), σ(z_2)` down as
well as `σ(z_3)`, while CORN pushes those three *up* for the same sample (grade 3 > 0, 1, 2). A
TRADE_OFF or grade-3 recall drop in §13 could come from this conflict and not from the
boundary hypothesis being wrong. An objective on the conditional `z_3` alone would not have this
conflict. (Analytical observation from the code; not an experimental finding.)

### L8 — Label reliability ceiling
APTOS labels come from a single-source competition set with no published adjudication. Severe NPDR
has the lowest inter-grader agreement of the ICDR categories in a recent UWF study, and missed NV
and PRP-scar interpretation are named sources of 3/4 disagreement (Krause et al., 2018). The
achievable 3-vs-4 AUROC on APTOS is therefore unknown and may be well below 1. Status: HYPOTHESIS
(ceiling of unknown size).

**Specific confound to check (HYPOTHESIS):** treated PDR eyes commonly carry panretinal
photocoagulation scars. A lesion segmenter trained on four IDRiD classes may fire on laser scars
as EX. `lesion_EX_foreground_fraction` is the strongest grade-4 variable (AUROC 0.7648, grade 4
vs rest). Part of the lesion "PDR signal" could be laser-scar detection. That would still be a
real image signal, but not PDR pathology. A visual audit of the top-EX grade-4 images would
settle it cheaply.

### L9 — Capacity vs data
39.7 M randomly initialised Swin parameters trained on 2,929 images at batch 2. Representation
quality is a global issue for every grade, not specific to 3-vs-4. Status: ESTABLISHED context.

---

## 5. Existing infrastructure relevant to a new experiment

| Asset | Reusable? |
|---|---|
| `no_racaf_model.build_no_racaf_joint_model_matched_init` + `InertReliabilityConnection` | Yes. A new builder must construct all baseline layers **first**, in the same order, then any new layers, so baseline init draws are bit-identical (same pattern as the C1 `global_projection` fix). |
| `weighted_corn.make_weighted_corn_loss`, `UnweightedCORNLoss` metric | Yes, unchanged. |
| `multiseed_runs` (locks, two-slot BEST, fingerprint, stop_decision, resume) | Yes. The aux notebook shows how to add an arm without editing it. |
| `improved_training_data.make_epoch_dataset` | Yields `((s5, s6, reliability), grade)`. A 4th input (precomputed lesion statistics) needs an **additive** data module; no cache regeneration is needed. |
| `colab/notebooks/grade4_aux_loss_experiment.ipynb` | Template for PREREGISTRATION.json + manifest + auto-run + verdict cells. |
| Frozen baselines `improved_multiseed_2026_09/NO_RACAF/seed_{42,123,2026}` | Reusable as the λ=0 / unmodified arm, exactly as §12 reuses them. |
| Saved per-sample artifacts (`per_sample_best.csv` with `p_gt_0..3`), `per_image_variables.csv` | Enough for Phase-0 probes P0.1 and part of P0.2 with no inference. |
| Auxiliary-loss infrastructure | Only inside the aux notebook (loss closure + recompile). No `.py` module; the recommended design needs none. |
