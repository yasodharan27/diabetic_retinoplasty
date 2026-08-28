# Reliability-Aware Cross-Attention Fusion (RACAF)

**Status:** Authoritative specification for the single approved downstream research innovation
referenced by `PROJECT_CODE.md`'s Approved Research Innovation section and
`IMPLEMENTATION_PLAN.md`'s roadmap. **Not yet implemented.** RACAF has no code, no dataset
loader, no training run, and no checkpoint — this document exists so that its design is fixed
and unambiguous *before* implementation begins, exactly as `SEGMENTATION_ARCHITECTURE.md`
already did for Stages 3–4 before their code was written.

RACAF depends on Stage 05 (Local Feature Extraction), Stage 06 (Global Feature Extraction), and
Stage 07 (Adaptive Cross-Attention) already having finalized output contracts — it has no
independent existence apart from what those three stages hand it. It must not be implemented
before that dependency is satisfied (§13).

No file described as frozen or finalized elsewhere in this repository (Stages 01–04, Experiment
2C) is modified, redesigned, or retrained by anything in this document.

---

## 1. Motivation

Stage 04 (Attention U-Net, Experiment 2C — Weighted-Pooled Dice) is now finalized. Its measured
per-class performance on the official IDRiD 27-image held-out test set is recorded here as
**evaluation history**, for motivation only:

| Class | Dice | IoU |
|---|---|---|
| Hard Exudate (EX) | 0.3574 | 0.2176 |
| Haemorrhage (HE) | 0.1273 | 0.0680 |
| Soft Exudate (SE) | 0.0244 | 0.0123 |
| Microaneurysm (MA) | 0.0165 | 0.0083 |
| **Mean** | **0.1314** | **0.0766** |

**These numbers are never used as an input to RACAF.** They motivate the problem this document
solves and describe evidence that the problem is real, not hypothetical — nothing more. The
21.7x spread between the best- and worst-performing classes shows that Stage 04's four lesion
channels are not equally trustworthy, on average, across the dataset. But average trustworthiness
is not the same thing as this-image trustworthiness, and a fixed, dataset-level number cannot
serve as a per-image signal without leaking held-out evaluation information into a trainable
downstream component (see §9). RACAF's actual reliability signal is derived entirely differently
— from Stage 04's own inference-time behavior on the specific image being processed, never from
its recorded test-set score.

Given that segmentation quality varies this much by class, and Stage 05's documented input
contract concatenates all four lesion channels with equal, unconditional weight
(`SEGMENTATION_ARCHITECTURE.md` §4), nothing in the currently-specified downstream architecture
lets the classifier know, per image, whether the segmentation-derived information it just
received is more likely to be signal or noise. RACAF exists to answer that question at the one
point in the pipeline where a segmentation-dependent representation and a segmentation-independent
representation are both already available for comparison — the Feature Fusion boundary.

**What is, and is not, the contribution.** Predictive disagreement from test-time augmentation is
a known, general technique — it is not invented here. Sigmoid gating of a weighted blend between
two representations is a known primitive — it is not invented here either. The contribution is
the specific integration of *frozen lesion segmentation* + *per-image TTA disagreement* +
*segmentation reliability estimation* + *reliability-controlled fusion of segmentation-dependent
and global representations* + *ordinal DR classification*, assembled to work within this exact
pipeline's constraints — most importantly, that Stage 04 is finalized and must never be retrained
or architecturally modified to obtain this signal. That constraint is what rules out the more
common alternative (MC-Dropout-style resampling, which needs dropout layers inside the frozen
model) and is what makes the TTA-based route the correct choice here, not an arbitrary one.

---

## 2. Baseline

The architecture immediately upstream and downstream of Feature Fusion, exactly as currently
documented, with no RACAF mechanism present:

```
Local Features (Stage 05)  +  Global Features (Stage 06)
                    │
                    ▼
      Adaptive Cross-Attention (Stage 07)
                    │
                    ▼  E  (fused embedding, dimensionality fixed by CORN's needs)
                    │
                    ▼
      CORN Ordinal Classification (Stage 08)
```

Every channel of segmentation-derived information reaches Stage 05 with identical, unconditional
trust, regardless of which lesion class actually drove the prediction on a given image, and
Stage 07's fusion output $E$ is passed to CORN with no signal at all about how reliable the
Local branch's segmentation-dependent content was for this particular image.

---

## 3. Proposed RACAF

```
Local Features (Stage 05)  +  Global Features (Stage 06)
                    │
                    ▼
      Adaptive Cross-Attention (Stage 07)  →  E
                    │
                    ▼
      [NEW] Reliability-Aware Cross-Attention Fusion (RACAF)
                    │
                    ▼  F  (same shape as E)
                    │
                    ▼
      CORN Ordinal Classification (Stage 08)  — receives F, not E
```

RACAF wraps Stage 07's output. It does not alter Stage 07's own Query/Key/Value computation or
softmax, does not alter Stage 05 or Stage 06's internals, and does not alter CORN's head or
decision logic — only the vector CORN receives as input changes, and only in value, not shape.

---

## 4. Reliability Estimation

For the image currently being processed, Stage 04 (frozen, Experiment 2C) is evaluated four
times, once per deterministic geometric transform:

1. identity
2. horizontal flip
3. vertical flip
4. 180° rotation

Each of these four transforms is an exact pixel permutation on Stage 04's fixed 512×512 working
resolution — no interpolation, no resampling artifact is introduced by the augmentation itself.
Each output is transformed back into canonical (original) orientation before anything else is
computed, so all four predictions are pixel-aligned.

For each lesion class $c \in \{MA, HE, EX, SE\}$:

- **Mean prediction:** $\bar p_c$ — the average of the four aligned probability maps.
- **Per-pixel disagreement:** $D_c(x,y) = \text{Var}_k\big[\tilde p_c^{(k)}(x,y)\big]$ across the
  four aligned views.
- **Foreground restriction:** disagreement is pooled only over the region the *averaged*
  prediction itself claims is foreground — $U_c = \{(x,y): \bar p_c(x,y) > 0.5\}$, reusing this
  project's existing `DEFAULT_THRESHOLD` convention, not a newly invented number. This avoids the
  signal being diluted by the overwhelming majority of confidently-background pixels that a
  whole-image average would otherwise be dominated by.
- **Mathematical normalization:** disagreement is rescaled against the maximum variance
  mathematically possible for a quantity bounded in $[0,1]$ across four samples — $0.25$, achieved
  at a perfect split between $0$ and $1$. This is a fixed constant derived from the definition of
  variance, not fit to any dataset, split, or model.

The result is one **per-class reliability vector**, computed fresh for every image, using nothing
but Stage 04's own frozen forward pass on that image, repeated four times with a different
deterministic input transform each time.

---

## 5. Mathematical Formulation

**Per-pixel disagreement and foreground restriction:**

$$D_c(x,y) = \text{Var}_k\big[\tilde p_c^{(k)}(x,y)\big], \qquad U_c = \{(x,y): \bar p_c(x,y) > 0.5\}$$

$$\Delta_c = \begin{cases} \dfrac{1}{|U_c|}\displaystyle\sum_{(x,y)\in U_c} D_c(x,y) & |U_c| > 0 \\[6pt] 0 & |U_c| = 0 \end{cases}$$

**Normalization against the mathematical variance ceiling:**

$$\kappa_c = 1 - \frac{\Delta_c}{\Delta_{\max}}, \qquad \Delta_{\max} = 0.25$$

**Per-class reliability vector:**

$$\kappa = [\kappa_{MA}, \kappa_{HE}, \kappa_{EX}, \kappa_{SE}]$$

**Predicted lesion burden and burden-weighted aggregation:**

$$B_c = \text{mean}(\bar p_c), \qquad w_c = \frac{B_c}{\sum_{c'} B_{c'} + \epsilon}$$

If total burden $\sum_c B_c$ is effectively zero (no lesion detected in any class), $w_c$
defaults to $\tfrac14$ for every class — a well-defined fallback, not undefined behavior.

**Image-level reliability:**

$$r = \sum_c w_c\,\kappa_c$$

**Learned gate:**

$$\text{gate} = \sigma(w_g\, r + b_g)$$

where $w_g, b_g$ are the only *learned* parameters in the reliability path — trained downstream,
jointly with Stages 05–08, never fit to any test-set statistic.

**Global readout and fusion:**

$$E = \text{AdaptiveCrossAttention}(L, G), \qquad \hat G = W_r\cdot\text{GAP}(G) + b_r$$

$$F = \text{gate}\cdot E + (1-\text{gate})\cdot\hat G$$

$F$ replaces $E$ as CORN's input. $W_r, b_r$ are learned; their only purpose is projecting the
Global branch's pooled representation to the same dimensionality as $E$ so the blend is
well-defined — this is the one small architectural addition RACAF requires beyond the reliability
computation and gate themselves.

---

## 6. Tensor Contracts

Using this project's own fixed values where they exist, and marking everything else as an
explicit, unresolved design parameter rather than an invented number:

| Point in the flow | Shape | Status |
|---|---|---|
| Stage 04 input (per TTA view) | $(512, 512, 4)$ | Fixed (`DEFAULT_INPUT_SHAPE`, `lesion_segmentation_model.py`) |
| Stage 04 output per view $\tilde p_c^{(k)}$ | $(512, 512, 4)$ | Fixed (model's own working resolution) |
| $\bar p_c$, $D_c$ | $(512, 512, 4)$ each | Derived, this document — same resolution as Stage 04's native output |
| $\kappa_c$, $B_c$, $w_c$ | $(4,)$ each | Derived — always exactly 4, fixed by `LESION_CLASSES` order |
| $r$, gate | scalar | Derived |
| Native-resolution lesion maps fed to Stage 05 | $(H_{img}, W_{img}, 4)$ | Fixed (`predict_lesion_mask`'s existing resize-back logic, unmodified by RACAF) |
| $L$ (Local FE output) | $(H_{local}, W_{local}, C_{local})$ | **Design parameter — not yet fixed.** `SEGMENTATION_ARCHITECTURE.md` §4 explicitly defers this; no shape is invented here. |
| $G$ (Global FE output, pre-pool) | $(64, 1152)$ | **Fixed** by Stage 06's implementation (`swin_transformer.py`'s `create_dual_scale_swin_model()`) — un-pooled token sequence, $N=64$, $C_G=1152$. Not a RACAF decision; recorded here only because RACAF's own formula below depends on it. |
| $\text{GAP}(G)$ | $(1152,)$ | Mean over $G$'s 64-token axis, computed by RACAF (not by Stage 06 — Stage 06 applies no pooling of its own). |
| $E = \text{CrossAttn}(L,G)$ | $(d_{model},)$ | **Design parameter.** "Fixed by CORN's own formulation," not yet numerically fixed by any governing document. |
| $\hat G$ | $(d_{model},)$ | Forced to match $E$ by construction |
| $F$ | $(d_{model},)$ | Identical shape to $E$ — CORN's documented input contract is unaffected in shape, only in value |
| CORN output | $(4,)$ cumulative logits | Fixed (5 APTOS grades → 4 cumulative thresholds) |

No shape in Stages 01–04, or in CORN's own head, is affected by RACAF. $C_G=1152$ is now fixed
by Stage 06's implementation (above). The one value RACAF is still waiting on is $d_{model}$, an
open question Stage 07's own eventual implementation must resolve independently of whether RACAF
is ever built.

---

## 7. Training

- **Stage 03 is frozen.** Its weights are not updated during Stage 05–08 (or RACAF) training; it
  is called only through its documented `predict()`/`predict_batch()` interface.
- **Stage 04 is frozen.** Same as Stage 03 — never retrained, never architecturally modified
  (specifically: no dropout is added) to support RACAF. Its outputs are wrapped in a
  stop-gradient before any reliability quantity is computed from them, so gradient from the
  downstream classification loss cannot reach Stage 04's parameters through this path even
  though the entropy/variance functions involved are technically differentiable.
- **RACAF's downstream parameters ($w_g, b_g, W_r, b_r$) are trained jointly with Stages 05–08**,
  under the same optimizer already planned for that joint run, as ordinary additional trainable
  variables. The reliability *estimation* itself (the four TTA passes, $D_c$, $U_c$, $\Delta_c$,
  $\kappa_c$, $B_c$, $w_c$, $r$) is fully deterministic and non-trainable — it produces a fixed
  input to the gate, computed once per image and cacheable exactly like this project's existing
  vessel-map disk-caching pattern (`_get_or_compute_vessel_map` in `lesion_segmentation_dataset.py`).
- **No test-set information is introduced anywhere in this training procedure** — see §9 for the
  full audit.
- **No auxiliary loss is required or used.** $w_g,b_g,W_r,b_r$ train purely through the existing
  downstream CORN loss, end-to-end. An auxiliary stabilization term (e.g. discouraging the gate
  from collapsing to exactly 0 or 1 across an entire batch) is not part of this specification and
  would require explicit approval before being added.
- **Dataset:** follows the existing downstream classification dataset specification —
  APTOS 2019, the dataset already documented as Stage 08's training set
  (`PROJECT_STRUCTURE.md`'s Dataset Organization table). Stage 03/04 are run in inference mode
  over APTOS 2019 to produce the vessel/lesion maps RACAF's reliability computation consumes —
  already their documented existing role for this dataset
  (`SEGMENTATION_ARCHITECTURE.md` §1.4). **IDRiD's test set is never part of this training
  procedure in any capacity** — not as a label source, not as a feature source, since RACAF's
  reliability computation is only ever run on the image currently being classified (an APTOS
  2019 image during training, or any new image at inference), never on an IDRiD test image.

---

## 8. Inference

For a completely unseen, real-world fundus image, with no ground-truth of any kind available:

```
Image
  ↓ IQA (accept)
  ↓ Preprocessing — Gamma + CLAHE
  ↓ Vessel Segmentation — LWNet (frozen) → v
  ↓ Lesion Segmentation — Attention U-Net, Experiment 2C (frozen)
        run 4x on {identity, h-flip, v-flip, 180°-rotation}(input)
        → 4 aligned probability maps per class → p̄_c, D_c
  ↓ Reliability estimation → κ_c → B_c, w_c → r
  ↓ Local Features (Stage 05, from RGB + v + lesion maps)   Global Features (Stage 06, from RGB)
  ↓ Adaptive Cross-Attention (Stage 07) → E ;  Ĝ = W_r·GAP(G) + b_r
  ↓ RACAF: gate = σ(w_g·r + b_g);  F = gate·E + (1-gate)·Ĝ
  ↓ CORN → DR grade
```

No step in this trace reads a lesion mask, a vessel mask, or a DR grade label for the image being
processed. The only additional computation RACAF introduces at inference is: three extra Stage
04 forward passes (on transformed copies of the same already-preprocessed image), a handful of
elementwise variance/mean reductions over already-computed probability maps, one affine+sigmoid
(two scalars), and one small dense-layer matmul (the Global readout). No new forward pass of any
other network, no resampling of Stage 03, no access to any label.

---

## 9. Leakage Prevention

Four explicit claims, each verifiable directly from the formulation above:

- **No test labels are used.** The reliability computation's only inputs are Stage 04's frozen
  weights and the image currently being processed. No IDRiD ground-truth mask is read at any
  point, for any image, in training or inference.
- **No test-set Dice is used.** The Experiment 2C numbers in §1 are recorded once, as motivating
  evaluation history, and do not appear anywhere in §4/§5's formulation. $\Delta_{\max}=0.25$ is a
  mathematical constant (the maximum possible variance of a $[0,1]$-bounded quantity), and the
  $0.5$ threshold reused for $U_c$ is this project's own pre-existing segmentation operating
  convention, not a number fit to IDRiD's test split or to any split.
- **No test-set statistics influence model parameters.** The only learned parameters
  ($w_g,b_g,W_r,b_r$) are updated by gradient descent against APTOS 2019's training data under the
  CORN loss. Stage 04's role in producing RACAF's features is restricted to APTOS 2019 images —
  IDRiD's test images are never passed through Stage 04 to generate a RACAF feature, so there is
  no path, structural or incidental, for IDRiD test information to reach a trainable parameter.
- **The final IDRiD test set remains untouched until final evaluation.** Its role in this
  document is exactly what it was before RACAF existed: the held-out set that produced Experiment
  2C's already-reported, already-frozen Dice/IoU numbers. It is never re-read by anything RACAF
  does.

---

## 10. Baseline vs RACAF Ablation

**Baseline:**

```
Local + Global
       ↓
Adaptive Cross Attention
       ↓
CORN
```

**Proposed:**

```
Local + Global
       ↓
Adaptive Cross Attention
       ↓
RACAF
       ↓
CORN
```

Everything else identical: same Local FE, same Global FE, same Cross-Attention internals, same
CORN head/loss, same optimizer, same data split, same training budget, same seed protocol.

**Metrics:** Accuracy, Macro F1, Quadratic Weighted Kappa, per-class recall, confusion matrix, and
calibration metrics where already supported by the project (`evaluation/` already implements
QWK and calibration reporting per `PROJECT_STRUCTURE.md`'s description of `evaluation.Evaluator`).
A stratified comparison by RACAF's own dynamic $r$ (low-$r$ vs. high-$r$ images) is the direct
test of whether the mechanism does what it claims, in addition to the aggregate numbers above.

**No result is fabricated or assumed here.** Neither run exists yet — this section defines the
comparison to be run once both are trained, not a claimed outcome.

---

## 11. Research Hypothesis

> Explicitly gating the fused Local (segmentation-dependent) representation by a per-image
> reliability score — derived from Stage 04's frozen test-time-augmentation predictive
> disagreement, never from any labeled statistic — will improve ordinal DR-grading performance
> (QWK, Macro F1) relative to an architecturally identical model that fuses the Local and Global
> branches unconditionally, with any improvement concentrated on images where the reliability
> score is low, and no degradation on images where it is high.

This is falsifiable in both halves: an aggregate-only gain with no relationship to the stratified
$r$-based split would not support the mechanism's actual claimed behavior, even if overall QWK
improved for unrelated reasons.

---

## 12. Limitations

**TTA disagreement cannot guarantee detection of an error that remains invariant across all four
transformations.** If Stage 04 is confidently, systematically wrong on a region of an image in a
way that horizontal flip, vertical flip, and 180° rotation all fail to disturb, $D_c$ for that
region will be low, $\kappa_c$ will be high, and RACAF will not flag it. This is a real, accepted,
disclosed residual gap — not a claim that this mechanism detects every failure mode of a poorly
segmented image. Closing it further would require a fundamentally different signal (e.g.
resampled internal stochasticity via dropout inside Stage 04 itself), which this specification
explicitly does not pursue, since it would require retraining or architecturally modifying Stage
04 — forbidden by this project's current constraints.

RACAF is also, by design, scoped only to Stage 04's lesion output — it does not estimate or
respond to Stage 03's vessel-segmentation reliability, which has no comparable frozen evaluation
signal to build against in this project. This is a deliberate scope boundary, not an oversight.

---

## 13. Implementation Requirements

RACAF must not be implemented until all of the following exist:

- A finalized Local Feature Extraction (Stage 05) output contract — specifically, a fixed
  $(H_{local}, W_{local}, C_{local})$ (or equivalent flattened) shape.
- A finalized Global Feature Extraction (Stage 06) output contract — specifically, a fixed
  input resolution and channel count $C_G$. **Satisfied**: Stage 06 is implemented
  (`swin_transformer.py`), input resolution 256x256, $G=(64, 1152)$, $C_G=1152$.
- A finalized Adaptive Cross-Attention (Stage 07) output contract — specifically, a fixed
  $d_{model}$, since $\hat G$'s projection dimensionality is defined relative to it.
- The frozen Experiment 2C checkpoint, unchanged from its currently committed state.
- Confirmation that Stage 04's inference interface (`predict_lesion_mask` /
  `predict_lesion_mask_batch` in `lesion_segmentation_model.py`) can be called repeatedly, with a
  transformed input, without requiring any change to the loaded model itself — already true today,
  since these are plain inference calls against a loaded `tf.keras.Model`.

Until these exist, this document is a specification only. No RACAF code, dataset loader,
training script, or checkpoint exists in this repository as of this document's authoring.
