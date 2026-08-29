# CORN Ordinal Classification (Stage 08)

**Status:** Authoritative specification for CORN, the final stage of the downstream pipeline.
**Implemented and unit-tested** (`corn.py`, `tests/test_corn.py`). **Not trained; not frozen** —
its `Dense(256→4)` weights do not exist yet; no training has been run.

CORN wraps RACAF's output `F` — it does not read anything else. It is established
ordinal-regression methodology (Shi, Cao & Raschka, 2021, "Deep Neural Networks for
Rank-Consistent Ordinal Regression Based On Conditional Probabilities", arXiv:2111.08851),
applied to this project's 5-grade APTOS2019 task, not a new mechanism. **RACAF remains this
project's ONE approved research innovation** (`PROJECT_CODE.md`'s "Approved Research Innovation"
section) — CORN introduces no attention, no new uncertainty mechanism, no feature extractor, no
second fusion, and no custom loss.

---

## 1. Purpose

Final ordinal DR-severity classification. The last stage of the downstream pipeline:

```
Stage 05 → Stage 06 → Stage 07 → RACAF → F=(B,256) → CORN → 5-class ordered DR grade
```

---

## 2. Classification Task

Diabetic retinopathy severity grading — ordinal, 5-class, verified directly against
`datasets/APTOS2019/raw/train.csv` (`id_code,diagnosis`, 3662 labeled rows):

| Grade | Name | Meaning |
|---|---|---|
| 0 | No DR | |
| 1 | Mild | |
| 2 | Moderate | |
| 3 | Severe | |
| 4 | Proliferative DR | |

Strictly ordinal: 0 < 1 < 2 < 3 < 4. Names reused from this repository's own existing
`evaluation/evaluator.py` docstring convention, not invented here.

---

## 3. Input Contract

**CORN receives exactly `F = (B, 256)` from RACAF — nothing else.**

Verified live against the actual, current `racaf.build_racaf_fusion()`:
`model.output_shape == (None, 256)`, linear activation. No Stage 4 mask, no Stage 5/6 features, no
Stage 7 raw `E`, no RACAF reliability vector `r`, no raw image, and no ground-truth label of any
kind reaches CORN's forward pass.

---

## 4. CORN Architecture

**A single `Dense(256 → 4)` layer applied directly to `F`. No hidden layer. No activation on the
output.**

```
F = (B, 256)
      │
      ▼
Dense(256 → 4), linear
      │
      ▼
z = (B, 4)   -- raw, unbounded logits
```

- Hidden layers: **none**
- Output dimension: **4** (= K−1 for K=5 grades, not K)
- Activation: **none** (linear) — sigmoid is applied only inside the loss (§6) and the decoding
  rule (§7), never inside the model itself
- No attention, no convolution, no second feature extractor, no second fusion mechanism

This matches the CORN reference implementation's own convention (`Raschka-research-group/coral-pytorch`):
the output layer attaches directly to the backbone's last hidden representation, with no
interposed MLP. `F` already *is* that last hidden representation here — the product of Stages
05→06→07→RACAF — so a hidden layer here would be unjustified additional depth with no basis in
either the literature or this project's approved design.

---

## 5. Ordinal Formulation

Four raw logits `z = (z₀, z₁, z₂, z₃)`. Each `zₖ` is the logit of a **conditional** probability —
not an independent binary classifier output:

$$z_k \;\Rightarrow\; P(Y > k \mid Y \ge k), \qquad k = 0, 1, 2, 3$$

| Output | Meaning |
|---|---|
| $z_0$ | logit of $P(Y>0 \mid Y\ge 0) = P(Y>0)$ |
| $z_1$ | logit of $P(Y>1 \mid Y\ge 1)$ |
| $z_2$ | logit of $P(Y>2 \mid Y\ge 2)$ |
| $z_3$ | logit of $P(Y>3 \mid Y\ge 3)$ |

By the chain rule (since `Y≥k` and `Y>k−1` are the same event for integer `Y`):

$$P(Y>k) = \prod_{i=0}^{k} P(Y>i \mid Y\ge i) = \prod_{i=0}^{k}\sigma(z_i)$$

This cumulative product forces $P(Y{>}0)\ge P(Y{>}1)\ge P(Y{>}2)\ge P(Y{>}3)$ automatically (each
factor ∈[0,1]) — ordinal information is preserved by construction. Unlike CORAL, CORN needs **no
weight-sharing constraint** across the four outputs to guarantee this; rank consistency instead
comes from the conditional training-subset construction (§6) and this cumulative decoding (§7).
The four outputs must never be treated as four independent, unrelated binary classifiers.

---

## 6. Loss — Standard CORN Conditional-Subset Objective

For a training example with true grade `y` and logits `z=(z₀,z₁,z₂,z₃)`, for each task `k=0,1,2,3`:

- **Conditional inclusion:** the example contributes to task `k`'s loss **iff `y ≥ k`** (task 0:
  always; task 3: only if `y≥3`).
- **Binary target:** $t_k = \mathbb{1}[y>k]$, defined only for included examples.

Per-(example, task) loss — standard, numerically stable binary cross-entropy with logits:

$$\ell_{n,k} = -\big[t_{n,k}\log\sigma(z_{n,k}) + (1-t_{n,k})\log(1-\sigma(z_{n,k}))\big]$$

**Total loss**, summed over all included (example, task) pairs, normalized by their total pooled
count `N_total` (across every task and every example together — not averaged per-task first):

$$\mathcal{L}_{CORN} = \frac{1}{N_{total}}\sum_{k=0}^{3}\;\sum_{n:\,y_n\ge k}\ell_{n,k}$$

Implemented (`corn.py`'s `corn_loss`) via `tf.nn.sigmoid_cross_entropy_with_logits` (TensorFlow's
own numerically stable BCE-with-logits, mathematically identical to the formula above) applied
elementwise to the full `(B,4)` logits tensor, masked to zero for excluded `(example, task)`
pairs, then summed and divided by the mask's own sum.

**No focal loss, class weighting, label smoothing, Dice loss, or auxiliary penalty is added.**
APTOS2019's class imbalance (§11) is a separate, future training decision, not folded into this
loss. Per `training/losses.py`'s own stated policy ("module-specific losses ... add those
alongside that module's model definition"), this loss lives in `corn.py`, not
`training/losses.py`.

---

## 7. Inference

$$p^{cond}_k=\sigma(z_k),\qquad p^{cum}_k=\prod_{i=0}^{k}p^{cond}_i\;(=P(Y{>}k)),\qquad k=0..3$$

$$\hat y=\sum_{k=0}^{3}\mathbb{1}[p^{cum}_k>0.5]\in\{0,1,2,3,4\}$$

`ŷ` counts how many of the 4 cumulative-probability thresholds were exceeded — never an index into
them. Implemented in `corn.py`'s `decode_logits()`.

**Probability reconstruction** (a forced finite-difference of the cumulative probabilities, not a
new design choice — satisfies `pipeline.ClassificationStage`'s existing `"probabilities"`
contract):

$$P(Y{=}0)=1-p^{cum}_0,\quad P(Y{=}k)=p^{cum}_{k-1}-p^{cum}_k\;(k{=}1,2,3),\quad P(Y{=}4)=p^{cum}_3$$

Verified (`tests/test_corn.py`) to sum to ≈1 and to be monotonically consistent with `p_cum`.

---

## 8. Parameter Count

`Dense(256→4)`: weights `256×4=1024` + bias `4` = **1,028 trainable parameters** — measured
directly (`sum(int(np.prod(v.shape)) for v in model.trainable_variables)`), not estimated.

| Component | Trainable parameters | Owned by CORN? |
|---|---|---|
| CORN (`Dense(256→4)`) | **1,028** | Yes |
| RACAF (gate + Global readout) | 295,170 | No — upstream |
| Stage 07 (Adaptive Cross-Attention) | 1,173,504 | No — upstream |
| Stage 06 (Swin backbones) | ~39.7M | No — upstream |
| Stage 05 | not measured here | No — upstream |

---

## 9. Training Boundary

**Architecture frozen ≠ weights frozen ≠ excluded from joint training** — explicitly distinguished:

- CORN's **architecture** (`Dense(256→4)`) is fixed by this document and `corn.py`.
- CORN's **weights** do not exist yet — nothing has been trained.
- Once trained, CORN's weights, RACAF's 295,170 parameters, and Stages 05/06/07's weights are all
  intended to train **jointly**, in one shared graph, through CORN's own loss (§6) backpropagated
  end-to-end — matching `RACAF_ARCHITECTURE.md` §7's statement that RACAF's parameters "train
  jointly with Stages 05–08." Only Stage 4 (frozen, via `tf.stop_gradient`+`trainable=False` inside
  RACAF's TTA) and Stage 3/1 (not part of the same graph at all) are excluded.
- `corn.py`'s `build_corn_model()` returns an **uncompiled** model — no optimizer, no training loop
  embedded. `CORNStage.train()`/`.evaluate()` raise `NotImplementedError`, mirroring
  `RACAFStage`/`AdaptiveCrossAttentionStage`'s identical convention — CORN has no standalone
  training procedure; the joint Stage 05–08+RACAF training script does not exist yet and is not
  implemented by this document.

---

## 10. Dataset / Split Contract

CORN uses **APTOS2019 only**, via the already-committed authoritative manifest
`dataset_splits/aptos2019_train_val_split.csv` (`downstream_split.get_authoritative_split()`) —
**no second split is created.** Re-verified this turn: 2929 train / 733 val = 3662, stratified by
`diagnosis` (80/20, seed 42), zero overlap. The same image IDs are already used by Stage 05/06
(`local_feature_extraction_dataset.split_train_val_ids()` delegates to this same manifest;
`global_feature_extraction_dataset.py` re-exports it unmodified). Stage 07/RACAF/CORN's eventual
joint training script should call `downstream_split.get_authoritative_split()` directly.

`corn.py` itself defines no dataset loader and reads no CSV — its only input is `F`, already
produced upstream from these same image IDs by the time it reaches CORN.

---

## 11. Class Imbalance (Reported, Not Acted On)

APTOS `train.csv` distribution, re-verified this turn:

| Grade | Count | % |
|---|---|---|
| 0 | 1805 | 49.3% |
| 1 | 370 | 10.1% |
| 2 | 999 | 27.3% |
| 3 | 193 | 5.3% |
| 4 | 295 | 8.1% |

Grade 0 outnumbers Grade 3 by ≈9.4×. A plain CORN loss trained on this distribution will likely
bias toward lower grades; per-class recall for grades 1/3/4 should be watched at evaluation time.
**No loss weighting, oversampling, or architecture change is introduced here** — this is an
explicit, separate, future training decision.

---

## 12. Evaluation Status

- **`test.csv`** (1928 rows, `id_code` only, no `diagnosis`) — confirmed unlabeled, unusable for
  supervised evaluation.
- **IDRiD `grading`'s 103-image official test split** — remains an optional external evaluation
  candidate, **PENDING USER APPROVAL**. Not adopted, not used anywhere in this implementation.
- CORN's architecture is indifferent to this decision — since it reads only `F` (§3), a different
  final-evaluation image set can be substituted later without any CORN architecture change.

**Evaluation metrics** (design contract only — not implemented, no result computed or claimed):
training loss = `corn_loss` (§6); classification metrics = accuracy, macro-F1, and quadratic
weighted kappa (QWK — the standard ordinal-appropriate metric, already implemented and reusable in
`evaluation/metrics.py`'s `quadratic_weighted_kappa()`), computed from `decode_logits()`'s
`predicted_grade` against true labels once a real evaluation set and trained weights exist.

---

## 13. RACAF → CORN Boundary

Exactly one tensor crosses it: `F=(B,256)`. Verified live this turn against the real
`racaf.build_racaf_fusion()` model. CORN reads nothing else RACAF or any earlier stage computed.

---

## 14. Innovation Boundary

RACAF remains the project's sole research innovation. CORN introduces zero novel mechanisms: the
head is a bare linear layer, the loss and decode rule are the unmodified, literature-verified CORN
methodology (Shi, Cao & Raschka, 2021).

---

## 15. Current Status

```
CORN: IMPLEMENTED
CORN: TESTED
CORN: NOT TRAINED
CORN: NOT FROZEN
```
