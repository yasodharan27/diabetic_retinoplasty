"""
CORN (Conditional Ordinal Regression for Neural networks) -- Stage 08, the
final ordinal DR-severity classifier, per the approved design
(`CORN_ARCHITECTURE.md`, `PROJECT_CODE.md`'s Models table). The last stage
in the downstream pipeline:

    Stage 05 -> Stage 06 -> Stage 07 -> RACAF -> F=(B,256) -> CORN -> DR grade

RACAF remains this project's ONE approved research innovation
(`PROJECT_CODE.md`'s "Approved Research Innovation" section). CORN is
established ordinal-regression methodology (Shi, Cao & Raschka, 2021,
"Deep Neural Networks for Rank-Consistent Ordinal Regression Based On
Conditional Probabilities", arXiv:2111.08851) applied to this project's
5-grade APTOS2019 task -- it introduces no new mechanism of its own:

- The architecture is a single `Dense(256 -> 4)` layer applied directly to
  `F` -- no hidden layer, no activation on the output, no attention, no
  convolution, no second feature extractor or fusion mechanism.
- The loss and inference/decoding rule below are the standard, unmodified
  CORN formulation -- not a custom ordinal loss.

Kept in independently testable pieces, mirroring `racaf.py`'s modularity
(do not hide the mechanism inside one function):

1. `build_corn_model` -- the architecture. An **uncompiled** Keras model;
   no optimizer, no training loop is embedded here. Full `.keras`
   serialization works out of the box (`Dense`/`Input` are built-in Keras
   layers with existing `get_config()` -- no custom, non-serializable layer
   class is introduced, unlike RACAF's `_OneMinus`).
2. `corn_loss` -- the standard CORN conditional-subset loss, computed from
   raw logits and integer grades. Pure TensorFlow, differentiable, no
   ground-truth mask or reliability signal read.
3. `decode_logits` -- deterministic, non-trainable inference: sigmoid ->
   cumulative product -> thresholded predicted grade -> per-class
   probability reconstruction. Pure NumPy.
4. `CORNQuadraticWeightedKappa` -- a Keras `Metric`, NOT a loss, for joint
   training's `monitor="val_QWK", mode="max"` checkpoint selection
   (`JOINT_TRAINING_ARCHITECTURE.md` Sec 23). Decodes raw CORN logits with
   EXACTLY `decode_logits`'s own sigmoid -> cumulative-product ->
   threshold-count rule (re-expressed in TensorFlow ops so it runs inside
   `model.fit()`'s graph-mode execution), then delegates confusion-matrix
   accumulation and kappa computation to
   `training.metrics.QuadraticWeightedKappa`, unmodified -- see this
   class's own docstring for why the generic metric cannot be attached to
   CORN's output directly.
5. `CORNStage` -- `pipeline.classification.ClassificationStage`
   implementation (this project's existing classification-stage contract,
   already binding on Stage 01/Stage 08 per that ABC's own docstring), not
   a new interface.

Boundary (`CORN_ARCHITECTURE.md`): this module reads only RACAF's `F`
(`(B, 256)`) -- never Stage 4's masks, Stage 5/6 features, Stage 7's raw
`E`, RACAF's reliability vector `r`, a ground-truth segmentation mask, or
any label other than the APTOS `diagnosis` column consumed by `corn_loss`
during (future, not-yet-implemented) training. It introduces no second
research innovation.
"""

import os

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers

import config
from pipeline.classification import ClassificationStage
from training.metrics import QuadraticWeightedKappa

# --- Fixed by the approved CORN design (CORN_ARCHITECTURE.md) ---

NUM_GRADES = 5  # APTOS2019 DR grades, 0-4
NUM_THRESHOLDS = NUM_GRADES - 1  # K-1 = 4 ordinal thresholds
D_MODEL = 256  # matches RACAF's F / feature_fusion.D_MODEL exactly

# Reused from evaluation/evaluator.py's own existing docstring convention
# (`Evaluator(class_names=["No DR", "Mild", "Moderate", "Severe",
# "Proliferative"])`), not invented here.
GRADE_NAMES = ("No DR", "Mild", "Moderate", "Severe", "Proliferative")

DEFAULT_MODEL_PATH = os.path.join(config.CORN_MODEL_DIR, "best_model.keras")


# =====================================================================
# 1. CORN architecture -- a single Dense(256 -> 4) layer, nothing else.
# =====================================================================

def build_corn_model(d_model=D_MODEL, num_thresholds=NUM_THRESHOLDS):
    """Builds CORN's entire architecture: `Dense(d_model -> num_thresholds)`
    applied directly to `F`, linear activation (raw logits -- sigmoid is
    applied only inside `corn_loss`/`decode_logits`, never baked into the
    layer itself, matching the reference CORN implementation's own
    convention). No hidden layer.

    Returns an **uncompiled** `keras.Model`, `F=(B, d_model) -> z=(B,
    num_thresholds)`. For the default `d_model=256`, `num_thresholds=4`:
    exactly `256*4 + 4 = 1,028` trainable parameters (measured, not
    estimated -- see `tests/test_corn.py`)."""
    f_input = Input(shape=(d_model,), name="F")
    logits = layers.Dense(num_thresholds, activation=None, name="corn_logits")(f_input)
    return Model(inputs=f_input, outputs=logits, name="corn")


# =====================================================================
# 2. CORN loss -- standard conditional-subset ordinal loss.
# =====================================================================

def corn_loss(logits, grades, num_thresholds=NUM_THRESHOLDS):
    """Standard CORN training objective (Shi, Cao & Raschka, 2021).

    `logits`: `(B, num_thresholds)` raw, unactivated CORN outputs.
    `grades`: `(B,)` integer ground-truth DR grades, `0..NUM_GRADES-1`.

    For each ordinal task `k = 0..num_thresholds-1`:
      - a sample is INCLUDED in task `k`'s loss iff `grade >= k` (task 0 is
        always included; task `num_thresholds-1` only for the highest
        grades) -- this conditional-subset construction is what makes CORN
        "conditional", distinct from treating the `num_thresholds` outputs
        as independent ordinary binary classifiers.
      - its binary target is `t_k = 1[grade > k]`.

    Per-(sample, task) loss is standard, numerically stable binary
    cross-entropy with logits (`tf.nn.sigmoid_cross_entropy_with_logits`,
    mathematically `max(z,0) - z*t + log(1+exp(-|z|))`, equal to
    `-[t*log(sigmoid(z)) + (1-t)*log(1-sigmoid(z))]`), computed for every
    `(sample, task)` pair, masked to zero for excluded pairs, then summed
    and divided by the total number of INCLUDED pairs (pooled across every
    task, not averaged per-task first) -- exactly the "sum of all included
    example/task losses / number of included example/task pairs" objective
    this module implements, not a novel variant.

    No focal loss, class weighting, label smoothing, or auxiliary penalty
    is added here -- APTOS2019's class imbalance (`CORN_ARCHITECTURE.md`)
    is a separate, future training decision, not folded into this loss.

    Accepts `logits` in any float dtype (in particular float16, CORN's Dense layer has no
    explicit float32 output override, so under a `mixed_float16` policy its raw output is
    float16) -- `tf.convert_to_tensor(logits, dtype=tf.float32)` does NOT cast an
    already-a-tensor input of a different dtype, it raises `ValueError: Tensor conversion
    requested dtype float32 for Tensor with dtype float16`; converting first, then casting
    explicitly, is what actually casts. The loss itself is still always computed in float32
    (`sigmoid_cross_entropy_with_logits`'s numerically-stable formulation, division), matching
    every other stage's "float16 activations, float32 loss" mixed-precision convention in this
    project -- no CORN mathematics changes."""
    logits = tf.convert_to_tensor(logits)
    logits = tf.cast(logits, tf.float32)
    grades = tf.cast(tf.convert_to_tensor(grades), tf.float32)

    thresholds = tf.range(num_thresholds, dtype=tf.float32)  # [0, 1, 2, 3]
    grades_column = grades[:, tf.newaxis]  # (B, 1)

    mask = tf.cast(grades_column >= thresholds[tf.newaxis, :], tf.float32)  # (B, T): grade >= k
    target = tf.cast(grades_column > thresholds[tf.newaxis, :], tf.float32)  # (B, T): grade > k

    per_element_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)  # (B, T)
    masked_loss = per_element_loss * mask

    total_loss = tf.reduce_sum(masked_loss)
    total_included = tf.reduce_sum(mask)
    return total_loss / total_included


# =====================================================================
# 3. Inference / decoding -- deterministic, non-trainable, pure NumPy.
# =====================================================================

def decode_logits(logits):
    """Converts CORN's raw `(B, num_thresholds)` logits into a decoded
    prediction -- exactly the approved inference rule, no ambiguity between
    threshold index, class index, and final grade:

        p_cond_k = sigmoid(z_k)                          -- P(Y>k | Y>=k)
        p_cum_k  = prod(i=0..k) p_cond_i                  -- P(Y>k), monotonically
                                                              non-increasing in k
        predicted_grade = sum(k)[p_cum_k > 0.5]           -- in {0,...,NUM_GRADES-1}

    Per-class probability reconstruction (finite-difference of the
    cumulative probabilities -- forced by `p_cum`, not a new design
    choice), satisfying `pipeline.ClassificationStage`'s existing
    `"probabilities"` contract:

        P(Y=0)   = 1 - p_cum_0
        P(Y=k)   = p_cum_(k-1) - p_cum_k     for k = 1, 2, 3
        P(Y=NUM_GRADES-1) = p_cum_(num_thresholds-1)

    Returns a dict: `"p_cond"`, `"p_cum"` (each `(B, num_thresholds)`),
    `"predicted_grade"` (`(B,)` int64, in `[0, NUM_GRADES-1]`),
    `"class_probabilities"` (`(B, NUM_GRADES)`, rows sum to ~1)."""
    logits = np.asarray(logits, dtype=np.float64)  # float64: keeps the cumprod/diff stable
    p_cond = 1.0 / (1.0 + np.exp(-logits))
    p_cum = np.cumprod(p_cond, axis=-1)

    predicted_grade = np.sum(p_cum > 0.5, axis=-1).astype(np.int64)

    p0 = 1.0 - p_cum[..., 0]
    middle = p_cum[..., :-1] - p_cum[..., 1:]  # (B, num_thresholds-1): P(Y=1..NUM_GRADES-2)
    p_last = p_cum[..., -1]
    class_probabilities = np.concatenate(
        [p0[..., np.newaxis], middle, p_last[..., np.newaxis]], axis=-1,
    ).astype(np.float32)

    return {
        "p_cond": p_cond.astype(np.float32),
        "p_cum": p_cum.astype(np.float32),
        "predicted_grade": predicted_grade,
        "class_probabilities": class_probabilities,
    }


# =====================================================================
# 4. CORN-aware Quadratic Weighted Kappa -- a Keras METRIC (never a loss),
#    for joint training's val_QWK checkpoint selection.
# =====================================================================

class CORNQuadraticWeightedKappa(QuadraticWeightedKappa):
    """Quadratic Weighted Kappa for CORN's raw ordinal logits -- for use as a
    `model.compile(metrics=[...])` entry during joint training, so Keras's own `logs` dict
    actually contains `"QWK"`/`"val_QWK"` for `JOINT_TRAINING_ARCHITECTURE.md` Sec 23's
    `monitor="val_QWK", mode="max"` checkpoint-selection policy to observe. This is a METRIC
    only -- `corn_loss` remains the sole training objective (see this module's docstring); no
    auxiliary loss is introduced by adding this metric.

    `training.metrics.QuadraticWeightedKappa.update_state()` cannot be applied to CORN's raw
    `(B, num_thresholds)` logits directly: it argmaxes `y_pred` over its last axis, which would
    treat CORN's `num_thresholds=4` conditional-task logits as 4 mutually exclusive classes --
    wrong both mathematically (an ordinal grade needs the sigmoid -> cumulative-product ->
    threshold-count decode below, not an argmax) and structurally (argmax over 4 values can
    never produce class index 4, silently making the highest DR grade unreachable).

    This subclass overrides ONLY the decode step. Confusion-matrix accumulation, `result()`'s
    kappa computation, and `reset_state()` are all inherited from `QuadraticWeightedKappa`
    UNCHANGED -- the kappa mathematics itself is not duplicated here. The decode is EXACTLY
    `decode_logits`'s own inference rule (item 3 above), re-expressed in pure TensorFlow ops
    since `decode_logits` itself is NumPy-only (used for eager, single-prediction inference) and
    this class must run inside `model.fit()`'s graph-mode execution:

        p_cond = sigmoid(logits)
        p_cum  = cumprod(p_cond, axis=-1)
        grade  = sum(p_cum > 0.5, axis=-1)   -- in [0, NUM_GRADES-1]

    `tests/test_corn.py`'s `CORNQuadraticWeightedKappaTests` proves this decode is numerically
    identical to `decode_logits` on the same logits, and that the resulting kappa matches an
    independent reference implementation."""

    def __init__(self, num_classes=NUM_GRADES, name="QWK", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Same dtype-safety fix as corn_loss (see its docstring): `y_pred` is CORN's raw
        # logits, float16 under a mixed_float16 policy -- convert first, THEN cast, since
        # `tf.convert_to_tensor(x, dtype=...)` does not cast an already-a-tensor input of a
        # different dtype.
        logits = tf.convert_to_tensor(y_pred)
        logits = tf.cast(logits, tf.float32)
        p_cond = tf.sigmoid(logits)
        p_cum = tf.math.cumprod(p_cond, axis=-1)
        predicted_grade = tf.reduce_sum(tf.cast(p_cum > 0.5, tf.int32), axis=-1)
        super().update_state(y_true, predicted_grade, sample_weight=sample_weight)


# =====================================================================
# 5. pipeline.ClassificationStage implementation.
# =====================================================================

class CORNStage(ClassificationStage):
    """`pipeline.classification.ClassificationStage` implementation for
    CORN -- this project's existing classification-stage contract
    (`predict`/`predict_batch` return `{"label", "class_index",
    "confidence", "probabilities"}`, already binding on Stage 01/Stage 08
    per that ABC's own docstring), not a new interface.

    `train`/`evaluate` raise `NotImplementedError` -- CORN has no
    standalone training procedure of its own: its `Dense(256->4)` weights
    are intended to train jointly with Stages 05-07 and RACAF through this
    module's own `corn_loss`, backpropagated end-to-end through the whole
    downstream graph (mirroring `RACAFStage.train()`/
    `AdaptiveCrossAttentionStage.train()`'s identical pattern). That joint
    training script does not exist yet and is explicitly not implemented
    here."""

    def __init__(self, d_model=D_MODEL, num_thresholds=NUM_THRESHOLDS):
        self.d_model = d_model
        self.num_thresholds = num_thresholds
        self.model = None

    def train(self, train_data, val_data=None, **kwargs):
        raise NotImplementedError(
            "CORN has no standalone training procedure in this project -- its "
            "Dense(256->4) weights train only jointly with Stages 05-07 and RACAF, "
            "through this module's own corn_loss, backpropagated end-to-end. That "
            "joint training script does not exist yet."
        )

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError(
            "CORN has no standalone evaluation procedure in this project -- its "
            "usefulness is only observable through the joint downstream pipeline's "
            "classification metrics (accuracy, macro-F1, quadratic weighted kappa), "
            "not reproduced here."
        )

    def save(self, path):
        if self.model is None:
            raise RuntimeError("No model to save -- call build() (or assign self.model) first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        self.model = tf.keras.models.load_model(path, compile=False)
        return self

    def build(self):
        """Builds a fresh, uncompiled model from `self.d_model`/
        `self.num_thresholds` and assigns it to `self.model`. Separate from
        `__init__`, mirroring `RACAFStage`/`AdaptiveCrossAttentionStage`'s
        identical lazy-build pattern."""
        self.model = build_corn_model(d_model=self.d_model, num_thresholds=self.num_thresholds)
        return self.model

    def _decode_one(self, class_probabilities, predicted_grade):
        grade = int(predicted_grade)
        return {
            "label": GRADE_NAMES[grade],
            "class_index": grade,
            "confidence": float(class_probabilities[grade]),
            "probabilities": {
                GRADE_NAMES[i]: float(class_probabilities[i]) for i in range(NUM_GRADES)
            },
        }

    def predict(self, input_data):
        """`input_data`: `F` for a single (un-batched) image, `(d_model,)`.
        Returns the `ClassificationStage` result dict."""
        if self.model is None:
            raise RuntimeError("CORNStage.build() or .load() must be called before predict().")
        f = np.asarray(input_data, dtype="float32")[None, ...]
        logits = self.model.predict(f, verbose=0)
        decoded = decode_logits(logits)
        return self._decode_one(decoded["class_probabilities"][0], decoded["predicted_grade"][0])

    def predict_batch(self, inputs):
        """`inputs`: a list of `F` vectors, each `(d_model,)`. Returns one
        result dict per input, in the same order."""
        if self.model is None:
            raise RuntimeError("CORNStage.build() or .load() must be called before predict_batch().")
        f_batch = np.stack([np.asarray(x, dtype="float32") for x in inputs], axis=0)
        logits = self.model.predict(f_batch, verbose=0)
        decoded = decode_logits(logits)
        return [
            self._decode_one(decoded["class_probabilities"][i], decoded["predicted_grade"][i])
            for i in range(len(inputs))
        ]
