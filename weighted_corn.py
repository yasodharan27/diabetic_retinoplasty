"""
Class-weighted CORN loss -- an ADDITIVE module. `corn.py` is not modified.

Motivation and audit: `docs/experiments/` multi-seed improved-training audit, Part C
(class-imbalance). APTOS2019's training-set class counts are moderately imbalanced
(grade 3 is 5.3% of training images), and Keras' built-in `class_weight=`/`sample_weight=`
machinery is mathematically WRONG for `corn.corn_loss`: that loss already reduces every
(sample, task) pair to ONE pooled scalar per batch (`total_loss / total_included`), and Keras 3's
`compile_loss` reduction (`keras/src/losses/loss.py::reduce_weighted_values`) then multiplies
that ALREADY-SCALAR loss by the batch's sample weights and divides by batch size again --
producing `corn_loss(batch) * mean(sample_weight)`, not a per-sample-weighted loss at all.

This module reimplements CORN's per-(sample, task) loss construction (the SAME mask/target/
`sigmoid_cross_entropy_with_logits` formulation `corn.corn_loss` uses -- not a new ordinal
mechanism) and applies the class weight to the NUMERATOR only, keeping the UNWEIGHTED
included-pair count as the denominator:

    numerator   = sum_i  w[y_i] * sum_k m_ik * loss_ik
    denominator = sum_i  sum_k m_ik                        -- UNWEIGHTED
    loss        = numerator / denominator

At uniform weights (`w_c == 1` for every class) this is IDENTICAL to `corn.corn_loss` --
`tests/test_weighted_corn.py` proves it numerically, including under mixed_float16 logits.
`corn.build_corn_model()`, `corn.decode_logits()` and `corn.CORNQuadraticWeightedKappa` are all
unchanged and reused unmodified: this module changes the TRAINING objective only, never the
architecture, the decode rule, or the reported QWK.

Weighting formula (pre-registered): square-root inverse frequency,
`w_c ∝ p_c^(-1/2)`, normalized so `sum(p_c * w_c) == 1` -- i.e. the weighted loss stays on the
same overall scale as the unweighted one; uniform weights are the special case `power=0`.
Weights are derived from TRAINING-ONLY class counts, never validation.
"""

import numpy as np
import tensorflow as tf

import corn

#: The committed APTOS2019 training split's per-grade image counts (grades 0-4), from
#: `dataset_splits/aptos2019_train_val_split.csv`, `split == "train"` rows only -- pinned here so
#: this module never silently recomputes a different value if the split file changes underfoot.
#: `tests/test_weighted_corn.py::test_pinned_train_counts_match_the_committed_split_manifest`
#: verifies this against the real file.
PREREGISTERED_TRAIN_COUNTS = (1444, 296, 799, 154, 236)
DEFAULT_WEIGHT_POWER = 0.5


def class_weights_from_counts(counts, power=DEFAULT_WEIGHT_POWER):
    """`w_c ∝ p_c^(-power)`, normalized so `sum(p_c * w_c) == 1`.

    `counts`: one positive count per class, TRAINING-ONLY (never touch a validation count --
    that would be using validation-set information to shape the training objective). `power=0`
    reproduces uniform weights (`w_c == 1` for every class) exactly, which is what makes this
    loss's unit-weight case identical to `corn.corn_loss` (see this module's docstring)."""
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1 or counts.size < 2:
        raise ValueError(f"counts must be a 1-D sequence of at least 2 class counts, got {counts!r}")
    if np.any(counts <= 0):
        raise ValueError(f"all class counts must be positive, got {counts.tolist()}")
    proportions = counts / counts.sum()
    raw_weights = proportions ** (-power)
    normalizer = float(np.sum(proportions * raw_weights))
    weights = raw_weights / normalizer
    return weights.astype(np.float64)


def _round4(values):
    return tuple(round(float(v), 4) for v in values)


#: Computed from `PREREGISTERED_TRAIN_COUNTS` at import time, then cross-checked against the
#: independently pre-registered, human-reviewed constants below -- so a change to either the
#: counts or the weighting formula is caught immediately (an import-time `RuntimeError`) rather
#: than silently producing different weights than what was pre-registered before any run started.
_COMPUTED_TRAIN_CLASS_WEIGHTS = class_weights_from_counts(PREREGISTERED_TRAIN_COUNTS, DEFAULT_WEIGHT_POWER)
PREREGISTERED_CLASS_WEIGHTS = tuple(float(w) for w in _COMPUTED_TRAIN_CLASS_WEIGHTS)
_EXPECTED_ROUNDED_CLASS_WEIGHTS = (0.6929, 1.5304, 0.9315, 2.1217, 1.7139)

if _round4(PREREGISTERED_CLASS_WEIGHTS) != _EXPECTED_ROUNDED_CLASS_WEIGHTS:
    raise RuntimeError(
        f"Computed class weights {_round4(PREREGISTERED_CLASS_WEIGHTS)} (from counts "
        f"{PREREGISTERED_TRAIN_COUNTS}, power={DEFAULT_WEIGHT_POWER}) do not match the "
        f"pre-registered constants {_EXPECTED_ROUNDED_CLASS_WEIGHTS}. Either the training class "
        "counts or the weighting formula changed since pre-registration -- this must be resolved "
        "deliberately, not silently, before any run uses this module."
    )


def _corn_pairs(logits, grades, num_thresholds):
    """The exact `mask`/`target`/`per_element_loss` construction `corn.corn_loss` uses --
    reproduced (not imported as a private helper, since `corn.py` exposes none) so this module's
    weighted numerator and unweighted denominator are built from IDENTICAL per-(sample, task)
    quantities. `logits`: `(B, num_thresholds)` raw CORN logits, any float dtype. `grades`:
    `(B,)` integer ground-truth grades. Returns `(mask, per_element_loss)`, each `(B,
    num_thresholds)` float32."""
    logits = tf.convert_to_tensor(logits)
    logits = tf.cast(logits, tf.float32)
    grades = tf.cast(tf.convert_to_tensor(grades), tf.float32)

    thresholds = tf.range(num_thresholds, dtype=tf.float32)
    grades_column = grades[:, tf.newaxis]

    mask = tf.cast(grades_column >= thresholds[tf.newaxis, :], tf.float32)
    target = tf.cast(grades_column > thresholds[tf.newaxis, :], tf.float32)
    per_element_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=logits)
    return mask, per_element_loss


def weighted_corn_loss_value(logits, grades, class_weights, num_thresholds=corn.NUM_THRESHOLDS):
    """The scalar weighted-CORN loss for one batch. Not a Keras `loss(y_true, y_pred)` callable
    itself -- see `make_weighted_corn_loss()` for that adapter; this is the plain function the
    tests exercise directly and `make_weighted_corn_loss()` wraps."""
    mask, per_element_loss = _corn_pairs(logits, grades, num_thresholds)
    masked_loss = per_element_loss * mask

    weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    num_classes = tf.shape(weights)[0]
    grade_indices = tf.cast(tf.convert_to_tensor(grades), tf.int32)
    # Ordinary training grades are always 0..num_classes-1 (corn.NUM_GRADES == 5); the clip is a
    # defensive bound only, so a corrupted label cannot index outside `weights` -- it never
    # silently reinterprets an out-of-range grade as a valid one for the mask/target computation
    # above, which already used the RAW grade value.
    grade_indices = tf.clip_by_value(grade_indices, 0, num_classes - 1)
    sample_weight = tf.gather(weights, grade_indices)  # (B,)

    weighted_numerator = tf.reduce_sum(masked_loss * sample_weight[:, tf.newaxis])
    # UNWEIGHTED denominator -- the total included (sample, task) pair count, identical to
    # corn.corn_loss's own `total_included`. This is what makes weight==1 reproduce corn.corn_loss
    # exactly, and what this module's docstring explicitly requires: NOT
    # `sum(weighted_loss) / sum(weighted_mask)`.
    unweighted_denominator = tf.reduce_sum(mask)
    return weighted_numerator / unweighted_denominator


def make_weighted_corn_loss(class_weights, num_thresholds=corn.NUM_THRESHOLDS):
    """Returns a Keras `loss(y_true, y_pred)` callable with the SAME argument convention as
    `joint_training_model.joint_corn_loss` (`y_true`=integer grades, `y_pred`=raw CORN logits),
    so it can be swapped in at `model.compile(loss=...)` in its place without touching
    `joint_training_model.py`. `class_weights`: a length-`NUM_GRADES` sequence of positive
    floats, e.g. `PREREGISTERED_CLASS_WEIGHTS`. Passing `[1.0] * NUM_GRADES` reproduces
    `joint_training_model.joint_corn_loss`/`corn.corn_loss` exactly (see this module's docstring
    and `tests/test_weighted_corn.py`)."""
    weights = tuple(float(w) for w in class_weights)

    def _loss(y_true, y_pred):
        return weighted_corn_loss_value(y_pred, y_true, weights, num_thresholds=num_thresholds)

    return _loss


class UnweightedCORNLoss(tf.keras.metrics.Metric):
    """Stateful Keras `Metric` reporting the plain, UNWEIGHTED `corn.corn_loss` value alongside a
    weighted training loss, so `val_corn_loss_unweighted` stays directly comparable across
    weighted and unweighted runs (and with the finalized RACAF/NO-RACAF reports' `val_loss`,
    which is this same unweighted quantity). A `Metric`, not a second `loss=`, so it never
    contributes a gradient -- exactly like `corn.CORNQuadraticWeightedKappa` is a metric, never a
    loss.

    Accumulates the TOTAL loss and the TOTAL included-pair count across every batch since the
    last `reset_state()` (matching `corn.corn_loss`'s own pooled reduction -- "sum of all
    included example/task losses / number of included example/task pairs" -- computed over the
    WHOLE accumulation period, not as a mean of per-batch means, which would over-weight a small
    final batch)."""

    def __init__(self, num_thresholds=corn.NUM_THRESHOLDS, name="corn_loss_unweighted", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_thresholds = num_thresholds
        self.total_loss = self.add_weight(name="total_loss", initializer="zeros", dtype=tf.float32)
        self.total_count = self.add_weight(name="total_count", initializer="zeros", dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask, per_element_loss = _corn_pairs(y_pred, y_true, self.num_thresholds)
        masked = per_element_loss * mask
        self.total_loss.assign_add(tf.reduce_sum(masked))
        self.total_count.assign_add(tf.reduce_sum(mask))

    def result(self):
        return self.total_loss / tf.maximum(self.total_count, 1.0)

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.total_count.assign(0.0)
