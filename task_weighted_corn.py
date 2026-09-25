"""
Per-task-weighted CORN loss -- an ADDITIVE module. `corn.py` and `weighted_corn.py` are not modified.

Purpose: the task-3 weighting vs mass-matched placebo experiment
(`colab/notebooks/task3_weighting_placebo_experiment.ipynb`). It asks whether specifically increasing
the loss pressure on CORN's conditional task 3 (P(y > 3 | y >= 3), i.e. Grade 3 vs Grade 4 among
images of grade >= 3) improves the model's Grade-3-vs-Grade-4 posterior `p_cond_3 = sigmoid(z_3)`
beyond the non-specific effect of perturbing the loss by the same amount.

The loss is `weighted_corn`'s own construction -- the SAME `_corn_pairs` mask/target/per-element
binary cross-entropy, the SAME class-weight lookup, the SAME UNWEIGHTED included-pair denominator --
with one extra per-task factor `beta_k` on the numerator only:

    numerator   = sum_i sum_k  beta_k * w[y_i] * m_ik * loss_ik
    denominator = sum_i sum_k  m_ik                                 -- UNWEIGHTED, unchanged
    loss        = numerator / denominator

With every `beta_k == 1` this is `weighted_corn.weighted_corn_loss_value` bit-for-bit (a
multiplication by 1.0 is exact in IEEE 754); `tests/test_task_weighted_corn.py` proves it, including
under float16 logits. A `beta_k` touches only task `k`'s terms, so its gradient reaches logit `z_k`
alone: the other three logits receive exactly the baseline gradient.

Weighted loss MASS of task k (the pre-registered matching criterion) is its expected numerator weight
over the training set:

    M_k = sum_c  n_c * w_c * 1[c >= k]          (n_c = TRAINING class counts, w_c = class weights)

Scaling task k by beta adds (beta - 1) * M_k. The treatment (task 3, beta_3 = 2) adds M_3; the
placebo scales task 1 by beta_1 = 1 + M_3 / M_1, which adds exactly the same mass while acting on a
boundary (Grade >= 2 vs Grade 1) far from Grade 3/4. Every quantity here is derived from the pinned
TRAINING counts (`weighted_corn.PREREGISTERED_TRAIN_COUNTS`); no validation count enters.
"""

import numpy as np
import tensorflow as tf

import corn
import weighted_corn

TREATMENT_TASK = 3
PLACEBO_TASK = 1
TREATMENT_BETA = 2.0
UNIT_TASK_WEIGHTS = (1.0,) * corn.NUM_THRESHOLDS


def task_loss_masses(train_counts=weighted_corn.PREREGISTERED_TRAIN_COUNTS,
                     class_weights=weighted_corn.PREREGISTERED_CLASS_WEIGHTS):
    """`M_k` for every task k: sum over training classes c >= k of count * class weight."""
    counts = np.asarray(train_counts, dtype=np.float64)
    weights = np.asarray(class_weights, dtype=np.float64)
    if counts.shape != (corn.NUM_GRADES,) or weights.shape != (corn.NUM_GRADES,):
        raise ValueError(f"expected {corn.NUM_GRADES} counts and weights, got "
                         f"{counts.shape} / {weights.shape}")
    return tuple(float(np.sum(counts[k:] * weights[k:])) for k in range(corn.NUM_THRESHOLDS))


def mass_matched_beta(target_task, reference_task, reference_beta, masses=None):
    """The factor on `target_task` that adds the same mass as `reference_beta` on `reference_task`:
    (beta_t - 1) * M_t == (beta_r - 1) * M_r."""
    masses = task_loss_masses() if masses is None else masses
    return 1.0 + (reference_beta - 1.0) * masses[reference_task] / masses[target_task]


def task_weights_for(task, beta):
    weights = list(UNIT_TASK_WEIGHTS)
    weights[task] = float(beta)
    return tuple(weights)


PLACEBO_BETA = mass_matched_beta(PLACEBO_TASK, TREATMENT_TASK, TREATMENT_BETA)
TREATMENT_TASK_WEIGHTS = task_weights_for(TREATMENT_TASK, TREATMENT_BETA)
PLACEBO_TASK_WEIGHTS = task_weights_for(PLACEBO_TASK, PLACEBO_BETA)

# Import-time cross-check against the values stated in the pre-registration request, so a change to
# the counts, the class weights or the formula can never silently produce a different placebo.
_MASSES = task_loss_masses()
if (round(_MASSES[TREATMENT_TASK], 1), round(_MASSES[PLACEBO_TASK], 1), round(PLACEBO_BETA, 4)) \
        != (731.2, 1928.5, 1.3792):
    raise RuntimeError(f"Task masses {_MASSES} / placebo beta {PLACEBO_BETA} differ from the "
                       "pre-registered 731.2 / 1928.5 / 1.3792.")


def _validated_task_weights(task_weights, num_thresholds):
    values = tuple(float(v) for v in task_weights)
    if len(values) != num_thresholds:
        raise ValueError(f"need {num_thresholds} task weights, got {len(values)}")
    if any(not np.isfinite(v) or v <= 0 for v in values):
        raise ValueError(f"task weights must be finite and positive, got {values}")
    return values


def task_weighted_corn_loss_value(logits, grades, class_weights, task_weights,
                                  num_thresholds=corn.NUM_THRESHOLDS):
    """Scalar per-task-weighted CORN loss for one batch. Mirrors
    `weighted_corn.weighted_corn_loss_value` line for line, plus the per-task factor."""
    task_weights = _validated_task_weights(task_weights, num_thresholds)
    mask, per_element_loss = weighted_corn._corn_pairs(logits, grades, num_thresholds)
    masked_loss = per_element_loss * mask

    weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
    num_classes = tf.shape(weights)[0]
    grade_indices = tf.cast(tf.convert_to_tensor(grades), tf.int32)
    grade_indices = tf.clip_by_value(grade_indices, 0, num_classes - 1)
    sample_weight = tf.gather(weights, grade_indices)  # (B,)

    per_task = tf.constant(task_weights, dtype=tf.float32)  # (T,)
    weighted_numerator = tf.reduce_sum(
        masked_loss * sample_weight[:, tf.newaxis] * per_task[tf.newaxis, :])
    unweighted_denominator = tf.reduce_sum(mask)  # identical to weighted_corn / corn.corn_loss
    return weighted_numerator / unweighted_denominator


def make_task_weighted_corn_loss(class_weights, task_weights, num_thresholds=corn.NUM_THRESHOLDS):
    """Keras `loss(y_true, y_pred)` adapter, same argument convention as
    `weighted_corn.make_weighted_corn_loss` (y_true = integer grades, y_pred = raw CORN logits)."""
    class_weights = tuple(float(w) for w in class_weights)
    task_weights = _validated_task_weights(task_weights, num_thresholds)

    def _loss(y_true, y_pred):
        return task_weighted_corn_loss_value(y_pred, y_true, class_weights, task_weights,
                                             num_thresholds=num_thresholds)

    return _loss
