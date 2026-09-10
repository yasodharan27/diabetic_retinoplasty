"""
Joint model composition for the Stage 05-08 + RACAF joint training run.

Authoritative design: `JOINT_TRAINING_ARCHITECTURE.md`. This module composes each stage's own,
already-approved, ALREADY-IMPLEMENTED `build_*()` function into one functional `keras.Model` --
it does not redefine, duplicate, or modify any stage's architecture:

    local_feature_extraction_model.build_local_feature_extractor()   -- Stage 05
    swin_transformer.create_dual_scale_swin_model()                  -- Stage 06
    feature_fusion.build_adaptive_cross_attention()                  -- Stage 07
    racaf.build_racaf_fusion()                                       -- RACAF
    corn.build_corn_model()                                          -- CORN

Stage 03/04 never appear in this graph at all -- they are frozen, and (per
`JOINT_TRAINING_ARCHITECTURE.md` §9-13/§19) run entirely inside `joint_training_dataset.py`'s
data/cache layer, producing plain NumPy arrays before anything reaches this model's `Input`
tensors. That is a stronger guarantee than any in-graph `stop_gradient` could provide: Stage 03/04
parameters are not `tf.Variable`s in this graph at all, so no optimizer, gradient tape, or
`model.fit()` call built from `build_joint_model()` can touch them, structurally.

Only Stage 05, Stage 06, Stage 07, RACAF's two learned pieces (`w_g,b_g,W_r,b_r`), and CORN are
trainable in the returned model -- exactly the boundary `JOINT_TRAINING_ARCHITECTURE.md` §4/§19
fixes. The training loss is `corn.corn_loss` alone (`compile_joint_model`) -- no auxiliary loss.
`compile_joint_model` also attaches `corn.CORNQuadraticWeightedKappa` as a reported METRIC (not
a loss) so `monitor="val_QWK", mode="max"` checkpoint selection (§23) has a real value to read.

Checkpoint format: weights-only (`save_joint_model_weights`/`load_joint_model_weights`), per
`JOINT_TRAINING_ARCHITECTURE.md` §25 -- Stage 06's underlying Swin layer classes have no
`get_config()`, so a full single-file `.keras` save of a model embedding Stage 06 is not reliably
reconstructable on load; `training.TrainingConfig.save_weights_only` already defaults to `True`
for exactly this reason, project-wide. This module does not change that.
"""

import os

import numpy as np
import tensorflow as tf
from keras import Input, Model

import corn
import feature_fusion
import local_feature_extraction_model
import racaf
import swin_transformer

STAGE5_INPUT_SHAPE = local_feature_extraction_model.DEFAULT_INPUT_SHAPE  # (512, 512, 8)
STAGE6_INPUT_SHAPE = swin_transformer.DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE  # (256, 256, 3)


def build_joint_model():
    """Composes Stage 05 -> Stage 06 -> Stage 07 -> RACAF -> CORN into one functional
    `keras.Model`, `[stage5_input, stage6_input, reliability] -> logits`.

    Each sub-model is built via its own, unmodified `build_*()` function and called as a Keras
    Functional-API layer on the previous stage's output tensor -- no architecture is
    reimplemented or duplicated here. Returns an **uncompiled** model (see `compile_joint_model`).
    Trainable by construction: every one of the five sub-models keeps its own default
    `trainable=True` (Keras Functional composition does not freeze a sub-model unless told to,
    and nothing here tells it to) -- Stage 05/06/07, RACAF's `w_g,b_g,W_r,b_r`, and CORN are all
    trainable variables of the returned model; Stage 03/04 are not part of this graph at all.
    """
    stage5_input = Input(shape=STAGE5_INPUT_SHAPE, name="stage5_input")
    stage6_input = Input(shape=STAGE6_INPUT_SHAPE, name="stage6_input")
    reliability_input = Input(shape=(1,), name="reliability")

    stage5_model = local_feature_extraction_model.build_local_feature_extractor()
    stage6_model = swin_transformer.create_dual_scale_swin_model()
    stage7_model = feature_fusion.build_adaptive_cross_attention()
    racaf_model = racaf.build_racaf_fusion()
    corn_model = corn.build_corn_model()

    local_features = stage5_model(stage5_input)
    global_features = stage6_model(stage6_input)
    fused_embedding = stage7_model([local_features, global_features])
    reliability_fused = racaf_model([fused_embedding, global_features, reliability_input])
    logits = corn_model(reliability_fused)

    return Model(
        inputs=[stage5_input, stage6_input, reliability_input],
        outputs=logits,
        name="joint_stage05_08_racaf",
    )


def joint_corn_loss(y_true, y_pred):
    """Keras `loss(y_true, y_pred)` adapter over `corn.corn_loss(logits, grades)` -- an
    argument-order adapter only, not a new loss. `y_pred` is the joint model's raw `(B,4)` CORN
    logits; `y_true` is the integer APTOS grade. No focal loss, Dice loss, segmentation loss,
    class weighting, label smoothing, or auxiliary term is added."""
    return corn.corn_loss(y_pred, y_true)


def compile_joint_model(model, optimizer=None):
    """Compiles `model` (from `build_joint_model()`) with exactly `corn.corn_loss` as the
    training objective -- the ONLY supervised loss for this joint training run
    (`JOINT_TRAINING_ARCHITECTURE.md` §21) -- plus `corn.CORNQuadraticWeightedKappa` as a
    reported METRIC, never a second loss, so Keras's own `logs` dict actually contains
    `"QWK"`/`"val_QWK"` for `JOINT_TRAINING_ARCHITECTURE.md` §23's `monitor="val_QWK",
    mode="max"` checkpoint-selection policy to observe during `model.fit()` -- previously this
    function compiled with no metric at all, so that monitor string had nothing to read and
    `ModelCheckpoint`/`EarlyStopping`/`ReduceLROnPlateau` would have silently skipped every
    epoch (see `tests/test_joint_training.py`'s `CORNQWKJointIntegrationTests`). `optimizer`
    defaults to a plain `Adam()` if not supplied; this function makes no other training-loop
    decision (batch size, callbacks, schedule) -- those belong to the notebook /
    `training.Trainer`, not this module."""
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=joint_corn_loss, metrics=[corn.CORNQuadraticWeightedKappa()])
    return model


def build_and_compile_joint_model(mixed_precision=True, optimizer=None, verbose=1):
    """Establish the dtype policy, THEN build, THEN compile -- the only order in
    which mixed precision actually takes effect. Use this instead of calling
    `build_joint_model()` + `compile_joint_model()` by hand.

    Keras 3 reads the global dtype policy in each layer's CONSTRUCTOR and decides
    whether to wrap the optimizer in a `LossScaleOptimizer` when the model is
    COMPILED. Setting the policy afterwards -- which is what happens when a model
    is built first and only later handed to `training.Trainer`, whose `prepare()`
    calls `enable_mixed_precision()` -- changes the global policy and nothing
    about the model. Verified on TF 2.21.0 / Keras 3.15.1: a model built under
    `float32` and then exposed to a `mixed_float16` global policy still reports
    `compute=float32, variable=float32` on every weighted layer and still carries
    a bare `Adam`, so the whole run trains in float32 with no loss scaling while
    every configuration flag says otherwise.

    `mixed_float16` keeps VARIABLES in float32 by design (`compute_dtype=float16`,
    `variable_dtype=float32`) -- float32 trainable tensors and float32 gradients
    are correct mixed-precision behaviour, not a symptom of a misconfiguration.
    What distinguishes a working setup is that the layers' compute dtype is
    float16 and the optimizer is a `LossScaleOptimizer`; both are asserted here.

    Returns the compiled model. Raises if the resulting configuration is not the
    one requested, rather than training a silently downgraded model."""
    from training import enable_mixed_precision, model_precision_policies, expected_policy_name
    from training.trainer import precision_is_consistent

    policy = enable_mixed_precision(mixed_precision)
    model = build_joint_model()
    compile_joint_model(model, optimizer=optimizer)

    expected = expected_policy_name(mixed_precision)
    actual = model_precision_policies(model)
    if not precision_is_consistent(expected, actual):
        raise RuntimeError(
            f"Joint model was built under {sorted(actual)} but the global policy is "
            f"'{policy.name}' and this call requested '{expected}'. Something set the dtype "
            "policy between enable_mixed_precision() and build_joint_model()."
        )

    resolved = model.optimizer
    inner = getattr(resolved, "inner_optimizer", None)
    if expected == "mixed_float16" and inner is None:
        raise RuntimeError(
            "Model is mixed_float16 but its optimizer was not wrapped in a LossScaleOptimizer. "
            "Without loss scaling, float16 gradients underflow to zero and training silently "
            "stops learning."
        )

    if verbose:
        print(f"Joint model built under dtype policy '{expected}' "
              f"(global '{policy.name}'), optimizer "
              f"{type(resolved).__name__}{f'(inner={type(inner).__name__})' if inner else ''}.")
        print(f"  trainable tensors: {len(model.trainable_variables)}  "
              f"parameters: {sum(int(np.prod(v.shape)) for v in model.trainable_variables):,}")
    return model


# --- Checkpoint infrastructure -- weights-only, path is always caller-supplied. ---
#
# Deliberately takes no default path and makes no Drive/local assumption of its own: the actual
# persistent location (`experiments/FinalClassification/<timestamp>/checkpoints/` on Drive) is
# resolved by the caller via the EXISTING `colab/common/experiment_manager.py` +
# `colab_config.DRIVE.experiment_dir("FinalClassification")` infrastructure (unmodified), exactly
# as `training.TrainingConfig.run_dir` is already caller-supplied, never hardcoded, project-wide.

def save_joint_model_weights(model, path):
    """Saves `model`'s weights (weights-only, per this module's docstring) to `path`. Does not
    create or resolve any directory beyond `path`'s own parent -- the caller decides where."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save_weights(path)
    return path


def load_joint_model_weights(path):
    """Rebuilds a fresh joint architecture via `build_joint_model()` and loads `path`'s weights
    into it -- the same "rebuild then load_weights" pattern
    `swin_transformer.GlobalFeatureExtractionStage.load()` already uses for Stage 06 alone,
    extended here to the whole joint graph. Returns the loaded, uncompiled model."""
    model = build_joint_model()
    model.load_weights(path)
    return model
