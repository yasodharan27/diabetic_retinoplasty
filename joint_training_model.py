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
    (`JOINT_TRAINING_ARCHITECTURE.md` §21). `optimizer` defaults to a plain `Adam()` if not
    supplied; this function makes no other training-loop decision (batch size, callbacks,
    schedule) -- those belong to the notebook / `training.Trainer`, not this module."""
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss=joint_corn_loss)
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
