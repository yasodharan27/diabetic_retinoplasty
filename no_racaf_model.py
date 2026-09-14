"""
NO-RACAF joint model builder -- an ADDITIVE module, not a modification of any frozen file.

Extracts the exact NO-RACAF construction that has, until now, existed only inline in
`colab/notebooks/stage08_corn_classifier_racaf_ablation.ipynb`'s `[7]` cell (the executed,
committed RACAF ablation) into an importable, independently testable module, so a NEW caller
(the multi-seed improved-training experiment, `multiseed_runs.py`) can build the identical
NO-RACAF architecture without duplicating that cell's code a second time and without importing
from a notebook.

This is byte-for-byte the same architecture as the committed ablation notebook's `[7]` cell:
`joint_training_model.build_joint_model()` with RACAF removed and nothing put in its place,
plus `InertReliabilityConnection` so the `reliability` Input stays declared (the dataset
pipeline's contract -- `((stage5, stage6, reliability), grade)` -- is identical for RACAF and
NO-RACAF) without ever influencing the logits. No repository `.py` file is modified to produce
this: `joint_training_model.py`, `corn.py`, `local_feature_extraction_model.py`,
`swin_transformer.py`, `feature_fusion.py` and `racaf.py` are all imported unmodified.

`RACAF_ARCHITECTURE.md`/`PROJECT_CODE.md` remain authoritative: RACAF is this project's one
approved research innovation. This module exists only to let a *controlled ablation* of it be
built without RACAF, exactly as the finalized NO-RACAF experiment (`2026-09-13_04-11-07`) already
did from notebook-only code.
"""

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers

import corn
import feature_fusion
import joint_training_model as jtm
import local_feature_extraction_model
import swin_transformer

NO_RACAF_MODEL_NAME = "joint_stage05_08_no_racaf"

# Measured, not estimated -- see tests/test_no_racaf_model.py and
# RACAF_Ablation_Experiment_1_Report.md Sec 6.
EXPECTED_TRAINABLE_TENSORS = 405
EXPECTED_TRAINABLE_PARAMETERS = 43_043_336
REFERENCE_TRAINABLE_TENSORS = 409
REFERENCE_TRAINABLE_PARAMETERS = 43_338_506
RACAF_PARAMETERS_REMOVED = 295_170
RACAF_TENSORS_REMOVED = 4


class InertReliabilityConnection(layers.Layer):
    """Connects the `reliability` Input to the graph with EXACTLY zero influence on the logits.

    Identical to the class of the same name in the committed ablation notebook's `[7]` cell
    (`colab/notebooks/stage08_corn_classifier_racaf_ablation.ipynb`) -- reproduced here, not
    imported from the notebook, so it is usable from ordinary Python. See that cell's docstring
    for the full rationale: Keras' Functional API requires every declared Input to reach the
    output, and RACAF was `reliability`'s only consumer, so removing RACAF requires this
    parameter-free edge to keep the graph constructible. `x + 0.0` is exact in IEEE 754, so the
    logits are bit-for-bit unchanged and the prediction cannot depend on reliability."""

    def call(self, inputs):
        logits, reliability = inputs
        zero = tf.zeros_like(reliability[:, :1])        # (B, 1) exact zeros, from reliability
        return logits + tf.cast(zero, logits.dtype)     # broadcasts over the NUM_THRESHOLDS axis

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def build_no_racaf_joint_model():
    """`joint_training_model.build_joint_model()` with RACAF removed and nothing put in its
    place. Same unmodified sub-builders, same construction order (Stage 05, Stage 06, Stage 07,
    CORN -- RACAF's own `build_racaf_fusion()` is never called here), same three inputs, same
    `(B, 4)` CORN output. See this module's docstring for the exact correspondence with the
    committed ablation notebook."""
    stage5_input = Input(shape=jtm.STAGE5_INPUT_SHAPE, name="stage5_input")
    stage6_input = Input(shape=jtm.STAGE6_INPUT_SHAPE, name="stage6_input")
    reliability_input = Input(shape=(1,), name="reliability")   # declared, deliberately unused

    stage5_model = local_feature_extraction_model.build_local_feature_extractor()
    stage6_model = swin_transformer.create_dual_scale_swin_model()
    stage7_model = feature_fusion.build_adaptive_cross_attention()
    corn_model = corn.build_corn_model()

    local_features = stage5_model(stage5_input)
    global_features = stage6_model(stage6_input)
    fused_embedding = stage7_model([local_features, global_features])
    logits = corn_model(fused_embedding)       # RACAF removed: E goes straight into CORN
    logits = InertReliabilityConnection(name="no_racaf_inert_reliability")(
        [logits, reliability_input])

    return Model(inputs=[stage5_input, stage6_input, reliability_input], outputs=logits,
                 name=NO_RACAF_MODEL_NAME)


def build_no_racaf_joint_model_matched_init():
    """`build_no_racaf_joint_model()`, but constructing (and discarding) RACAF's own
    `racaf.build_racaf_fusion()` at the same point in the build order `joint_training_model.
    build_joint_model()` calls it -- so that, under an identical `keras.utils.set_random_seed()`
    call beforehand, Stage 05/06/07 and CORN consume the SAME sequence of random weight-
    initialisation draws in both arms. Keras draws each layer's initial weights when the layer is
    CONSTRUCTED, in construction order, from the global RNG state that `set_random_seed()` fixes
    -- so if RACAF's construction is simply skipped (as `build_no_racaf_joint_model()` does),
    every later layer's initializer call consumes a different position in that draw sequence than
    it would in the WITH-RACAF model, and the two arms are no longer matched at the same seed
    (`tests/test_no_racaf_model.py::test_matched_init_...` proves both the parity this function
    restores and the mismatch `build_no_racaf_joint_model()` alone would have).

    RACAF's own module is imported here ONLY to construct-and-discard it -- its parameters never
    enter the returned model, are never part of `trainable_variables`, and never receive a
    gradient. `RACAF_PARAMETERS_REMOVED`/`RACAF_TENSORS_REMOVED` and the parameter/tensor counts
    this module asserts elsewhere are unaffected."""
    import racaf  # local import: only this function needs it, keeping the ablation-identical
                  # build_no_racaf_joint_model() free of any RACAF import, exactly as the
                  # committed ablation notebook's [7] cell is.

    stage5_input = Input(shape=jtm.STAGE5_INPUT_SHAPE, name="stage5_input")
    stage6_input = Input(shape=jtm.STAGE6_INPUT_SHAPE, name="stage6_input")
    reliability_input = Input(shape=(1,), name="reliability")

    stage5_model = local_feature_extraction_model.build_local_feature_extractor()
    stage6_model = swin_transformer.create_dual_scale_swin_model()
    stage7_model = feature_fusion.build_adaptive_cross_attention()
    _discarded_racaf_model = racaf.build_racaf_fusion()  # construct, consume RNG draws, discard
    del _discarded_racaf_model
    corn_model = corn.build_corn_model()

    local_features = stage5_model(stage5_input)
    global_features = stage6_model(stage6_input)
    fused_embedding = stage7_model([local_features, global_features])
    logits = corn_model(fused_embedding)
    logits = InertReliabilityConnection(name="no_racaf_inert_reliability")(
        [logits, reliability_input])

    return Model(inputs=[stage5_input, stage6_input, reliability_input], outputs=logits,
                 name=NO_RACAF_MODEL_NAME)


def compile_no_racaf_model(model, optimizer=None):
    """Reuses `joint_training_model.compile_joint_model()` UNMODIFIED -- identical loss
    (`joint_corn_loss`) and reported metric (`CORNQuadraticWeightedKappa`) as every other arm.
    Callers needing the weighted CORN loss (`weighted_corn.py`) pass it as `optimizer=...`'s
    sibling by calling `model.compile(...)` themselves instead of this convenience wrapper."""
    return jtm.compile_joint_model(model, optimizer=optimizer)


def build_and_compile_no_racaf_model(mixed_precision=True, optimizer=None, matched_init=False,
                                     verbose=1):
    """The NO-RACAF twin of `joint_training_model.build_and_compile_joint_model()`: establishes
    the dtype policy FIRST, then builds, then compiles -- see that function's docstring for why
    this order is the only one under which mixed precision actually takes effect.

    `matched_init=True` uses `build_no_racaf_joint_model_matched_init()` instead of
    `build_no_racaf_joint_model()`, for cross-arm initialisation parity under a shared seed
    (`multiseed_runs.py`'s use). `matched_init=False` (the default) reproduces the exact,
    already-verified ablation notebook construction."""
    from training import enable_mixed_precision, model_precision_policies, expected_policy_name
    from training.trainer import precision_is_consistent

    policy = enable_mixed_precision(mixed_precision)
    model = (build_no_racaf_joint_model_matched_init() if matched_init
             else build_no_racaf_joint_model())
    compile_no_racaf_model(model, optimizer=optimizer)

    expected = expected_policy_name(mixed_precision)
    actual = model_precision_policies(model)
    if not precision_is_consistent(expected, actual):
        raise RuntimeError(
            f"No-RACAF model was built under {sorted(actual)} but the global policy is "
            f"'{policy.name}' and this call requested '{expected}'."
        )
    resolved = model.optimizer
    inner = getattr(resolved, "inner_optimizer", None)
    if expected == "mixed_float16" and inner is None:
        raise RuntimeError(
            "No-RACAF model is mixed_float16 but its optimizer was not wrapped in a "
            "LossScaleOptimizer. Without loss scaling, float16 gradients underflow to zero and "
            "training silently stops learning."
        )
    if verbose:
        print(f"No-RACAF model built under dtype policy '{expected}' (global '{policy.name}'), "
              f"optimizer {type(resolved).__name__}"
              f"{f'(inner={type(inner).__name__})' if inner else ''}.")
        print(f"  trainable tensors: {len(model.trainable_variables)}  "
              f"parameters: {sum(int(np.prod(v.shape)) for v in model.trainable_variables):,}")
    return model


def verify_no_racaf_model(model):
    """Structural + inertness proof, mirroring the ablation notebook's `[7]` model-check block.
    Raises on the first failing check; returns the checked parameter/tensor counts on success."""
    parameters = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    tensors = len(model.trainable_variables)
    layer_names = {layer.name for layer in model.layers}
    variable_paths = " ".join(v.path for v in model.trainable_variables).lower()
    input_names = {tensor.name.split(":")[0] for tensor in model.inputs}
    inert_layers = [layer for layer in model.layers if isinstance(layer, InertReliabilityConnection)]

    probe_rng = np.random.default_rng(0)
    probe5 = probe_rng.random((2, *jtm.STAGE5_INPUT_SHAPE), dtype=np.float32)
    probe6 = probe_rng.random((2, *jtm.STAGE6_INPUT_SHAPE), dtype=np.float32)
    logits_r0 = np.asarray(model.predict_on_batch([probe5, probe6, np.zeros((2, 1), np.float32)]))
    logits_r1 = np.asarray(model.predict_on_batch([probe5, probe6, np.ones((2, 1), np.float32)]))
    reliability_max_abs_diff = float(np.max(np.abs(
        logits_r0.astype(np.float64) - logits_r1.astype(np.float64))))

    checks = {
        f"model is named {NO_RACAF_MODEL_NAME}": model.name == NO_RACAF_MODEL_NAME,
        f"{EXPECTED_TRAINABLE_TENSORS} trainable tensors": tensors == EXPECTED_TRAINABLE_TENSORS,
        f"{EXPECTED_TRAINABLE_PARAMETERS:,} trainable parameters": parameters == EXPECTED_TRAINABLE_PARAMETERS,
        "NO racaf_fusion layer in the graph": "racaf_fusion" not in layer_names,
        "NO trainable variable belongs to RACAF": "racaf" not in variable_paths,
        "NO reliability_gate / global_projection variable": not any(
            key in variable_paths for key in ("reliability_gate", "global_projection")),
        "Stage 05 local features present":
            any(name.startswith("local_feature_extraction") for name in layer_names),
        "Stage 06 dual-scale Swin present":
            any(name.startswith("global_feature_extraction") for name in layer_names),
        "Stage 07 adaptive cross-attention present":
            any(name.startswith("feature_fusion") for name in layer_names),
        "CORN head present": "corn" in layer_names,
        "reliability input still declared": "reliability" in input_names,
        "reliability is CONNECTED via exactly one inert layer": len(inert_layers) == 1,
        "the inert reliability connection has no weights at all":
            bool(inert_layers) and not inert_layers[0].weights,
        "reliability is INERT: r=0 and r=1 give bit-identical logits":
            np.array_equal(logits_r0, logits_r1),
        "reliability inertness max |diff| is exactly 0.0": reliability_max_abs_diff == 0.0,
        "output is (batch, num_thresholds) CORN logits":
            tuple(model.outputs[0].shape) == (None, corn.NUM_THRESHOLDS),
        f"exactly RACAF's {RACAF_PARAMETERS_REMOVED:,} parameters were removed":
            REFERENCE_TRAINABLE_PARAMETERS - parameters == RACAF_PARAMETERS_REMOVED,
        f"exactly RACAF's {RACAF_TENSORS_REMOVED} tensors were removed":
            REFERENCE_TRAINABLE_TENSORS - tensors == RACAF_TENSORS_REMOVED,
    }
    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"No-RACAF model verification failed: {failed}")
    return {"trainable_parameters": parameters, "trainable_tensors": tensors,
            "reliability_max_abs_diff": reliability_max_abs_diff}
