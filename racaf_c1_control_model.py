"""
RACAF C1 decomposition-control model -- an ADDITIVE module, not a modification of any frozen file.

Research-design context (not the final improved-training experiment): the pre-registered RACAF vs
NO_RACAF comparison confounds two simultaneous architectural changes --

    RACAF: gate = sigmoid(w_g * r + b_g)      F = gate*E + (1-gate)*Ghat
    where Ghat = Dense(d_model)(GAP(G))

  1. the reliability-CONDITIONED gate (`w_g, b_g`, 2 trainable parameters, driven by `r`), and
  2. the added global-readout pathway `Ghat = Dense(d_model)(GAP(G))` (295,168 trainable
     parameters) that RACAF's fusion blends against `E`, which NO_RACAF has never had at all.

C1 isolates change 1 alone: it keeps `Ghat` and the blend structure exactly as RACAF's, and
replaces `gate = sigmoid(w_g*r + b_g)` with `gate = sigmoid(b_g)` -- a single learnable scalar,
completely independent of the reliability input `r`.

    C1:    gate = sigmoid(b_g)                F = gate*E + (1-gate)*Ghat        (r removed)

Once C1/seed_42 is trained under the identical protocol and compared against the EXISTING
(unmodified) RACAF/seed_42 and NO_RACAF/seed_42 checkpoints, the three-way comparison decomposes:

    RACAF   - C1       = value attributable to reliability CONDITIONING (the actual RACAF hypothesis)
    C1      - NO_RACAF = value attributable to the added Ghat global-readout pathway alone

`racaf.py`, `joint_training_model.py`, `no_racaf_model.py`, `corn.py`, `feature_fusion.py`,
`local_feature_extraction_model.py`, `swin_transformer.py` and every other frozen module are
imported unmodified; nothing here changes RACAF's own approved architecture or the finalized
RACAF/NO_RACAF checkpoints. The `reliability` Input is kept declared in C1's graph (dataset-
contract parity -- the shared data pipeline always yields `((stage5, stage6, reliability), grade)`
-- mirroring `no_racaf_model.InertReliabilityConnection`'s established "connected, zero influence"
idiom for that SAME Input) but C1's gate never reads `r`'s VALUE, only `tf.shape(r)` to broadcast
the constant gate to the batch.
"""

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers
from keras.saving import register_keras_serializable

import corn
import feature_fusion
import joint_training_model as jtm
import local_feature_extraction_model
import racaf
import swin_transformer

C1_MODEL_NAME = "joint_stage05_08_racaf_c1_constant_gate"

# Measured, not estimated -- see tests/test_racaf_c1_control_model.py. RACAF's own
# `reliability_gate` is a `Dense(1)` on `r`: kernel `(1,1)` + bias `(1,)` = 2 params, 2 tensors.
# C1's `ConstantGate` keeps only the bias-equivalent `b_g:(1,)` = 1 param, 1 tensor -- removing the
# kernel is the ENTIRE, and only, architectural difference between C1 and RACAF.
RACAF_REFERENCE_TRAINABLE_TENSORS = 409             # no_racaf_model.REFERENCE_TRAINABLE_TENSORS
RACAF_REFERENCE_TRAINABLE_PARAMETERS = 43_338_506   # no_racaf_model.REFERENCE_TRAINABLE_PARAMETERS
GATE_KERNEL_PARAMETERS_REMOVED = 1
GATE_KERNEL_TENSORS_REMOVED = 1
EXPECTED_TRAINABLE_TENSORS = RACAF_REFERENCE_TRAINABLE_TENSORS - GATE_KERNEL_TENSORS_REMOVED
EXPECTED_TRAINABLE_PARAMETERS = RACAF_REFERENCE_TRAINABLE_PARAMETERS - GATE_KERNEL_PARAMETERS_REMOVED


@register_keras_serializable(package="racaf_c1")
class ConstantGate(layers.Layer):
    """`gate = sigmoid(b_g)` -- ONE learnable scalar bias, read from no input. `inputs` (`r`,
    shape `(B,1)`) is consumed only via `tf.shape` to broadcast the gate to the batch; its VALUE
    never reaches the returned tensor, so the reliability signal is completely removed from C1's
    computational path while `r` can still be declared and fed real data every epoch.

    Named `bias`, not `b_g`: `multiseed_runs.build_optimizer()`'s AdamW weight-decay exclusion
    (every variable whose name contains "bias", "gamma", or "beta") is applied identically to
    RACAF, NO_RACAF and C1 with no arm-specific exception -- RACAF's own `b_g` is swept in by that
    same generic rule because Keras' default `Dense` bias variable is named `"bias"`. Naming this
    weight anything else would silently exempt C1's only trainable parameter from a policy every
    other bias in the network (RACAF's included) receives, breaking "same AdamW regularization"."""

    def build(self, input_shape):
        self.bias = self.add_weight(name="bias", shape=(1,), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        bias = tf.cast(self.bias, inputs.dtype)
        return tf.sigmoid(tf.broadcast_to(bias, (batch_size, 1)))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


@register_keras_serializable(package="racaf_c1")
class _OneMinusC1(layers.Layer):
    """Computes `1 - x`. Reproduced here (not imported from `racaf.py`'s own `_OneMinus`) so this
    module stays independently loadable/testable, mirroring `no_racaf_model.py`'s identical choice
    to reproduce `InertReliabilityConnection`-adjacent helpers rather than import module-private
    symbols across files."""

    def call(self, inputs):
        return 1.0 - inputs


def build_racaf_fusion_c1(d_model=racaf.D_MODEL, global_tokens=racaf.GLOBAL_TOKENS,
                          global_channels=racaf.GLOBAL_CHANNELS, global_projection_layer=None):
    """`racaf.build_racaf_fusion()` with `gate = sigmoid(w_g*r + b_g)` replaced by
    `gate = sigmoid(b_g)` -- everything else (the `Ghat` global-readout projection, the
    `Multiply`/`Add` blend, the output activation) byte-for-byte identical. Returns an
    **uncompiled** `keras.Model`, `[E, G, r] -> F`, matching RACAF's own input/output contract
    exactly so it drops into `build_c1_joint_model()` the same way `racaf.build_racaf_fusion()`
    drops into `joint_training_model.build_joint_model()`.

    `global_projection_layer`, if given, is an ALREADY-BUILT `Dense(d_model)` layer object reused
    as-is (Keras' standard shared-layer pattern -- calling a built layer again on a new tensor
    reuses its existing weights and draws no new random initialisation) instead of constructing a
    fresh one -- see `build_c1_joint_model_matched_init()`, which passes RACAF's own
    `global_projection` layer here so C1's `Ghat` pathway starts from bit-identical initial
    weights to RACAF's, at zero extra RNG-stream cost."""
    e_input = Input(shape=(d_model,), name="E")
    g_input = Input(shape=(global_tokens, global_channels), name="G")
    r_input = Input(shape=(1,), name="r")   # declared for dataset-contract parity; VALUE unused below

    gap_g = layers.GlobalAveragePooling1D(name="gap_g")(g_input)
    projection_layer = global_projection_layer or layers.Dense(d_model, name="global_projection")
    g_hat = projection_layer(gap_g)

    gate = ConstantGate(name="c1_constant_gate")(r_input)
    one_minus_gate = _OneMinusC1(name="one_minus_gate")(gate)

    gated_e = layers.Multiply(name="gated_e")([e_input, gate])
    gated_g_hat = layers.Multiply(name="gated_g_hat")([g_hat, one_minus_gate])
    fused = layers.Add(name="fusion")([gated_e, gated_g_hat])

    outputs = layers.Activation("linear", dtype="float32", name="racaf_output")(fused)
    return Model(inputs=[e_input, g_input, r_input], outputs=outputs, name="racaf_fusion_c1")


def build_c1_joint_model():
    """`joint_training_model.build_joint_model()` with RACAF's fusion replaced by
    `build_racaf_fusion_c1()` -- same sub-builders, same construction order, same three Inputs,
    same `(B,4)` CORN output as every other arm (RACAF, NO_RACAF)."""
    stage5_input = Input(shape=jtm.STAGE5_INPUT_SHAPE, name="stage5_input")
    stage6_input = Input(shape=jtm.STAGE6_INPUT_SHAPE, name="stage6_input")
    reliability_input = Input(shape=(1,), name="reliability")

    stage5_model = local_feature_extraction_model.build_local_feature_extractor()
    stage6_model = swin_transformer.create_dual_scale_swin_model()
    stage7_model = feature_fusion.build_adaptive_cross_attention()
    c1_fusion_model = build_racaf_fusion_c1()
    corn_model = corn.build_corn_model()

    local_features = stage5_model(stage5_input)
    global_features = stage6_model(stage6_input)
    fused_embedding = stage7_model([local_features, global_features])
    reliability_fused = c1_fusion_model([fused_embedding, global_features, reliability_input])
    logits = corn_model(reliability_fused)

    return Model(inputs=[stage5_input, stage6_input, reliability_input], outputs=logits,
                name=C1_MODEL_NAME)


def build_c1_joint_model_matched_init():
    """`build_c1_joint_model()`, but constructing RACAF's own `racaf.build_racaf_fusion()` at the
    same point in the build order `build_joint_model()` calls it, then reusing that reference
    model's `global_projection` layer object (not its `reliability_gate`) inside C1's own fusion,
    before building CORN -- so that, under an identical `keras.utils.set_random_seed()` call
    beforehand, Stage 05/06/07 and CORN consume the SAME sequence of random weight-initialisation
    draws as RACAF at the same run seed, AND C1's `Ghat` pathway starts from bit-identical initial
    weights to RACAF's own `Ghat`.

    An earlier version of this function DISCARDED the reference `racaf.build_racaf_fusion()`
    entirely (mirroring `no_racaf_model.build_no_racaf_joint_model_matched_init()`'s pattern) and
    built C1's fusion with a brand-new `Dense(d_model)`. That is WRONG for C1 specifically: unlike
    NO_RACAF (which puts nothing in RACAF's place, so CORN is built immediately after the discard
    -- exactly matching RACAF's own draw count), C1 puts a REAL fusion model in RACAF's place, and
    a fresh `Dense(d_model)` there consumes 2 more RNG draws (its own kernel + bias) before CORN
    is built, shifting CORN's initial weights away from RACAF's. Reusing the reference model's
    already-built `global_projection` layer object (Keras' standard shared-layer pattern -- a
    second call to a built layer reuses its weights and draws nothing new) consumes zero extra
    draws, restoring exact parity. Caught by, and verified against, a real failing test first --
    see `tests/test_racaf_c1_control_model.py::CrossArmInitializationParityTests`."""
    stage5_input = Input(shape=jtm.STAGE5_INPUT_SHAPE, name="stage5_input")
    stage6_input = Input(shape=jtm.STAGE6_INPUT_SHAPE, name="stage6_input")
    reliability_input = Input(shape=(1,), name="reliability")

    stage5_model = local_feature_extraction_model.build_local_feature_extractor()
    stage6_model = swin_transformer.create_dual_scale_swin_model()
    stage7_model = feature_fusion.build_adaptive_cross_attention()
    reference_racaf_fusion = racaf.build_racaf_fusion()   # consumes RACAF's exact 4 draws
    c1_fusion_model = build_racaf_fusion_c1(
        global_projection_layer=reference_racaf_fusion.get_layer("global_projection"))
    del reference_racaf_fusion   # its reliability_gate is discarded; global_projection lives on,
                                 # now owned by c1_fusion_model, which still references the layer
    corn_model = corn.build_corn_model()

    local_features = stage5_model(stage5_input)
    global_features = stage6_model(stage6_input)
    fused_embedding = stage7_model([local_features, global_features])
    reliability_fused = c1_fusion_model([fused_embedding, global_features, reliability_input])
    logits = corn_model(reliability_fused)

    return Model(inputs=[stage5_input, stage6_input, reliability_input], outputs=logits,
                name=C1_MODEL_NAME)


def verify_c1_model(model):
    """Structural + r-inertness-of-the-GATE proof -- the whole point of C1. Unlike NO_RACAF's
    inertness proof (the WHOLE reliability path is inert), C1's gate is the ONLY thing proven
    inert to `r`: `Ghat` and the blend itself are exactly RACAF's, so varying `r` must leave the
    scalar gate value (and therefore the fused output) bit-identical, while the model still has a
    real, non-degenerate `Ghat` pathway wired in from `G`. Raises on the first failing check;
    returns the checked parameter/tensor counts and the learned gate value on success."""
    parameters = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    tensors = len(model.trainable_variables)
    layer_names = {layer.name for layer in model.layers}
    variable_paths = " ".join(v.path for v in model.trainable_variables).lower()

    probe_rng = np.random.default_rng(0)
    probe5 = probe_rng.random((2, *jtm.STAGE5_INPUT_SHAPE), dtype=np.float32)
    probe6 = probe_rng.random((2, *jtm.STAGE6_INPUT_SHAPE), dtype=np.float32)
    logits_r0 = np.asarray(model.predict_on_batch([probe5, probe6, np.zeros((2, 1), np.float32)]))
    logits_r1 = np.asarray(model.predict_on_batch([probe5, probe6, np.ones((2, 1), np.float32)]))
    reliability_max_abs_diff = float(np.max(np.abs(
        logits_r0.astype(np.float64) - logits_r1.astype(np.float64))))

    # Fetched defensively -- a wrong/tampered model (e.g. the actual RACAF model, which has no
    # `racaf_fusion_c1` sub-model at all) must fail as an ordinary, informative check below, never
    # crash this function with a raw Keras `ValueError` before the checks dict is even built.
    gate_layer = None
    if "racaf_fusion_c1" in layer_names:
        fusion_submodel = model.get_layer("racaf_fusion_c1")
        if "c1_constant_gate" in {layer.name for layer in fusion_submodel.layers}:
            gate_layer = fusion_submodel.get_layer("c1_constant_gate")
    gate_value = float(tf.sigmoid(gate_layer.bias).numpy()[0]) if gate_layer is not None else None

    checks = {
        f"model is named {C1_MODEL_NAME}": model.name == C1_MODEL_NAME,
        f"{EXPECTED_TRAINABLE_TENSORS} trainable tensors": tensors == EXPECTED_TRAINABLE_TENSORS,
        f"{EXPECTED_TRAINABLE_PARAMETERS:,} trainable parameters": parameters == EXPECTED_TRAINABLE_PARAMETERS,
        "NO RACAF r-dependent reliability_gate variable present": "reliability_gate" not in variable_paths,
        "the C1 constant-gate sub-layer is present": gate_layer is not None,
        "the constant gate has exactly ONE weight (bias)":
            gate_layer is not None and len(gate_layer.trainable_weights) == 1,
        "global_projection (Ghat pathway) variable IS present": "global_projection" in variable_paths,
        "Stage 05 local features present":
            any(name.startswith("local_feature_extraction") for name in layer_names),
        "Stage 06 dual-scale Swin present":
            any(name.startswith("global_feature_extraction") for name in layer_names),
        "Stage 07 adaptive cross-attention present":
            any(name.startswith("feature_fusion") for name in layer_names),
        "CORN head present": "corn" in layer_names,
        "the gate is INERT to r: r=0 and r=1 give bit-identical logits":
            np.array_equal(logits_r0, logits_r1),
        "gate inertness-to-r max |diff| is exactly 0.0": reliability_max_abs_diff == 0.0,
        "output is (batch, num_thresholds) CORN logits":
            tuple(model.outputs[0].shape) == (None, corn.NUM_THRESHOLDS),
        f"exactly {GATE_KERNEL_PARAMETERS_REMOVED} parameter removed vs RACAF":
            RACAF_REFERENCE_TRAINABLE_PARAMETERS - parameters == GATE_KERNEL_PARAMETERS_REMOVED,
        f"exactly {GATE_KERNEL_TENSORS_REMOVED} tensor removed vs RACAF":
            RACAF_REFERENCE_TRAINABLE_TENSORS - tensors == GATE_KERNEL_TENSORS_REMOVED,
    }
    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"C1 model verification failed: {failed}")
    return {"trainable_parameters": parameters, "trainable_tensors": tensors,
            "reliability_max_abs_diff": reliability_max_abs_diff, "gate_value": gate_value}
