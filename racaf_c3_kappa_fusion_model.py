"""
C3: per-lesion-class reliability conditioning a channel-wise RESIDUAL fusion -- an ADDITIVE
module, not a modification of any frozen file.

This is a CORRECTED IMPLEMENTATION of this project's original reliability hypothesis, not a new
mechanism and not a novelty claim. Affine feature modulation of this kind is established
(FiLM, Perez et al. 2017; FiLM-Ensemble applies it to uncertainty in medical imaging). What is
being tested here is narrower: whether the original hypothesis was previously UNDER-tested because
the mechanism carrying it -- RACAF's 2-parameter scalar gate -- was poorly conditioned and weakly
trainable.

The evidence that motivated C3 (see docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md):
RACAF's gate weight `w_g` never meaningfully learned. Across seeds 42/123/2026, AdamW weight decay
moved it 3-10x more than the gradient did, and the decay-free bias `b_g` moved only +0.016..+0.023
in total. The scalar `r` feeding that gate has SD 0.0425, while the per-class `kappa` it is
collapsed from has per-class SD up to ~0.15.

    RACAF: gate = sigmoid(w_g*r + b_g)     F = gate*E + (1 - gate)*Ghat     (convex, scalar)
    C3:    gamma = Dense(4 -> 256)(kappa)  F = E + gamma * Ghat             (residual, channel-wise)

Three concrete differences from RACAF, all deliberate:
  1. the conditioning input is the 4-d per-class `kappa`, not the burden-collapsed scalar `r`;
  2. the modulation is 256-d channel-wise, not a single scalar -- 256x the bandwidth;
  3. the fusion is RESIDUAL, so `E` is always fully present. RACAF's convex blend scales `E` down
     whenever `Ghat` is scaled up, which is what made the gate-forcing ablation collapse to a
     constant prediction; a residual has no such mode.

KAPPA CHANNEL ORDER. `kappa` is `(4,)` in `lesion_segmentation_dataset.LESION_CLASSES` order,
which is **(MA, HE, EX, SE)** -- Microaneurysm, Haemorrhage, HardExudate, SoftExudate. This module
never reorders it; `racaf.compute_reliability` produces it in exactly this order and the cached
`.npz` stores it unchanged.

TWO KNOWN PROPERTIES OF THE CACHED KAPPA, recorded here because they bound what C3 can do:
  - `kappa[0]` (MA) is exactly 1.0 for 100% of cached images, so input column 0 is a constant and
    the effective input dimensionality is ~3, not 4.
  - `racaf.compute_reliability`'s empty-foreground branch sets `delta = 0 -> kappa = 1.0` when a
    class has NO predicted foreground, so `kappa = 1.0` conflates "perfectly consistent" with
    "nothing detected". A positive C3 result may therefore reflect lesion PRESENCE rather than
    reliability. Distinguishing the two would need a burden-conditioned arm; the shuffled-kappa
    control does not separate them (it separates content from capacity).

`racaf.py`, `joint_training_model.py`, `no_racaf_model.py`, `corn.py`, `feature_fusion.py`,
`local_feature_extraction_model.py`, `swin_transformer.py` and `multiseed_runs.py` are all imported
UNMODIFIED. The `Ghat` pathway is RACAF's own `Dense(1152 -> 256)` implementation, reused rather
than reimplemented.
"""

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers

import corn
import feature_fusion
import joint_training_model as jtm
import local_feature_extraction_model
import racaf
import swin_transformer

C3_MODEL_NAME = "joint_stage05_08_c3_kappa_residual"
C3_FUSION_NAME = "c3_kappa_fusion"
GAMMA_LAYER_NAME = "gamma_projection"

#: `kappa`'s dimensionality, reused from RACAF rather than hardcoded as 4.
KAPPA_DIM = racaf.NUM_LESION_CLASSES

# Measured, not estimated -- see tests/test_racaf_c3_kappa_fusion_model.py.
#   NO_RACAF                     43,043,336   (405 tensors)
#   + Ghat Dense(1152->256)        +295,168   (+2 tensors)   -- reused from racaf.py unchanged
#   + gamma Dense(4->256)+bias       +1,280   (+2 tensors)   -- C3's only new parameters
NO_RACAF_TRAINABLE_PARAMETERS = 43_043_336
GHAT_PARAMETERS = 295_168
GAMMA_PARAMETERS = KAPPA_DIM * 256 + 256          # 1,280
EXPECTED_TRAINABLE_PARAMETERS = 43_339_784
EXPECTED_TRAINABLE_TENSORS = 409
RACAF_REFERENCE_TRAINABLE_PARAMETERS = 43_338_506  # no_racaf_model.REFERENCE_TRAINABLE_PARAMETERS

#: Pre-registered mechanism gate. A median per-sample ||gamma*Ghat|| / ||E|| below this means the
#: residual never meaningfully engaged, so the run's QWK carries NO information about the
#: reliability hypothesis -- a distinct outcome from "reliability does not help". Do not change
#: this value to fit a result.
MECHANISM_RATIO_THRESHOLD = 0.01


def build_c3_kappa_fusion(d_model=racaf.D_MODEL, global_tokens=racaf.GLOBAL_TOKENS,
                          global_channels=racaf.GLOBAL_CHANNELS, global_projection_layer=None):
    """C3's fusion: `F = E + gamma * Ghat`, with `gamma = Dense(KAPPA_DIM -> d_model)(kappa)`.

    `gamma` is LINEAR with a bias and is ZERO-initialised in both kernel and bias, so at
    construction `gamma == 0` and `F == E` exactly -- C3 starts at the NO-RACAF representation and
    any later difference is learned residual fusion, never a random starting advantage.

    Linear, not sigmoid/tanh, deliberately. `sigmoid` would force a strictly non-negative bounded
    residual AND put `gamma = 0.5` at zero-init (so `F = E + 0.5*Ghat`, losing the `F == E` start),
    and its `sigma' <= 0.25` would attenuate the gradient at exactly the point where RACAF's gate
    already proved too weakly driven. `tanh` preserves the zero point but still saturates.
    Magnitude control is already provided by AdamW decay on `gamma`'s kernel.

    The bias is kept even though `kappa[0]` (MA) is a constant 1.0 and therefore already acts as a
    bias column: the two are NOT equivalent under AdamW, because the kernel is weight-decayed
    (`variable.name == "kernel"`) while the bias is excluded (`"bias"`). The bias is the residual's
    only decay-immune unconditional term.

    `global_projection_layer`, if given, is an ALREADY-BUILT `Dense(d_model)` reused as-is (Keras'
    standard shared-layer pattern: calling a built layer again draws no new initialisation) rather
    than constructing a fresh one -- see `build_c3_joint_model_matched_init()`.

    Returns an **uncompiled** `keras.Model`, `[E, G, kappa] -> F`."""
    e_input = Input(shape=(d_model,), name="E")
    g_input = Input(shape=(global_tokens, global_channels), name="G")
    kappa_input = Input(shape=(KAPPA_DIM,), name="kappa")

    gap_g = layers.GlobalAveragePooling1D(name="gap_g")(g_input)
    projection_layer = global_projection_layer or layers.Dense(d_model, name="global_projection")
    g_hat = projection_layer(gap_g)

    gamma = layers.Dense(
        d_model, activation="linear", use_bias=True,
        kernel_initializer="zeros", bias_initializer="zeros", name=GAMMA_LAYER_NAME,
    )(kappa_input)

    modulated_g_hat = layers.Multiply(name="gamma_times_g_hat")([gamma, g_hat])
    fused = layers.Add(name="fusion")([e_input, modulated_g_hat])

    # Same float32 cast pattern as racaf.build_racaf_fusion's own output, so CORN always receives
    # float32 logits input under the mixed_float16 policy.
    outputs = layers.Activation("linear", dtype="float32", name="c3_output")(fused)

    return Model(inputs=[e_input, g_input, kappa_input], outputs=outputs, name=C3_FUSION_NAME)


def build_c3_joint_model():
    """`joint_training_model.build_joint_model()` with RACAF's fusion replaced by
    `build_c3_kappa_fusion()` and the scalar `reliability` Input replaced by the `(4,)` `kappa`
    Input -- same unmodified sub-builders, same construction order, same `(B,4)` CORN output."""
    stage5_input = Input(shape=jtm.STAGE5_INPUT_SHAPE, name="stage5_input")
    stage6_input = Input(shape=jtm.STAGE6_INPUT_SHAPE, name="stage6_input")
    kappa_input = Input(shape=(KAPPA_DIM,), name="kappa")

    stage5_model = local_feature_extraction_model.build_local_feature_extractor()
    stage6_model = swin_transformer.create_dual_scale_swin_model()
    stage7_model = feature_fusion.build_adaptive_cross_attention()
    c3_fusion_model = build_c3_kappa_fusion()
    corn_model = corn.build_corn_model()

    local_features = stage5_model(stage5_input)
    global_features = stage6_model(stage6_input)
    fused_embedding = stage7_model([local_features, global_features])
    reliability_fused = c3_fusion_model([fused_embedding, global_features, kappa_input])
    logits = corn_model(reliability_fused)

    return Model(inputs=[stage5_input, stage6_input, kappa_input], outputs=logits,
                name=C3_MODEL_NAME)


def build_c3_joint_model_matched_init():
    """`build_c3_joint_model()`, but with Stage 05/06/07 and CORN consuming the SAME sequence of
    random weight-initialisation draws as RACAF and NO_RACAF at the same run seed.

    How the RNG accounting works out. RACAF's build draws, in order: Stage 05, Stage 06, Stage 07,
    then its fusion (`global_projection` kernel + bias, `reliability_gate` kernel + bias = 4
    draws), then CORN. C3's own fusion would draw only 2 (`global_projection`) because `gamma` is
    zero-initialised and a zeros-initializer consumes NO RNG at all -- which would put CORN two
    draws earlier than RACAF and break parity. So this builder constructs a reference
    `racaf.build_racaf_fusion()` (consuming RACAF's exact 4 draws) and REUSES its already-built
    `global_projection` layer object inside C3's fusion. That reuse draws nothing further, leaving
    CORN at exactly RACAF's position, and additionally makes C3's `Ghat` start from bit-identical
    weights to RACAF's own `Ghat`. The reference fusion's `reliability_gate` is discarded.

    This is the same correction `racaf_c1_control_model.build_c1_joint_model_matched_init()` needed
    and for the same reason; see `tests/..._c3_...py::CrossArmInitializationParityTests`."""
    stage5_input = Input(shape=jtm.STAGE5_INPUT_SHAPE, name="stage5_input")
    stage6_input = Input(shape=jtm.STAGE6_INPUT_SHAPE, name="stage6_input")
    kappa_input = Input(shape=(KAPPA_DIM,), name="kappa")

    stage5_model = local_feature_extraction_model.build_local_feature_extractor()
    stage6_model = swin_transformer.create_dual_scale_swin_model()
    stage7_model = feature_fusion.build_adaptive_cross_attention()
    reference_racaf_fusion = racaf.build_racaf_fusion()   # consumes RACAF's exact 4 draws
    c3_fusion_model = build_c3_kappa_fusion(
        global_projection_layer=reference_racaf_fusion.get_layer("global_projection"))
    del reference_racaf_fusion   # its reliability_gate is discarded; global_projection lives on
    corn_model = corn.build_corn_model()

    local_features = stage5_model(stage5_input)
    global_features = stage6_model(stage6_input)
    fused_embedding = stage7_model([local_features, global_features])
    reliability_fused = c3_fusion_model([fused_embedding, global_features, kappa_input])
    logits = corn_model(reliability_fused)

    return Model(inputs=[stage5_input, stage6_input, kappa_input], outputs=logits,
                name=C3_MODEL_NAME)


def verify_c3_model(model, expect_zero_initialized_gamma=True):
    """Structural proof for a C3 joint model. Raises on the first failing check; returns the
    checked counts on success.

    `expect_zero_initialized_gamma=True` (the default, correct for a FRESHLY BUILT model) also
    proves `gamma == 0` and, by a forward pass through the fusion sub-model alone, that `F == E`
    bit-exactly. Pass `False` after loading trained weights, where `gamma` is expected to have
    moved."""
    parameters = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    tensors = len(model.trainable_variables)
    layer_names = {layer.name for layer in model.layers}
    variable_paths = " ".join(v.path for v in model.trainable_variables).lower()

    fusion = model.get_layer(C3_FUSION_NAME) if C3_FUSION_NAME in layer_names else None
    gamma_layer = None
    if fusion is not None and GAMMA_LAYER_NAME in {l.name for l in fusion.layers}:
        gamma_layer = fusion.get_layer(GAMMA_LAYER_NAME)

    gamma_parameters = (
        sum(int(np.prod(v.shape)) for v in gamma_layer.trainable_variables)
        if gamma_layer is not None else None)
    gamma_is_zero = None
    fusion_equals_e = None
    if gamma_layer is not None:
        gamma_is_zero = bool(np.all(np.asarray(gamma_layer.kernel) == 0.0)
                             and np.all(np.asarray(gamma_layer.bias) == 0.0))
    if expect_zero_initialized_gamma and fusion is not None:
        probe = np.random.default_rng(0)
        # Probe E is rounded through float16 and cast back. Under `mixed_float16` (the training
        # policy) the fusion's Add layer computes in float16, so an arbitrary float32 E would come
        # back rounded by up to ~2.4e-4 and a bit-exact `F == E` check would fail even though the
        # property holds. fp16-representable values pass through both the float32 and the
        # float16 compute paths losslessly, so one probe is exact under either policy.
        e_probe = probe.random((2, racaf.D_MODEL), dtype=np.float32).astype(np.float16).astype(np.float32)
        g_probe = probe.random((2, racaf.GLOBAL_TOKENS, racaf.GLOBAL_CHANNELS), dtype=np.float32)
        k_probe = probe.random((2, KAPPA_DIM), dtype=np.float32)
        fused = np.asarray(fusion.predict_on_batch([e_probe, g_probe, k_probe]))
        fusion_equals_e = bool(np.array_equal(fused, e_probe))

    kappa_inputs = [t for t in model.inputs if t.shape[-1] == KAPPA_DIM and len(t.shape) == 2]

    checks = {
        f"model is named {C3_MODEL_NAME}": model.name == C3_MODEL_NAME,
        f"{EXPECTED_TRAINABLE_TENSORS} trainable tensors": tensors == EXPECTED_TRAINABLE_TENSORS,
        f"{EXPECTED_TRAINABLE_PARAMETERS:,} trainable parameters":
            parameters == EXPECTED_TRAINABLE_PARAMETERS,
        "the C3 kappa fusion sub-model is present": fusion is not None,
        "the gamma projection layer is present": gamma_layer is not None,
        f"gamma has exactly {GAMMA_PARAMETERS:,} parameters": gamma_parameters == GAMMA_PARAMETERS,
        "gamma is linear (no activation)":
            gamma_layer is not None and gamma_layer.activation is tf.keras.activations.linear,
        "gamma uses a bias": gamma_layer is not None and gamma_layer.use_bias,
        f"a (None, {KAPPA_DIM}) kappa Input is declared": len(kappa_inputs) == 1,
        "NO RACAF r-dependent reliability_gate variable present":
            "reliability_gate" not in variable_paths,
        "global_projection (Ghat pathway) variable IS present": "global_projection" in variable_paths,
        "Stage 05 local features present":
            any(name.startswith("local_feature_extraction") for name in layer_names),
        "Stage 06 dual-scale Swin present":
            any(name.startswith("global_feature_extraction") for name in layer_names),
        "Stage 07 adaptive cross-attention present":
            any(name.startswith("feature_fusion") for name in layer_names),
        "CORN head present": "corn" in layer_names,
        "output is (batch, num_thresholds) CORN logits":
            tuple(model.outputs[0].shape) == (None, corn.NUM_THRESHOLDS),
        f"exactly {GAMMA_PARAMETERS - 2:,} parameters more than RACAF":
            parameters - RACAF_REFERENCE_TRAINABLE_PARAMETERS == GAMMA_PARAMETERS - 2,
    }
    if expect_zero_initialized_gamma:
        checks["gamma is zero-initialised (kernel and bias)"] = gamma_is_zero is True
        checks["F == E exactly at initialisation"] = fusion_equals_e is True

    failed = [label for label, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"C3 model verification failed: {failed}")
    return {"trainable_parameters": parameters, "trainable_tensors": tensors,
            "gamma_parameters": gamma_parameters, "gamma_is_zero": gamma_is_zero,
            "fusion_equals_e_at_init": fusion_equals_e}


# =====================================================================================
# Mechanism diagnostics -- did the residual actually engage?
# =====================================================================================

def gamma_from_kappa(model, kappa):
    """`gamma` for each row of `kappa` (`(N, KAPPA_DIM)`), by applying ONLY the trained
    `gamma_projection` layer. Needs no images and no forward pass through Stage 05/06/07, so it is
    usable at initialisation and in tests. Returns `(N, d_model)` float32."""
    gamma_layer = model.get_layer(C3_FUSION_NAME).get_layer(GAMMA_LAYER_NAME)
    kappa = np.asarray(kappa, dtype=np.float32)
    return np.asarray(gamma_layer(tf.convert_to_tensor(kappa)), dtype=np.float32)


def c3_submodels(model):
    """`(stage5, stage6, stage7, fusion)` sub-model handles from a built C3 joint model, resolved
    by the name prefixes each builder assigns rather than by position.

    Deliberately returns the sub-models themselves rather than building an intermediate
    `keras.Model` over the parent graph: Stage 06 and Stage 07 both contain an operation named
    `global_features`, so any `Model(model.inputs, [...])` spanning both is rejected by Keras with
    a duplicate-operation-name error. Calling the sub-models eagerly reproduces the parent's exact
    forward path (it is literally the sequence `build_c3_joint_model()` wires) with no graph
    surgery."""
    def by_prefix(prefix):
        return next(layer for layer in model.layers if layer.name.startswith(prefix))

    return (by_prefix("local_feature_extraction"), by_prefix("global_feature_extraction"),
            by_prefix("feature_fusion"), model.get_layer(C3_FUSION_NAME))


def compute_mechanism_diagnostics(model, stage5, stage6, kappa, batch_size=8):
    """The pre-registered C3 mechanism diagnostic.

    AGGREGATION, stated explicitly: the residual ratio is computed PER SAMPLE as the L2 norm over
    the 256 channels of `gamma * Ghat` divided by the L2 norm over the 256 channels of `E`, and
    only then aggregated across samples (mean and median are both reported). It is deliberately
    NOT a ratio of two separately-summed global quantities, which would let a few large-norm
    samples dominate and would not correspond to any per-image statement.

    Samples whose `||E||` is exactly zero are excluded from the ratio and counted separately.

    Returns a dict with the per-sample ratio summary, the gamma summary
    (mean|gamma|, min, max, mean, std and percentiles), and
    `mechanism_engaged = median_ratio >= MECHANISM_RATIO_THRESHOLD`.

    `residual_ratio_per_sample` is returned in full so a caller can evaluate a large validation
    split in chunks (one `(512,512,8)` sample is 8 MiB, so a whole split cannot be held at once)
    and aggregate the concatenated per-sample ratios itself."""
    stage5_model, stage6_model, stage7_model, fusion = c3_submodels(model)
    gap_layer = fusion.get_layer("gap_g")
    projection_layer = fusion.get_layer("global_projection")
    gamma_layer = fusion.get_layer(GAMMA_LAYER_NAME)

    stage5 = np.asarray(stage5, dtype=np.float32)
    stage6 = np.asarray(stage6, dtype=np.float32)
    kappa = np.asarray(kappa, dtype=np.float32)

    # Only the 256-d quantities are accumulated; the (B,64,1152) G of each batch is consumed and
    # dropped immediately, so this stays flat in memory over a full validation split.
    e_parts, g_hat_parts, gamma_parts = [], [], []
    for start in range(0, len(kappa), batch_size):
        stop = start + batch_size
        local_b = stage5_model(tf.convert_to_tensor(stage5[start:stop]), training=False)
        g_b = stage6_model(tf.convert_to_tensor(stage6[start:stop]), training=False)
        e_b = stage7_model([local_b, g_b], training=False)
        g_hat_b = projection_layer(gap_layer(g_b))
        gamma_b = gamma_layer(tf.convert_to_tensor(kappa[start:stop]))
        e_parts.append(np.asarray(e_b, dtype=np.float64))
        g_hat_parts.append(np.asarray(g_hat_b, dtype=np.float64))
        gamma_parts.append(np.asarray(gamma_b, dtype=np.float64))

    e = np.concatenate(e_parts, axis=0)
    g_hat = np.concatenate(g_hat_parts, axis=0)
    gamma = np.concatenate(gamma_parts, axis=0)

    residual = gamma * g_hat
    residual_norm = np.linalg.norm(residual, axis=-1)
    e_norm = np.linalg.norm(e, axis=-1)
    usable = e_norm > 0.0
    ratio = residual_norm[usable] / e_norm[usable]

    percentiles = [1, 25, 50, 75, 99]
    return {
        "n_samples": int(len(kappa)),
        "n_excluded_zero_norm_E": int((~usable).sum()),
        "residual_ratio_definition":
            "per-sample ||gamma*Ghat||_2 / ||E||_2 over the 256 channels, then aggregated",
        "residual_ratio_per_sample": [float(v) for v in ratio],
        "residual_ratio_mean": float(ratio.mean()) if ratio.size else None,
        "residual_ratio_median": float(np.median(ratio)) if ratio.size else None,
        "residual_ratio_min": float(ratio.min()) if ratio.size else None,
        "residual_ratio_max": float(ratio.max()) if ratio.size else None,
        "mean_abs_gamma": float(np.abs(gamma).mean()),
        "gamma_min": float(gamma.min()),
        "gamma_max": float(gamma.max()),
        "gamma_mean": float(gamma.mean()),
        "gamma_std": float(gamma.std()),
        "gamma_percentiles": {f"p{p}": float(np.percentile(gamma, p)) for p in percentiles},
        "mean_norm_E": float(e_norm.mean()),
        "mean_norm_residual": float(residual_norm.mean()),
        "mechanism_ratio_threshold": MECHANISM_RATIO_THRESHOLD,
        "mechanism_engaged": bool(ratio.size and float(np.median(ratio)) >= MECHANISM_RATIO_THRESHOLD),
    }
