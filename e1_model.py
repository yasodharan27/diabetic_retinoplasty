"""E1 multi-task grader: P's ConvNeXt-Tiny + CORN, plus one auxiliary 1x1 head on the final feature map.

    rgb (B, 512, 512, 3) in [0, 1]
      -> ImageNet normalisation -> ConvNeXt-Tiny stages 0..3 -> F3 (B, 16, 16, 768)
           F3 -> global average pooling -> LayerNorm -> CORN Dense(768 -> 4)        output "corn"
           F3 -> Conv2D(4, 1) -> linear, float32                                    output "lesion_logits"

The encoder is the stage-by-stage rebuild of arch1_model (same layers, same names as Keras' ConvNeXtTiny)
with the prior encoder, the vessel / pathology inputs and the gated injections removed, so the grading path is
P's graph. The auxiliary head reads the same tensor the grader pools; it has 768 * 4 + 4 = 3,076 parameters and
sits beside the grading path, never on it. Channel order of the auxiliary output: MA, HE, EX, SE.

Nothing here trains, compiles or reads data. arch1_model.py and pl_convnext.py are imported, not modified.
"""
import numpy as np

import arch1_model as a1
import pl_convnext as pl

IMAGE_SIZE = pl.IMAGE_SIZE
BACKBONE_STRIDE = 32
LESION_CLASSES = ("MA", "HE", "EX", "SE")
GRADING_OUTPUT = "corn"
LESION_OUTPUT = "lesion_logits"
LESION_CONV_NAME = "lesion_head_conv"
FEATURE_MAP_NAME = f"{a1.BACKBONE_NAME}_stage_3_block_{a1.DEPTHS[3] - 1}_residual"
ADDED_PARAMETERS = a1.DIMS[3] * len(LESION_CLASSES) + len(LESION_CLASSES)        # 3,076
PRIOR_CLIP = (1e-4, 1.0 - 1e-4)


def prior_logits(lesion_prior):
    """Bias initialisation of the auxiliary head: logit of the per-class mean training target, the mean
    clipped to PRIOR_CLIP. `lesion_prior`: four values in [0, 1], order LESION_CLASSES."""
    prior = np.asarray(lesion_prior, dtype=np.float64)
    if prior.shape != (len(LESION_CLASSES),) or not np.all(np.isfinite(prior)) or prior.min() < 0 or prior.max() > 1:
        raise ValueError(f"lesion_prior must be {len(LESION_CLASSES)} finite values in [0, 1], got {lesion_prior!r}")
    clipped = np.clip(prior, *PRIOR_CLIP)
    return np.log(clipped / (1.0 - clipped)).astype(np.float32)


def build_e1_model(seed, lesion_prior, reference=None, image_size=IMAGE_SIZE):
    """The E1 model for one seed. `lesion_prior`: per-class mean training target (sets the auxiliary bias).
    `reference`: Keras' ConvNeXtTiny (`pl_convnext.load_reference`) whose weights are copied in by layer name;
    None leaves the backbone randomly initialised (tests, or before `copy_from_p`). Build under the run's dtype
    policy. Uncompiled. Outputs: [corn logits (B, 4), lesion logits (B, S/32, S/32, 4) float32]."""
    import keras
    from keras import Input, Model, initializers, layers

    import corn

    if image_size % BACKBONE_STRIDE:
        raise ValueError(f"image_size must be a multiple of {BACKBONE_STRIDE}, got {image_size}")
    bias = prior_logits(lesion_prior)
    ImageNetNormalization, _ = a1._layers()
    keras.utils.set_random_seed(int(seed))
    rgb = Input(shape=(image_size, image_size, 3), name="rgb")
    x = ImageNetNormalization(name="rgb_imagenet_normalization")(rgb)
    for s in range(4):
        x = a1._downsampling(s)(x)
        for j in range(a1.DEPTHS[s]):
            x = a1._convnext_block(x, a1.DIMS[s], f"{a1.BACKBONE_NAME}_stage_{s}_block_{j}")
    feature_map = x                                                       # F3: (B, S/32, S/32, 768)

    pooled = layers.GlobalAveragePooling2D(name=f"{a1.BACKBONE_NAME}_gap")(feature_map)
    features = layers.LayerNormalization(epsilon=1e-6, name=a1.HEAD_NORM_NAME)(pooled)
    head = corn.build_corn_model(d_model=pl.FEATURE_DIM)
    kernel, head_bias = pl.corn_head_initial_weights(seed)
    head.get_layer("corn_logits").set_weights([kernel, head_bias])
    grading = head(features)

    lesion = layers.Conv2D(len(LESION_CLASSES), 1, name=LESION_CONV_NAME,
                           kernel_initializer=initializers.GlorotUniform(seed=int(seed)),
                           bias_initializer=initializers.Constant(bias.tolist()))(feature_map)
    lesion = layers.Activation("linear", dtype="float32", name=LESION_OUTPUT)(lesion)

    model = Model(inputs=rgb, outputs=[grading, lesion], name="e1_multitask_grader")
    if tuple(model.output_names) != (GRADING_OUTPUT, LESION_OUTPUT):
        raise RuntimeError(f"unexpected output names {model.output_names}")
    model.e1_copy_report = a1.copy_backbone_weights(model, reference) if reference is not None else None
    model.e1_lesion_prior = [float(p) for p in np.asarray(lesion_prior, dtype=np.float64)]
    return model


def encoder_layers(model):
    """The ConvNeXt layers the auxiliary loss can reach: the rebuilt backbone without its head LayerNorm
    (that LayerNorm sits after the pooling, on the grading path only)."""
    return [l for l in a1.backbone_layers(model) if l.name != a1.HEAD_NORM_NAME]


def copy_from_p(model, p_model):
    """Copies a P model's weights into E1's grading path: every ConvNeXt array by layer name (each exactly
    once; arch1_model.copy_backbone_weights raises on any unmatched, mis-shaped or unconsumed array) and the
    CORN head. The auxiliary head is not touched. Returns the copy report."""
    backbone = p_model.get_layer(pl.BACKBONE_NAME)
    report = a1.copy_backbone_weights(model, backbone)
    if report["copied_arrays"] != len(backbone.weights):
        raise RuntimeError(f"copied {report['copied_arrays']} arrays, the P backbone has {len(backbone.weights)}")
    source, target = p_model.get_layer("corn"), model.get_layer("corn")
    arrays = source.get_weights()
    if [a.shape for a in arrays] != [tuple(w.shape) for w in target.weights]:
        raise RuntimeError("CORN head shapes differ between P and E1")
    target.set_weights(arrays)
    return dict(report, corn_arrays=len(arrays))


def grading_model(model):
    """The grading output alone (rgb -> CORN logits): what is compared with P."""
    from keras import Model
    return Model(model.input, model.outputs[0], name="e1_grading_path")


def parameter_report(model):
    count = lambda ls: int(sum(int(np.prod(w.shape)) for l in ls for w in l.trainable_weights))
    report = {"backbone": count(a1.backbone_layers(model)), "head": count([model.get_layer("corn")]),
              "lesion_head": count([model.get_layer(LESION_CONV_NAME)]),
              "total_trainable": int(sum(int(np.prod(v.shape)) for v in model.trainable_variables)),
              "non_trainable": int(sum(int(np.prod(v.shape)) for v in model.non_trainable_variables))}
    report["p_total"] = pl.EXPECTED_BACKBONE_PARAMETERS["P"] + pl.EXPECTED_HEAD_PARAMETERS
    return report
