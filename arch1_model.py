"""Architecture 1 -- Stages 5-8 of the v2 pipeline (Keras), research record §40,
`research/Grade3vs4_Architecture_Research/IMPLEMENTATION_SPEC_STAGE3_8_V2.md` §8-§9.

    rgb (512^2x3, Stage-2 [0,1]) -> ImageNet normalisation -> Stage 6 ConvNeXt-T (pinned ImageNet
        weights, drop_path 0), rebuilt block-by-block so Stage 7 can sit between its stages
    vessel (512^2x1) + pathology (512^2x2K, [c:mean, c:max] per Stage-4 class)
        -> Stage 5 prior encoder -> P1 128^2x32, P2 64^2x64, P3 32^2x128, P4 16^2x256
    Stage 7, after ConvNeXt stage s and before downsampling layer s:
        F_s <- F_s * (1 + alpha_s * tanh(Conv1x1_a(P_s))) + gamma_s * Conv1x1_b(P_s),
        alpha_s, gamma_s per-channel, ZERO-initialised (the initial model computes exactly P)
    -> GAP -> LayerNorm (the backbone head norm) -> Stage 8 CORN Dense(768 -> 4)

K is never hard-coded: the pathology stem is built from the Stage-4 manifest's channel list.
There is no reliability input and no legacy cache anywhere in this graph.
"""
import numpy as np

import pl_convnext as pl
import stage34_cache_v2 as cache

IMAGE_SIZE = pl.IMAGE_SIZE
BACKBONE_NAME = pl.BACKBONE_NAME
DEPTHS = (3, 3, 9, 3)
DIMS = (96, 192, 384, 768)
PRIOR_DIMS = (32, 64, 128, 256)
VESSEL_STEM, PATHOLOGY_STEM = 16, 32
GN_GROUPS = 8
INPUT_NAMES = ("rgb", "vessel", "pathology")
HEAD_NORM_NAME = f"{BACKBONE_NAME}_head_layernorm"


def _layers():
    from keras import layers, ops

    class ImageNetNormalization(layers.Layer):
        """Weight-free (x - ImageNet mean) / std on Stage-2 [0, 1] RGB, as P's ChannelAdapter."""

        def call(self, x):
            mean = ops.convert_to_tensor(pl.IMAGENET_MEAN, dtype=x.dtype)
            std = ops.convert_to_tensor(pl.IMAGENET_STD, dtype=x.dtype)
            return (x - mean) / std

    class GatedPriorInjection(layers.Layer):
        """F * (1 + alpha * tanh(Conv1x1_a(P))) + gamma * Conv1x1_b(P); alpha = gamma = 0 at init."""

        def __init__(self, channels, **kwargs):
            super().__init__(**kwargs)
            self.channels = int(channels)
            self.conv_a = layers.Conv2D(self.channels, 1, name="conv_a")
            self.conv_b = layers.Conv2D(self.channels, 1, name="conv_b")

        def build(self, input_shape):
            _, prior_shape = input_shape
            self.conv_a.build(prior_shape)
            self.conv_b.build(prior_shape)
            self.alpha = self.add_weight(name="alpha", shape=(self.channels,),
                                         initializer="zeros", trainable=True)
            self.gamma = self.add_weight(name="gamma", shape=(self.channels,),
                                         initializer="zeros", trainable=True)

        def call(self, inputs):
            features, prior = inputs
            alpha = ops.cast(self.alpha, features.dtype)
            gamma = ops.cast(self.gamma, features.dtype)
            gate = ops.tanh(ops.cast(self.conv_a(prior), features.dtype))
            shift = ops.cast(self.conv_b(prior), features.dtype)
            return features * (1.0 + alpha * gate) + gamma * shift

        def compute_output_shape(self, input_shape):
            return input_shape[0]

        def get_config(self):
            return dict(super().get_config(), channels=self.channels)

    return ImageNetNormalization, GatedPriorInjection


def _conv_gn(x, filters, groups, stride, name, activation=True):
    from keras import layers
    x = layers.Conv2D(filters, 3, strides=stride, padding="same", name=f"{name}_conv")(x)
    x = layers.GroupNormalization(groups=groups, epsilon=1e-5, name=f"{name}_gn")(x)
    if activation:
        x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    return x


def prior_encoder(vessel, pathology):
    """Stage 5 (K-agnostic apart from the pathology stem's input channels). Returns [P1..P4]."""
    from keras import layers
    v = _conv_gn(vessel, VESSEL_STEM, 4, 2, "prior_vessel_stem")
    q = _conv_gn(pathology, PATHOLOGY_STEM, 8, 2, "prior_pathology_stem")
    x = layers.Concatenate(name="prior_concat")([v, q])
    pyramid = []
    for i, filters in enumerate(PRIOR_DIMS):
        name = f"prior_down_{i + 1}"
        y = _conv_gn(x, filters, GN_GROUPS, 2, f"{name}_a")
        y = _conv_gn(y, filters, GN_GROUPS, 1, f"{name}_b", activation=False)
        shortcut = layers.Conv2D(filters, 1, strides=2, name=f"{name}_shortcut")(x)
        x = layers.Activation("gelu", name=f"{name}_out")(layers.Add(name=f"{name}_add")([y, shortcut]))
        pyramid.append(x)
    return pyramid


def _convnext_block(x, dim, name):
    """Identical layers and names to `keras.src.applications.convnext.ConvNeXtBlock`
    (drop_path 0 -> the linear identity layer)."""
    from keras import layers
    from keras.src.applications.convnext import LayerScale
    inputs = x
    x = layers.Conv2D(dim, 7, padding="same", groups=dim, name=f"{name}_depthwise_conv")(x)
    x = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_layernorm")(x)
    x = layers.Dense(4 * dim, name=f"{name}_pointwise_conv_1")(x)
    x = layers.Activation("gelu", name=f"{name}_gelu")(x)
    x = layers.Dense(dim, name=f"{name}_pointwise_conv_2")(x)
    x = LayerScale(1e-6, dim, name=f"{name}_layer_scale")(x)
    x = layers.Activation("linear", name=f"{name}_identity")(x)
    return layers.Add(name=f"{name}_residual")([inputs, x])


def _downsampling(i, name=BACKBONE_NAME):
    from keras import Sequential, layers
    if i == 0:
        return Sequential([layers.Conv2D(DIMS[0], 4, strides=4, name=f"{name}_stem_conv"),
                           layers.LayerNormalization(epsilon=1e-6, name=f"{name}_stem_layernorm")],
                          name=f"{name}_stem")
    return Sequential([layers.LayerNormalization(epsilon=1e-6, name=f"{name}_downsampling_layernorm_{i - 1}"),
                       layers.Conv2D(DIMS[i], 2, strides=2, name=f"{name}_downsampling_conv_{i - 1}")],
                      name=f"{name}_downsampling_block_{i - 1}")


def build_arch1_model(channels, seed, reference=None, image_size=IMAGE_SIZE):
    """The Architecture-1 model for one seed. `channels`: the Stage-4 manifest's ordered channel
    list (validated; its length sets the pathology stem). `reference`: Keras' ConvNeXtTiny
    (`pl_convnext.load_reference`) whose weights are copied in by layer name; None leaves the
    backbone randomly initialised (tests only). Build under the run's dtype policy. Uncompiled."""
    import keras
    from keras import Input, Model, layers

    import corn

    channels = tuple(channels)
    cache.classes_from_channels(channels)
    ImageNetNormalization, GatedPriorInjection = _layers()
    keras.utils.set_random_seed(int(seed))
    rgb = Input(shape=(image_size, image_size, 3), name="rgb")
    vessel = Input(shape=(image_size, image_size, 1), name="vessel")
    pathology = Input(shape=(image_size, image_size, len(channels)), name="pathology")

    priors = prior_encoder(vessel, pathology)
    x = ImageNetNormalization(name="rgb_imagenet_normalization")(rgb)
    for s in range(4):
        x = _downsampling(s)(x)
        for j in range(DEPTHS[s]):
            x = _convnext_block(x, DIMS[s], f"{BACKBONE_NAME}_stage_{s}_block_{j}")
        x = GatedPriorInjection(DIMS[s], name=f"stage7_injection_{s + 1}")([x, priors[s]])
    x = layers.GlobalAveragePooling2D(name=f"{BACKBONE_NAME}_gap")(x)
    features = layers.LayerNormalization(epsilon=1e-6, name=HEAD_NORM_NAME)(x)
    head = corn.build_corn_model(d_model=pl.FEATURE_DIM)
    kernel, bias = pl.corn_head_initial_weights(seed)
    head.get_layer("corn_logits").set_weights([kernel, bias])
    logits = head(features)
    model = Model(inputs=[rgb, vessel, pathology], outputs=logits, name="arch1_convnext_priors")
    model.arch1_copy_report = copy_backbone_weights(model, reference) if reference is not None else None
    model.arch1_channels = channels
    return model


def backbone_layers(model):
    """The layers of the rebuilt ConvNeXt-T incl. its head LayerNorm (weights come from the reference)."""
    return [l for l in model.layers if l.name.startswith(f"{BACKBONE_NAME}_") and l.weights]


def copy_backbone_weights(model, reference):
    """Copies every weighted ConvNeXt layer from `reference` by LAYER name (layer names are stable;
    variable paths are not -- LayerScale weights are unnamed). The reference's unnamed final
    LayerNormalization maps to HEAD_NORM_NAME; its PreStem normalisation is skipped. Every
    reference weight must be consumed exactly once, with identical shapes."""
    ref = {l.name: l for l in reference.layers if l.weights and "prestem" not in l.name}
    final_norm = [l for l in reference.layers if l.weights and l.name not in
                  {b.name for b in backbone_layers(model)} and "prestem" not in l.name]
    if len(final_norm) != 1 or final_norm[0].__class__.__name__ != "LayerNormalization":
        raise RuntimeError(f"expected exactly one unmatched reference layer (the head LayerNorm), "
                           f"got {[l.name for l in final_norm]}")
    ref[HEAD_NORM_NAME] = ref.pop(final_norm[0].name)
    copied = 0
    targets = backbone_layers(model)
    for layer in targets:
        source = ref.pop(layer.name, None)
        if source is None:
            raise RuntimeError(f"no reference layer for {layer.name}")
        arrays = source.get_weights()
        if [a.shape for a in arrays] != [tuple(w.shape) for w in layer.weights]:
            raise RuntimeError(f"{layer.name}: shape mismatch")
        layer.set_weights(arrays)
        copied += len(arrays)
    if ref:
        raise RuntimeError(f"reference layers not consumed: {sorted(ref)[:5]}")
    expected = len([v for v in reference.weights if "prestem" not in v.path])
    if copied != expected:
        raise RuntimeError(f"copied {copied} arrays, reference has {expected}")
    return {"copied_arrays": copied, "layers": len(targets)}


def injection_layers(model):
    return [l for l in model.layers if l.name.startswith("stage7_injection_")]


def prior_layers(model):
    return [l for l in model.layers if l.name.startswith("prior_")]


def parameter_report(model):
    count = lambda ls: int(sum(int(np.prod(w.shape)) for l in ls for w in l.trainable_weights))
    return {"backbone": count(backbone_layers(model)),
            "prior_encoder": count(prior_layers(model)),
            "injection": count(injection_layers(model)),
            "head": count([model.get_layer("corn")]),
            "total": pl.trainable_parameter_count(model)}
