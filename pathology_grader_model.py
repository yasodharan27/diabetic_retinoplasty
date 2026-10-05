"""Pathology-only grader -- the Stage-3 / Stage-4 branch of the dual-branch pipeline.

    vessel (512^2x1, Stage 3, [0,1]) + pathology (512^2x2K, Stage 4 v2, [0,1]; [c:mean, c:max] per class)
        -> Stage 5 pathology encoder (arch1_model.prior_encoder, unchanged: vessel stem 16 + pathology stem 32,
           four residual down-blocks -> 16^2 x 256)
        -> GAP -> LayerNorm -> CORN Dense(256 -> 4)

There is NO RGB input and no ConvNeXt anywhere in this graph: the model's only inputs are the two
segmentation-derived tensors, so whatever it grades, it grades from Stage 3 and Stage 4 alone. The encoder is
Architecture 1's Stage-5 encoder as designed (no attention, no extra scale); only its last scale is read.
"""
import numpy as np

import arch1_model as a1
import stage34_cache_v2 as cache

IMAGE_SIZE = a1.IMAGE_SIZE
INPUT_NAMES = ("vessel", "pathology")
FEATURE_DIM = a1.PRIOR_DIMS[-1]
NUM_THRESHOLDS = 4
MODEL_NAME = "pathology_grader"
HEAD_NORM_NAME = "pathology_head_layernorm"


def input_channel_names(channels):
    """The ordered content of the branch input: the vessel map, then every Stage-4 channel."""
    channels = tuple(channels)
    cache.classes_from_channels(channels)
    return ("vessel",) + channels


def head_initial_weights(seed):
    """The CORN head start: GlorotUniform(seed) kernel, zero bias -- pl_convnext.corn_head_initial_weights's
    rule at this branch's feature width, independent of the global RNG."""
    import keras
    kernel = keras.initializers.GlorotUniform(seed=int(seed))((FEATURE_DIM, NUM_THRESHOLDS))
    return np.asarray(kernel, dtype=np.float32), np.zeros((NUM_THRESHOLDS,), dtype=np.float32)


def build_pathology_grader(channels, seed, image_size=IMAGE_SIZE):
    """The pathology-only grader for one seed. `channels`: the Stage-4 manifest's ordered channel list
    (validated; its length sets the pathology stem). Build under the run's dtype policy. Uncompiled."""
    import keras
    from keras import Input, Model, layers

    import corn

    channels = tuple(channels)
    cache.classes_from_channels(channels)
    keras.utils.set_random_seed(int(seed))
    vessel = Input(shape=(image_size, image_size, 1), name="vessel")
    pathology = Input(shape=(image_size, image_size, len(channels)), name="pathology")
    features = a1.prior_encoder(vessel, pathology)[-1]
    x = layers.GlobalAveragePooling2D(name="pathology_gap")(features)
    x = layers.LayerNormalization(epsilon=1e-6, name=HEAD_NORM_NAME)(x)
    head = corn.build_corn_model(d_model=FEATURE_DIM)
    kernel, bias = head_initial_weights(seed)
    head.get_layer("corn_logits").set_weights([kernel, bias])
    model = Model(inputs=[vessel, pathology], outputs=head(x), name=MODEL_NAME)
    model.pathology_channels = channels
    return model


def parameter_report(model):
    count = lambda ls: int(sum(int(np.prod(w.shape)) for l in ls for w in l.trainable_weights))      # noqa: E731
    return {"encoder": count(a1.prior_layers(model)), "head_norm": count([model.get_layer(HEAD_NORM_NAME)]),
            "head": count([model.get_layer("corn")]),
            "total": int(sum(int(np.prod(v.shape)) for v in model.trainable_variables))}
