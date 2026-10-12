"""Stage 11 -- Grad-CAM for the final grader (research record §96). PREPARED, NOT RUN.

The map is computed on the grader's last feature map (16 x 16 x 768 for a 512 frame), reached through the graph
the lesion-readability probes already use (`e1_probe.build_frozen_encoder`, kind 'p': the grader's weights
copied into a graph that exposes that feature map; its grading logits are those of the grader). No weight is
trained or changed.

What is explained: by default the model's EXPECTED GRADE (the sum of its four cumulative probabilities), so one
map answers "what moved this image up the severity scale"; a single cumulative cut can be asked for instead
(e.g. cut 1 = "referable or worse"). The map is the standard Grad-CAM: feature channels weighted by the spatial
mean of the gradient of that score, summed, rectified, scaled to [0, 1] and resized to the frame.

Limits to state with any figure: 16 x 16 resolution (each cell is 32 x 32 pixels); a saliency map shows where the
score is sensitive, not that the model uses clinically correct evidence; `lesion_agreement` below measures
overlap with expert lesion masks and is the only quantitative check offered.
"""
import numpy as np

FEATURE_POLICY = "float32"                     # gradients are taken in full precision
ARM_FRAME = {"direct": "direct", "crop": "crop"}


def load_explainer(seed, weights_path, reference_arrays):
    """(grad model, full model) for one frozen grader checkpoint. The grad model maps {'rgb'} to
    (feature map, grading logits)."""
    import keras

    import e1_model as em
    import e1_probe as ep
    _, full = ep.build_frozen_encoder("p", int(seed), weights_path, [0.1] * 4, reference_arrays, policy=FEATURE_POLICY)
    grad_model = keras.Model(full.inputs, [full.get_layer(em.FEATURE_MAP_NAME).output, full.outputs[0]])
    return grad_model, full


def score_from_logits(logits, cut=None):
    """The explained scalar per image: expected grade, or the cumulative probability of one cut (0..3)."""
    import tensorflow as tf
    p = tf.math.cumprod(tf.sigmoid(tf.cast(logits, tf.float32)), axis=1)
    return tf.reduce_sum(p, axis=1) if cut is None else p[:, int(cut)]


def grad_cam(grad_model, rgb, cut=None):
    """rgb: (N, 512, 512, 3) float32 in [0, 1] -> (maps (N, 512, 512) in [0, 1], logits (N, 4))."""
    import tensorflow as tf
    rgb = tf.convert_to_tensor(np.asarray(rgb, np.float32))
    with tf.GradientTape() as tape:
        tape.watch(rgb)                        # the model is frozen (no trainable variable): watching the input records the graph
        features, logits = grad_model({"rgb": rgb}, training=False)
        score = score_from_logits(logits, cut)
    gradients = tape.gradient(score, features)   # with respect to the feature map itself, an intermediate of that call
    if gradients is None:
        raise RuntimeError("no gradient reached the feature map")
    features, gradients = tf.cast(features, tf.float32), tf.cast(gradients, tf.float32)
    weights = tf.reduce_mean(gradients, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(weights * features, axis=-1))
    peak = tf.reduce_max(cam, axis=(1, 2), keepdims=True)
    cam = tf.where(peak > 0, cam / tf.maximum(peak, 1e-12), tf.zeros_like(cam))
    cam = tf.image.resize(cam[..., None], rgb.shape[1:3], method="bilinear")[..., 0]
    return np.asarray(cam, np.float32), np.asarray(logits, np.float64)


def ensemble_cam(grad_models, rgb, cut=None):
    """The mean map of several seeds' explainers (each scaled to [0, 1] first) and the per-seed logits."""
    maps, logits = zip(*(grad_cam(g, rgb, cut) for g in grad_models))
    return np.mean(maps, axis=0), np.stack(logits)


def overlay(rgb, cam, alpha=0.45):
    """An RGB uint8 picture: the frame with the map over it (red = high)."""
    rgb = np.asarray(rgb, np.float32)
    heat = np.stack([cam, np.zeros_like(cam), 1.0 - cam], axis=-1) * cam[..., None]
    return np.clip(255.0 * ((1.0 - alpha * cam[..., None]) * rgb + alpha * heat), 0, 255).astype(np.uint8)


def lesion_agreement(cam, lesion_mask, top_fraction=0.1):
    """Agreement of one map with a binary expert lesion mask of the same size: the pointing game (does the
    map's maximum fall on a lesion?), the share of the map's top `top_fraction` of pixels that are lesion, and
    the lesion share of the image (what a uniform map would score)."""
    mask = np.asarray(lesion_mask, bool)
    if mask.shape != cam.shape:
        raise ValueError("the map and the mask must have the same shape")
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    k = max(1, int(round(top_fraction * cam.size)))
    top = np.argsort(cam, axis=None)[-k:]
    return {"pointing_hit": bool(mask[peak]), "top_fraction_precision": float(mask.reshape(-1)[top].mean()),
            "lesion_fraction": float(mask.mean()), "has_lesion": bool(mask.any())}
