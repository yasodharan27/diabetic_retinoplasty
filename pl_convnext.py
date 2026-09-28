"""
P/PL controlled experiment -- pretrained ConvNeXt-Tiny, RGB only (P) vs RGB + frozen Stage 3/4 priors
(PL). An ADDITIVE module: no existing stage, cache, split or checkpoint is modified.

Protocol: the P/PL protocol-lock report (research record / docs/experiments/PL_ConvNeXt_Priors_Report.md).
This is NOT a novel architecture. It is a controlled test of one question:
    do frozen vessel/lesion segmentation-derived priors add information when supplied to the SAME
    strong pretrained CNN?

Model (both arms, identical except the input channels of the first convolution):
    (stage5_input 512x512x8, stage6_input, reliability)          -- the project's 3-input contract
      -> ChannelAdapter: RGB (channels 0-2, Stage-02 [0,1]) normalised with fixed ImageNet constants;
                         P keeps RGB only; PL appends channels 3-7 (vessel, MA, HE, EX, SE) unchanged
      -> ConvNeXt-Tiny (include_top=False, include_preprocessing=False, pooling="avg")
         = ImageNet-1k weights copied from the pinned Keras file; PL's first-layer kernel is
           [pretrained RGB | zeros for 5 extra channels], bias copied unchanged
      -> 768-d feature -> CORN Dense(768->4), kernel GlorotUniform(seed=run_seed), bias zeros
      -> + exact zeros from stage6_input / reliability (Keras requires every Input to be connected)
Optimiser: AdamW (lr, wd from the frozen six-run protocol) with NO weight decay on any 1-D parameter
(biases, LayerNorm scale/shift and ConvNeXt LayerScale, whose weights are unnamed `variable_N`).
"""

import hashlib
import os
import shutil
import urllib.request

import numpy as np

EXPERIMENT_VERSION = "pl-convnext-v1"
EXPERIMENT_NAME = "PL_ConvNeXtPriors"
SEEDS = (42, 123, 2026)
ARMS = {"P": {"channels": 3, "dirname": "P_convnext_rgb"},
        "PL": {"channels": 8, "dirname": "PL_convnext_rgb_priors"}}
RUN_ORDER = [(arm, seed) for seed in SEEDS for arm in ("P", "PL")]

IMAGE_SIZE = 512
FEATURE_DIM = 768
NUM_THRESHOLDS = 4
CHANNEL_ORDER = ("R", "G", "B", "vessel", "MA", "HE", "EX", "SE")
EXPECTED_LESION_CLASSES = ("Microaneurysm", "Haemorrhage", "HardExudate", "SoftExudate")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Pinned pretrained source (protocol-locked; never substituted).
WEIGHTS_FILENAME = "convnext_tiny_notop.h5"
WEIGHTS_URL = "https://storage.googleapis.com/tensorflow/keras-applications/convnext/" + WEIGHTS_FILENAME
WEIGHTS_SHA256 = "d547c096cabd03329d7be5562c5e14798aa39ed24b474157cef5e85ab9e49ef1"
BACKBONE_NAME = "convnext_tiny"

# Measured structure (local build, weights=None): backbone trainable parameters.
EXPECTED_BACKBONE_PARAMETERS = {"P": 27_820_128, "PL": 27_827_808}
EXPECTED_HEAD_PARAMETERS = FEATURE_DIM * NUM_THRESHOLDS + NUM_THRESHOLDS      # 3,076
EXPECTED_LAYER_SCALE_VARIABLES = 18

REFERENCE_PARITY_TOL = 1e-4        # P(normalised x) vs the Keras-preprocessed reference R(x*255)
PL_P_EQUIVALENCE_TOL = 1e-5        # initial PL(x8) vs P(x_rgb), float32
SPECIFICITY = 0.95

# Locked decision rule (A: PL - P; B: PL - frozen NO_RACAF reference).
SUPPORTIVE_MEAN = 0.03
NOT_SUPPORTIVE_MEAN = 0.01
NOT_SUPPORTIVE_MIN_NONPOSITIVE = 2
GUARDRAILS = {"qwk": ("min", -0.02), "grade3_recall": ("min", -0.10),
              "false_urgent_rate": ("max", 0.02)}


# ---------------------------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------------------------

def _layers():
    from keras import layers, ops

    class ChannelAdapter(layers.Layer):
        """Fixed, weight-free input adapter. RGB: (x - ImageNet mean) / ImageNet std (identical to
        Keras' own ConvNeXt PreStem applied to x*255). P returns RGB only; PL appends channels 3-7
        (vessel + lesion probabilities, already in [0,1]) unchanged."""

        def __init__(self, arm, **kwargs):
            super().__init__(**kwargs)
            if arm not in ARMS:
                raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
            self.arm = arm

        def call(self, x):
            mean = ops.convert_to_tensor(IMAGENET_MEAN, dtype=x.dtype)
            std = ops.convert_to_tensor(IMAGENET_STD, dtype=x.dtype)
            rgb = (x[..., :3] - mean) / std
            if self.arm == "P":
                return rgb
            return ops.concatenate([rgb, x[..., 3:8]], axis=-1)

        def compute_output_shape(self, input_shape):
            return tuple(input_shape[:-1]) + (ARMS[self.arm]["channels"],)

        def get_config(self):
            return dict(super().get_config(), arm=self.arm)

    class InertAuxiliaryInputs(layers.Layer):
        """Connects stage6_input and reliability to the graph with EXACTLY zero influence (zeros_like,
        never a multiplication, so no value -- not even NaN -- can propagate)."""

        def call(self, inputs):
            logits, stage6, reliability = inputs
            zero = ops.zeros_like(reliability[:, :1]) + ops.zeros_like(stage6[:, 0, 0, :1])
            return logits + ops.cast(zero, logits.dtype)

        def compute_output_shape(self, input_shape):
            return input_shape[0]

    return ChannelAdapter, InertAuxiliaryInputs


# ---------------------------------------------------------------------------------------------
# Pretrained weights (pinned file, cached on Drive, never substituted)
# ---------------------------------------------------------------------------------------------

def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def internet_reachable(url=WEIGHTS_URL, timeout=20):
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 400
    except Exception:  # noqa: BLE001 -- any failure means "not reachable"
        return False


def ensure_pretrained_weights(drive_cache_dir, local_dir="/content/pretrained_weights"):
    """The pinned ImageNet-1k ConvNeXt-Tiny notop file, SHA-256 verified. Reused from the Drive cache
    when present (no re-download); otherwise downloaded once from the pinned URL and copied to the
    Drive cache for the remaining runs/sessions. Returns a provenance record."""
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(drive_cache_dir, exist_ok=True)
    local_path = os.path.join(local_dir, WEIGHTS_FILENAME)
    drive_path = os.path.join(drive_cache_dir, WEIGHTS_FILENAME)
    how = None
    if os.path.exists(local_path) and sha256_file(local_path) == WEIGHTS_SHA256:
        how = "local_runtime_copy"
    elif os.path.exists(drive_path) and sha256_file(drive_path) == WEIGHTS_SHA256:
        shutil.copyfile(drive_path, local_path)
        how = "drive_cache"
    else:
        if not internet_reachable():
            raise RuntimeError(f"No cached copy of {WEIGHTS_FILENAME} and {WEIGHTS_URL} is not "
                               "reachable from this runtime (check internet access).")
        tmp = local_path + ".part"
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
        os.replace(tmp, local_path)
        how = "downloaded"
    digest = sha256_file(local_path)
    if digest != WEIGHTS_SHA256:
        raise RuntimeError(f"{WEIGHTS_FILENAME} sha256 {digest} != pinned {WEIGHTS_SHA256}.")
    if not (os.path.exists(drive_path) and sha256_file(drive_path) == WEIGHTS_SHA256):
        tmp = drive_path + ".part"
        shutil.copyfile(local_path, tmp)
        os.replace(tmp, drive_path)
    return {"source_url": WEIGHTS_URL, "file": WEIGHTS_FILENAME, "sha256": digest,
            "pinned_sha256": WEIGHTS_SHA256, "local_path": local_path, "drive_cache": drive_path,
            "obtained_via": how, "pretraining": "ImageNet-1k (Keras applications ConvNeXtTiny, "
            "converted from facebookresearch/ConvNeXt)"}


def load_reference(weights_path, image_size=IMAGE_SIZE):
    """R: Keras' own ConvNeXt-Tiny with its built-in ImageNet preprocessing (the path Keras uses when
    `weights="imagenet"`), loaded from the pinned file. Returns (R, arrays) where `arrays` are R's
    weights in order with the preprocessing layer's variables (if any) removed -- positionally aligned
    with a model built with include_preprocessing=False."""
    from keras.applications import ConvNeXtTiny
    reference = ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=True,
                             pooling="avg", input_shape=(image_size, image_size, 3),
                             name=BACKBONE_NAME)
    reference.load_weights(weights_path)
    arrays = [np.asarray(v.numpy()) for v in reference.weights if "prestem" not in v.path]
    return reference, arrays


# ---------------------------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------------------------

def build_backbone(arm, image_size=IMAGE_SIZE):
    from keras.applications import ConvNeXtTiny
    return ConvNeXtTiny(include_top=False, weights=None, include_preprocessing=False,
                        pooling="avg", input_shape=(image_size, image_size, ARMS[arm]["channels"]),
                        name=BACKBONE_NAME)


def copy_pretrained(backbone, reference_arrays, arm):
    """Positional copy with shape checks (names are unsafe: LayerScale weights are `variable_N`,
    numbered by build order). The first-layer kernel is the only permitted shape difference (PL):
    [:, :, :3, :] = pretrained RGB kernel exactly, [:, :, 3:, :] = 0. Every other array, including
    the first-layer bias, is copied unchanged. Returns a report."""
    weights = backbone.weights
    if len(weights) != len(reference_arrays):
        raise RuntimeError(f"{arm}: {len(weights)} backbone weights vs {len(reference_arrays)} "
                           "reference arrays -- architectures do not align.")
    expanded = []
    for variable, array in zip(weights, reference_arrays):
        target = tuple(variable.shape)
        if target == array.shape:
            variable.assign(array.astype(variable.dtype))
            continue
        is_stem = ("stem_conv" in variable.path and len(target) == 4 and array.ndim == 4
                   and target[:2] == array.shape[:2] and target[3] == array.shape[3]
                   and array.shape[2] == 3 and target[2] == ARMS[arm]["channels"])
        if not is_stem:
            raise RuntimeError(f"{arm}: unexpected shape mismatch at {variable.path}: "
                               f"{target} vs {array.shape}")
        kernel = np.zeros(target, dtype=np.float32)
        kernel[:, :, :3, :] = array
        variable.assign(kernel.astype(variable.dtype))
        expanded.append(variable.path)
    if arm == "P" and expanded:
        raise RuntimeError(f"P must not expand any kernel, expanded {expanded}")
    if arm == "PL" and len(expanded) != 1:
        raise RuntimeError(f"PL must expand exactly the first-layer kernel, expanded {expanded}")
    return {"arm": arm, "copied": len(weights), "expanded": expanded}


def stem_kernel(backbone):
    return [v for v in backbone.weights if "stem_conv" in v.path and len(v.shape) == 4][0]


def corn_head_initial_weights(seed):
    """Identical for P and PL at a given seed, independent of the global RNG (PL's larger first
    layer would otherwise consume extra random draws and change the head's initialisation)."""
    import keras
    kernel = keras.initializers.GlorotUniform(seed=int(seed))((FEATURE_DIM, NUM_THRESHOLDS))
    return np.asarray(kernel, dtype=np.float32), np.zeros((NUM_THRESHOLDS,), dtype=np.float32)


def build_pl_model(arm, seed, reference_arrays, image_size=IMAGE_SIZE):
    """The P or PL model for one seed, pretrained weights copied in, head initialised. Uncompiled.
    Build under the dtype policy the run uses (set it BEFORE calling)."""
    import keras
    from keras import Input, Model

    import corn
    import joint_training_model as jtm

    ChannelAdapter, InertAuxiliaryInputs = _layers()
    keras.utils.set_random_seed(int(seed))
    stage5 = Input(shape=(image_size, image_size, 8), name="stage5_input")
    stage6 = Input(shape=jtm.STAGE6_INPUT_SHAPE, name="stage6_input")
    reliability = Input(shape=(1,), name="reliability")
    backbone = build_backbone(arm, image_size)
    copy_report = copy_pretrained(backbone, reference_arrays, arm)
    head = corn.build_corn_model(d_model=FEATURE_DIM)
    kernel, bias = corn_head_initial_weights(seed)
    head.get_layer("corn_logits").set_weights([kernel, bias])
    features = backbone(ChannelAdapter(arm, name=f"channel_adapter_{arm.lower()}")(stage5))
    logits = head(features)
    logits = InertAuxiliaryInputs(name="inert_auxiliary_inputs")([logits, stage6, reliability])
    model = Model(inputs=[stage5, stage6, reliability], outputs=logits,
                  name=f"{BACKBONE_NAME}_{arm.lower()}")
    model.pl_copy_report = copy_report
    return model


def trainable_parameter_count(model_or_layer):
    return int(sum(int(np.prod(v.shape)) for v in model_or_layer.trainable_variables))


def build_optimizer(model, learning_rate, weight_decay, exclude_names=("bias", "gamma", "beta")):
    """AdamW; no weight decay on ANY 1-D trainable parameter (biases, LayerNorm scale/shift, and the
    unnamed ConvNeXt LayerScale `variable_N`), plus the project's name-based exclusions. Call after
    the model is built and before compile."""
    import tensorflow as tf
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    one_d = [v for v in model.trainable_variables if len(v.shape) <= 1]
    optimizer.exclude_from_weight_decay(var_list=one_d, var_names=list(exclude_names))
    return optimizer


def compile_pl_model(model, class_weights, learning_rate, weight_decay):
    import corn
    import weighted_corn
    model.compile(optimizer=build_optimizer(model, learning_rate, weight_decay),
                  loss=weighted_corn.make_weighted_corn_loss(class_weights),
                  metrics=[corn.CORNQuadraticWeightedKappa(), weighted_corn.UnweightedCORNLoss()])
    return model


def inner_optimizer(model):
    optimizer = model.optimizer
    return getattr(optimizer, "inner_optimizer", None) or optimizer


def weight_decay_report(model):
    """Which trainable variables AdamW would decay. Correct iff exactly the 1-D ones are exempt."""
    optimizer = inner_optimizer(model)
    variables = model.trainable_variables
    one_d = [v for v in variables if len(v.shape) <= 1]
    decayed_one_d = [v.path for v in one_d if optimizer._use_weight_decay(v)]
    exempt_multi_d = [v.path for v in variables if len(v.shape) > 1
                      and not optimizer._use_weight_decay(v)]
    layer_scale = [v for v in variables if "layer_scale" in v.path]
    return {"trainable": len(variables), "one_d": len(one_d),
            "layer_scale": len(layer_scale),
            "layer_scale_exempt": sum(not optimizer._use_weight_decay(v) for v in layer_scale),
            "decayed_one_d": decayed_one_d, "exempt_multi_d": exempt_multi_d,
            "ok": (not decayed_one_d and not exempt_multi_d
                   and len(layer_scale) == EXPECTED_LAYER_SCALE_VARIABLES)}


# ---------------------------------------------------------------------------------------------
# Metrics and the locked decision rule
# ---------------------------------------------------------------------------------------------

def sensitivity_at_specificity(grades, p_ge3, specificity=SPECIFICITY):
    """Grade-4 sensitivity at the P(grade>=3) value that leaves >= `specificity` of grade 0-2 images
    at or below it (an evaluation point on the ROC curve, not a deployment threshold)."""
    g = np.asarray(grades, dtype=int)
    s = np.asarray(p_ge3, dtype=np.float64)
    negatives, positives = s[g <= 2], s[g == 4]
    threshold = float(np.quantile(negatives, specificity, method="higher"))
    return {"sensitivity": float(np.mean(positives > threshold)),
            "specificity_achieved": float(np.mean(negatives <= threshold)),
            "threshold": threshold}


def run_metrics(frame):
    """Every locked endpoint for one run from its per-sample CSV (multiseed schema)."""
    import icdr_two_route_experiment as exp
    g = frame["true_grade"].to_numpy(int)
    pred = frame["predicted_grade"].to_numpy(int)
    p_ge3 = frame["p_gt_2"].to_numpy(np.float64)        # CORN: P(grade >= 3) = p_gt_2
    p4 = frame["p_gt_3"].to_numpy(np.float64)
    metrics = exp.head_metrics(g, pred, p_ge3, p4, persistent_mask=np.zeros(len(g), dtype=bool))
    metrics.pop("persistent21_decoded_le2", None)      # not an endpoint in this experiment
    sens = sensitivity_at_specificity(g, p_ge3)
    metrics.update(grade4_sensitivity_at_95_specificity=sens["sensitivity"],
                   specificity_achieved=sens["specificity_achieved"])
    return metrics


def decide(deltas, guardrail_mean_deltas):
    """The locked rule: SUPPORTIVE = mean >= +0.03 AND > 0 in all seeds AND guardrails hold;
    NOT_SUPPORTIVE = mean < +0.01 OR <= 0 in >= 2 seeds; otherwise INCONCLUSIVE (including a met
    primary with a failed guardrail)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    mean = float(deltas.mean())
    positive, nonpositive = int(np.sum(deltas > 0)), int(np.sum(deltas <= 0))
    guard = {}
    for name, (kind, bound) in GUARDRAILS.items():
        value = float(guardrail_mean_deltas[name])
        holds = value >= bound - 1e-12 if kind == "min" else value <= bound + 1e-12
        guard[name] = {"mean_delta": value, "bound": bound, "kind": kind, "holds": bool(holds)}
    guards_hold = all(g["holds"] for g in guard.values())
    if mean < NOT_SUPPORTIVE_MEAN or nonpositive >= NOT_SUPPORTIVE_MIN_NONPOSITIVE:
        verdict = "NOT_SUPPORTIVE"
    elif mean >= SUPPORTIVE_MEAN and positive == deltas.size and guards_hold:
        verdict = "SUPPORTIVE"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "mean_delta": mean,
            "sd_delta": float(deltas.std(ddof=1)) if deltas.size > 1 else float("nan"),
            "per_seed_delta": deltas.tolist(), "seeds_positive": positive,
            "guardrails": guard, "guardrails_hold": bool(guards_hold)}


# ---------------------------------------------------------------------------------------------
# Read-only fingerprint of Stage 3/4 models and caches
# ---------------------------------------------------------------------------------------------

def directory_fingerprint(paths):
    """(relative path, size, mtime) of every file under each existing path -- recorded before and
    after the experiment to prove Stage 3/4 models and caches were not written."""
    record = {}
    for root in paths:
        if not root or not os.path.exists(root):
            record[str(root)] = "absent"
            continue
        entries = []
        for directory, _dirs, files in os.walk(root):
            for name in sorted(files):
                full = os.path.join(directory, name)
                stat = os.stat(full)
                entries.append([os.path.relpath(full, root), stat.st_size, int(stat.st_mtime)])
        entries.sort()
        record[str(root)] = hashlib.sha256(repr(entries).encode("utf-8")).hexdigest() + \
            f" ({len(entries)} files)"
    return record
