"""Joint APTOS + EyePACS training of the P grader ("J") -- research record §93.

One change from P: the training data. Everything else is P's or the EyePACS adaptation run's, reused and not
re-implemented: ConvNeXt-Tiny from ImageNet + GAP + LayerNorm + CORN (pl_convnext), the direct-resize 512 frames
already cached for both datasets, P's augmentation and per-image RNG, weighted CORN, AdamW 1e-4 / wd 0.05, the
resumable generation-based loop (eyepacs_adaptation_train.training_loop), batch 16 as in the adaptation run, the
fixed 0.5 CORN decode. No crop, no new augmentation, no architecture change.

What is specific to this experiment and lives here:

  * the epoch: all APTOS training images plus twice as many EyePACS training images, the EyePACS part being a
    rotating slice of ONE seeded shuffle -- a pure function of (seed, epoch), so a resumed run sees exactly the
    epochs an uninterrupted run would;
  * the class weights: P's formula (weighted_corn) on the grade counts of that mixture;
  * checkpoint selection: BEST = the highest MEAN of the APTOS validation QWK and the EyePACS validation QWK,
    both on their FULL sets (730 and 3,512), scored every epoch -- equal influence, whatever the set sizes;
  * the continuation rule: seeds 123 and 2026 run only if seed 42 is at most 0.01 below P on APTOS validation
    and at most 0.02 below the EyePACS-only model on EyePACS validation.

Validation images of either dataset are never in a training epoch. The EyePACS held-out test, IDRiD and DDR are
not read: this module has no path to them.
"""
import hashlib
import json
import os
import posixpath

import numpy as np

import arch1_train as at
import eyepacs_adaptation_train as eat

EXPERIMENT = "JointAptosEyePACS"
PREFIX = "j"
SEEDS = at.SEEDS                                # 42 first; 123 and 2026 only through the continuation rule
PROTOCOL = dict(at.P_PROTOCOL, batch_size=16)   # P's protocol at the adaptation run's batch size
EYEPACS_PER_APTOS = 2                           # EyePACS training images per APTOS training image in every epoch
ORDER_TAG = "joint-eyepacs-rotation-v1"
APTOS_TRAIN, EYEPACS_TRAIN = 2921, 31610
#: P's formula on the expected grade counts of one epoch (1,444 / 296 / 791 / 154 / 236 APTOS + 5,842 EyePACS at
#: its training prevalence), to four decimals; recorded in §93 before any run.
APPROVED_CLASS_WEIGHTS = (0.6456, 1.8421, 1.1954, 2.8293, 2.5958)
WEIGHT_TOLERANCE = 0.02                         # the weights of the mixture actually sampled must be this close
SELECTION = ("mean of the APTOS validation QWK (730 images) and the EyePACS validation QWK (3,512 images), both full "
             "sets, every epoch; first epoch with the highest value")
APTOS_MARGIN, EYEPACS_MARGIN = 0.01, 0.02       # continuation rule
VAL_BATCH = 16
#: The two arms (record §93, §94). "direct" is the primary model and its configuration must not change; "crop"
#: differs in the frames only (every image cut to the fundus box before the resize) and has no continuation rule.
ARMS = {"direct": {"prefix": PREFIX, "continuation_rule": True},
        "crop": {"prefix": "jc", "continuation_rule": False}}
CROP_PREPROCESSING = "Stage 2 + cut to the fundus bounding box (geometry_diagnostic.fundus_box) + direct resize to 512 (crop_frames cache)"


class JointError(RuntimeError):
    """The data, the run or the continuation rule is not what §93 describes."""


# --------------------------------------------------------------------------- data

class JointFrames:
    """`frame(image)` over both caches. APTOS ids and EyePACS file names never coincide; this is checked."""

    def __init__(self, aptos_frames, eyepacs_frames, aptos_ids, eyepacs_ids):
        self._aptos, self._eyepacs = aptos_frames, eyepacs_frames
        self._is_aptos, self._is_eyepacs = set(map(str, aptos_ids)), set(map(str, eyepacs_ids))
        if self._is_aptos & self._is_eyepacs:
            raise JointError("an image name is in both datasets")

    def frame(self, image):
        image = str(image)
        if image in self._is_aptos:
            return self._aptos.frame(image)
        if image in self._is_eyepacs:
            return self._eyepacs.frame(image)
        raise JointError(f"{image} is in neither dataset's permitted images")


def eyepacs_rotation(eyepacs_train_entries, seed):
    """The EyePACS training images in ONE fixed shuffled order for this seed (sorted by name, then permuted with a
    generator seeded from (ORDER_TAG, seed)). Depends on nothing else: not on the epoch, not on a checkpoint."""
    entries = sorted((str(i), int(g)) for i, g in eyepacs_train_entries)
    digest = hashlib.sha256(f"{ORDER_TAG}|{int(seed)}".encode()).digest()
    order = np.random.default_rng(int.from_bytes(digest[:8], "little")).permutation(len(entries))
    return [entries[k] for k in order]


def epoch_entries(aptos_train_entries, rotation, epoch):
    """The images of one epoch: every APTOS training image and the next 2 x len(APTOS) images of the rotation,
    wrapping round. A pure function of (rotation, epoch)."""
    n = EYEPACS_PER_APTOS * len(aptos_train_entries)
    if n > len(rotation):
        raise JointError("the EyePACS training part is smaller than one epoch's share")
    start = (int(epoch) * n) % len(rotation)
    part = [rotation[(start + k) % len(rotation)] for k in range(n)]
    return [(str(i), int(g)) for i, g in aptos_train_entries] + part


def grade_counts(entries):
    return np.bincount([int(g) for _, g in entries], minlength=5)


def class_weights(aptos_train_entries, eyepacs_train_entries, seed=None, epochs=None):
    """(expected per-epoch grade counts, weights) by P's formula. The weights must be the recorded ones, and --
    when a seed is given -- within WEIGHT_TOLERANCE of the weights of the mixture ACTUALLY sampled over the
    run's epochs (its mean per-epoch grade counts), which is returned as the third value."""
    import weighted_corn
    share = EYEPACS_PER_APTOS * len(aptos_train_entries) / len(eyepacs_train_entries)
    expected = grade_counts(aptos_train_entries) + grade_counts(eyepacs_train_entries) * share
    weights = weighted_corn.class_weights_from_counts(expected)
    if not np.allclose(weights, APPROVED_CLASS_WEIGHTS, atol=5e-5):
        raise JointError(f"class weights {np.round(weights, 4).tolist()} are not the recorded {APPROVED_CLASS_WEIGHTS}")
    sampled = None
    if seed is not None:
        rotation = eyepacs_rotation(eyepacs_train_entries, seed)
        counts = np.mean([grade_counts(epoch_entries(aptos_train_entries, rotation, e)) for e in range(int(epochs or PROTOCOL["max_epochs"]))], axis=0)
        actual = weighted_corn.class_weights_from_counts(counts)
        if np.abs(actual - weights).max() > WEIGHT_TOLERANCE:
            raise JointError(f"the sampled mixture gives weights {np.round(actual, 4).tolist()}, not {np.round(weights, 4).tolist()}")
        sampled = {"mean_counts_per_epoch": counts.tolist(), "weights": actual.tolist(), "max_abs_difference": float(np.abs(actual - weights).max())}
    return expected.tolist(), [float(w) for w in weights], sampled


def check_data(aptos_train, aptos_val, eyepacs_train, eyepacs_val):
    """The four parts are the permitted ones and no validation image can reach a training epoch."""
    names = [set(str(i) for i, _ in part) for part in (aptos_train, aptos_val, eyepacs_train, eyepacs_val)]
    if (len(aptos_train), len(eyepacs_train)) != (APTOS_TRAIN, EYEPACS_TRAIN):
        raise JointError(f"training parts hold {len(aptos_train)} / {len(eyepacs_train)} images, not {APTOS_TRAIN} / {EYEPACS_TRAIN}")
    if (len(aptos_val), len(eyepacs_val)) != (730, 3512):
        raise JointError("the validation parts are not the 730 / 3,512 images")
    if (names[0] | names[2]) & (names[1] | names[3]):
        raise JointError("a validation image is in a training part")
    return True


# --------------------------------------------------------------------------- selection metric

def selection_metric(aptos_qwk, eyepacs_qwk):
    """The monitored value: the two full-set QWKs with equal weight."""
    return 0.5 * (float(aptos_qwk) + float(eyepacs_qwk))


def validation_qwk(model, frames, entries):
    import arch1_posthoc as ph
    import corn
    ids, grades = [str(i) for i, _ in entries], np.asarray([int(g) for _, g in entries])
    logits = eat.predict_logits(model, frames, ids, VAL_BATCH)
    return float(ph._fast_qwk(grades, np.asarray(corn.decode_logits(logits)["predicted_grade"]).astype(int)))


def dual_validation_callback(run_dir, frames, aptos_val, eyepacs_val, qwk_fn=validation_qwk):
    """Scores BOTH full validation sets at the end of every epoch and makes their mean the monitored `val_QWK`
    (Keras' own value, computed on the APTOS part passed to fit, is kept as `val_QWK_keras_aptos`). It must come
    before the trainer's callbacks, which read `val_QWK`. Every epoch's two values are also written to
    run_dir/dual_validation/, one file per epoch (rewriting the same epoch gives the same file)."""
    import keras

    class DualValidation(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                raise JointError("no epoch logs to write the selection metric into")
            a, e = qwk_fn(self.model, frames, aptos_val), qwk_fn(self.model, frames, eyepacs_val)
            logs["val_QWK_keras_aptos"] = logs.get("val_QWK")
            logs["val_QWK_aptos"], logs["val_QWK_eyepacs"] = a, e
            logs["val_QWK"] = selection_metric(a, e)
            out = posixpath.join(run_dir, "dual_validation")
            os.makedirs(out, exist_ok=True)
            with open(posixpath.join(out, f"epoch_{int(epoch):04d}.json"), "w") as fh:
                json.dump({"epoch": int(epoch), "val_QWK_aptos": a, "val_QWK_eyepacs": e, "selection_metric": logs["val_QWK"],
                           "val_QWK_keras_aptos": logs["val_QWK_keras_aptos"]}, fh)

    return DualValidation()


# --------------------------------------------------------------------------- run identity

def run_dir_for(experiments_root, seed, arm="direct"):
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
    return posixpath.join(experiments_root, EXPERIMENT, f"{ARMS[arm]['prefix']}_seed{int(seed)}")


def run_mapping(seed, weights, expected_counts, aptos_identity, eyepacs_identity, protocol=PROTOCOL, repo_dir=None, arm="direct"):
    """Everything that identifies a run. The direct arm's mapping is exactly the one of §93 (a run directory is
    bound to its configuration, and seed 42 of that arm was started with it); the cropped arm adds its arm name
    and crop rule and names its preprocessing."""
    mapping = _direct_mapping(seed, weights, expected_counts, aptos_identity, eyepacs_identity, protocol, repo_dir)
    if arm == "crop":
        import crop_frames as cf
        mapping.update(arm="crop", preprocessing=CROP_PREPROCESSING, crop_rule=dict(cf.RULE), crop_frames_version=cf.VERSION)
    elif arm != "direct":
        raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
    return mapping


def _direct_mapping(seed, weights, expected_counts, aptos_identity, eyepacs_identity, protocol=PROTOCOL, repo_dir=None):
    import pl_convnext as pl
    return {"experiment": EXPERIMENT, "arch": "P: convnext_tiny (ImageNet) + GAP + LayerNorm + CORN", "seed": int(seed),
            "encoder_initialisation": "ImageNet (as P); not the EyePACS-adapted encoder", "backbone_weights_sha256": pl.WEIGHTS_SHA256,
            "data": "APTOS training split + EyePACS adaptation-split training part", "aptos": aptos_identity, "eyepacs": eyepacs_identity,
            "epoch": f"all {APTOS_TRAIN} APTOS + {EYEPACS_PER_APTOS * APTOS_TRAIN} EyePACS (rotating slice)", "eyepacs_order": ORDER_TAG,
            "class_weights": [float(w) for w in weights], "class_weight_source": "P's formula on the expected per-epoch grade counts",
            "expected_grade_counts_per_epoch": [float(c) for c in expected_counts],
            "optimizer": "AdamW (no decay on 1-D)", "augmentation": "P: lfed._augment_spatial + _augment_intensity_rgb",
            "preprocessing": "Stage 2 + full-frame direct resize to 512 (cached frames); no crop",
            "checkpoint_selection": SELECTION, **protocol, "mixed_precision": True, "ema": at.EMA, "git_commit": at.git_commit(repo_dir)}


# --------------------------------------------------------------------------- training

def train_seed(run_dir, seed, frames, parts, reference_arrays, weights, mapping, *, repo_dir, staging_dir, protocol=PROTOCOL,
               log=print, max_epochs=None, mixed_precision=True, image_size=None, workers=2, qwk_fn=validation_qwk, arm="direct"):
    """One resumable run. `parts` = (APTOS train, APTOS val, EyePACS train, EyePACS val) entries."""
    aptos_train, aptos_val, eyepacs_train, eyepacs_val = parts
    rotation = eyepacs_rotation(eyepacs_train, seed)
    batch = protocol["batch_size"]
    dual = lambda callbacks: [dual_validation_callback(run_dir, frames, aptos_val, eyepacs_val, qwk_fn)] + list(callbacks)   # noqa: E731
    return eat.training_loop(
        run_dir, mapping,
        lambda: eat.build_compiled_model(seed, reference_arrays, weights, protocol, mixed_precision, image_size),
        lambda epoch: eat.make_epoch_sequence(frames, epoch_entries(aptos_train, rotation, epoch), epoch, seed, batch, True, workers),
        lambda: eat.make_epoch_sequence(frames, aptos_val, 0, seed, batch, False, workers),
        experiment_id=f"{EXPERIMENT}/seed_{seed}" if arm == "direct" else f"{EXPERIMENT}/{arm}/seed_{seed}", dataset_version=f"joint:{mapping['aptos'].get('fingerprint')}:{mapping['eyepacs'].get('fingerprint')}",
        protocol=protocol, repo_dir=repo_dir, staging_dir=staging_dir, log=log, max_epochs=max_epochs,
        mixed_precision=mixed_precision, wrap_callbacks=dual)


# --------------------------------------------------------------------------- evaluation of BEST

def top_label_ece(frame, bins=15):
    grade = frame["true_grade"].to_numpy(int)
    prob = np.stack([frame[f"class_prob_{k}"].to_numpy(np.float64) for k in range(5)], 1)
    confidence, correct = prob.max(1), (prob.argmax(1) == grade).astype(np.float64)
    edges, total = np.linspace(0.0, 1.0, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def describe(frame):
    """The reported quantities of one per-image table."""
    import arch1_posthoc as ph
    table = ph.table_arrays(frame)
    m = ph.metrics(table)
    referable, called = table["grade"] >= 2, table["pred"] >= 2
    return {"qwk": m["qwk"], "auroc_cuts": m["auroc_cuts"], "recall_per_grade": m["recall_per_grade"], "confusion_matrix": m["confusion_matrix"],
            "referable_sensitivity": float(called[referable].mean()), "referable_specificity": float((~called[~referable]).mean()),
            "ece_top_label": top_label_ece(frame), "mae": m["mae"]}


def evaluate_best(run_dir, build_model, frames, aptos_val, eyepacs_val):
    """The BEST checkpoint on both full validation sets: per-image tables (P's schema) and metrics."""
    import pandas as pd
    import tensorflow as tf

    import multiseed_runs as msr
    from training import checkpointing as ckpt
    best_dir, pointer = msr.read_best(run_dir)
    if best_dir is None:
        raise JointError(f"{run_dir}: no BEST checkpoint")
    tf.keras.backend.clear_session()
    model = build_model()
    path = os.path.join(best_dir, ckpt.MODEL_WEIGHTS_FILENAME)
    ckpt.load_model_weights_only(model, path)
    out_dir = posixpath.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    result = {"checkpoint": {"weights_sha256": _sha256(path), "slot": pointer["active"],
                             "epoch_index": int(pointer["epoch"]), "selection_metric": float(pointer["val_QWK"])}, "sets": {}}
    for name, entries in (("aptos_val", aptos_val), ("eyepacs_val", eyepacs_val)):
        ids, grades = [str(i) for i, _ in entries], [int(g) for _, g in entries]
        _, rows = at.metrics_from_logits(ids, grades, eat.predict_logits(model, frames, ids, VAL_BATCH))
        frame = pd.DataFrame(rows)
        frame.to_csv(posixpath.join(out_dir, f"per_sample_best_{name}.csv"), index=False)
        result["sets"][name] = describe(frame)
    with open(posixpath.join(out_dir, "metrics_best.json"), "w") as fh:
        json.dump(_plain(result), fh, indent=1)
    return result


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _plain(obj):
    import arch1_posthoc as ph
    return ph._jsonable(obj)


# --------------------------------------------------------------------------- comparison and continuation rule

def compare(joint_table_path, reference_table_path):
    """Joint model minus a reference model on the SAME images in the same order: QWK difference with the
    project's paired grade-stratified bootstrap, and the reference's own description."""
    import pandas as pd

    import arch1_posthoc as ph
    joint = pd.read_csv(joint_table_path, dtype={"image_id": str})
    reference = pd.read_csv(reference_table_path, dtype={"image_id": str})
    a, b = ph.table_arrays(joint), ph.table_arrays(reference)
    delta = ph.paired_delta(a, b, indices=ph.bootstrap_indices(a["grade"]))["qwk"]
    return {"joint_qwk": ph.metrics(a)["qwk"], "reference_qwk": ph.metrics(b)["qwk"], "difference": float(delta["delta"]),
            "ci_95": [float(delta["ci_low"]), float(delta["ci_high"])], "reference": describe(reference)}


def continuation(aptos, eyepacs):
    """The rule for seeds 123 and 2026, applied to seed 42's two comparisons (from `compare`)."""
    aptos_ok = aptos["difference"] >= -APTOS_MARGIN
    eyepacs_ok = eyepacs["difference"] >= -EYEPACS_MARGIN
    return {"rule": f"APTOS validation QWK no more than {APTOS_MARGIN} below P (same seed) AND EyePACS validation QWK no more than "
                    f"{EYEPACS_MARGIN} below the EyePACS-only model",
            "aptos_difference": aptos["difference"], "eyepacs_difference": eyepacs["difference"],
            "aptos_ok": bool(aptos_ok), "eyepacs_ok": bool(eyepacs_ok), "continue": bool(aptos_ok and eyepacs_ok)}


def state(run_dir):
    if os.path.exists(posixpath.join(run_dir, "result.json")):
        return "complete"
    return "started" if os.path.isdir(posixpath.join(run_dir, "checkpoints")) else "not started"


def run_seed(seed, experiments_root, frames, parts, reference_arrays, identities, references, *, repo_dir, staging_root, log=print,
             train_fn=None, evaluate_fn=None, mixed_precision=True, image_size=None, max_epochs=None, arm="direct"):
    """Train (or resume) one seed, evaluate BEST on both validation sets, compare with the references, write
    result.json. `references`: {'p_aptos': path of P's stored APTOS validation table for this seed,
    'eyepacs_only': path of the EyePACS-only model's stored EyePACS validation table}. Seeds 123 and 2026 are
    refused unless seed 42 is complete and its continuation rule holds. A completed seed is returned as recorded.
    `arm="crop"`: `frames` must be the cropped-frame caches; the run has its own directory, no continuation rule,
    and -- when `references` also names 'joint_aptos' / 'joint_eyepacs' (the direct arm's tables of the same seed)
    -- is compared with the direct arm as well."""
    import multiseed_runs as msr
    seed = int(seed)
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")
    run_dir = run_dir_for(experiments_root, seed, arm)
    if seed != SEEDS[0] and ARMS[arm]["continuation_rule"]:
        first = posixpath.join(run_dir_for(experiments_root, SEEDS[0]), "result.json")
        if not os.path.exists(first):
            raise JointError(f"seed {seed} needs the result of seed {SEEDS[0]} first")
        with open(first) as fh:
            if not json.load(fh)["continuation"]["continue"]:
                raise JointError(f"seed {SEEDS[0]} did not meet the continuation rule: seed {seed} is not run")
    if state(run_dir) == "complete":
        with open(posixpath.join(run_dir, "result.json")) as fh:
            return json.load(fh)
    check_data(*parts)
    aptos_train, aptos_val, eyepacs_train, eyepacs_val = parts
    expected, weights, sampled = class_weights(aptos_train, eyepacs_train, seed)
    mapping = run_mapping(seed, weights, expected, identities["aptos"], identities["eyepacs"], repo_dir=repo_dir, arm=arm)
    log(f"=== joint APTOS + EyePACS ({arm}), seed {seed}: {state(run_dir)} | {run_dir}")
    (train_fn or train_seed)(run_dir, seed, frames, parts, reference_arrays, weights, mapping, repo_dir=repo_dir,
                             staging_dir=posixpath.join(staging_root, f"{arm}_seed_{seed}"), log=log, mixed_precision=mixed_precision,
                             image_size=image_size, max_epochs=max_epochs, arm=arm)
    if msr.read_stop_decision(run_dir) is None:
        raise JointError(f"{run_dir}: the run returned without a stop decision -- training is not finished")
    build = lambda: eat.build_compiled_model(seed, reference_arrays, weights, PROTOCOL, mixed_precision, image_size)      # noqa: E731
    evaluated = (evaluate_fn or evaluate_best)(run_dir, build, frames, aptos_val, eyepacs_val)
    metrics_dir = posixpath.join(run_dir, "metrics")
    comparisons = {"aptos_val_vs_p": compare(posixpath.join(metrics_dir, "per_sample_best_aptos_val.csv"), references["p_aptos"]),
                   "eyepacs_val_vs_eyepacs_only": compare(posixpath.join(metrics_dir, "per_sample_best_eyepacs_val.csv"), references["eyepacs_only"])}
    for key, name in (("joint_aptos", "aptos_val"), ("joint_eyepacs", "eyepacs_val")):      # the cropped arm against the direct arm
        if arm == "crop" and references.get(key) and os.path.exists(references[key]):
            comparisons[f"{name}_vs_direct_arm"] = compare(posixpath.join(metrics_dir, f"per_sample_best_{name}.csv"), references[key])
    history = sorted(os.listdir(posixpath.join(run_dir, "dual_validation"))) if os.path.isdir(posixpath.join(run_dir, "dual_validation")) else []
    result = {"experiment": EXPERIMENT, "arm": arm, "seed": seed, "selection": SELECTION, "class_weights": weights, "sampled_mixture": sampled,
              "best": evaluated, "comparisons": comparisons, "epochs_scored": len(history), "stop": msr.read_stop_decision(run_dir),
              "limitations": ["batch size 16 here against 2 for P: the comparison with P confounds joint data with batch size",
                              "both validation sets selected the checkpoint: their values are selection-set values",
                              "EyePACS validation is in-domain for this model: nothing here is evidence of generalisation to unseen data"]}
    if seed == SEEDS[0] and ARMS[arm]["continuation_rule"]:
        result["continuation"] = continuation(comparisons["aptos_val_vs_p"], comparisons["eyepacs_val_vs_eyepacs_only"])
    with open(posixpath.join(run_dir, "result.json.tmp"), "w") as fh:
        json.dump(_plain(result), fh, indent=1)
    os.replace(posixpath.join(run_dir, "result.json.tmp"), posixpath.join(run_dir, "result.json"))
    return result
