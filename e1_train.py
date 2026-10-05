"""E1 multi-task grader: compile, loss, log aliases, training loop and evaluation (Colab, GPU, for training).

Hypothesis under test: auxiliary supervision from the frozen Stage-4 lesion maps, applied to the feature map the
grader pools, can be added to P at no material cost in grading, and changes what that feature map encodes.

What is reused, unchanged: the v2 bundle and its integrity checks (arch1_data.Arch1Bundle), the APTOS split,
the P protocol (arch1_train.P_PROTOCOL: batch 2, AdamW 1e-4 / wd 0.05 with no decay on 1-D parameters, weighted
CORN with the pre-registered class weights, <= 50 epochs, early stopping on val QWK, ReduceLROnPlateau, mixed
precision), P's epoch order and per-image augmentation RNG, the generation-based checkpoints (training.Trainer,
multiseed_runs), seeds 42 -> 123 -> 2026, one run directory per seed, BEST by validation QWK, no EMA.

What differs from P: the auxiliary head (e1_model) and its loss, with weight LAMBDA = 1.0 exactly:

    L_total = L_CORN + 1.0 * L_lesion
    L_lesion = mean over batch, 16 x 16 cells and 4 classes of sigmoid_cross_entropy_with_logits(T, Z), float32

A two-output Keras model prefixes its metric names with the output name (`val_corn_QWK`), while the run
machinery (multiseed_runs, training.checkpointing) reads `val_QWK`. `LogAliases` copies the grading output's
metrics to the names P's runs use; it must be the FIRST callback. The total Keras `loss` / `val_loss` is CORN +
lesion and is not comparable with P's; `corn_loss` and `corn_loss_unweighted` are.

The training loop mirrors arch1_train.train_seed. It is repeated here rather than parameterised there because the
Architecture-1 module is a completed experiment's code and is not modified. Nothing here defines a success
threshold or tunes anything.
"""
import json
import os
import posixpath

import numpy as np

import arch1_train as at
import e1_data as ed
import e1_model as em

SEEDS = at.SEEDS
EXPERIMENT = "E1MultiTask"
PREFIX = "e1"
LAMBDA = 1.0                                                    # locked; never tuned
#: Keras' names for the grading output's metrics -> the names the run machinery reads.
ALIASES = {"val_corn_QWK": "val_QWK", "corn_QWK": "QWK",
           "val_corn_corn_loss_unweighted": "val_corn_loss_unweighted",
           "corn_corn_loss_unweighted": "corn_loss_unweighted"}
REQUIRED_TRAIN_KEYS = ("corn_QWK", "corn_corn_loss_unweighted")
REQUIRED_VAL_KEYS = ("val_corn_QWK", "val_corn_corn_loss_unweighted")


# --------------------------------------------------------------------------- loss

def lesion_loss(y_true, y_pred):
    """Mean over batch, cells and classes of the sigmoid cross-entropy between the soft teacher targets and
    the auxiliary logits, in float32."""
    import tensorflow as tf
    logits = tf.cast(tf.convert_to_tensor(y_pred), tf.float32)
    targets = tf.cast(tf.convert_to_tensor(y_true), tf.float32)
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))


def lesion_loss_numpy(targets, logits):
    """The same quantity in NumPy float64 (reference for the tests and the evaluation)."""
    t, z = np.asarray(targets, np.float64), np.asarray(logits, np.float64)
    return float(np.mean(np.maximum(z, 0.0) - z * t + np.log1p(np.exp(-np.abs(z)))))


# --------------------------------------------------------------------------- log aliases

def make_log_aliases(strict=True):
    """The alias callback. Copies each source key that exists to its alias (never the other way, never a key
    that is absent). `strict`: an epoch whose logs lack the grading output's metrics raises instead of training
    on with a monitor that would silently be missing."""
    import keras

    class LogAliases(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.observed_keys = []
            self.created = []

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                raise RuntimeError("no epoch logs to alias")
            self.observed_keys = sorted(logs)
            required = REQUIRED_TRAIN_KEYS + (REQUIRED_VAL_KEYS if any(k.startswith("val_") for k in logs) else ())
            missing = [k for k in required if k not in logs]
            if strict and missing:
                raise RuntimeError(f"Keras did not log {missing}; logged keys: {sorted(logs)}")
            created = []
            for source, alias in ALIASES.items():
                if source not in logs:
                    continue
                if alias in logs and logs[alias] is not logs[source] and logs[alias] != logs[source]:
                    raise RuntimeError(f"log key {alias!r} already exists with another value")
                logs[alias] = logs[source]
                created.append(alias)
            self.created = created

    return LogAliases()


def with_aliases(callbacks, strict=True):
    """[LogAliases] + callbacks: the alias callback first, so every later callback sees the aliased keys."""
    callbacks = list(callbacks)
    if any(type(c).__name__ == "LogAliases" for c in callbacks):
        raise RuntimeError("the alias callback is already in the list")
    return [make_log_aliases(strict)] + callbacks


# --------------------------------------------------------------------------- model + compile

def compile_e1_model(model, class_weights, learning_rate, weight_decay):
    """P's optimizer and grading loss / metrics (pl_convnext.compile_pl_model), plus the lesion loss at LAMBDA."""
    import corn
    import pl_convnext as pl
    import weighted_corn
    model.compile(optimizer=pl.build_optimizer(model, learning_rate, weight_decay),
                  loss=[weighted_corn.make_weighted_corn_loss(class_weights), lesion_loss],
                  loss_weights=[1.0, LAMBDA],
                  metrics=[[corn.CORNQuadraticWeightedKappa(), weighted_corn.UnweightedCORNLoss()], []])
    return model


def build_compiled_model(seed, lesion_prior, reference, class_weights, protocol=at.P_PROTOCOL, mixed_precision=True,
                         image_size=em.IMAGE_SIZE):
    from training import enable_mixed_precision
    enable_mixed_precision(bool(mixed_precision))
    model = em.build_e1_model(seed, lesion_prior, reference, image_size=image_size)
    return compile_e1_model(model, list(class_weights), protocol["learning_rate"], protocol["weight_decay"])


def run_mapping(bundle, seed, class_weights, lesion_prior, protocol=at.P_PROTOCOL, repo_dir=None):
    """Everything that identifies a run (written to config.json and hashed into the checkpoint config)."""
    import arch1_model
    import pl_convnext as pl
    return {"experiment": EXPERIMENT, "arch": "convnext_tiny + CORN + 1x1 lesion head on the final feature map",
            "seed": int(seed), "split_sha256": bundle.bundle["split_sha256"],
            "population_sha256": bundle.bundle.get("population_sha256"), "bundle_id": bundle.bundle["bundle_id"],
            "bundle_fingerprint": bundle.fingerprint, "stage4_sha256": bundle.stage4_sha256,
            "stage4_generation": bundle.stage4_generation, "channels": list(bundle.channels),
            "backbone_weights_sha256": pl.WEIGHTS_SHA256, "convnext_dims": list(arch1_model.DIMS),
            "inputs": ["rgb"], "caches_read": list(ed.KINDS), "lesion_classes": list(em.LESION_CLASSES),
            "lesion_target": {"source_channels": [f"{c}:{ed.POOLING}" for c in em.LESION_CLASSES],
                              "grid": ed.GRID, "pooling": "block maximum", "soft": True},
            "lesion_loss": "mean sigmoid cross-entropy with logits (batch, cells, classes), float32",
            "lesion_loss_weight": LAMBDA, "lesion_prior": [float(p) for p in lesion_prior],
            "lesion_prior_clip": list(em.PRIOR_CLIP), "added_parameters": em.ADDED_PARAMETERS,
            "log_aliases": dict(ALIASES), "vessel_supervision": False,
            "class_weights": [float(w) for w in class_weights], "optimizer": "AdamW (no decay on 1-D)",
            "augmentation": "P: lfed._augment_spatial (RGB + targets) + _augment_intensity_rgb (RGB only)",
            **protocol, "mixed_precision": True, "ema": at.EMA, "git_commit": at.git_commit(repo_dir)}


# --------------------------------------------------------------------------- prediction + evaluation

def predict_outputs(model, bundle, image_ids, batch_size=8):
    """(grading logits (N, 4) float64, lesion logits (N, 16, 16, 4) float32) for `image_ids`, unaugmented."""
    grading, lesion = [], []
    ids = [str(i) for i in image_ids]
    for s in range(0, len(ids), batch_size):
        rgb = np.stack([ed.load_inputs(bundle, i)["rgb"] for i in ids[s:s + batch_size]])
        g, z = model.predict_on_batch({"rgb": rgb})
        grading.append(np.asarray(g, np.float64))
        lesion.append(np.asarray(z, np.float32))
    return np.concatenate(grading, 0), np.concatenate(lesion, 0)


def validation_targets(bundle, image_ids):
    return np.stack([ed.pool_targets(ed.load_inputs(bundle, str(i))["pathology"], bundle.channels) for i in image_ids])


def lesion_head_report(targets, logits):
    """Agreement of the auxiliary head with the teacher on a set of images (descriptive): the loss, and the
    cell-level AUROC of each class against target >= 0.5."""
    from sklearn.metrics import roc_auc_score
    report = {"lesion_loss": lesion_loss_numpy(targets, logits), "cell_auroc": {}}
    for c, name in enumerate(em.LESION_CLASSES):
        y = (targets[..., c] >= ed.POSITIVE).ravel()
        report["cell_auroc"][name] = float(roc_auc_score(y, logits[..., c].ravel())) if 0 < y.sum() < y.size else float("nan")
    values = [v for v in report["cell_auroc"].values() if np.isfinite(v)]
    report["mean_cell_auroc"] = float(np.mean(values)) if values else float("nan")
    return report


def evaluate_run(run_dir, bundle, seed, lesion_prior, reference, class_weights, protocol=at.P_PROTOCOL, grade_of=None,
                 mixed_precision=True):
    """BEST and LAST checkpoints -> per-sample grading tables (P's schema), grading metrics and the auxiliary
    head's agreement with the teacher on the validation images. Written under run_dir/metrics."""
    import pandas as pd
    import tensorflow as tf

    import multiseed_runs as msr
    from training import checkpointing as ckpt
    val_ids = list(bundle.val_ids)
    grades = at._grades(bundle, val_ids, grade_of)
    targets = validation_targets(bundle, val_ids)
    out_dir = posixpath.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    best_dir, _ = msr.read_best(run_dir)
    last_dir = ckpt.find_resumable_generation(posixpath.join(run_dir, "checkpoints"), verbose=False)
    for which, gen_dir in (("best", best_dir), ("last", last_dir)):
        if gen_dir is None:
            raise RuntimeError(f"{run_dir}: no {which.upper()} checkpoint")
        tf.keras.backend.clear_session()
        model = build_compiled_model(seed, lesion_prior, reference, class_weights, protocol, mixed_precision)
        weights = os.path.join(gen_dir, ckpt.MODEL_WEIGHTS_FILENAME)
        ckpt.load_model_weights_only(model, weights)
        grading, lesion = predict_outputs(model, bundle, val_ids)
        metrics, rows = at.metrics_from_logits(val_ids, grades, grading)
        state = ckpt.read_state(gen_dir)
        res = {"metrics": metrics, "lesion_head_validation": lesion_head_report(targets, lesion),
               "checkpoint": {"which": which.upper(), "generation": gen_dir, "weights_sha256": at._sha256(weights),
                              "completed_epoch": state.completed_epoch, "best_epoch": state.best_epoch,
                              "best_metric": state.best_metric, "monitor": state.monitor,
                              "learning_rate": state.learning_rate}}
        pd.DataFrame(rows).to_csv(posixpath.join(out_dir, f"per_sample_{which}.csv"), index=False)
        with open(posixpath.join(out_dir, f"metrics_{which}.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        results[which] = res
    return results


# --------------------------------------------------------------------------- training (P's loop)

def train_seed(run_dir, bundle, seed, lesion_prior, reference, class_weights, *, repo_dir, staging_dir,
               protocol=at.P_PROTOCOL, log=print, max_epochs=None, grade_of=None, mixed_precision=True):
    """One resumable E1 run, structured exactly like arch1_train.train_seed; the alias callback runs first."""
    import tensorflow as tf

    import multiseed_runs as msr
    from training import CheckpointOptions, Trainer, TrainingConfig, TrainingStateCheckpoint
    from training import checkpointing as ckpt
    epochs = int(max_epochs or protocol["max_epochs"])
    msr.ensure_run_dir(run_dir)
    chash = at.ensure_run_config(run_dir, run_mapping(bundle, seed, class_weights, lesion_prior, protocol, repo_dir))
    if msr.read_stop_decision(run_dir) is not None:
        log("  run already stopped")
        return
    sealed = msr._sealed_stop(posixpath.join(run_dir, "checkpoints"), epochs)
    if sealed is not None:
        msr.write_stop_decision(run_dir, sealed[0], sealed[1])
        return
    tf.keras.backend.clear_session()
    model = build_compiled_model(seed, lesion_prior, reference, class_weights, protocol, mixed_precision)
    at.acquire_lock_waiting(run_dir, log)
    try:
        trainer = Trainer(TrainingConfig(
            run_dir=run_dir, epochs=epochs, monitor=protocol["monitor"], mode=protocol["mode"],
            mixed_precision=bool(mixed_precision),
            resume=True, early_stopping_patience=protocol["early_stopping_patience"],
            reduce_lr_patience=protocol["reduce_lr_patience"], reduce_lr_factor=protocol["reduce_lr_factor"],
            min_lr=protocol["min_lr"], precision_check="error", repo_dir=repo_dir,
            checkpoint_options=CheckpointOptions(
                experiment_id=f"{EXPERIMENT}/seed_{seed}", config_hash=chash,
                dataset_version=f"bundle:{bundle.fingerprint}", staging_dir=staging_dir, keep_generations=2, verbose=1)))
        trainer.prepare(model)
        initial_epoch = trainer.resolve_initial_epoch()
        if initial_epoch > 0:
            trainer.restore(model)
            log(f"  resumed at epoch {initial_epoch}")
        else:                                       # a fresh model: record what this seed started from
            with open(posixpath.join(run_dir, "initialization.json"), "w") as fh:
                json.dump({"seed": int(seed), "initial_epoch": 0, "weights_sha256": at.weights_sha256(model),
                           "pretrained_backbone_file_sha256": at._pretrained_sha256()}, fh, indent=1)
        if initial_epoch >= epochs:
            msr.write_stop_decision(run_dir, initial_epoch, "epoch_cap")
            return
        early = next(c for c in trainer.callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping))
        state_cb = next(c for c in trainer.callbacks if isinstance(c, TrainingStateCheckpoint))
        callbacks = with_aliases(trainer.callbacks)
        train_entries = list(zip(bundle.train_ids, at._grades(bundle, bundle.train_ids, grade_of)))
        val_entries = list(zip(bundle.val_ids, at._grades(bundle, bundle.val_ids, grade_of)))
        val_seq = ed.make_epoch_sequence(bundle, val_entries, 0, seed, protocol["batch_size"], augment=False)
        for epoch in range(initial_epoch, epochs):
            msr.heartbeat_lock(run_dir, owner_id=msr.OWNER_ID)
            train_seq = ed.make_epoch_sequence(bundle, train_entries, epoch, seed, protocol["batch_size"], augment=True)
            model.fit(train_seq, validation_data=val_seq, epochs=epoch + 1, initial_epoch=epoch,
                      callbacks=callbacks, verbose=1)
            generation = state_cb.last_generation_dir
            if generation is None:
                raise RuntimeError(f"epoch {epoch}: no checkpoint generation written")
            state = ckpt.read_state(generation)
            if state.best_metric is None or (state.extra or {}).get("epoch_logs", {}).get("val_QWK") is None:
                raise RuntimeError(f"epoch {epoch}: the checkpoint state holds no val_QWK -- the log aliases did not "
                                   "reach the checkpoint callback")
            msr.write_epoch_history(run_dir, state)
            if state.best_epoch == epoch:
                msr.publish_best(run_dir, generation, state, repo_dir=repo_dir, verbose=1)
            if early.stopped_epoch:
                msr.write_stop_decision(run_dir, state.completed_epoch, "early_stopping")
                break
            if state.completed_epoch >= epochs:
                msr.write_stop_decision(run_dir, state.completed_epoch, "epoch_cap")
                break
    finally:
        msr.release_lock(run_dir, owner_id=msr.OWNER_ID)


# --------------------------------------------------------------------------- three-seed sequence

def run_dir_for(experiments_root, stage4_sha256, seed):
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")
    return posixpath.join(experiments_root, EXPERIMENT, f"{PREFIX}_{stage4_sha256[:12]}_seed{int(seed)}")


def seed_state(run_dir):
    """"complete" (result written) | "trained" (stop decided, not yet evaluated) | "in_progress" | "new"."""
    import multiseed_runs as msr
    if os.path.exists(posixpath.join(run_dir, "result.json")):
        return "complete"
    if msr.read_stop_decision(run_dir) is not None:
        return "trained"
    return "in_progress" if os.path.exists(posixpath.join(run_dir, "config.json")) else "new"


def write_result(run_dir, results, gates=None):
    """result.json for one seed: the BEST / LAST evaluation, the run's identity, history and stop decision.
    It states no verdict: the comparison with P is made afterwards under the pre-registered rules."""
    import multiseed_runs as msr
    with open(posixpath.join(run_dir, "config.json")) as fh:
        config = json.load(fh)
    payload = {"experiment": EXPERIMENT, "seed": config["seed"], "config": config, "best": results["best"],
               "last": results["last"], "history": msr.read_history(run_dir), "stop": msr.read_stop_decision(run_dir),
               "gates": gates}
    path = posixpath.join(run_dir, "result.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    os.replace(path + ".tmp", path)
    return payload


def run_sequence(experiments_root, bundle, lesion_prior, reference, class_weights, *, repo_dir, staging_root,
                 gates=None, log=print, train_fn=None, evaluate_fn=None):
    """Seeds 42 -> 123 -> 2026, one after another, each model built fresh. Refuses to start unless the three
    pre-training gates are recorded as passed. Any exception stops the sequence there. A completed seed is
    kept as recorded."""
    import gc

    import multiseed_runs as msr
    missing = [g for g in ("p_parity", "log_alias", "targets") if not (gates or {}).get(g, {}).get("PASS")]
    if missing:
        raise RuntimeError(f"pre-training gates not passed: {missing}")
    results = {}
    for n, seed in enumerate(SEEDS, start=1):
        run_dir = run_dir_for(experiments_root, bundle.stage4_sha256, seed)
        log(f"=== [{n}/{len(SEEDS)}] seed {seed}: {seed_state(run_dir)} | {run_dir}")
        if seed_state(run_dir) == "complete":
            with open(posixpath.join(run_dir, "result.json")) as fh:
                results[seed] = json.load(fh)
            continue
        (train_fn or train_seed)(run_dir, bundle, seed, lesion_prior, reference, class_weights, repo_dir=repo_dir,
                                 staging_dir=posixpath.join(staging_root, f"seed_{seed}"), log=log)
        if msr.read_stop_decision(run_dir) is None:
            raise RuntimeError(f"{run_dir}: seed {seed} returned without a stop decision -- training is not finished")
        evaluated = (evaluate_fn or evaluate_run)(run_dir, bundle, seed, lesion_prior, reference, class_weights)
        results[seed] = write_result(run_dir, evaluated, gates)
        best = results[seed]["best"]
        log(f"=== [{n}/{len(SEEDS)}] seed {seed} DONE: QWK {best['metrics']['qwk']:.4f} | lesion head mean cell AUROC "
            f"{best['lesion_head_validation']['mean_cell_auroc']:.4f}")
        gc.collect()
    return results
