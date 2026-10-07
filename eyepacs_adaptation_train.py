"""EyePACS adaptation of P (research record §69, §69.2): the trainer. Colab, GPU. Nothing here reads an APTOS file.

ONE adaptation run (seed ADAPTATION_SEED). The model is P itself (pl_convnext.build_pl_model("P", ...): ImageNet
ConvNeXt-Tiny, average pooling, LayerNorm, CORN Dense 768 -> 4) compiled by P's own function
(pl_convnext.compile_pl_model: AdamW with no decay on 1-D parameters, weighted CORN, QWK and unweighted CORN loss
as metrics). The loop is P's: generation-based checkpoints (training.Trainer), BEST published by validation QWK,
early stopping and ReduceLROnPlateau on validation QWK, mixed precision, P's epoch order and per-image
augmentation RNG (improved_training_data), P's augmentation (flips, rot90, brightness / contrast).

What differs from P, and only this:
  * the data      -- the EyePACS frame cache (eyepacs_adaptation_data.FrameCache), bound to the split manifest;
  * batch size    -- 16 (P: 2);
  * class weights -- P's formula on the EyePACS TRAINING counts.

Order (§69.2 D): train on the EyePACS training / validation parts -> `pin_checkpoint` pins the selected weights
by SHA-256 (frozen_checkpoint.json, written once) -> `evaluate_heldout_test` scores that pinned model once on
the held-out labelled EyePACS test images -> only then do the APTOS runs (ep_aptos_train) start from the pinned
encoder. The held-out result is read by no other step.

`training_loop` is the one resumable loop of this phase; the adaptation run and the APTOS runs of the EP arms
both use it. It follows arch1_train.train_seed / e1_train.train_seed step for step; those modules belong to
completed experiments and are not modified.
"""
import json
import os
import posixpath

import numpy as np

import arch1_train as at
import eyepacs_adaptation_data as ea

EXPERIMENT = "EyePACSAdaptation"
PREFIX = "p_eyepacs"
ADAPTATION_SEED = 42                       # exactly one adaptation run (record §69.2 A)
ADAPTATION_RUNS = 1
#: P's protocol with the one approved change (batch 16).
PROTOCOL = dict(at.P_PROTOCOL, batch_size=16)
#: The class weights recorded in §69 (P's formula on the EyePACS training counts), to four decimals.
APPROVED_CLASS_WEIGHTS = (0.6450, 2.0906, 1.4230, 3.5138, 3.8780)
EVAL_BATCH = 8
HEARTBEAT_EVERY_BATCHES = 200
FROZEN_NAME = "frozen_checkpoint.json"
HELDOUT_DIR = "heldout_test"
SELECTION = "EyePACS validation QWK only (no APTOS image is read in this phase)"
HELDOUT_ROLE = "held-out evaluation of the frozen adaptation model; in-domain; never used for any selection"
#: config.json keys copied into a pinned-checkpoint record when present.
IDENTITY_KEYS = ("split_manifest_sha256", "frame_cache_fingerprint", "split_sha256", "bundle_fingerprint", "arm",
                 "adapted_encoder_sha256", "checkpoint_selection")


def class_weights(rows):
    """(training counts, weights) from the manifest; the weights must be the ones recorded in §69."""
    counts, weights = ea.class_weights(rows)
    if not np.allclose(weights, APPROVED_CLASS_WEIGHTS, atol=5e-5):
        raise RuntimeError(f"class weights {weights} are not the recorded {APPROVED_CLASS_WEIGHTS}")
    return counts, weights


def environment(repo_dir=None):
    """Commit, library versions, device and time of an invocation (descriptive; never part of a config hash)."""
    import datetime
    import platform
    import sys
    out = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "git_commit": at.git_commit(repo_dir), "python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__}
    try:
        import keras
        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        out.update(tensorflow=tf.__version__, keras=keras.__version__,
                   gpus=[tf.config.experimental.get_device_details(g).get("device_name", g.name) for g in gpus])
    except Exception as error:  # noqa: BLE001
        out["tensorflow_error"] = repr(error)
    return out


def record_invocation(run_dir, record):
    """Appends one invocation to run_metadata.json (a resumed run has several)."""
    path = posixpath.join(run_dir, "run_metadata.json")
    runs = []
    if os.path.exists(path):
        with open(path) as fh:
            runs = json.load(fh)["invocations"]
    runs.append(record)
    with open(path + ".tmp", "w") as fh:
        json.dump({"invocations": runs}, fh, indent=1)
    os.replace(path + ".tmp", path)
    return runs


# --------------------------------------------------------------------------- inputs (P's three-input contract)

def p_inputs(rgb):
    """P's inputs for a batch of RGB frames: stage5_input = [RGB | five zero channels] (P's ChannelAdapter reads
    the first three), stage6_input and reliability zeros (connected to the graph with exactly zero influence)."""
    import joint_training_model as jtm
    rgb = np.asarray(rgb, dtype=np.float32)
    stage5 = np.zeros(rgb.shape[:3] + (8,), dtype=np.float32)
    stage5[..., :3] = rgb
    return {"stage5_input": stage5,
            "stage6_input": np.zeros((len(rgb),) + tuple(jtm.STAGE6_INPUT_SHAPE), dtype=np.float32),
            "reliability": np.zeros((len(rgb), 1), dtype=np.float32)}


def augment_rgb(rgb, rng):
    """P's augmentation of one frame with the given per-image RNG: lfed._augment_spatial, then
    lfed._augment_intensity_rgb (the call order of P, Architecture 1 and E1)."""
    import local_feature_extraction_dataset as lfed
    return lfed._augment_intensity_rgb(lfed._augment_spatial(rgb, rng), rng)


def epoch_order(entries, run_seed, epoch, augment):
    """P's order: a seeded permutation per (run seed, epoch) for training, the given order for validation."""
    import improved_training_data as itd
    entries = [(str(i), int(g)) for i, g in entries]
    return itd.epoch_training_order(entries, run_seed, epoch) if augment else entries


def make_epoch_sequence(frames, entries, epoch, run_seed, batch_size, augment, workers=1):
    """A keras PyDataset for ONE epoch yielding (P's inputs, grades). `frames`: an object with
    `frame(image) -> float32 (S, S, 3)`. Every batch is a pure function of (run seed, epoch, batch index), so the
    result does not depend on `workers`."""
    import keras

    import improved_training_data as itd
    ordered = epoch_order(entries, run_seed, epoch, augment)

    class _EpochSequence(keras.utils.PyDataset):
        def __len__(self):
            return int(np.ceil(len(ordered) / batch_size))

        def __getitem__(self, index):
            rows = ordered[index * batch_size:(index + 1) * batch_size]
            batch = []
            for image, _ in rows:
                rgb = frames.frame(image)
                if augment:
                    rgb = augment_rgb(rgb, itd.per_image_augmentation_rng(run_seed, epoch, image))
                batch.append(rgb)
            return p_inputs(np.stack(batch)), np.asarray([g for _, g in rows], dtype=np.int32)

    return _EpochSequence(workers=int(workers), use_multiprocessing=False, max_queue_size=max(2, 2 * int(workers)))


# --------------------------------------------------------------------------- model + identity

def build_compiled_model(seed, reference_arrays, weights, protocol=PROTOCOL, mixed_precision=True, image_size=None):
    """P for one seed from `reference_arrays` (the encoder it starts from), compiled exactly like P."""
    import pl_convnext as pl
    from training import enable_mixed_precision
    enable_mixed_precision(bool(mixed_precision))
    model = pl.build_pl_model("P", int(seed), reference_arrays, image_size=image_size or pl.IMAGE_SIZE)
    return pl.compile_pl_model(model, list(weights), protocol["learning_rate"], protocol["weight_decay"])


def run_mapping(frames, weights, protocol=PROTOCOL, repo_dir=None):
    """Everything that identifies the adaptation run (written to config.json and hashed into the checkpoints)."""
    import pl_convnext as pl
    return {"experiment": EXPERIMENT, "arch": "P: convnext_tiny (ImageNet) + GAP + LayerNorm + CORN",
            "seed": ADAPTATION_SEED, "adaptation_runs": ADAPTATION_RUNS,
            "data": "EyePACS (Kaggle training set), patient-level split", "split_manifest_sha256": frames.manifest_sha256,
            "frame_cache_fingerprint": frames.fingerprint, "frame_cache_version": ea.CACHE_VERSION,
            "frame_dtype": ea.FRAME_DTYPE, "excluded_blank": list(ea.BLANK_TRAINING_IMAGES),
            "quality_filter": "none", "backbone_weights_sha256": pl.WEIGHTS_SHA256,
            "class_weights": [float(w) for w in weights], "class_weight_source": "EyePACS training counts (P's formula)",
            "optimizer": "AdamW (no decay on 1-D)", "augmentation": "P: lfed._augment_spatial + _augment_intensity_rgb",
            "checkpoint_selection": SELECTION, **protocol, "mixed_precision": True, "ema": at.EMA,
            "git_commit": at.git_commit(repo_dir)}


def run_dir_for(experiments_root, manifest_sha256):
    """The one adaptation run directory."""
    return posixpath.join(experiments_root, EXPERIMENT, f"{PREFIX}_{manifest_sha256[:12]}_seed{ADAPTATION_SEED}")


# --------------------------------------------------------------------------- the resumable loop (P's)

def _heartbeat_callback(run_dir):
    """A long epoch can outlast the run lock: refresh the lock inside the epoch."""
    import keras

    import multiseed_runs as msr

    class LockHeartbeat(keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if batch % HEARTBEAT_EVERY_BATCHES == 0:
                msr.heartbeat_lock(run_dir, owner_id=msr.OWNER_ID)

        def on_test_begin(self, logs=None):
            msr.heartbeat_lock(run_dir, owner_id=msr.OWNER_ID)

    return LockHeartbeat()


def training_loop(run_dir, mapping, build_model, train_sequence, val_sequence, *, experiment_id, dataset_version,
                  protocol, repo_dir, staging_dir, log=print, max_epochs=None, mixed_precision=True,
                  wrap_callbacks=None, initialisation=None):
    """One resumable run, step for step the loop of arch1_train.train_seed / e1_train.train_seed.

    `mapping` identifies the run (a directory holds one configuration); `build_model()` returns the compiled
    model; `train_sequence(epoch)` and `val_sequence()` return the epoch's data; `wrap_callbacks` (E1: the log
    aliases) may put callbacks before the trainer's; `initialisation(model)` returns extra facts recorded in
    initialization.json for a fresh run. Checkpoint selection is the trainer's: BEST = highest validation QWK."""
    import tensorflow as tf

    import multiseed_runs as msr
    from training import CheckpointOptions, Trainer, TrainingConfig, TrainingStateCheckpoint
    from training import checkpointing as ckpt
    if protocol["monitor"] != "val_QWK" or protocol["mode"] != "max":
        raise RuntimeError("the run must be selected by validation QWK")
    epochs = int(max_epochs or protocol["max_epochs"])
    msr.ensure_run_dir(run_dir)
    chash = at.ensure_run_config(run_dir, mapping)
    if msr.read_stop_decision(run_dir) is not None:
        log("  run already stopped")
        return
    sealed = msr._sealed_stop(posixpath.join(run_dir, "checkpoints"), epochs)
    if sealed is not None:
        msr.write_stop_decision(run_dir, sealed[0], sealed[1])
        return
    tf.keras.backend.clear_session()
    model = build_model()
    at.acquire_lock_waiting(run_dir, log)
    try:
        record_invocation(run_dir, dict(environment(repo_dir), config_hash=chash))
        trainer = Trainer(TrainingConfig(
            run_dir=run_dir, epochs=epochs, monitor=protocol["monitor"], mode=protocol["mode"],
            mixed_precision=bool(mixed_precision),
            resume=True, early_stopping_patience=protocol["early_stopping_patience"],
            reduce_lr_patience=protocol["reduce_lr_patience"], reduce_lr_factor=protocol["reduce_lr_factor"],
            min_lr=protocol["min_lr"], precision_check="error", repo_dir=repo_dir,
            checkpoint_options=CheckpointOptions(
                experiment_id=experiment_id, config_hash=chash, dataset_version=dataset_version,
                staging_dir=staging_dir, keep_generations=2, verbose=1)))
        trainer.prepare(model)
        initial_epoch = trainer.resolve_initial_epoch()
        if initial_epoch > 0:
            trainer.restore(model)
            log(f"  resumed at epoch {initial_epoch}")
        else:                                       # a fresh model: record what this run started from
            facts = {"seed": int(mapping["seed"]), "initial_epoch": 0, "weights_sha256": at.weights_sha256(model),
                     "pretrained_backbone_file_sha256": at._pretrained_sha256()}
            facts.update(initialisation(model) if initialisation else {})
            with open(posixpath.join(run_dir, "initialization.json"), "w") as fh:
                json.dump(facts, fh, indent=1)
        if initial_epoch >= epochs:
            msr.write_stop_decision(run_dir, initial_epoch, "epoch_cap")
            return
        early = next(c for c in trainer.callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping))
        state_cb = next(c for c in trainer.callbacks if isinstance(c, TrainingStateCheckpoint))
        callbacks = (wrap_callbacks(trainer.callbacks) if wrap_callbacks else list(trainer.callbacks)) + [_heartbeat_callback(run_dir)]
        val_seq = val_sequence()
        for epoch in range(initial_epoch, epochs):
            msr.heartbeat_lock(run_dir, owner_id=msr.OWNER_ID)
            model.fit(train_sequence(epoch), validation_data=val_seq, epochs=epoch + 1, initial_epoch=epoch,
                      callbacks=callbacks, verbose=1)
            generation = state_cb.last_generation_dir
            if generation is None:
                raise RuntimeError(f"epoch {epoch}: no checkpoint generation written")
            state = ckpt.read_state(generation)
            if state.monitor != "val_QWK" or state.best_metric is None or \
                    (state.extra or {}).get("epoch_logs", {}).get("val_QWK") is None:
                raise RuntimeError(f"epoch {epoch}: the checkpoint state does not hold val_QWK as its monitor")
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


def train(run_dir, frames, reference_arrays, weights, *, repo_dir, staging_dir, protocol=PROTOCOL, log=print,
          max_epochs=None, mixed_precision=True, image_size=None, workers=2):
    """The adaptation run: P from the ImageNet arrays on the EyePACS training part, validated on the EyePACS
    validation part. Only those two parts are arguments; no test set can be passed."""
    if set(getattr(frames, "report", {}).get("frames", ("train", "val"))) != {"train", "val"}:
        raise RuntimeError("the adaptation run takes the training / validation cache only")
    train_entries, val_entries = frames.entries("train"), frames.entries("val")
    return training_loop(
        run_dir, run_mapping(frames, weights, protocol, repo_dir),
        lambda: build_compiled_model(ADAPTATION_SEED, reference_arrays, weights, protocol, mixed_precision, image_size),
        lambda epoch: make_epoch_sequence(frames, train_entries, epoch, ADAPTATION_SEED, protocol["batch_size"], True, workers),
        lambda: make_epoch_sequence(frames, val_entries, 0, ADAPTATION_SEED, protocol["batch_size"], False, workers),
        experiment_id=f"{EXPERIMENT}/seed_{ADAPTATION_SEED}",
        dataset_version=f"eyepacs:{frames.manifest_sha256[:16]}:{frames.fingerprint[:16]}",
        protocol=protocol, repo_dir=repo_dir, staging_dir=staging_dir, log=log, max_epochs=max_epochs,
        mixed_precision=mixed_precision)


# --------------------------------------------------------------------------- evaluation

def predict_logits(model, frames, images, batch_size=EVAL_BATCH):
    """CORN logits (N, 4) float64 for `images` in order, unaugmented, through P's inputs."""
    out = []
    images = [str(i) for i in images]
    for s in range(0, len(images), batch_size):
        rgb = np.stack([frames.frame(i) for i in images[s:s + batch_size]])
        out.append(np.asarray(model.predict_on_batch(p_inputs(rgb)), np.float64))
    return np.concatenate(out, 0)


def evaluate_checkpoints(run_dir, build_model, frames, entries, set_name):
    """BEST and LAST checkpoints of a P-type run on `entries`: per-sample tables (P's schema) and metrics under
    run_dir/metrics."""
    import pandas as pd
    import tensorflow as tf

    import multiseed_runs as msr
    from training import checkpointing as ckpt
    ids, grades = [str(i) for i, _ in entries], [int(g) for _, g in entries]
    out_dir = posixpath.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    best_dir, _ = msr.read_best(run_dir)
    last_dir = ckpt.find_resumable_generation(posixpath.join(run_dir, "checkpoints"), verbose=False)
    for which, gen_dir in (("best", best_dir), ("last", last_dir)):
        if gen_dir is None:
            raise RuntimeError(f"{run_dir}: no {which.upper()} checkpoint")
        tf.keras.backend.clear_session()
        model = build_model()
        path = os.path.join(gen_dir, ckpt.MODEL_WEIGHTS_FILENAME)
        ckpt.load_model_weights_only(model, path)
        metrics, rows = at.metrics_from_logits(ids, grades, predict_logits(model, frames, ids))
        state = ckpt.read_state(gen_dir)
        res = {"set": set_name, "metrics": metrics,
               "checkpoint": {"which": which.upper(), "generation": gen_dir, "weights_sha256": at._sha256(path),
                              "completed_epoch": state.completed_epoch, "best_epoch": state.best_epoch,
                              "best_metric": state.best_metric, "monitor": state.monitor,
                              "learning_rate": state.learning_rate}}
        pd.DataFrame(rows).to_csv(posixpath.join(out_dir, f"per_sample_{which}.csv"), index=False)
        with open(posixpath.join(out_dir, f"metrics_{which}.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        results[which] = res
    return results


def evaluate_run(run_dir, frames, reference_arrays, weights, protocol=PROTOCOL, mixed_precision=True, image_size=None):
    """The adaptation run on the EyePACS validation part (the selection set, never reported as a test)."""
    return evaluate_checkpoints(
        run_dir, lambda: build_compiled_model(ADAPTATION_SEED, reference_arrays, weights, protocol, mixed_precision, image_size),
        frames, frames.entries("val"), "EyePACS adaptation validation (selection set)")


# --------------------------------------------------------------------------- pinning the selected checkpoint

def selected_epoch(history):
    """The epoch a max-QWK selection must have published: the FIRST epoch with the highest validation QWK."""
    values = [(row["val_QWK"], row["epoch"]) for row in history if row.get("val_QWK") is not None]
    if not values:
        raise RuntimeError("the run has no validation QWK")
    top = max(v for v, _ in values)
    return min(e for v, e in values if v == top), float(top)


def pin_checkpoint(run_dir, repo_dir=None):
    """Pins a finished run's BEST weights by SHA-256 in frozen_checkpoint.json. BEST must be the first epoch
    with the highest validation QWK. Written once: a later call must find the same file, otherwise it raises."""
    import multiseed_runs as msr
    from training import checkpointing as ckpt
    stop = msr.read_stop_decision(run_dir)
    if stop is None:
        raise RuntimeError(f"{run_dir}: training has not finished; nothing is pinned")
    best_dir, _ = msr.read_best(run_dir)
    if best_dir is None:
        raise RuntimeError(f"{run_dir}: no BEST checkpoint")
    history = msr.read_history(run_dir)
    completed, top = selected_epoch(history)                   # history rows count completed epochs (1-based)
    state = ckpt.read_state(best_dir)
    if state.best_epoch != completed - 1 or abs(float(state.best_metric) - top) > 1e-12:
        raise RuntimeError(f"{run_dir}: BEST (epoch index {state.best_epoch}, {state.best_metric}) is not the first "
                           f"epoch with the highest validation QWK (completed epoch {completed}, {top})")
    weights_path = os.path.join(best_dir, ckpt.MODEL_WEIGHTS_FILENAME)
    with open(posixpath.join(run_dir, "config.json")) as fh:
        config = json.load(fh)
    record = {"experiment": config["experiment"], "seed": config["seed"],
              "weights": posixpath.join("checkpoints", os.path.basename(best_dir), ckpt.MODEL_WEIGHTS_FILENAME),
              "sha256": at._sha256(weights_path), "best_epoch_index": state.best_epoch, "val_qwk": float(state.best_metric),
              "epochs_run": len(history), "stop": stop, "config_hash": config["config_hash"],
              "training_git_commit": config.get("git_commit"),
              "identity": {k: config[k] for k in IDENTITY_KEYS if k in config},
              "training_configuration": {k: config[k] for k in ("batch_size", "max_epochs", "learning_rate", "weight_decay",
                                                                "early_stopping_patience", "reduce_lr_patience",
                                                                "reduce_lr_factor", "min_lr", "class_weights",
                                                                "mixed_precision") if k in config},
              "pinned": environment(repo_dir)}
    path = posixpath.join(run_dir, FROZEN_NAME)
    if os.path.exists(path):
        with open(path) as fh:
            old = json.load(fh)
        if old["sha256"] != record["sha256"] or old["weights"] != record["weights"]:
            raise RuntimeError(f"{run_dir}: a different checkpoint is already pinned; a pinned checkpoint is never replaced")
        return old
    with open(path + ".tmp", "w") as fh:
        json.dump(record, fh, indent=1)
    os.replace(path + ".tmp", path)
    return record


def read_frozen(run_dir):
    """(absolute weights path, record) of a pinned run; the file must still have the pinned SHA-256."""
    path = posixpath.join(run_dir, FROZEN_NAME)
    if not os.path.exists(path):
        raise RuntimeError(f"{run_dir}: no pinned checkpoint")
    with open(path) as fh:
        record = json.load(fh)
    weights_path = os.path.join(run_dir, *record["weights"].split("/"))
    if not os.path.exists(weights_path) or at._sha256(weights_path) != record["sha256"]:
        raise RuntimeError(f"{run_dir}: the weights file is missing or is not the pinned checkpoint")
    return weights_path, record


def load_frozen(run_dir, reference_arrays, mixed_precision=True, image_size=None):
    """The pinned adaptation model (P's graph, uncompiled, not trainable) and its record."""
    import keras

    import pl_convnext as pl
    from training import checkpointing as ckpt
    from training import enable_mixed_precision
    weights_path, record = read_frozen(run_dir)
    if record["experiment"] != EXPERIMENT:
        raise RuntimeError(f"{run_dir} is not the EyePACS adaptation run")
    keras.backend.clear_session()
    enable_mixed_precision(bool(mixed_precision))               # after clear_session (it resets the policy)
    model = pl.build_pl_model("P", int(record["seed"]), reference_arrays, image_size=image_size or pl.IMAGE_SIZE)
    ckpt.load_model_weights_only(model, weights_path)
    model.trainable = False
    return model, record


def adapted_backbone_arrays(model):
    """The adapted encoder as `reference_arrays` for pl_convnext.build_pl_model / copy_pretrained: the ConvNeXt
    weights in order. The CORN head is deliberately not included: the APTOS runs re-initialise it per seed."""
    import pl_convnext as pl
    return [np.asarray(v.numpy()) for v in model.get_layer(pl.BACKBONE_NAME).weights]


# --------------------------------------------------------------------------- held-out EyePACS test (once)

def heldout_result(run_dir):
    """The recorded held-out evaluation of a run, or None."""
    path = posixpath.join(run_dir, HELDOUT_DIR, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def evaluate_heldout_test(run_dir, test_frames, reference_arrays, *, repo_dir=None, mixed_precision=True, image_size=None,
                          log=print):
    """Scores the PINNED adaptation model once on the held-out labelled EyePACS test images and writes
    heldout_test/{per_sample.csv, metrics.json}. Requires the pinned checkpoint; a recorded result is returned
    as it is and never recomputed. Nothing is selected, tuned or decided from the output."""
    import pandas as pd
    recorded = heldout_result(run_dir)
    _, frozen = read_frozen(run_dir)
    if recorded is not None:
        if recorded["checkpoint_sha256"] != frozen["sha256"]:
            raise RuntimeError(f"{run_dir}: the recorded held-out result belongs to another checkpoint")
        log("  held-out test: already evaluated -- kept as recorded")
        return recorded
    if test_frames.manifest_sha256 != ea.TEST_MANIFEST_SHA256 or set(test_frames.report["frames"]) != {"test"}:
        raise RuntimeError("the frames are not the cache of the pinned held-out test manifest")
    entries = test_frames.entries("test")
    if len(entries) != ea.EXPECTED_TEST_IMAGES:
        raise RuntimeError(f"{len(entries)} test images, expected {ea.EXPECTED_TEST_IMAGES}")
    model, _ = load_frozen(run_dir, reference_arrays, mixed_precision, image_size)
    ids, grades = [i for i, _ in entries], [g for _, g in entries]
    metrics, rows = at.metrics_from_logits(ids, grades, predict_logits(model, test_frames, ids))
    out_dir = posixpath.join(run_dir, HELDOUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(posixpath.join(out_dir, "per_sample.csv"), index=False)
    result = {"set": "EyePACS held-out labelled test images", "role": HELDOUT_ROLE, "images": len(ids),
              "grade_counts": np.bincount(grades, minlength=5).tolist(), "test_manifest_sha256": test_frames.manifest_sha256,
              "frame_cache_fingerprint": test_frames.fingerprint, "checkpoint_sha256": frozen["sha256"],
              "checkpoint_selected_by": SELECTION, "metrics": metrics, "evaluated": environment(repo_dir)}
    path = posixpath.join(out_dir, "metrics.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    os.replace(path + ".tmp", path)
    return result


# --------------------------------------------------------------------------- the one adaptation run, end to end

def state(run_dir):
    """"complete" (pinned, validation result written) | "trained" | "in_progress" | "new"."""
    import multiseed_runs as msr
    if os.path.exists(posixpath.join(run_dir, "result.json")):
        return "complete"
    if msr.read_stop_decision(run_dir) is not None:
        return "trained"
    return "in_progress" if os.path.exists(posixpath.join(run_dir, "config.json")) else "new"


def write_result(run_dir, results, frozen):
    import multiseed_runs as msr
    with open(posixpath.join(run_dir, "config.json")) as fh:
        config = json.load(fh)
    payload = {"experiment": config["experiment"], "seed": config["seed"], "config": config, "frozen": frozen,
               "best": results["best"], "last": results["last"], "history": msr.read_history(run_dir),
               "stop": msr.read_stop_decision(run_dir)}
    path = posixpath.join(run_dir, "result.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    os.replace(path + ".tmp", path)
    return payload


def run_adaptation(experiments_root, frames, reference_arrays, weights, *, repo_dir, staging_root, log=print,
                   train_fn=None, evaluate_fn=None):
    """THE adaptation run: train (or resume), pin the selected checkpoint, score it on the EyePACS validation
    part. One run, one directory; a completed run is returned as recorded. The held-out test evaluation is a
    separate call (`evaluate_heldout_test`) made after this returns."""
    import multiseed_runs as msr
    run_dir = run_dir_for(experiments_root, frames.manifest_sha256)
    log(f"=== EyePACS adaptation (one run, seed {ADAPTATION_SEED}): {state(run_dir)} | {run_dir}")
    if state(run_dir) == "complete":
        read_frozen(run_dir)
        with open(posixpath.join(run_dir, "result.json")) as fh:
            return json.load(fh)
    (train_fn or train)(run_dir, frames, reference_arrays, weights, repo_dir=repo_dir,
                        staging_dir=posixpath.join(staging_root, f"seed_{ADAPTATION_SEED}"), log=log)
    if msr.read_stop_decision(run_dir) is None:
        raise RuntimeError(f"{run_dir}: the run returned without a stop decision -- training is not finished")
    frozen = pin_checkpoint(run_dir, repo_dir)
    evaluated = (evaluate_fn or evaluate_run)(run_dir, frames, reference_arrays, weights)
    if evaluated["best"]["checkpoint"]["weights_sha256"] != frozen["sha256"]:
        raise RuntimeError(f"{run_dir}: the evaluated BEST is not the pinned checkpoint")
    result = write_result(run_dir, evaluated, frozen)
    log(f"=== adaptation DONE: EyePACS validation QWK {frozen['val_qwk']:.4f} at epoch index {frozen['best_epoch_index']} "
        f"| pinned {frozen['sha256']}")
    return result
