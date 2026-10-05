"""Pathology-only grader: data, training, evaluation and the three-seed sequence (Colab, GPU, for training).

Question this experiment answers: can the segmentation outputs (Stage-3 vessel map + Stage-4 v2 lesion maps)
support DR grading on their own, and does their information add anything when combined with the stored RGB
grader P?

What is reused, unchanged: the v2 bundle and its integrity checks (arch1_data.Arch1Bundle), the APTOS split,
the P protocol (arch1_train.P_PROTOCOL: batch 2, AdamW 1e-4 / wd 0.05, weighted CORN with the pre-registered
class weights, <= 50 epochs, early stopping on val QWK, ReduceLROnPlateau, mixed precision), P's epoch order
and per-image augmentation RNG, the generation-based checkpoints (training.Trainer, multiseed_runs), seeds
42 -> 123 -> 2026, one run directory per seed, BEST by validation QWK.

What differs from Architecture 1: the model (pathology_grader_model -- no RGB, no ConvNeXt) and the batches
(only the vessel and pathology tensors are read; the RGB cache is never opened). Augmentation is P's spatial
flips / rot90 with the same per-image RNG; P's intensity jitter is RGB-only and therefore does not apply.

The training loop below mirrors arch1_train.train_seed. It is repeated here rather than parameterised there
because the Architecture-1 module is a completed experiment's code and is not modified.

Nothing here defines a success threshold. The fusion rule is fixed in pathology_grader_fusion.FUSION and is
written into config.json before the first epoch.
"""
import json
import os
import posixpath

import numpy as np

import arch1_data as ad
import arch1_train as at
import pathology_grader_fusion as pf
import pathology_grader_model as pm
import stage34_cache_v2 as cache

SEEDS = at.SEEDS
EXPERIMENT = "PathologyGrader"
PREFIX = "pathgrader"
KINDS = ("stage3_cache_v2", "stage4_cache_v2")                 # the only caches this branch reads
PERMUTATION_SEED = at.PERMUTATION_SEED                          # the Architecture-1 derangement seed


# --------------------------------------------------------------------------- data (vessel + pathology only)

def require_inputs(bundle, image_ids, check_files=True):
    """Completeness of the two generations this branch reads (the RGB generation is not required)."""
    for kind in KINDS:
        cache.assert_complete(bundle.manifests[kind], image_ids, bundle.dirs[kind] if check_files else None,
                              ad._FILENAMES[kind])
    return True


def load_inputs(bundle, image_id):
    """{'vessel': (512,512,1), 'pathology': (512,512,2K)} float32 in [0, 1], through the verified readers
    (per-file SHA, embedded Stage-4 / Stage-3 SHA, generation id, channel order). No RGB file is opened."""
    vessel = cache.read_stage3_vessel(bundle._path("stage3_cache_v2", image_id),
                                      expected_file_sha256=bundle._sha("stage3_cache_v2", image_id))
    maps = cache.read_pathology_npz(
        bundle._path("stage4_cache_v2", image_id), expected_stage4_sha256=bundle.stage4_sha256,
        expected_stage3_sha256=bundle.stage3_sha256, expected_gen_id=bundle.stage4_generation,
        expected_channels=bundle.channels, expected_file_sha256=bundle._sha("stage4_cache_v2", image_id))
    return {"vessel": vessel, "pathology": cache.from_uint8(maps)}


def augment_inputs(sample, rng):
    """P's spatial augmentation (flips / rot90, lfed._augment_spatial) on the aligned stack [vessel | pathology].
    With the same per-image RNG it is the transform Architecture 1 and P applied to that image in that epoch
    (the spatial draws come first). Probabilities are never intensity-jittered."""
    import local_feature_extraction_dataset as lfed
    stack = lfed._augment_spatial(np.concatenate([sample["vessel"], sample["pathology"]], axis=-1), rng)
    return {"vessel": np.ascontiguousarray(stack[..., :1]), "pathology": np.ascontiguousarray(stack[..., 1:])}


def make_epoch_sequence(bundle, entries, epoch, run_seed, batch_size, augment):
    """A keras PyDataset for ONE epoch yielding ({'vessel', 'pathology'}, grade) batches in P's order."""
    import keras

    import improved_training_data as itd
    ordered = ad.epoch_entries([(str(i), int(g)) for i, g in entries], run_seed, epoch, augment)
    require_inputs(bundle, [i for i, _ in ordered])

    class _EpochSequence(keras.utils.PyDataset):
        def __len__(self):
            return int(np.ceil(len(ordered) / batch_size))

        def __getitem__(self, index):
            rows = ordered[index * batch_size:(index + 1) * batch_size]
            samples = []
            for image_id, _ in rows:
                s = load_inputs(bundle, image_id)
                if augment:
                    s = augment_inputs(s, itd.per_image_augmentation_rng(run_seed, epoch, image_id))
                samples.append(s)
            inputs = {k: np.stack([s[k] for s in samples]) for k in pm.INPUT_NAMES}
            return inputs, np.asarray([g for _, g in rows], dtype=np.int32)

    return _EpochSequence()


# --------------------------------------------------------------------------- shuffles

def shuffle_groups(channels):
    """{name: (shuffle vessel?, pathology channel indices)} -- the vessel map, each lesion class (its mean and
    max channels together), every lesion class at once, and everything."""
    channels = tuple(channels)
    classes = cache.classes_from_channels(channels)
    groups = {"vessel": (True, ())}
    for c in classes:
        groups[c] = (False, tuple(i for i, name in enumerate(channels) if name.split(":")[0] == c))
    groups["lesions"] = (False, tuple(range(len(channels))))
    groups["all"] = (True, tuple(range(len(channels))))
    return groups


def derangement(n, seed=PERMUTATION_SEED):
    """A fixed permutation with no fixed point (the rule of arch1_train.predict_logits)."""
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if n < 2 or not np.any(perm == np.arange(n)):
            return perm


def predict_logits(model, bundle, entries, batch_size=8, permute=None):
    """Logits for `entries` in order. `permute`: a key of shuffle_groups -- that part of the input is taken
    from another validation image (one fixed derangement); everything else stays the image's own."""
    ids = [i for i, _ in entries]
    if permute is not None:
        shuffle_vessel, idx = shuffle_groups(bundle.channels)[permute]
        perm = derangement(len(ids))
    out = []
    for s in range(0, len(ids), batch_size):
        rows = range(s, min(s + batch_size, len(ids)))
        samples = [load_inputs(bundle, ids[r]) for r in rows]
        if permute is not None:
            donors = [load_inputs(bundle, ids[perm[r]]) for r in rows]
            mixed = []
            for own, donor in zip(samples, donors):
                pathology = own["pathology"].copy()
                if idx:
                    pathology[..., list(idx)] = donor["pathology"][..., list(idx)]
                mixed.append({"vessel": donor["vessel"] if shuffle_vessel else own["vessel"], "pathology": pathology})
            samples = mixed
        x = {k: np.stack([m[k] for m in samples]) for k in pm.INPUT_NAMES}
        out.append(np.asarray(model.predict_on_batch(x), np.float64))
    return np.concatenate(out, 0)


# --------------------------------------------------------------------------- configuration and model

def run_dir_for(experiments_root, stage4_sha256, seed):
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")
    return posixpath.join(experiments_root, EXPERIMENT, f"{PREFIX}_{stage4_sha256[:12]}_seed{int(seed)}")


def summary_dir_for(experiments_root, stage4_sha256):
    return posixpath.join(experiments_root, EXPERIMENT, f"{PREFIX}_{stage4_sha256[:12]}_3seed_summary")


def run_mapping(bundle, seed, class_weights, protocol=at.P_PROTOCOL, repo_dir=None):
    """Everything that identifies a run (config.json; hashed into the checkpoint config). The fusion rule is
    part of it, so it is on disk before the first epoch."""
    import arch1_model as a1
    return {"experiment": EXPERIMENT,
            "arch": "pathology-only grader: vessel + Stage-4 maps -> Stage-5 encoder -> GAP -> LN -> CORN (no RGB)",
            "seed": int(seed), "split_sha256": bundle.bundle["split_sha256"],
            "population_sha256": bundle.bundle.get("population_sha256"), "bundle_id": bundle.bundle["bundle_id"],
            "bundle_fingerprint": bundle.fingerprint, "stage4_sha256": bundle.stage4_sha256,
            "stage4_generation": bundle.stage4_generation, "stage3_sha256": bundle.stage3_sha256,
            "input_channels": list(pm.input_channel_names(bundle.channels)), "inputs": list(pm.INPUT_NAMES),
            "rgb_input": False, "encoder_dims": list(a1.PRIOR_DIMS), "feature_dim": pm.FEATURE_DIM,
            "class_weights": [float(w) for w in class_weights], "optimizer": "AdamW (no decay on 1-D)",
            "augmentation": "P spatial only: lfed._augment_spatial on [vessel | pathology], P's per-image RNG",
            **protocol, "mixed_precision": True, "ema": "none", "shuffle_seed": PERMUTATION_SEED,
            "fusion": pf.FUSION, "git_commit": at.git_commit(repo_dir)}


def build_compiled_model(channels, seed, class_weights, protocol=at.P_PROTOCOL, mixed_precision=True,
                         image_size=pm.IMAGE_SIZE):
    """The pathology grader compiled exactly like P (pl_convnext.compile_pl_model: weighted CORN, AdamW with
    no decay on 1-D parameters, CORN QWK and unweighted CORN loss as metrics)."""
    import pl_convnext as pl
    from training import enable_mixed_precision
    enable_mixed_precision(bool(mixed_precision))
    model = pm.build_pathology_grader(channels, seed, image_size=image_size)
    return pl.compile_pl_model(model, list(class_weights), protocol["learning_rate"], protocol["weight_decay"])


# --------------------------------------------------------------------------- training (P's loop)

def train_seed(run_dir, bundle, seed, class_weights, *, repo_dir, staging_dir, protocol=at.P_PROTOCOL, log=print,
               max_epochs=None, grade_of=None, mixed_precision=True):
    """One resumable run, structured as arch1_train.train_seed (see the module docstring)."""
    import tensorflow as tf

    import multiseed_runs as msr
    from training import CheckpointOptions, Trainer, TrainingConfig, TrainingStateCheckpoint
    from training import checkpointing as ckpt
    epochs = int(max_epochs or protocol["max_epochs"])
    msr.ensure_run_dir(run_dir)
    chash = at.ensure_run_config(run_dir, run_mapping(bundle, seed, class_weights, protocol, repo_dir))
    if msr.read_stop_decision(run_dir) is not None:
        log("  run already stopped")
        return
    sealed = msr._sealed_stop(posixpath.join(run_dir, "checkpoints"), epochs)
    if sealed is not None:
        msr.write_stop_decision(run_dir, sealed[0], sealed[1])
        return
    tf.keras.backend.clear_session()
    model = build_compiled_model(bundle.channels, seed, class_weights, protocol, mixed_precision)
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
                           "pretrained_weights": None}, fh, indent=1)
        if initial_epoch >= epochs:
            msr.write_stop_decision(run_dir, initial_epoch, "epoch_cap")
            return
        early = next(c for c in trainer.callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping))
        state_cb = next(c for c in trainer.callbacks if isinstance(c, TrainingStateCheckpoint))
        train_entries = list(zip(bundle.train_ids, at._grades(bundle, bundle.train_ids, grade_of)))
        val_entries = list(zip(bundle.val_ids, at._grades(bundle, bundle.val_ids, grade_of)))
        val_seq = make_epoch_sequence(bundle, val_entries, 0, seed, protocol["batch_size"], augment=False)
        for epoch in range(initial_epoch, epochs):
            msr.heartbeat_lock(run_dir, owner_id=msr.OWNER_ID)
            train_seq = make_epoch_sequence(bundle, train_entries, epoch, seed, protocol["batch_size"], augment=True)
            model.fit(train_seq, validation_data=val_seq, epochs=epoch + 1, initial_epoch=epoch,
                      callbacks=trainer.callbacks, verbose=1)
            generation = state_cb.last_generation_dir
            if generation is None:
                raise RuntimeError(f"epoch {epoch}: no checkpoint generation written")
            state = ckpt.read_state(generation)
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


# --------------------------------------------------------------------------- evaluation

def evaluate_run(run_dir, bundle, seed, class_weights, protocol=at.P_PROTOCOL, grade_of=None, mixed_precision=True):
    """BEST and LAST checkpoints -> per-sample tables (with the cumulative CORN probabilities `p_gt_k` used
    by the fusion) and metrics. For BEST, every shuffle of shuffle_groups is evaluated and its per-sample
    table is saved as well. Written under run_dir/metrics."""
    import pandas as pd
    import tensorflow as tf

    import multiseed_runs as msr
    from training import checkpointing as ckpt
    ids = list(bundle.val_ids)
    grades = at._grades(bundle, ids, grade_of)
    val_entries = list(zip(ids, grades))
    out_dir = posixpath.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    best_dir, _ = msr.read_best(run_dir)
    last_dir = ckpt.find_resumable_generation(posixpath.join(run_dir, "checkpoints"), verbose=False)
    for which, gen_dir in (("best", best_dir), ("last", last_dir)):
        if gen_dir is None:
            raise RuntimeError(f"{run_dir}: no {which.upper()} checkpoint")
        tf.keras.backend.clear_session()
        model = build_compiled_model(bundle.channels, seed, class_weights, protocol, mixed_precision)
        weights = os.path.join(gen_dir, ckpt.MODEL_WEIGHTS_FILENAME)
        ckpt.load_model_weights_only(model, weights)
        metrics, rows = at.metrics_from_logits(ids, grades, predict_logits(model, bundle, val_entries))
        pd.DataFrame(rows).to_csv(posixpath.join(out_dir, f"per_sample_{which}.csv"), index=False)
        state = ckpt.read_state(gen_dir)
        res = {"metrics": metrics,
               "checkpoint": {"which": which.upper(), "generation": gen_dir, "weights_sha256": at._sha256(weights),
                              "completed_epoch": state.completed_epoch, "best_epoch": state.best_epoch,
                              "best_metric": state.best_metric, "monitor": state.monitor,
                              "learning_rate": state.learning_rate}}
        if which == "best":
            res["shuffle"] = {}
            for name in shuffle_groups(bundle.channels):
                m, shuffled_rows = at.metrics_from_logits(ids, grades, predict_logits(model, bundle, val_entries, permute=name))
                pd.DataFrame(shuffled_rows).to_csv(posixpath.join(out_dir, f"per_sample_best_shuffle_{name}.csv"), index=False)
                res["shuffle"][name] = {"qwk": m["qwk"], "dqwk": m["qwk"] - metrics["qwk"]}
        with open(posixpath.join(out_dir, f"metrics_{which}.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        results[which] = res
    return results


def load_p_tables(experiments_root):
    """The stored P runs' BEST per-sample tables for the three seeds (frozen; never retrained)."""
    return {s: read_table(pf.p_prediction_path(experiments_root, s)) for s in SEEDS}


def read_table(path):
    import arch1_posthoc as ph
    return ph._read_table(path)


def seed_result(run_dir, experiments_root, results, sources=None, p_tables=None, indices=None):
    """The fusion / control / shuffle report for one finished seed, written as result.md and result.json (last,
    atomically -- its presence marks the seed complete). Reads the per-sample tables evaluate_run wrote."""
    import arch1_posthoc as ph

    import multiseed_runs as msr
    config = at._read_json(posixpath.join(run_dir, "config.json"))
    seed = config["seed"]
    metrics_dir = posixpath.join(run_dir, "metrics")
    pathology = read_table(posixpath.join(metrics_dir, "per_sample_best.csv"))
    shuffled = {name: read_table(posixpath.join(metrics_dir, f"per_sample_best_shuffle_{name}.csv"))
                for name in results["best"]["shuffle"]}
    p_sources = None if p_tables else {str(s): pf.p_prediction_path(experiments_root, s) for s in SEEDS}
    p_tables = p_tables or load_p_tables(experiments_root)
    indices = indices if indices is not None else ph.bootstrap_indices(pathology["grade"])
    report = pf.seed_report(seed, pathology, p_tables, shuffled, indices)
    report.update(
        experiment=EXPERIMENT, status="RECORDED", fusion=pf.FUSION,
        identity={k: config.get(k) for k in ("git_commit", "config_hash", "stage3_sha256", "stage4_sha256",
                                              "stage4_generation", "bundle_id", "bundle_fingerprint", "split_sha256",
                                              "population_sha256", "input_channels", "ema")},
        checkpoints={"best": results["best"]["checkpoint"], "last": results["last"]["checkpoint"]},
        last_checkpoint_metrics={k: results["last"]["metrics"][k] for k in ("qwk", "grade3_recall", "false_urgent_rate")},
        p_sources=p_sources,
        stop=msr.read_stop_decision(run_dir), history=msr.read_history(run_dir), sources=sources,
        initialization=at._read_json(posixpath.join(run_dir, "initialization.json")))
    m = report["models"]
    report["headline"] = (f"pathology QWK {m['pathology']['qwk']:.4f} | P {m['p']['qwk']:.4f} | fused {m['fused']['qwk']:.4f} | "
                          f"P+P control {m['control']['qwk']:.4f} | pathology lesion-shuffle dQWK "
                          f"{report['pathology_shuffle']['lesions']['dqwk']:+.4f}, vessel {report['pathology_shuffle']['vessel']['dqwk']:+.4f}")
    with open(posixpath.join(run_dir, "result.md"), "w", encoding="utf-8") as fh:
        fh.write(f"# Pathology grader -- seed {seed}\n\n{report['headline']}\n\nFusion: {pf.FUSION['rule']} "
                 f"(weights {pf.FUSION['weights']}, not tuned). Control seed: P-{report['control_seed']}.\n")
    at._write_json_last(posixpath.join(run_dir, "result.json"), ph._jsonable(report))
    return report


# --------------------------------------------------------------------------- three-seed sequence

def seed_state(run_dir):
    """"complete" (result written) | "trained" (stop decided, not yet evaluated) | "in_progress" | "new"."""
    import multiseed_runs as msr
    if os.path.exists(posixpath.join(run_dir, "result.json")):
        return "complete"
    if msr.read_stop_decision(run_dir) is not None:
        return "trained"
    return "in_progress" if os.path.exists(posixpath.join(run_dir, "config.json")) else "new"


def run_seed(run_dir, bundle, seed, class_weights, *, experiments_root, repo_dir, staging_dir, sources=None,
             log=print, train_fn=None, evaluate_fn=None, p_tables=None):
    """One seed end to end in its own directory: train (or resume THIS seed's checkpoint), evaluate BEST and
    LAST with every shuffle, fuse with the stored P runs, write the result. A completed seed is returned as
    recorded. `train_fn` / `evaluate_fn` default to train_seed / evaluate_run (tests substitute them)."""
    import multiseed_runs as msr
    if seed_state(run_dir) == "complete":
        report = at._read_json(posixpath.join(run_dir, "result.json"))
        ident = report["identity"]
        if (report["seed"] != int(seed) or ident["stage4_sha256"] != bundle.stage4_sha256
                or ident["bundle_fingerprint"] != bundle.fingerprint):
            raise RuntimeError(f"{run_dir}: the recorded result belongs to another seed, model or bundle")
        log(f"  seed {seed}: already complete -- kept as recorded")
        return report
    (train_fn or train_seed)(run_dir, bundle, seed, class_weights, repo_dir=repo_dir, staging_dir=staging_dir, log=log)
    if msr.read_stop_decision(run_dir) is None:
        raise RuntimeError(f"{run_dir}: seed {seed} returned without a stop decision -- training is not finished")
    results = (evaluate_fn or evaluate_run)(run_dir, bundle, seed, class_weights)
    return seed_result(run_dir, experiments_root, results, sources, p_tables)


def write_summary(summary_dir, reports):
    """summary.md, then summary.json (last, atomically). Only with all three seeds."""
    import arch1_posthoc as ph
    summary = pf.summarise(reports)
    os.makedirs(summary_dir, exist_ok=True)
    with open(posixpath.join(summary_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write(pf.summary_markdown(summary, reports))
    at._write_json_last(posixpath.join(summary_dir, "summary.json"), ph._jsonable(summary))
    return summary


def run_sequence(experiments_root, bundle, class_weights, *, repo_dir, staging_root, sources=None, log=print,
                 train_fn=None, evaluate_fn=None, p_tables=None):
    """Seeds 42 -> 123 -> 2026, one after another, each model built fresh from its own seed. Before anything is
    trained the stored P predictions are checked to be the same validation images in the same order. Any
    exception stops the sequence: no later seed starts and no summary is written. Rerunning resumes the
    unfinished seed from its own checkpoint. The summary is written only when all three seeds are complete."""
    import gc
    p_tables = p_tables or load_p_tables(experiments_root)
    for s in SEEDS:
        if list(p_tables[s]["ids"]) != [str(i) for i in bundle.val_ids]:
            raise RuntimeError(f"stored P-{s} predictions are not the bundle's validation images in the same order")
    reports = {}
    for n, seed in enumerate(SEEDS, start=1):
        run_dir = run_dir_for(experiments_root, bundle.stage4_sha256, seed)
        log(f"=== [{n}/{len(SEEDS)}] seed {seed}: {seed_state(run_dir)} | {run_dir}")
        reports[seed] = run_seed(run_dir, bundle, seed, class_weights, experiments_root=experiments_root,
                                 repo_dir=repo_dir, staging_dir=posixpath.join(staging_root, f"seed_{seed}"),
                                 sources=sources, log=log, train_fn=train_fn, evaluate_fn=evaluate_fn, p_tables=p_tables)
        log(f"=== [{n}/{len(SEEDS)}] seed {seed} DONE | {reports[seed]['headline']}")
        gc.collect()
    summary_dir = summary_dir_for(experiments_root, bundle.stage4_sha256)
    summary = write_summary(summary_dir, reports)
    log(f"=== all {len(SEEDS)} seeds complete | {summary_dir}")
    return reports, summary


# --------------------------------------------------------------------------- Colab staging (two caches only)

def stage_inputs(bundle_id, drive_roots, local_roots, log=print, stage_files=None):
    """Copies the Stage-3 and Stage-4 generations of a bundle from Drive to local disk with per-file SHA checks
    (colab stage4_v2_setup.stage_files), plus every manifest. The RGB generation's FILES are not copied: this
    branch never reads them. Manifests are written last."""
    if stage_files is None:
        import stage4_v2_setup
        stage_files = stage4_v2_setup.stage_files
    names = {"stage2_rgb_v2": cache.rgb_filename, "stage3_cache_v2": cache.vessel_filename,
             "stage4_cache_v2": cache.pathology_filename}
    bundle_manifest = os.path.join(drive_roots["bundle_v2"], bundle_id, cache.MANIFEST_NAMES["bundle_v2"])
    with open(bundle_manifest, "rb") as fh:
        bundle_payload = fh.read()
    bundle = json.loads(bundle_payload.decode("utf-8"))
    gens = {"stage2_rgb_v2": bundle["stage2_generation"], "stage3_cache_v2": bundle["stage3_generation"],
            "stage4_cache_v2": bundle["stage4_generation"]}
    out, manifests = {}, {}
    for kind, gen in gens.items():
        src_gen, dst_gen = os.path.join(drive_roots[kind], gen), os.path.join(local_roots[kind], gen)
        with open(os.path.join(src_gen, cache.MANIFEST_NAMES[kind]), "rb") as fh:
            manifests[kind] = (os.path.join(dst_gen, cache.MANIFEST_NAMES[kind]), fh.read())
        if kind not in KINDS:
            continue
        files = json.loads(manifests[kind][1].decode("utf-8"))["files"]
        sub = cache.DATA_SUBDIRS[kind]
        pairs = [(os.path.join(src_gen, sub, names[kind](i)), os.path.join(dst_gen, sub, names[kind](i)), sha)
                 for i, sha in sorted(files.items())]
        out[kind] = stage_files(pairs, log=log, label=kind)
    targets = list(manifests.values()) + [(os.path.join(local_roots["bundle_v2"], bundle_id,
                                                        cache.MANIFEST_NAMES["bundle_v2"]), bundle_payload)]
    for path, payload in targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as fh:
            fh.write(payload)
        os.replace(path + ".tmp", path)
    log(f"  pathology inputs staged (vessel + lesion maps only): {out}")
    return out
