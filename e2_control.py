"""E2 -- the shuffled-target control of the E1 multi-task grader (research record §62 decision tree, §63.2).

Question: does E1's representation change need IMAGE-ALIGNED lesion supervision, or does an auxiliary spatial
loss of the same form produce it by itself?

E2 is E1 in every respect -- model (e1_model), loss and lambda = 1.0 (e1_train.lesion_loss, LAMBDA), optimiser,
P protocol, augmentation, seeds, bias prior, BEST by validation QWK -- except for ONE thing: during training the
lesion target of an image is the cached Stage-4 map of ANOTHER training image, its partner in a fixed
derangement of the training set (seed 20261001; one derangement, the same for all three model seeds). The
shuffled targets have exactly the same overall distribution as E1's (every training map is used once) but never
belong to the image they are paired with. Validation uses the true, image-aligned targets, as E1 does.

How the partner's map is used: it replaces the image's own map BEFORE the augmentation, so E1's augmentation
code runs unchanged on the stack [image RGB | partner map] (the image's geometric transform, RGB-only
intensity). The image's own cached map is never opened by the training sequence.

The training loop mirrors e1_train.train_seed (which mirrors arch1_train.train_seed). It is repeated here because
the E1 module is a completed experiment's code and is not modified. E2 is a control: it is not selected against
E1, replaces nothing and tunes nothing.

After training, the same fresh frozen-encoder lesion probe as §63.2 (e1_probe) is run on E2's BEST encoder and
compared with the stored P and E1 probes on the same validation images and the same bootstrap resamples.
"""
import hashlib
import json
import os
import posixpath

import numpy as np

import arch1_data as ad
import arch1_train as at
import e1_data as ed
import e1_model as em
import e1_probe as ep
import e1_train as et

SEEDS = et.SEEDS
EXPERIMENT = "E2ShuffledTargets"
PREFIX = "e2"
DERANGEMENT_SEED = 20261001                       # the pre-registered seed (record §62)
MODELS = ("p", "e1", "e2")


# --------------------------------------------------------------------------- the derangement

def derangement(train_ids, seed=DERANGEMENT_SEED):
    """{image id: partner id} over the training images, in the bundle's order: a permutation with no fixed
    point, drawn as the project's other derangements are (arch1_train.predict_logits): seeded permutations
    until none maps an index to itself. One mapping; it does not depend on the model seed."""
    ids = [str(i) for i in train_ids]
    if len(set(ids)) != len(ids) or len(ids) < 2:
        raise ValueError("the training ids must be unique and at least two")
    rng = np.random.default_rng(int(seed))
    while True:
        perm = rng.permutation(len(ids))
        if not np.any(perm == np.arange(len(ids))):
            break
    return {ids[n]: ids[int(perm[n])] for n in range(len(ids))}


def verify_derangement(partner, train_ids, val_ids=()):
    """Raises unless `partner` is a true derangement of the training images. Returns its report."""
    ids = [str(i) for i in train_ids]
    if list(partner) != ids:
        raise RuntimeError("the mapping does not cover the training images in order")
    values = list(partner.values())
    fixed = [i for i in ids if partner[i] == i]
    if fixed:
        raise RuntimeError(f"{len(fixed)} images map to themselves (e.g. {fixed[:3]})")
    if sorted(values) != sorted(ids):
        raise RuntimeError("the partners are not a permutation of the training images (each exactly once)")
    if set(values) & {str(i) for i in val_ids}:
        raise RuntimeError("a validation image is used as a partner")
    return {"seed": DERANGEMENT_SEED, "images": len(ids), "fixed_points": 0, "is_permutation": True,
            "sha256": hashlib.sha256(json.dumps(partner, sort_keys=True).encode()).hexdigest(),
            "first_pairs": [[i, partner[i]] for i in ids[:3]]}


# --------------------------------------------------------------------------- data

def make_epoch_sequence(bundle, entries, epoch, run_seed, batch_size, partner):
    """The E2 TRAINING sequence: ({'rgb'}, (grades, lesion targets)) in P's order with P's augmentation, where
    the lesion target comes from the partner's cached map. Only the image's RGB and the PARTNER's Stage-4 file
    are read; the image's own Stage-4 file is never opened. (Validation uses e1_data.make_epoch_sequence.)"""
    import keras

    import improved_training_data as itd
    import stage34_cache_v2 as cache
    ordered = ad.epoch_entries([(str(i), int(g)) for i, g in entries], run_seed, epoch, True)
    ids = [i for i, _ in ordered]
    ed.require_inputs(bundle, ids)
    missing = [i for i in ids if i not in partner]
    if missing:
        raise RuntimeError(f"{len(missing)} training images have no partner")
    channels = bundle.channels

    class _EpochSequence(keras.utils.PyDataset):
        def __len__(self):
            return int(np.ceil(len(ordered) / batch_size))

        def __getitem__(self, index):
            rows = ordered[index * batch_size:(index + 1) * batch_size]
            rgb, targets = [], []
            for image_id, _ in rows:
                donor = partner[image_id]
                if donor == image_id:
                    raise RuntimeError(f"{image_id} is its own partner")
                sample = {"rgb": cache.read_stage2_rgb(bundle._path("stage2_rgb_v2", image_id),
                                                       expected_file_sha256=bundle._sha("stage2_rgb_v2", image_id)),
                          "pathology": cache.from_uint8(ed.load_pathology(bundle, donor))}
                sample = ed.augment_inputs(sample, itd.per_image_augmentation_rng(run_seed, epoch, image_id))
                rgb.append(sample["rgb"])
                targets.append(ed.pool_targets(sample["pathology"], channels))
            return ({"rgb": np.stack(rgb)},
                    (np.asarray([g for _, g in rows], dtype=np.int32), np.stack(targets).astype(np.float32)))

    return _EpochSequence()


def run_mapping(bundle, seed, class_weights, lesion_prior, derangement_report, protocol=at.P_PROTOCOL, repo_dir=None):
    """E1's run identity with the experiment name and the shuffled-target definition."""
    mapping = et.run_mapping(bundle, seed, class_weights, lesion_prior, protocol, repo_dir)
    mapping.update(experiment=EXPERIMENT,
                   arch="E1 unchanged: convnext_tiny + CORN + 1x1 lesion head on the final feature map",
                   control_of="E1MultiTask",
                   training_lesion_targets="shuffled: the cached Stage-4 map of the image's partner in a fixed derangement",
                   validation_lesion_targets="true, image-aligned (as E1)",
                   derangement_seed=DERANGEMENT_SEED, derangement_sha256=derangement_report["sha256"],
                   derangement_images=derangement_report["images"], derangement_fixed_points=0,
                   derangement_shared_by_all_model_seeds=True)
    return mapping


# --------------------------------------------------------------------------- training (E1's loop, shuffled targets)

def train_seed(run_dir, bundle, seed, lesion_prior, reference, class_weights, partner, *, repo_dir, staging_dir,
               protocol=at.P_PROTOCOL, log=print, max_epochs=None, grade_of=None, mixed_precision=True):
    """One resumable E2 run: e1_train.train_seed with the training sequence above and E2's run identity."""
    import tensorflow as tf

    import multiseed_runs as msr
    from training import CheckpointOptions, Trainer, TrainingConfig, TrainingStateCheckpoint
    from training import checkpointing as ckpt
    report = verify_derangement(partner, bundle.train_ids, bundle.val_ids)
    epochs = int(max_epochs or protocol["max_epochs"])
    msr.ensure_run_dir(run_dir)
    chash = at.ensure_run_config(run_dir, run_mapping(bundle, seed, class_weights, lesion_prior, report, protocol, repo_dir))
    if msr.read_stop_decision(run_dir) is not None:
        log("  run already stopped")
        return
    sealed = msr._sealed_stop(posixpath.join(run_dir, "checkpoints"), epochs)
    if sealed is not None:
        msr.write_stop_decision(run_dir, sealed[0], sealed[1])
        return
    tf.keras.backend.clear_session()
    model = et.build_compiled_model(seed, lesion_prior, reference, class_weights, protocol, mixed_precision)
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
        else:
            with open(posixpath.join(run_dir, "initialization.json"), "w") as fh:
                json.dump({"seed": int(seed), "initial_epoch": 0, "weights_sha256": at.weights_sha256(model),
                           "pretrained_backbone_file_sha256": at._pretrained_sha256()}, fh, indent=1)
        if initial_epoch >= epochs:
            msr.write_stop_decision(run_dir, initial_epoch, "epoch_cap")
            return
        early = next(c for c in trainer.callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping))
        state_cb = next(c for c in trainer.callbacks if isinstance(c, TrainingStateCheckpoint))
        callbacks = et.with_aliases(trainer.callbacks)
        train_entries = list(zip(bundle.train_ids, at._grades(bundle, bundle.train_ids, grade_of)))
        val_entries = list(zip(bundle.val_ids, at._grades(bundle, bundle.val_ids, grade_of)))
        val_seq = ed.make_epoch_sequence(bundle, val_entries, 0, seed, protocol["batch_size"], augment=False)   # aligned
        for epoch in range(initial_epoch, epochs):
            msr.heartbeat_lock(run_dir, owner_id=msr.OWNER_ID)
            train_seq = make_epoch_sequence(bundle, train_entries, epoch, seed, protocol["batch_size"], partner)
            model.fit(train_seq, validation_data=val_seq, epochs=epoch + 1, initial_epoch=epoch,
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


def run_dir_for(experiments_root, stage4_sha256, seed):
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")
    return posixpath.join(experiments_root, EXPERIMENT, f"{PREFIX}_{stage4_sha256[:12]}_seed{int(seed)}")


seed_state = et.seed_state


def write_result(run_dir, results, gates, derangement_report):
    import multiseed_runs as msr
    with open(posixpath.join(run_dir, "config.json")) as fh:
        config = json.load(fh)
    payload = {"experiment": EXPERIMENT, "seed": config["seed"], "config": config, "derangement": derangement_report,
               "best": results["best"], "last": results["last"], "history": msr.read_history(run_dir),
               "stop": msr.read_stop_decision(run_dir), "gates": gates,
               "note": "control run: the lesion head's validation numbers are against the TRUE aligned targets"}
    path = posixpath.join(run_dir, "result.json")
    with open(path + ".tmp", "w") as fh:
        json.dump(payload, fh, indent=1, default=float)
    os.replace(path + ".tmp", path)
    return payload


def run_sequence(experiments_root, bundle, lesion_prior, reference, class_weights, *, repo_dir, staging_root, gates,
                 log=print, train_fn=None, evaluate_fn=None):
    """Seeds 42 -> 123 -> 2026 with ONE derangement. E1's gates (same model, same targets, same bookkeeping)
    must be recorded as passed. Any exception stops the sequence. A completed seed is kept as recorded."""
    import gc

    import multiseed_runs as msr
    missing = [g for g in ("p_parity", "log_alias", "targets") if not (gates or {}).get(g, {}).get("PASS")]
    if missing:
        raise RuntimeError(f"E1's pre-training gates are not recorded as passed: {missing}")
    partner = derangement(bundle.train_ids)
    report = verify_derangement(partner, bundle.train_ids, bundle.val_ids)
    log(f"derangement: seed {report['seed']} | {report['images']} images | fixed points {report['fixed_points']} | "
        f"sha256 {report['sha256'][:16]}")
    results = {}
    for n, seed in enumerate(SEEDS, start=1):
        run_dir = run_dir_for(experiments_root, bundle.stage4_sha256, seed)
        log(f"=== [{n}/{len(SEEDS)}] E2 seed {seed}: {seed_state(run_dir)} | {run_dir}")
        if seed_state(run_dir) == "complete":
            with open(posixpath.join(run_dir, "result.json")) as fh:
                results[seed] = json.load(fh)
            if results[seed]["derangement"]["sha256"] != report["sha256"]:
                raise RuntimeError(f"{run_dir}: recorded with another derangement")
            continue
        (train_fn or train_seed)(run_dir, bundle, seed, lesion_prior, reference, class_weights, partner, repo_dir=repo_dir,
                                 staging_dir=posixpath.join(staging_root, f"seed_{seed}"), log=log)
        if msr.read_stop_decision(run_dir) is None:
            raise RuntimeError(f"{run_dir}: seed {seed} returned without a stop decision -- training is not finished")
        evaluated = (evaluate_fn or et.evaluate_run)(run_dir, bundle, seed, lesion_prior, reference, class_weights)
        results[seed] = write_result(run_dir, evaluated, gates, report)
        best = results[seed]["best"]
        log(f"=== [{n}/{len(SEEDS)}] E2 seed {seed} DONE: QWK {best['metrics']['qwk']:.4f} | trained lesion head vs the TRUE "
            f"targets, mean cell AUROC {best['lesion_head_validation']['mean_cell_auroc']:.4f}")
        gc.collect()
    return results


# --------------------------------------------------------------------------- the same fresh probe, on E2

def run_probe(bundle, e2_run_dirs, convnext_weights_path, lesion_prior, out_dir, e1_probe_dir, *, seeds=SEEDS,
              work_dir="/content/e2_probe_features", log=print):
    """The §63.2 probe (e1_probe: same fresh 1x1 probe, protocol, seed, batch order, targets) on the frozen
    encoder of each E2 BEST checkpoint. E2's trained lesion head and grading head are not used. The validation
    targets must be byte-identical to those the P and E1 probes were scored on. Resumable."""
    import gc

    import keras

    import pl_convnext as pl
    train_ids, val_ids = [str(i) for i in bundle.train_ids], [str(i) for i in bundle.val_ids]
    os.makedirs(out_dir, exist_ok=True)
    previous = keras.mixed_precision.global_policy().name
    with open(os.path.join(e1_probe_dir, "summary.json")) as fh:
        e1_summary = json.load(fh)
    if e1_summary["probe"] != json.loads(json.dumps(ep.PROBE)) or e1_summary["bundle_fingerprint"] != bundle.fingerprint:
        raise RuntimeError("the stored P / E1 probe was run with another protocol or bundle")
    if [float(p) for p in lesion_prior] != e1_summary["lesion_prior"]:
        raise RuntimeError("the lesion prior differs from the one the P / E1 probes used")
    identity = {"probe": ep.PROBE, "lesion_prior": [float(p) for p in lesion_prior], "train_images": len(train_ids),
                "val_images": len(val_ids), "bundle_fingerprint": bundle.fingerprint}
    summary = dict(identity, seeds={})
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            old = json.load(fh)
        if all(old.get(k) == v for k, v in json.loads(json.dumps(identity)).items()):
            summary["seeds"] = {int(k): v for k, v in old.get("seeds", {}).items()}
    try:
        keras.mixed_precision.set_global_policy("float32")
        _, reference_arrays = pl.load_reference(convnext_weights_path)
        log("  targets ...")
        train_targets, val_targets = ep.targets_for(bundle, train_ids), ep.targets_for(bundle, val_ids)
        with np.load(os.path.join(e1_probe_dir, "validation_targets.npz")) as data:
            if [str(i) for i in data["image_ids"]] != val_ids or not np.array_equal(data["targets"], val_targets):
                raise RuntimeError("the validation targets differ from those of the P / E1 probes")
        np.savez_compressed(os.path.join(out_dir, "validation_targets.npz"), targets=val_targets, image_ids=np.asarray(val_ids))
        for seed in seeds:
            seed = int(seed)
            path, sha = ep.e1_best_weights(e2_run_dirs[seed])
            saved = os.path.join(out_dir, f"probe_e2_seed{seed}.npz")
            if summary["seeds"].get(seed, {}).get("e2", {}).get("weights_sha256") == sha and os.path.exists(saved):
                log(f"  seed {seed} / e2: already done -- kept")
                continue
            log(f"  seed {seed} / e2: frozen features  {ep._ram()}")
            features, full = ep.build_frozen_encoder("e1", seed, path, lesion_prior, reference_arrays)   # E1's graph
            files = [os.path.join(work_dir, f"e2_seed{seed}_{part}.npy") for part in ("train", "val")]
            train_f, _ = ep.extract(bundle, features, full, train_ids, files[0], log=log)
            val_f, val_grading = ep.extract(bundle, features, full, val_ids, files[1], log=log)
            del features, full
            keras.backend.clear_session()
            gc.collect()
            val_logits, curve, final = ep.train_probe(seed, lesion_prior, train_f, train_targets, val_f, val_targets, log=log)
            np.savez_compressed(saved, val_logits=val_logits, val_grading_logits=val_grading, image_ids=np.asarray(val_ids))
            summary["seeds"][seed] = {"e2": {"weights_sha256": sha, "curve": curve, "final": final,
                                             "feature_abs_mean": float(np.mean([np.abs(val_f[i].astype(np.float32)).mean()
                                                                                for i in range(len(val_f))]))}}
            log(f"  seed {seed} / e2: val mean cell AUROC {final['val_scores']['mean']:.4f} "
                f"{ {k: round(v, 4) for k, v in final['val_scores'].items() if k != 'mean'} }")
            del train_f, val_f
            keras.backend.clear_session()
            gc.collect()
            for f in files:
                try:
                    os.remove(f)
                except OSError:
                    pass
            with open(summary_path + ".tmp", "w") as fh:
                json.dump(summary, fh, indent=1, default=float)
            os.replace(summary_path + ".tmp", summary_path)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    return summary


# --------------------------------------------------------------------------- comparison P / E1 / E2

def pair_counts(positive, score):
    """U[i, j] = number of (positive cell of image i, negative cell of image j) pairs in which the positive
    cell scores higher (ties count one half). With image weights w (how often each image is in a resample),
    the cell-level AUROC of the resample is  w.U.w / ((w.n_pos) (w.n_neg))  -- exactly the AUROC of the
    concatenated cells, without re-sorting them for every resample."""
    n = positive.shape[0]
    pos = positive.reshape(n, -1)
    s = np.asarray(score, np.float64).reshape(n, -1)
    owner = np.repeat(np.arange(n), pos.sum(axis=1))
    pos_scores = s[pos]
    u = np.zeros((n, n), np.float64)
    for j in range(n):
        neg = np.sort(s[j][~pos[j]])
        if neg.size == 0 or pos_scores.size == 0:
            continue
        below = np.searchsorted(neg, pos_scores, side="left")
        ties = np.searchsorted(neg, pos_scores, side="right") - below
        u[:, j] = np.bincount(owner, weights=below + 0.5 * ties, minlength=n)
    return u, pos.sum(axis=1).astype(np.float64), (~pos).sum(axis=1).astype(np.float64)


def analyse(e1_probe_dir, e2_probe_dir, grades, seeds=SEEDS, n_boot=ep.N_BOOT, boot_seed=ep.BOOT_SEED, log=print):
    """P / E1 / E2 probe scores on the same validation images, and E1 - P, E2 - P, E2 - E1 per seed and for the
    three-seed mean with the §63.2 paired bootstrap (the same grade-stratified image resamples, seed 20260927,
    shared by every model and seed)."""
    import arch1_posthoc as ph
    with np.load(os.path.join(e1_probe_dir, "validation_targets.npz")) as data:
        targets, ids = data["targets"], [str(i) for i in data["image_ids"]]
    with np.load(os.path.join(e2_probe_dir, "validation_targets.npz")) as data:
        if [str(i) for i in data["image_ids"]] != ids or not np.array_equal(data["targets"], targets):
            raise RuntimeError("the E2 probe was scored on other targets")
    grades = np.asarray(grades, int)
    if len(grades) != len(ids):
        raise RuntimeError("grades do not match the validation images")
    positive = targets >= ed.POSITIVE
    classes = em.LESION_CLASSES
    logits = {}
    for seed in seeds:
        for kind, folder in (("p", e1_probe_dir), ("e1", e1_probe_dir), ("e2", e2_probe_dir)):
            with np.load(os.path.join(folder, f"probe_{kind}_seed{int(seed)}.npz")) as data:
                if [str(i) for i in data["image_ids"]] != ids:
                    raise RuntimeError(f"{kind}-{seed}: other images")
                logits[(kind, int(seed))] = data["val_logits"].astype(np.float32)
    point = {int(s): {kind: ep.probe_scores(targets, logits[(kind, int(s))]) for kind in MODELS} for s in seeds}
    stats = {}
    for key, z in logits.items():
        stats[key] = [pair_counts(positive[..., c], z[..., c]) for c in range(len(classes))]
        if log:
            log(f"  pair counts {key}")
    indices = ph.bootstrap_indices(grades, n_boot=n_boot, seed=boot_seed)
    n = len(ids)
    draws = {key: np.zeros(len(indices)) for key in logits}
    for b, idx in enumerate(indices):
        w = np.bincount(idx, minlength=n).astype(np.float64)
        for key, per_class in stats.items():
            draws[key][b] = np.mean([(w @ u @ w) / ((w @ n_pos) * (w @ n_neg)) for u, n_pos, n_neg in per_class])
    interval = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
    contrasts = {"e1_minus_p": ("e1", "p"), "e2_minus_p": ("e2", "p"), "e2_minus_e1": ("e2", "e1")}
    result = {"per_seed": {}, "mean": {}, "n_boot": int(len(indices)), "boot_seed": int(boot_seed),
              "resampling": "grade-stratified image resamples (arch1_posthoc.bootstrap_indices), shared by all models and seeds",
              "bootstrap_method": "exact pair counts per image pair (ties one half)"}
    for s in point:
        row = {kind: point[s][kind] for kind in MODELS}
        for name, (a, b) in contrasts.items():
            row[name] = {k: point[s][a][k] - point[s][b][k] for k in point[s][a]}
            row[name]["mean_ci"] = interval(draws[(a, s)] - draws[(b, s)])
        result["per_seed"][s] = row
    for name, (a, b) in contrasts.items():
        per_seed = [result["per_seed"][int(s)][name]["mean"] for s in seeds]
        mean_draws = np.mean(np.stack([draws[(a, int(s))] - draws[(b, int(s))] for s in seeds]), axis=0)
        ci = interval(mean_draws)
        result["mean"][name] = {"mean": float(np.mean(per_seed)), "ci": ci, "per_seed": per_seed,
                                "positive_seeds": int(sum(v > 0 for v in per_seed)),
                                "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                                "per_class": {c: float(np.mean([result["per_seed"][int(s)][name][c] for s in seeds])) for c in classes}}
    gain = result["mean"]["e1_minus_p"]["mean"]
    result["share_of_e1_gain_reproduced_by_e2"] = float(result["mean"]["e2_minus_p"]["mean"] / gain) if gain else float("nan")
    result["bootstrap_point_check"] = {f"{k}-{s}": float(abs(ep.probe_scores(targets, logits[(k, s)])["mean"]
                                                             - np.mean([(np.ones(n) @ u @ np.ones(n)) / (p.sum() * q.sum())
                                                                        for u, p, q in stats[(k, s)]])))
                                       for (k, s) in logits}
    with open(os.path.join(e2_probe_dir, "e2_probe_comparison.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    return result
