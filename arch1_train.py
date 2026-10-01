"""Architecture-1 training and evaluation (research record §40 step 6 "D", §46). Colab, GPU; ONE seed first.

The P protocol is reused exactly so the run is comparable with P-42 (the frozen ImageNet ConvNeXt-T RGB
baseline): AdamW (lr 1e-4, weight decay 0.05, no decay on 1-D parameters), weighted CORN with the
pre-registered class weights, batch 2, <= 50 epochs, early stopping on val QWK (patience 12),
ReduceLROnPlateau (patience 4, factor 0.5, min 1e-6), mixed precision, P's augmentation and epoch order
(arch1_data), the project's generation-based checkpoints (training.Trainer) with BEST and LAST published
(multiseed_runs). Only the model (arch1_model) and the data (a verified v2 bundle) differ.

Pre-specified one-seed checks (§40 step 6), reported -- never used to tune anything:
  * Q-permutation contribution: QWK(Q permuted across validation images) - QWK <= -0.01;
  * non-inferiority vs P-42: QWK >= P-42 - 0.02 and AUROC(P>=3, grade 4 vs 0-2) >= P-42 - 0.01;
  * guardrails vs P-42 (pl_convnext.GUARDRAILS): grade-3 recall >= -0.10, false-urgent rate <= +0.02.
Permutations of V and of V+Q are reported descriptively (spec §9 contribution tests).
"""
import hashlib
import json
import os
import posixpath
import subprocess

import numpy as np

#: The locked P protocol (verified against the six-run pre-registration in the notebook).
P_PROTOCOL = {"batch_size": 2, "max_epochs": 50, "learning_rate": 1e-4, "weight_decay": 0.05,
              "monitor": "val_QWK", "mode": "max", "early_stopping_patience": 12, "reduce_lr_patience": 4,
              "reduce_lr_factor": 0.5, "min_lr": 1e-6}
FIRST_SEED = 42
PERMUTATION_SEED = 20261001
ONE_SEED_CHECKS = {"q_permutation_dqwk_max": -0.01, "noninferiority_qwk": -0.02, "noninferiority_auroc": -0.01}


def assert_protocol(prereg, class_weights):
    """The six-run pre-registration must still equal the locked P protocol (as the P/PL notebook asserts)."""
    import weighted_corn
    got = {"batch_size": prereg["batch_size"], "max_epochs": prereg["max_epochs"],
           "learning_rate": prereg["learning_rate"], "weight_decay": prereg["weight_decay"],
           "monitor": prereg["primary_metric"], "mode": "max",
           "early_stopping_patience": prereg["early_stopping"]["patience"],
           "reduce_lr_patience": prereg["reduce_lr_on_plateau"]["patience"],
           "reduce_lr_factor": prereg["reduce_lr_on_plateau"]["factor"],
           "min_lr": prereg["reduce_lr_on_plateau"]["min_lr"]}
    bad = {k: (got[k], v) for k, v in P_PROTOCOL.items() if got[k] != v}
    if bad:
        raise RuntimeError(f"pre-registered protocol differs from the locked P protocol: {bad}")
    if not np.allclose(class_weights, weighted_corn.PREREGISTERED_CLASS_WEIGHTS):
        raise RuntimeError("class weights differ from the pre-registered CORN class weights")
    return True


def git_commit(repo_dir=None):
    try:
        return subprocess.run(["git", "-C", repo_dir or os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def run_mapping(bundle, seed, class_weights, protocol=P_PROTOCOL, repo_dir=None):
    """Everything that identifies a run (written to config.json and hashed into the checkpoint config)."""
    import arch1_model
    import pl_convnext as pl
    return {"experiment": "Architecture1", "arch": "convnext_tiny + prior encoder + gated injection + CORN",
            "seed": int(seed), "split_sha256": bundle.bundle["split_sha256"],
            "population_sha256": bundle.bundle.get("population_sha256"), "bundle_id": bundle.bundle["bundle_id"],
            "bundle_fingerprint": bundle.fingerprint, "stage4_sha256": bundle.stage4_sha256,
            "stage4_generation": bundle.stage4_generation, "stage3_sha256": bundle.stage3_sha256,
            "channels": list(bundle.channels), "backbone_weights_sha256": pl.WEIGHTS_SHA256,
            "prior_dims": list(arch1_model.PRIOR_DIMS), "convnext_dims": list(arch1_model.DIMS),
            "class_weights": [float(w) for w in class_weights], "optimizer": "AdamW (no decay on 1-D)",
            "augmentation": "P: lfed._augment_spatial (all channels) + _augment_intensity_rgb (RGB only)",
            **protocol, "mixed_precision": True, "git_commit": git_commit(repo_dir)}


def config_hash(mapping):
    keep = {k: v for k, v in mapping.items() if k != "git_commit"}
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def build_compiled_model(channels, seed, reference, class_weights, protocol=P_PROTOCOL, mixed_precision=True,
                         image_size=512):
    """Architecture 1 compiled exactly like P (pl_convnext.compile_pl_model)."""
    import arch1_model
    import pl_convnext as pl
    from training import enable_mixed_precision
    enable_mixed_precision(bool(mixed_precision))
    model = arch1_model.build_arch1_model(channels, seed, reference, image_size=image_size)
    return pl.compile_pl_model(model, list(class_weights), protocol["learning_rate"], protocol["weight_decay"])


# --------------------------------------------------------------------------- evaluation

def predict_logits(model, bundle, entries, batch_size=8, permute=None, seed=PERMUTATION_SEED):
    """Logits for `entries` in order. `permute` in {None, "vessel", "pathology", "both"} shuffles that prior
    across images (a fixed derangement) to measure its contribution."""
    ids = [i for i, _ in entries]
    perm = np.arange(len(ids))
    if permute:
        rng = np.random.default_rng(seed)
        while True:
            perm = rng.permutation(len(ids))
            if len(ids) < 2 or not np.any(perm == np.arange(len(ids))):
                break
    out = []
    for s in range(0, len(ids), batch_size):
        rows = range(s, min(s + batch_size, len(ids)))
        samples = [bundle.load_sample(ids[r]) for r in rows]
        donors = [bundle.load_sample(ids[perm[r]]) for r in rows] if permute else samples
        x = {"rgb": np.stack([m["rgb"] for m in samples]),
             "vessel": np.stack([(d if permute in ("vessel", "both") else m)["vessel"] for m, d in zip(samples, donors)]),
             "pathology": np.stack([(d if permute in ("pathology", "both") else m)["pathology"]
                                    for m, d in zip(samples, donors)])}
        out.append(np.asarray(model.predict_on_batch(x), np.float64))
    return np.concatenate(out, 0)


def metrics_from_logits(ids, grades, logits):
    """The P per-sample schema (multiseed_runs.build_per_sample_rows) + pl_convnext.run_metrics, plus
    cumulative-cut AUROCs, MAE and per-grade recalls."""
    import pandas as pd

    import corn
    import multiseed_runs as msr
    import pl_convnext as pl
    from sklearn.metrics import roc_auc_score
    grades = np.asarray(grades, int)
    decoded = corn.decode_logits(logits)
    rows = msr.build_per_sample_rows(list(ids), grades, logits, decoded)
    frame = pd.DataFrame(rows)
    m = pl.run_metrics(frame)
    for k in range(4):
        y = grades > k
        m[f"auroc_cut_ge{k + 1}"] = float(roc_auc_score(y, frame[f"p_gt_{k}"])) if 0 < y.sum() < len(y) else float("nan")
    pred = frame["predicted_grade"].to_numpy(int)
    m["recall_per_grade"] = {g: float(np.mean(pred[grades == g] == g)) if np.any(grades == g) else float("nan")
                             for g in range(5)}
    m["mae"] = float(np.mean(np.abs(pred - grades)))
    return m, rows


def evaluate(model, bundle, entries, batch_size=8):
    ids, grades = [i for i, _ in entries], [g for _, g in entries]
    base, rows = metrics_from_logits(ids, grades, predict_logits(model, bundle, entries, batch_size))
    perms = {}
    for which in ("pathology", "vessel", "both"):
        m, _ = metrics_from_logits(ids, grades, predict_logits(model, bundle, entries, batch_size, permute=which))
        perms[which] = {"qwk": m["qwk"], "dqwk": m["qwk"] - base["qwk"],
                        "auroc_ge3_g4_vs_g012": m["auroc_ge3_g4_vs_g012"]}
    return {"metrics": base, "permutation": perms}, rows


def one_seed_checks(arch1, p42):
    """The pre-specified one-seed checks (§40 step 6). `arch1` = evaluate()[0]; `p42` = P-42 metrics dict."""
    import pl_convnext as pl
    m = arch1["metrics"]
    checks = {
        "q_permutation_contributes": arch1["permutation"]["pathology"]["dqwk"] <= ONE_SEED_CHECKS["q_permutation_dqwk_max"],
        "noninferior_qwk": m["qwk"] >= p42["qwk"] + ONE_SEED_CHECKS["noninferiority_qwk"],
        "noninferior_auroc": m["auroc_ge3_g4_vs_g012"] >= p42["auroc_ge3_g4_vs_g012"] + ONE_SEED_CHECKS["noninferiority_auroc"],
    }
    for name, (kind, bound) in pl.GUARDRAILS.items():
        if name == "qwk":
            continue
        delta = m[name] - p42[name]
        checks[f"guardrail_{name}"] = bool(delta >= bound - 1e-12 if kind == "min" else delta <= bound + 1e-12)
    return {"checks": {k: bool(v) for k, v in checks.items()}, "all_pass": bool(all(checks.values())),
            "deltas_vs_p42": {k: (m[k] - p42[k]) for k in ("qwk", "auroc_ge3_g4_vs_g012", "grade3_recall",
                                                             "false_urgent_rate")}}


# --------------------------------------------------------------------------- training (P's loop)

def train_seed(run_dir, bundle, seed, reference, class_weights, *, repo_dir, staging_dir, protocol=P_PROTOCOL,
               log=print, max_epochs=None, grade_of=None, mixed_precision=True):
    """One resumable Architecture-1 run, structured exactly like the P/PL notebook's train_arm_run."""
    import tensorflow as tf

    import multiseed_runs as msr
    from training import CheckpointOptions, Trainer, TrainingConfig, TrainingStateCheckpoint
    from training import checkpointing as ckpt
    epochs = int(max_epochs or protocol["max_epochs"])
    msr.ensure_run_dir(run_dir)
    mapping = run_mapping(bundle, seed, class_weights, protocol, repo_dir)
    chash = config_hash(mapping)
    cfg_path = posixpath.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as fh:
            old = json.load(fh)
        if old.get("config_hash") != chash:
            raise RuntimeError(f"{run_dir}: existing run has a different configuration (data/model/protocol); "
                               "start a new run directory instead of overwriting")
    else:
        with open(cfg_path, "w") as fh:
            json.dump({**mapping, "config_hash": chash}, fh, indent=1, default=str)
    if msr.read_stop_decision(run_dir) is not None:
        log("  run already stopped")
        return
    sealed = msr._sealed_stop(posixpath.join(run_dir, "checkpoints"), epochs)
    if sealed is not None:
        msr.write_stop_decision(run_dir, sealed[0], sealed[1])
        return
    tf.keras.backend.clear_session()
    model = build_compiled_model(bundle.channels, seed, reference, class_weights, protocol, mixed_precision)
    msr.acquire_lock(run_dir, owner_id=msr.OWNER_ID)
    try:
        trainer = Trainer(TrainingConfig(
            run_dir=run_dir, epochs=epochs, monitor=protocol["monitor"], mode=protocol["mode"],
            mixed_precision=bool(mixed_precision),
            resume=True, early_stopping_patience=protocol["early_stopping_patience"],
            reduce_lr_patience=protocol["reduce_lr_patience"], reduce_lr_factor=protocol["reduce_lr_factor"],
            min_lr=protocol["min_lr"], precision_check="error", repo_dir=repo_dir,
            checkpoint_options=CheckpointOptions(
                experiment_id=f"Architecture1/seed_{seed}", config_hash=chash,
                dataset_version=f"bundle:{bundle.fingerprint}", staging_dir=staging_dir, keep_generations=2, verbose=1)))
        trainer.prepare(model)
        initial_epoch = trainer.resolve_initial_epoch()
        if initial_epoch > 0:
            trainer.restore(model)
            log(f"  resumed at epoch {initial_epoch}")
        if initial_epoch >= epochs:
            msr.write_stop_decision(run_dir, initial_epoch, "epoch_cap")
            return
        early = next(c for c in trainer.callbacks if isinstance(c, tf.keras.callbacks.EarlyStopping))
        state_cb = next(c for c in trainer.callbacks if isinstance(c, TrainingStateCheckpoint))
        import arch1_data as ad
        train_entries = [(i, g) for i, g in zip(bundle.train_ids, _grades(bundle, bundle.train_ids, grade_of))]
        val_entries = [(i, g) for i, g in zip(bundle.val_ids, _grades(bundle, bundle.val_ids, grade_of))]
        val_seq = ad.make_epoch_sequence(bundle, val_entries, 0, seed, protocol["batch_size"], augment=False)
        for epoch in range(initial_epoch, epochs):
            msr.heartbeat_lock(run_dir, owner_id=msr.OWNER_ID)
            train_seq = ad.make_epoch_sequence(bundle, train_entries, epoch, seed, protocol["batch_size"], augment=True)
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


def _grades(bundle, ids, grade_of=None):
    """Grades come from the authoritative split (never from file names). `grade_of` is for tests only."""
    if grade_of is not None:
        return [int(grade_of[i]) for i in ids]
    import multiseed_runs as msr
    train_entries, val_entries, split_sha = msr.verify_split()
    if split_sha != bundle.bundle["split_sha256"]:
        raise RuntimeError("bundle split differs from the authoritative split")
    grade = {str(i): int(g) for i, g in list(train_entries) + list(val_entries)}
    return [grade[i] for i in ids]


def evaluate_run(run_dir, bundle, seed, reference, class_weights, protocol=P_PROTOCOL, p42_metrics=None,
                 grade_of=None, mixed_precision=True):
    """BEST and LAST checkpoints -> per-sample CSVs, metrics (+ permutation contributions) and, when the P-42
    metrics are given, the pre-specified one-seed checks. Written under run_dir/metrics."""
    import pandas as pd
    import tensorflow as tf

    import multiseed_runs as msr
    from training import checkpointing as ckpt
    val_entries = [(i, g) for i, g in zip(bundle.val_ids, _grades(bundle, bundle.val_ids, grade_of))]
    out_dir = posixpath.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    best_dir, pointer = msr.read_best(run_dir)
    last_dir = ckpt.find_resumable_generation(posixpath.join(run_dir, "checkpoints"), verbose=False)
    for which, gen_dir in (("best", best_dir), ("last", last_dir)):
        if gen_dir is None:
            raise RuntimeError(f"{run_dir}: no {which.upper()} checkpoint")
        tf.keras.backend.clear_session()
        model = build_compiled_model(bundle.channels, seed, reference, class_weights, protocol, mixed_precision)
        weights = os.path.join(gen_dir, ckpt.MODEL_WEIGHTS_FILENAME)
        ckpt.load_model_weights_only(model, weights)
        res, rows = evaluate(model, bundle, val_entries)
        res["checkpoint"] = {"which": which.upper(), "generation": gen_dir,
                             "weights_sha256": _sha256(weights)}
        if p42_metrics is not None and which == "best":
            res["one_seed_checks"] = one_seed_checks(res, p42_metrics)
        pd.DataFrame(rows).to_csv(posixpath.join(out_dir, f"per_sample_{which}.csv"), index=False)
        with open(posixpath.join(out_dir, f"metrics_{which}.json"), "w") as fh:
            json.dump(res, fh, indent=1, default=float)
        results[which] = res
    return results


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()
