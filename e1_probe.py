"""The pre-registered mechanism probe of the E1 multi-task grader (research record §62, "Mechanism").

Question: can the E1 representation support lesion prediction better than the P representation?

For each seed, on the frozen encoder of P BEST and of E1 BEST, the SAME fresh probe is trained:

    F3 (16, 16, 768) -> Conv2D(4, 1x1) -> logits (MA, HE, EX, SE)           the form of E1's auxiliary head

with the lesion loss only (e1_train.lesion_loss), the E1 targets (e1_data.pool_targets of the cached Stage-4
maps), Adam 1e-3, batch 16, 5 epochs, no augmentation, the final epoch, a fixed seed. Score: cell-level AUROC on
the 730 validation images against target >= 0.5, per class, and the mean of the four classes.

Criterion (fixed in §62, not restated differently here): the mechanism is supported only if
probe(E1) - probe(P) is positive in 3 / 3 seeds AND the 95 % paired bootstrap interval of the three-seed mean
(2,000 resamples over the validation images, seed 20260927) excludes zero.

What the code does to honour "frozen": the encoder is never trained -- its final feature map is computed once
per image (no augmentation, so it is a constant) and the probe is trained on those stored maps, which is the
same optimisation as training the head on top of a frozen encoder. Both encoders are run through the same graph
(e1_model's; gate 1 showed it reproduces P bit for bit). E1's trained lesion head and both grading heads are
not used: the probe's kernel is a fresh GlorotUniform(seed) and its bias the training-target prior, identical
for the P probe and the E1 probe of a seed, as is the order of the training batches.

Choices §62 left open, fixed here before any probe result exists: the probe seed is the run seed; the training
order is a fresh seeded permutation each epoch; features are taken under mixed_float16 (the precision both
models were trained and evaluated in) and stored as float16; the probe itself is float32; the bootstrap
resamples are the project's grade-stratified image resamples (arch1_posthoc.bootstrap_indices).
"""
import json
import os
import posixpath

import numpy as np

import e1_data as ed
import e1_model as em
import e1_train as et

PROBE = {"form": "Conv2D(4, 1x1) on the final 16x16x768 feature map", "loss": "e1_train.lesion_loss",
         "optimizer": "Adam", "learning_rate": 1e-3, "batch_size": 16, "epochs": 5, "augmentation": "none",
         "checkpoint": "final epoch", "feature_policy": "mixed_float16", "probe_dtype": "float32"}
N_BOOT = 2000
BOOT_SEED = 20260927
EXTRACT_BATCH = 16
SEEDS = et.SEEDS
MODELS = ("p", "e1")


# --------------------------------------------------------------------------- metrics

def cell_auroc(positive, score):
    """AUROC of `score` for the boolean `positive` (rank-sum; ties broken by order -- scores are continuous)."""
    positive = np.asarray(positive, bool).ravel()
    score = np.asarray(score).ravel()
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def probe_scores(targets, logits):
    """{class: cell AUROC against target >= 0.5} over all cells of all images, and their mean."""
    out = {name: cell_auroc(targets[..., c] >= ed.POSITIVE, logits[..., c]) for c, name in enumerate(em.LESION_CLASSES)}
    out["mean"] = float(np.mean([out[name] for name in em.LESION_CLASSES]))
    return out


# --------------------------------------------------------------------------- frozen features

def feature_model(model):
    """rgb -> F3, the final (16, 16, 768) feature map (the tensor the grader pools and E1's head reads)."""
    from keras import Model
    return Model(model.input, model.get_layer(em.FEATURE_MAP_NAME).output, name="frozen_encoder")


def build_frozen_encoder(kind, seed, weights_path, lesion_prior, reference_arrays, policy=PROBE["feature_policy"]):
    """The encoder of P BEST (`kind` 'p') or of E1 BEST ('e1') for one seed, in E1's graph, never trained here.
    Returns (feature model, the full model for a grading-logit sanity check)."""
    import keras

    import pl_convnext as pl
    from training import checkpointing as ckpt
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy(policy)          # after clear_session (it resets the policy)
    model = em.build_e1_model(int(seed), lesion_prior)
    if kind == "p":
        p = pl.build_pl_model("P", int(seed), reference_arrays)
        ckpt.load_model_weights_only(p, weights_path)
        em.copy_from_p(model, p)
        del p
    elif kind == "e1":
        ckpt.load_model_weights_only(model, weights_path)
    else:
        raise ValueError(f"kind must be 'p' or 'e1', got {kind!r}")
    model.trainable = False
    expected = "float16" if policy == "mixed_float16" else "float32"
    if str(model.outputs[0].dtype) != expected:
        raise RuntimeError(f"the encoder was not built under {policy}")
    return feature_model(model), model


def _ram():
    try:
        import psutil
        m = psutil.virtual_memory()
        return f"RAM {m.used / 2 ** 30:.1f}/{m.total / 2 ** 30:.1f} GB"
    except Exception:  # noqa: BLE001
        return ""


def extract(bundle, features, full, image_ids, path, batch_size=EXTRACT_BATCH, log=None):
    """F3 of every image, unaugmented, written batch by batch to a float16 .npy memmap at `path` (never held
    in RAM as a whole), and the grading logits (N, 4) float64. Returns (read-only memmap, grading logits)."""
    import stage34_cache_v2 as cache
    ids = [str(i) for i in image_ids]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    shape = (len(ids),) + tuple(int(d) for d in features.output.shape[1:])
    store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    grading = np.zeros((len(ids), 4), np.float64)
    for s in range(0, len(ids), batch_size):
        chunk = ids[s:s + batch_size]
        rgb = np.stack([cache.read_stage2_rgb(bundle._path("stage2_rgb_v2", i),
                                              expected_file_sha256=bundle._sha("stage2_rgb_v2", i)) for i in chunk])
        store[s:s + len(chunk)] = np.asarray(features.predict_on_batch({"rgb": rgb})).astype(np.float16)
        grading[s:s + len(chunk)] = np.asarray(full.predict_on_batch({"rgb": rgb})[0], np.float64)
        if log and (s // batch_size) % 50 == 0:
            log(f"    features {min(s + batch_size, len(ids))}/{len(ids)}  {_ram()}")
    store.flush()
    del store
    return np.load(path, mmap_mode="r"), grading


def targets_for(bundle, image_ids):
    return np.stack([ed.pool_targets(ed.load_inputs(bundle, str(i))["pathology"], bundle.channels)
                     for i in image_ids]).astype(np.float32)


# --------------------------------------------------------------------------- the probe

def build_probe(seed, lesion_prior, feature_shape):
    """The fresh probe: E1's auxiliary head form, float32, GlorotUniform(seed) kernel, prior-logit bias."""
    import keras
    from keras import Input, Model, initializers, layers
    x = Input(shape=tuple(feature_shape), dtype="float16", name="frozen_features")
    h = layers.Lambda(lambda t: keras.ops.cast(t, "float32"), dtype="float32", name="to_float32")(x)
    z = layers.Conv2D(len(em.LESION_CLASSES), 1, dtype="float32", name="probe_conv",
                      kernel_initializer=initializers.GlorotUniform(seed=int(seed)),
                      bias_initializer=initializers.Constant(em.prior_logits(lesion_prior).tolist()))(h)
    model = Model(x, z, name="lesion_probe")
    model.compile(optimizer=keras.optimizers.Adam(PROBE["learning_rate"]), loss=et.lesion_loss)
    return model


def _predict(probe, features, batch=256):
    """Probe logits for stored features, read from the (memory-mapped) array in batches."""
    out = np.zeros((len(features), features.shape[1], features.shape[2], len(em.LESION_CLASSES)), np.float32)
    for s in range(0, len(features), batch):
        out[s:s + batch] = np.asarray(probe.predict_on_batch(np.ascontiguousarray(features[s:s + batch])), np.float32)
    return out


def train_probe(seed, lesion_prior, train_features, train_targets, val_features, val_targets, log=None):
    """Trains the fresh probe for PROBE['epochs'] epochs and returns the FINAL epoch's validation logits with
    the per-epoch behaviour (training loss as it ran, then validation loss and validation AUROC with the
    epoch's weights). The per-epoch numbers are reported only; nothing is selected with them. The feature
    arrays may be memory-mapped: only one batch is in RAM at a time."""
    import keras
    keras.mixed_precision.set_global_policy("float32")
    probe = build_probe(seed, lesion_prior, train_features.shape[1:])
    initial = [w.copy() for w in probe.get_weights()]
    size = PROBE["batch_size"]

    class _Epoch(keras.utils.PyDataset):
        def __init__(self, order):
            super().__init__()
            self.order = order

        def __len__(self):
            return int(np.ceil(len(self.order) / size))

        def __getitem__(self, index):
            rows = self.order[index * size:(index + 1) * size]
            return np.stack([train_features[int(r)] for r in rows]), train_targets[rows]

    curve = []
    start = _predict(probe, val_features)
    curve.append({"epoch": 0, "val_loss": et.lesion_loss_numpy(val_targets, start),
                  "val_mean_cell_auroc": probe_scores(val_targets, start)["mean"]})
    for epoch in range(PROBE["epochs"]):
        order = np.random.default_rng([int(seed), epoch]).permutation(len(train_features))
        history = probe.fit(_Epoch(order), epochs=1, shuffle=False, verbose=0)
        val_logits = _predict(probe, val_features)
        row = {"epoch": epoch + 1, "running_train_loss": float(history.history["loss"][0]),
               "val_loss": et.lesion_loss_numpy(val_targets, val_logits),
               "val_mean_cell_auroc": probe_scores(val_targets, val_logits)["mean"]}
        curve.append(row)
        if log:
            log(f"    probe epoch {epoch + 1}: train loss {row['running_train_loss']:.4f} | val loss {row['val_loss']:.4f} | "
                f"val mean cell AUROC {row['val_mean_cell_auroc']:.4f}  {_ram()}")
    train_logits = _predict(probe, train_features)
    final = {"train_loss": et.lesion_loss_numpy(train_targets, train_logits),
             "train_scores": probe_scores(train_targets, train_logits),
             "val_loss": curve[-1]["val_loss"], "val_scores": probe_scores(val_targets, val_logits),
             "kernel_moved": float(np.abs(probe.get_weights()[0] - initial[0]).max())}
    return val_logits, curve, final


def e1_best_weights(run_dir):
    """(path, sha256) of an E1 run's BEST weights; the sha must equal the one its result.json recorded."""
    import multiseed_runs as msr
    import stage34_cache_v2 as cache
    from training import checkpointing as ckpt
    best_dir, _ = msr.read_best(run_dir)
    if best_dir is None:
        raise RuntimeError(f"{run_dir}: no BEST checkpoint")
    path = os.path.join(best_dir, ckpt.MODEL_WEIGHTS_FILENAME)
    sha = cache.sha256_file(path)
    with open(posixpath.join(run_dir, "result.json")) as fh:
        recorded = json.load(fh)["best"]["checkpoint"]["weights_sha256"]
    if sha != recorded:
        raise RuntimeError(f"{run_dir}: BEST weights {sha} are not the evaluated ones ({recorded})")
    return path, sha


def run(bundle, p_weight_paths, e1_run_dirs, convnext_weights_path, lesion_prior, out_dir, *, p_sha256=None,
        seeds=SEEDS, train_ids=None, val_ids=None, work_dir="/content/e1_probe_features", log=print):
    """The probe for every seed and both encoders. Writes, per (model, seed), the final validation logits and
    the behaviour, and summary.json with the point estimates. The bootstrap is `analyse`.

    Memory: features are written to a float16 memmap under `work_dir` (local disk, ~1.5 GB per model, deleted
    when that model's probe is done) and read in batches. Resumable: a (model, seed) whose result is already in
    summary.json -- for the same bundle, prior, protocol and weights -- is not recomputed."""
    import gc

    import keras

    import pl_convnext as pl
    import stage34_cache_v2 as cache
    train_ids = [str(i) for i in (train_ids if train_ids is not None else bundle.train_ids)]
    val_ids = [str(i) for i in (val_ids if val_ids is not None else bundle.val_ids)]
    os.makedirs(out_dir, exist_ok=True)
    previous = keras.mixed_precision.global_policy().name
    identity = {"probe": PROBE, "lesion_prior": [float(p) for p in lesion_prior], "train_images": len(train_ids),
                "val_images": len(val_ids), "bundle_fingerprint": bundle.fingerprint}
    summary = dict(identity, criterion="probe(E1) - probe(P) > 0 in 3/3 seeds AND the 95% paired bootstrap interval of the "
                                       "three-seed mean excludes zero (record 62)", seeds={})
    summary_path = os.path.join(out_dir, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            old = json.load(fh)
        if all(old.get(k) == v for k, v in json.loads(json.dumps(identity)).items()):
            summary["seeds"] = {int(k): v for k, v in old.get("seeds", {}).items()}
            log(f"  resuming: {[(s, sorted(k for k in v if k in MODELS)) for s, v in summary['seeds'].items()]} already done")

    def save():
        with open(summary_path + ".tmp", "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        os.replace(summary_path + ".tmp", summary_path)

    try:
        keras.mixed_precision.set_global_policy("float32")
        _, reference_arrays = pl.load_reference(convnext_weights_path)
        log("  targets ...")
        train_targets, val_targets = targets_for(bundle, train_ids), targets_for(bundle, val_ids)
        np.savez_compressed(os.path.join(out_dir, "validation_targets.npz"), targets=val_targets, image_ids=np.asarray(val_ids))
        for seed in seeds:
            seed = int(seed)
            e1_path, e1_sha = e1_best_weights(e1_run_dirs[seed])
            p_sha = cache.sha256_file(p_weight_paths[seed])
            if p_sha256 and p_sha != p_sha256[seed]:
                raise RuntimeError(f"P-{seed}: checkpoint {p_sha} is not the pinned {p_sha256[seed]}")
            done = summary["seeds"].setdefault(seed, {})
            for kind, path, sha in (("p", p_weight_paths[seed], p_sha), ("e1", e1_path, e1_sha)):
                saved = os.path.join(out_dir, f"probe_{kind}_seed{seed}.npz")
                if done.get(kind, {}).get("weights_sha256") == sha and os.path.exists(saved):
                    log(f"  seed {seed} / {kind}: already done (val mean cell AUROC "
                        f"{done[kind]['final']['val_scores']['mean']:.4f}) -- kept")
                    continue
                log(f"  seed {seed} / {kind}: frozen features  {_ram()}")
                features, full = build_frozen_encoder(kind, seed, path, lesion_prior, reference_arrays)
                files = [os.path.join(work_dir, f"{kind}_seed{seed}_{part}.npy") for part in ("train", "val")]
                train_f, _ = extract(bundle, features, full, train_ids, files[0], log=log)
                val_f, val_grading = extract(bundle, features, full, val_ids, files[1], log=log)
                del features, full
                keras.backend.clear_session()
                gc.collect()
                val_logits, curve, final = train_probe(seed, lesion_prior, train_f, train_targets, val_f, val_targets, log=log)
                np.savez_compressed(saved, val_logits=val_logits, val_grading_logits=val_grading, image_ids=np.asarray(val_ids))
                done[kind] = {"weights_sha256": sha, "curve": curve, "final": final,
                              "feature_abs_mean": float(np.mean([np.abs(val_f[i].astype(np.float32)).mean()
                                                                 for i in range(len(val_f))]))}
                log(f"  seed {seed} / {kind}: val mean cell AUROC {final['val_scores']['mean']:.4f} "
                    f"{ {k: round(v, 4) for k, v in final['val_scores'].items() if k != 'mean'} }")
                del train_f, val_f
                keras.backend.clear_session()
                gc.collect()
                for f in files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
                save()
            done["difference_e1_minus_p"] = {k: done["e1"]["final"]["val_scores"][k] - done["p"]["final"]["val_scores"][k]
                                             for k in done["p"]["final"]["val_scores"]}
            log(f"  seed {seed}: probe(E1) - probe(P) = {done['difference_e1_minus_p']['mean']:+.4f}")
            save()
    finally:
        keras.mixed_precision.set_global_policy(previous)
    return summary


# --------------------------------------------------------------------------- the pre-registered comparison

def analyse(out_dir, grades, seeds=SEEDS, n_boot=N_BOOT, boot_seed=BOOT_SEED, log=print):
    """probe(E1) - probe(P) per seed and for the three-seed mean, with the paired bootstrap over the validation
    images (the same resamples for both probes and all seeds), and the §62 criterion. `grades`: the validation
    grades in the stored order, used only to stratify the resamples (arch1_posthoc.bootstrap_indices)."""
    import arch1_posthoc as ph
    with np.load(os.path.join(out_dir, "validation_targets.npz")) as data:
        targets, ids = data["targets"], [str(i) for i in data["image_ids"]]
    grades = np.asarray(grades, int)
    if len(grades) != len(ids):
        raise RuntimeError("grades do not match the validation images")
    logits = {}
    for seed in seeds:
        for kind in MODELS:
            with np.load(os.path.join(out_dir, f"probe_{kind}_seed{int(seed)}.npz")) as data:
                if [str(i) for i in data["image_ids"]] != ids:
                    raise RuntimeError(f"{kind}-{seed}: other images")
                logits[(kind, int(seed))] = data["val_logits"].astype(np.float32)
    positive = targets >= ed.POSITIVE
    classes = em.LESION_CLASSES
    point = {int(s): {kind: probe_scores(targets, logits[(kind, int(s))]) for kind in MODELS} for s in seeds}
    for s in point:
        point[s]["difference"] = {k: point[s]["e1"][k] - point[s]["p"][k] for k in point[s]["p"]}
    indices = ph.bootstrap_indices(grades, n_boot=n_boot, seed=boot_seed)
    draws = {int(s): [] for s in seeds}
    for n, idx in enumerate(indices):
        pos = positive[idx]
        for s in seeds:
            means = {}
            for kind in MODELS:
                z = logits[(kind, int(s))][idx]
                means[kind] = np.mean([cell_auroc(pos[..., c], z[..., c]) for c in range(len(classes))])
            draws[int(s)].append(means["e1"] - means["p"])
        if log and (n + 1) % 250 == 0:
            log(f"  bootstrap {n + 1}/{len(indices)}")
    draws = {s: np.asarray(v, np.float64) for s, v in draws.items()}
    mean_draws = np.mean(np.stack([draws[int(s)] for s in seeds]), axis=0)
    interval = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
    per_seed = {s: {"p": point[s]["p"], "e1": point[s]["e1"], "difference": point[s]["difference"],
                    "difference_mean_ci": interval(draws[s])} for s in point}
    differences = [point[int(s)]["difference"]["mean"] for s in seeds]
    ci = interval(mean_draws)
    positive_seeds = int(sum(d > 0 for d in differences))
    supported = bool(positive_seeds == len(differences) and (ci[0] > 0 or ci[1] < 0) and np.mean(differences) > 0)
    result = {"per_seed": per_seed, "mean_difference": float(np.mean(differences)), "mean_difference_ci": ci,
              "per_class_mean_difference": {c: float(np.mean([point[int(s)]["difference"][c] for s in seeds])) for c in classes},
              "positive_seeds": positive_seeds, "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
              "mechanism_supported": supported, "n_boot": int(len(indices)), "boot_seed": int(boot_seed),
              "resampling": "grade-stratified image resamples (arch1_posthoc.bootstrap_indices)",
              "criterion": "positive in 3/3 seeds AND the 95% interval of the three-seed mean excludes zero (record 62)"}
    with open(os.path.join(out_dir, "mechanism_probe_result.json"), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    return result
