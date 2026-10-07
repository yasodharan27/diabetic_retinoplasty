"""DDR ground-truth lesion probe (research record §68).

Question: does E1's frozen representation contain more information predictive of REAL, expert-annotated
retinal lesions than P's? (The APTOS probe of §63.2 used Stage-4 pseudo-labels; this one uses DDR's masks.)

No grader is trained and no encoder is changed. For each seed, the frozen BEST encoders of P, E1 and E2 give
the final 16 x 16 x 768 feature map of every DDR image (Stage 2 once, then the locked 512 frame), and the SAME
fresh 1 x 1 probe (e1_probe.build_probe: the form of E1's auxiliary head) is trained on the DDR training split
to predict, per cell, whether an expert-annotated lesion of each class (MA, HE, EX, SE) is present.

Two results per (encoder, seed), both pre-declared and both reported:
  * FIVE-EPOCH  -- the protocol of §63.2 unchanged: Adam 1e-3, batch 16, 5 epochs, no augmentation, seeded
                   kernel and batch order, final epoch.
  * CONVERGED   -- the same probe continued with the same optimiser and batch-order rule until the DDR
                   VALIDATION loss stops improving (STOPPING below); the weights of the best validation epoch.
The test split is predicted exactly twice per probe -- once with each of those two weight sets -- after the
stopping decision is final. Nothing is chosen from test performance.

Targets: DDR's masks are binary images at the native size, one file per class. The image is used full-frame
(no crop), so cell (r, c) of the 16 x 16 grid covers the native pixels rows [floor(r H / 16), floor((r+1) H / 16))
and columns [floor(c W / 16), floor((c+1) W / 16)); the target is 1 if any lesion pixel lies in the cell, else 0.
The class channels stay separate, in E1's order MA, HE, EX, SE.

Score: cell-level AUROC per class over all test cells, and the mean of the four classes. Comparisons are paired
on the same test images with a bootstrap over images (2,000 resamples, seed 20260927).
"""
import csv
import hashlib
import json
import os

import numpy as np

import e1_model as em
import e1_probe as ep
import e1_train as et

SEEDS = (42, 123, 2026)
MODELS = ("p", "e1", "e2")
CLASSES = em.LESION_CLASSES                                   # MA, HE, EX, SE
SPLITS = {"train": ("train", "train"), "val": ("val", "val"), "test": ("test", "tet")}   # split -> (image dir, mask dir)
GRID = 16
MANIFEST_NAME = "ddr_probe_manifest.csv"
MANIFEST_SHA256 = "2f3e4a40fa17e0af8706c859002163a18d56d594ca43f856a54cac65dbdcc2d0"
EXCLUDED = ("007-5869-300.jpg",)                             # research record §66.1
EXPECTED_COUNTS = {"train": 383, "val": 148, "test": 225}
FIVE_EPOCHS = ep.PROBE["epochs"]
STOPPING = {"monitor": "validation lesion loss (DDR validation split)", "min_delta": 1e-4, "patience": 5,
            "max_epochs": 100, "weights": "the epoch with the lowest validation loss",
            "min_epochs": FIVE_EPOCHS}
N_BOOT = 2000
BOOT_SEED = 20260927
CONTRASTS = {"e1_minus_p": ("e1", "p"), "e2_minus_p": ("e2", "p"), "e1_minus_e2": ("e1", "e2")}
PRIMARY = "e1_minus_p"
VARIANTS = ("five_epoch", "converged")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- data

def read_manifest(audit_dir, lesion_root=None, verify_images=True):
    """The locked 756-image probe set: {split: [(image name, sha256)]}. The manifest file must be the pinned
    one; with `lesion_root`, every listed image (and its four masks) must exist and match its SHA-256."""
    path = os.path.join(audit_dir, MANIFEST_NAME)
    if sha256_file(path) != MANIFEST_SHA256:
        raise RuntimeError("the DDR probe manifest is not the pinned file")
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {s: [] for s in SPLITS}
    for r in rows:
        if r["in_probe_set"] == "1":
            out[r["split"]].append((r["image"], r["sha256"]))
        elif r["image"] not in EXCLUDED:
            raise RuntimeError(f"{r['image']} is excluded but is not a recorded exclusion")
    if {s: len(v) for s, v in out.items()} != EXPECTED_COUNTS:
        raise RuntimeError(f"probe set counts {({s: len(v) for s, v in out.items()})} are not {EXPECTED_COUNTS}")
    names = [n for v in out.values() for n, _ in v]
    if len(set(names)) != len(names) or set(names) & set(EXCLUDED):
        raise RuntimeError("the probe set has a repeated or an excluded image")
    if lesion_root is not None:
        for split, items in out.items():
            for name, sha in items:
                if verify_images and sha256_file(image_path(lesion_root, split, name)) != sha:
                    raise RuntimeError(f"{name}: image differs from the manifest")
                for c in CLASSES:
                    if not os.path.exists(mask_path(lesion_root, split, name, c)):
                        raise RuntimeError(f"{name}: no {c} mask")
    return out


def image_path(lesion_root, split, name):
    return os.path.join(lesion_root, "images", SPLITS[split][0], name)


def mask_path(lesion_root, split, name, lesion_class):
    return os.path.join(lesion_root, "annotations", SPLITS[split][1], lesion_class, os.path.splitext(name)[0] + ".tif")


def cell_edges(size, grid=GRID):
    """The start index of each of the `grid` cells along an axis of `size` pixels (full-frame, no crop)."""
    edges = np.floor(np.arange(grid) * size / grid).astype(int)
    if size < grid or len(np.unique(edges)) != grid:
        raise ValueError(f"an axis of {size} pixels cannot be divided into {grid} cells")
    return edges


def cell_targets(mask, grid=GRID):
    """(H, W) mask -> (grid, grid) float32 in {0, 1}: 1 where any lesion pixel (value > 0) lies in the cell."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"expected a single-channel mask, got shape {mask.shape}")
    present = (mask > 0).astype(np.uint8)
    rows = np.maximum.reduceat(present, cell_edges(mask.shape[0], grid), axis=0)
    return np.maximum.reduceat(rows, cell_edges(mask.shape[1], grid), axis=1).astype(np.float32)


def cell_targets_bruteforce(mask, grid=GRID):
    """The same by explicit loops (an independent implementation for the tests)."""
    mask = np.asarray(mask)
    h, w = mask.shape
    out = np.zeros((grid, grid), np.float32)
    for r in range(grid):
        for c in range(grid):
            r0, r1 = int(np.floor(r * h / grid)), int(np.floor((r + 1) * h / grid))
            c0, c1 = int(np.floor(c * w / grid)), int(np.floor((c + 1) * w / grid))
            out[r, c] = float((mask[r0:r1, c0:c1] > 0).any())
    return out


def load_targets(lesion_root, split, name):
    """(16, 16, 4) targets of one image, channels MA, HE, EX, SE; the masks must be the size of the image."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path(lesion_root, split, name)) as im:
        size = im.size
    channels = []
    for c in CLASSES:
        with Image.open(mask_path(lesion_root, split, name, c)) as m:
            if m.size != size:
                raise RuntimeError(f"{name}: the {c} mask is {m.size}, the image {size}")
            channels.append(cell_targets(np.asarray(m)))
    return np.stack(channels, axis=-1)


def frame_from_raw(raw_path):
    """The locked image path: Stage 2 (DR profile, once, native size), then the canonical 512 frame."""
    import stage4_v2_aptos_cache as ac
    import stage4_v2_data as sd
    return ac.recompute_rgb_512(sd.stage2_rgb(raw_path))


# --------------------------------------------------------------------------- the probe (five-epoch + converged)

def train_probes(seed, prior, train_f, train_t, val_f, val_t, stopping=STOPPING, log=None):
    """One fresh probe (e1_probe.build_probe: seeded kernel, prior bias, Adam 1e-3) trained with the §63.2 rule
    -- batch 16, a seeded permutation per epoch, no augmentation. Returns the weights after epoch 5 (the
    five-epoch probe), the weights of the best validation epoch under `stopping` (the converged probe), and
    the per-epoch behaviour. Only training and validation data enter; the test split is not an argument."""
    import keras
    keras.mixed_precision.set_global_policy("float32")
    probe = ep.build_probe(seed, prior, train_f.shape[1:])
    size = ep.PROBE["batch_size"]

    class _Epoch(keras.utils.PyDataset):
        def __init__(self, order):
            super().__init__()
            self.order = order

        def __len__(self):
            return int(np.ceil(len(self.order) / size))

        def __getitem__(self, index):
            rows = self.order[index * size:(index + 1) * size]
            return np.stack([train_f[int(r)] for r in rows]), train_t[rows]

    def validation():
        logits = ep._predict(probe, val_f)
        return et.lesion_loss_numpy(val_t, logits), ep.probe_scores(val_t, logits)["mean"]

    loss0, auroc0 = validation()
    curve = [{"epoch": 0, "val_loss": loss0, "val_mean_cell_auroc": auroc0}]
    five = None
    best = {"epoch": 0, "val_loss": np.inf, "weights": None}
    stale, epoch = 0, 0
    while epoch < stopping["max_epochs"]:
        order = np.random.default_rng([int(seed), epoch]).permutation(len(train_f))
        history = probe.fit(_Epoch(order), epochs=1, shuffle=False, verbose=0)
        epoch += 1
        val_loss, val_auroc = validation()
        curve.append({"epoch": epoch, "running_train_loss": float(history.history["loss"][0]), "val_loss": val_loss,
                      "val_mean_cell_auroc": val_auroc})
        if epoch == FIVE_EPOCHS:
            five = [w.copy() for w in probe.get_weights()]
        if val_loss < best["val_loss"] - stopping["min_delta"]:
            best, stale = {"epoch": epoch, "val_loss": val_loss, "weights": [w.copy() for w in probe.get_weights()]}, 0
        else:
            stale += 1
        if log and (epoch <= FIVE_EPOCHS or epoch % 10 == 0):
            log(f"      epoch {epoch}: val loss {val_loss:.4f} | val mean cell AUROC {val_auroc:.4f}")
        if epoch >= stopping["min_epochs"] and stale >= stopping["patience"]:
            break
    if five is None or best["weights"] is None:
        raise RuntimeError("the probe did not reach five epochs or never improved")
    behaviour = {"curve": curve, "epochs_run": epoch, "converged_epoch": best["epoch"],
                 "stopped_by": "patience" if epoch < stopping["max_epochs"] else "epoch cap",
                 "val_loss_at_five": curve[FIVE_EPOCHS]["val_loss"], "val_loss_converged": best["val_loss"]}
    return probe, {"five_epoch": five, "converged": best["weights"]}, behaviour


def evaluate_probe(probe, weights, features, targets):
    probe.set_weights(weights)
    logits = ep._predict(probe, features)
    return logits, {"loss": et.lesion_loss_numpy(targets, logits), "scores": ep.probe_scores(targets, logits)}


# --------------------------------------------------------------------------- pinned checkpoints

def pinned_checkpoints(repo_dir, drive_root):
    """{(model, seed): (path, sha256)} -- P from the locked IDRiD protocol, E1 / E2 from the batch-1 protocol.
    Every file is hashed and must match."""
    with open(os.path.join(repo_dir, "idrid_grading_protocol.json"), encoding="utf-8") as fh:
        p_pins = json.load(fh)["checkpoints"]
    with open(os.path.join(repo_dir, "idrid_batch1_protocol.json"), encoding="utf-8") as fh:
        e_pins = json.load(fh)["checkpoints"]
    out = {}
    for seed in SEEDS:
        for model, entry in (("p", p_pins[f"p_seed{seed}"]), ("e1", e_pins[f"e1_seed{seed}"]), ("e2", e_pins[f"e2_seed{seed}"])):
            path = os.path.join(drive_root, *entry["path"].split("/"))
            if not os.path.exists(path) or sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"{model}-{seed}: {path} is missing or is not the pinned checkpoint")
            out[(model, seed)] = (path, entry["sha256"])
    pretrained = p_pins["convnext_pretrained"]
    return out, os.path.join(drive_root, *pretrained["path"].split("/")), pretrained["sha256"]


# --------------------------------------------------------------------------- run metadata

def environment(repo_dir=None):
    """Where and with what a run was made: commit, library versions, device, time. Descriptive only -- it is
    kept out of configuration.json so that resuming on another runtime is not a configuration change."""
    import datetime
    import platform
    import sys

    import arch1_train as at
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


def record_invocation(out_dir, record):
    """Appends one invocation to run_metadata.json (a resumed run has several)."""
    path = os.path.join(out_dir, "run_metadata.json")
    runs = []
    if os.path.exists(path):
        with open(path) as fh:
            runs = json.load(fh)["invocations"]
    runs.append(record)
    with open(path + ".tmp", "w") as fh:
        json.dump({"invocations": runs}, fh, indent=1)
    os.replace(path + ".tmp", path)
    return runs


# --------------------------------------------------------------------------- shared steps of a probe run

def prepare_data(lesion_root, ids, work_dir, log=print):
    """Frames (Stage 2 once per image, then the 512 frame; a float32 memmap per split under `work_dir`) and the
    (16, 16, 4) cell targets of every image of the probe set. Returns (frames, targets)."""
    frames, targets = {}, {}
    for split in SPLITS:
        path = os.path.join(work_dir, f"frames_{split}.npy")
        done = os.path.join(work_dir, f"frames_{split}.done")
        if not os.path.exists(done):
            store = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(ids[split]), 512, 512, 3))
            for n, name in enumerate(ids[split]):
                store[n] = frame_from_raw(image_path(lesion_root, split, name))
                if (n + 1) % 50 == 0 or n + 1 == len(ids[split]):
                    log(f"  frames {split} {n + 1}/{len(ids[split])}  {ep._ram()}")
            store.flush()
            del store
            open(done, "w").close()
        frames[split] = np.load(path, mmap_mode="r")
        targets[split] = np.stack([load_targets(lesion_root, split, name) for name in ids[split]]).astype(np.float32)
    return frames, targets


def probe_encoder(kind, seed, weights_path, reference_arrays, frames, ids, targets, prior, log=print):
    """One frozen encoder (`kind` 'p' = P's graph, 'e1' = E1's graph) and its probe: features of every split,
    the probe trained on the training split with the validation split for the stopping rule, and exactly two
    predictions of the test split (one per weight set). Returns (result, arrays to save)."""
    import gc

    import keras
    features, full = ep.build_frozen_encoder(kind, seed, weights_path, [0.1] * 4, reference_arrays)
    f = {}
    for split in SPLITS:
        chunks = []
        for s in range(0, len(ids[split]), ep.EXTRACT_BATCH):
            rgb = np.ascontiguousarray(frames[split][s:s + ep.EXTRACT_BATCH])
            chunks.append(np.asarray(features.predict_on_batch({"rgb": rgb})).astype(np.float16))
        f[split] = np.concatenate(chunks, 0)
    del features, full
    keras.backend.clear_session()
    gc.collect()
    probe, weights, behaviour = train_probes(seed, prior, f["train"], targets["train"], f["val"], targets["val"], log=log)
    result = {"behaviour": behaviour}
    arrays = {"test_ids": np.asarray(ids["test"])}
    for variant in VARIANTS:                         # the only two test predictions of this probe
        test_logits, test = evaluate_probe(probe, weights[variant], f["test"], targets["test"])
        _, val = evaluate_probe(probe, weights[variant], f["val"], targets["val"])
        _, train = evaluate_probe(probe, weights[variant], f["train"], targets["train"])
        result[variant] = {"test": test, "val": val, "train": train}
        arrays[f"test_logits_{variant}"] = test_logits
    del probe, f
    keras.backend.clear_session()
    gc.collect()
    return result, arrays


# --------------------------------------------------------------------------- the run (GPU for the encoders)

def run(drive_root, lesion_root, audit_dir, out_dir, *, repo_dir=None, work_dir="/content/ddr_probe_work", seeds=SEEDS,
        log=print):
    """Frames and targets for the 756 images, then for every seed and encoder: frozen features, the probe
    (five-epoch and converged), and the two test predictions. Resumable per (model, seed). Writes
    configuration.json, run_metadata.json, targets.npz, one npz per (model, seed) and summary.json. The comparison is `analyse`."""
    import gc

    import keras

    import pl_convnext as pl
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    manifest = read_manifest(audit_dir, lesion_root)
    checkpoints, pretrained, pretrained_sha = pinned_checkpoints(repo_dir, drive_root)
    if sha256_file(pretrained) != pretrained_sha:
        raise RuntimeError("the ImageNet ConvNeXt file is not the pinned one")
    ids = {s: [n for n, _ in manifest[s]] for s in SPLITS}
    identity = {"experiment": "DDR ground-truth lesion probe", "manifest_sha256": MANIFEST_SHA256, "excluded": list(EXCLUDED),
                "counts": {s: len(v) for s, v in ids.items()}, "classes": list(CLASSES), "grid": GRID,
                "target": "cell = 1 if any expert lesion pixel lies in it (full frame, native size), else 0",
                "probe": ep.PROBE, "stopping": STOPPING, "variants": list(VARIANTS),
                "checkpoints": {f"{m}_seed{s}": sha for (m, s), (_, sha) in checkpoints.items()},
                "contrasts": {k: list(v) for k, v in CONTRASTS.items()}, "primary": PRIMARY,
                "bootstrap": {"n": N_BOOT, "seed": BOOT_SEED, "unit": "test image", "stratified": False}}
    config_path = os.path.join(out_dir, "configuration.json")
    if os.path.exists(config_path):
        with open(config_path) as fh:
            if json.load(fh) != json.loads(json.dumps(identity)):
                raise RuntimeError(f"{out_dir} holds a run with another configuration")
    else:
        with open(config_path, "w") as fh:
            json.dump(identity, fh, indent=1)
    record_invocation(out_dir, dict(environment(repo_dir), manifest_sha256=MANIFEST_SHA256, pretrained_sha256=pretrained_sha,
                                    checkpoints=identity["checkpoints"], seeds=[int(s) for s in seeds],
                                    feature_policy=ep.PROBE["feature_policy"]))
    # frames (Stage 2 once per image) and targets
    frames, targets = prepare_data(lesion_root, ids, work_dir, log)
    prior = targets["train"].mean(axis=(0, 1, 2)).astype(np.float64).tolist()
    np.savez_compressed(os.path.join(out_dir, "targets.npz"), **{f"{s}_targets": targets[s] for s in SPLITS},
                        **{f"{s}_ids": np.asarray(ids[s]) for s in SPLITS})
    summary_path = os.path.join(out_dir, "summary.json")
    summary = {"prior": prior, "positive_cell_rate": {s: targets[s].mean(axis=(0, 1, 2)).tolist() for s in SPLITS}, "results": {}}
    if os.path.exists(summary_path):
        with open(summary_path) as fh:
            old = json.load(fh)
        if old.get("prior") == prior:
            summary["results"] = old.get("results", {})
    previous = keras.mixed_precision.global_policy().name
    try:
        keras.mixed_precision.set_global_policy("float32")
        _, reference_arrays = pl.load_reference(pretrained)
        for seed in seeds:
            for model in MODELS:
                key = f"{model}_seed{seed}"
                saved = os.path.join(out_dir, f"probe_{key}.npz")
                path, sha = checkpoints[(model, int(seed))]
                if summary["results"].get(key, {}).get("weights_sha256") == sha and os.path.exists(saved):
                    log(f"  {key}: already done -- kept")
                    continue
                log(f"  {key}: frozen features  {ep._ram()}")
                result, arrays = probe_encoder("p" if model == "p" else "e1", seed, path, reference_arrays, frames, ids,
                                               targets, prior, log)
                result = {"weights_sha256": sha, **result}
                behaviour = result["behaviour"]
                np.savez_compressed(saved, **arrays)
                summary["results"][key] = result
                log(f"  {key}: test mean cell AUROC five-epoch {result['five_epoch']['test']['scores']['mean']:.4f} | converged "
                    f"{result['converged']['test']['scores']['mean']:.4f} (epoch {behaviour['converged_epoch']} of "
                    f"{behaviour['epochs_run']}, {behaviour['stopped_by']})")
                with open(summary_path + ".tmp", "w") as fh:
                    json.dump(summary, fh, indent=1, default=float)
                os.replace(summary_path + ".tmp", summary_path)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    return summary


# --------------------------------------------------------------------------- the pre-registered comparison

def analyse(out_dir, seeds=SEEDS, n_boot=N_BOOT, boot_seed=BOOT_SEED, *, models=MODELS, contrasts=None, primary=PRIMARY,
            result_name="ddr_probe_result.json", criterion=None, sources=None):
    """P, E1, E2 on the DDR test split for both probe variants; E1 - P, E2 - P, E1 - E2 per seed and for the
    three-seed mean, with a paired bootstrap over the test images (the same resamples for every model, seed and
    variant); and the §68 criterion on the five-epoch result: E1 - P positive in 3 / 3 seeds and the interval of
    the three-seed mean excluding zero.

    The keyword arguments exist for the same analysis on other encoders (ddr_probe_ep): `models` and `contrasts`
    name them, `primary` is the contrast the criterion is evaluated on, `sources` maps a model to the directory
    holding its saved test logits (default `out_dir`). The defaults are the recorded §68 analysis."""
    import e2_control as e2
    contrasts = CONTRASTS if contrasts is None else contrasts
    sources = sources or {}
    with np.load(os.path.join(out_dir, "targets.npz")) as data:
        targets, ids = data["test_targets"], [str(i) for i in data["test_ids"]]
    positive = targets >= 0.5
    n = len(ids)
    indices = np.random.default_rng(boot_seed).integers(0, n, size=(n_boot, n))
    weights = np.stack([np.bincount(idx, minlength=n) for idx in indices]).astype(np.float64)
    result = {"test_images": n, "n_boot": int(n_boot), "boot_seed": int(boot_seed), "bootstrap": "paired, over test images, unstratified",
              "variants": {}}
    for variant in VARIANTS:
        point, draws = {}, {}
        for seed in seeds:
            for model in models:
                with np.load(os.path.join(sources.get(model, out_dir), f"probe_{model}_seed{int(seed)}.npz")) as data:
                    if [str(i) for i in data["test_ids"]] != ids:
                        raise RuntimeError(f"{model}-{seed}: other test images")
                    z = data[f"test_logits_{variant}"].astype(np.float32)
                point[(model, int(seed))] = ep.probe_scores(targets, z)
                per_class = []
                for c in range(len(CLASSES)):
                    u, n_pos, n_neg = e2.pair_counts(positive[..., c], z[..., c])
                    per_class.append(np.einsum("bi,ij,bj->b", weights, u, weights) / ((weights @ n_pos) * (weights @ n_neg)))
                draws[(model, int(seed))] = np.stack(per_class, axis=1)                    # (n_boot, 4)
        interval = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]
        out = {"per_seed": {}, "mean": {}}
        for seed in seeds:
            seed = int(seed)
            row = {m: point[(m, seed)] for m in models}
            for name, (a, b) in contrasts.items():
                d = draws[(a, seed)] - draws[(b, seed)]
                row[name] = {**{k: point[(a, seed)][k] - point[(b, seed)][k] for k in point[(a, seed)]},
                             "mean_ci": interval(d.mean(axis=1)),
                             "class_ci": {c: interval(d[:, i]) for i, c in enumerate(CLASSES)}}
            out["per_seed"][seed] = row
        for m in models:
            out["mean"][m] = {k: float(np.mean([point[(m, int(s))][k] for s in seeds])) for k in point[(m, int(seeds[0]))]}
        for name, (a, b) in contrasts.items():
            d = np.mean(np.stack([draws[(a, int(s))] - draws[(b, int(s))] for s in seeds]), axis=0)
            per_seed = [out["per_seed"][int(s)][name]["mean"] for s in seeds]
            ci = interval(d.mean(axis=1))
            out["mean"][name] = {"mean": float(np.mean(per_seed)), "ci": ci, "per_seed": per_seed,
                                 "positive_seeds": int(sum(v > 0 for v in per_seed)),
                                 "ci_excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
                                 "per_class": {c: {"mean": float(np.mean([out["per_seed"][int(s)][name][c] for s in seeds])),
                                                   "ci": interval(d[:, i])} for i, c in enumerate(CLASSES)}}
        first = out["mean"][primary]
        out["criterion_met"] = bool(first["positive_seeds"] == len(seeds) and first["ci"][0] > 0)
        result["variants"][variant] = out
    result["primary_variant"] = "five_epoch"
    result["criterion"] = criterion or "E1 - P positive in 3/3 seeds AND the 95% paired bootstrap interval of the three-seed mean excludes zero"
    result["criterion_met"] = result["variants"]["five_epoch"]["criterion_met"]
    with open(os.path.join(out_dir, result_name), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    return result
