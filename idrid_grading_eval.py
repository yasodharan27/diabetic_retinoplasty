"""One-time external evaluation of the frozen Stage 1-8 pipeline on the IDRiD Disease Grading Testing Set
(research record §60). Evaluation only: nothing is trained, tuned or selected here.

    raw image -> Stage 2 "DR" preprocessing (once) -> native RGB
        -> 512^2 RGB ------------------------------------------> P (ConvNeXt-Tiny, CORN), seeds 42 / 123 / 2026
        -> Stage 3 LWNet (TTA) -> 512^2 vessel map ----.
        -> Stage 4 v2 U-Net at 1536^2 -> 512^2 maps ----+------> pathology grader (CORN), seeds 42 / 123 / 2026
    -> locked severity-aware fusion of the two branches of the same seed -> grade

Everything that defines the run is in `idrid_grading_protocol.json` (dataset hashes, the three excluded
images, the eight model files with SHA-256 and size, the inference settings, the fusion rule, the parity
tolerances). This module refuses to run on anything that differs from it.

Two phases, in this order:
  1. `run_parity` -- the APTOS PARITY GATE. A fixed subset of the authoritative APTOS validation images is
     pushed through this exact code, from the raw image, and compared with the cached Stage-2 / 3 / 4 arrays
     and with the stored BEST prediction tables. It must pass on the machine that will run IDRiD.
  2. `run_idrid` -- needs a passed gate from the same environment, the confirmation token and a free lock.
     The 100-image primary result is computed and written first; the 103-image sensitivity result after it.

No IDRiD-specific preprocessing exists here: the image functions are the ones the APTOS pipeline used.
"""
import datetime
import hashlib
import json
import os
import platform
import posixpath

import numpy as np

import arch1_posthoc as ph
import pathology_grader_fusion as pf
import pathology_grader_severity_fusion as sf

SEEDS = (42, 123, 2026)
HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "idrid_grading_protocol.json")
CONFIRMATION_TOKEN = "RUN_THE_ONE_TIME_IDRID_GRADING_TEST"
MODELS = ("p", "pathology", "fused")


class ProtocolError(RuntimeError):
    pass


# --------------------------------------------------------------------------- protocol and files

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_protocol(path=PROTOCOL_PATH):
    with open(path, encoding="utf-8") as fh:
        protocol = json.load(fh)
    rule = protocol["fusion"]
    for key in ("fused_0", "fused_1", "fused_2", "fused_3", "decode"):
        if rule[key] != sf.RULE[key]:
            raise ProtocolError(f"protocol fusion {key!r} is not the implemented locked rule")
    if tuple(protocol["seeds"]) != SEEDS:
        raise ProtocolError("protocol seeds are not 42, 123, 2026")
    if len(protocol["dataset"]["primary_exclusions"]) != 3:
        raise ProtocolError("the primary exclusion list must be exactly the three predefined images")
    return protocol, sha256_file(path)


def checkpoint_path(drive_root, entry):
    return os.path.join(drive_root, *entry["path"].split("/"))


def verify_checkpoints(protocol, drive_root):
    """Every pinned model file exists with the pinned size and SHA-256; BEST pointers still name the pinned
    slot. Returns {name: local path}. Raises on the first mismatch -- nothing is substituted."""
    paths = {}
    for name, entry in protocol["checkpoints"].items():
        path = checkpoint_path(drive_root, entry)
        if not os.path.exists(path):
            raise ProtocolError(f"{name}: {path} not found")
        if os.path.getsize(path) != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"{name}: {path} is not the pinned file (size / SHA-256 mismatch)")
        if entry.get("best_pointer"):
            with open(os.path.join(drive_root, *entry["best_pointer"].split("/"))) as fh:
                pointer = json.load(fh)
            if pointer["active"] != entry["best_slot"] or int(pointer["epoch"]) != entry["best_epoch_index"]:
                raise ProtocolError(f"{name}: the run's BEST pointer no longer names the pinned checkpoint")
        paths[name] = path
    return paths


def environment(settings):
    import tensorflow as tf
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    import keras
    return {"device": settings["device"], "gpu": gpu, "stage4_amp": bool(settings["stage4_amp"]),
            "keras_policy": keras.mixed_precision.global_policy().name, "torch": torch.__version__,
            "tensorflow": tf.__version__, "keras": keras.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform()}


def git_commit():
    import arch1_train as at
    return at.git_commit(HERE)


# --------------------------------------------------------------------------- models (frozen, loaded once)

def prepare_gpu():
    """TensorFlow and PyTorch share the GPU here: turn TensorFlow's memory growth on before it touches the
    device (training.trainer.check_gpu, the project's helper), so it does not reserve the whole card."""
    from training.trainer import check_gpu
    return check_gpu()


def load_segmentation_models(paths, settings):
    import stage4_v2 as s4
    import pipeline_v2_config as v2cfg
    from vessel_segmentation_inference import load_vessel_model
    vessel = load_vessel_model(paths["stage3_lwnet"])
    stage4 = s4.load_stage4_v2(paths["stage4_unet"], expected_sha256=sha256_file(paths["stage4_unet"]),
                               classes=v2cfg.STAGE4_V2A_CLASSES).to(settings["device"])
    return vessel, stage4


def load_graders(paths, settings):
    """{'p': {seed: model}, 'pathology': {seed: model}} built under the protocol's dtype policy and loaded
    from the pinned BEST weights (weights only)."""
    import keras
    import pipeline_v2_config as v2cfg
    import pl_convnext as pl
    import pathology_grader_model as pm
    import stage34_cache_v2 as cache
    from training import checkpointing as ckpt
    keras.mixed_precision.set_global_policy("float32")
    _, reference_arrays = pl.load_reference(paths["convnext_pretrained"])       # arrays are taken in float32
    keras.mixed_precision.set_global_policy(settings["keras_policy"])
    channels = cache.channel_names(v2cfg.STAGE4_V2A_CLASSES)
    graders = {"p": {}, "pathology": {}}
    for seed in SEEDS:
        p = pl.build_pl_model("P", seed, reference_arrays)
        ckpt.load_model_weights_only(p, paths[f"p_seed{seed}"])
        graders["p"][seed] = p
        q = pm.build_pathology_grader(channels, seed)
        ckpt.load_model_weights_only(q, paths[f"pathology_seed{seed}"])
        graders["pathology"][seed] = q
    return graders


# --------------------------------------------------------------------------- image -> inputs -> logits

def image_inputs(raw_path, vessel_model, stage4_model, settings):
    """The frozen Stage 2 -> 3 -> 4 path for ONE raw image, with the functions the APTOS pipeline used:
    stage4_v2_data.stage2_rgb (DR profile, once), the canonical 512 resize, LWNet with TTA, and the Stage-4
    full-frame 1536 inference with exact 3x3 mean + max pooling to uint8."""
    import stage4_v2 as s4
    import stage4_v2_aptos_cache as ac
    import stage4_v2_data as sd
    native = sd.stage2_rgb(raw_path)
    return {"native_shape": list(native.shape), "rgb": ac.recompute_rgb_512(native),
            "vessel": ac.recompute_vessel_512(native, vessel_model)[..., None].astype(np.float32),
            "maps": s4.pathology_cache_maps(stage4_model, native, device=settings["device"],
                                            amp=bool(settings["stage4_amp"]))}, native


def predict_logits(graders, inputs, batch_size=8):
    """{model: {seed: (N, 4) float64 logits}} for a list of per-image inputs. P receives the RGB frame in
    channels 0-2 of its 8-channel input (it reads nothing else); the pathology grader receives the vessel
    map and the Stage-4 maps and no RGB."""
    import stage34_cache_v2 as cache
    out = {"p": {s: [] for s in SEEDS}, "pathology": {s: [] for s in SEEDS}}
    for start in range(0, len(inputs), batch_size):
        chunk = inputs[start:start + batch_size]
        rgb = np.stack([c["rgb"] for c in chunk]).astype(np.float32)
        x8 = np.concatenate([rgb, np.zeros(rgb.shape[:3] + (5,), np.float32)], axis=-1)
        aux = [np.zeros((len(chunk), 256, 256, 3), np.float32), np.zeros((len(chunk), 1), np.float32)]
        xq = {"vessel": np.stack([c["vessel"] for c in chunk]).astype(np.float32),
              "pathology": np.stack([cache.from_uint8(c["maps"]) for c in chunk]).astype(np.float32)}
        for seed in SEEDS:
            out["p"][seed].append(np.asarray(graders["p"][seed].predict_on_batch([x8] + aux), np.float64))
            out["pathology"][seed].append(np.asarray(graders["pathology"][seed].predict_on_batch(xq), np.float64))
    return {m: {s: np.concatenate(v, 0) for s, v in d.items()} for m, d in out.items()}


def tables_from_logits(ids, grades, logits):
    """{seed: {'p', 'pathology', 'fused'}} per-sample tables (multiseed schema) + the frames to save."""
    import pandas as pd

    import arch1_train as at
    tables, frames = {}, {}
    for seed in SEEDS:
        tables[seed], frames[seed] = {}, {}
        for model in ("p", "pathology"):
            frame = pd.DataFrame(at.metrics_from_logits(list(ids), np.asarray(grades), logits[model][seed])[1])
            frames[seed][model], tables[seed][model] = frame, ph.table_arrays(frame)
        fused = sf.fuse_severity(tables[seed]["p"], tables[seed]["pathology"], f"seed {seed}")
        tables[seed]["fused"] = fused
        frames[seed]["fused"] = pd.DataFrame({"image_id": fused["ids"], "true_grade": fused["grade"],
                                              "predicted_grade": fused["pred"],
                                              **{f"p_gt_{k}": fused["p_gt"][:, k] for k in range(4)}})
    return tables, frames


# --------------------------------------------------------------------------- phase 1: the APTOS parity gate

def parity_subset(ids, grades, per_grade):
    """The first `per_grade` validation images of every grade, in the authoritative order."""
    chosen = []
    for g in range(5):
        chosen += [i for i, y in zip(ids, grades) if y == g][:per_grade]
    order = {i: n for n, i in enumerate(ids)}
    return sorted(chosen, key=order.get)


def run_parity(drive_root, aptos_raw_dir, stored_root, out_dir, *, bundle=None, aptos_processed_dir=None,
               protocol_path=PROTOCOL_PATH, settings=None, official=True, log=print):
    """Pushes the parity subset through this module from the RAW images and compares every stage with what the
    project stored. `bundle`: the verified v2 bundle (arch1_data.Arch1Bundle) holding the cached arrays.
    `stored_root`: the experiments root with the stored BEST per-sample tables. Writes parity.json and
    returns it; `official=False` marks a pre-check on another machine (it can never unlock IDRiD)."""
    protocol, protocol_sha = load_protocol(protocol_path)
    settings = dict(settings or protocol["inference"])
    if official and settings != protocol["inference"]:
        raise ProtocolError("the official gate runs with the protocol's inference settings only")
    tol = protocol["parity"]["tolerances"]
    prepare_gpu()
    paths = verify_checkpoints(protocol, drive_root)
    val_ids, val_grades = sf.authoritative_validation_ids()
    subset = parity_subset(val_ids, val_grades, protocol["parity"]["images_per_grade"])
    grade_of = dict(zip(val_ids, val_grades))
    vessel_model, stage4_model = load_segmentation_models(paths, settings)
    inputs, stage_checks = [], []
    for image_id in subset:
        raw = os.path.join(aptos_raw_dir, f"{image_id}.png")
        item, native = image_inputs(raw, vessel_model, stage4_model, settings)
        row = {"image_id": image_id, "grade": int(grade_of[image_id])}
        processed = os.path.join(aptos_processed_dir, f"{image_id}.png") if aptos_processed_dir else None
        if processed and os.path.exists(processed):      # Stage 2 from raw == the stored Stage-2 file, exactly
            import stage4_v2_data as sd
            stored_native = sd.read_rgb(processed)
            row["stage2_exact"] = bool(stored_native.shape == native.shape and np.array_equal(stored_native, native))
        if bundle is not None:
            cached = bundle.load_sample(image_id)
            import stage34_cache_v2 as cache
            row.update(rgb_max_abs=float(np.abs(item["rgb"] - cached["rgb"]).max()),
                       vessel_max_abs=float(np.abs(item["vessel"] - cached["vessel"]).max()),
                       stage4_max_counts=int(np.abs(item["maps"].astype(np.int16)
                                                    - np.round(cached["pathology"] * 255).astype(np.int16)).max()),
                       stage4_mean_counts=float(np.abs(cache.from_uint8(item["maps"]) - cached["pathology"]).mean() * 255))
        stage_checks.append(row)
        inputs.append(item)
        log(f"  parity inputs {image_id} (grade {row['grade']}): " + ", ".join(f"{k} {v:.3g}" for k, v in row.items()
                                                                                if k not in ("image_id", "grade", "stage2_exact"))
            + (f", stage2_exact {row['stage2_exact']}" if "stage2_exact" in row else ""))
    del vessel_model, stage4_model
    graders = load_graders(paths, settings)
    logits = predict_logits(graders, inputs)
    tables, _ = tables_from_logits(subset, [grade_of[i] for i in subset], logits)
    index = {i: n for n, i in enumerate(val_ids)}
    rows = [index[i] for i in subset]
    prediction_checks = {}
    for seed in SEEDS:
        stored = {"p": ph._read_table(pf.p_prediction_path(stored_root, seed, "best")),
                  "pathology": ph._read_table(sf.pathology_path(stored_root, seed))}
        import pandas as pd
        stored_logits = {"p": pd.read_csv(pf.p_prediction_path(stored_root, seed, "best"))[[f"logit_{k}" for k in range(4)]].to_numpy(),
                         "pathology": pd.read_csv(sf.pathology_path(stored_root, seed))[[f"logit_{k}" for k in range(4)]].to_numpy()}
        stored_fused = sf.fuse_severity(stored["p"], stored["pathology"])
        check = {}
        for model in ("p", "pathology"):
            if list(stored[model]["ids"][rows]) != subset:
                raise ProtocolError(f"{model}-{seed}: stored table is not in the authoritative order")
            check[model] = {"logit_max_abs": float(np.abs(logits[model][seed] - stored_logits[model][rows]).max()),
                            "p_max_abs": float(np.abs(tables[seed][model]["p_gt"] - stored[model]["p_gt"][rows]).max()),
                            "grades_equal": bool(np.array_equal(tables[seed][model]["pred"], stored[model]["pred"][rows]))}
        check["fused"] = {"p_max_abs": float(np.abs(tables[seed]["fused"]["p_gt"] - stored_fused["p_gt"][rows]).max()),
                          "grades_equal": bool(np.array_equal(tables[seed]["fused"]["pred"], stored_fused["pred"][rows]))}
        prediction_checks[seed] = check
    failures = []
    for row in stage_checks:
        if row.get("stage2_exact") is False:
            failures.append(f"{row['image_id']}: Stage 2 recomputed from the raw image differs from the stored Stage-2 file")
        if "rgb_max_abs" in row:
            if row["rgb_max_abs"] > tol["rgb_max_abs"]:
                failures.append(f"{row['image_id']}: RGB frame differs from the cache by {row['rgb_max_abs']:.3g}")
            if row["vessel_max_abs"] > tol["vessel_max_abs"]:
                failures.append(f"{row['image_id']}: vessel map differs from the cache by {row['vessel_max_abs']:.3g}")
            if row["stage4_max_counts"] > tol["stage4_max_counts"]:
                failures.append(f"{row['image_id']}: Stage-4 maps differ from the cache by {row['stage4_max_counts']} counts")
    if official and bundle is None:
        failures.append("the official gate needs the cached arrays (bundle) to compare against")
    for seed, check in prediction_checks.items():
        for model in ("p", "pathology"):
            if check[model]["logit_max_abs"] > tol["logit_max_abs"]:
                failures.append(f"{model}-{seed}: logits differ by {check[model]['logit_max_abs']:.3g}")
        for model in MODELS:
            if check[model]["p_max_abs"] > tol["probability_max_abs"]:
                failures.append(f"{model}-{seed}: cumulative probabilities differ by {check[model]['p_max_abs']:.3g}")
            if not check[model]["grades_equal"]:
                failures.append(f"{model}-{seed}: decoded grades differ")
    result = {"kind": "official gate" if official else "pre-check (cannot unlock IDRiD)", "official": bool(official),
              "PASS": not failures, "failures": failures, "subset": subset,
              "subset_grades": [int(grade_of[i]) for i in subset], "tolerances": tol, "stage_checks": stage_checks,
              "prediction_checks": {str(s): c for s, c in prediction_checks.items()},
              "settings": settings, "environment": environment(settings), "protocol_sha256": protocol_sha,
              "git_commit": git_commit(), "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "parity.json"), "w") as fh:
        json.dump(ph._jsonable(result), fh, indent=1)
    log(f"APTOS PARITY {'PASS' if result['PASS'] else 'FAIL'} ({result['kind']}): {len(subset)} images, grades "
        f"{result['subset_grades']}" + ("" if result["PASS"] else " | " + "; ".join(failures)))
    return result


# --------------------------------------------------------------------------- phase 2: IDRiD, once

def read_idrid(idrid_raw_dir, protocol):
    """The 103 test images and labels, verified against the pinned hashes. Raw files only."""
    import csv
    data = protocol["dataset"]
    if "processed" in idrid_raw_dir.replace("\\", "/").lower().split("/"):
        raise ProtocolError("the RAW IDRiD tree must be used: the processed tree would apply Stage 2 twice")
    image_dir = os.path.join(idrid_raw_dir, *data["image_dir"].split("/"))
    label_file = os.path.join(idrid_raw_dir, *data["label_file"].split("/"))
    names = sorted(n for n in os.listdir(image_dir) if n.lower().endswith(".jpg"))
    if len(names) != data["n_images"]:
        raise ProtocolError(f"expected {data['n_images']} test images, found {len(names)}")
    digest, hashes = hashlib.sha256(), {}
    for name in names:
        hashes[name] = sha256_file(os.path.join(image_dir, name))
        digest.update(f"{name}:{hashes[name]}\n".encode())
    if digest.hexdigest() != data["images_manifest_sha256"]:
        raise ProtocolError("the IDRiD test images are not the pinned files")
    if sha256_file(label_file) != data["label_file_sha256"]:
        raise ProtocolError("the IDRiD label file is not the pinned file")
    with open(label_file, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    grade = {r["Image name"].strip(): int(r["Retinopathy grade"]) for r in rows}
    ids = [os.path.splitext(n)[0] for n in names]
    if set(ids) != set(grade):
        raise ProtocolError("image files and label rows do not match")
    grades = np.array([grade[i] for i in ids])
    if np.bincount(grades, minlength=5).tolist() != data["grade_counts"]:
        raise ProtocolError("the grade distribution is not the recorded one")
    missing = [x for x in data["primary_exclusions"] if x not in ids]
    if missing:
        raise ProtocolError(f"excluded images not in the test set: {missing}")
    return {"ids": ids, "grades": grades, "paths": [os.path.join(image_dir, n) for n in names], "hashes": hashes,
            "primary": np.array([i not in set(data["primary_exclusions"]) for i in ids])}


def subset_table(table, mask):
    return {k: v[mask] for k, v in table.items()}


def summarise(tables, mask, n_boot=ph.N_BOOT):
    """Per-seed metrics for P, pathology and the fused pipeline on the masked images, mean +/- SD over the
    three seeds (no ensemble, no seed selection), and grade-stratified bootstrap intervals for the seed mean."""
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        for model in MODELS:
            m = ph.metrics(subset_table(tables[seed][model], mask))
            per_seed[seed][model] = {**pf.headline(m), "recall_per_grade": m["recall_per_grade"],
                                     "confusion_matrix": m["confusion_matrix"]}
    keys = pf.HEADLINE_KEYS
    aggregate = {}
    for model in MODELS:
        aggregate[model] = {}
        for k in keys:
            x = np.array([per_seed[s][model][k] for s in SEEDS], np.float64)
            aggregate[model][k] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "per_seed": [float(v) for v in x]}
        for g in range(5):
            x = np.array([per_seed[s][model]["recall_per_grade"][g] for s in SEEDS], np.float64)
            aggregate[model][f"recall_grade{g}"] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)),
                                                    "per_seed": [float(v) for v in x]}
    grades = subset_table(tables[SEEDS[0]]["fused"], mask)["grade"]
    indices = ph.bootstrap_indices(grades, n_boot)
    intervals = {}
    for model in MODELS:
        draws = {k: [] for k in ph.BOOT_KEYS}
        for idx in indices:
            vals = [ph.bootstrap_metrics(subset_table(tables[s][model], mask), idx) for s in SEEDS]
            for k in ph.BOOT_KEYS:
                draws[k].append(np.mean([v[k] for v in vals]))
        intervals[model] = {k: [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))] for k, d in draws.items()}
    reference = {}
    for name, other in (("fused_minus_p", "p"), ("fused_minus_pathology", "pathology")):
        point, draws = {k: [] for k in ph.METRICS}, {k: [] for k in ph.BOOT_KEYS}
        for s in SEEDS:
            d = ph.paired_delta(subset_table(tables[s]["fused"], mask), subset_table(tables[s][other], mask), indices=indices)
            for k in ph.METRICS:
                point[k].append(d[k]["delta"])
            for k in ph.BOOT_KEYS:
                draws[k].append(d[k]["_draws"])
        reference[name] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "per_seed": [float(x) for x in v]}
                           for k, v in point.items()}
        for k in ph.BOOT_KEYS:
            lo, hi = np.nanpercentile(np.mean(np.stack(draws[k], 0), 0), [2.5, 97.5])
            reference[name][k].update(ci_low=float(lo), ci_high=float(hi))
    return {"n": int(mask.sum()), "grade_counts": np.bincount(grades, minlength=5).tolist(), "per_seed": per_seed,
            "aggregate": aggregate, "seed_mean_interval_95": intervals, "reference_differences": reference,
            "n_boot": int(n_boot), "boot_seed": ph.BOOT_SEED}


def _f(v, signed=False):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.4f}" if signed else f"{v:.4f}")


def report_markdown(title, note, summary, protocol):
    labels = {"fused": "FINAL PIPELINE (locked severity-aware fusion)", "p": "reference: P alone",
              "pathology": "reference: pathology grader alone"}
    rows = (("qwk", "QWK"), ("auroc_ge1", "AUROC ≥1"), ("auroc_ge2", "AUROC ≥2"), ("auroc_ge3", "AUROC ≥3"),
            ("auroc_ge4", "AUROC ≥4"), ("recall_grade0", "recall, grade 0"), ("recall_grade1", "recall, grade 1"),
            ("recall_grade2", "recall, grade 2"), ("recall_grade3", "recall, grade 3"), ("recall_grade4", "recall, grade 4"),
            ("false_urgent_rate", "false-urgent rate"))
    lines = [f"# {title}", "", note, "",
             f"Images: {summary['n']} (grades 0-4: {summary['grade_counts']}). Seeds 42, 123, 2026; mean ± SD over seeds; no "
             "ensemble, no seed selection. P and the pathology grader are reference rows only.",
             "Fusion: fused_k = 0.5·P_k + 0.5·Path_k (k = 0, 1, 2); fused_3 = min(P_3, fused_2); grade = #{k: fused_k > 0.5}.", ""]
    for model in ("fused", "p", "pathology"):
        lines += [f"## {labels[model]}", "", "| | seed 42 | seed 123 | seed 2026 | mean | SD |", "|---|---|---|---|---|---|"]
        for key, name in rows:
            st = summary["aggregate"][model][key]
            lines.append(f"| {name} | " + " | ".join(_f(v) for v in st["per_seed"]) + f" | {_f(st['mean'])} | {_f(st['sd'])} |")
        ci = summary["seed_mean_interval_95"][model]
        lines += ["", f"95 % interval of the seed mean (bootstrap over images): QWK {_f(ci['qwk'][0])} to {_f(ci['qwk'][1])}; "
                      f"mean AUROC over the four cuts {_f(ci['mean_cut_auroc'][0])} to {_f(ci['mean_cut_auroc'][1])}.", ""]
        for seed in SEEDS:
            lines.append(f"Confusion matrix, seed {seed} (rows = true grade): {summary['per_seed'][seed][model]['confusion_matrix']}")
        lines.append("")
    d = summary["reference_differences"]["fused_minus_p"]
    lines += ["## Final pipeline minus P, same seed (reference only)", "",
              f"QWK {_f(d['qwk']['mean'], True)} ({_f(d['qwk']['ci_low'], True)} to {_f(d['qwk']['ci_high'], True)}); mean AUROC "
              f"over the four cuts {_f(d['mean_cut_auroc']['mean'], True)} ({_f(d['mean_cut_auroc']['ci_low'], True)} to "
              f"{_f(d['mean_cut_auroc']['ci_high'], True)}); grade-3 recall {_f(d['grade3_recall']['mean'], True)}; false-urgent "
              f"{_f(d['false_urgent_rate']['mean'], True)}.", ""]
    return "\n".join(lines) + "\n"


def lock_path(out_root, protocol_sha):
    return os.path.join(out_root, f"idrid_grading_lock_{protocol_sha[:12]}.json")


def require_gate(parity, protocol, protocol_sha, settings):
    """The gate must be an official PASS from this environment, this protocol and this code."""
    if not parity.get("official") or not parity.get("PASS"):
        raise ProtocolError("the APTOS parity gate has not passed (an official PASS is required)")
    if parity["protocol_sha256"] != protocol_sha or parity["settings"] != protocol["inference"]:
        raise ProtocolError("the parity gate was run with another protocol or other settings")
    if parity["git_commit"] != git_commit():
        raise ProtocolError("the parity gate was run on another commit")
    env = environment(settings)
    for key in ("gpu", "torch", "tensorflow", "keras", "keras_policy"):
        if parity["environment"].get(key) != env[key]:
            raise ProtocolError(f"the parity gate was run in another environment ({key})")
    return True


def run_idrid(drive_root, idrid_raw_dir, out_root, parity, *, confirm, technical_rerun_reason=None,
              protocol_path=PROTOCOL_PATH, command=None, log=print):
    """THE ONE-TIME EVALUATION. Refuses without the token, without an official parity PASS from this
    environment, or when the lock exists (a rerun needs a written technical reason and is recorded)."""
    protocol, protocol_sha = load_protocol(protocol_path)
    settings = dict(protocol["inference"])
    if confirm != CONFIRMATION_TOKEN:
        raise ProtocolError("the IDRiD grading test runs once: pass confirm=CONFIRMATION_TOKEN to run it")
    require_gate(parity, protocol, protocol_sha, settings)
    prepare_gpu()
    os.makedirs(out_root, exist_ok=True)
    lock = lock_path(out_root, protocol_sha)
    previous = None
    if os.path.exists(lock):
        with open(lock) as fh:
            previous = json.load(fh)
        if not (isinstance(technical_rerun_reason, str) and technical_rerun_reason.strip()):
            raise ProtocolError(f"the IDRiD grading test has already been run ({previous['timestamp_utc']}); a rerun is "
                                "allowed only for a technical failure, with technical_rerun_reason stated")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_root, f"idrid_grading_{protocol_sha[:12]}_{stamp}")
    paths = verify_checkpoints(protocol, drive_root)
    data = read_idrid(idrid_raw_dir, protocol)
    os.makedirs(out_dir)
    record = {"timestamp_utc": stamp, "protocol_sha256": protocol_sha, "git_commit": git_commit(), "out_dir": out_dir,
              "technical_rerun_reason": technical_rerun_reason, "previous_run": previous}
    with open(lock, "w") as fh:                          # written BEFORE any prediction exists
        json.dump(record, fh, indent=1)
    vessel_model, stage4_model = load_segmentation_models(paths, settings)
    inputs = []
    for n, path in enumerate(data["paths"], start=1):
        item, _ = image_inputs(path, vessel_model, stage4_model, settings)
        inputs.append(item)
        if n % 20 == 0 or n == len(data["paths"]):
            log(f"  inputs {n}/{len(data['paths'])}")
    del vessel_model, stage4_model
    graders = load_graders(paths, settings)
    tables, frames = tables_from_logits(data["ids"], data["grades"], predict_logits(graders, inputs))
    for seed in SEEDS:
        for model in MODELS:
            frame = frames[seed][model].copy()
            frame["in_primary_set"] = data["primary"]
            frame.to_csv(os.path.join(out_dir, f"per_image_{model}_seed{seed}.csv"), index=False)
    common = {"protocol": protocol, "protocol_sha256": protocol_sha, "environment": environment(settings),
              "git_commit": git_commit(), "command": command, "timestamp_utc": stamp, "parity": parity,
              "checkpoints_verified": {k: protocol["checkpoints"][k]["sha256"] for k in paths},
              "image_hashes": data["hashes"], "excluded_from_primary": protocol["dataset"]["primary_exclusions"],
              "native_shapes": {i: x["native_shape"] for i, x in zip(data["ids"], inputs)}}
    with open(os.path.join(out_dir, "run_configuration.json"), "w") as fh:
        json.dump(ph._jsonable(common), fh, indent=1)
    primary = summarise(tables, data["primary"])         # the primary result is computed and written first
    with open(os.path.join(out_dir, "primary_results_100.json"), "w") as fh:
        json.dump(ph._jsonable(primary), fh, indent=1)
    with open(os.path.join(out_dir, "primary_report_100.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown("IDRiD grading test -- PRIMARY RESULT (100 images)",
                                 "The three test images that are copies of Stage-4 v2 training images (IDRiD_088, IDRiD_089, "
                                 "IDRiD_091) are excluded. Evaluation only; run once.", primary, protocol))
    sensitivity = summarise(tables, np.ones(len(data["ids"]), bool))
    with open(os.path.join(out_dir, "sensitivity_results_103.json"), "w") as fh:
        json.dump(ph._jsonable(sensitivity), fh, indent=1)
    with open(os.path.join(out_dir, "sensitivity_report_103.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown("IDRiD grading test -- SENSITIVITY ANALYSIS ONLY (103 images)",
                                 "Includes three images that are copies of Stage-4 v2 training images. This is NOT the primary "
                                 "result and never replaces it.", sensitivity, protocol))
    log(f"IDRiD grading test written: {out_dir}")
    return {"out_dir": out_dir, "primary": primary, "sensitivity": sensitivity}
