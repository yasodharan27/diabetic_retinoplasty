"""Geometry diagnostic v1 -- how sensitive are the frozen P graders to the framing of the input? (research record §89)

INFERENCE ONLY. The three pinned P checkpoints are run on the same development images twice: with the project's
frame (Stage 2, then a direct resize of the WHOLE image to 512 x 512) and with the same image cropped to the
bounding box of the fundus before the same resize. Nothing is trained, no weight changes, and no parameter of the
crop is chosen from a result. What this measures is the sensitivity of models trained on uncropped frames to input
geometry. It does NOT measure what retraining on geometry-normalised input would give.

Data: APTOS validation (730 images) and EyePACS adaptation-split validation (3,512 images) -- development data.
The EyePACS held-out test, IDRiD and DDR are not read: this file has no path to them.

Two commands:
  build     (laptop, CPU, no model)  both frames of every image from the RAW file in one pass, the crop boxes, and
                                     a check that the direct frame is the cached one.
  evaluate  (T4)                     parity of the direct arm against P's stored APTOS validation table, then the
                                     paired comparison and its report.
"""
import argparse
import datetime
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 123, 2026)
#: ---- the crop rule, fixed before any prediction exists (research record §89) ----
GRAY_THRESHOLD = 12          # a pixel is "lit" when its grey value (cv2 BGR2GRAY of the RAW image) exceeds this
LINE_FRACTION = 0.02         # a row / column belongs to the fundus when more than this fraction of its pixels is lit
MIN_SIDE_FRACTION = 0.10     # a box narrower than this fraction of the image side is treated as a failure: no crop
#: ---- the criteria, fixed before any prediction exists ----
PRIMARY_SET = "eyepacs_val"
MIN_EFFECT = 0.02            # supported: interval wholly above zero AND three-seed mean difference >= this
APTOS_GUARDRAIL = -0.01      # APTOS validation: three-seed mean difference must not be below this
SETS = ("aptos_val", "eyepacs_val")
ARMS = ("direct", "crop")
P_VALIDATION_TABLE = "experiments/PL_ConvNeXtPriors/2026-09-28_05-22-44/P_convnext_rgb/seed_{seed}/evaluation/per_sample_best.csv"
OUT_ROOT = "experiments/GeometryDiagnostic"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- the crop rule

def fundus_box(bgr):
    """(top, bottom, left, right) of the fundus in a RAW image, bottom / right exclusive -- or the whole image
    when no usable box is found. Deterministic; uses pixels only."""
    import cv2
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lit = gray > GRAY_THRESHOLD
    h, w = lit.shape
    rows = np.flatnonzero(lit.mean(axis=1) > LINE_FRACTION)
    cols = np.flatnonzero(lit.mean(axis=0) > LINE_FRACTION)
    if rows.size == 0 or cols.size == 0:
        return 0, h, 0, w
    top, bottom, left, right = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    if bottom - top < MIN_SIDE_FRACTION * h or right - left < MIN_SIDE_FRACTION * w:
        return 0, h, 0, w
    return top, bottom, left, right


def two_frames(raw_path):
    """(direct frame, cropped frame, box, native (h, w)) of one RAW image. Stage 2 runs ONCE on the whole native
    image; the two frames differ only in whether the Stage-2 image is cut to the box before the same resize."""
    import cv2

    import stage4_v2_aptos_cache as ac
    import stage4_v2_data as sd
    native = sd.stage2_rgb(raw_path)                                 # RGB uint8, native size (gamma, then CLAHE)
    bgr = cv2.imread(raw_path)
    if bgr is None or bgr.shape[:2] != native.shape[:2]:
        raise RuntimeError(f"{raw_path}: the raw image and its Stage-2 image differ in size")
    top, bottom, left, right = fundus_box(bgr)
    direct = ac.recompute_rgb_512(native)
    crop = direct if (top, bottom, left, right) == (0, native.shape[0], 0, native.shape[1]) \
        else ac.recompute_rgb_512(np.ascontiguousarray(native[top:bottom, left:right]))
    for frame in (direct, crop):
        if frame.shape != (512, 512, 3) or frame.dtype != np.float32 or frame.min() < 0.0 or frame.max() > 1.0:
            raise RuntimeError(f"{raw_path}: a frame is not float32 (512, 512, 3) in [0, 1]")
    return direct, crop, (top, bottom, left, right), native.shape[:2]


# --------------------------------------------------------------------------- development image sets

def aptos_validation(aptos_raw_dir):
    import pathology_grader_severity_fusion as sf
    ids, grades = sf.authoritative_validation_ids()
    return [{"image": i, "grade": int(g), "unit": i, "path": os.path.join(aptos_raw_dir, f"{i}.png")} for i, g in zip(ids, grades)]


def eyepacs_validation(eyepacs_train_dir):
    import eyepacs_adaptation_data as ead
    rows = ead.read_split(os.path.join(HERE, "dataset_splits"), ead.MANIFEST_SHA256)
    rows = rows[0] if isinstance(rows, tuple) else rows
    return [{"image": r["image"], "grade": int(r["grade"]), "unit": str(r["patient"]), "path": os.path.join(eyepacs_train_dir, r["image"])}
            for r in rows if r["split"] == "val"]


def build(out_dir, aptos_raw_dir, eyepacs_train_dir, eyepacs_cache_dir, workers=3, log=print):
    """Writes, per set: <set>_direct.npy and <set>_crop.npy (float16, the dtype of the project's caches) and
    <set>_index.json (order, grades, units, boxes, native sizes, source hashes). For EyePACS every direct frame is
    compared with the cached validation frame."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for name, rows in (("aptos_val", aptos_validation(aptos_raw_dir)), ("eyepacs_val", eyepacs_validation(eyepacs_train_dir))):
        cached = None
        if name == "eyepacs_val":
            index = json.load(open(os.path.join(eyepacs_cache_dir, "index.json")))
            where = {image: (shard, k) for shard, e in index["shards"].items() if e["split"] == "val" for k, image in enumerate(e["images"])}
            blocks = {shard: np.load(os.path.join(eyepacs_cache_dir, shard), mmap_mode="r") for shard in {s for s, _ in where.values()}}
            cached = lambda image: np.asarray(blocks[where[image][0]][where[image][1]])      # noqa: E731
        arrays = {arm: np.lib.format.open_memmap(os.path.join(out_dir, f"{name}_{arm}.npy"), mode="w+", dtype=np.float16, shape=(len(rows), 512, 512, 3))
                  for arm in ARMS}
        boxes, natives, hashes, cache_diff = [], [], [], []

        def work(row):
            return two_frames(row["path"]), sha256_file(row["path"])
        with ThreadPoolExecutor(workers) as pool:
            for k, ((direct, crop, box, native), digest) in enumerate(pool.map(work, rows)):
                arrays["direct"][k], arrays["crop"][k] = direct.astype(np.float16), crop.astype(np.float16)
                boxes.append(list(box)), natives.append(list(native)), hashes.append(digest)
                if cached is not None:
                    cache_diff.append(float(np.abs(arrays["direct"][k].astype(np.float32) - cached(rows[k]["image"]).astype(np.float32)).max()))
                if (k + 1) % 250 == 0 or k + 1 == len(rows):
                    log(f"  {name}: {k + 1}/{len(rows)}")
        for a in arrays.values():
            a.flush()
        del arrays
        b, n = np.asarray(boxes, np.float64), np.asarray(natives, np.float64)
        height, width = (b[:, 1] - b[:, 0]) / n[:, 0], (b[:, 3] - b[:, 2]) / n[:, 1]
        quantiles = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]    # noqa: E731
        summary[name] = {"images": len(rows), "grade_counts": np.bincount([r["grade"] for r in rows], minlength=5).tolist(),
                         "units": len({r["unit"] for r in rows}), "crop_is_identity": int(np.sum((height == 1.0) & (width == 1.0))),
                         "box_width_fraction_p10_50_90": quantiles(width), "box_height_fraction_p10_50_90": quantiles(height),
                         "direct_vs_cache_max_abs": (max(cache_diff) if cache_diff else None),
                         "direct_equal_to_cache": (int(np.sum(np.asarray(cache_diff) == 0.0)) if cache_diff else None)}
        with open(os.path.join(out_dir, f"{name}_index.json"), "w") as fh:
            json.dump({"set": name, "images": [r["image"] for r in rows], "grades": [r["grade"] for r in rows], "units": [r["unit"] for r in rows],
                       "boxes": boxes, "native_hw": natives, "source_sha256": hashes,
                       "rule": {"gray_threshold": GRAY_THRESHOLD, "line_fraction": LINE_FRACTION, "min_side_fraction": MIN_SIDE_FRACTION},
                       "sha256": {arm: sha256_file(os.path.join(out_dir, f"{name}_{arm}.npy")) for arm in ARMS}}, fh)
        log(json.dumps({name: summary[name]}))
    with open(os.path.join(out_dir, "geometry_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


# --------------------------------------------------------------------------- evaluation (T4)

def evaluate(drive_root, frames_dir, log=print):
    import pandas as pd

    import arch1_posthoc as ph
    import ddr_grading_eval as ev            # only its dataset-independent statistics (QWK difference, calibration, ...)
    import eyepacs_adaptation_train as eat
    import idrid_batch1_eval as b1
    import idrid_batch2_eval as b2
    import idrid_grading_eval as ig
    root = lambda rel: os.path.join(drive_root, *rel.split("/"))     # noqa: E731
    parent = ig.load_protocol()[0]
    settings = dict(parent["inference"])
    ig.prepare_gpu()
    files = {"checkpoints": {}, "kinds": {}, "lesion_prior": {}, "pretrained": root(parent["checkpoints"]["convnext_pretrained"]["path"])}
    for seed in SEEDS:
        entry = parent["checkpoints"][f"p_seed{seed}"]
        if ig.sha256_file(root(entry["path"])) != entry["sha256"]:
            raise RuntimeError(f"P seed {seed}: not the pinned checkpoint")
        files["checkpoints"][f"p_seed{seed}"], files["kinds"][f"p_seed{seed}"] = root(entry["path"]), "P"
    models = b2.load_models(files, settings, arms=("p",))
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(root(OUT_ROOT), f"geometry_diag_v1_{stamp}")
    os.makedirs(out_dir)
    tables, frames_tab, index = {}, {}, {}
    for name in SETS:
        index[name] = json.load(open(os.path.join(frames_dir, f"{name}_index.json")))
        ids, grades = index[name]["images"], np.asarray(index[name]["grades"])
        tables[name], frames_tab[name] = {s: {} for s in SEEDS}, {}
        for arm in ARMS:
            path = os.path.join(frames_dir, f"{name}_{arm}.npy")
            if sha256_file(path) != index[name]["sha256"][arm]:
                raise RuntimeError(f"{name} / {arm}: the frame file is not the one that was built")
            block = np.load(path, mmap_mode="r")
            logits = {s: [] for s in SEEDS}
            for start in range(0, len(ids), settings["batch_size"]):
                rgb = np.asarray(block[start:start + settings["batch_size"]], np.float32)
                for s in SEEDS:
                    logits[s].append(np.asarray(models[("p", s)].predict_on_batch(eat.p_inputs(rgb)), np.float64))
            for s in SEEDS:
                frame, tables[name][s][arm] = b1.table_from_logits(ids, grades, np.concatenate(logits[s], 0))
                frame.to_csv(os.path.join(out_dir, f"per_image_{name}_{arm}_p_seed{s}.csv"), index=False)
                frames_tab[name][(arm, s)] = frame
            log(f"  predicted {name} / {arm}")
    # the direct arm on APTOS validation must be P's stored validation prediction (the parity tolerances of the locked protocol)
    tolerance, parity = parent["parity"]["tolerances"], {}
    for s in SEEDS:
        stored = pd.read_csv(root(P_VALIDATION_TABLE.format(seed=s)), dtype={"image_id": str})
        mine = frames_tab["aptos_val"][("direct", s)]
        if list(stored["image_id"]) != list(mine["image_id"]):
            raise RuntimeError("the APTOS validation order is not the stored one")
        cols = [f"logit_{k}" for k in range(4)]
        parity[str(s)] = {"logit_max_abs": float(np.abs(stored[cols].to_numpy() - mine[cols].to_numpy()).max()),
                          "grades_equal": bool((stored["predicted_grade"].to_numpy() == mine["predicted_grade"].to_numpy()).all())}
    parity_ok = all(v["logit_max_abs"] <= tolerance["logit_max_abs"] and v["grades_equal"] for v in parity.values())
    result = {"protocol": "geometry diagnostic v1 (research record 89)", "timestamp_utc": stamp, "git_commit": ig.git_commit(),
              "script_sha256": sha256_file(os.path.abspath(__file__)), "environment": ig.environment(settings),
              "rule": index[PRIMARY_SET]["rule"], "parity_direct_arm_vs_stored_aptos_table": parity, "parity_pass": parity_ok, "sets": {}}
    if parity_ok:
        for name in SETS:
            grades, units = np.asarray(index[name]["grades"]), np.asarray(index[name]["units"])
            everything = np.ones(len(grades), bool)
            difference = ev.qwk_difference(tables[name], "crop", "direct", everything, ph.bootstrap_indices(grades))
            by_unit = ev.qwk_difference(tables[name], "crop", "direct", everything, ev.cluster_indices(units, np.zeros(len(units), int)))
            arms = {}
            for arm in ARMS:
                per_seed = [tables[name][s][arm] for s in SEEDS]
                arms[arm] = {"qwk_per_seed": [ph._fast_qwk(t["grade"], t["pred"]) for t in per_seed],
                             "recall_per_grade_mean": [float(np.mean([np.mean(t["pred"][t["grade"] == g] == g) for t in per_seed])) for g in range(5)],
                             "referable_dr_mean": {k: float(np.mean([ev.referable(t)[k] for t in per_seed])) for k in ("sensitivity", "specificity")},
                             "auroc_cut_mean": {f"ge{c}": float(np.mean([ph._auroc(t["grade"] >= c, t["p_gt"][:, c - 1]) for t in per_seed])) for c in ph.CUTS},
                             "calibration_mean": {k: float(np.mean([ev.calibration(frames_tab[name][(arm, s)])[k] for s in SEEDS]))
                                                  for k in ("ece_top_label", "nll", "mean_confidence", "top_label_accuracy")},
                             "decoded_grade_counts_per_seed": [np.bincount(t["pred"], minlength=5).tolist() for t in per_seed]}
            result["sets"][name] = {"images": int(len(grades)), "grade_counts": np.bincount(grades, minlength=5).tolist(),
                                    "units": int(len(set(units.tolist()))), "crop_minus_direct_qwk": difference,
                                    "crop_minus_direct_qwk_unit_bootstrap": by_unit, "arms": arms,
                                    "decoded_grades_differ_per_seed": [int((tables[name][s]["crop"]["pred"] != tables[name][s]["direct"]["pred"]).sum()) for s in SEEDS]}
        primary, guard = result["sets"][PRIMARY_SET]["crop_minus_direct_qwk"], result["sets"]["aptos_val"]["crop_minus_direct_qwk"]
        result["reading"] = {
            "primary_supported": bool(primary["ci_low"] > 0 and primary["mean"] >= MIN_EFFECT),
            "primary_rule": f"EyePACS validation: interval wholly above zero AND three-seed mean >= {MIN_EFFECT}",
            "aptos_guardrail_held": bool(guard["mean"] >= APTOS_GUARDRAIL),
            "aptos_guardrail_rule": f"APTOS validation: three-seed mean difference >= {APTOS_GUARDRAIL}",
            "scope": "inference only: this is the sensitivity of models trained on uncropped frames to input geometry. It is not the "
                     "benefit of retraining on geometry-normalised input, and it selects nothing."}
    with open(os.path.join(out_dir, "result.json"), "w") as fh:
        json.dump(ph._jsonable(result), fh, indent=1)
    log(json.dumps(ph._jsonable({k: v for k, v in result.items() if k != "environment"}), indent=1, default=float))
    log(f"written to {out_dir}" if parity_ok else "PARITY FAILED: no comparison was computed")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("build", "evaluate"))
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--aptos-raw-dir", default=os.path.join(HERE, "datasets", "APTOS2019", "raw", "train_images"))
    parser.add_argument("--eyepacs-train-dir", default=os.path.join(HERE, "datasets", "EyePACS", "raw", "train"))
    parser.add_argument("--eyepacs-cache-dir", default=os.path.join(HERE, "datasets", "EyePACS", "processed", "adaptation_frames_v1"))
    parser.add_argument("--drive-root", default=None)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args.frames_dir, args.aptos_raw_dir, args.eyepacs_train_dir, args.eyepacs_cache_dir, args.workers)
    else:
        evaluate(args.drive_root, args.frames_dir)


if __name__ == "__main__":
    main()
