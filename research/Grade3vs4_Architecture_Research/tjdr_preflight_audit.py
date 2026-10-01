"""TJDR pre-training audit (research record §40 step 2 / §42). Read-only: no training, no caches.

Inputs (read-only):
  datasets/TJDR/raw/{train,test}/{image,annotation}/*.png   (downloaded from the authors' public
      Google Drive folder linked from github.com/NekoPii/TJDR)
  datasets/APTOS2019/raw/{train_images,test_images}, datasets/IDRiD/grading/raw,
  datasets/IDRiD/segmentation/raw

Outputs: datasets/TJDR/audit/  (inventory.csv, components.csv, hashes.npz, overlap.json,
  summary.json, overlays/*.jpg)

Survival rule measured = the CURRENT Step-1 rule, unchanged:
  native binary mask -> stage4_v2.resize_mask_full_frame(mask, 1536, threshold=0.5)
  -> stage34_cache_v2.pack_pathology_maps (exact 3x3 block mean + max) -> 512 uint8.
Usage:  .venv/Scripts/python research/Grade3vs4_Architecture_Research/tjdr_preflight_audit.py
"""
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO)

DS = os.path.join(REPO, "datasets")
TJDR_RAW = os.path.join(DS, "TJDR", "raw")
OUT = os.path.join(DS, "TJDR", "audit")
SIZE = 1536
FOV_LEVEL = 10                              # the Step-1 fov_mask threshold (10/255)
TJDR_CODES = {"EX": 1, "HE": 2, "MA": 3, "SE": 4}      # paper: EX(1), HE(2), MA(3), SE(4)
CLASSES = ("MA", "HE", "EX", "SE")
IDRID_DIRS = {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates",
              "SE": "4. Soft Exudates"}
IDRID_SUFFIX = {"MA": "MA", "HE": "HE", "EX": "EX", "SE": "SE"}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- masks / survival

def _soft_resize(mask):
    """The exact pre-threshold operation inside stage4_v2.resize_mask_full_frame."""
    from skimage.transform import resize as sk_resize
    return sk_resize(mask.astype(np.float32), (SIZE, SIZE), order=1, mode="reflect",
                     anti_aliasing=True, preserve_range=True)


def component_survival(mask, dataset, group, image_id, cls):
    """Per native 8-connected component: size, whether it survives the current rule at 1536
    (any positive target pixel in its footprint), in the 512 max channel, and its 512 mean-channel
    value. Returns (records, pixel summary)."""
    from scipy import ndimage

    import stage34_cache_v2 as cache
    h, w = mask.shape
    soft = _soft_resize(mask)
    hard = (soft >= 0.5).astype(np.uint8)                       # current rule, threshold 0.5
    packed = cache.pack_pathology_maps(hard[..., None].astype(np.float32))   # (512, 512, 2)
    labels, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    records = []
    sy, sx = SIZE / h, SIZE / w
    for idx, sl in enumerate(ndimage.find_objects(labels), start=1):
        ys, xs = np.nonzero(labels[sl] == idx)
        ys = ys + sl[0].start
        xs = xs + sl[1].start
        area = ys.size
        fy = np.minimum(((ys + 0.5) * sy).astype(int), SIZE - 1)
        fx = np.minimum(((xs + 0.5) * sx).astype(int), SIZE - 1)
        py, px = fy // 3, fx // 3
        records.append({
            "dataset": dataset, "group": group, "image": image_id, "cls": cls, "area_native": int(area),
            "eqdiam_native": float(2 * np.sqrt(area / np.pi)),
            "eqdiam_1536": float(2 * np.sqrt(area * sy * sx / np.pi)),
            "max_soft_1536": float(soft[fy, fx].max()),
            "survives_1536": bool(hard[fy, fx].any()),
            "survives_512_max": bool((packed[py, px, 1] == 255).any()),
            "mean512_u8_max": int(packed[py, px, 0].max()),
        })
    return records, {"pixels_native": int(mask.sum()), "pixels_1536": int(hard.sum()),
                     "pixels_1536_expected": float(mask.sum() * sy * sx), "components": int(n)}


def _fov_stats(rgb):
    fov = rgb.max(axis=-1) > FOV_LEVEL
    ys, xs = np.nonzero(fov)
    corners = [rgb[:20, :20], rgb[:20, -20:], rgb[-20:, :20], rgb[-20:, -20:]]
    return fov, {"fov_fraction": float(fov.mean()),
                 "fov_bbox": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if ys.size else None,
                 "fov_eqdiam": float(2 * np.sqrt(fov.sum() / np.pi)),
                 "corner_max": int(max(c.max() for c in corners))}


def tjdr_worker(args):
    split, stem = args
    from PIL import Image
    img_path = f"{TJDR_RAW}/{split}/image/{stem}.png"
    ann_path = f"{TJDR_RAW}/{split}/annotation/{stem}.png"
    row = {"split": split, "image": stem, "image_sha256": None, "ok": False, "error": None}
    try:
        row["image_sha256"] = sha256_file(img_path)
        with Image.open(img_path) as im:
            row["image_mode"], row["image_size"] = im.mode, list(im.size)
            rgb = np.asarray(im.convert("RGB"))
        if not os.path.exists(ann_path):
            row["error"] = "missing annotation"
            return row, []
        row["mask_sha256"] = sha256_file(ann_path)
        with Image.open(ann_path) as mk:
            row["mask_mode"], row["mask_size"] = mk.mode, list(mk.size)
            row["mask_palette16"] = (mk.getpalette() or [])[:15]
            mask = np.asarray(mk)
        if mask.ndim != 2:
            row["error"] = f"mask ndim {mask.ndim}"
            return row, []
        values, counts = np.unique(mask, return_counts=True)
        row["mask_values"] = {int(v): int(c) for v, c in zip(values, counts)}
        row["size_match"] = tuple(mask.shape) == rgb.shape[:2]
        fov, fstats = _fov_stats(rgb)
        row.update(fstats)
        group = {2048: "TJDR-TRC50DX", 3912: "TJDR-CLARUS500"}.get(rgb.shape[1], f"TJDR-{rgb.shape[1]}")
        row["group"] = group
        records = []
        for cls in CLASSES:
            m = mask == TJDR_CODES[cls]
            row[f"{cls}_present"] = bool(m.any())
            row[f"{cls}_outside_fov_px"] = int((m & ~fov).sum())
            row[f"{cls}_rgb_sum"] = rgb[m].astype(np.float64).sum(axis=0).tolist() if m.any() else [0, 0, 0]
            if m.any():
                recs, pix = component_survival(m, "TJDR-" + split, group, stem, cls)
                records += recs
                row.update({f"{cls}_{k}": v for k, v in pix.items()})
        row["ok"] = True
        return row, records
    except Exception as exc:  # noqa: BLE001 -- recorded per file
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row, []


def idrid_worker(args):
    split, stem = args
    from PIL import Image
    root = f"{DS}/IDRiD/segmentation/raw/2. All Segmentation Groundtruths/{split}"
    records, row = [], {"split": split, "image": stem}
    for cls in CLASSES:
        path = f"{root}/{IDRID_DIRS[cls]}/{stem}_{IDRID_SUFFIX[cls]}.tif"
        if not os.path.exists(path):
            row[f"{cls}_present"] = False
            continue
        with Image.open(path) as mk:
            m = np.asarray(mk.convert("L")) > 0
        row[f"{cls}_present"] = bool(m.any())
        if m.any():
            recs, pix = component_survival(m, "IDRiD-seg-" + split[:1], "IDRiD", stem, cls)
            records += recs
            row.update({f"{cls}_{k}": v for k, v in pix.items()})
    return row, records


# ----------------------------------------------------------------------------- hashing

def _phash(gray32):
    from scipy.fft import dctn
    d = dctn(gray32.astype(np.float64), norm="ortho")[:8, :8].reshape(-1)
    return np.packbits(d > np.median(d[1:]))          # 8 bytes


def image_fingerprint(path, transform=None):
    """File sha256 + pixel sha256 + pHash of the FOV-cropped grey image (and of its mirror) +
    a 16x16 thumbnail for confirmation + FOV geometry."""
    from PIL import Image
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    if transform is not None:
        rgb = transform(rgb)
    fov = rgb.max(axis=-1) > FOV_LEVEL
    ys, xs = np.nonzero(fov)
    crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if ys.size else rgb
    gray = Image.fromarray(crop).convert("L")
    g32 = np.asarray(gray.resize((32, 32), Image.BILINEAR, reducing_gap=3.0), dtype=np.float32)
    g16 = np.asarray(gray.resize((16, 16), Image.BILINEAR, reducing_gap=3.0), dtype=np.float32).reshape(-1)
    g16 = (g16 - g16.mean()) / (g16.std() + 1e-6)
    return {"sha256": sha256_file(path) if transform is None else None,
            "pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "phash": _phash(g32), "phash_mirror": _phash(g32[:, ::-1]), "thumb": g16,
            "size": list(rgb.shape[:2]),
            "fov_bbox": [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if ys.size else None}


def fingerprint_worker(item):
    name, path = item
    try:
        return name, path, image_fingerprint(path), None
    except Exception as exc:  # noqa: BLE001
        return name, path, None, f"{type(exc).__name__}: {exc}"


def _control_transform(rgb):
    """Positive control: Stage-2 DR profile + downscale to 1024 + JPEG q=85 round trip."""
    import cv2
    from image_preprocessing import preprocess_array
    bgr = preprocess_array(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), profile="DR")
    h, w = bgr.shape[:2]
    bgr = cv2.resize(bgr, (1024, int(round(1024 * h / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def control_worker(path):
    return path, image_fingerprint(path), image_fingerprint(path, _control_transform)


def hamming(a, b):
    """a (n, 8) uint8, b (m, 8) uint8 -> (n, m) Hamming distances."""
    pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = np.empty((a.shape[0], b.shape[0]), dtype=np.int32)
    for s in range(0, a.shape[0], 128):
        x = np.bitwise_xor(a[s:s + 128, None, :], b[None, :, :])
        out[s:s + 128] = pop[x].sum(axis=-1, dtype=np.int32)
    return out


# ----------------------------------------------------------------------------- stage-2 near duplicates
# pHash on whole fundus images is not discriminative (unrelated APTOS pairs collide: see overlap.json
# APTOS_train_internal_pairs). Stage 2 compares the high-pass texture (vessels, lesions) of the
# FOV-cropped green channel at 128x128, mirror-aware, by normalised correlation.

HP_SIDE = 128


def hp_descriptor(rgb):
    from PIL import Image
    from scipy import ndimage
    fov = rgb.max(axis=-1) > FOV_LEVEL
    ys, xs = np.nonzero(fov)
    crop = rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1, 1] if ys.size else rgb[..., 1]
    g = np.asarray(Image.fromarray(crop).resize((HP_SIDE, HP_SIDE), Image.BOX), dtype=np.float32)
    hp = g - ndimage.gaussian_filter(g, 3.0)
    yy, xx = np.mgrid[:HP_SIDE, :HP_SIDE]
    inner = (yy - HP_SIDE / 2 + .5) ** 2 + (xx - HP_SIDE / 2 + .5) ** 2 <= (0.45 * HP_SIDE) ** 2
    v = np.where(inner, hp, 0.0)
    v = (v - v[inner].mean()) * inner
    v = v / (np.sqrt((v ** 2).sum()) + 1e-6)
    return v.astype(np.float32)


def hp_worker(item):
    name, path = item
    from PIL import Image
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"))
        v = hp_descriptor(rgb)
        return name, v.reshape(-1), v[:, ::-1].reshape(-1), None
    except Exception as exc:  # noqa: BLE001
        return name, None, None, f"{type(exc).__name__}: {exc}"


def hp_control_worker(path):
    from PIL import Image
    with Image.open(path) as im:
        rgb = np.asarray(im.convert("RGB"))
    return path, float(hp_descriptor(rgb).reshape(-1) @ hp_descriptor(_control_transform(rgb)).reshape(-1))


def overlap_stage2(sets):
    t0 = time.time()
    desc = {}
    with cf.ProcessPoolExecutor(max_workers=6) as pool:
        for key, items in sets.items():
            res = list(pool.map(hp_worker, items, chunksize=8))
            desc[key] = ([r[0] for r in res if r[1] is not None], np.stack([r[1] for r in res if r[1] is not None]),
                         np.stack([r[2] for r in res if r[1] is not None]), [r for r in res if r[1] is None])
            print(f"hp descriptors {key}: {len(desc[key][0])} {time.time() - t0:.0f}s", flush=True)
        rng = np.random.default_rng(1)
        ctrl = list(pool.map(hp_control_worker,
                             [sets["TJDR"][i][1] for i in rng.choice(len(sets["TJDR"]), 24, replace=False)]))
    tn, tv, tm, _ = desc["TJDR"]
    out = {"positive_control_corr": [round(c, 4) for _, c in ctrl]}
    for key in sets:
        on, ov, _, err = desc[key]
        corr = np.maximum(tv @ ov.T, tm @ ov.T)
        if key == "TJDR":
            np.fill_diagonal(corr, -1)
        best = corr.max(axis=1)
        pairs = np.argwhere(corr >= 0.5)
        out[key] = {"n": len(on), "errors": err, "max_corr": float(corr.max()),
                    "best_corr_quantiles_per_tjdr": np.quantile(best, [.5, .9, .99, 1]).round(4).tolist(),
                    "pairs_corr>=0.5": sorted([(tn[i], on[j], round(float(corr[i, j]), 4)) for i, j in pairs
                                               if key != "TJDR" or i < j], key=lambda z: -z[2])[:300]}
    an, av, am, _ = desc["APTOS_train"]
    ca = np.maximum(av @ av.T, am @ av.T)
    np.fill_diagonal(ca, -1)
    out["APTOS_train_internal"] = {f">={k}": int(np.triu(ca >= k, 1).sum()) for k in (0.3, 0.5, 0.7, 0.9)}
    out["APTOS_train_internal_null_quantiles"] = np.quantile(ca[np.triu_indices(len(an), 1)],
                                                             [.5, .99, .9999]).round(4).tolist()
    with open(f"{OUT}/overlap_stage2.json", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("stage2 DONE", round(time.time() - t0), flush=True)


def dataset_sets():
    import pandas as pd
    inv = pd.read_csv(f"{OUT}/inventory.csv")
    return {
        "TJDR": [(f"{r.split}/{r.image}", f"{TJDR_RAW}/{r.split}/image/{r.image}.png") for r in inv.itertuples()],
        "APTOS_train": [(os.path.basename(p), p) for p in sorted(glob.glob(f"{DS}/APTOS2019/raw/train_images/*.png"))],
        "APTOS_test": [(os.path.basename(p), p) for p in sorted(glob.glob(f"{DS}/APTOS2019/raw/test_images/*.png"))],
        "IDRiD_grading": [(os.path.relpath(p, f"{DS}/IDRiD/grading/raw"), p) for p in
                          sorted(glob.glob(f"{DS}/IDRiD/grading/raw/1. Original Images/*/*.jpg"))],
        "IDRiD_seg": [(os.path.relpath(p, f"{DS}/IDRiD/segmentation/raw"), p) for p in
                      sorted(glob.glob(f"{DS}/IDRiD/segmentation/raw/1. Original Images/*/*.jpg"))],
    }


# ----------------------------------------------------------------------------- main

def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S")}

    # --- 1. TJDR inventory + survival -------------------------------------------------------
    pairs, orphans = [], {}
    for split in ("train", "test"):
        imgs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{TJDR_RAW}/{split}/image/*")}
        anns = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{TJDR_RAW}/{split}/annotation/*")}
        orphans[split] = {"image_without_annotation": sorted(imgs - anns),
                          "annotation_without_image": sorted(anns - imgs),
                          "images": len(imgs), "annotations": len(anns)}
        pairs += [(split, s) for s in sorted(imgs)]
    summary["files"] = orphans
    rows, comps = [], []
    with cf.ProcessPoolExecutor(max_workers=5) as pool:
        for i, (row, recs) in enumerate(pool.map(tjdr_worker, pairs, chunksize=2)):
            rows.append(row)
            comps += recs
            if i % 50 == 0:
                print(f"TJDR {i}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
    idrid_items = []
    for split in ("a. Training Set", "b. Testing Set"):
        for p in sorted(glob.glob(f"{DS}/IDRiD/segmentation/raw/1. Original Images/{split}/*.jpg")):
            idrid_items.append((split, os.path.splitext(os.path.basename(p))[0]))
    idrid_rows = []
    with cf.ProcessPoolExecutor(max_workers=5) as pool:
        for row, recs in pool.map(idrid_worker, idrid_items):
            idrid_rows.append(row)
            comps += recs
    print(f"survival done {time.time() - t0:.0f}s", flush=True)

    import pandas as pd
    inv = pd.DataFrame(rows)
    inv.to_csv(f"{OUT}/inventory.csv", index=False)
    pd.DataFrame(idrid_rows).to_csv(f"{OUT}/idrid_seg_inventory.csv", index=False)
    comp = pd.DataFrame(comps)
    comp.to_csv(f"{OUT}/components.csv", index=False)

    # rule identity check: the audit's resize == stage4_v2.resize_mask_full_frame
    import stage4_v2 as s4
    from PIL import Image
    first = inv[inv.ok].iloc[0]
    mk = np.asarray(Image.open(f"{TJDR_RAW}/{first.split}/annotation/{first.image}.png"))
    for cls in CLASSES:
        m = mk == TJDR_CODES[cls]
        if m.any():
            assert np.array_equal((_soft_resize(m) >= 0.5).astype(np.uint8), s4.resize_mask_full_frame(m, SIZE))
    summary["rule_identity_checked_on"] = str(first.image)

    # --- 2. fingerprints for overlap ----------------------------------------------------------
    sets = {
        "TJDR": [(f"{r.split}/{r.image}", f"{TJDR_RAW}/{r.split}/image/{r.image}.png") for r in inv.itertuples()],
        "APTOS_train": [(os.path.basename(p), p) for p in sorted(glob.glob(f"{DS}/APTOS2019/raw/train_images/*.png"))],
        "APTOS_test": [(os.path.basename(p), p) for p in sorted(glob.glob(f"{DS}/APTOS2019/raw/test_images/*.png"))],
        "IDRiD_grading": [(os.path.relpath(p, f"{DS}/IDRiD/grading/raw"), p) for p in
                          sorted(glob.glob(f"{DS}/IDRiD/grading/raw/1. Original Images/*/*.jpg"))],
        "IDRiD_seg": [(os.path.relpath(p, f"{DS}/IDRiD/segmentation/raw"), p) for p in
                      sorted(glob.glob(f"{DS}/IDRiD/segmentation/raw/1. Original Images/*/*.jpg"))],
    }
    fps, errors = {}, {}
    with cf.ProcessPoolExecutor(max_workers=6) as pool:
        for key, items in sets.items():
            res = list(pool.map(fingerprint_worker, items, chunksize=8))
            fps[key] = [r for r in res if r[2] is not None]
            errors[key] = [(r[0], r[3]) for r in res if r[2] is None]
            print(f"fingerprinted {key}: {len(fps[key])} ({len(errors[key])} errors) {time.time() - t0:.0f}s",
                  flush=True)
        rng = np.random.default_rng(0)
        control_paths = [p for _, p in rng.permutation(np.array(sets["TJDR"], dtype=object))[:24]]
        controls = list(pool.map(control_worker, control_paths))
    np.savez(f"{OUT}/hashes.npz", **{f"{k}_phash": np.stack([f[2]["phash"] for f in v]) for k, v in fps.items()},
             **{f"{k}_names": np.array([f[0] for f in v]) for k, v in fps.items()})

    ctrl_d = [int(hamming(o["phash"][None], t["phash"][None])[0, 0]) for _, o, t in controls]
    ctrl_r = [float(np.dot(o["thumb"], t["thumb"]) / 256) for _, o, t in controls]
    overlap = {"errors": errors, "positive_control": {
        "transform": "Stage-2 DR profile + downscale to 1024 + JPEG q85", "n": len(controls),
        "hamming": ctrl_d, "thumb_corr": ctrl_r}}

    t_ph = np.stack([f[2]["phash"] for f in fps["TJDR"]])
    t_mi = np.stack([f[2]["phash_mirror"] for f in fps["TJDR"]])
    t_th = np.stack([f[2]["thumb"] for f in fps["TJDR"]])
    t_names = [f[0] for f in fps["TJDR"]]
    t_sha = {f[2]["sha256"] for f in fps["TJDR"]} | {f[2]["pixel_sha256"] for f in fps["TJDR"]}
    for key in ("APTOS_train", "APTOS_test", "IDRiD_grading", "IDRiD_seg", "TJDR"):
        o_ph = np.stack([f[2]["phash"] for f in fps[key]])
        o_th = np.stack([f[2]["thumb"] for f in fps[key]])
        d = np.minimum(hamming(t_ph, o_ph), hamming(t_mi, o_ph))
        corr = t_th @ o_th.T / 256
        if key == "TJDR":
            np.fill_diagonal(d, 99)
            np.fill_diagonal(corr, -1)
        exact = sorted({f[0] for f in fps[key] if f[2]["sha256"] in t_sha or f[2]["pixel_sha256"] in t_sha}) \
            if key != "TJDR" else None
        cand = np.argwhere((d <= 12) | (corr >= 0.95))
        overlap[key] = {
            "n": len(fps[key]), "exact_sha_matches": exact,
            "min_hamming": int(d.min()), "hamming_hist_0_16": np.bincount(d.min(axis=1), minlength=65)[:17].tolist(),
            "max_thumb_corr": float(corr.max()),
            "candidates_d<=12_or_corr>=0.95": [
                {"tjdr": t_names[i], "other": fps[key][j][0], "hamming": int(d[i, j]), "corr": float(corr[i, j])}
                for i, j in cand[:200]]}
        if key == "TJDR":
            overlap[key]["cross_split_pairs_d<=8"] = [
                (t_names[i], t_names[j], int(d[i, j])) for i, j in np.argwhere(d <= 8)
                if i < j and t_names[i].split("/")[0] != t_names[j].split("/")[0]]
            overlap[key]["duplicate_file_sha"] = int(len(fps["TJDR"]) - len({f[2]["sha256"] for f in fps["TJDR"]}))
    # APTOS-internal positive control (APTOS 2019 is known to contain near-duplicates)
    a_ph = np.stack([f[2]["phash"] for f in fps["APTOS_train"]])
    da = hamming(a_ph, a_ph)
    np.fill_diagonal(da, 99)
    overlap["APTOS_train_internal_pairs"] = {f"d<={k}": int((np.triu(da <= k, 1)).sum()) for k in (0, 2, 4, 6, 8)}
    # geometry of the other datasets (FOV bbox in the 1536 frame)
    geo = {}
    for key in ("APTOS_train", "IDRiD_seg", "TJDR"):
        vals = []
        for f in fps[key]:
            (h, w), b = f[2]["size"], f[2]["fov_bbox"]
            if b:
                vals.append(((b[1] - b[0] + 1) * SIZE / h, (b[3] - b[2] + 1) * SIZE / w, w / h, h, w))
        v = np.array(vals)
        geo[key] = {"fov_height_1536_q": np.percentile(v[:, 0], [5, 25, 50, 75, 95]).round(1).tolist(),
                    "fov_width_1536_q": np.percentile(v[:, 1], [5, 25, 50, 75, 95]).round(1).tolist(),
                    "aspect_w_over_h_q": np.percentile(v[:, 2], [5, 50, 95]).round(3).tolist(),
                    "native_sizes_top": pd.Series([f"{int(a)}x{int(b)}" for a, b in v[:, 3:5]])
                    .value_counts().head(8).to_dict()}
    overlap["geometry"] = geo
    with open(f"{OUT}/overlap.json", "w") as fh:
        json.dump(overlap, fh, indent=1, default=str)
    summary["elapsed_s"] = round(time.time() - t0)
    with open(f"{OUT}/summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print("DONE", summary["elapsed_s"], "s", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["overlap2"]:
        overlap_stage2(dataset_sets())
    else:
        main()
        overlap_stage2(dataset_sets())
