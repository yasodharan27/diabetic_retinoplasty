"""Fundus-cropped 512 frames for the cropped arm of the joint training experiment (research record §94).

The frame of an image here is the project's frame with ONE step added: Stage 2 (gamma, CLAHE) on the native
image, the image cut to the bounding box of the fundus, then the SAME resize to 512 x 512. The box rule is the one
fixed in §89 and used by the geometry diagnostic (`geometry_diagnostic.fundus_box`): no parameter is changed.

Layout (the layout of the EyePACS frame cache, so the same staging and reading code applies): shards of at most
1,000 float16 frames, `frames_<split>_<nnn>.npy`, and `index.json` naming every shard's images, source hashes,
boxes and SHA-256. A shard is written whole or not at all, with a sidecar record, so an interrupted build
resumes at the next missing shard.

Two sets are built: the EyePACS adaptation split (training and validation parts) and the APTOS training and
validation images. The EyePACS held-out test, IDRiD and DDR are not read here.
"""
import hashlib
import json
import os

import numpy as np

import geometry_diagnostic as gd

HERE = os.path.dirname(os.path.abspath(__file__))
VERSION = "crop-frames-v1: stage4_v2_data.stage2_rgb -> geometry_diagnostic.fundus_box cut -> stage4_v2_aptos_cache.recompute_rgb_512 -> float16"
RULE = {"gray_threshold": gd.GRAY_THRESHOLD, "line_fraction": gd.LINE_FRACTION, "min_side_fraction": gd.MIN_SIDE_FRACTION}
SHARD_SIZE = 1000
EYEPACS_DIR = os.path.join(HERE, "datasets", "EyePACS", "processed", "adaptation_frames_crop_v1")
APTOS_DIR = os.path.join(HERE, "datasets", "APTOS2019", "processed", "frames_crop_v1")


class CropCacheError(RuntimeError):
    """A cropped-frame cache is incomplete, altered, or not the one that was asked for."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def crop_frame(raw_path):
    """(frame float32 (512, 512, 3) in [0, 1], box (top, bottom, left, right), native (h, w)) of one raw image."""
    import cv2

    import stage4_v2_aptos_cache as ac
    import stage4_v2_data as sd
    native = sd.stage2_rgb(raw_path)
    bgr = cv2.imread(raw_path)
    if bgr is None or bgr.shape[:2] != native.shape[:2]:
        raise CropCacheError(f"{raw_path}: the raw image and its Stage-2 image differ in size")
    top, bottom, left, right = gd.fundus_box(bgr)
    frame = ac.recompute_rgb_512(np.ascontiguousarray(native[top:bottom, left:right]))
    if frame.shape != (512, 512, 3) or frame.dtype != np.float32 or frame.min() < 0.0 or frame.max() > 1.0:
        raise CropCacheError(f"{raw_path}: the frame is not float32 (512, 512, 3) in [0, 1]")
    return frame, (top, bottom, left, right), tuple(native.shape[:2])


# --------------------------------------------------------------------------- what is built

def eyepacs_plan(raw_dir):
    """{split: [(image, grade, raw path), ...]} of the pinned adaptation split, in the cache's shard order."""
    import eyepacs_adaptation_data as ead
    rows = ead.read_split(os.path.join(HERE, "dataset_splits"), ead.MANIFEST_SHA256)
    grade = {r["image"]: int(r["grade"]) for r in rows}
    return {split: [(image, grade[image], os.path.join(raw_dir, image)) for shard in shards for image in shard]
            for split, shards in ead.shard_plan(rows).items()}


def aptos_plan(raw_dir):
    """{split: [(image id, grade, raw path), ...]} of the authoritative APTOS split without the empty-FOV images,
    in the order of the split file."""
    import csv

    import pipeline_v2_config as v2cfg
    empty = set(v2cfg.APTOS_EMPTY_FOV_IDS)
    with open(os.path.join(HERE, "dataset_splits", "aptos2019_train_val_split.csv"), newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["id_code"] not in empty]
    return {split: [(r["id_code"], int(r["diagnosis"]), os.path.join(raw_dir, r["id_code"] + ".png")) for r in rows if r["split"] == split]
            for split in ("train", "val")}


# --------------------------------------------------------------------------- build (CPU, no model)

def build(plan, cache_dir, set_name, workers=4, shard_size=SHARD_SIZE, log=print):
    """Builds every missing shard of `plan` and then writes index.json. Returns the index."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(cache_dir, exist_ok=True)
    shards = {}
    for split, items in plan.items():
        for k in range(0, len(items), shard_size):
            name = f"frames_{split}_{k // shard_size:03d}.npy"
            part, record_path, path = items[k:k + shard_size], os.path.join(cache_dir, name + ".json"), os.path.join(cache_dir, name)
            if os.path.exists(record_path) and os.path.exists(path):
                with open(record_path) as fh:
                    record = json.load(fh)
                if record["images"] == [i for i, _, _ in part] and record["bytes"] == os.path.getsize(path):
                    shards[name] = record
                    continue
            block = np.lib.format.open_memmap(path + ".part", mode="w+", dtype=np.float16, shape=(len(part), 512, 512, 3))
            boxes, natives, hashes = [], [], []

            def work(item):
                return crop_frame(item[2]), sha256_file(item[2])
            with ThreadPoolExecutor(workers) as pool:
                for row, ((frame, box, native), digest) in enumerate(pool.map(work, part)):
                    block[row] = frame.astype(np.float16)
                    boxes.append(list(box)), natives.append(list(native)), hashes.append(digest)
            block.flush()
            del block
            os.replace(path + ".part", path)
            record = {"split": split, "images": [i for i, _, _ in part], "grades": [g for _, g, _ in part], "source_sha256": hashes,
                      "boxes": boxes, "native_hw": natives, "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
            with open(record_path + ".tmp", "w") as fh:
                json.dump(record, fh)
            os.replace(record_path + ".tmp", record_path)
            shards[name] = record
            log(f"  {set_name}: {name} ({len(part)} frames) | {len(shards)} of {sum(-(-len(v) // shard_size) for v in plan.values())} shards")
    index = {"version": VERSION, "set": set_name, "rule": RULE, "dtype": "float16", "frame_size": 512, "shard_size": shard_size,
             "frames": {split: len(items) for split, items in plan.items()}, "shards": shards}
    with open(os.path.join(cache_dir, "index.json.tmp"), "w") as fh:
        json.dump(index, fh)
    os.replace(os.path.join(cache_dir, "index.json.tmp"), os.path.join(cache_dir, "index.json"))
    return index


def summary(index):
    """Label-free description of what the rule did: images left uncut and box sizes."""
    boxes = np.asarray([b for s in index["shards"].values() for b in s["boxes"]], np.float64)
    native = np.asarray([n for s in index["shards"].values() for n in s["native_hw"]], np.float64)
    height, width = (boxes[:, 1] - boxes[:, 0]) / native[:, 0], (boxes[:, 3] - boxes[:, 2]) / native[:, 1]
    q = lambda v: [round(float(x), 3) for x in np.percentile(v, [10, 50, 90])]      # noqa: E731
    return {"set": index["set"], "frames": index["frames"], "left_uncut": int(np.sum((height == 1.0) & (width == 1.0))),
            "box_width_fraction_p10_50_90": q(width), "box_height_fraction_p10_50_90": q(height)}


# --------------------------------------------------------------------------- read

class CropFrames:
    """Read access to a cropped-frame cache: `entries(split)` in build order and `frame(image)` as float32
    (512, 512, 3) in [0, 1]. Construction checks the version, the rule, the expected images of every split and
    (optionally) every shard's SHA-256."""

    def __init__(self, cache_dir, expected, check_hashes=True, log=None):
        with open(os.path.join(cache_dir, "index.json")) as fh:
            index = json.load(fh)
        if index["version"] != VERSION or index["rule"] != RULE or index["dtype"] != "float16" or index["frame_size"] != 512:
            raise CropCacheError("the cache was not built with this version of the crop rule")
        self.cache_dir, self.set_name = cache_dir, index["set"]
        self._where, self._entries = {}, {split: [] for split in index["frames"]}
        for name in sorted(index["shards"]):
            entry = index["shards"][name]
            path = os.path.join(cache_dir, name)
            if not os.path.exists(path) or os.path.getsize(path) != entry["bytes"]:
                raise CropCacheError(f"{name} is missing or has another size")
            if check_hashes:
                if log:
                    log(f"  hashing {name}")
                if sha256_file(path) != entry["sha256"]:
                    raise CropCacheError(f"{name} is not the shard that was built")
            for row, (image, grade) in enumerate(zip(entry["images"], entry["grades"])):
                if image in self._where:
                    raise CropCacheError(f"{image} is in the cache twice")
                self._where[image] = (name, row)
                self._entries[entry["split"]].append((image, int(grade)))
        for split, wanted in expected.items():
            if [(str(i), int(g)) for i, g in wanted] != self._entries.get(split):
                raise CropCacheError(f"the {split} part of the cache is not the expected images in the expected order")
        self.fingerprint = hashlib.sha256(json.dumps({n: e["sha256"] for n, e in sorted(index["shards"].items())}, sort_keys=True).encode()).hexdigest()
        self._open = {}

    def entries(self, split):
        return list(self._entries[split])

    def frame(self, image):
        name, row = self._where[str(image)]
        block = self._open.get(name)
        if block is None:
            block = self._open[name] = np.load(os.path.join(self.cache_dir, name), mmap_mode="r")
        return np.asarray(block[row], dtype=np.float32)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Build the fundus-cropped frame caches (CPU; no model)")
    parser.add_argument("set", choices=("aptos", "eyepacs"))
    parser.add_argument("--aptos-raw-dir", default=os.path.join(HERE, "datasets", "APTOS2019", "raw", "train_images"))
    parser.add_argument("--eyepacs-raw-dir", default=os.path.join(HERE, "datasets", "EyePACS", "raw", "train"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.set == "aptos":
        index = build(aptos_plan(args.aptos_raw_dir), APTOS_DIR, "aptos", args.workers)
    else:
        index = build(eyepacs_plan(args.eyepacs_raw_dir), EYEPACS_DIR, "eyepacs_adaptation", args.workers)
    print(json.dumps(summary(index)))


if __name__ == "__main__":
    main()
