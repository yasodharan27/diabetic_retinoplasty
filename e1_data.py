"""E1 data: Stage-2 RGB input and the Stage-4 teacher targets, from the verified v2 bundle.

Target definition (locked): from the cached Stage-4 maps -- uint8 (512, 512, 8), channels
[MA:mean, MA:max, HE:mean, HE:max, EX:mean, EX:max, SE:mean, SE:max] -- only the four `max` channels are used,
as float32 in [0, 1], and reduced to a 16 x 16 grid by an exact 32 x 32 block maximum:

    T = M.reshape(16, 32, 16, 32, 4).max(axis=(1, 3))            # (16, 16, 4), order MA, HE, EX, SE

T is the largest teacher probability inside each 96 x 96 region of the 1536 x 1536 Stage-4 output, after the
cache's uint8 quantisation. Stage 3 is not read: no vessel file is opened by this module.

Augmentation is P's, applied to the aligned stack [RGB | pathology] with P's per-image RNG: flips and rot90 on
every channel, brightness / contrast on RGB only. Targets are pooled after the augmentation; the flips and
rotations are exact permutations and 512 = 16 * 32, so the order does not change the result. There is no
interpolation anywhere.
"""
import os

import numpy as np

import arch1_data as ad
import e1_model as em
import stage34_cache_v2 as cache

KINDS = ("stage2_rgb_v2", "stage4_cache_v2")                    # the only caches E1 reads
GRID = 16
POOLING = "max"
POSITIVE = 0.5                                                  # a cell counted as "positive" in the statistics


def max_channel_indices(channels):
    """Indices of the `max` channels in LESION_CLASSES order; the channel list must be exactly the cache's."""
    channels = tuple(str(c) for c in channels)
    if channels != cache.channel_names(em.LESION_CLASSES):
        raise cache.ManifestMismatchError(f"E1 needs the channels {cache.channel_names(em.LESION_CLASSES)}, got {channels}")
    return tuple(channels.index(f"{c}:{POOLING}") for c in em.LESION_CLASSES)


def pool_targets(pathology, channels, grid=GRID):
    """(S, S, 2K) cached maps in [0, 1] -> (grid, grid, K) float32 targets: block maximum of the max channels."""
    pathology = np.asarray(pathology)
    if pathology.ndim != 3 or pathology.shape[0] != pathology.shape[1] or pathology.shape[0] % grid:
        raise ValueError(f"expected a square (S, S, C) map with S divisible by {grid}, got {pathology.shape}")
    if pathology.shape[2] != len(tuple(channels)):
        raise ValueError(f"{pathology.shape[2]} channels, {len(tuple(channels))} names")
    maxima = pathology[..., list(max_channel_indices(channels))].astype(np.float32)
    block = pathology.shape[0] // grid
    return np.ascontiguousarray(maxima.reshape(grid, block, grid, block, maxima.shape[-1]).max(axis=(1, 3)))


def pool_targets_bruteforce(pathology, channels, grid=GRID):
    """The same targets by explicit loops (an independent implementation for the tests and the target gate)."""
    pathology = np.asarray(pathology)
    index = max_channel_indices(channels)
    block = pathology.shape[0] // grid
    out = np.zeros((grid, grid, len(index)), dtype=np.float32)
    for row in range(grid):
        for col in range(grid):
            for k, channel in enumerate(index):
                out[row, col, k] = np.float32(pathology[row * block:(row + 1) * block,
                                                        col * block:(col + 1) * block, channel].max())
    return out


def require_inputs(bundle, image_ids, check_files=True):
    """Completeness of the two generations E1 reads (the Stage-3 generation is not required)."""
    for kind in KINDS:
        cache.assert_complete(bundle.manifests[kind], image_ids, bundle.dirs[kind] if check_files else None,
                              ad._FILENAMES[kind])
    return True


def load_pathology(bundle, image_id):
    """The cached Stage-4 maps of one image as uint8 (512, 512, 8), through the verified reader (file SHA,
    embedded Stage-4 / Stage-3 SHA, generation id, channel order)."""
    return cache.read_pathology_npz(
        bundle._path("stage4_cache_v2", image_id), expected_stage4_sha256=bundle.stage4_sha256,
        expected_stage3_sha256=bundle.stage3_sha256, expected_gen_id=bundle.stage4_generation,
        expected_channels=bundle.channels, expected_file_sha256=bundle._sha("stage4_cache_v2", image_id))


def load_inputs(bundle, image_id):
    """{'rgb': (512, 512, 3), 'pathology': (512, 512, 8)} float32 in [0, 1]. No vessel file is opened."""
    rgb = cache.read_stage2_rgb(bundle._path("stage2_rgb_v2", image_id),
                                expected_file_sha256=bundle._sha("stage2_rgb_v2", image_id))
    return {"rgb": rgb, "pathology": cache.from_uint8(load_pathology(bundle, image_id))}


def augment_inputs(sample, rng):
    """P's augmentation on the aligned stack [RGB(3) | pathology(8)]: lfed._augment_spatial on every channel,
    then lfed._augment_intensity_rgb on channels 0-2 only. With the same per-image RNG the RGB result is the
    one P and Architecture 1 produced for that image in that epoch."""
    import local_feature_extraction_dataset as lfed
    stack = np.concatenate([sample["rgb"], sample["pathology"]], axis=-1)
    stack = lfed._augment_intensity_rgb(lfed._augment_spatial(stack, rng), rng)
    return {"rgb": np.ascontiguousarray(stack[..., :3]), "pathology": np.ascontiguousarray(stack[..., 3:])}


def make_epoch_sequence(bundle, entries, epoch, run_seed, batch_size, augment):
    """A keras PyDataset for ONE epoch yielding ({'rgb'}, (grades, lesion targets)) batches in P's order with
    P's per-image augmentation (augment=True), or in the given order unaugmented (validation)."""
    import keras

    import improved_training_data as itd
    ordered = ad.epoch_entries([(str(i), int(g)) for i, g in entries], run_seed, epoch, augment)
    require_inputs(bundle, [i for i, _ in ordered])
    channels = bundle.channels

    class _EpochSequence(keras.utils.PyDataset):
        def __len__(self):
            return int(np.ceil(len(ordered) / batch_size))

        def __getitem__(self, index):
            rows = ordered[index * batch_size:(index + 1) * batch_size]
            rgb, targets = [], []
            for image_id, _ in rows:
                s = load_inputs(bundle, image_id)
                if augment:
                    s = augment_inputs(s, itd.per_image_augmentation_rng(run_seed, epoch, image_id))
                rgb.append(s["rgb"])
                targets.append(pool_targets(s["pathology"], channels))
            return ({"rgb": np.stack(rgb)},
                    (np.asarray([g for _, g in rows], dtype=np.int32), np.stack(targets).astype(np.float32)))

    return _EpochSequence()


# --------------------------------------------------------------------------- target statistics (label-free)

def embedded_image_id(bundle, image_id):
    """The image id written inside the cached Stage-4 file (alignment check: must equal the requested id)."""
    with np.load(bundle._path("stage4_cache_v2", image_id), allow_pickle=False) as data:
        return str(data["image_id"])


def target_statistics(bundle, image_ids, bruteforce_every=1, log=None):
    """Statistics of the 16 x 16 targets of `image_ids`, with the hard checks of the target gate. No grade
    label is read. `bruteforce_every`: every n-th image is also pooled by explicit loops and compared exactly.
    Returns {'statistics', 'lesion_prior', 'failures'}."""
    k = len(em.LESION_CLASSES)
    total = np.zeros(k, dtype=np.float64)
    above = {t: np.zeros(k, dtype=np.int64) for t in (0.0, POSITIVE, 0.9)}
    images_without_positive = np.zeros(k, dtype=np.int64)
    images_without_any_positive = 0
    histogram = np.zeros((k, 256), dtype=np.int64)
    lo, hi, non_finite, cells, bruteforce_checked = np.inf, -np.inf, 0, 0, 0
    failures = []
    ids = [str(i) for i in image_ids]
    for n, image_id in enumerate(ids):
        maps = load_pathology(bundle, image_id)
        found = embedded_image_id(bundle, image_id)
        if found != image_id:
            failures.append(f"{image_id}: the cached file belongs to image {found}")
        pathology = cache.from_uint8(maps)
        target = pool_targets(pathology, bundle.channels)
        if target.shape != (GRID, GRID, k) or target.dtype != np.float32:
            failures.append(f"{image_id}: target is {target.dtype} {target.shape}")
            continue
        if bruteforce_every and n % bruteforce_every == 0:
            bruteforce_checked += 1
            if not np.array_equal(target, pool_targets_bruteforce(pathology, bundle.channels)):
                failures.append(f"{image_id}: pooled target differs from the brute-force block maximum")
        finite = np.isfinite(target)
        non_finite += int((~finite).sum())
        if not finite.all():
            continue
        lo, hi = min(lo, float(target.min())), max(hi, float(target.max()))
        cells += GRID * GRID
        total += target.sum(axis=(0, 1), dtype=np.float64)
        for threshold, counter in above.items():
            counter += (target > threshold).sum(axis=(0, 1))
        positive = (target >= POSITIVE).sum(axis=(0, 1))
        images_without_positive += positive == 0
        images_without_any_positive += int(positive.sum() == 0)
        levels = np.rint(target * 255.0).astype(np.int64)
        if np.abs(levels / 255.0 - target).max() > 1e-6:
            failures.append(f"{image_id}: target values are not uint8 levels / 255")
        for c in range(k):
            histogram[c] += np.bincount(levels[..., c].ravel(), minlength=256)
        if log and (n + 1) % 500 == 0:
            log(f"  targets {n + 1}/{len(ids)}")
    if non_finite:
        failures.append(f"{non_finite} non-finite target values")
    if cells and (lo < 0.0 or hi > 1.0):
        failures.append(f"target range [{lo}, {hi}] is outside [0, 1]")
    mean = total / max(cells, 1)
    for c, name in enumerate(em.LESION_CLASSES):
        distinct = int((histogram[c] > 0).sum())
        if cells and distinct <= 1:
            failures.append(f"{name}: every target cell has the same value (collapse)")
        if cells and (mean[c] <= 0.0 or mean[c] >= 1.0):
            failures.append(f"{name}: mean target {mean[c]} (all-zero or all-one collapse)")
    per_class = lambda values: {name: float(v) for name, v in zip(em.LESION_CLASSES, values)}
    stats = {"images": len(ids), "cells_per_class": int(cells), "shape": [GRID, GRID, k], "min": float(lo), "max": float(hi),
             "non_finite": int(non_finite), "mean": float(mean.mean()), "mean_per_class": per_class(mean),
             "positive_threshold": POSITIVE,
             "positive_cell_rate_per_class": per_class((histogram[:, int(np.ceil(POSITIVE * 255)):].sum(axis=1)) / max(cells, 1)),
             "nonzero_cell_rate_per_class": per_class(above[0.0] / max(cells, 1)),
             "fraction_cells_above_0.5_per_class": per_class(above[POSITIVE] / max(cells, 1)),
             "fraction_cells_above_0.9_per_class": per_class(above[0.9] / max(cells, 1)),
             "fraction_cells_above_0.5": float(above[POSITIVE].sum() / max(cells * k, 1)),
             "fraction_cells_above_0.9": float(above[0.9].sum() / max(cells * k, 1)),
             "fraction_images_without_positive_cell_per_class": per_class(images_without_positive / max(len(ids), 1)),
             "fraction_images_without_any_positive_cell": float(images_without_any_positive / max(len(ids), 1)),
             "distinct_levels_per_class": {name: int((histogram[c] > 0).sum()) for c, name in enumerate(em.LESION_CLASSES)},
             "fraction_cells_exactly_0_per_class": per_class(histogram[:, 0] / max(cells, 1)),
             "fraction_cells_exactly_1_per_class": per_class(histogram[:, 255] / max(cells, 1)),
             "bruteforce_checked_images": int(bruteforce_checked)}
    return {"statistics": stats, "lesion_prior": [float(v) for v in mean], "failures": failures}


def stage_inputs(bundle_id, drive_roots, local_roots, log=print, stage_files=None):
    """Copies the Stage-2 RGB and Stage-4 generations of a bundle from Drive to local disk with per-file SHA
    checks (colab stage4_v2_setup.stage_files), plus every manifest. The Stage-3 generation's FILES are not
    copied: E1 never reads them. Manifests are written last."""
    import json
    if stage_files is None:
        import stage4_v2_setup
        stage_files = stage4_v2_setup.stage_files
    names = {"stage2_rgb_v2": cache.rgb_filename, "stage3_cache_v2": cache.vessel_filename,
             "stage4_cache_v2": cache.pathology_filename}
    with open(os.path.join(drive_roots["bundle_v2"], bundle_id, cache.MANIFEST_NAMES["bundle_v2"]), "rb") as fh:
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
    log(f"  E1 inputs staged (RGB + lesion maps only): {out}")
    return out
