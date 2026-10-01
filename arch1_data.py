"""Architecture-1 input pipeline -- reads ONLY a v2 bundle (research record §40 §11, §46).

A bundle binds one Stage-2 RGB generation, one Stage-3 vessel generation and one Stage-4 v2 pathology
generation. Opening it REFUSES (section E of the §46 request) when:
  * the caller does not state the Stage-4 model SHA, or the bundle/Stage-4 manifest names another model;
  * a generation id is not a v2 id (legacy namespaces are refused, not just legacy paths);
  * Stage-3 or RGB parity is not recorded as passed;
  * the APTOS split SHA or the population (3,651 ids, train/val disjoint, APTOS ids only) is wrong;
  * the bundle fingerprint does not recompute from its generation manifests;
  * any required file is missing (completeness) -- every sample is then read through the verified readers
    (per-file SHA, embedded Stage-4/Stage-3 SHA, generation id, channel order).
There is no compute-on-miss path and no fallback to any other cache.

Training batches follow the P protocol exactly (improved_training_data): epoch order
`epoch_training_order(entries, run_seed, epoch)`, per-image RNG `per_image_augmentation_rng(run_seed, epoch,
id)`, augmentation `lfed._augment_spatial` (all channels together) then `lfed._augment_intensity_rgb` (RGB only;
vessel / lesion probabilities untouched).
"""
import hashlib
import json
import os
import re

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache

DEFAULT_ROOTS = {"stage2_rgb_v2": v2cfg.STAGE2_ROOT, "stage3_cache_v2": v2cfg.STAGE3_ROOT,
                 "stage4_cache_v2": v2cfg.STAGE4_ROOT, "bundle_v2": v2cfg.BUNDLE_ROOT}
_FILENAMES = {"stage2_rgb_v2": cache.rgb_filename, "stage3_cache_v2": cache.vessel_filename,
              "stage4_cache_v2": cache.pathology_filename}
_GEN_PATTERNS = {"stage2_rgb_v2": rf"^{re.escape(v2cfg.STAGE2_RGB_GENERATION)}$",
                 "stage3_cache_v2": rf"^{re.escape(v2cfg.STAGE3_GENERATION)}$",
                 "stage4_cache_v2": r"^s4v2-[0-9a-f]{12}-K\d+$"}
_APTOS_ID = re.compile(r"^[0-9a-f]{12}$")


class Arch1Bundle:
    """A validated v2 bundle. `expected_bundle_id`, `expected_split_sha256` and `expected_stage4_sha256` are
    REQUIRED."""

    def __init__(self, *, expected_bundle_id, expected_split_sha256, expected_stage4_sha256, roots=None,
                 expected_population=v2cfg.APTOS_POPULATION):
        cache.assert_not_deny_listed(expected_stage4_sha256, "requested Stage-4 model")
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected_stage4_sha256 or "")):
            raise cache.ManifestMismatchError("the Stage-4 model SHA (64 hex) must be stated explicitly")
        self.roots = dict(DEFAULT_ROOTS, **(roots or {}))
        for root in self.roots.values():
            cache.assert_not_legacy_path(root)
        bundle_path = os.path.join(self.roots["bundle_v2"], expected_bundle_id, cache.MANIFEST_NAMES["bundle_v2"])
        self.bundle = cache.load_manifest(bundle_path, expected_kind="bundle_v2",
                                          expected_generation_id=expected_bundle_id)
        generations = {"stage2_rgb_v2": self.bundle["stage2_generation"],
                       "stage3_cache_v2": self.bundle["stage3_generation"],
                       "stage4_cache_v2": self.bundle["stage4_generation"]}
        for kind, gen in generations.items():
            if not re.match(_GEN_PATTERNS[kind], gen):
                raise cache.LegacyArtifactError(f"{kind} generation {gen!r} is not a v2 generation")
        self.dirs, self.manifests = {}, {}
        for kind, gen in generations.items():
            directory = os.path.join(self.roots[kind], gen)
            self.dirs[kind] = os.path.join(directory, cache.DATA_SUBDIRS[kind])
            self.manifests[kind] = cache.load_manifest(
                os.path.join(directory, cache.MANIFEST_NAMES[kind]), expected_kind=kind,
                expected_generation_id=gen)
        s2, s3, s4 = (self.manifests[k] for k in ("stage2_rgb_v2", "stage3_cache_v2", "stage4_cache_v2"))
        rebuilt = cache.bundle_manifest(s2, s3, s4)
        for key in ("bundle_id", "stage3_sha256", "stage4_sha256", "channels", "split_sha256", "population"):
            if rebuilt[key] != self.bundle[key]:
                raise cache.ManifestMismatchError(f"bundle {expected_bundle_id}: {key} disagrees with its generation manifests")
        if self.bundle["stage4_sha256"] != expected_stage4_sha256 or s4["stage4_sha256"] != expected_stage4_sha256:
            raise cache.ManifestMismatchError("the bundle's Stage-4 maps come from a different model than requested")
        if s4["generation_id"] != cache.stage4_generation_id(expected_stage4_sha256, len(s4["classes"])):
            raise cache.ManifestMismatchError("Stage-4 generation id does not encode the requested model SHA")
        if not s3.get("parity", {}).get("passed"):
            raise cache.StaleCacheError("Stage-3 parity has not passed for this bundle")
        if not s2.get("parity", {}).get("passed"):
            raise cache.StaleCacheError("RGB parity has not passed for this bundle")
        if self.bundle["split_sha256"] != expected_split_sha256:
            raise cache.ManifestMismatchError(f"bundle split {self.bundle['split_sha256']} != {expected_split_sha256}")
        fp = {k: hashlib.sha256(json.dumps(m["files"], sort_keys=True).encode()).hexdigest()
              for k, m in (("stage2", s2), ("stage3", s3), ("stage4", s4))}
        if self.bundle.get("generation_fingerprints") != fp:
            raise cache.ManifestMismatchError("bundle generation fingerprints do not recompute (files changed)")
        recomputed = hashlib.sha256(json.dumps(
            {k: self.bundle[k] for k in ("bundle_id", "stage3_sha256", "stage4_sha256", "split_sha256",
                                         "population_sha256", "generation_fingerprints", "channels")},
            sort_keys=True).encode()).hexdigest()
        if recomputed != self.bundle.get("fingerprint"):
            raise cache.ManifestMismatchError("bundle fingerprint does not recompute")
        self.train_ids = tuple(self.bundle["train_ids"])
        self.val_ids = tuple(self.bundle["val_ids"])
        ids = self.train_ids + self.val_ids
        if (set(self.train_ids) & set(self.val_ids) or len(set(ids)) != expected_population
                or set(ids) != set(self.bundle["population"])):
            raise cache.ManifestMismatchError("bundle train/val population is overlapping, incomplete or inconsistent")
        bad = [i for i in ids if not _APTOS_ID.match(i)]
        if bad:
            raise cache.ManifestMismatchError(f"non-APTOS ids (e.g. IDRiD) in the bundle: {bad[:5]}")
        self.channels = tuple(s4["channels"])
        self.classes = cache.classes_from_channels(self.channels)
        self.stage4_sha256 = s4["stage4_sha256"]
        self.stage3_sha256 = self.bundle["stage3_sha256"]
        self.stage4_generation = s4["generation_id"]
        self.population = tuple(self.bundle["population"])
        self.fingerprint = self.bundle["fingerprint"]

    def require(self, image_ids, check_files=True):
        """Completeness: every id is in all three generations (and on disk)."""
        for kind, manifest in self.manifests.items():
            cache.assert_complete(manifest, image_ids, self.dirs[kind] if check_files else None, _FILENAMES[kind])
        return True

    def _path(self, kind, image_id):
        return os.path.join(self.dirs[kind], _FILENAMES[kind](image_id))

    def _sha(self, kind, image_id):
        files = self.manifests[kind]["files"]
        if str(image_id) not in files:
            raise cache.IncompleteCacheError(f"{image_id} not in {self.manifests[kind]['generation_id']}")
        return files[str(image_id)]

    def load_sample(self, image_id):
        """{'rgb': (512,512,3), 'vessel': (512,512,1), 'pathology': (512,512,2K)} float32 in [0, 1]."""
        rgb = cache.read_stage2_rgb(self._path("stage2_rgb_v2", image_id),
                                    expected_file_sha256=self._sha("stage2_rgb_v2", image_id))
        vessel = cache.read_stage3_vessel(self._path("stage3_cache_v2", image_id),
                                          expected_file_sha256=self._sha("stage3_cache_v2", image_id))
        maps = cache.read_pathology_npz(
            self._path("stage4_cache_v2", image_id),
            expected_stage4_sha256=self.stage4_sha256, expected_stage3_sha256=self.stage3_sha256,
            expected_gen_id=self.stage4_generation, expected_channels=self.channels,
            expected_file_sha256=self._sha("stage4_cache_v2", image_id))
        return {"rgb": rgb, "vessel": vessel, "pathology": cache.from_uint8(maps)}

    def describe(self):
        return {"bundle_id": self.bundle["bundle_id"], "fingerprint": self.fingerprint,
                "channels": list(self.channels), "stage4_sha256": self.stage4_sha256,
                "stage4_generation": self.stage4_generation, "stage3_sha256": self.stage3_sha256,
                "population": len(self.population), "split_sha256": self.bundle["split_sha256"],
                "population_sha256": self.bundle.get("population_sha256")}


def augment_sample(sample, rng):
    """P's augmentation on the aligned stack [RGB(3) | vessel(1) | pathology(2K)]: spatial flips/rot90 on all
    channels together, then brightness/contrast on RGB only."""
    import local_feature_extraction_dataset as lfed
    stack = np.concatenate([sample["rgb"], sample["vessel"], sample["pathology"]], axis=-1)
    stack = lfed._augment_intensity_rgb(lfed._augment_spatial(stack, rng), rng)
    return {"rgb": np.ascontiguousarray(stack[..., :3]), "vessel": np.ascontiguousarray(stack[..., 3:4]),
            "pathology": np.ascontiguousarray(stack[..., 4:])}


def epoch_entries(entries, run_seed, epoch, augment):
    import improved_training_data as itd
    return itd.epoch_training_order(entries, run_seed, epoch) if augment else list(entries)


def make_epoch_sequence(bundle, entries, epoch, run_seed, batch_size, augment):
    """A keras PyDataset for ONE epoch, yielding ({'rgb','vessel','pathology'}, grade) batches in P's order
    with P's per-image augmentation (augment=True), or in the given order unaugmented (validation)."""
    import keras

    import improved_training_data as itd
    ordered = epoch_entries([(str(i), int(g)) for i, g in entries], run_seed, epoch, augment)
    bundle.require([i for i, _ in ordered])

    class _EpochSequence(keras.utils.PyDataset):
        def __len__(self):
            return int(np.ceil(len(ordered) / batch_size))

        def __getitem__(self, index):
            rows = ordered[index * batch_size:(index + 1) * batch_size]
            samples = []
            for image_id, _ in rows:
                s = bundle.load_sample(image_id)
                if augment:
                    s = augment_sample(s, itd.per_image_augmentation_rng(run_seed, epoch, image_id))
                samples.append(s)
            inputs = {k: np.stack([s[k] for s in samples]) for k in ("rgb", "vessel", "pathology")}
            return inputs, np.asarray([g for _, g in rows], dtype=np.int32)

    return _EpochSequence()
