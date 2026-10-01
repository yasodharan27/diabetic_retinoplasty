"""Architecture-1 input pipeline -- reads ONLY a v2 bundle (research record §40; spec §11).

A bundle binds one Stage-2 RGB generation, one Stage-3 vessel generation and one Stage-4 v2
pathology generation. Every sample is read through the verified readers of `stage34_cache_v2`
(per-file SHA-256 from the manifests, embedded Stage-4/Stage-3 SHA, generation id and channel order
in every pathology file). There is no compute-on-miss path, no reliability signal and no fallback to
any legacy cache: a missing or mismatching file raises.
"""
import os

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache

DEFAULT_ROOTS = {"stage2_rgb_v2": v2cfg.STAGE2_ROOT, "stage3_cache_v2": v2cfg.STAGE3_ROOT,
                 "stage4_cache_v2": v2cfg.STAGE4_ROOT, "bundle_v2": v2cfg.BUNDLE_ROOT}
_FILENAMES = {"stage2_rgb_v2": cache.rgb_filename, "stage3_cache_v2": cache.vessel_filename,
              "stage4_cache_v2": cache.pathology_filename}


class Arch1Bundle:
    """A validated v2 bundle. `expected_bundle_id` and `expected_split_sha256` are REQUIRED."""

    def __init__(self, *, expected_bundle_id, expected_split_sha256, roots=None):
        self.roots = dict(DEFAULT_ROOTS, **(roots or {}))
        for root in self.roots.values():
            cache.assert_not_legacy_path(root)
        bundle_path = os.path.join(self.roots["bundle_v2"], expected_bundle_id,
                                   cache.MANIFEST_NAMES["bundle_v2"])
        self.bundle = cache.load_manifest(bundle_path, expected_kind="bundle_v2",
                                          expected_generation_id=expected_bundle_id)
        generations = {"stage2_rgb_v2": self.bundle["stage2_generation"],
                       "stage3_cache_v2": self.bundle["stage3_generation"],
                       "stage4_cache_v2": self.bundle["stage4_generation"]}
        self.dirs, self.manifests = {}, {}
        for kind, gen in generations.items():
            directory = os.path.join(self.roots[kind], gen)
            self.dirs[kind] = os.path.join(directory, cache.DATA_SUBDIRS[kind])
            self.manifests[kind] = cache.load_manifest(
                os.path.join(directory, cache.MANIFEST_NAMES[kind]), expected_kind=kind,
                expected_generation_id=gen)
        rebuilt = cache.bundle_manifest(self.manifests["stage2_rgb_v2"],
                                        self.manifests["stage3_cache_v2"],
                                        self.manifests["stage4_cache_v2"])
        for key in ("bundle_id", "stage3_sha256", "stage4_sha256", "channels", "split_sha256",
                    "population"):
            if rebuilt[key] != self.bundle[key]:
                raise cache.ManifestMismatchError(f"bundle {expected_bundle_id}: {key} disagrees with "
                                                  "its generation manifests")
        if self.bundle["split_sha256"] != expected_split_sha256:
            raise cache.ManifestMismatchError(f"bundle split {self.bundle['split_sha256']} != "
                                              f"{expected_split_sha256}")
        s4 = self.manifests["stage4_cache_v2"]
        self.channels = tuple(s4["channels"])
        self.classes = cache.classes_from_channels(self.channels)
        self.stage4_sha256 = s4["stage4_sha256"]
        self.stage3_sha256 = self.bundle["stage3_sha256"]
        self.stage4_generation = s4["generation_id"]
        self.population = tuple(self.bundle["population"])

    def require(self, image_ids, check_files=True):
        """Completeness: every id is in all three generations (and on disk)."""
        for kind, manifest in self.manifests.items():
            cache.assert_complete(manifest, image_ids, self.dirs[kind] if check_files else None,
                                  _FILENAMES[kind])
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
        return {"bundle_id": self.bundle["bundle_id"], "channels": list(self.channels),
                "stage4_sha256": self.stage4_sha256, "stage3_sha256": self.stage3_sha256,
                "population": len(self.population), "split_sha256": self.bundle["split_sha256"]}


def make_sequence(bundle, image_ids, grades, batch_size, shuffle, seed):
    """A keras PyDataset yielding ({'rgb','vessel','pathology'}, grade) batches from the bundle."""
    import keras

    image_ids = [str(i) for i in image_ids]
    grades = np.asarray(grades, dtype=np.int32)
    if len(image_ids) != len(grades):
        raise ValueError("image_ids and grades differ in length")
    bundle.require(image_ids)

    class _Arch1Sequence(keras.utils.PyDataset):
        def __init__(self):
            super().__init__()
            self.order = np.arange(len(image_ids))
            self.epoch = 0
            self._shuffle()

        def _shuffle(self):
            if shuffle:
                self.order = np.random.default_rng([int(seed), self.epoch]).permutation(len(image_ids))

        def __len__(self):
            return int(np.ceil(len(image_ids) / batch_size))

        def __getitem__(self, index):
            rows = self.order[index * batch_size:(index + 1) * batch_size]
            samples = [bundle.load_sample(image_ids[r]) for r in rows]
            inputs = {k: np.stack([s[k] for s in samples]) for k in ("rgb", "vessel", "pathology")}
            return inputs, grades[rows]

        def on_epoch_end(self):
            self.epoch += 1
            self._shuffle()

    return _Arch1Sequence()

