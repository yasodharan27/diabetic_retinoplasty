"""EyePACS adaptation -- data only (research record §69): inventory, split, frame cache and its reader. Nothing
here trains a model, reads an APTOS file, reads an EyeQ quality label or uses the Stage-1 model.

Purpose of the adaptation (fixed elsewhere): one supervised DR-grading run of P's model on EyePACS, whose
encoder then initialises P-EP and E1-EP on APTOS. This module fixes WHICH images it uses and HOW they are
preprocessed:

  * inventory  -- the 35,126 labelled Kaggle EyePACS training images (`trainLabels.csv`), named
                  `<patient>_<left|right>.jpeg`;
  * exclusion  -- exactly the four images recorded as blank (BLANK_TRAINING_IMAGES); nothing else is removed,
                  and no quality label or quality model is consulted;
  * split      -- by PATIENT, 90 % training / 10 % validation, stratified by the higher grade of the patient's
                  two eyes, seeded; both eyes of a patient are always in the same split;
  * frames     -- P's image path: Stage 2 (DR profile, once, native size) and the full-frame resize to 512 x 512
                  (the functions used for APTOS and IDRiD), cached once as float16 shards;
  * cache      -- bound to the split manifest's SHA-256, the preprocessing version, the dtype and the shard
                  layout; every shard records its images, their grades, the SHA-256 of each source file and its
                  own SHA-256. `verify_cache` / `FrameCache` refuse a cache that is incomplete, was built for
                  another manifest or configuration, or whose shards do not match their recorded hashes.

The split manifest is written once and is immutable: `write_split` refuses to overwrite a different one.

The held-out test set (record §69.2) is a second, separate set with its own immutable manifest and its own
cache: the Kaggle EyePACS TEST images that carry a DR grade in the EyeQ repository's label file. Only the image
name and the grade are read from that file -- never its quality column. It is evaluation-only.

Declared deviation from P's APTOS cache: frames are stored as float16, not float32 (largest rounding error
measured 0.00024; about 55 GB instead of 110 GB). Under mixed_float16 the model computes in float16 anyway.
"""
import csv
import hashlib
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.abspath(__file__))
SPLIT_SEED = 20261008
VAL_FRACTION = 0.10
#: Blank photographs (every grey value <= 22 of 255): found by the overlap screen of record §64 and inspected
#: individually. They are readable files; they are excluded because they contain no fundus.
BLANK_TRAINING_IMAGES = ("1986_left.jpeg", "32253_right.jpeg", "34689_left.jpeg", "43457_left.jpeg")
EXPECTED_LABELLED = 35126
EXPECTED_GRADE_COUNTS = (25810, 2443, 5292, 873, 708)
MANIFEST_NAME = "eyepacs_adaptation_split_v1.csv"
MANIFEST_SHA256 = "c5f7e41501b9b803be3cc4fdbaf655942366a961531e1b0570ae5fec9a13fd86"
SUMMARY_NAME = "eyepacs_adaptation_split_v1.json"
MANIFEST_FIELDS = ("image", "patient", "eye", "grade", "patient_max_grade", "split")
FRAME_SIZE = 512
FRAME_DTYPE = "float16"
SHARD_SIZE = 1000
SPLITS = ("train", "val", "test")
#: Held-out labelled EyePACS test set (record §69.2): evaluation only, never selection.
TEST_LABELS_SHA256 = "5e3a80415311b9513a957da351bb04b16b9ef661db6ef2161794d5b7da6aaaae"   # EyeQ repo, Label_EyeQ_test.csv
TEST_MANIFEST_NAME = "eyepacs_heldout_test_v1.csv"
TEST_SUMMARY_NAME = "eyepacs_heldout_test_v1.json"
TEST_MANIFEST_SHA256 = "84a344ab1c780899170def378b0efbc021ae642005ba585512a772b2268b7e6d"
EXPECTED_TEST_IMAGES = 16249
EXPECTED_TEST_GRADE_COUNTS = (11362, 1398, 2644, 448, 397)
CACHE_VERSION = "eyepacs-frames-v1: stage4_v2_data.stage2_rgb -> stage4_v2_aptos_cache.recompute_rgb_512 -> float16"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- inventory

def parse_name(image):
    """'<patient>_<left|right>' -> (patient, eye). Anything else is an error, never guessed."""
    stem = image[:-5] if image.endswith(".jpeg") else image
    patient, _, eye = stem.rpartition("_")
    if not patient.isdigit() or eye not in ("left", "right"):
        raise ValueError(f"unexpected EyePACS image name {image!r}")
    return patient, eye


def read_labels(labels_csv):
    """[{image, patient, eye, grade}] for every labelled training image, in file order."""
    rows = []
    with open(labels_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            patient, eye = parse_name(r["image"])
            grade = int(r["level"])
            if grade not in range(5):
                raise ValueError(f"{r['image']}: grade {grade}")
            rows.append({"image": r["image"] + ".jpeg", "patient": patient, "eye": eye, "grade": grade})
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("repeated image names in the label file")
    return rows


def inventory(labels_csv, train_dir=None):
    """The usable images: every labelled image except the four recorded blank ones. With `train_dir`, every
    usable image must exist as a file and every file must have a label."""
    rows = read_labels(labels_csv)
    names = {r["image"] for r in rows}
    missing_blank = [b for b in BLANK_TRAINING_IMAGES if b not in names]
    if missing_blank:
        raise ValueError(f"recorded blank images are not in the label file: {missing_blank}")
    usable = [r for r in rows if r["image"] not in BLANK_TRAINING_IMAGES]
    report = {"labelled": len(rows), "grade_counts_labelled": np.bincount([r["grade"] for r in rows], minlength=5).tolist(),
              "excluded_blank": list(BLANK_TRAINING_IMAGES), "usable": len(usable),
              "patients_labelled": len({r["patient"] for r in rows}), "patients_usable": len({r["patient"] for r in usable})}
    if train_dir is not None:
        files = set(os.listdir(train_dir))
        report["files_in_train_dir"] = len(files)
        report["labelled_without_file"] = sorted(names - files)[:10]
        report["files_without_label"] = sorted(files - names)[:10]
        if names - files or files - names:
            raise ValueError(f"label file and image folder disagree: {report['labelled_without_file']} / {report['files_without_label']}")
    return usable, report


# --------------------------------------------------------------------------- patient-level split

def build_split(usable, seed=SPLIT_SEED, val_fraction=VAL_FRACTION):
    """Rows with `patient_max_grade` and `split`. Patients are grouped by the higher grade of their eyes;
    inside each group they are sorted by numeric id, permuted with a generator seeded by (seed, grade), and the
    first round(val_fraction * n) go to validation. Deterministic in the set of usable images alone."""
    by_patient = {}
    for r in usable:
        by_patient.setdefault(r["patient"], []).append(r)
    max_grade = {p: max(x["grade"] for x in rows) for p, rows in by_patient.items()}
    split_of = {}
    for grade in range(5):
        patients = sorted((p for p, g in max_grade.items() if g == grade), key=int)
        order = np.random.default_rng([int(seed), grade]).permutation(len(patients))
        n_val = int(round(val_fraction * len(patients)))
        for rank, index in enumerate(order):
            split_of[patients[index]] = "val" if rank < n_val else "train"
    return [dict(r, patient_max_grade=max_grade[r["patient"]], split=split_of[r["patient"]]) for r in usable]


def split_report(rows):
    """Counts and the integrity facts of a split; raises if a patient is in both parts."""
    patients = {"train": set(), "val": set()}
    for r in rows:
        patients[r["split"]].add(r["patient"])
    both = patients["train"] & patients["val"]
    if both:
        raise ValueError(f"{len(both)} patients are in both splits")
    eyes = {}
    for r in rows:
        eyes.setdefault(r["patient"], []).append(r["eye"])
    out = {"seed": SPLIT_SEED, "val_fraction": VAL_FRACTION, "stratified_by": "the higher grade of the patient's eyes",
           "images": {}, "patients": {}, "grade_counts": {}, "patient_max_grade_counts": {},
           "patients_in_both_splits": 0, "patients_with_one_usable_eye": int(sum(len(v) == 1 for v in eyes.values())),
           "patients_with_two_usable_eyes": int(sum(len(v) == 2 for v in eyes.values()))}
    for split in ("train", "val"):
        part = [r for r in rows if r["split"] == split]
        out["images"][split] = len(part)
        out["patients"][split] = len(patients[split])
        out["grade_counts"][split] = np.bincount([r["grade"] for r in part], minlength=5).tolist()
        firsts = {r["patient"]: r["patient_max_grade"] for r in part}
        out["patient_max_grade_counts"][split] = np.bincount(list(firsts.values()), minlength=5).tolist()
    out["images"]["total"] = len(rows)
    out["patients"]["total"] = len(patients["train"]) + len(patients["val"])
    return out


def write_split(rows, out_dir):
    """Writes the manifest and its summary ONCE. If a manifest already exists it must be byte-identical to what
    would be written; otherwise this raises -- the split is immutable."""
    os.makedirs(out_dir, exist_ok=True)
    path, tmp = os.path.join(out_dir, MANIFEST_NAME), os.path.join(out_dir, MANIFEST_NAME + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (int(r["patient"]), r["eye"])))
    sha = sha256_file(tmp)
    if os.path.exists(path):
        if sha256_file(path) != sha:
            os.remove(tmp)
            raise RuntimeError(f"{path} exists and differs: the EyePACS adaptation split is immutable")
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    summary = dict(split_report(rows), manifest=MANIFEST_NAME, manifest_sha256=sha, excluded_blank=list(BLANK_TRAINING_IMAGES),
                   quality_filter="none (no EyeQ label and no Stage-1 prediction is used)")
    with open(os.path.join(out_dir, SUMMARY_NAME), "w") as fh:
        json.dump(summary, fh, indent=1)
    return path, summary


def read_split(out_dir, expected_sha256=None):
    path = os.path.join(out_dir, MANIFEST_NAME)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError("the EyePACS adaptation split is not the pinned manifest")
    with open(path, newline="") as fh:
        rows = [dict(r, grade=int(r["grade"]), patient_max_grade=int(r["patient_max_grade"])) for r in csv.DictReader(fh)]
    split_report(rows)
    return rows


def class_weights(rows):
    """Weighted-CORN class weights from the EyePACS TRAINING part only (P's formula, weighted_corn)."""
    import weighted_corn
    counts = np.bincount([r["grade"] for r in rows if r["split"] == "train"], minlength=5)
    return counts.tolist(), [float(w) for w in weighted_corn.class_weights_from_counts(counts)]


# --------------------------------------------------------------------------- held-out test set (labels only)

def read_test_labels(labels_csv, test_dir=None):
    """Rows {image, patient, eye, grade, patient_max_grade, split='test'} for the EyePACS test images that have a
    DR grade in the EyeQ label file. The file must be the pinned one. Its `quality` column is not read: nothing
    is filtered or weighted by quality, and no image is excluded. With `test_dir`, every image must exist."""
    if sha256_file(labels_csv) != TEST_LABELS_SHA256:
        raise RuntimeError("the EyePACS test label file is not the pinned Label_EyeQ_test.csv")
    rows = []
    with open(labels_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            patient, eye = parse_name(r["image"])
            grade = int(r["DR_grade"])
            if grade not in range(5):
                raise ValueError(f"{r['image']}: grade {grade}")
            rows.append({"image": r["image"], "patient": patient, "eye": eye, "grade": grade, "split": "test"})
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("repeated image names in the test label file")
    counts = tuple(np.bincount([r["grade"] for r in rows], minlength=5).tolist())
    if len(rows) != EXPECTED_TEST_IMAGES or counts != EXPECTED_TEST_GRADE_COUNTS:
        raise RuntimeError(f"test set is {len(rows)} images with grades {counts}, not the recorded set")
    best = {}
    for r in rows:
        best[r["patient"]] = max(best.get(r["patient"], 0), r["grade"])
    rows = [dict(r, patient_max_grade=best[r["patient"]]) for r in rows]
    if test_dir is not None:
        files = set(os.listdir(test_dir))
        missing = sorted(r["image"] for r in rows if r["image"] not in files)
        if missing:
            raise ValueError(f"{len(missing)} test images have no file (first {missing[:3]})")
    return rows


def assert_disjoint_patients(test_rows, adaptation_rows):
    """No patient of the held-out test set may be in the adaptation data (training or validation)."""
    shared = {r["patient"] for r in test_rows} & {r["patient"] for r in adaptation_rows}
    if shared:
        raise RuntimeError(f"{len(shared)} patients are in both the adaptation data and the held-out test set")
    return True


def write_test_manifest(rows, out_dir):
    """Writes the held-out test manifest and its summary ONCE (immutable, like the split manifest)."""
    if any(r["split"] != "test" for r in rows):
        raise ValueError("the held-out manifest holds test rows only")
    os.makedirs(out_dir, exist_ok=True)
    path, tmp = os.path.join(out_dir, TEST_MANIFEST_NAME), os.path.join(out_dir, TEST_MANIFEST_NAME + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (int(r["patient"]), r["eye"])))
    sha = sha256_file(tmp)
    if os.path.exists(path):
        if sha256_file(path) != sha:
            os.remove(tmp)
            raise RuntimeError(f"{path} exists and differs: the held-out test manifest is immutable")
        os.remove(tmp)
    else:
        os.replace(tmp, path)
    summary = {"manifest": TEST_MANIFEST_NAME, "manifest_sha256": sha, "images": len(rows),
               "patients": len({r["patient"] for r in rows}),
               "grade_counts": np.bincount([r["grade"] for r in rows], minlength=5).tolist(),
               "label_file": "EyeQ repository data/Label_EyeQ_test.csv (columns image, DR_grade only)",
               "label_file_sha256": TEST_LABELS_SHA256, "excluded": [],
               "quality_filter": "none (the quality column is not read)",
               "role": "held-out evaluation of the frozen adaptation model; never selection"}
    with open(os.path.join(out_dir, TEST_SUMMARY_NAME), "w") as fh:
        json.dump(summary, fh, indent=1)
    return path, summary


def read_test_manifest(out_dir, expected_sha256=None):
    path = os.path.join(out_dir, TEST_MANIFEST_NAME)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError("the held-out EyePACS test manifest is not the pinned manifest")
    with open(path, newline="") as fh:
        rows = [dict(r, grade=int(r["grade"]), patient_max_grade=int(r["patient_max_grade"])) for r in csv.DictReader(fh)]
    if any(r["split"] != "test" for r in rows):
        raise RuntimeError("the held-out manifest holds a row that is not a test row")
    return rows


# --------------------------------------------------------------------------- frames (P's image path)

def frame_from_raw(raw_path):
    """Stage 2 (DR profile, once, native size) and the canonical 512 frame: float32 (512, 512, 3) in [0, 1].
    The same two functions produce the APTOS cache and the IDRiD inputs."""
    import stage4_v2_aptos_cache as ac
    import stage4_v2_data as sd
    return ac.recompute_rgb_512(sd.stage2_rgb(raw_path))


def assert_not_aptos(path):
    """EyePACS files only: the adaptation cache must never contain an APTOS image."""
    norm = os.path.normpath(path).lower().split(os.sep)
    if "aptos2019" in norm or not os.path.basename(path).endswith(".jpeg"):
        raise RuntimeError(f"{path} is not an EyePACS training image")
    parse_name(os.path.basename(path))
    return True


def cache_identity(manifest_sha256, shard_size=SHARD_SIZE):
    """What a cache is bound to. Two caches with different identities are never interchangeable."""
    import pipeline_v2_config as v2cfg
    return {"version": CACHE_VERSION, "preproc_version": v2cfg.PREPROC_VERSION, "manifest_sha256": manifest_sha256,
            "dtype": FRAME_DTYPE, "frame_size": FRAME_SIZE, "shard_size": int(shard_size)}


def _frame_and_source_hash(path):
    return frame_from_raw(path), sha256_file(path)


def shard_plan(rows, shard_size=SHARD_SIZE):
    """{split: [[image, ...] per shard]} in manifest order (patient id, eye), for the splits present in `rows`."""
    unknown = {r["split"] for r in rows} - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}")
    plan = {}
    for split in SPLITS:
        names = [r["image"] for r in sorted(rows, key=lambda r: (int(r["patient"]), r["eye"])) if r["split"] == split]
        if names:
            plan[split] = [names[i:i + shard_size] for i in range(0, len(names), shard_size)]
    return plan


def verify_sources(cache_dir, image_dir, log=None):
    """Re-hashes every source image a cache was built from and compares with the hashes recorded per shard.
    Raises on the first image that is missing or differs (the cache then no longer describes those files)."""
    with open(os.path.join(cache_dir, "index.json")) as fh:
        shards = json.load(fh)["shards"]
    checked = 0
    for name in sorted(shards):
        entry = shards[name]
        for image, recorded in zip(entry["images"], entry["source_sha256"]):
            path = os.path.join(image_dir, image)
            if not os.path.exists(path) or sha256_file(path) != recorded:
                raise RuntimeError(f"{image}: the source image is missing or differs from the one the cache was built from")
            checked += 1
        if log:
            log(f"  sources of {name} match")
    return checked


def build_cache(rows, train_dir, cache_dir, manifest_sha256, workers=4, shard_size=SHARD_SIZE, limit_shards=None, log=print):
    """Writes the frames as float16 shards `frames_<split>_<k>.npy` (n, 512, 512, 3) with an index. Resumable:
    a shard that exists with its recorded SHA-256 is kept. Stops on the first unreadable image and reports it
    (the inventory is never changed silently)."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(cache_dir, exist_ok=True)
    index_path = os.path.join(cache_dir, "index.json")
    identity = cache_identity(manifest_sha256, shard_size)
    index = dict(identity, shards={})
    if os.path.exists(index_path):
        with open(index_path) as fh:
            old = json.load(fh)
        if any(old.get(k) != v for k, v in identity.items()):
            raise RuntimeError(f"{cache_dir} holds a cache for another manifest or configuration")
        index["shards"] = old["shards"]
    grade_of = {r["image"]: r["grade"] for r in rows}
    done = 0
    for split, shards in shard_plan(rows, shard_size).items():
        for k, names in enumerate(shards):
            name = f"frames_{split}_{k:03d}.npy"
            path = os.path.join(cache_dir, name)
            entry = index["shards"].get(name)
            if entry and os.path.exists(path) and entry["images"] == names and os.path.getsize(path) == entry["bytes"]:
                continue
            if limit_shards is not None and done >= limit_shards:
                continue
            paths = [os.path.join(train_dir, n) for n in names]
            for p in paths:
                assert_not_aptos(p)
            with ThreadPoolExecutor(workers) as ex:
                results = list(ex.map(_frame_and_source_hash, paths))
            block = np.stack([f for f, _ in results]).astype(np.float16)
            if block.shape != (len(names), FRAME_SIZE, FRAME_SIZE, 3) or not np.isfinite(block).all():
                raise RuntimeError(f"{name}: unexpected frames")
            np.save(path + ".tmp.npy", block)
            os.replace(path + ".tmp.npy", path)
            index["shards"][name] = {"split": split, "images": names, "grades": [grade_of[n] for n in names],
                                     "source_sha256": [h for _, h in results],
                                     "sha256": sha256_file(path), "bytes": os.path.getsize(path)}
            with open(index_path + ".tmp", "w") as fh:
                json.dump(index, fh)
            os.replace(index_path + ".tmp", index_path)
            done += 1
            log(f"  {name}: {len(names)} frames")
    return index


# --------------------------------------------------------------------------- verification and reading

def verify_cache(cache_dir, rows, manifest_sha256, shard_size=SHARD_SIZE, check_hashes=True, log=None):
    """Raises unless `cache_dir` holds the COMPLETE cache of exactly `rows` under the identity of
    `cache_identity`: same manifest hash, preprocessing version, dtype, frame size and shard layout; every
    planned shard present with the planned images in the planned order and the manifest's grades; no other
    shard; and (check_hashes) every shard file equal to its recorded SHA-256. Returns the index and a report."""
    index_path = os.path.join(cache_dir, "index.json")
    if not os.path.exists(index_path):
        raise RuntimeError(f"{cache_dir} holds no frame cache")
    with open(index_path) as fh:
        index = json.load(fh)
    identity = cache_identity(manifest_sha256, shard_size)
    different = {k: (index.get(k), v) for k, v in identity.items() if index.get(k) != v}
    if different:
        raise RuntimeError(f"{cache_dir} holds a cache for another manifest or configuration: {different}")
    grade_of = {r["image"]: int(r["grade"]) for r in rows}
    plan = shard_plan(rows, shard_size)
    expected = {f"frames_{split}_{k:03d}.npy": (split, names) for split, shards in plan.items() for k, names in enumerate(shards)}
    extra = sorted(set(index["shards"]) - set(expected))
    missing = sorted(set(expected) - set(index["shards"]))
    if extra or missing:
        raise RuntimeError(f"{cache_dir}: {len(missing)} shards missing (first {missing[:3]}), {len(extra)} unexpected (first {extra[:3]})")
    frames = {split: 0 for split in plan}
    total = 0
    for n, (name, (split, names)) in enumerate(sorted(expected.items())):
        entry = index["shards"][name]
        path = os.path.join(cache_dir, name)
        if entry["split"] != split or entry["images"] != names or entry["grades"] != [grade_of[i] for i in names]:
            raise RuntimeError(f"{name}: images or grades differ from the split manifest")
        if len(entry.get("source_sha256", [])) != len(names):
            raise RuntimeError(f"{name}: the source-file hashes are not recorded")
        if not os.path.exists(path) or os.path.getsize(path) != entry["bytes"]:
            raise RuntimeError(f"{name}: the shard file is missing or has another size")
        if check_hashes and sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"{name}: the shard file does not match its recorded SHA-256")
        frames[split] += len(names)
        total += entry["bytes"]
        if log and ((n + 1) % 5 == 0 or n + 1 == len(expected)):
            log(f"  verified {n + 1}/{len(expected)} shards")
    fingerprint = hashlib.sha256(json.dumps({"identity": identity, "shards": {k: index["shards"][k]["sha256"] for k in sorted(expected)}},
                                            sort_keys=True).encode()).hexdigest()
    return index, {"shards": len(expected), "frames": frames, "bytes": total, "hashes_checked": bool(check_hashes),
                   "identity": identity, "fingerprint": fingerprint}


class FrameCache:
    """Read access to a verified cache: `entries(split)` in manifest order and `frame(image)` as float32
    (512, 512, 3) in [0, 1]. Construction verifies the cache against the split (`verify_cache`)."""

    def __init__(self, cache_dir, rows, manifest_sha256=MANIFEST_SHA256, shard_size=SHARD_SIZE, check_hashes=True, log=None):
        self.cache_dir = cache_dir
        self.manifest_sha256 = manifest_sha256
        index, self.report = verify_cache(cache_dir, rows, manifest_sha256, shard_size, check_hashes, log)
        self.fingerprint = self.report["fingerprint"]
        self._where, self._entries = {}, {split: [] for split in self.report["frames"]}
        for name in sorted(index["shards"]):
            entry = index["shards"][name]
            for row, (image, grade) in enumerate(zip(entry["images"], entry["grades"])):
                parse_name(image)
                self._where[image] = (name, row)
                self._entries[entry["split"]].append((image, int(grade)))
        self._open = {}

    def entries(self, split):
        return list(self._entries[split])

    def frame(self, image):
        name, row = self._where[image]
        block = self._open.get(name)
        if block is None:
            block = self._open[name] = np.load(os.path.join(self.cache_dir, name), mmap_mode="r")
        return np.asarray(block[row], dtype=np.float32)

    def close(self):
        self._open.clear()


def stage_cache(drive_cache_dir, local_cache_dir, log=print, stage_files=None):
    """Copies a cache from Drive to local disk with a SHA-256 check of every shard (resumable; the index is
    written last). Reading shards through the Drive mount during training is not attempted."""
    if stage_files is None:
        import stage4_v2_setup
        stage_files = stage4_v2_setup.stage_files
    with open(os.path.join(drive_cache_dir, "index.json"), "rb") as fh:
        payload = fh.read()
    shards = json.loads(payload.decode("utf-8"))["shards"]
    pairs = [(os.path.join(drive_cache_dir, name), os.path.join(local_cache_dir, name), entry["sha256"])
             for name, entry in sorted(shards.items())]
    os.makedirs(local_cache_dir, exist_ok=True)
    out = stage_files(pairs, log=log, label="EyePACS frame shards")
    with open(os.path.join(local_cache_dir, "index.json.tmp"), "wb") as fh:
        fh.write(payload)
    os.replace(os.path.join(local_cache_dir, "index.json.tmp"), os.path.join(local_cache_dir, "index.json"))
    return out


# --------------------------------------------------------------------------- command line (CPU; no model)

def main(argv=None):
    """`build`: write (or resume) a frame cache from the raw EyePACS folder. `verify`: check a cache against its
    pinned manifest, hashing every shard (and, with --sources, every source image). `--set adaptation` (default)
    is the pinned training / validation split; `--set heldout-test` is the pinned held-out test set."""
    import argparse
    parser = argparse.ArgumentParser(description="EyePACS frame caches (research record 69, 69.2)")
    parser.add_argument("action", choices=("build", "verify"))
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--set", dest="which", choices=("adaptation", "heldout-test"), default="adaptation")
    parser.add_argument("--image-dir", default=None, help="raw image folder (default: the set's folder under datasets/EyePACS/raw)")
    parser.add_argument("--split-dir", default=os.path.join(REPO, "dataset_splits"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-shards", type=int, default=None)
    parser.add_argument("--sources", action="store_true", help="verify: also re-hash every source image")
    args = parser.parse_args(argv)
    if args.which == "adaptation":
        rows, sha, folder = read_split(args.split_dir, MANIFEST_SHA256), MANIFEST_SHA256, "train"
    else:
        rows, sha, folder = read_test_manifest(args.split_dir, TEST_MANIFEST_SHA256), TEST_MANIFEST_SHA256, "test"
    image_dir = args.image_dir or os.path.join(REPO, "datasets", "EyePACS", "raw", folder)
    if args.action == "build":
        build_cache(rows, image_dir, args.cache_dir, sha, workers=args.workers, limit_shards=args.limit_shards)
        if args.limit_shards is not None:
            return 0
    _, report = verify_cache(args.cache_dir, rows, sha, log=print)
    if args.sources:
        report["sources_checked"] = verify_sources(args.cache_dir, image_dir, log=print)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
