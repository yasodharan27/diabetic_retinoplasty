"""DDR grading test set -- population, duplicate handling, near-duplicate screen and acquisition for the restricted
grading evaluation (research record §81, §82; protocol ddr_grading_protocol_v1.json).

No model is imported here and nothing in this module predicts a grade. What it holds:

  * the eligible population, rebuilt from three sources only -- the DDR test label list, the lesion-subset
    manifest and the mirror's file metadata (name, size, SHA-256) at the pinned revision -- with every exclusion
    counted and named: ungradable images, the lesion-segmentation subset (by name and by content), and the extra
    members of byte-identical groups (the lexicographically smallest file name of a group is kept);
  * the two image lists the protocol pins by hash (primary, and the broader sensitivity list);
  * the label-free signals used to examine source effects: the file-name family and chains of file times;
  * the near-duplicate screen against APTOS and the EyePACS adaptation images, with its frozen thresholds and
    decision rule;
  * the download of the listed images from the pinned revision, each checked against its recorded SHA-256. It needs
    an explicit confirmation token.

Images, image lists and everything derived from images stay under the git-ignored datasets/ folder.
"""
import collections
import csv
import datetime
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = "ctmedtech/DDR-dataset"
REVISION = "50541319cfcdc963e3edc8b15508bc822d6b3ab5"
DDR_DIR = os.path.join(HERE, "datasets", "DDR")
AUDIT_DIR = os.path.join(DDR_DIR, "audit")
LABEL_FILE = os.path.join(DDR_DIR, "raw", "DR_grading", "test.txt")
IMAGE_DIR = os.path.join(DDR_DIR, "raw", "DR_grading", "test")
LESION_MANIFEST = os.path.join(AUDIT_DIR, "ddr_lesion_manifest.csv")
FILE_METADATA = os.path.join(AUDIT_DIR, f"ddr_grading_test_files_{REVISION[:8]}.csv")
PRIMARY_LIST = "ddr_grading_primary_v1.csv"
BROADER_LIST = "ddr_grading_broader_v1.csv"
SCREEN_FILE = "ddr_grading_near_duplicate_screen_v1.json"
OFFICIAL_DIRECTORY = os.path.join(AUDIT_DIR, "ddr_official_archive_directory.csv")
OFFICIAL_CHECK_FILE = "ddr_grading_official_crc_check_v1.json"
OFFICIAL_TEST_FOLDER = "DDR-dataset/DR_grading/test/"
UNGRADABLE = 5
LIST_COLUMNS = ("image", "grade", "sha256", "bytes", "family", "lesion_subset")
DOWNLOAD_TOKEN = "DOWNLOAD_DDR_GRADING_TEST_V1"
#: near-duplicate screen (the signature and the two distances of research record §64, §64.1)
SIGNATURE_SIZE = 32
COPY_BAND = 0.01          # below: a copy or re-encoding of the same photograph -> excluded
AMBIGUOUS_BAND = 0.06     # from COPY_BAND up to here: not decidable by distance -> kept, flagged, sensitivity analysis
CHAIN_GAP_SECONDS = 60
SIGNATURE_CACHE = os.path.join(HERE, "results", "EyePACS_APTOS_overlap")
ADAPTATION_MANIFEST = os.path.join(HERE, "dataset_splits", "eyepacs_adaptation_split_v1.csv")


class PopulationError(RuntimeError):
    """The sources, a list or the screen are not what the protocol describes. Nothing is repaired or substituted."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- sources

def read_labels(path=LABEL_FILE):
    """{file name: label} from the DDR list (`<name> <label>` per line; label 5 = ungradable)."""
    labels = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                name, label = line.split()
                if name in labels:
                    raise PopulationError(f"{name} is listed twice in {path}")
                labels[name] = int(label)
    return labels


def read_lesion_manifest(path=LESION_MANIFEST):
    """{file name: {'sha256', 'split'}} for the 757 images of the lesion-segmentation subset."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["image"]: {"sha256": row["sha256"], "split": row["split"]} for row in csv.DictReader(fh)}


def read_file_metadata(path=FILE_METADATA):
    """{file name: {'sha256', 'bytes'}} as recorded from the mirror at the pinned revision."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["image"]: {"sha256": row["sha256"], "bytes": int(row["bytes"])} for row in csv.DictReader(fh)}


def fetch_file_metadata(path=FILE_METADATA, opener=None):
    """Writes the file metadata of DR_grading/test at the pinned revision (Hugging Face tree API: path, size and
    the LFS SHA-256 of every file). METADATA ONLY -- no image is requested."""
    import urllib.request
    opener = opener or urllib.request.urlopen
    url = f"https://huggingface.co/api/datasets/{MIRROR}/tree/{REVISION}/DR_grading/test?limit=1000"
    rows = []
    while url:
        with opener(urllib.request.Request(url, headers={"User-Agent": "ddr-grading-population"}), timeout=60) as response:
            rows += json.loads(response.read().decode("utf-8"))
            link = response.headers.get("Link", "") or ""
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1:part.index(">")]
    files = {}
    for row in rows:
        if row.get("type") == "file":
            lfs = row.get("lfs") or {}
            if not lfs.get("oid"):
                raise PopulationError(f"{row['path']}: the mirror records no SHA-256 for this file")
            files[os.path.basename(row["path"])] = (int(lfs["size"]), lfs["oid"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("image,bytes,sha256\n")
        for name in sorted(files):
            fh.write(f"{name},{files[name][0]},{files[name][1]}\n")
    return len(files)


# --------------------------------------------------------------------------- label-free source signals

def family(name):
    """'007' for the 007-NNNN-SSS names, 'timestamp' for the 17-digit file-time names."""
    if name.startswith("007-"):
        return "007"
    if re.fullmatch(r"\d{17}\.jpg", name):
        return "timestamp"
    raise PopulationError(f"{name}: neither of the two DDR file-name families")


def timestamp_chains(names, gap_seconds=CHAIN_GAP_SECONDS):
    """A unit label per image for the same-visit PROXY: timestamp-named images whose file times (to the second)
    follow one another within `gap_seconds` share a unit; every 007-named image is its own unit. File times are
    not acquisition or patient identifiers -- byte-identical copies carry different times."""
    units = {}
    stamped = sorted((datetime.datetime.strptime(n[:14], "%Y%m%d%H%M%S"), n) for n in names if family(n) == "timestamp")
    chain = -1
    for k, (when, name) in enumerate(stamped):
        if k == 0 or (when - stamped[k - 1][0]).total_seconds() > gap_seconds:
            chain += 1
        units[name] = f"chain{chain:05d}"
    return [units.get(n, f"single:{n}") for n in names]


# --------------------------------------------------------------------------- population

def build_population(labels, lesion, metadata, extra_exclusions=None):
    """The eligible images, in the order of the exclusions the protocol states. Returns
    {'primary': rows, 'broader': rows, 'exclusions': {...}, 'duplicate_groups': {kept: [removed, ...]}}; a row is
    a dict with LIST_COLUMNS. `extra_exclusions` ({name: reason}) is how a LATER, separately versioned population
    removes near-duplicates; version 1 passes none."""
    if set(labels) != set(metadata):
        raise PopulationError("the label list and the mirror's file metadata do not name the same images")
    extra = dict(extra_exclusions or {})
    if set(extra) - set(labels):
        raise PopulationError("an extra exclusion names an image that is not in the test list")
    sha = {n: metadata[n]["sha256"] for n in labels}
    lesion_hashes = {entry["sha256"] for entry in lesion.values()}
    ungradable = sorted(n for n, g in labels.items() if g == UNGRADABLE)
    gradable = sorted(n for n, g in labels.items() if g != UNGRADABLE)
    if any(g not in range(5) for g in (labels[n] for n in gradable)):
        raise PopulationError("a gradable image carries a label outside 0-4")
    lesion_named = [n for n in gradable if n in lesion]
    if any(lesion[n]["sha256"] != sha[n] for n in lesion_named):
        raise PopulationError("a lesion-subset image and the grading image of the same name are different files")
    lesion_content = [n for n in gradable if n not in lesion and sha[n] in lesion_hashes]
    removed_extra = sorted(n for n in gradable if n in extra)

    def deduplicate(names):
        groups = collections.defaultdict(list)
        for n in sorted(names):
            groups[sha[n]].append(n)
        kept = sorted(members[0] for members in groups.values())          # the lexicographically smallest file name
        return kept, {members[0]: members[1:] for members in groups.values() if len(members) > 1}

    remaining = [n for n in gradable if n not in lesion and sha[n] not in lesion_hashes and n not in extra]
    primary, groups = deduplicate(remaining)
    conflicts = [k for k, others in groups.items() if {labels[k]} != {labels[o] for o in others} | {labels[k]}]
    if conflicts:
        raise PopulationError(f"byte-identical images carry different grades: {conflicts[:3]}")
    lesion_kept, lesion_groups = deduplicate([n for n in lesion_named if n not in extra])
    if lesion_groups or {sha[n] for n in lesion_kept} & {sha[n] for n in primary}:
        raise PopulationError("the lesion-subset images are not distinct files outside the primary population")
    broader = sorted(primary + lesion_kept)

    def rows(names):
        return [{"image": n, "grade": labels[n], "sha256": sha[n], "bytes": metadata[n]["bytes"], "family": family(n),
                 "lesion_subset": int(n in lesion)} for n in names]

    return {"primary": rows(primary), "broader": rows(broader), "duplicate_groups": groups,
            "exclusions": {"test_list": len(labels), "ungradable": len(ungradable), "gradable": len(gradable),
                           "lesion_subset_by_name": len(lesion_named), "lesion_subset_by_content": len(lesion_content),
                           "near_duplicate_or_other": removed_extra, "after_lesion_exclusion": len(remaining),
                           "duplicate_groups": len(groups), "duplicate_extras_removed": sum(map(len, groups.values())),
                           "lesion_roles": dict(collections.Counter(lesion[n]["split"] for n in lesion_named))}}


def grade_counts(rows):
    return np.bincount([int(r["grade"]) for r in rows], minlength=5).tolist()


def list_text(rows):
    lines = [",".join(LIST_COLUMNS)] + [",".join(str(r[c]) for c in LIST_COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def write_list(path, rows):
    """Writes one image list (LF line ends, fixed column order) and returns its SHA-256."""
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write(list_text(rows))
    return sha256_file(path)


def read_list(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and tuple(rows[0]) != LIST_COLUMNS:
        raise PopulationError(f"{path}: unexpected columns {tuple(rows[0])}")
    for r in rows:
        r["grade"], r["bytes"], r["lesion_subset"] = int(r["grade"]), int(r["bytes"]), int(r["lesion_subset"])
    return rows


def describe(population):
    """The counts the protocol pins, from a built population."""
    primary, broader = population["primary"], population["broader"]
    by_family = {f: [r for r in primary if r["family"] == f] for f in ("timestamp", "007")}
    units = timestamp_chains([r["image"] for r in primary])
    sizes = collections.Counter(collections.Counter(u for u in units if u.startswith("chain")).values())
    return {"primary": {"n": len(primary), "grades": grade_counts(primary),
                        "by_family": {f: {"n": len(v), "grades": grade_counts(v)} for f, v in by_family.items()}},
            "broader": {"n": len(broader), "grades": grade_counts(broader),
                        "lesion_subset": sum(r["lesion_subset"] for r in broader)},
            "exclusions": population["exclusions"],
            "duplicate_group_sizes": dict(collections.Counter(len(v) + 1 for v in population["duplicate_groups"].values())),
            "same_visit_proxy": {"timestamp_images": len(by_family["timestamp"]), "chains": sum(sizes.values()),
                                 "chains_of_two_or_more": sum(c for size, c in sizes.items() if size > 1),
                                 "longest_chain": max(sizes) if sizes else 0}}


def verify_population(spec, audit_dir=AUDIT_DIR, label_file=LABEL_FILE, lesion_manifest=LESION_MANIFEST, metadata_file=FILE_METADATA):
    """Rebuilds the population from the three source files and requires it to be the one the protocol pins:
    source hashes, every count, and the two stored lists byte for byte. `spec` is the protocol's `data` and
    `populations` sections ({'data': ..., 'populations': ...}). Returns the description; raises on any difference."""
    data, pinned = spec["data"], spec["populations"]
    for label, path, key in (("label file", label_file, "label_file"), ("lesion manifest", lesion_manifest, "lesion_manifest"),
                             ("mirror file metadata", metadata_file, "file_metadata")):
        if not os.path.exists(path):
            raise PopulationError(f"{label} not found: {path}")
        if sha256_file(path) != data[key]["sha256"]:
            raise PopulationError(f"the {label} is not the pinned file")
    population = build_population(read_labels(label_file), read_lesion_manifest(lesion_manifest), read_file_metadata(metadata_file))
    found = describe(population)
    expected = pinned["exclusion_counts"]
    for key in ("test_list", "ungradable", "gradable", "lesion_subset_by_name", "lesion_subset_by_content", "after_lesion_exclusion",
                "duplicate_groups", "duplicate_extras_removed"):
        if found["exclusions"][key] != expected[key]:
            raise PopulationError(f"{key}: {found['exclusions'][key]} found, {expected[key]} pinned")
    for name, rows in (("primary", population["primary"]), ("broader", population["broader"])):
        if len(rows) != pinned[name]["n"] or grade_counts(rows) != pinned[name]["grades_0_to_4"]:
            raise PopulationError(f"the {name} population is {len(rows)} images {grade_counts(rows)}, not the pinned one")
        text = list_text(rows).encode("utf-8")
        if hashlib.sha256(text).hexdigest() != pinned[name]["list_sha256"]:
            raise PopulationError(f"the rebuilt {name} list does not have the pinned SHA-256")
        stored = os.path.join(audit_dir, pinned[name]["list"])
        if not os.path.exists(stored) or sha256_file(stored) != pinned[name]["list_sha256"]:
            raise PopulationError(f"the stored {name} list is missing or is not the pinned file")
    if found["primary"]["by_family"] != {f: {"n": v["n"], "grades": v["grades_0_to_4"]} for f, v in pinned["primary"]["by_family"].items()}:
        raise PopulationError("the file-name families of the primary population are not the pinned ones")
    return found


# --------------------------------------------------------------------------- near-duplicate screen

def signature(path):
    """The screen's image signature (§64): grey image, cropped to the fundus bounding box, 32 x 32, standardised.
    None when the image cannot be read or has no usable fundus area."""
    from PIL import Image
    try:
        im = Image.open(path)
        im.draft("RGB", (320, 320))
        im = im.convert("L")
        im.thumbnail((256, 256), Image.BILINEAR)
        a = np.asarray(im, np.float32)
        mask = a > 12
        if mask.sum() < 100:
            return None
        rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
        crop = im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)).resize((SIGNATURE_SIZE,) * 2, Image.BILINEAR)
        v = np.asarray(crop, np.float32).ravel()
        sd = v.std()
        return None if sd < 1e-3 else ((v - v.mean()) / sd).astype(np.float32)
    except Exception:  # noqa: BLE001 -- an unreadable file is reported by the caller, not skipped silently
        return None


def signatures(paths, workers=6):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(workers) as pool:
        out = list(pool.map(signature, paths, chunksize=32))
    ok = np.asarray([o is not None for o in out], bool)
    return np.stack([o if o is not None else np.zeros(SIGNATURE_SIZE ** 2, np.float32) for o in out]), ok


def nearest(a, b, exclude_self=False, block=512):
    """For every row of `a`: (distance to its nearest row of `b`, that row's index). Distance = L2 / 32, which is
    0 for identical signatures and about 1.41 for unrelated ones."""
    d = float(SIGNATURE_SIZE ** 2)
    best, arg = np.zeros(len(a), np.float32), np.zeros(len(a), np.int64)
    for s in range(0, len(a), block):
        d2 = 2.0 * d - 2.0 * (a[s:s + block] @ b.T)
        if exclude_self:
            d2[np.arange(d2.shape[0]), np.arange(s, s + d2.shape[0])] = np.inf
        arg[s:s + block] = d2.argmin(1)
        best[s:s + block] = np.sqrt(np.maximum(d2.min(1), 0) / d)
    return best, arg


def band(distance):
    """The frozen decision for one image's nearest development image."""
    if distance < COPY_BAND:
        return "copy"
    return "ambiguous" if distance < AMBIGUOUS_BAND else "distinct"


def reference_signatures(cache_dir=SIGNATURE_CACHE, adaptation_manifest=ADAPTATION_MANIFEST):
    """The development images the screen compares against, from the signature files of the §64 screen: every
    image of the APTOS 2019 training set (training and validation splits) and the 35,122 EyePACS adaptation
    images. Returns {name: (signatures, image names, source file SHA-256)}."""
    out = {}
    with open(adaptation_manifest, newline="", encoding="utf-8") as fh:
        adaptation = {row["image"] for row in csv.DictReader(fh)}
    for name, file, keep in (("aptos_train_and_validation", "sig_aptos_train.npz", None),
                             ("eyepacs_adaptation", "sig_eyepacs_train.npz", adaptation)):
        path = os.path.join(cache_dir, file)
        d = np.load(path, allow_pickle=False)
        names = np.asarray([os.path.basename(str(p)) for p in d["paths"]])
        use = np.asarray(d["ok"], bool) & (np.isin(names, sorted(keep)) if keep is not None else True)
        if keep is not None and set(names[use]) != keep:
            raise PopulationError("the stored EyePACS signatures do not cover the adaptation manifest")
        out[name] = (np.asarray(d["sig"], np.float32)[use], names[use].tolist(), sha256_file(path))
    return out


def screen(rows, image_dir, references, list_sha256):
    """The near-duplicate screen of the listed images (label-free: only file names and pixels are used).

    Every listed image must exist with its recorded SHA-256. For each, the distance to its nearest image of every
    reference set; the frozen rule (`band`) then names the images to EXCLUDE (copy band) and to FLAG (ambiguous
    band). Distances between the listed images themselves are reported for information and change nothing."""
    names = [r["image"] for r in rows]
    paths = [os.path.join(image_dir, n) for n in names]
    for r, p in zip(rows, paths):
        if not os.path.exists(p) or sha256_file(p) != r["sha256"]:
            raise PopulationError(f"{r['image']} is missing or is not the listed file")
    sig, ok = signatures(paths)
    report = {"method": "fundus bounding-box crop, grey, 32x32, standardised; distance = L2 / 32 (research record 64, 64.1)",
              "thresholds": {"copy_band_below": COPY_BAND, "ambiguous_band_below": AMBIGUOUS_BAND},
              "rule": "copy band: excluded from every population (a new, versioned population and protocol are issued); "
                      "ambiguous band: kept and flagged, with a sensitivity analysis without the flagged images; "
                      "no visual review changes a decision",
              "list_sha256": list_sha256, "images": len(names), "labels_used": False,
              "no_signature": sorted(n for n, good in zip(names, ok) if not good), "against": {}}
    best = np.full(len(names), np.inf, np.float32)
    partner = [None] * len(names)
    for name, (ref, ref_names, ref_sha) in references.items():
        d, j = nearest(sig, ref)
        d = np.where(ok, d, np.inf).astype(np.float32)
        report["against"][name] = {"images": len(ref_names), "signature_file_sha256": ref_sha, "min": float(d.min()),
                                   "p1": float(np.percentile(d[ok], 1)), "median": float(np.median(d[ok])),
                                   "copy_band": int((d < COPY_BAND).sum()), "ambiguous_band": int(((d >= COPY_BAND) & (d < AMBIGUOUS_BAND)).sum())}
        for k in np.flatnonzero(d < best):
            best[k], partner[k] = d[k], f"{name}:{ref_names[j[k]]}"
    within, _ = nearest(sig[ok], sig[ok], exclude_self=True)
    report["within_listed_images_information_only"] = {"copy_band": int((within < COPY_BAND).sum()),
                                                       "ambiguous_band": int(((within >= COPY_BAND) & (within < AMBIGUOUS_BAND)).sum())}
    close = [{"image": names[k], "nearest": partner[k], "distance": round(float(best[k]), 5), "band": band(best[k])}
             for k in np.argsort(best) if best[k] < AMBIGUOUS_BAND]
    report["exclude"] = sorted(c["image"] for c in close if c["band"] == "copy")
    report["flagged"] = sorted(c["image"] for c in close if c["band"] == "ambiguous")
    report["closest"] = close
    return report


CONTROL_VARIANTS = {"JPEG q95, same size": (1.0, 95), "JPEG q75, same size": (1.0, 75), "75% size, JPEG q90": (0.75, 90),
                    "50% size, JPEG q75": (0.5, 75), "25% size, JPEG q75": (0.25, 75)}
CONTROL_SEED, CONTROL_IMAGES = 20261011, 150


def positive_control(image_dirs, cache_dir=SIGNATURE_CACHE, n=CONTROL_IMAGES, seed=CONTROL_SEED):
    """What the screen can and cannot find, measured on DEVELOPMENT images only (no DDR image, no label, no model):
    `n` images of each development set are re-encoded and resized as JPEG, and the distance of every such copy to
    its original's stored signature is placed in the two bands. `image_dirs`: {signature file: folder of the
    original images}."""
    import tempfile

    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    rng = np.random.default_rng(seed)
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for file, folder in image_dirs.items():
            d = np.load(os.path.join(cache_dir, file), allow_pickle=False)
            sig, ok, names = d["sig"], d["ok"], [os.path.basename(str(p)) for p in d["paths"]]
            pick = [int(i) for i in rng.choice(len(names), 400, replace=False) if ok[i] and os.path.exists(os.path.join(folder, names[i]))][:n]
            distances = {k: [] for k in CONTROL_VARIANTS}
            for i in pick:
                im = Image.open(os.path.join(folder, names[i])).convert("RGB")
                for k, (scale, quality) in CONTROL_VARIANTS.items():
                    v = im if scale == 1.0 else im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
                    v.save(os.path.join(tmp, "copy.jpg"), quality=quality)
                    s = signature(os.path.join(tmp, "copy.jpg"))
                    distances[k].append(np.inf if s is None else float(np.sqrt(max(2.0 * s.size - 2.0 * float(s @ sig[i]), 0.0) / s.size)))
            out[file] = {"images": len(pick)}
            for k, values in distances.items():
                v = np.asarray(values)
                out[file][k] = {"median": round(float(np.median(v)), 4), "p95": round(float(np.percentile(v, 95)), 4), "max": round(float(v.max()), 4),
                                "copy_band": int((v < COPY_BAND).sum()), "ambiguous_band": int(((v >= COPY_BAND) & (v < AMBIGUOUS_BAND)).sum()),
                                "outside_both": int((v >= AMBIGUOUS_BAND).sum())}
    return out


# --------------------------------------------------------------------------- the official release's own directory

def parse_zip_directory(tail, total_size):
    """[(path, uncompressed bytes, CRC-32 as 8 hex digits, compression method)] for every file of a zip archive,
    read from its LAST bytes only (`tail`; `total_size` is the size of the whole archive). The official DDR release
    is one stored (uncompressed) zip in ten chunks: its directory sits at the end of the last chunk, so the names,
    sizes and CRC-32 of all files can be read without downloading the images. Plain parsing; nothing is extracted."""
    import struct
    base = total_size - len(tail)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise PopulationError("no end-of-central-directory record in the given bytes")
    n_total, size, offset = struct.unpack("<HII", tail[eocd + 10:eocd + 20])
    if n_total == 0xFFFF or 0xFFFFFFFF in (size, offset):
        locator = tail.rfind(b"PK\x06\x07", 0, eocd)
        record = (struct.unpack("<Q", tail[locator + 8:locator + 16])[0] - base) if locator >= 0 else -1
        if record < 0 or tail[record:record + 4] != b"PK\x06\x06":
            raise PopulationError("the zip64 directory record is not inside the given bytes")
        n_total, size, offset = struct.unpack("<QQQ", tail[record + 32:record + 56])
    p = offset - base
    if p < 0:
        raise PopulationError("the central directory begins before the given bytes")
    entries = []
    for _ in range(n_total):
        if tail[p:p + 4] != b"PK\x01\x02":
            raise PopulationError("the central directory is damaged")
        flags, method = struct.unpack("<HH", tail[p + 8:p + 12])
        crc, _packed, plain, name_len, extra_len, comment_len = struct.unpack("<IIIHHH", tail[p + 16:p + 34])
        name = tail[p + 46:p + 46 + name_len].decode("utf-8" if flags & 0x800 else "cp437", "replace")
        extra, q = tail[p + 46 + name_len:p + 46 + name_len + extra_len], 0
        while q + 4 <= len(extra):
            tag, length = struct.unpack("<HH", extra[q:q + 4])
            if tag == 1 and plain == 0xFFFFFFFF:
                plain = struct.unpack("<Q", extra[q + 4:q + 12])[0]
            q += 4 + length
        if not name.endswith("/"):
            entries.append((name, plain, f"{crc:08x}", method))
        p += 46 + name_len + extra_len + comment_len
    return entries


def write_official_directory(tail_path, total_size, path=OFFICIAL_DIRECTORY):
    with open(tail_path, "rb") as fh:
        entries = parse_zip_directory(fh.read(), total_size)
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("path,bytes,crc32\n")
        for name, plain, crc, _ in entries:
            fh.write(f"{name},{plain},{crc}\n")
    return len(entries)


def read_official_directory(path=OFFICIAL_DIRECTORY, folder=OFFICIAL_TEST_FOLDER):
    """{file name: (bytes, crc32)} for the files of one folder of the official archive."""
    with open(path, newline="", encoding="utf-8") as fh:
        return {row["path"][len(folder):]: (int(row["bytes"]), row["crc32"]) for row in csv.DictReader(fh) if row["path"].startswith(folder)}


def crc32_file(path):
    import zlib
    value = 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            value = zlib.crc32(block, value)
    return f"{value & 0xFFFFFFFF:08x}"


def official_check(rows, image_dir, official, list_sha256, directory_sha256):
    """Every listed image against the official archive's directory entry of the same name: size and CRC-32.
    Agreement is evidence that the mirror's file is the official one; a CRC-32 is 32 bits and is NOT a proof of
    byte identity. The SHA-256 of each file is checked first, so the comparison is of the listed file."""
    mismatches, missing = [], []
    for row in rows:
        path = os.path.join(image_dir, row["image"])
        if not os.path.exists(path) or sha256_file(path) != row["sha256"]:
            raise PopulationError(f"{row['image']} is missing or is not the listed file")
        entry = official.get(row["image"])
        if entry is None:
            missing.append(row["image"])
        elif entry != (os.path.getsize(path), crc32_file(path)):
            mismatches.append(row["image"])
    return {"list_sha256": list_sha256, "official_directory_sha256": directory_sha256, "images": len(rows),
            "size_and_crc32_equal": len(rows) - len(mismatches) - len(missing), "mismatches": mismatches, "not_in_official_directory": missing,
            "meaning": "size and CRC-32 agreement with the official archive's directory; not a proof of byte identity"}


# --------------------------------------------------------------------------- acquisition

def download(rows, image_dir=IMAGE_DIR, *, confirm, opener=None, log=print):
    """Fetches the listed images from the pinned revision, one file each, and keeps a file only when its SHA-256
    is the listed one. A file already present with the right hash is not fetched again. Nothing but the listed
    images is requested; ungradable images are never listed."""
    if confirm != DOWNLOAD_TOKEN:
        raise PopulationError("the DDR test images are downloaded only after approval: pass confirm=<the download token>")
    import urllib.parse
    import urllib.request
    opener = opener or urllib.request.urlopen
    os.makedirs(image_dir, exist_ok=True)
    fetched = 0
    for n, row in enumerate(rows, start=1):
        path = os.path.join(image_dir, row["image"])
        if os.path.exists(path) and sha256_file(path) == row["sha256"]:
            continue
        url = f"https://huggingface.co/datasets/{MIRROR}/resolve/{REVISION}/DR_grading/test/{urllib.parse.quote(row['image'])}"
        part = path + ".part"
        with opener(urllib.request.Request(url, headers={"User-Agent": "ddr-grading-evaluation"}), timeout=120) as response, \
                open(part, "wb") as fh:
            for block in iter(lambda: response.read(1 << 20), b""):
                fh.write(block)
        if sha256_file(part) != row["sha256"]:
            os.remove(part)
            raise PopulationError(f"{row['image']}: the downloaded file does not have the listed SHA-256")
        os.replace(part, path)
        fetched += 1
        if n % 200 == 0 or n == len(rows):
            log(f"  {n}/{len(rows)} images present ({fetched} fetched in this call)")
    return fetched


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="DDR grading population: metadata, lists, verification, screen, download")
    parser.add_argument("command", choices=("metadata", "build", "verify", "control", "official", "screen", "download"))
    parser.add_argument("--protocol", default=os.path.join(HERE, "ddr_grading_protocol_v1.json"))
    parser.add_argument("--confirm", default=None)
    args = parser.parse_args(argv)
    if args.command == "metadata":
        print("files recorded:", fetch_file_metadata(), "| sha256", sha256_file(FILE_METADATA))
        return
    if args.command == "build":
        population = build_population(read_labels(), read_lesion_manifest(), read_file_metadata())
        for name, file in (("primary", PRIMARY_LIST), ("broader", BROADER_LIST)):
            print(name, len(population[name]), grade_counts(population[name]), "| sha256", write_list(os.path.join(AUDIT_DIR, file), population[name]))
        print(json.dumps(describe(population), indent=1))
        return
    if args.command == "control":
        report = positive_control({"sig_aptos_train.npz": os.path.join(HERE, "datasets", "APTOS2019", "raw", "train_images"),
                                   "sig_eyepacs_train.npz": os.path.join(HERE, "datasets", "EyePACS", "raw", "train")})
        with open(os.path.join(AUDIT_DIR, "ddr_grading_screen_positive_control.json"), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(json.dumps(report, indent=1))
        return
    with open(args.protocol, encoding="utf-8") as fh:
        protocol = json.load(fh)
    found = verify_population(protocol)
    if args.command == "verify":
        print(json.dumps(found, indent=1))
        print("POPULATION VERIFIED")
        return
    rows = read_list(os.path.join(AUDIT_DIR, protocol["populations"]["broader"]["list"]))
    if args.command == "download":
        download(rows, confirm=args.confirm)
        return
    if args.command == "official":
        report = official_check(rows, IMAGE_DIR, read_official_directory(), protocol["populations"]["broader"]["list_sha256"],
                                sha256_file(OFFICIAL_DIRECTORY))
        with open(os.path.join(AUDIT_DIR, OFFICIAL_CHECK_FILE), "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(json.dumps({k: v for k, v in report.items() if k not in ("mismatches", "not_in_official_directory")}, indent=1),
              "| mismatches", len(report["mismatches"]), "| not in the official directory", len(report["not_in_official_directory"]))
        return
    report = screen(rows, IMAGE_DIR, reference_signatures(), protocol["populations"]["broader"]["list_sha256"])
    with open(os.path.join(AUDIT_DIR, SCREEN_FILE), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: report[k] for k in ("against", "exclude", "flagged", "no_signature")}, indent=1))


if __name__ == "__main__":
    main()
