"""
The ONE authoritative train/validation split of APTOS2019's labeled
classification images (`datasets/APTOS2019/raw/train.csv`), shared by every
downstream trainable stage that reads that dataset -- Stage 05 (Local
Feature Extraction), Stage 06 (Global Feature Extraction), and eventually
Stage 07 / RACAF / CORN's joint training script.

This is deliberately a standalone, stage-agnostic infrastructure module, not
something owned by any one stage's model code -- `local_feature_extraction_dataset.py`
previously defined this split itself (`split_train_val_ids`) and
`global_feature_extraction_dataset.py` re-exported it; both now delegate to
this module instead, so there is exactly one place the partition is computed,
regardless of how many stages eventually need it (PROJECT_CODE.md's Modular
Stage Principle -- a data contract, not a per-stage implementation detail).

Two properties, both required, not just one:
  - Deterministic / reconstructable: given the same `train.csv`, `val_split`,
    and `seed`, `compute_split()` always returns the identical partition (a
    seeded, stratified `sklearn.train_test_split` -- stratified so each DR
    grade is represented proportionally in both halves, which the prior
    interim split did not do).
  - Saved: `get_authoritative_split()` persists the *default* split (the one
    every stage should actually use) to a small, version-controlled CSV
    manifest (`dataset_splits/aptos2019_train_val_split.csv`) the first time
    it runs, and reads that file on every subsequent call/environment/session
    instead of recomputing it -- removing any dependency on identical
    scikit-learn behavior across the local machine and every future Colab
    session. Non-default parameters (e.g. a test's own synthetic CSV) always
    compute fresh and never touch the manifest.

The manifest lives outside `datasets/` (gitignored entirely -- real dataset
content is never committed, PROJECT_CODE.md's Training policy) and outside
`results/`/`models/` (also gitignored, reserved for regenerable model
outputs) so it is an actual, checked-in, cross-environment source of truth --
a small (id_code, diagnosis, split) manifest, not "a dataset."
"""

import csv
import os

from sklearn.model_selection import train_test_split

import config

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

APTOS_RAW_DIR = config.dataset_raw_dir("APTOS2019")
DEFAULT_TRAIN_CSV = os.path.join(APTOS_RAW_DIR, "train.csv")

# Unchanged from the prior interim split (local_feature_extraction_dataset.py's
# original DEFAULT_VAL_SPLIT/DEFAULT_SEED) -- this module promotes that same
# ratio/seed to authoritative status and adds stratification; it does not
# silently pick a different ratio.
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42

DEFAULT_SPLIT_DIR = config.DOWNSTREAM_SPLIT_DIR
DEFAULT_SPLIT_MANIFEST = os.path.join(DEFAULT_SPLIT_DIR, "aptos2019_train_val_split.csv")

_MANIFEST_FIELDNAMES = ("id_code", "diagnosis", "split")


def list_labeled_images(csv_path=DEFAULT_TRAIN_CSV):
    """Reads `csv_path` (APTOS2019's train.csv: `id_code`, `diagnosis`
    columns), returning a sorted list of `(id_code, diagnosis)` with
    `diagnosis` already cast to `int`. Deliberately not imported from
    `local_feature_extraction_dataset.py` -- that module now delegates *to*
    this one (see module docstring), so importing back from it would be
    circular; this is the authoritative reader instead."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"APTOS2019 labeled CSV not found at {csv_path}. Expected the raw APTOS2019 "
            "train.csv to already be present under datasets/APTOS2019/raw/ "
            "(PROJECT_CODE.md's Dataset Handling rule -- datasets are never auto-downloaded)."
        )
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append((row["id_code"], int(row["diagnosis"])))
    if not entries:
        raise FileNotFoundError(f"No labeled rows found in {csv_path}.")
    return sorted(entries)


def compute_split(csv_path=DEFAULT_TRAIN_CSV, val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED):
    """Deterministic, stratified train/val split of `csv_path`'s labeled
    entries -- `train_test_split` with `stratify=` on the diagnosis column,
    so each of the 5 DR grades is represented proportionally in both halves
    (the prior interim split in `local_feature_extraction_dataset.py` did
    not stratify). Same seeded mechanism otherwise (fixed `random_state`,
    no overlap by construction). Pure function -- no disk I/O beyond reading
    `csv_path` itself; does not touch the saved manifest (see
    `get_authoritative_split` for that)."""
    entries = list_labeled_images(csv_path)
    labels = [diagnosis for _, diagnosis in entries]
    train_entries, val_entries = train_test_split(
        entries, test_size=val_split, random_state=seed, stratify=labels,
    )
    return sorted(train_entries), sorted(val_entries)


def save_split(train_entries, val_entries, path=DEFAULT_SPLIT_MANIFEST):
    """Writes `train_entries`/`val_entries` (each a list of `(id_code,
    diagnosis)`) to `path` as a single `id_code,diagnosis,split` CSV, sorted
    by `id_code` -- one manifest covering both halves, so `split` membership
    is never ambiguous or split across two files that could drift apart."""
    rows = sorted(
        [(id_code, diagnosis, "train") for id_code, diagnosis in train_entries]
        + [(id_code, diagnosis, "val") for id_code, diagnosis in val_entries]
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_MANIFEST_FIELDNAMES)
        for id_code, diagnosis, split in rows:
            writer.writerow([id_code, diagnosis, split])
    return path


def load_split(path=DEFAULT_SPLIT_MANIFEST):
    """Reads a manifest written by `save_split` back into `(train_entries,
    val_entries)`, each sorted `(id_code, diagnosis)` lists -- the inverse
    of `save_split`."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Authoritative split manifest not found at {path}.")
    train_entries, val_entries = [], []
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entry = (row["id_code"], int(row["diagnosis"]))
            if row["split"] == "train":
                train_entries.append(entry)
            elif row["split"] == "val":
                val_entries.append(entry)
            else:
                raise ValueError(f"Unknown split label {row['split']!r} in {path} for {row['id_code']!r}.")
    return sorted(train_entries), sorted(val_entries)


def get_authoritative_split(csv_path=DEFAULT_TRAIN_CSV, val_split=DEFAULT_VAL_SPLIT,
                             seed=DEFAULT_SEED, manifest_path=DEFAULT_SPLIT_MANIFEST):
    """The single entry point every downstream stage should call.

    The one shared, committed manifest (`manifest_path` left at its default,
    `DEFAULT_SPLIT_MANIFEST`) is only ever read or written when the request
    is unambiguously "give me *the* authoritative split" -- `csv_path`,
    `val_split`, and `seed` must ALL match their defaults too. This is what
    protects the real, committed manifest from being corrupted by a
    differently-parameterized caller (a test's own synthetic CSV, or a
    deliberately different ratio/seed): such a call always computes fresh
    via `compute_split` and never touches `DEFAULT_SPLIT_MANIFEST` at all.

    A caller who explicitly passes their OWN `manifest_path` (anything
    other than `DEFAULT_SPLIT_MANIFEST`) is opting into the same
    load-if-present/compute-and-save-otherwise caching behavior at that
    other location, regardless of `csv_path` -- there is no shared state to
    protect there, so no additional restriction applies.
    """
    if manifest_path == DEFAULT_SPLIT_MANIFEST:
        is_default_request = (
            csv_path == DEFAULT_TRAIN_CSV and val_split == DEFAULT_VAL_SPLIT and seed == DEFAULT_SEED
        )
        if not is_default_request:
            return compute_split(csv_path, val_split=val_split, seed=seed)

    if os.path.exists(manifest_path):
        return load_split(manifest_path)

    train_entries, val_entries = compute_split(csv_path, val_split=val_split, seed=seed)
    save_split(train_entries, val_entries, manifest_path)
    return train_entries, val_entries
