"""
Multi-seed RACAF vs NO-RACAF improved-training experiment orchestration -- an ADDITIVE module.

Nothing in `training/`, `joint_training_model.py`, `joint_training_dataset.py`, `corn.py`,
`racaf.py`, or `downstream_split.py` is modified. This module composes those UNCHANGED pieces
(`training.Trainer`/`training.checkpointing`/`training.callbacks` for checkpoint/resume/
early-stopping/LR-scheduling; `improved_training_data` for the epoch-indexed, position-
independent data stream; `weighted_corn` for the class-weighted CORN loss; `no_racaf_model` and
`joint_training_model` for the two arms) into the six matched, independently resumable runs
this experiment needs.

Six runs: {RACAF, NO_RACAF} x seed in {42, 123, 2026}, all sharing the SAME committed APTOS2019
split (`SPLIT_SEED`, always 42 -- see `verify_split()`), the same cache, and the same protocol.

--- The key resume design decision -----------------------------------------------------------

`joint_training_dataset._make_joint_dataset()`'s existing `tf.data.Dataset` re-seeds its
augmentation RNG on every generator iteration (`JOINT_TRAINING_ARCHITECTURE.md` Sec 49) and its
shuffle order is not resume-aware. `improved_training_data.make_epoch_dataset()` fixes both by
making the data order and every image's augmentation a pure function of `(run_seed, epoch,
image_id)` -- so a given epoch's stream is identical however many times it is rebuilt.

That means each Keras epoch needs its OWN freshly built dataset object, so `train_run()` below
calls `model.fit(train_ds, validation_data=val_ds, epochs=e+1, initial_epoch=e, callbacks=...)`
ONCE PER EPOCH in a plain Python loop, rebuilding `train_ds` for epoch `e` each time, instead of
the usual single `model.fit(epochs=N)` call `training.Trainer.fit()` makes. The SAME `Trainer`-
built callback list (`EarlyStopping`, `ReduceLROnPlateau`, `training.TrainingStateCheckpoint`) is
reused, UNCHANGED, across every iteration of that loop.

This works because `TrainingConfig(resume=True)` is used UNCONDITIONALLY, for every run, from
its very first epoch onward -- not only when genuinely continuing a previous session:

  - Keras resets `EarlyStopping`/`ReduceLROnPlateau`'s counters in `on_train_begin` on EVERY
    `model.fit()` call, including each of this loop's one-epoch calls.
  - `training.TrainingStateCheckpoint.on_train_begin` (placed AFTER those two in the callback
    list, per `training.callbacks.build_callbacks()`) immediately restores the persisted
    counters/global-best -- but ONLY when `restore_state=True` (wired from `config.resume`).
  - With `resume=True` fixed, this restore fires after EVERY one-epoch `fit()` call, correctly
    carrying `EarlyStopping.wait`, `ReduceLROnPlateau.wait`/`cooldown_counter`, the optimizer, and
    the global best epoch-to-epoch within one process AND across a completely fresh Colab
    runtime, using the exact same code path either way.
  - On a genuinely fresh run (no checkpoint exists yet), `find_resumable_generation()` returns
    `None` and there is no checkpoint evidence, so the restore step is a harmless no-op --
    `TrainingConfig(resume=True)` on a brand-new run behaves identically to `resume=False` there.

So this module never has to reimplement checkpoint/callback-state persistence: it is entirely
`training/checkpointing.py`'s and `training/callbacks.py`'s existing, tested machinery, invoked
once per epoch instead of once per multi-epoch run.

--- What THIS module adds on top -----------------------------------------------------------

`training/checkpointing.py` already gives per-epoch generations (LAST) with a single, delete-
then-copy `best/` (BEST). This experiment needs more than that:

  - a two-slot BEST (`best_a/`/`best_b/` + `best.json`) so a crash can never leave the run
    without ANY valid BEST, unlike `ckpt.save_best()`'s delete-then-write;
  - a `stop_decision.json` sidecar recording WHY a run stopped (`early_stopping`/`epoch_cap`), so
    a resumed session that finds it never trains one more epoch after a sealed stop decision;
  - one immutable `run_manifest.json` per run (identity: experiment id, arm, seed, split hash,
    config hash, environment) and one frozen `PREREGISTRATION.json` per experiment;
  - a lightweight, heartbeat-based single-writer lock, so two Colab runtimes cannot train the
    same run at once without an explicit override;
  - one small JSON file per completed epoch under `history/`, built from that epoch's own sealed
    `state.json` -- never an appended CSV, which cannot be made crash-safe against a duplicate row.

None of this touches `training/checkpointing.py`'s own generation format, or any other production
module -- everything here is new, additive orchestration on top of unchanged infrastructure.

One consequence of reusing `training.TrainingStateCheckpoint` completely unmodified: its own
`on_epoch_end` still calls `ckpt.save_best()` (the original, single-slot, delete-then-copy `best/`)
on every improving epoch, exactly as it always has -- that call is never suppressed. `train_run()`
below publishes THIS module's two-slot BEST (`best_a`/`best_b`/`best.json`) as an explicit,
separate step alongside it. The legacy `checkpoints/best/` this produces is harmless but
unused: `read_best()`/the evaluation and comparison code read only `best.json`.
"""

import ast
import hashlib
import io
import json
import os
import platform
import shutil
import time
import tokenize
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

import corn
import downstream_split
import joint_training_model as jtm
import no_racaf_model
import weighted_corn
import improved_training_data as itd
from training import (
    Trainer, TrainingConfig, CheckpointOptions, TrainingStateCheckpoint,
    enable_mixed_precision, model_precision_policies, expected_policy_name,
)
from training import checkpointing as ckpt
from training.trainer import precision_is_consistent

# =====================================================================================
# 1. Fixed protocol constants -- the pre-registered configuration, not free parameters.
# =====================================================================================

PROTOCOL_VERSION = "improved-multiseed-v1"
ARMS = ("RACAF", "NO_RACAF")
RUN_SEEDS = (42, 123, 2026)
#: The authoritative dataset/split seed -- NEVER a run seed. `verify_split()` below hardcodes
#: this rather than accepting it as a parameter, so there is no call signature through which a
#: RUN_SEED could ever be mistaken for the split seed (the audited experiment's #1 MUST-FIX item).
SPLIT_SEED = 42
EXPECTED_SPLIT_SHA256 = "bc80fd450340b09307fbd80a1b00553e70e34d64a3cdf94635162b6c1e99aca5"
EXPECTED_TRAIN_COUNT = 2929
EXPECTED_VAL_COUNT = 733

BATCH_SIZE = 2
MAX_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
WEIGHT_DECAY_EXCLUDE_NAMES = ("bias", "gamma", "beta")
MONITOR_METRIC = "val_QWK"
MONITOR_MODE = "max"
EARLY_STOPPING_PATIENCE = 12
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.5
MIN_LR = 1e-6

DEFAULT_LOCK_TTL_SECONDS = 1800  # 30 min; comfortably longer than one measured ~8-13 min epoch.

#: Stable for the lifetime of this Python process/Colab kernel -- computed once at import time,
#: not per call. This is what makes re-running a training cell in the SAME runtime look like the
#: SAME lock owner (no spurious RunLockedError), while a genuinely NEW runtime (a fresh process,
#: hence a fresh import of this module) gets a different owner id and correctly sees the previous
#: runtime's lock as either fresh (refused) or stale (recoverable).
OWNER_ID = f"{platform.node()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"


class RunLockedError(RuntimeError):
    """Another runtime appears to be actively training this run."""


class RunConfigurationError(RuntimeError):
    """A run's persisted identity does not match what this session is configured for."""


def _now():
    import datetime
    return datetime.datetime.now().isoformat(timespec="seconds")


def _atomic_write_json(path, payload):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}-{uuid.uuid4().hex[:6]}"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(tmp, path)


def _read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


# =====================================================================================
# 2. Directory layout
# =====================================================================================
#
# experiments/ImprovedTraining/<experiment_id>/
#     PREREGISTRATION.json
#     experiment_manifest.json
#     RACAF/seed_42/ seed_123/ seed_2026/
#     NO_RACAF/seed_42/ seed_123/ seed_2026/
#         run_manifest.json      -- immutable identity, written once
#         training_behavior.json  -- active training-behaviour fingerprint + restart/commit events
#         superseded/restart_NNN/ -- a previous trajectory, moved aside when training behaviour changed
#         LOCK.json               -- heartbeat lease
#         stop_decision.json      -- present only once EarlyStopping/the epoch cap has decided
#         checkpoints/            -- gen_NNNNN/ (LAST), best_a/ best_b/ best.json (BEST)
#         history/                -- epoch_0001.json, epoch_0002.json, ...
#         logs/                   -- TensorBoard scalars only (histogram_freq=0)
#         evaluation/             -- per_sample_{best,last}.csv/.json, metrics_{best,last}.json

ARM_DIRNAMES = {"RACAF": "RACAF", "NO_RACAF": "NO_RACAF"}


def experiment_root(experiments_root, experiment_id):
    return os.path.join(experiments_root, experiment_id)


def run_dir(experiments_root, experiment_id, arm, run_seed):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    if run_seed not in RUN_SEEDS:
        raise ValueError(f"run_seed must be one of {RUN_SEEDS}, got {run_seed!r}")
    return os.path.join(experiment_root(experiments_root, experiment_id),
                        ARM_DIRNAMES[arm], f"seed_{run_seed}")


def all_run_ids():
    """Every (arm, run_seed) pair this experiment trains, in the pre-registered order: matched
    pairs seed by seed -- RACAF 42, NO_RACAF 42, RACAF 123, NO_RACAF 123, RACAF 2026, NO_RACAF 2026."""
    return [(arm, seed) for seed in RUN_SEEDS for arm in ARMS]


def ensure_run_dir(path):
    """Idempotent: creates the run's subfolders if missing, never touches existing content."""
    for sub in ("checkpoints", "history", "logs", "evaluation"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    return path


# =====================================================================================
# 3. Split verification -- SPLIT_SEED only, hardcoded, never a caller-supplied parameter.
# =====================================================================================

def verify_split():
    """Reads the authoritative, committed APTOS2019 split with the fixed `SPLIT_SEED` (42) --
    never a run seed, and never accepts one as a parameter -- verifies its file hash and the
    2929/733 counts, and returns `(train_entries, val_entries, split_sha256)`."""
    csv_path = downstream_split.DEFAULT_SPLIT_MANIFEST
    if not os.path.exists(csv_path):
        raise RunConfigurationError(f"Split manifest not found at {csv_path}.")
    with open(csv_path, "rb") as handle:
        raw = handle.read()
    sha256 = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if sha256 != EXPECTED_SPLIT_SHA256:
        raise RunConfigurationError(
            f"Split manifest sha256 {sha256} does not match the pinned "
            f"{EXPECTED_SPLIT_SHA256} at {csv_path}. Refusing to proceed with an unverified split."
        )
    train_entries, val_entries = downstream_split.get_authoritative_split(seed=SPLIT_SEED)
    if (len(train_entries), len(val_entries)) != (EXPECTED_TRAIN_COUNT, EXPECTED_VAL_COUNT):
        raise RunConfigurationError(
            f"Split counts ({len(train_entries)}, {len(val_entries)}) do not match the expected "
            f"({EXPECTED_TRAIN_COUNT}, {EXPECTED_VAL_COUNT})."
        )
    return train_entries, val_entries, sha256


# =====================================================================================
# 4. Pre-registration
# =====================================================================================

PREREGISTRATION_FILENAME = "PREREGISTRATION.json"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"


def build_preregistration(experiment_id):
    return {
        "experiment_id": experiment_id,
        "protocol_version": PROTOCOL_VERSION,
        "arms": list(ARMS),
        "run_seeds": list(RUN_SEEDS),
        "split_seed": SPLIT_SEED,
        "split_sha256": EXPECTED_SPLIT_SHA256,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "weight_decay_exclude_name_substrings": list(WEIGHT_DECAY_EXCLUDE_NAMES),
        "weight_decay_applies_to_racaf_gate": True,
        "class_weighting_method": "sqrt_inverse_frequency",
        "class_weight_power": weighted_corn.DEFAULT_WEIGHT_POWER,
        "class_weight_train_counts": list(weighted_corn.PREREGISTERED_TRAIN_COUNTS),
        "class_weights": list(weighted_corn.PREREGISTERED_CLASS_WEIGHTS),
        "reduce_lr_on_plateau": {
            "monitor": MONITOR_METRIC, "mode": MONITOR_MODE, "patience": REDUCE_LR_PATIENCE,
            "factor": REDUCE_LR_FACTOR, "min_lr": MIN_LR,
        },
        "early_stopping": {
            "monitor": MONITOR_METRIC, "mode": MONITOR_MODE, "patience": EARLY_STOPPING_PATIENCE,
            "restore_best_weights": False,
        },
        "primary_metric": MONITOR_METRIC,
        "delta_definition": "Delta = QWK_RACAF - QWK_NO_RACAF (positive means RACAF is better)",
        "primary_comparison_rule": {
            "support_racaf": "mean(delta) > 0 AND all 3 seed-level deltas > 0 AND the paired "
                             "per-image bootstrap CI excludes zero in at least 2 of 3 seeds",
            "support_no_racaf": "the mirror condition with every sign reversed",
            "otherwise": "INCONCLUSIVE",
        },
        "secondary_metrics": [
            "accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "MAE",
            "confusion_matrix", "per_grade_precision_recall_f1", "prediction_histogram",
            "error_distance_distribution", "grade_3_recall", "grade_4_recall",
            "errors_of_2_or_more_grades", "mcnemar_paired_correctness", "ece", "brier",
        ],
        "duplicate_excluded_analysis": (
            "secondary; reuses the existing pinned 41-image train/validation duplicate list "
            "(docs/experiments/RACAF_Duplicate_Contamination_Audit.md), not rediscovered"
        ),
        "idrid_external_evaluation": (
            "explicitly out of scope for this notebook; a separate downstream step using the "
            "existing colab/notebooks/idrid_external_evaluation.ipynb, applied to all 6 BEST "
            "checkpoints once this experiment completes, never a hand-picked seed"
        ),
        "notes": (
            "This is a bundled protocol change (augmentation-RNG fix + class-weighted CORN loss "
            "+ AdamW weight decay) compared against the finalized single RACAF/NO-RACAF runs. It "
            "cannot isolate the individual contribution of any one of those three changes. Three "
            "seeds give very limited seed-level statistical power (2 degrees of freedom). "
            "Validation-based BEST selection is optimistic for both arms equally."
        ),
    }


def write_preregistration(path, experiment_id):
    """Writes `PREREGISTRATION.json`. Refuses to overwrite an existing one -- the
    pre-registration is frozen the first time it is written, exactly as its own name promises.
    Returns `(preregistration_dict, sha256_hex)`."""
    if os.path.exists(path):
        raise RunConfigurationError(
            f"{path} already exists. The pre-registration is frozen once written and must not "
            "be modified after the first training run -- read it with "
            "load_and_verify_preregistration() instead."
        )
    prereg = build_preregistration(experiment_id)
    payload_bytes = json.dumps(prereg, indent=2, sort_keys=True).encode("utf-8")
    sha256 = hashlib.sha256(payload_bytes).hexdigest()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    with open(tmp, "wb") as handle:
        handle.write(payload_bytes)
    os.replace(tmp, path)
    return prereg, sha256


def load_and_verify_preregistration(path):
    """Reads `PREREGISTRATION.json` and verifies it is still byte-identical to its own canonical
    serialization -- catching a hand-edit after the fact, which would otherwise silently change
    the experiment's frozen protocol mid-run. Returns `(preregistration_dict, sha256_hex)`."""
    if not os.path.exists(path):
        raise RunConfigurationError(f"No PREREGISTRATION.json at {path} -- write it before any run.")
    with open(path, "rb") as handle:
        payload_bytes = handle.read()
    prereg = json.loads(payload_bytes)
    recomputed = json.dumps(prereg, indent=2, sort_keys=True).encode("utf-8")
    if recomputed != payload_bytes:
        raise RunConfigurationError(
            f"{path} is not byte-identical to its own canonical JSON serialization -- it may "
            "have been hand-edited after being written. Refusing to trust a possibly-altered "
            "pre-registration."
        )
    return prereg, hashlib.sha256(payload_bytes).hexdigest()


# =====================================================================================
# 5. Run identity: config hash + immutable run manifest
# =====================================================================================

RUN_MANIFEST_FILENAME = "run_manifest.json"


def run_config_mapping(experiment_id, arm, run_seed, split_sha256):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "experiment_id": experiment_id,
        "arm": arm,
        "run_seed": run_seed,
        "split_seed": SPLIT_SEED,
        "split_sha256": split_sha256,
        "model": "joint_stage05_08_racaf" if arm == "RACAF" else no_racaf_model.NO_RACAF_MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "loss": "weighted_corn_loss",
        "class_weight_power": weighted_corn.DEFAULT_WEIGHT_POWER,
        "class_weights": list(weighted_corn.PREREGISTERED_CLASS_WEIGHTS),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "weight_decay_exclude": list(WEIGHT_DECAY_EXCLUDE_NAMES),
        "monitor": MONITOR_METRIC,
        "mode": MONITOR_MODE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "reduce_lr_factor": REDUCE_LR_FACTOR,
        "min_lr": MIN_LR,
    }


def run_config_hash(experiment_id, arm, run_seed, split_sha256):
    return ckpt.config_hash(run_config_mapping(experiment_id, arm, run_seed, split_sha256))


def write_run_manifest(path, mapping, config_hash_value, repo_dir=None):
    if os.path.exists(path):
        raise RunConfigurationError(
            f"{path} already exists -- run manifests are immutable; a run is never re-initialized."
        )
    payload = dict(mapping)
    payload["config_hash"] = config_hash_value
    payload["environment"] = ckpt.environment_fingerprint(repo_dir)
    payload["created"] = _now()
    _atomic_write_json(path, payload)
    return payload


def read_run_manifest(run_dir_path):
    return _read_json(os.path.join(run_dir_path, RUN_MANIFEST_FILENAME))


def verify_run_manifest(run_dir_path, expected_mapping, expected_hash):
    """Raises `RunConfigurationError` if the persisted run identity does not match what this
    session is configured for -- experiment id, arm, run seed, split hash, protocol version, or
    the full config hash.

    The git commit recorded in the manifest is provenance only and is never compared here: a
    commit that does not change training behaviour must not block a run. Whether persisted
    checkpoints may be resumed is decided by the training-behaviour fingerprint in `train_run()`
    (`reconcile_training_behavior()`)."""
    manifest = read_run_manifest(run_dir_path)
    if manifest is None:
        raise RunConfigurationError(f"No {RUN_MANIFEST_FILENAME} at {run_dir_path} -- not initialized.")
    if manifest.get("config_hash") != expected_hash:
        raise RunConfigurationError(
            f"config_hash mismatch at {run_dir_path}: manifest has {manifest.get('config_hash')}, "
            f"this session computed {expected_hash}. Refusing to resume with a materially "
            "different configuration."
        )
    for key in ("experiment_id", "arm", "run_seed", "split_seed", "split_sha256", "protocol_version"):
        if manifest.get(key) != expected_mapping.get(key):
            raise RunConfigurationError(
                f"{key} mismatch at {run_dir_path}: manifest has {manifest.get(key)!r}, this "
                f"session has {expected_mapping.get(key)!r}."
            )
    return manifest


def initialize_run(experiments_root, experiment_id, arm, run_seed, split_sha256, repo_dir=None):
    """Idempotent: creates the run directory and its `run_manifest.json` if this run has never
    been initialized; otherwise verifies the existing manifest matches. Never overwrites an
    existing manifest. Returns `(run_dir_path, config_hash_value)`."""
    path = run_dir(experiments_root, experiment_id, arm, run_seed)
    ensure_run_dir(path)
    mapping = run_config_mapping(experiment_id, arm, run_seed, split_sha256)
    config_hash_value = ckpt.config_hash(mapping)
    manifest_path = os.path.join(path, RUN_MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        write_run_manifest(manifest_path, mapping, config_hash_value, repo_dir=repo_dir)
    else:
        verify_run_manifest(path, mapping, config_hash_value)
    return path, config_hash_value


# =====================================================================================
# 5b. Training-behaviour fingerprint -- decides whether a run's checkpoints may be resumed.
# =====================================================================================
#
# The git commit is provenance only. What decides whether persisted model + optimizer state may be
# resumed is a hash of the code and configuration that materially determine how the model is
# fitted: the model/loss/metric/callback/augmentation/data-path source listed below, the run's
# training configuration, and the exact train/validation population. Source is normalised before
# hashing -- comments, docstrings, blank lines and trailing whitespace are dropped -- so a comment
# or docstring edit leaves the hash unchanged while any change to executable code in a listed
# module or symbol changes it. No timestamp, path, git SHA or random value enters the hash.

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
TRAINING_BEHAVIOR_FILENAME = "training_behavior.json"
SUPERSEDED_DIRNAME = "superseded"
TRAINING_BEHAVIOR_FINGERPRINT_VERSION = 1
WHOLE_MODULE = None

#: `(repo-relative path, symbols)`. `WHOLE_MODULE` where every definition in the file belongs to
#: the model, loss, metric or callbacks; an explicit symbol tuple where the file also holds
#: infrastructure (cache generation/staging, locking, status, evaluation, checkpoint persistence)
#: whose changes must NOT restart a run.
TRAINING_BEHAVIOR_SOURCES = (
    # Model architecture, both arms.
    ("local_feature_extraction_model.py", WHOLE_MODULE),
    ("swin_transformer.py", WHOLE_MODULE),
    ("feature_fusion.py", WHOLE_MODULE),
    ("racaf.py", WHOLE_MODULE),
    ("corn.py", WHOLE_MODULE),
    ("joint_training_model.py", WHOLE_MODULE),
    ("no_racaf_model.py", ("InertReliabilityConnection", "build_no_racaf_joint_model",
                           "build_no_racaf_joint_model_matched_init")),
    # Loss, metric, optimizer, seeding + matched initialisation + compile.
    ("weighted_corn.py", WHOLE_MODULE),
    ("training/metrics.py", ("QuadraticWeightedKappa",)),
    ("multiseed_runs.py", ("build_optimizer", "build_arm_model")),
    # Early stopping, ReduceLROnPlateau, callback-counter restore; precision policy.
    ("training/callbacks.py", WHOLE_MODULE),
    ("training/trainer.py", ("enable_mixed_precision",)),
    # Training inputs: cached-sample construction, augmentation and its RNG, epoch order.
    ("local_feature_extraction_dataset.py", ("NUM_CHANNELS", "_cache_path", "_augment_spatial",
                                             "_augment_intensity_rgb", "_resize_input")),
    ("joint_training_dataset.py", ("STAGE5_IMAGE_SIZE", "STAGE6_IMAGE_SIZE", "_expected_cache_shape",
                                   "_validate_cached_array", "_load_local_array",
                                   "_canonical_rgb_cache_path", "_get_or_compute_canonical_rgb",
                                   "_get_or_compute_joint_frozen_outputs", "_augment",
                                   "_build_joint_sample")),
    ("joint_cache_diagnostics.py", ("artifact_paths",)),
    ("improved_training_data.py", ("_AUGMENTATION_TAG", "_ORDER_TAG", "_seed_from_key",
                                   "per_image_augmentation_rng", "epoch_training_order",
                                   "missing_local_artifacts", "locally_cached_entries",
                                   "load_cached_sample", "make_epoch_dataset")),
)


def _char_col(line, byte_col):
    """AST column offsets are UTF-8 byte offsets; tokenize's are character offsets."""
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="ignore"))


def _strip_comments_and_docstrings(source):
    """`(tree, lines)`: `source` with every comment and module/class/function docstring blanked
    and trailing whitespace removed. Line numbering is preserved so AST line ranges still index
    `lines`."""
    tree = ast.parse(source)
    lines = source.split("\n")
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                doc = body[0]
                spans.append((doc.lineno, _char_col(lines[doc.lineno - 1], doc.col_offset),
                              doc.end_lineno, _char_col(lines[doc.end_lineno - 1], doc.end_col_offset)))
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            spans.append((token.start[0], token.start[1], token.end[0], token.end[1]))

    cleaned = list(lines)
    for start_line, start_col, end_line, end_col in sorted(spans, reverse=True):
        if start_line == end_line:
            text = cleaned[start_line - 1]
            cleaned[start_line - 1] = text[:start_col] + text[end_col:]
        else:
            cleaned[start_line - 1] = cleaned[start_line - 1][:start_col]
            for index in range(start_line, end_line - 1):
                cleaned[index] = ""
            cleaned[end_line - 1] = cleaned[end_line - 1][end_col:]
    return tree, [line.rstrip() for line in cleaned]


def _top_level_span(tree, symbol, relative_path):
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        else:
            continue
        if symbol in names:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            return start, node.end_lineno
    raise RunConfigurationError(
        f"Training-behaviour source {relative_path}::{symbol} no longer exists -- update "
        "TRAINING_BEHAVIOR_SOURCES deliberately rather than fingerprinting less code silently.")


def normalized_source_digests(relative_path, symbols=WHOLE_MODULE, repo_root=REPO_ROOT):
    """`{component_key: sha256}` of the normalised source of `relative_path` -- one entry for the
    whole module, or one `path::symbol` entry per listed top-level definition."""
    with open(os.path.join(repo_root, *relative_path.split("/")), encoding="utf-8") as handle:
        source = handle.read()
    tree, lines = _strip_comments_and_docstrings(source)

    def digest(selected):
        text = "\n".join(line for line in selected if line.strip())
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    if symbols is WHOLE_MODULE:
        return {relative_path: digest(lines)}
    digests = {}
    for symbol in symbols:
        start, end = _top_level_span(tree, symbol, relative_path)
        digests[f"{relative_path}::{symbol}"] = digest(lines[start - 1:end])
    return digests


def _canonical_hash(payload):
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def population_digest(entries):
    """Order-independent digest of `(id_code, grade)` membership."""
    rows = sorted((str(id_code), int(grade)) for id_code, grade in entries)
    text = "\n".join(f"{id_code},{grade}" for id_code, grade in rows)
    return {"count": len(rows), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def training_behavior_fingerprint(arm, run_seed, train_entries, val_entries, batch_size=BATCH_SIZE,
                                  max_epochs=MAX_EPOCHS,
                                  early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                  reduce_lr_patience=REDUCE_LR_PATIENCE,
                                  reduce_lr_factor=REDUCE_LR_FACTOR, min_lr=MIN_LR,
                                  monitor=MONITOR_METRIC, mode=MONITOR_MODE, mixed_precision=True,
                                  repo_root=REPO_ROOT):
    """`{"training_behavior_hash", "components"}` for one run. `components` is stored in full so
    two fingerprints can be diffed component by component."""
    configuration = run_config_mapping(None, arm, run_seed, EXPECTED_SPLIT_SHA256)
    for label_only in ("experiment_id", "protocol_version"):
        configuration.pop(label_only)
    configuration.update({
        "batch_size": int(batch_size), "max_epochs": int(max_epochs),
        "early_stopping_patience": int(early_stopping_patience),
        "reduce_lr_patience": int(reduce_lr_patience), "reduce_lr_factor": float(reduce_lr_factor),
        "min_lr": float(min_lr), "monitor": monitor, "mode": mode,
        "mixed_precision": bool(mixed_precision),
    })
    sources = {}
    for relative_path, symbols in TRAINING_BEHAVIOR_SOURCES:
        sources.update(normalized_source_digests(relative_path, symbols, repo_root))
    components = {
        "fingerprint_version": TRAINING_BEHAVIOR_FINGERPRINT_VERSION,
        "configuration": configuration,
        "train_population": population_digest(train_entries),
        "validation_population": population_digest(val_entries),
        "sources": sources,
    }
    return {"training_behavior_hash": _canonical_hash(components), "components": components}


def changed_behavior_components(old_components, new_components):
    """Which fingerprint components differ, e.g. `sources.improved_training_data.py::make_epoch_dataset`."""
    if not old_components:
        return ["<no recorded training-behaviour fingerprint>"]
    changed = []
    for section in sorted(set(old_components) | set(new_components)):
        old, new = old_components.get(section), new_components.get(section)
        if section in ("configuration", "sources") and isinstance(old, dict) and isinstance(new, dict):
            changed += [f"{section}.{key}" for key in sorted(set(old) | set(new))
                        if old.get(key) != new.get(key)]
        elif old != new:
            changed.append(section)
    return changed


def _current_git_commit(repo_dir=None):
    return ckpt.environment_fingerprint(repo_dir).get("git_commit_hash")


def read_training_behavior(run_dir_path):
    return _read_json(os.path.join(run_dir_path, TRAINING_BEHAVIOR_FILENAME))


def run_has_training_state(run_dir_path):
    """True once any epoch of this run's trajectory has been persisted: a checkpoint generation,
    `latest.json`, a BEST slot/pointer, or an epoch history file."""
    checkpoint_dir = os.path.join(run_dir_path, "checkpoints")
    if ckpt.checkpoint_evidence(checkpoint_dir):
        return True
    if os.path.isdir(checkpoint_dir) and any(
            name in (BEST_POINTER_FILENAME,) + _BEST_SLOTS for name in os.listdir(checkpoint_dir)):
        return True
    return bool(read_history(run_dir_path))


def _superseded_dirs(run_dir_path):
    root = os.path.join(run_dir_path, SUPERSEDED_DIRNAME)
    if not os.path.isdir(root):
        return []
    return sorted(os.path.join(root, name) for name in os.listdir(root) if name.startswith("restart_"))


def _finalize_pending_supersede(run_dir_path):
    for directory in _superseded_dirs(run_dir_path):
        marker = os.path.join(directory, "restart.json")
        payload = _read_json(marker)
        if payload is not None and not payload.get("completed"):
            payload["completed"] = True
            _atomic_write_json(marker, payload)


#: Moved aside in this order -- `checkpoints/` LAST, so a crash part-way leaves the checkpoints in
#: place and the next session repeats the restart instead of treating the run as never trained.
_TRAJECTORY_DIRS = ("history", "logs", "evaluation", "checkpoints")


def _supersede_trajectory(run_dir_path, restart):
    """Moves this run's current trajectory into `superseded/restart_NNN/` (never deleting it) and
    returns that directory. Reuses a restart directory left incomplete by an interrupted attempt."""
    pending = [d for d in _superseded_dirs(run_dir_path)
               if not (_read_json(os.path.join(d, "restart.json")) or {}).get("completed")]
    if pending:
        target = pending[-1]
    else:
        target = os.path.join(run_dir_path, SUPERSEDED_DIRNAME,
                              f"restart_{len(_superseded_dirs(run_dir_path)) + 1:03d}")
        os.makedirs(target, exist_ok=True)
    _atomic_write_json(os.path.join(target, "restart.json"), dict(restart, completed=False))
    for name in _TRAJECTORY_DIRS:
        source = os.path.join(run_dir_path, name)
        if not os.path.isdir(source) or not os.listdir(source):
            continue
        destination, suffix = os.path.join(target, name), 1
        while os.path.exists(destination):
            destination, suffix = os.path.join(target, f"{name}_{suffix}"), suffix + 1
        os.replace(source, destination)
    return target


def reconcile_training_behavior(run_dir_path, behavior, git_commit, verbose=1):
    """Decides, BEFORE any checkpoint is read, whether this unfinished run may resume.

      - same fingerprint as recorded        -> "resume" (a git commit change is only logged);
      - no training state persisted yet     -> "established" (the current fingerprint becomes active);
      - training state + different/missing fingerprint -> "restarted": the trajectory (checkpoints
        incl. BEST, history, logs, evaluation) is moved to `superseded/restart_NNN/`, the run starts
        again from epoch 0, and the event is recorded in `training_behavior.json` and `restart.json`.

    Completed runs never reach this function (`train_run()` returns first)."""
    path = os.path.join(run_dir_path, TRAINING_BEHAVIOR_FILENAME)
    record = read_training_behavior(run_dir_path)
    current_hash = behavior["training_behavior_hash"]
    active_hash = record.get("active_training_behavior_hash") if record else None
    now = _now()

    if record is not None and active_hash == current_hash:
        _finalize_pending_supersede(run_dir_path)
        if record.get("last_git_commit") != git_commit:
            record["events"].append({
                "event": "git_commit_changed_resume_allowed", "training_behavior_hash": current_hash,
                "old_git_commit": record.get("last_git_commit"), "new_git_commit": git_commit,
                "recorded": now,
            })
            record["last_git_commit"] = git_commit
            _atomic_write_json(path, record)
        return {"action": "resume", "training_behavior_hash": current_hash}

    environment = ckpt.environment_fingerprint(None)
    environment["git_commit_hash"] = git_commit
    events = list(record["events"]) if record else []

    if not run_has_training_state(run_dir_path):
        _finalize_pending_supersede(run_dir_path)
        events.append({"event": "established", "training_behavior_hash": current_hash,
                       "previous_training_behavior_hash": active_hash, "git_commit": git_commit,
                       "environment": environment, "recorded": now})
        decision = {"action": "established", "training_behavior_hash": current_hash}
    else:
        restart = {
            "event": "restarted_from_epoch_0",
            "training_behavior_changed": True,
            "old_training_behavior_hash": active_hash,
            "new_training_behavior_hash": current_hash,
            "old_git_commit": record.get("last_git_commit") if record else None,
            "new_git_commit": git_commit,
            "restart_reason": ("training_behavior_fingerprint_changed" if record
                               else "training_behavior_fingerprint_missing"),
            "changed_components": changed_behavior_components(
                record.get("active_components") if record else None, behavior["components"]),
            "environment": environment,
            "recorded": now,
        }
        superseded = _supersede_trajectory(run_dir_path, restart)
        restart["superseded_dir"] = os.path.relpath(superseded, run_dir_path).replace(os.sep, "/")
        ensure_run_dir(run_dir_path)
        events.append(restart)
        decision = dict(restart, action="restarted", training_behavior_hash=current_hash)
        if verbose:
            print(f"{run_dir_path}: training behaviour changed ({restart['restart_reason']}; "
                  f"{restart['changed_components']}). The previous trajectory was moved to "
                  f"{restart['superseded_dir']} and this run restarts from epoch 0.")

    _atomic_write_json(path, {
        "active_training_behavior_hash": current_hash,
        "active_components": behavior["components"],
        "established_git_commit": git_commit,
        "last_git_commit": git_commit,
        "events": events,
    })
    _finalize_pending_supersede(run_dir_path)
    return decision


# =====================================================================================
# 6. Lightweight single-writer lock
# =====================================================================================

LOCK_FILENAME = "LOCK.json"


def _lock_path(run_dir_path):
    return os.path.join(run_dir_path, LOCK_FILENAME)


def acquire_lock(run_dir_path, owner_id=OWNER_ID, ttl_seconds=DEFAULT_LOCK_TTL_SECONDS, force=False):
    """Refuses if a DIFFERENT, still-fresh owner holds the lock. Safe to call repeatedly by the
    SAME owner (re-running a cell in the same runtime just refreshes the heartbeat)."""
    path = _lock_path(run_dir_path)
    existing = _read_json(path)
    if existing is not None and existing.get("owner") != owner_id and not force:
        age = time.time() - float(existing.get("heartbeat", 0))
        if age < ttl_seconds:
            raise RunLockedError(
                f"{run_dir_path} is locked by {existing.get('owner')!r} (heartbeat {age:.0f}s "
                f"ago, ttl {ttl_seconds}s). If that runtime is genuinely gone, call with "
                "force=True only after confirming no other runtime is actually training this run."
            )
    _atomic_write_json(path, {"owner": owner_id, "heartbeat": time.time(), "heartbeat_iso": _now()})


def heartbeat_lock(run_dir_path, owner_id=OWNER_ID):
    """Refreshes the lock; raises if another owner has since taken it over (this runtime must
    stop training immediately if that happens)."""
    existing = _read_json(_lock_path(run_dir_path))
    if existing is not None and existing.get("owner") not in (owner_id, None):
        raise RunLockedError(
            f"Lock on {run_dir_path} is now held by {existing.get('owner')!r}, not {owner_id!r} "
            "-- another runtime took over this run. Stop training immediately."
        )
    _atomic_write_json(_lock_path(run_dir_path), {"owner": owner_id, "heartbeat": time.time(),
                                                  "heartbeat_iso": _now()})


def release_lock(run_dir_path, owner_id=OWNER_ID):
    path = _lock_path(run_dir_path)
    existing = _read_json(path)
    if existing is not None and existing.get("owner") == owner_id:
        try:
            os.remove(path)
        except OSError:
            pass


# =====================================================================================
# 7. Stop-decision sidecar -- "this run is finished; never train another epoch."
# =====================================================================================

STOP_DECISION_FILENAME = "stop_decision.json"


def _stop_decision_path(run_dir_path):
    return os.path.join(run_dir_path, "checkpoints", STOP_DECISION_FILENAME)


def read_stop_decision(run_dir_path):
    return _read_json(_stop_decision_path(run_dir_path))


def write_stop_decision(run_dir_path, epoch, reason):
    if reason not in ("early_stopping", "epoch_cap"):
        raise ValueError(f"reason must be 'early_stopping' or 'epoch_cap', got {reason!r}")
    payload = {"stop_decided": True, "stop_reason": reason, "epoch": int(epoch), "recorded": _now()}
    _atomic_write_json(_stop_decision_path(run_dir_path), payload)
    return payload


# =====================================================================================
# 8. Epoch history -- one sealed file per completed epoch, never appended.
# =====================================================================================

def write_epoch_history(run_dir_path, state):
    """`state`: the `training.TrainingState` read back from the generation JUST sealed for this
    epoch. Writing is idempotent by filename: retrying the SAME epoch after a crash overwrites
    its own file with (deterministically identical, per this experiment's data design) content,
    never appends a duplicate row."""
    logs = (state.extra or {}).get("epoch_logs") or {}
    payload = {
        "epoch": state.completed_epoch,
        "val_QWK": logs.get("val_QWK"),
        "val_loss": logs.get("val_loss"),
        "val_corn_loss_unweighted": logs.get("val_corn_loss_unweighted"),
        "QWK": logs.get("QWK"),
        "loss": logs.get("loss"),
        "corn_loss_unweighted": logs.get("corn_loss_unweighted"),
        "learning_rate": state.learning_rate,
        "best_epoch": state.best_epoch,
        "best_metric": state.best_metric,
        "early_stopping": state.early_stopping,
        "reduce_lr": state.reduce_lr,
        "generation": state.generation,
        "created": state.created,
    }
    path = os.path.join(run_dir_path, "history", f"epoch_{state.completed_epoch:04d}.json")
    _atomic_write_json(path, payload)
    return path


def read_history(run_dir_path):
    directory = os.path.join(run_dir_path, "history")
    if not os.path.isdir(directory):
        return []
    rows = []
    for name in sorted(os.listdir(directory)):
        if name.startswith("epoch_") and name.endswith(".json"):
            payload = _read_json(os.path.join(directory, name))
            if payload is not None:
                rows.append(payload)
    return rows


# =====================================================================================
# 9. Two-slot BEST -- never delete the active slot before the new one is safely written.
# =====================================================================================

BEST_POINTER_FILENAME = "best.json"
_BEST_SLOTS = ("best_a", "best_b")


def _best_pointer_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, BEST_POINTER_FILENAME)


def _seal_best_slot(slot_dir, manifest_path):
    """Writes a READY marker in the exact format `training.checkpointing._seal()` writes
    (reproduced here rather than imported, since that helper is private) -- so
    `ckpt.validate_generation()` accepts this slot exactly as it accepts a LAST generation."""
    payload = {
        "manifest_sha256": ckpt.sha256_file(manifest_path),
        "checkpoint_format": ckpt.CHECKPOINT_FORMAT,
        "format_version": ckpt.CHECKPOINT_FORMAT_VERSION,
        "sealed": _now(),
    }
    _atomic_write_json(os.path.join(slot_dir, ckpt.READY_FILENAME), payload)


def publish_best(run_dir_path, source_generation_dir, state, repo_dir=None, verbose=0):
    """Publishes `source_generation_dir`'s model weights as this run's new global BEST, into
    whichever of `best_a/`/`best_b/` is currently INACTIVE. The active slot (the previously
    published, still-valid BEST) is never touched until the new one is fully written and
    validated -- only then does `best.json` flip to point at it, so a crash at any point during
    this function leaves the PREVIOUS valid BEST exactly as it was."""
    checkpoint_dir = os.path.join(run_dir_path, "checkpoints")
    pointer = _read_json(_best_pointer_path(checkpoint_dir))
    active = pointer.get("active") if pointer else None
    inactive = "best_b" if active == "best_a" else "best_a"
    inactive_dir = os.path.join(checkpoint_dir, inactive)

    shutil.rmtree(inactive_dir, ignore_errors=True)
    os.makedirs(inactive_dir, exist_ok=True)
    shutil.copy2(os.path.join(source_generation_dir, ckpt.MODEL_WEIGHTS_FILENAME),
                os.path.join(inactive_dir, ckpt.MODEL_WEIGHTS_FILENAME))
    with open(os.path.join(inactive_dir, ckpt.STATE_FILENAME), "w") as handle:
        json.dump(state.to_dict(), handle, indent=2)
    manifest = ckpt.build_manifest(inactive_dir, state, filenames=ckpt.BEST_REQUIRED_FILES,
                                   repo_dir=repo_dir)
    manifest["role"] = "best"
    manifest["source_generation"] = os.path.basename(source_generation_dir.rstrip("/\\"))
    manifest_path = os.path.join(inactive_dir, ckpt.MANIFEST_FILENAME)
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2)

    check = ckpt.validate_generation(inactive_dir, required=ckpt.BEST_REQUIRED_FILES, require_ready=False)
    if not check.ok:
        raise ckpt.CheckpointIntegrityError(f"New BEST slot failed local validation: {check.reason}")
    _seal_best_slot(inactive_dir, manifest_path)
    sealed_check = ckpt.validate_generation(inactive_dir, required=ckpt.BEST_REQUIRED_FILES, require_ready=True)
    if not sealed_check.ok:
        raise ckpt.CheckpointIntegrityError(f"New BEST slot failed sealed validation: {sealed_check.reason}")

    _atomic_write_json(_best_pointer_path(checkpoint_dir), {
        "active": inactive, "epoch": state.best_epoch, "val_QWK": state.best_metric,
        "generation": os.path.basename(source_generation_dir.rstrip("/\\")), "updated": _now(),
    })
    if verbose:
        print(f"New BEST ({state.monitor}={state.best_metric} at epoch {state.best_epoch}): {inactive_dir}")
    return inactive_dir


def read_best(run_dir_path):
    """Returns `(slot_dir, pointer_dict)`, or `(None, None)` if no BEST has ever been published
    (a legitimate state before the first improving epoch). If `best.json` names a slot that
    fails integrity validation, this RAISES rather than returning `(None, None)` -- a corrupted
    BEST must never be silently treated as "no BEST yet"."""
    checkpoint_dir = os.path.join(run_dir_path, "checkpoints")
    pointer = _read_json(_best_pointer_path(checkpoint_dir))
    if pointer is None:
        return None, None
    slot_dir = os.path.join(checkpoint_dir, pointer["active"])
    check = ckpt.validate_generation(slot_dir, required=ckpt.BEST_REQUIRED_FILES)
    if not check.ok:
        raise ckpt.CheckpointIntegrityError(
            f"best.json points at {slot_dir}, but it fails integrity validation: {check.reason}. "
            "This is a corrupted BEST, not an absent one -- it is reported rather than ignored."
        )
    return slot_dir, pointer


# =====================================================================================
# 10. AdamW with weight decay excluded from bias / normalization scale-shift parameters
# =====================================================================================

def build_optimizer(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY):
    """AdamW, decoupled weight decay excluded from every variable whose name contains "bias",
    "gamma" or "beta" -- i.e. every Dense/Conv bias and every BatchNorm/LayerNorm scale-shift
    pair. Applied identically to BOTH arms; RACAF's own `reliability_gate`/`global_projection`
    kernels receive the SAME decay as every other kernel, with no RACAF-specific exception (this
    is a deliberate, pre-registered decision -- see `PREREGISTRATION.json`'s
    `weight_decay_applies_to_racaf_gate`, and `tests/test_multiseed_runs.py` for a check that the
    exclusion pattern matches real variable names in the real model, not merely in principle)."""
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    optimizer.exclude_from_weight_decay(var_names=list(WEIGHT_DECAY_EXCLUDE_NAMES))
    return optimizer


# =====================================================================================
# 11. Arm model construction -- shared seed, matched initialisation, weighted loss.
# =====================================================================================

def _verify_racaf_model(model):
    parameters = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    tensors = len(model.trainable_variables)
    if tensors != no_racaf_model.REFERENCE_TRAINABLE_TENSORS:
        raise RuntimeError(f"RACAF model has {tensors} trainable tensors, expected "
                           f"{no_racaf_model.REFERENCE_TRAINABLE_TENSORS}.")
    if parameters != no_racaf_model.REFERENCE_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"RACAF model has {parameters:,} trainable parameters, expected "
                           f"{no_racaf_model.REFERENCE_TRAINABLE_PARAMETERS:,}.")
    return {"trainable_parameters": parameters, "trainable_tensors": tensors}


def build_arm_model(arm, run_seed, class_weights=weighted_corn.PREREGISTERED_CLASS_WEIGHTS,
                    learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
                    mixed_precision=True, verbose=1):
    """Builds and compiles one arm's model for one seed: `keras.utils.set_random_seed(run_seed)`
    immediately before construction, RACAF's own components constructed-and-discarded inside the
    NO_RACAF builder for cross-arm initialisation parity (`no_racaf_model.
    build_no_racaf_joint_model_matched_init`), AdamW + the pre-registered class-weighted CORN
    loss + both the QWK and the unweighted-CORN-loss metrics. Verifies parameter/tensor counts
    and (NO_RACAF only) the reliability-inertness proof ONCE here -- not repeated per epoch."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}, got {arm!r}")

    tf.keras.utils.set_random_seed(run_seed)
    policy = enable_mixed_precision(mixed_precision)
    if arm == "RACAF":
        model = jtm.build_joint_model()
    else:
        model = no_racaf_model.build_no_racaf_joint_model_matched_init()

    optimizer = build_optimizer(learning_rate=learning_rate, weight_decay=weight_decay)
    loss = weighted_corn.make_weighted_corn_loss(class_weights)
    model.compile(optimizer=optimizer, loss=loss,
                 metrics=[corn.CORNQuadraticWeightedKappa(), weighted_corn.UnweightedCORNLoss()])

    expected = expected_policy_name(mixed_precision)
    actual = model_precision_policies(model)
    if not precision_is_consistent(expected, actual):
        raise RuntimeError(f"{arm} model was built under {sorted(actual)} but the global policy "
                           f"is '{policy.name}' and this call requested '{expected}'.")
    resolved = model.optimizer
    inner = getattr(resolved, "inner_optimizer", None)
    if expected == "mixed_float16" and inner is None:
        raise RuntimeError(f"{arm} model is mixed_float16 but its optimizer was not wrapped in a "
                           "LossScaleOptimizer.")

    if arm == "RACAF":
        counts = _verify_racaf_model(model)
    else:
        counts = no_racaf_model.verify_no_racaf_model(model)
    if verbose:
        print(f"{arm} seed={run_seed}: dtype policy '{expected}', optimizer "
              f"{type(resolved).__name__}(inner={type(inner).__name__ if inner else None}), "
              f"{counts['trainable_tensors']} trainable tensors, "
              f"{counts['trainable_parameters']:,} trainable parameters.")
    return model


# =====================================================================================
# 12. Training loop -- one epoch per model.fit() call; resume ALWAYS on. See module docstring.
# =====================================================================================

@dataclass
class TrainRunOutcome:
    completed_epoch: int
    stopped: bool
    stop_reason: Optional[str]
    epochs_trained_this_call: int
    #: `reconcile_training_behavior()`'s decision ("resume"/"established"/"restarted"), or None
    #: when the run was already finished before that decision was needed.
    training_behavior: Optional[Dict[str, Any]] = None


def _sealed_stop(checkpoint_dir, max_epochs):
    """`(epoch, reason)` if the last sealed generation already satisfies a stop condition, else
    None -- read from its state.json only, without restoring anything. Closes the crash window
    between sealing that generation and writing stop_decision.json."""
    generation_dir = ckpt.find_resumable_generation(checkpoint_dir, verbose=False)
    if generation_dir is None:
        return None
    state = ckpt.read_state(generation_dir)
    early_stopping = state.early_stopping or {}
    wait, patience = early_stopping.get("wait"), early_stopping.get("patience")
    if wait is not None and patience is not None and wait >= patience:
        return state.completed_epoch, "early_stopping"
    if state.completed_epoch >= max_epochs:
        return state.completed_epoch, "epoch_cap"
    return None


def train_run(model, run_dir_path, arm, run_seed, train_entries, val_entries,
             cache_dir, racaf_cache_dir, config_hash_value, batch_size=BATCH_SIZE,
             max_epochs=MAX_EPOCHS, early_stopping_patience=EARLY_STOPPING_PATIENCE,
             reduce_lr_patience=REDUCE_LR_PATIENCE, reduce_lr_factor=REDUCE_LR_FACTOR,
             min_lr=MIN_LR, monitor=MONITOR_METRIC, mode=MONITOR_MODE,
             staging_dir="/content/checkpoint_staging", repo_dir=None,
             precision_check="error", mixed_precision=True,
             session_epoch_budget=None, owner_id=OWNER_ID, verbose=1):
    """Trains `model` for at most `max_epochs` total epochs (across every session that has ever
    called this function for this run), stopping early on: a pre-existing sealed stop decision,
    EarlyStopping firing this call, the epoch cap being reached, or `session_epoch_budget` new
    epochs having been trained THIS call (if given -- lets one Colab session deliberately bound
    how much of its time budget goes to one run before moving to the next).

    `model` must already be built by `build_arm_model()` for this exact `(arm, run_seed)` --
    this function does not build or seed the model itself, so a caller resuming a later session
    must rebuild it identically first (`build_arm_model` is itself deterministic given the same
    seed, so this is safe; the OPTIMIZER's actual values are then overwritten by the restored
    checkpoint, only its structure/type need match).

    `train_entries`/`val_entries` must already be cached locally under `cache_dir`/
    `racaf_cache_dir` (`improved_training_data.locally_cached_entries()`, after the notebook's
    one-time `complete_local_cache()`): every epoch reads that local cache only, so neither a
    fresh run nor a resumed one ever runs Stage 02-04 or reads a raw image."""
    ensure_run_dir(run_dir_path)   # idempotent -- also protects a direct caller (e.g. a test) that
                                   # skips initialize_run() from Trainer.prepare()'s
                                   # assert_resume_location(), which requires checkpoints/ to exist.
    acquire_lock(run_dir_path, owner_id=owner_id)
    try:
        existing_stop = read_stop_decision(run_dir_path)
        if existing_stop is not None:
            if verbose:
                print(f"{run_dir_path}: stop_decision.json already records "
                      f"{existing_stop['stop_reason']!r} at epoch {existing_stop['epoch']} -- "
                      "finished; no further epoch will be trained.")
            return TrainRunOutcome(existing_stop["epoch"], True, existing_stop["stop_reason"], 0)

        # A run whose last sealed generation already reached its stop condition is finished --
        # decided before any fingerprint comparison or restore, so a completed run is never
        # restarted and never has weights loaded here.
        sealed_stop = _sealed_stop(os.path.join(run_dir_path, "checkpoints"), max_epochs)
        if sealed_stop is not None:
            write_stop_decision(run_dir_path, sealed_stop[0], sealed_stop[1])
            return TrainRunOutcome(sealed_stop[0], True, sealed_stop[1], 0)

        # Resume only under the SAME training behaviour. Decided before the Trainer is built and
        # before any weight/optimizer restore: a changed fingerprint moves this unfinished run's
        # trajectory aside, so resolve_initial_epoch() below finds nothing and training starts at
        # epoch 0 from the model exactly as build_arm_model() produced it.
        behavior = training_behavior_fingerprint(
            arm, run_seed, train_entries, val_entries, batch_size=batch_size, max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience, reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor, min_lr=min_lr, monitor=monitor, mode=mode,
            mixed_precision=mixed_precision)
        behavior_decision = reconcile_training_behavior(
            run_dir_path, behavior, _current_git_commit(repo_dir), verbose=verbose)

        config = TrainingConfig(
            run_dir=run_dir_path, epochs=max_epochs, monitor=monitor, mode=mode,
            mixed_precision=mixed_precision,
            resume=True,   # ALWAYS -- see this module's docstring for why this is correct even
                           # for a run's very first epoch.
            early_stopping_patience=early_stopping_patience, reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor, min_lr=min_lr, precision_check=precision_check,
            repo_dir=repo_dir,
            checkpoint_options=CheckpointOptions(
                experiment_id=f"{arm}/seed_{run_seed}", config_hash=config_hash_value,
                dataset_version="aptos2019-joint-cache-v1", staging_dir=staging_dir,
                keep_generations=2, verbose=verbose,
            ),
        )
        trainer = Trainer(config)
        trainer.prepare(model)
        initial_epoch = trainer.resolve_initial_epoch()
        if initial_epoch > 0:
            trainer.restore(model)

        if initial_epoch >= max_epochs:
            write_stop_decision(run_dir_path, initial_epoch, "epoch_cap")
            return TrainRunOutcome(initial_epoch, True, "epoch_cap", 0, behavior_decision)

        early_stopping_cb = next(c for c in trainer.callbacks
                                 if isinstance(c, tf.keras.callbacks.EarlyStopping))
        state_checkpoint_cb = next(c for c in trainer.callbacks
                                   if isinstance(c, TrainingStateCheckpoint))

        val_ds = itd.make_epoch_dataset(
            val_entries, epoch=0, run_seed=run_seed, cache_dir=cache_dir,
            racaf_cache_dir=racaf_cache_dir, batch_size=batch_size, augment=False,
        )

        completed_epoch = initial_epoch
        epochs_trained_this_call = 0
        stopped, stop_reason = False, None

        for epoch in range(initial_epoch, max_epochs):
            heartbeat_lock(run_dir_path, owner_id=owner_id)
            train_ds = itd.make_epoch_dataset(
                train_entries, epoch=epoch, run_seed=run_seed, cache_dir=cache_dir,
                racaf_cache_dir=racaf_cache_dir, batch_size=batch_size, augment=True,
            )
            model.fit(train_ds, validation_data=val_ds, epochs=epoch + 1, initial_epoch=epoch,
                     callbacks=trainer.callbacks, verbose=verbose)

            generation_dir = state_checkpoint_cb.last_generation_dir
            if generation_dir is None:
                raise RuntimeError(f"Epoch {epoch}: no checkpoint generation was written for "
                                   f"{run_dir_path} -- refusing to continue silently.")
            completed_state = ckpt.read_state(generation_dir)
            completed_epoch = completed_state.completed_epoch
            write_epoch_history(run_dir_path, completed_state)
            epochs_trained_this_call += 1

            # `TrainingStateCheckpoint.on_epoch_end` (unmodified) already published its OWN
            # single-slot `checkpoints/best/` via `ckpt.save_best()` when this epoch improved --
            # that call is reused as-is (never touched) rather than suppressed, per this module's
            # "reuse unmodified infrastructure" design. It is harmless but NOT what this
            # experiment's evaluation/comparison code reads from: publish THIS run's authoritative,
            # crash-safe two-slot BEST here explicitly. `completed_state.best_epoch == epoch`
            # (both 0-based, Keras' own convention) is true iff THIS epoch just became the new
            # global best -- not merely "a best exists somewhere from an earlier epoch".
            if completed_state.best_epoch == epoch:
                publish_best(run_dir_path, generation_dir, completed_state, repo_dir=repo_dir,
                            verbose=verbose)

            if early_stopping_cb.stopped_epoch:
                write_stop_decision(run_dir_path, completed_epoch, "early_stopping")
                stopped, stop_reason = True, "early_stopping"
                break
            if completed_epoch >= max_epochs:
                write_stop_decision(run_dir_path, completed_epoch, "epoch_cap")
                stopped, stop_reason = True, "epoch_cap"
                break
            if session_epoch_budget is not None and epochs_trained_this_call >= session_epoch_budget:
                break

        return TrainRunOutcome(completed_epoch, stopped, stop_reason, epochs_trained_this_call,
                               behavior_decision)
    finally:
        release_lock(run_dir_path, owner_id=owner_id)


# =====================================================================================
# 13. Status
# =====================================================================================

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_CREATED = "CREATED"
STATUS_RUNNING = "RUNNING"
STATUS_INTERRUPTED = "INTERRUPTED"
STATUS_COMPLETED = "COMPLETED"
STATUS_FAILED = "FAILED"


def run_status(run_dir_path, lock_ttl_seconds=DEFAULT_LOCK_TTL_SECONDS):
    if read_run_manifest(run_dir_path) is None:
        return STATUS_NOT_STARTED
    if read_stop_decision(run_dir_path) is not None:
        return STATUS_COMPLETED
    checkpoint_dir = os.path.join(run_dir_path, "checkpoints")
    generation_dir = ckpt.find_resumable_generation(checkpoint_dir, verbose=False)
    if generation_dir is None:
        return STATUS_FAILED if ckpt.checkpoint_evidence(checkpoint_dir) else STATUS_CREATED
    lock = _read_json(_lock_path(run_dir_path))
    if lock is not None and (time.time() - float(lock.get("heartbeat", 0))) < lock_ttl_seconds:
        return STATUS_RUNNING
    return STATUS_INTERRUPTED


def experiment_status_table(experiments_root, experiment_id):
    return {
        f"{arm}/seed_{seed}": run_status(run_dir(experiments_root, experiment_id, arm, seed))
        for arm, seed in all_run_ids()
    }


# =====================================================================================
# 14. Evaluation from disk -- per-sample rows with the same schema as
#     evaluation/per_sample_corn_predictions.csv (the finalized RACAF/NO-RACAF experiments'
#     own schema), for direct compatibility with the existing duplicate-audit tooling.
# =====================================================================================

def evaluate_arm_from_disk(model, entries, cache_dir, racaf_cache_dir, batch_size=8):
    """Runs `model` (already loaded from disk by the caller) over `entries` deterministically
    (no augmentation, manifest order), building each sample from the LOCAL cache via
    `improved_training_data.load_cached_sample()` -- one at a time, batched only for
    `model.predict_on_batch()` -- so each row can be tied back to its `image_id`, which a
    `tf.data` output signature does not carry. `entries` must already be cached locally."""
    ids, grades, s5, s6, rel = [], [], [], [], []
    all_logits, all_true, all_ids = [], [], []

    def flush():
        if not ids:
            return
        # `reliability` is fed as (N, 1), matching the model's own `Input(shape=(1,))`. It must NOT
        # be the rank-1 (N,) that `np.stack` of per-sample scalars produces: `build_arm_model()`
        # already traces this model's predict function with rank-2 (2, 1) probes for the NO_RACAF
        # arm (`no_racaf_model.verify_no_racaf_model()`), and calling it afterwards with a
        # different RANK makes TensorFlow relax the traced signature to an unknown TensorShape --
        # after which the first static-shape-dependent op (e.g. Swin's window reshapes) fails with
        # "as_list() is not defined on an unknown TensorShape". Keras adjusts either rank to the
        # declared input spec, so this changes no predicted value (verified bit-identical).
        logits = model.predict_on_batch(
            [np.stack(s5), np.stack(s6), np.stack(rel).reshape(-1, 1)]
        )
        all_logits.append(np.asarray(logits, dtype=np.float64))
        all_true.extend(grades)
        all_ids.extend(ids)
        ids.clear(); grades.clear(); s5.clear(); s6.clear(); rel.clear()

    for id_code, diagnosis in entries:
        sample = itd.load_cached_sample(id_code, diagnosis, cache_dir, racaf_cache_dir, False, None)
        ids.append(id_code)
        grades.append(int(diagnosis))
        s5.append(sample["stage5_input"])
        s6.append(sample["stage6_input"])
        rel.append(np.float32(sample["reliability"]))
        if len(ids) >= batch_size:
            flush()
    flush()

    logits = (np.concatenate(all_logits, axis=0) if all_logits
             else np.zeros((0, corn.NUM_THRESHOLDS), dtype=np.float64))
    true_grades = np.asarray(all_true, dtype=int)
    decoded = corn.decode_logits(logits)
    return build_per_sample_rows(all_ids, true_grades, logits, decoded)


def build_per_sample_rows(ids, true_grades, logits, decoded):
    """Reproduces exactly the 27-column schema the finalized RACAF/NO-RACAF experiments' own
    `[D5]` post-run cell writes to `evaluation/per_sample_corn_predictions.csv` -- so this
    experiment's per-sample tables are readable by the same duplicate-contamination audit
    tooling without modification."""
    if len(ids) == 0:
        return []
    p_cum = decoded["p_cum"].astype(np.float64)
    p_cond = decoded["p_cond"].astype(np.float64)
    class_probabilities = decoded["class_probabilities"].astype(np.float64)
    predicted = decoded["predicted_grade"]
    row_index = np.arange(len(true_grades))
    predicted_class_probability = class_probabilities[row_index, predicted]
    argmax_class = class_probabilities.argmax(axis=-1).astype(int)
    threshold_margin = np.min(np.abs(p_cum - 0.5), axis=-1)
    safe = np.clip(class_probabilities, 1e-12, 1.0)
    class_entropy = -np.sum(class_probabilities * np.log(safe), axis=-1)
    error_distance = np.abs(true_grades - predicted)

    rows = []
    for index, image_id in enumerate(ids):
        row = {
            "image_id": image_id, "true_grade": int(true_grades[index]),
            "predicted_grade": int(predicted[index]),
            "correct": bool(true_grades[index] == predicted[index]),
            "error_distance": int(error_distance[index]),
        }
        row.update({f"logit_{k}": float(logits[index, k]) for k in range(corn.NUM_THRESHOLDS)})
        row.update({f"p_gt_{k}": float(p_cum[index, k]) for k in range(corn.NUM_THRESHOLDS)})
        row.update({f"p_cond_{k}": float(p_cond[index, k]) for k in range(corn.NUM_THRESHOLDS)})
        row.update({f"class_prob_{g}": float(class_probabilities[index, g])
                    for g in range(corn.NUM_GRADES)})
        row.update({
            "predicted_class_probability": float(predicted_class_probability[index]),
            "argmax_class": int(argmax_class[index]),
            "argmax_matches_decode": bool(argmax_class[index] == predicted[index]),
            "nearest_threshold_margin": float(threshold_margin[index]),
            "class_entropy_nats": float(class_entropy[index]),
        })
        rows.append(row)
    return rows


# =====================================================================================
# 15. Metrics summary from per-sample rows (BEST/LAST evaluation, and the six-run comparison)
# =====================================================================================

GRADES = (0, 1, 2, 3, 4)


def summarize_predictions(rows):
    """Standard classification/ordinal metrics from `build_per_sample_rows()`'s output --
    accuracy, balanced accuracy, macro/weighted F1, MAE, QWK, confusion matrix, per-grade
    precision/recall/F1, prediction histogram, error-distance distribution, and calibration
    (ECE/Brier -- reported, never used for model selection; the class-weighted training
    objective shifts the predicted distribution away from the empirical prior by design, so a
    calibration change here is an expected consequence of Change C, not necessarily a defect)."""
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support,
        confusion_matrix, mean_absolute_error, cohen_kappa_score,
    )
    if not rows:
        return {"n": 0}
    y_true = np.array([r["true_grade"] for r in rows])
    y_pred = np.array([r["predicted_grade"] for r in rows])
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=GRADES, zero_division=0)
    error_distance = np.abs(y_true - y_pred)

    predicted_probability = np.array([r["predicted_class_probability"] for r in rows])
    class_probabilities = np.array([[r[f"class_prob_{g}"] for g in GRADES] for r in rows])
    one_hot = np.eye(len(GRADES))[y_true]
    brier = float(np.mean(np.sum((class_probabilities - one_hot) ** 2, axis=-1)))
    correct = (y_true == y_pred).astype(np.float64)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (predicted_probability > lo) & (predicted_probability <= hi)
        if mask.any():
            ece += (mask.mean()) * abs(correct[mask].mean() - predicted_probability[mask].mean())

    return {
        "n": len(rows),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=GRADES, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=GRADES, average="weighted", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=GRADES).tolist(),
        "precision": {g: float(p) for g, p in zip(GRADES, prec)},
        "recall": {g: float(r) for g, r in zip(GRADES, rec)},
        "f1": {g: float(f) for g, f in zip(GRADES, f1)},
        "support": {g: int(s) for g, s in zip(GRADES, support)},
        "prediction_histogram": {g: int((y_pred == g).sum()) for g in GRADES},
        "error_distance_histogram": {int(d): int((error_distance == d).sum()) for d in range(5)},
        "errors_ge_2_grades": int((error_distance >= 2).sum()),
        "grade_3_recall": float(rec[3]),
        "grade_4_recall": float(rec[4]),
        "brier": brier,
        "ece": float(ece),
    }
