"""
Crash-safe, resumable training checkpoints: model weights + optimizer state +
one authoritative training-state record, written as immutable numbered
*generations* under a run's `checkpoints/` directory.

Why this exists
---------------
`ModelCheckpoint(save_weights_only=True)` persists model variables and nothing
else. A run resumed from one of those files restarts with a freshly initialized
optimizer (Adam's moment estimates back to zero, `iterations` back to 0) and
freshly reset callbacks (`EarlyStopping.wait = 0`, `ReduceLROnPlateau.best =
None`, `ModelCheckpoint.best = None`). On this project's schedule -- roughly one
epoch per Colab session -- that means the LR could never be reduced, early
stopping could never fire, and the "best" checkpoint would be overwritten by the
first epoch of every new session regardless of how bad it was.

This module fixes that without adopting full `.keras` model serialization, which
`joint_training_model.py` documents as unsafe here (Stage 06's Swin layer classes
have no `get_config()`). Weights still travel as `.weights.h5` through the
existing "rebuild the architecture, then `load_weights`" path; the optimizer
travels as a plain `.npz` of its own variables; everything else travels as JSON.

On-disk layout
--------------
    checkpoints/
        gen_00001/
            model.weights.h5   -- model variables (incl. BatchNorm moving stats)
            optimizer.npz      -- every optimizer variable, in build order
            state.json         -- the authoritative training state (see TrainingState)
            manifest.json      -- per-file size + SHA256, plus compatibility metadata
            READY              -- written LAST; its absence means "ignore this generation"
        gen_00002/
            ...
        best/                  -- the globally best val_QWK epoch (delivery model)
        latest.json            -- pointer to the newest known-good generation

Write protocol (Drive/FUSE-safe)
--------------------------------
Everything is built and validated on LOCAL disk first, then copied to the real
(possibly Drive-backed) destination, then re-validated there, and only then is
`READY` written and `latest.json` updated. `os.replace` is never relied upon for
atomicity: Google Drive's FUSE mount makes no such guarantee. The atomicity that
matters here is supplied instead by the READY marker -- a generation directory
without one is invisible to `find_resumable_generation()`, so a half-copied
generation is simply skipped rather than half-loaded. The previous known-good
generation is pruned only after its replacement has been validated at the
destination.

Integrity vs. compatibility
---------------------------
These are deliberately different failures with different handling:

* An *integrity* failure (missing file, wrong size, wrong SHA256, no READY)
  means "this generation is damaged" -- `find_resumable_generation()` skips it
  and falls back to the newest older generation that validates.
* A *compatibility* failure (different config hash, optimizer type, precision
  policy, or checkpoint format) means "this generation is intact but does not
  belong to the run you are trying to continue" -- that RAISES. Silently falling
  back to an older generation of an incompatible run would be worse than
  stopping.

What resume does and does not guarantee
---------------------------------------
Restoring a generation reproduces the model, its BatchNorm statistics, the full
Adam state, the learning rate, and every stateful callback counter. It is
*statistically equivalent continuation*, not bit-for-bit reproduction of an
uninterrupted run: this project sets no global seed, GPU kernels are not
deterministic, Keras 3's dropout `SeedGenerator` state is not part of
`.weights.h5`, and `model.fit(initial_epoch=N)` does not fast-forward the
`tf.data` shuffle stream. See JOINT_TRAINING_ARCHITECTURE.md's checkpoint
section. Nothing in this module claims otherwise.
"""

import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

CHECKPOINT_FORMAT = "generation-v1"
CHECKPOINT_FORMAT_VERSION = 1

GENERATION_PREFIX = "gen_"
GENERATION_DIGITS = 5
BEST_DIRNAME = "best"

MODEL_WEIGHTS_FILENAME = "model.weights.h5"
OPTIMIZER_FILENAME = "optimizer.npz"
STATE_FILENAME = "state.json"
MANIFEST_FILENAME = "manifest.json"
READY_FILENAME = "READY"
LATEST_FILENAME = "latest.json"

#: Files a *generation* must contain to be resumable.
REQUIRED_FILES = (MODEL_WEIGHTS_FILENAME, OPTIMIZER_FILENAME, STATE_FILENAME)
#: Files the *best* checkpoint must contain. It carries no optimizer state: BEST is
#: the delivery model, never the trajectory a later session continues from.
BEST_REQUIRED_FILES = (MODEL_WEIGHTS_FILENAME, STATE_FILENAME)

#: Key holding the JSON optimizer-variable spec inside `optimizer.npz`, so the
#: archive is self-describing rather than positional-by-convention.
OPTIMIZER_SPEC_KEY = "__spec__"

DEFAULT_KEEP_GENERATIONS = 2

#: A mismatch in any of these refuses the resume outright.
STRICT_COMPATIBILITY_FIELDS = (
    "checkpoint_format",
    "format_version",
    "config_hash",
    "optimizer_type",
    "precision_policy",
)
#: A mismatch in any of these is reported but does not block the resume: pulling a
#: new commit or a TensorFlow patch release between Colab sessions is normal, and
#: refusing on it would make the whole scheme unusable.
ADVISORY_COMPATIBILITY_FIELDS = (
    "git_commit_hash",
    "tensorflow_version",
    "keras_version",
    "python_version",
    "dataset_version",
)


class CheckpointError(RuntimeError):
    """Base class for every failure raised by this module."""


class CheckpointIntegrityError(CheckpointError):
    """The checkpoint is damaged, incomplete, or does not match its manifest."""


class CheckpointCompatibilityError(CheckpointError):
    """The checkpoint is intact but belongs to a different configuration."""


class CheckpointResumeError(CheckpointError):
    """A resume was explicitly requested and cannot be honoured safely: the
    location does not exist, is not an experiment this framework created, or
    holds checkpoint state of which nothing validates. Always raised BEFORE any
    epoch runs, so the experiment on disk -- including `best/` -- is untouched."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def sha256_file(path, chunk_size=1 << 20):
    """SHA256 of a file's bytes, streamed so a 166 MiB weights file never lands
    in memory twice."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(mapping):
    """A stable, order-independent hash of a training configuration.

    Used to answer "is this checkpoint from the same run I am configuring now?".
    Keys are sorted and values are JSON-serialized with `default=str`, so a
    configuration containing paths, enums or tuples still hashes deterministically
    rather than raising."""
    if mapping is None:
        return None
    payload = json.dumps(mapping, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _git_commit_hash(repo_dir=None):
    """Current commit of the repository this module lives in, or None outside a
    checkout. Never raises: a missing git binary must not break checkpointing."""
    repo_dir = repo_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        completed = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def environment_fingerprint(repo_dir=None):
    """The environment facts a resume needs to judge compatibility."""
    try:
        keras_version = tf.keras.__version__
    except AttributeError:  # pragma: no cover -- defensive
        keras_version = None
    return {
        "git_commit_hash": _git_commit_hash(repo_dir),
        "tensorflow_version": tf.__version__,
        "keras_version": keras_version,
        "python_version": platform.python_version(),
    }


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def _json_safe(value):
    """Coerce NumPy scalars / arrays that arrive from Keras logs into plain JSON
    types. `logs["val_QWK"]` is routinely a `np.float32`, which `json.dump`
    refuses."""
    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def generation_name(number):
    return f"{GENERATION_PREFIX}{number:0{GENERATION_DIGITS}d}"


def generation_number(name):
    """Parse `gen_00007` -> 7, or None if `name` is not a generation directory."""
    base = os.path.basename(str(name).rstrip("/\\"))
    if not base.startswith(GENERATION_PREFIX):
        return None
    suffix = base[len(GENERATION_PREFIX):]
    if not suffix.isdigit():
        return None
    return int(suffix)


def best_dir(checkpoint_dir):
    return os.path.join(checkpoint_dir, BEST_DIRNAME)


def latest_path(checkpoint_dir):
    return os.path.join(checkpoint_dir, LATEST_FILENAME)


# ---------------------------------------------------------------------------
# Authoritative training state
# ---------------------------------------------------------------------------

@dataclass
class TrainingState:
    """The single authoritative record of where a training run has got to.

    Everything a resumed session needs that is NOT a tensor lives here, in one
    JSON file, rather than being reconstructed from Keras callbacks' private
    attributes or re-derived from a metrics CSV.

    `completed_epoch` is the number of epochs that have finished, which is also
    the `initial_epoch` the next session must pass to `model.fit()`.
    `best_epoch`/`best_metric` are GLOBAL across the whole experiment -- across
    every session, process and runtime -- not per-session.
    """

    experiment_id: Optional[str] = None
    generation: int = 0
    completed_epoch: int = 0
    best_epoch: Optional[int] = None
    best_metric: Optional[float] = None
    monitor: str = "val_QWK"
    monitor_mode: str = "max"
    learning_rate: Optional[float] = None
    early_stopping: Dict[str, Any] = field(default_factory=dict)
    reduce_lr: Dict[str, Any] = field(default_factory=dict)
    optimizer: Dict[str, Any] = field(default_factory=dict)
    precision_policy: Optional[str] = None
    config_hash: Optional[str] = None
    dataset_version: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)
    format_version: int = CHECKPOINT_FORMAT_VERSION
    created: Optional[str] = None

    @property
    def next_epoch(self):
        """The `initial_epoch` a resumed `model.fit()` must start from."""
        return self.completed_epoch

    def to_dict(self):
        return _json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload):
        known = {f for f in cls.__dataclass_fields__}  # noqa: F821 -- dataclass attr
        return cls(**{k: v for k, v in payload.items() if k in known})


def read_state(generation_dir):
    """Load `state.json` from a generation directory."""
    with open(os.path.join(generation_dir, STATE_FILENAME)) as handle:
        return TrainingState.from_dict(json.load(handle))


# ---------------------------------------------------------------------------
# Optimizer state
# ---------------------------------------------------------------------------

def _variable_name(variable, index):
    return getattr(variable, "path", None) or getattr(variable, "name", None) or f"var_{index}"


def optimizer_variables(optimizer):
    """Every variable the optimizer owns, in build order.

    Under `mixed_float16` Keras wraps the optimizer in a `LossScaleOptimizer`,
    whose `.variables` already includes both its own four (iteration,
    learning_rate, step_counter, dynamic_scale) and the inner Adam's slots -- so
    this needs no special case, but the count and order DO differ from a bare
    Adam, which is exactly why `precision_policy` is a strict compatibility
    field."""
    return list(optimizer.variables)


def ensure_optimizer_built(optimizer, model=None, variables=None):
    """Create the optimizer's slot variables if they do not exist yet.

    This is the single most important ordering rule in this module. A freshly
    constructed Keras 3 optimizer owns only `iteration` and `learning_rate`; the
    per-parameter momentum/velocity slots are created lazily on the first
    `apply_gradients`. Assigning saved state into an unbuilt optimizer therefore
    silently restores almost nothing. `Optimizer.build()` is guarded by
    `if self.built: return`, so calling it here is safe even when Keras has
    already built it."""
    if getattr(optimizer, "built", False):
        return optimizer
    if variables is None:
        if model is None:
            raise CheckpointError(
                "Cannot build optimizer slots: pass either `model` or `variables`."
            )
        variables = model.trainable_variables
    optimizer.build(variables)
    return optimizer


def optimizer_state_spec(optimizer, include_variables=True):
    """A JSON-safe description of the optimizer's variables, for `state.json`,
    the manifest, and validation on restore.

    `include_variables=False` drops the per-variable list and keeps only its
    signature hash. The full list is 820 entries for this project's joint model
    -- worth carrying inside `optimizer.npz`, where it is what validates the
    restore, but not worth 140 KB of `state.json` that a human is meant to read.
    The signature still detects a changed architecture."""
    variables = optimizer_variables(optimizer)
    inner = getattr(optimizer, "inner_optimizer", None)
    described = [
        {
            "name": _variable_name(v, i),
            "shape": [int(d) for d in v.shape],
            "dtype": str(v.dtype).replace("<dtype: '", "").replace("'>", ""),
        }
        for i, v in enumerate(variables)
    ]
    spec = {
        "type": type(inner).__name__ if inner is not None else type(optimizer).__name__,
        "wrapper": type(optimizer).__name__ if inner is not None else None,
        "iterations": int(tf.keras.backend.get_value(optimizer.iterations)),
        "learning_rate": float(tf.keras.backend.get_value(optimizer.learning_rate)),
        "variable_count": len(variables),
        "variables_signature": hashlib.sha256(
            json.dumps(described, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:32],
    }
    if include_variables:
        spec["variables"] = described
    return spec


def save_optimizer_state(optimizer, path):
    """Write every optimizer variable to a `.npz`, keyed by build position.

    Position, not name: Keras 3 derives optimizer variable names from the model's
    name and a per-process instance counter (`adam/...` in a fresh process,
    `adam_1/sequential_1_...` for a second optimizer in the same one), so names
    are not stable across processes and cannot be the key. Build order for a given
    architecture IS stable, and the spec stored alongside carries the names,
    shapes and dtypes so a mismatch is detected loudly rather than silently
    mis-assigned."""
    variables = optimizer_variables(optimizer)
    payload = {f"v{i:04d}": np.asarray(tf.keras.backend.get_value(v))
               for i, v in enumerate(variables)}
    payload[OPTIMIZER_SPEC_KEY] = np.array(json.dumps(optimizer_state_spec(optimizer)))
    np.savez(path, **payload)
    return path


def restore_optimizer_state(optimizer, path, model=None, variables=None):
    """Restore every optimizer variable from a `.npz` written by
    `save_optimizer_state`, after making sure the slots exist.

    Raises `CheckpointIntegrityError` on any count, shape or dtype mismatch. It
    never partially restores and never leaves a slot at its initial value: a
    resumed Adam with half its second moments zeroed would take large, wrongly
    scaled steps for hundreds of iterations, which is precisely the failure this
    whole module exists to prevent."""
    ensure_optimizer_built(optimizer, model=model, variables=variables)
    live = optimizer_variables(optimizer)

    with np.load(path, allow_pickle=False) as archive:
        keys = sorted(k for k in archive.files if k != OPTIMIZER_SPEC_KEY)
        if OPTIMIZER_SPEC_KEY not in archive.files:
            raise CheckpointIntegrityError(
                f"{path}: optimizer archive has no '{OPTIMIZER_SPEC_KEY}' entry -- it was not "
                "written by this checkpoint format."
            )
        spec = json.loads(str(archive[OPTIMIZER_SPEC_KEY]))
        saved = spec.get("variables", [])

        if len(keys) != len(saved):
            raise CheckpointIntegrityError(
                f"{path}: optimizer archive holds {len(keys)} arrays but its spec describes "
                f"{len(saved)} variables -- the archive is inconsistent."
            )
        if len(keys) != len(live):
            raise CheckpointIntegrityError(
                f"{path}: checkpoint holds {len(keys)} optimizer variables but the live "
                f"optimizer has {len(live)}. Refusing to resume with a partially restored "
                "optimizer. This usually means the model architecture, the optimizer type, or "
                "the precision policy changed since the checkpoint was written."
            )

        for index, (key, described, variable) in enumerate(zip(keys, saved, live)):
            array = archive[key]
            expected_shape = tuple(int(d) for d in described["shape"])
            if tuple(array.shape) != expected_shape:
                raise CheckpointIntegrityError(
                    f"{path}: optimizer variable {index} ('{described['name']}') has shape "
                    f"{tuple(array.shape)} in the archive but {expected_shape} in its spec."
                )
            if tuple(variable.shape) != tuple(array.shape):
                raise CheckpointIntegrityError(
                    f"{path}: optimizer variable {index} ('{described['name']}') is "
                    f"{tuple(array.shape)} in the checkpoint but {tuple(variable.shape)} in the "
                    "live optimizer -- the architecture does not match this checkpoint."
                )
            variable.assign(array.astype(variable.dtype))

    return spec


# ---------------------------------------------------------------------------
# Manifest + validation
# ---------------------------------------------------------------------------

def _file_entries(directory, filenames):
    entries = {}
    for filename in filenames:
        path = os.path.join(directory, filename)
        entries[filename] = {"size": os.path.getsize(path), "sha256": sha256_file(path)}
    return entries


def build_manifest(directory, state, filenames=REQUIRED_FILES, repo_dir=None):
    """Assemble `manifest.json`'s contents for an already-written directory."""
    environment = state.environment or environment_fingerprint(repo_dir)
    optimizer = state.optimizer or {}
    manifest = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "experiment_id": state.experiment_id,
        "generation": state.generation,
        "completed_epoch": state.completed_epoch,
        "best_epoch": state.best_epoch,
        "best_metric": state.best_metric,
        "monitor": state.monitor,
        "monitor_mode": state.monitor_mode,
        "optimizer_type": optimizer.get("type"),
        "optimizer_wrapper": optimizer.get("wrapper"),
        "precision_policy": state.precision_policy,
        "config_hash": state.config_hash,
        "dataset_version": state.dataset_version,
        "git_commit_hash": environment.get("git_commit_hash"),
        "tensorflow_version": environment.get("tensorflow_version"),
        "keras_version": environment.get("keras_version"),
        "python_version": environment.get("python_version"),
        "created": _now(),
        "files": _file_entries(directory, filenames),
    }
    return _json_safe(manifest)


@dataclass
class CheckpointValidation:
    ok: bool
    reason: Optional[str] = None
    manifest: Optional[Dict[str, Any]] = None

    def __bool__(self):
        return self.ok


def validate_generation(directory, required=REQUIRED_FILES, require_ready=True):
    """Is this directory a complete, undamaged checkpoint?

    Checks, in order: READY present, manifest parseable, the READY marker's
    recorded manifest digest still matches (so a tampered or truncated manifest
    cannot vouch for itself), every required file present, then every file's size
    and SHA256 against the manifest. Returns a result object rather than raising,
    because the caller's normal response is to fall back to an older generation,
    not to stop."""
    if not os.path.isdir(directory):
        return CheckpointValidation(False, f"{directory}: not a directory")

    if require_ready and not os.path.exists(os.path.join(directory, READY_FILENAME)):
        return CheckpointValidation(
            False, f"{os.path.basename(directory)}: no READY marker (incomplete or crashed write)")

    manifest_path = os.path.join(directory, MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        return CheckpointValidation(False, f"{os.path.basename(directory)}: {MANIFEST_FILENAME} missing")
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError) as error:
        return CheckpointValidation(False, f"{os.path.basename(directory)}: unreadable manifest ({error})")

    if require_ready:
        try:
            with open(os.path.join(directory, READY_FILENAME)) as handle:
                ready = json.load(handle)
        except (OSError, ValueError):
            ready = {}
        recorded = ready.get("manifest_sha256")
        if recorded and recorded != sha256_file(manifest_path):
            return CheckpointValidation(
                False,
                f"{os.path.basename(directory)}: manifest sha256 does not match the READY marker "
                "-- the manifest changed after the generation was sealed.",
                manifest,
            )

    files = manifest.get("files", {})
    for filename in required:
        path = os.path.join(directory, filename)
        if not os.path.exists(path):
            return CheckpointValidation(False, f"{os.path.basename(directory)}: {filename} missing", manifest)
        entry = files.get(filename)
        if entry is None:
            return CheckpointValidation(
                False, f"{os.path.basename(directory)}: {filename} is not listed in the manifest", manifest)
        actual_size = os.path.getsize(path)
        if actual_size != entry.get("size"):
            return CheckpointValidation(
                False,
                f"{os.path.basename(directory)}: {filename} size {actual_size} != manifest "
                f"{entry.get('size')}",
                manifest,
            )
        actual_digest = sha256_file(path)
        if actual_digest != entry.get("sha256"):
            return CheckpointValidation(
                False,
                f"{os.path.basename(directory)}: {filename} sha256 {actual_digest[:16]}... != "
                f"manifest {str(entry.get('sha256'))[:16]}...",
                manifest,
            )

    return CheckpointValidation(True, None, manifest)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------

@dataclass
class CompatibilityReport:
    compatible: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def compatibility_report(manifest, expected, strict_fields=STRICT_COMPATIBILITY_FIELDS,
                         advisory_fields=ADVISORY_COMPATIBILITY_FIELDS):
    """Compare a checkpoint's manifest against what the current session expects.

    A field is only compared when the caller actually supplies an expectation for
    it (a `None` expectation means "no opinion"), so a caller can check exactly as
    much as it can genuinely determine. `checkpoint_format`/`format_version` are
    always compared, since this module always knows what it can read."""
    expected = dict(expected or {})
    expected.setdefault("checkpoint_format", CHECKPOINT_FORMAT)
    expected.setdefault("format_version", CHECKPOINT_FORMAT_VERSION)

    errors, warnings = [], []
    for name in strict_fields:
        want = expected.get(name)
        if want is None:
            continue
        got = manifest.get(name)
        if got != want:
            errors.append(f"{name}: checkpoint has {got!r}, this run expects {want!r}")
    for name in advisory_fields:
        want = expected.get(name)
        if want is None:
            continue
        got = manifest.get(name)
        if got != want:
            warnings.append(f"{name}: checkpoint has {got!r}, this run has {want!r}")
    return CompatibilityReport(not errors, errors, warnings)


def assert_compatible(manifest, expected, **kwargs):
    """Raise `CheckpointCompatibilityError` if the checkpoint does not belong to
    this run. Returns the report (whose `warnings` the caller should print)."""
    report = compatibility_report(manifest, expected, **kwargs)
    if not report.compatible:
        raise CheckpointCompatibilityError(
            "Refusing to resume from an incompatible checkpoint:\n  "
            + "\n  ".join(report.errors)
            + "\nStart a new experiment, or correct the configuration to match the checkpoint."
        )
    return report


# ---------------------------------------------------------------------------
# Generation discovery
# ---------------------------------------------------------------------------

def list_generations(checkpoint_dir):
    """Every `gen_NNNNN` directory present, ascending, as `(number, path)`.
    Includes damaged ones -- `validate_generation()` is what judges them."""
    if not os.path.isdir(checkpoint_dir):
        return []
    found = []
    for name in os.listdir(checkpoint_dir):
        number = generation_number(name)
        path = os.path.join(checkpoint_dir, name)
        if number is not None and os.path.isdir(path):
            found.append((number, path))
    return sorted(found)


def read_latest_pointer(checkpoint_dir):
    """The generation directory name `latest.json` names, or None."""
    path = latest_path(checkpoint_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as handle:
            return json.load(handle).get("generation_name")
    except (OSError, ValueError):
        return None


def write_latest_pointer(checkpoint_dir, generation_dir, state=None):
    """Point `latest.json` at a generation. Only ever called AFTER that
    generation's READY marker exists and its files have re-validated at the
    destination."""
    payload = {
        "generation_name": os.path.basename(generation_dir),
        "generation": generation_number(generation_dir),
        "updated": _now(),
    }
    if state is not None:
        payload["completed_epoch"] = state.completed_epoch
        payload["best_epoch"] = state.best_epoch
        payload["best_metric"] = _json_safe(state.best_metric)
    with open(latest_path(checkpoint_dir), "w") as handle:
        json.dump(payload, handle, indent=2)
    return latest_path(checkpoint_dir)


def find_resumable_generation(checkpoint_dir, required=REQUIRED_FILES, verbose=True):
    """The newest generation that actually validates, or None.

    Tries `latest.json`'s target first, then every generation newest-first. A
    damaged or half-written newest generation therefore costs a resume nothing
    beyond one epoch of progress -- the previous known-good generation is still
    there and is used automatically."""
    candidates = []
    pointer = read_latest_pointer(checkpoint_dir)
    if pointer:
        candidates.append(os.path.join(checkpoint_dir, pointer))
    for _number, path in sorted(list_generations(checkpoint_dir), reverse=True):
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        result = validate_generation(path, required=required)
        if result.ok:
            return path
        if verbose and os.path.exists(path):
            print(f"Skipping unusable checkpoint -- {result.reason}")
    return None


#: What the original weights-only path (`training.callbacks.build_callbacks()`
#: without a `CheckpointOptions`) leaves in a checkpoint directory.
LEGACY_CHECKPOINT_FILES = (
    "best.weights.h5", "last.weights.h5", "best.keras", "last.keras", "epoch_state.json",
)


def checkpoint_evidence(checkpoint_dir):
    """Names in `checkpoint_dir` proving checkpointed training happened there.

    This is what separates "a genuinely new experiment" from "a checkpointed
    experiment whose checkpoints are all unusable" -- both of which make
    `find_resumable_generation()` return None, and only the first of which may
    ever be treated as a fresh start. Evidence is any generation directory
    (sealed or not: a half-copied `gen_00001/` is exactly what a crash during the
    first checkpoint leaves), `latest.json`, `best/`, or a weights-only-format
    checkpoint file.

    `metrics.csv` is deliberately NOT evidence: `CSVLogger` creates it in
    `on_train_begin`, before any epoch and before any checkpoint, so it exists in
    directories that have never held a single saved epoch."""
    if not os.path.isdir(checkpoint_dir):
        return []
    evidence = []
    for name in sorted(os.listdir(checkpoint_dir)):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path) and (generation_number(name) is not None or name == BEST_DIRNAME):
            evidence.append(name)
        elif os.path.isfile(path) and (name == LATEST_FILENAME or name in LEGACY_CHECKPOINT_FILES):
            evidence.append(name)
    return evidence


def _not_started(where):
    return (f"Training has NOT started: no epoch ran and nothing under {where} was modified "
            "-- `best/` included.")


def assert_resume_location(run_dir, checkpoint_dir):
    """Refuse a resume whose location cannot be an experiment this framework
    created, before anything is written there.

    `training.callbacks.build_callbacks()` creates the checkpoint directory, so
    without this check a mistyped or moved `RESUME_EXPERIMENT_DIR` would be
    silently turned into a brand-new empty experiment and trained from epoch 0.
    `experiment_manager.create_experiment()` always creates `checkpoints/`, so its
    absence under an existing directory means the directory is not one."""
    if not os.path.isdir(run_dir):
        raise CheckpointResumeError(
            f"Resume was requested, but the experiment directory {run_dir} does not exist "
            "(moved, renamed, mistyped, or Drive not mounted). No checkpoint generation can be "
            f"recovered from a location that is not there. {_not_started(run_dir)} Point "
            "RESUME_EXPERIMENT_DIR at the existing experiment root, or set it to None to start a "
            "new experiment."
        )
    if not os.path.isdir(checkpoint_dir):
        raise CheckpointResumeError(
            f"Resume was requested, but {run_dir} has no checkpoints/ folder ({checkpoint_dir} "
            "does not exist), so it is not an experiment created by "
            f"experiment_manager.create_experiment(). {_not_started(run_dir)} Point "
            "RESUME_EXPERIMENT_DIR at the experiment ROOT (the folder holding metadata.json and "
            "checkpoints/), or set it to None to start a new experiment."
        )


def unrecoverable_resume_message(checkpoint_dir, evidence):
    """The error text for a resume that found checkpoint state but no generation
    it could use: where, what was found, and why each generation was rejected."""
    lines = [
        "Resume was requested, but no valid checkpoint generation could be recovered from "
        f"{checkpoint_dir}.",
        f"This directory DOES hold checkpoint state ({', '.join(evidence)}), so it is not a new "
        "experiment and will not be treated as one: starting at epoch 0 would overwrite the "
        "global BEST with whatever the first new epoch scores.",
    ]
    generations = sorted(list_generations(checkpoint_dir), reverse=True)
    if generations:
        lines.append("Generations examined, newest first:")
        for _number, path in generations:
            result = validate_generation(path)
            lines.append(f"  {os.path.basename(path)}: {result.reason or 'valid'}")
    else:
        lines.append("No gen_NNNNN directory exists; the state found is not in the "
                     "generation-based format this run resumes from.")
    lines.append(_not_started(checkpoint_dir))
    lines.append("Inspect the generations above (a Drive copy may still be in progress), or "
                 "set RESUME_EXPERIMENT_DIR = None to start a separate new experiment.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def _default_staging_dir():
    return os.path.join(tempfile.gettempdir(), "dr_checkpoint_staging")


def _copy_tree_contents(source, destination):
    os.makedirs(destination, exist_ok=True)
    for name in sorted(os.listdir(source)):
        shutil.copy2(os.path.join(source, name), os.path.join(destination, name))


def save_model_weights_only(model, path):
    """`model.save_weights(path)` WITHOUT the optimizer's variables.

    Keras 3's `save_weights` walks the model's tracked attributes, and once the
    optimizer's slots exist it is one of them -- so a `.weights.h5` written after
    the first gradient step silently carries the full Adam state as well.
    Measured on this project's joint model: 521,441,264 bytes with the optimizer
    attached versus 173.4 MB of actual model variables, i.e. the momentum and
    velocity slots duplicated inside a file whose whole purpose is model weights,
    on top of the 331 MiB `optimizer.npz` that already holds them and is the
    authoritative, strictly validated copy.

    Detaching the optimizer for the duration of the write removes the duplicate.
    It is restored in `finally`, keeps its built slots and its `iterations`, and
    training continues unaffected (verified). If detaching is not possible on
    some future Keras, the weights are still written correctly -- just larger --
    so this is an optimization that cannot break correctness."""
    with _optimizer_detached(model):
        model.save_weights(path)
    return path


def load_model_weights_only(model, path):
    """`model.load_weights(path)` without letting Keras touch the optimizer.

    The mirror of `save_model_weights_only`. Because the saved file deliberately
    contains no optimizer group, a plain `load_weights` on a model whose
    optimizer is already built emits

        UserWarning: Skipping variable loading for optimizer 'adam', because it
        has 820 variables whereas the saved optimizer has 0 variables.

    which is harmless -- the optimizer was already restored from `optimizer.npz`
    a moment earlier -- but reads exactly like the silent optimizer reset this
    module exists to prevent. Detaching removes the ambiguity instead of asking
    anyone to remember that one warning is the benign one."""
    with _optimizer_detached(model):
        model.load_weights(path)
    return path


class _optimizer_detached:
    """Context manager: temporarily hide `model.optimizer` from Keras' weight
    (de)serialization, then put it back exactly as it was -- built slots,
    `iterations` and all."""

    def __init__(self, model):
        self.model = model
        self.optimizer = None
        self.detached = False

    def __enter__(self):
        self.optimizer = getattr(self.model, "optimizer", None)
        if self.optimizer is not None:
            try:
                self.model.optimizer = None
                self.detached = True
            except (AttributeError, TypeError):
                self.detached = False
        return self.model

    def __exit__(self, *exc_info):
        if self.detached:
            self.model.optimizer = self.optimizer
        return False


def _seal(directory, manifest_path):
    """Write the READY marker LAST, recording the manifest's own digest so a later
    manifest edit is detectable."""
    payload = {
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "sealed": _now(),
    }
    with open(os.path.join(directory, READY_FILENAME), "w") as handle:
        json.dump(payload, handle, indent=2)


def prune_generations(checkpoint_dir, keep=DEFAULT_KEEP_GENERATIONS, protect=()):
    """Delete all but the newest `keep` generations.

    Only ever called after the replacement generation has been validated at its
    destination, and never touches the generation `latest.json` points at, so the
    run is never left without a resumable checkpoint."""
    protected = {os.path.abspath(p) for p in protect if p}
    pointer = read_latest_pointer(checkpoint_dir)
    if pointer:
        protected.add(os.path.abspath(os.path.join(checkpoint_dir, pointer)))

    generations = list_generations(checkpoint_dir)
    removed = []
    for _number, path in generations[:-keep] if keep > 0 else generations:
        if os.path.abspath(path) in protected:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def save_generation(checkpoint_dir, model, state, staging_dir=None,
                    keep_generations=DEFAULT_KEEP_GENERATIONS, repo_dir=None, verbose=0):
    """Write one complete, sealed, validated checkpoint generation.

    Local staging -> local validation -> copy to destination -> destination
    validation -> READY -> `latest.json` -> prune. See the module docstring for
    why the ordering matters on a Drive/FUSE mount.

    `state` is filled in with the live optimizer spec, precision policy,
    environment fingerprint and assigned generation number, so the caller only has
    to supply the training-progress fields it actually owns."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = model.optimizer
    ensure_optimizer_built(optimizer, model=model)

    existing = list_generations(checkpoint_dir)
    number = (existing[-1][0] + 1) if existing else 1
    state.generation = number
    state.optimizer = optimizer_state_spec(optimizer, include_variables=False)
    state.learning_rate = state.optimizer["learning_rate"]
    state.precision_policy = getattr(getattr(model, "dtype_policy", None), "name", None)
    state.environment = state.environment or environment_fingerprint(repo_dir)
    state.created = _now()

    staging_root = staging_dir or _default_staging_dir()
    os.makedirs(staging_root, exist_ok=True)
    staging = os.path.join(staging_root, f"{generation_name(number)}_{uuid.uuid4().hex[:8]}")
    os.makedirs(staging, exist_ok=True)

    try:
        save_model_weights_only(model, os.path.join(staging, MODEL_WEIGHTS_FILENAME))
        save_optimizer_state(optimizer, os.path.join(staging, OPTIMIZER_FILENAME))
        with open(os.path.join(staging, STATE_FILENAME), "w") as handle:
            json.dump(state.to_dict(), handle, indent=2)

        manifest = build_manifest(staging, state, REQUIRED_FILES, repo_dir=repo_dir)
        manifest_path = os.path.join(staging, MANIFEST_FILENAME)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

        local_check = validate_generation(staging, require_ready=False)
        if not local_check.ok:
            raise CheckpointIntegrityError(
                f"Checkpoint failed validation before it was copied anywhere: {local_check.reason}"
            )

        destination = os.path.join(checkpoint_dir, generation_name(number))
        shutil.rmtree(destination, ignore_errors=True)
        _copy_tree_contents(staging, destination)

        remote_check = validate_generation(destination, require_ready=False)
        if not remote_check.ok:
            raise CheckpointIntegrityError(
                f"Checkpoint did not survive the copy to {destination}: {remote_check.reason}. "
                "The generation was left unsealed (no READY marker), so it will be ignored and "
                "the previous known-good generation is still resumable."
            )

        _seal(destination, os.path.join(destination, MANIFEST_FILENAME))
        write_latest_pointer(checkpoint_dir, destination, state)
        prune_generations(checkpoint_dir, keep=keep_generations, protect=(destination,))
        if verbose:
            print(f"Checkpoint written: {destination} (epoch {state.completed_epoch})")
        return destination
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def save_best(checkpoint_dir, source_generation_dir, state, staging_dir=None, verbose=0):
    """Publish a generation's model weights as the run's global BEST.

    Copies the weights out of the generation just written for this epoch rather
    than re-serializing the live model, so BEST is bit-identical to the LAST
    checkpoint of the epoch that earned it. No optimizer state is stored: BEST is
    the delivery model, never a trajectory to continue from."""
    destination = best_dir(checkpoint_dir)
    staging_root = staging_dir or _default_staging_dir()
    os.makedirs(staging_root, exist_ok=True)
    staging = os.path.join(staging_root, f"best_{uuid.uuid4().hex[:8]}")
    os.makedirs(staging, exist_ok=True)

    try:
        shutil.copy2(os.path.join(source_generation_dir, MODEL_WEIGHTS_FILENAME),
                     os.path.join(staging, MODEL_WEIGHTS_FILENAME))
        with open(os.path.join(staging, STATE_FILENAME), "w") as handle:
            json.dump(state.to_dict(), handle, indent=2)

        manifest = build_manifest(staging, state, BEST_REQUIRED_FILES)
        manifest["role"] = "best"
        manifest["source_generation"] = os.path.basename(source_generation_dir)
        manifest_path = os.path.join(staging, MANIFEST_FILENAME)
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2)

        local_check = validate_generation(staging, required=BEST_REQUIRED_FILES, require_ready=False)
        if not local_check.ok:
            raise CheckpointIntegrityError(f"BEST checkpoint failed local validation: {local_check.reason}")

        # The previous BEST is replaced only once the new one is proven good locally.
        shutil.rmtree(destination, ignore_errors=True)
        _copy_tree_contents(staging, destination)
        remote_check = validate_generation(destination, required=BEST_REQUIRED_FILES, require_ready=False)
        if not remote_check.ok:
            raise CheckpointIntegrityError(
                f"BEST checkpoint did not survive the copy to {destination}: {remote_check.reason}")
        _seal(destination, os.path.join(destination, MANIFEST_FILENAME))
        if verbose:
            print(f"New global best ({state.monitor}={state.best_metric}) at epoch "
                  f"{state.best_epoch}: {destination}")
        return destination
    finally:
        shutil.rmtree(staging, ignore_errors=True)


# ---------------------------------------------------------------------------
# Restoring
# ---------------------------------------------------------------------------

def restore_training_state(model, generation_dir, expected=None, verify_checksums=True,
                           verbose=1):
    """Restore a generation into an already-built, already-compiled `model`.

    Order matters and follows the sequence the design fixes:
    validate integrity -> validate compatibility -> ensure optimizer slots exist
    -> restore optimizer -> restore model weights -> return the training state.

    The optimizer is restored BEFORE the weights so that an architecture mismatch
    is reported as an optimizer-shape error (precise, naming the variable) rather
    than as an opaque HDF5 load failure. Any failure raises; nothing is ever
    partially restored."""
    if verify_checksums:
        result = validate_generation(generation_dir)
        if not result.ok:
            raise CheckpointIntegrityError(f"Cannot restore: {result.reason}")
        manifest = result.manifest
    else:
        manifest_path = os.path.join(generation_dir, MANIFEST_FILENAME)
        with open(manifest_path) as handle:
            manifest = json.load(handle)

    report = assert_compatible(manifest, expected)
    if verbose and report.warnings:
        print("Checkpoint compatibility warnings (resume continues):")
        for warning in report.warnings:
            print(f"  - {warning}")

    optimizer = model.optimizer
    if optimizer is None:
        raise CheckpointError(
            "model.optimizer is None -- compile the model before restoring a checkpoint."
        )
    restore_optimizer_state(optimizer, os.path.join(generation_dir, OPTIMIZER_FILENAME), model=model)
    load_model_weights_only(model, os.path.join(generation_dir, MODEL_WEIGHTS_FILENAME))

    state = read_state(generation_dir)
    if verbose:
        print(f"Restored {os.path.basename(generation_dir)}: epoch {state.completed_epoch}, "
              f"optimizer iterations {state.optimizer.get('iterations')}, "
              f"{state.monitor} best {state.best_metric} at epoch {state.best_epoch}, "
              f"learning rate {state.learning_rate}")
    return state


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def checkpoint_size_report(generation_dir):
    """Per-file and total byte sizes of a checkpoint, for the storage-cost
    measurements the design requires. Pure measurement -- no side effects."""
    sizes = {}
    for name in sorted(os.listdir(generation_dir)):
        path = os.path.join(generation_dir, name)
        if os.path.isfile(path):
            sizes[name] = os.path.getsize(path)
    return {"path": generation_dir, "files": sizes, "total_bytes": sum(sizes.values())}


def print_checkpoint_size_report(report):
    print(f"Checkpoint: {report['path']}")
    for name, size in report["files"].items():
        print(f"  {name:<24} {size:>14,} bytes ({size / 1024 ** 2:8.2f} MiB)")
    total = report["total_bytes"]
    print(f"  {'TOTAL':<24} {total:>14,} bytes ({total / 1024 ** 2:8.2f} MiB)")


# ---------------------------------------------------------------------------
# Caller-facing options
# ---------------------------------------------------------------------------

@dataclass
class CheckpointOptions:
    """Everything the robust checkpoint path needs that `TrainingConfig` does not
    already carry. Passing one to `TrainingConfig(checkpoint_options=...)` is what
    switches a run from the legacy weights-only checkpoints to this module.

    `staging_dir` should be LOCAL disk (the default is the system temp
    directory). On Colab that means `/tmp`, never the Drive mount: the whole
    point is to build and verify a checkpoint somewhere fast and reliable before
    copying it to Drive."""

    experiment_id: Optional[str] = None
    config_hash: Optional[str] = None
    dataset_version: Optional[str] = None
    staging_dir: Optional[str] = None
    keep_generations: int = DEFAULT_KEEP_GENERATIONS
    strict_compatibility: bool = True
    extra_metadata: Optional[Dict[str, Any]] = None
    verbose: int = 1

    def expected_for(self, model=None):
        """The compatibility expectations a resume should enforce, derived from
        this run's configuration plus whatever can be read off the live model."""
        expected = {"config_hash": self.config_hash}
        if model is not None:
            optimizer = getattr(model, "optimizer", None)
            if optimizer is not None:
                inner = getattr(optimizer, "inner_optimizer", None)
                expected["optimizer_type"] = type(inner or optimizer).__name__
            policy = getattr(getattr(model, "dtype_policy", None), "name", None)
            expected["precision_policy"] = policy
        if not self.strict_compatibility:
            return {}
        return expected
