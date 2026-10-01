"""Post-training APTOS cache generation for the v2 pipeline (research record §40 step 4, §46).

Builds, for the 3,651-image APTOS population, the three generations a bundle binds:

  cache/Stage2/rgb-v1/             canonical 512 RGB (copied from the loose legacy RGB files after an exact
                                   parity check against the Stage-2 output; never regenerated differently)
  cache/Stage3/s3-91f0cada/        Stage-3 vessel maps (copied after the 25-image LWNet parity check; never
                                   regenerated unless parity fails)
  cache/Stage4/s4v2-<sha12>-K4/    Stage-4 v2 lesion maps of ONE trained model: uint8 512x512x8, channels
                                   [MA:mean, MA:max, HE:mean, HE:max, EX:mean, EX:max, SE:mean, SE:max],
                                   exact 3x3 block mean+max of the 1536 probabilities
  cache/Bundle/<rgb>__<s3>__<s4>/  bundle manifest binding the three (incl. Stage-3 SHA and model SHA)

Rules: a Stage-4 generation is immutable (its directory is named by the model SHA; a completed generation is
never rewritten; a partial one resumes only for the same model SHA); legacy Stage-4 caches are never read;
the only legacy reads are loose `_vessel_` / `_rgb_` files (stage34_cache_v2 guards); every file is
SHA-recorded; every step fails loudly on a mismatch. Heavy I/O work runs in thread pools (Drive FUSE).
The model-dependent functions take the model / loader as arguments so they are CPU-testable."""
import concurrent.futures as cf
import datetime
import hashlib
import io
import json
import os
import re

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache

CLASSES = v2cfg.STAGE4_V2A_CLASSES
_APTOS_ID = re.compile(r"^[0-9a-f]{12}$")


class AptosCacheError(RuntimeError):
    """A downstream cache contract was violated."""


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    cache.assert_not_legacy_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- population

def aptos_population(train_entries, val_entries, split_sha256, expected_counts=None):
    """The 3,651-image population: the verified split minus the 11 pinned empty-FOV ids.
    Returns {"train": [(id, grade)], "val": [...], "split_sha256", "population_sha256"}."""
    if split_sha256 != v2cfg.SPLIT_SHA256:
        raise AptosCacheError(f"split sha {split_sha256} != pinned {v2cfg.SPLIT_SHA256}")
    empty = set(v2cfg.APTOS_EMPTY_FOV_IDS)
    pop = {"train": sorted((str(i), int(g)) for i, g in train_entries if i not in empty),
           "val": sorted((str(i), int(g)) for i, g in val_entries if i not in empty)}
    counts = {k: len(v) for k, v in pop.items()}
    expected_counts = expected_counts or v2cfg.APTOS_EXPECTED_COUNTS
    if counts != expected_counts:
        raise AptosCacheError(f"population {counts} != {expected_counts}")
    ids = [i for i, _ in pop["train"] + pop["val"]]
    if len(set(ids)) != len(ids) or not all(_APTOS_ID.match(i) for i in ids):
        raise AptosCacheError("population ids are not unique APTOS ids (IDRiD or other ids are refused)")
    pop["split_sha256"] = split_sha256
    pop["population_sha256"] = population_sha256(pop)
    return pop


def population_sha256(pop):
    payload = "\n".join(f"{role},{i},{g}" for role in ("train", "val") for i, g in pop[role])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def population_ids(pop):
    return sorted(i for role in ("train", "val") for i, _ in pop[role])


def assert_aptos_ids(ids):
    bad = [i for i in ids if not _APTOS_ID.match(str(i))]
    if bad:
        raise AptosCacheError(f"non-APTOS ids in a downstream manifest (e.g. IDRiD): {bad[:5]}")
    return True


def parity_ids(ids, n, seed=v2cfg.PARITY_SEED):
    """A fixed-seed sample of the population (sorted first, so independent of input order)."""
    return sorted(np.random.default_rng(seed).choice(sorted(ids), n, replace=False).tolist())


# --------------------------------------------------------------------------- generation directories

def generation_dir(kind, generation_id, roots=None):
    roots = roots or {"stage2_rgb_v2": v2cfg.STAGE2_ROOT, "stage3_cache_v2": v2cfg.STAGE3_ROOT,
                      "stage4_cache_v2": v2cfg.STAGE4_ROOT, "bundle_v2": v2cfg.BUNDLE_ROOT}
    return os.path.join(roots[kind], generation_id)


def data_dir(kind, generation_id, roots=None):
    return os.path.join(generation_dir(kind, generation_id, roots), cache.DATA_SUBDIRS[kind])


def manifest_path(kind, generation_id, roots=None):
    return os.path.join(generation_dir(kind, generation_id, roots), cache.MANIFEST_NAMES[kind])


# --------------------------------------------------------------------------- parity (Stage 3 and RGB)

def run_parity(ids, read_cached, recompute, tol, label):
    """max |cached - recomputed| per id; raises StaleCacheError above `tol`."""
    deltas = {}
    for i in ids:
        a = np.asarray(read_cached(i), np.float64).squeeze()
        b = np.asarray(recompute(i), np.float64).squeeze()
        if a.shape != b.shape:
            raise cache.StaleCacheError(f"{label} parity {i}: shape {a.shape} vs {b.shape}")
        deltas[i] = float(np.max(np.abs(a - b)))
    worst = max(deltas.values())
    report = {"label": label, "ids": list(ids), "n": len(ids), "max_abs_delta": worst, "tol": tol,
              "per_id": deltas, "passed": bool(worst <= tol), "checked_utc": _now()}
    if not report["passed"]:
        raise cache.StaleCacheError(f"{label} parity FAILED: max |delta| {worst:.3g} > {tol} -- do not reuse; "
                                    "regenerate this generation instead")
    return report


def recompute_vessel_512(rgb_native, vessel_model):
    """Exactly how the reused vessel cache was produced (joint_training_dataset): LWNet on the native
    Stage-2 RGB (VESSEL_SEG_TTA=True), then channel 3 of racaf.prepare_stage4_input (joint 512 resize)."""
    import racaf
    from vessel_segmentation_inference import predict_vessel_mask
    vessel = predict_vessel_mask(rgb_native, model=vessel_model, tta=True)["probability_map"].astype(np.float32)
    return racaf.prepare_stage4_input(rgb_native, vessel)[0, ..., 3]


def recompute_rgb_512(rgb_native):
    """joint_training_dataset._resize_rgb_01 -- the canonical 512 frame."""
    import joint_training_dataset as jtd
    return jtd._resize_rgb_01(rgb_native, (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE))


def assert_lwnet(model_path):
    sha = cache.sha256_file(model_path)
    if sha != v2cfg.STAGE3_LWNET_SHA256:
        raise AptosCacheError(f"Stage-3 model {model_path} sha {sha} != pinned LWNet {v2cfg.STAGE3_LWNET_SHA256}")
    return sha


# --------------------------------------------------------------------------- copying reused generations

def _copy_verified(src, dst, validate):
    cache.assert_not_legacy_path(dst)
    with open(src, "rb") as fh:
        payload = fh.read()
    arr = np.load(io.BytesIO(payload), allow_pickle=False)
    validate(arr)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    with open(tmp, "wb") as fh:
        fh.write(payload)
    os.replace(tmp, dst)
    return hashlib.sha256(payload).hexdigest(), arr


def _validate_vessel(arr):
    a = np.asarray(arr).squeeze()
    if a.shape != (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise AptosCacheError(f"vessel map invalid: shape {a.shape}, range [{a.min()}, {a.max()}]")


def _validate_rgb(arr):
    a = np.asarray(arr)
    if a.shape != (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE, 3) or not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise AptosCacheError(f"RGB frame invalid: shape {a.shape}")


def _kind_io(kind):
    if kind == "stage3_cache_v2":
        return cache.vessel_filename, cache.assert_legacy_vessel_source, _validate_vessel
    if kind == "stage2_rgb_v2":
        return cache.rgb_filename, cache.assert_legacy_rgb_source, _validate_rgb
    raise AptosCacheError(f"{kind} cannot be copied from legacy files")


def copy_reused_generation(kind, ids, legacy_dir, dest_dir, workers=16, log=print):
    """Copies loose legacy files for `kind` in {"stage3_cache_v2" (vessel), "stage2_rgb_v2" (rgb)} into the
    v2 generation directory, resumably (copy_progress.json), validating each array. Returns {id: sha256}."""
    name_of, guard, validate = _kind_io(kind)
    assert_aptos_ids(ids)
    progress_path = os.path.join(os.path.dirname(dest_dir), "copy_progress.json")
    done = _read_json(progress_path) if os.path.exists(progress_path) else {}
    todo = [i for i in ids if not (i in done and os.path.exists(os.path.join(dest_dir, name_of(i))))]

    def one(i):
        src = guard(os.path.join(legacy_dir, name_of(i)))
        sha, _ = _copy_verified(src, os.path.join(dest_dir, name_of(i)), validate)
        return i, sha

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (i, sha) in enumerate(pool.map(one, todo), start=1):
            done[i] = sha
            if n % 200 == 0:
                _write_json(progress_path, done)
                log(f"  {kind}: copied {n}/{len(todo)}")
    _write_json(progress_path, done)
    missing = sorted(set(ids) - set(done))
    if missing:
        raise cache.IncompleteCacheError(f"{kind}: {len(missing)} ids not copied, e.g. {missing[:5]}")
    return {i: done[i] for i in ids}


def verify_reused_generation(kind, ids, dest_dir, shas, workers=16, features=False):
    """Full pass over a copied generation: every file's SHA must equal its recorded SHA and its array must
    validate. With features=True also returns the C2 pyramid features ({id: vector})."""
    name_of, _, validate = _kind_io(kind)

    def one(i):
        path = os.path.join(dest_dir, name_of(i))
        with open(path, "rb") as fh:
            payload = fh.read()
        if hashlib.sha256(payload).hexdigest() != shas[i]:
            raise cache.ManifestMismatchError(f"{path}: sha differs from the recorded copy")
        arr = np.load(io.BytesIO(payload), allow_pickle=False)
        validate(arr)
        return i, (cache.pyramid_features(np.asarray(arr, np.float32).reshape(512, 512, -1)) if features else None)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        out = dict(pool.map(one, ids))
    return out if features else True


# --------------------------------------------------------------------------- Stage 4

def configure_determinism():
    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {"cudnn_deterministic": True, "cudnn_benchmark": False}


def bounded_prefetch(keys, load, max_ahead=v2cfg.STAGE4_CACHE_PREFETCH, readers=v2cfg.STAGE4_CACHE_READERS):
    """Yields (key, load(key)) in the order of `keys`, with at most `max_ahead` items resident at any time --
    loading, loaded-and-waiting, or currently held by the consumer (a strict producer/consumer bound;
    ThreadPoolExecutor.map would submit every key at once). A load exception is raised in the consumer at that key's position. On early exit the
    not-yet-started loads are cancelled. The consumer should drop its reference to each item once used."""
    import collections
    if max_ahead < 1 or readers < 1:
        raise ValueError("max_ahead and readers must be >= 1")
    keys = list(keys)
    window = collections.deque()
    nxt = 0
    pool = cf.ThreadPoolExecutor(max_workers=min(readers, max_ahead))
    try:
        while nxt < len(keys) and len(window) < max_ahead:
            window.append((keys[nxt], pool.submit(load, keys[nxt])))
            nxt += 1
        while window:
            key, fut = window.popleft()
            item = fut.result()                         # re-raises a reader failure here, in order
            del fut
            yield key, item
            del item                                    # the consumer is done with this image ...
            if nxt < len(keys):                         # ... only then is one new read started, so queued +
                window.append((keys[nxt], pool.submit(load, keys[nxt])))   # in-use images never exceed max_ahead
                nxt += 1
    finally:
        for _, fut in window:
            fut.cancel()
        pool.shutdown(wait=True, cancel_futures=True)


def generate_stage4_maps(model, model_sha256, ids, load_native, stage3_sha256, device="cuda", amp=True,
                         roots=None, prefetch=v2cfg.STAGE4_CACHE_PREFETCH, readers=v2cfg.STAGE4_CACHE_READERS,
                         writers=v2cfg.STAGE4_CACHE_WRITERS, max_pending_writes=v2cfg.STAGE4_CACHE_MAX_PENDING_WRITES,
                         log=print, compute_maps=None):
    """Stage-4 v2 inference for every id -> uint8 512x512x8 npz per image in the model's own generation.
    `load_native(id)` returns the native Stage-2 RGB uint8. Resumes a partial run of the SAME model; refuses
    a completed generation (immutable) and any directory whose progress names a different model.
    Memory is strictly bounded: at most `prefetch` native images ahead of the GPU (bounded_prefetch) and at most
    `max_pending_writes` finished map arrays waiting to be written. Returns {id: file sha}."""
    import stage4_v2 as s4
    cache.assert_not_deny_listed(model_sha256)
    assert_aptos_ids(ids)
    gen = cache.stage4_generation_id(model_sha256, len(CLASSES))
    out_dir = data_dir("stage4_cache_v2", gen, roots)
    if os.path.exists(manifest_path("stage4_cache_v2", gen, roots)):
        raise AptosCacheError(f"{gen} is complete and immutable; verify it instead of regenerating")
    progress_path = os.path.join(os.path.dirname(out_dir), "progress.json")
    progress = _read_json(progress_path) if os.path.exists(progress_path) else {
        "stage4_sha256": model_sha256, "generation_id": gen, "files": {}}
    if progress["stage4_sha256"] != model_sha256 or progress["generation_id"] != gen:
        raise AptosCacheError(f"{out_dir} holds maps of another model ({progress['stage4_sha256']}); refusing to mix")
    channels = cache.channel_names(CLASSES)
    compute_maps = compute_maps or (lambda rgb: s4.pathology_cache_maps(model, rgb, device=device, amp=amp))
    todo = [i for i in ids if not (i in progress["files"] and os.path.exists(os.path.join(out_dir, cache.pathology_filename(i))))]

    def write(i, maps):
        path = os.path.join(out_dir, cache.pathology_filename(i))
        sha = cache.write_pathology_npz(path, maps, channels=channels, image_id=i, stage4_sha256=model_sha256,
                                        stage3_sha256=stage3_sha256, gen_id=gen)
        return i, sha

    _write_json(progress_path, progress)                   # ownership recorded before any map is written
    pending = []

    def drain(block_until=None):
        """Record finished writes; with `block_until`, first wait until fewer than that many are pending."""
        if block_until is not None:
            running = [f for f in pending if not f.done()]
            while len(running) >= block_until:
                cf.wait(running, return_when=cf.FIRST_COMPLETED)
                running = [f for f in running if not f.done()]
        for f in pending:
            if f.done() and f.exception() is None:
                k, sha = f.result()
                progress["files"][k] = sha
        errors = [f.exception() for f in pending if f.done() and f.exception() is not None]
        pending[:] = [f for f in pending if not f.done()]       # finished futures (and their arrays) released
        if errors:
            raise errors[0]

    import contextlib
    with cf.ThreadPoolExecutor(max_workers=writers) as wpool,             contextlib.closing(bounded_prefetch(todo, load_native, prefetch, readers)) as images:
        try:
            for n, (i, rgb) in enumerate(images, start=1):
                maps = compute_maps(rgb)
                del rgb                                          # native image released before the next read
                if maps.shape != (v2cfg.CACHE_SIZE, v2cfg.CACHE_SIZE, len(channels)) or maps.dtype != np.uint8:
                    raise AptosCacheError(f"{i}: Stage-4 maps {maps.dtype} {maps.shape}")
                drain(block_until=max_pending_writes)            # back-pressure: bounded queued map arrays
                pending.append(wpool.submit(write, i, maps))
                del maps
                if n % 100 == 0:
                    cf.wait(pending)
                    drain()
                    _write_json(progress_path, progress)
                    log(f"  stage-4 maps {n}/{len(todo)}")
        finally:                                            # a crash still records every completed write
            cf.wait(pending)
            try:
                drain()
            finally:
                _write_json(progress_path, progress)
    missing = sorted(set(ids) - set(progress["files"]))
    if missing:
        raise cache.IncompleteCacheError(f"{gen}: {len(missing)} ids without maps, e.g. {missing[:5]}")
    return {i: progress["files"][i] for i in ids}


def verify_stage4_generation(ids, model_sha256, stage3_sha256, shas, roots=None, workers=16):
    """Full pass: every map file is read through the verified reader (file SHA, embedded Stage-4/Stage-3 SHA,
    generation id, channel order, dtype, shape). Returns the C2 Q-pyramid features ({id: vector})."""
    gen = cache.stage4_generation_id(model_sha256, len(CLASSES))
    out_dir = data_dir("stage4_cache_v2", gen, roots)
    if not os.path.isdir(out_dir):
        raise cache.IncompleteCacheError(f"no Stage-4 generation {gen} (model {model_sha256[:12]}) at {out_dir}")
    channels = cache.channel_names(CLASSES)

    def one(i):
        maps = cache.read_pathology_npz(os.path.join(out_dir, cache.pathology_filename(i)),
                                        expected_stage4_sha256=model_sha256, expected_stage3_sha256=stage3_sha256,
                                        expected_gen_id=gen, expected_channels=channels, expected_file_sha256=shas[i])
        return i, cache.pyramid_features(maps)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, ids))


def freshness_canary(ids, model_sha256, stage3_sha256, load_native, compute_maps, roots=None,
                     n=v2cfg.FRESHNESS_CANARY_N, seed=v2cfg.PARITY_SEED):
    """Recompute `n` fixed-seed images and compare with the stored maps (<= 2/255)."""
    gen = cache.stage4_generation_id(model_sha256, len(CLASSES))
    out_dir = data_dir("stage4_cache_v2", gen, roots)
    channels = cache.channel_names(CLASSES)
    deltas = {}
    for i in parity_ids(ids, n, seed + 1):
        stored = cache.read_pathology_npz(os.path.join(out_dir, cache.pathology_filename(i)),
                                          expected_stage4_sha256=model_sha256, expected_stage3_sha256=stage3_sha256,
                                          expected_gen_id=gen, expected_channels=channels)
        deltas[i] = cache.freshness_canary(stored, compute_maps(load_native(i)), v2cfg.CANARY_TOL_COUNTS)
    return {"ids": sorted(deltas), "max_delta_counts": max(deltas.values()), "tol_counts": v2cfg.CANARY_TOL_COUNTS,
            "passed": True, "checked_utc": _now()}


def save_pyramid_features(path, features, ids, channels, provenance):
    """C2 inputs: one row per id (population order), names from stage34_cache_v2.pyramid_feature_names."""
    cache.assert_not_legacy_path(path)
    x = np.stack([features[i] for i in ids]).astype(np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part.npz"
    np.savez_compressed(tmp, ids=np.asarray(ids), features=x,
                        names=np.asarray(cache.pyramid_feature_names(channels)),
                        provenance=np.asarray(json.dumps(provenance, sort_keys=True)))
    os.replace(tmp, path)
    return cache.sha256_file(path)


def load_pyramid_features(path, *, expected):
    """Reads a C2 feature file and checks its provenance against `expected` (e.g. {"stage4_sha256": ...})."""
    with np.load(path, allow_pickle=False) as z:
        prov = json.loads(str(z["provenance"]))
        for k, v in expected.items():
            if prov.get(k) != v:
                raise AptosCacheError(f"{path}: provenance {k} = {prov.get(k)!r}, expected {v!r}")
        return [str(i) for i in z["ids"]], z["features"].astype(np.float64), [str(n) for n in z["names"]], prov


# --------------------------------------------------------------------------- manifests and bundle

def write_generation_manifest(kind, generation_id, manifest, roots=None, immutable=True):
    path = manifest_path(kind, generation_id, roots)
    if os.path.exists(path):
        old = _read_json(path)
        if immutable and old.get("files") != manifest.get("files"):
            raise AptosCacheError(f"{path} exists with different files; generations are immutable")
        return cache.sha256_file(path)
    return cache.write_manifest(path, manifest)


def bundle_manifest_v2(s2, s3, s4, population, expected_population=v2cfg.APTOS_POPULATION):
    """stage34_cache_v2.bundle_manifest plus the downstream contract fields."""
    if s4.get("stage4_sha256") in (None, ""):
        raise AptosCacheError("Stage-4 manifest without a model SHA")
    if not s3.get("parity", {}).get("passed"):
        raise AptosCacheError("Stage-3 parity has not passed")
    if not s2.get("parity", {}).get("passed"):
        raise AptosCacheError("RGB parity has not passed")
    b = cache.bundle_manifest(s2, s3, s4)
    want = set(population_ids(population))
    if set(b["population"]) != want or len(want) != expected_population:
        raise cache.IncompleteCacheError(f"bundle population {len(b['population'])} != the {expected_population} ids")
    fp = {k: hashlib.sha256(json.dumps(m["files"], sort_keys=True).encode()).hexdigest()
          for k, m in (("stage2", s2), ("stage3", s3), ("stage4", s4))}
    b.update({"population_sha256": population["population_sha256"],
              "train_ids": [i for i, _ in population["train"]], "val_ids": [i for i, _ in population["val"]],
              "generation_fingerprints": fp, "stage4_generation": s4["generation_id"], "k": len(CLASSES),
              "classes": list(CLASSES), "preproc_version": v2cfg.PREPROC_VERSION,
              "cache_dims": {"rgb": [512, 512, 3], "vessel": [512, 512, 1], "pathology": [512, 512, 2 * len(CLASSES)]},
              "dtype": {"rgb": "float32 [0,1]", "vessel": "float32 [0,1]", "pathology": "uint8 round(p*255)"},
              "pooling": s4["pooling"], "stage4_gate": s4.get("stage4_gate"), "created_utc": _now()})
    b["fingerprint"] = hashlib.sha256(json.dumps(
        {k: b[k] for k in ("bundle_id", "stage3_sha256", "stage4_sha256", "split_sha256", "population_sha256",
                           "generation_fingerprints", "channels")}, sort_keys=True).encode()).hexdigest()
    return b


def write_bundle(bundle, roots=None):
    path = manifest_path("bundle_v2", bundle["bundle_id"], roots)
    if os.path.exists(path):
        if _read_json(path).get("fingerprint") != bundle["fingerprint"]:
            raise AptosCacheError(f"{path} exists with a different fingerprint; bundles are immutable")
        return path
    cache.write_manifest(path, bundle)
    return path


def require_gate(gate_report_path, model_sha256, override_token=None):
    """The APTOS cache is generated only for a model whose one-time IDRiD gate was run. A FAIL proceeds only
    with the explicit override token (recorded in the Stage-4 manifest)."""
    if not gate_report_path or not os.path.exists(gate_report_path):
        raise AptosCacheError("no IDRiD test-gate report for this model; run the one-time gate first")
    report = _read_json(gate_report_path)
    if report.get("model_sha256") not in (None, model_sha256):
        raise AptosCacheError("the gate report belongs to another Stage-4 model")
    passed = bool(report.get("decision", {}).get("PASS"))
    if not passed and override_token != v2cfg.STAGE4_GATE_OVERRIDE_TOKEN:
        raise AptosCacheError("the Stage-4 model FAILED the one-time gate; set the explicit override token to "
                              "proceed (it is recorded in the manifest)")
    return {"report": gate_report_path, "report_sha256": cache.sha256_file(gate_report_path), "PASS": passed,
            "override": None if passed else override_token}
