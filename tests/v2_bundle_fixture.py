"""Shared tiny synthetic v2 bundle, built through the REAL stage4_v2_aptos_cache pipeline functions
(legacy-style loose files -> parity -> copies -> Stage-4 generation -> canary -> manifests -> bundle).
Only the Stage-4 network is replaced by a deterministic fake map function; nothing large is generated."""
import os

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import stage4_v2_aptos_cache as ac

MODEL_SHA = "b" * 64
TRAIN = [("00000000000a", 0), ("00000000000b", 2), ("00000000000c", 4)]
VAL = [("00000000000d", 1), ("00000000000e", 3)]
EMPTY_FOV_ONE = (v2cfg.APTOS_EMPTY_FOV_IDS[0], 2)        # present in the split, must be dropped


def fake_native(image_id):
    rng = np.random.default_rng(int(image_id, 16))
    return rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)


def fake_maps(rgb_native):
    """Deterministic stand-in for stage4_v2.pathology_cache_maps (uint8 512x512x8)."""
    rng = np.random.default_rng(int(rgb_native.sum()) % (2 ** 31))
    probs = rng.random((1536, 1536, 4), dtype=np.float32) ** 4
    return cache.pack_pathology_maps(probs)


def build(tmp, model_sha=MODEL_SHA):
    legacy = os.path.join(tmp, "legacy_loose")
    os.makedirs(legacy)
    roots = {k: os.path.join(tmp, "cache", n) for k, n in (("stage2_rgb_v2", "Stage2"), ("stage3_cache_v2", "Stage3"),
                                                            ("stage4_cache_v2", "Stage4"), ("bundle_v2", "Bundle"))}
    pop = ac.aptos_population(TRAIN + [EMPTY_FOV_ONE], VAL, v2cfg.SPLIT_SHA256,
                              expected_counts={"train": len(TRAIN), "val": len(VAL)})
    ids = ac.population_ids(pop)
    rng = np.random.default_rng(3)
    vessels, rgbs = {}, {}
    for i in ids:
        vessels[i] = rng.uniform(0, 1, (512, 512)).astype(np.float32)
        rgbs[i] = rng.uniform(0, 1, (512, 512, 3)).astype(np.float32)
        np.save(os.path.join(legacy, cache.vessel_filename(i)), vessels[i])
        np.save(os.path.join(legacy, cache.rgb_filename(i)), rgbs[i])
    log = lambda *a: None
    # Stage 3
    p3 = ac.run_parity(ids[:2], lambda i: np.load(os.path.join(legacy, cache.vessel_filename(i))),
                       lambda i: vessels[i] + 5e-5, v2cfg.STAGE3_PARITY_TOL, "stage3")
    d3 = ac.data_dir("stage3_cache_v2", v2cfg.STAGE3_GENERATION, roots)
    sha3 = ac.copy_reused_generation("stage3_cache_v2", ids, legacy, d3, workers=2, log=log)
    vfeat = ac.verify_reused_generation("stage3_cache_v2", ids, d3, sha3, workers=2, features=True)
    s3 = cache.stage3_cache_manifest(lwnet_sha256=v2cfg.STAGE3_LWNET_SHA256, tta=True, parity=p3,
                                     split_sha256=v2cfg.SPLIT_SHA256, files=sha3)
    ac.write_generation_manifest("stage3_cache_v2", v2cfg.STAGE3_GENERATION, s3, roots)
    # Stage 2 RGB
    p2 = ac.run_parity(ids[:2], lambda i: np.load(os.path.join(legacy, cache.rgb_filename(i))),
                       lambda i: rgbs[i], v2cfg.RGB_PARITY_TOL, "rgb")
    d2 = ac.data_dir("stage2_rgb_v2", v2cfg.STAGE2_RGB_GENERATION, roots)
    sha2 = ac.copy_reused_generation("stage2_rgb_v2", ids, legacy, d2, workers=2, log=log)
    ac.verify_reused_generation("stage2_rgb_v2", ids, d2, sha2, workers=2)
    s2 = dict(cache.stage2_cache_manifest(split_sha256=v2cfg.SPLIT_SHA256, files=sha2), parity=p2)
    ac.write_generation_manifest("stage2_rgb_v2", v2cfg.STAGE2_RGB_GENERATION, s2, roots)
    # Stage 4
    sha4 = ac.generate_stage4_maps(None, model_sha, ids, fake_native, v2cfg.STAGE3_LWNET_SHA256, roots=roots,
                                   prefetch=2, writers=2, log=log, compute_maps=fake_maps)
    canary = ac.freshness_canary(ids, model_sha, v2cfg.STAGE3_LWNET_SHA256, fake_native, fake_maps, roots=roots, n=2)
    qfeat = ac.verify_stage4_generation(ids, model_sha, v2cfg.STAGE3_LWNET_SHA256, sha4, roots=roots, workers=2)
    gen4 = cache.stage4_generation_id(model_sha, 4)
    s4 = dict(cache.stage4_cache_manifest(stage4_sha256=model_sha, stage3_sha256=v2cfg.STAGE3_LWNET_SHA256,
                                          classes=v2cfg.STAGE4_V2A_CLASSES, split_sha256=v2cfg.SPLIT_SHA256, files=sha4),
              canary=canary, population_sha256=pop["population_sha256"], stage4_gate={"PASS": True})
    ac.write_generation_manifest("stage4_cache_v2", gen4, s4, roots)
    bundle = ac.bundle_manifest_v2(s2, s3, s4, pop, expected_population=len(ids))
    ac.write_bundle(bundle, roots)
    return {"roots": roots, "population": pop, "ids": ids, "bundle": bundle, "s2": s2, "s3": s3, "s4": s4,
            "vfeat": vfeat, "qfeat": qfeat, "legacy": legacy, "gen4": gen4, "vessels": vessels, "rgbs": rgbs,
            "sha4": sha4}
