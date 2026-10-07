"""The DDR ground-truth lesion probe (research record §68) applied to the EyePACS-adapted APTOS arms (§69.2).

A downstream runner, not a new method. Everything that defines the probe is ddr_probe's and is used as it is:
the pinned 756-image manifest (383 / 148 / 225; `007-5869-300.jpg` excluded), Stage 2 + the 512 frame, the
expert masks reduced to 16 x 16 any-pixel cell targets, the fresh 1 x 1 probe on the frozen 16 x 16 x 768 map
(e1_probe.build_probe; bias from the DDR training-target prior; Adam 1e-3, batch 16, seeded batch order, no
augmentation), the five-epoch probe (primary) and the converged probe (validation-loss stopping rule:
min-delta 1e-4, patience 5, at least 5 and at most 100 epochs, best validation epoch), two test predictions
per probe, and the paired bootstrap over the 225 test images (2,000 resamples, seed 20260927).

Only the encoders differ: the pinned BEST checkpoints of P-EP and E1-EP (APTOS seeds 42, 123, 2026), and of
E2-EP if -- and only if -- it exists by the rule of §69.2.

Primary criterion (§69.2 C; the §68 criterion with the EP arms): on the five-epoch probe, E1-EP - P-EP in mean
cell AUROC positive in 3 / 3 seeds AND the 95 % interval of the three-seed mean excluding zero. The converged
result is reported next to it and never replaces it. `record_e2_ep_gate` writes the criterion result that
ep_aptos_train.require_e2_ep_eligibility reads.

Checkpoints are named by a checkpoint manifest (`build_checkpoint_manifest` writes one from the pinned runs;
no hash is hard-coded here). `verify_checkpoint_manifest` refuses a wrong probe-protocol version, a wrong DDR
manifest hash, a missing or repeated seed, a file whose SHA-256 is not the recorded one, an architecture that
does not belong to its arm, and any checkpoint that is not a pinned run of the EP experiment started from the
adapted encoder -- in particular a P, E1 or E2 checkpoint.
"""
import json
import os
import posixpath

import numpy as np

import ddr_probe as dp
import e1_probe as ep

PROTOCOL_VERSION = "ddr-ground-truth-probe/record-68/v1"
SEEDS = dp.SEEDS
#: arm -> (the graph its checkpoint is loaded in, the model type its run must record)
ARMS = {"p_ep": ("p", "P"), "e1_ep": ("e1", "E1"), "e2_ep": ("e1", "E1")}
REQUIRED_ARMS = ("p_ep", "e1_ep")
CONTRASTS = {"e1_ep_minus_p_ep": ("e1_ep", "p_ep")}
E2_CONTRASTS = {"e2_ep_minus_p_ep": ("e2_ep", "p_ep"), "e1_ep_minus_e2_ep": ("e1_ep", "e2_ep")}
PRIMARY = "e1_ep_minus_p_ep"
CRITERION = ("E1-EP - P-EP mean cell AUROC positive in 3/3 seeds AND the 95% paired bootstrap interval of the "
             "three-seed mean excludes zero (five-epoch probe; 2,000 resamples, seed 20260927)")
#: Descriptive only (record §69.2, Q3): each EP arm against the encoder it mirrors, from the saved §70 logits.
BASELINE_CONTRASTS = {"p_ep_minus_p": ("p_ep", "p"), "e1_ep_minus_e1": ("e1_ep", "e1")}
RESULT_NAME = "ddr_probe_ep_result.json"
BASELINE_RESULT_NAME = "ddr_probe_ep_vs_baseline.json"
E2_EP_PROBE_CONFIRMATION = "PROBE E2-EP"
MANIFEST_KEYS = ("probe_protocol_version", "ddr_manifest_sha256", "adapted_encoder_sha256", "git_commit", "checkpoints")
ENTRY_KEYS = ("arm", "seed", "model_type", "run_dir", "weights", "sha256")


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


def baseline_checkpoint_hashes(repo_dir):
    """The SHA-256 of every pinned P, E1 and E2 checkpoint (and of the ImageNet weights): never an EP checkpoint."""
    hashes = set()
    for name in ("idrid_grading_protocol.json", "idrid_batch1_protocol.json"):
        with open(os.path.join(repo_dir, name), encoding="utf-8") as fh:
            hashes |= {entry["sha256"] for entry in json.load(fh)["checkpoints"].values()}
    return hashes


# --------------------------------------------------------------------------- the checkpoint manifest

def build_checkpoint_manifest(experiments_root, adapted_sha256, arms=REQUIRED_ARMS, *, repo_dir=None, seeds=SEEDS):
    """The checkpoint manifest of the pinned EP runs: one entry per (arm, seed) with the pinned weights' path
    (relative to the experiments root) and SHA-256. Every run must be pinned (its file is re-hashed)."""
    import arch1_train as at
    import ep_aptos_train as epa
    import eyepacs_adaptation_train as eat
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    entries = []
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
        for seed in seeds:
            run_dir = epa.run_dir_for(experiments_root, arm, adapted_sha256, seed)
            _, frozen = eat.read_frozen(run_dir)
            entries.append({"arm": arm, "seed": int(seed), "model_type": ARMS[arm][1],
                            "run_dir": posixpath.relpath(run_dir, experiments_root), "weights": frozen["weights"],
                            "sha256": frozen["sha256"]})
    return {"probe_protocol_version": PROTOCOL_VERSION, "ddr_manifest_sha256": dp.MANIFEST_SHA256,
            "adapted_encoder_sha256": adapted_sha256, "git_commit": at.git_commit(repo_dir), "checkpoints": entries}


def write_checkpoint_manifest(path, manifest):
    """Writes the manifest once; an existing file must hold the same checkpoints."""
    if os.path.exists(path):
        old = _read_json(path)
        if {k: old.get(k) for k in MANIFEST_KEYS if k != "git_commit"} != \
                json.loads(json.dumps({k: manifest[k] for k in MANIFEST_KEYS if k != "git_commit"})):
            raise RuntimeError(f"{path} already names other checkpoints")
        return old
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(path + ".tmp", path)
    return manifest


def verify_checkpoint_manifest(manifest, experiments_root, *, repo_dir=None, eligibility=None, e2_ep_confirmation=None,
                               seeds=SEEDS):
    """{(arm, seed): {"path", "sha256", "kind", "model_type"}} after every check, or an exception.

    Checked: the manifest's fields; the probe-protocol version; the DDR manifest hash; that P-EP and E1-EP are
    both present; that every arm has exactly the seeds 42 / 123 / 2026, each once; that each entry's run is a
    pinned run of the EP experiment for that arm, seed and model type, started from the adapted encoder named
    by the manifest with a fresh head; that the weights file has the pinned SHA-256; that no checkpoint is one
    of the pinned P / E1 / E2 checkpoints. E2-EP entries are accepted only with a met criterion record
    (`eligibility`) AND the explicit confirmation phrase."""
    import ep_aptos_train as epa
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    missing = [k for k in MANIFEST_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"the checkpoint manifest lacks {missing}")
    if manifest["probe_protocol_version"] != PROTOCOL_VERSION:
        raise RuntimeError(f"probe protocol {manifest['probe_protocol_version']!r} is not {PROTOCOL_VERSION!r}")
    if manifest["ddr_manifest_sha256"] != dp.MANIFEST_SHA256:
        raise RuntimeError("the checkpoint manifest names another DDR probe manifest")
    if not manifest["git_commit"] or not manifest["adapted_encoder_sha256"]:
        raise ValueError("the checkpoint manifest must record the commit and the adapted encoder")
    entries = manifest["checkpoints"]
    by_arm = {}
    for entry in entries:
        lacking = [k for k in ENTRY_KEYS if k not in entry]
        if lacking:
            raise ValueError(f"a checkpoint entry lacks {lacking}")
        if entry["arm"] not in ARMS:
            raise RuntimeError(f"{entry['arm']!r} is not an EP arm")
        by_arm.setdefault(entry["arm"], []).append(int(entry["seed"]))
    for arm in REQUIRED_ARMS:
        if arm not in by_arm:
            raise RuntimeError(f"the checkpoint manifest has no {arm} checkpoints")
    for arm, found in by_arm.items():
        if len(found) != len(set(found)):
            raise RuntimeError(f"{arm}: a seed is listed more than once ({sorted(found)})")
        if set(found) != {int(s) for s in seeds}:
            raise RuntimeError(f"{arm}: seeds {sorted(found)} are not exactly {tuple(int(s) for s in seeds)}")
    if "e2_ep" in by_arm:
        epa.require_e2_ep_eligibility(eligibility)               # raises unless the recorded criterion was met
        if e2_ep_confirmation != E2_EP_PROBE_CONFIRMATION:
            raise RuntimeError(f"E2-EP is eligible but not confirmed: pass e2_ep_confirmation={E2_EP_PROBE_CONFIRMATION!r}")
    baseline = baseline_checkpoint_hashes(repo_dir)
    out = {}
    for entry in entries:
        arm, seed = entry["arm"], int(entry["seed"])
        kind, model_type = ARMS[arm]
        label = f"{arm}-{seed}"
        if entry["model_type"] != model_type:
            raise RuntimeError(f"{label}: model type {entry['model_type']!r} is not {model_type!r}")
        if entry["sha256"] in baseline:
            raise RuntimeError(f"{label}: this is a pinned P / E1 / E2 checkpoint, not an EP checkpoint")
        run_dir = os.path.join(experiments_root, *entry["run_dir"].split("/"))
        for name in ("config.json", "frozen_checkpoint.json", "initialization.json"):
            if not os.path.exists(os.path.join(run_dir, name)):
                raise RuntimeError(f"{label}: {entry['run_dir']} is not a pinned EP run (no {name})")
        config, frozen = _read_json(os.path.join(run_dir, "config.json")), _read_json(os.path.join(run_dir, "frozen_checkpoint.json"))
        started = _read_json(os.path.join(run_dir, "initialization.json"))
        if config.get("experiment") != epa.EXPERIMENT or config.get("arm") != arm or int(config.get("seed", -1)) != seed \
                or config.get("model_type") != model_type:
            raise RuntimeError(f"{label}: the run is {config.get('experiment')} / {config.get('arm')} / seed {config.get('seed')} "
                               f"/ {config.get('model_type')}")
        if config.get("adapted_encoder_sha256") != manifest["adapted_encoder_sha256"] or not started.get("encoder_is_adapted") \
                or started.get("adapted_encoder_sha256") != manifest["adapted_encoder_sha256"] or not started.get("fresh_corn_head"):
            raise RuntimeError(f"{label}: the run did not start from the adapted encoder of this manifest with a fresh head")
        if frozen["sha256"] != entry["sha256"] or frozen["weights"] != entry["weights"]:
            raise RuntimeError(f"{label}: the manifest entry is not the run's pinned checkpoint")
        path = os.path.join(run_dir, *entry["weights"].split("/"))
        if not os.path.exists(path) or dp.sha256_file(path) != entry["sha256"]:
            raise RuntimeError(f"{label}: the weights file is missing or does not have the recorded SHA-256")
        out[(arm, seed)] = {"path": path, "sha256": entry["sha256"], "kind": kind, "model_type": model_type}
    return out


# --------------------------------------------------------------------------- the run (GPU for the encoders)

def run(experiments_root, lesion_root, audit_dir, out_dir, checkpoint_manifest, pretrained_path, *, repo_dir=None,
        eligibility=None, e2_ep_confirmation=None, work_dir="/content/ddr_probe_work", seeds=SEEDS, log=print):
    """ddr_probe.run for the EP encoders: the same data preparation (`ddr_probe.prepare_data`) and the same
    per-encoder step (`ddr_probe.probe_encoder`). Resumable per (arm, seed). Writes configuration.json,
    run_metadata.json, targets.npz, probe_<arm>_seed<seed>.npz and summary.json. The comparison is `analyse`."""
    import keras

    import pl_convnext as pl
    repo_dir = repo_dir or os.path.dirname(os.path.abspath(__file__))
    checkpoints = verify_checkpoint_manifest(checkpoint_manifest, experiments_root, repo_dir=repo_dir, eligibility=eligibility,
                                             e2_ep_confirmation=e2_ep_confirmation, seeds=seeds)
    if dp.sha256_file(pretrained_path) != pl.WEIGHTS_SHA256:
        raise RuntimeError("the ImageNet ConvNeXt file is not the pinned one")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    manifest = dp.read_manifest(audit_dir, lesion_root)          # pinned manifest, every image hash, every mask
    ids = {s: [n for n, _ in manifest[s]] for s in dp.SPLITS}
    arms = [a for a in ARMS if any(k[0] == a for k in checkpoints)]
    contrasts = dict(CONTRASTS, **(E2_CONTRASTS if "e2_ep" in arms else {}))
    identity = {"experiment": "DDR ground-truth lesion probe -- EyePACS-adapted arms", "probe_protocol_version": PROTOCOL_VERSION,
                "manifest_sha256": dp.MANIFEST_SHA256, "excluded": list(dp.EXCLUDED), "counts": {s: len(v) for s, v in ids.items()},
                "classes": list(dp.CLASSES), "grid": dp.GRID,
                "target": "cell = 1 if any expert lesion pixel lies in it (full frame, native size), else 0",
                "probe": ep.PROBE, "stopping": dp.STOPPING, "variants": list(dp.VARIANTS), "arms": arms,
                "adapted_encoder_sha256": checkpoint_manifest["adapted_encoder_sha256"],
                "checkpoints": {f"{a}_seed{s}": v["sha256"] for (a, s), v in sorted(checkpoints.items())},
                "contrasts": {k: list(v) for k, v in contrasts.items()}, "primary": PRIMARY, "criterion": CRITERION,
                "bootstrap": {"n": dp.N_BOOT, "seed": dp.BOOT_SEED, "unit": "test image", "stratified": False}}
    config_path = os.path.join(out_dir, "configuration.json")
    if os.path.exists(config_path):
        if _read_json(config_path) != json.loads(json.dumps(identity)):
            raise RuntimeError(f"{out_dir} holds a run with another configuration")
    else:
        with open(config_path, "w") as fh:
            json.dump(identity, fh, indent=1)
    dp.record_invocation(out_dir, dict(dp.environment(repo_dir), probe_protocol_version=PROTOCOL_VERSION,
                                       manifest_sha256=dp.MANIFEST_SHA256, pretrained_sha256=pl.WEIGHTS_SHA256,
                                       checkpoints=identity["checkpoints"], checkpoint_manifest_commit=checkpoint_manifest["git_commit"],
                                       seeds=[int(s) for s in seeds], bootstrap=identity["bootstrap"],
                                       feature_policy=ep.PROBE["feature_policy"]))
    frames, targets = dp.prepare_data(lesion_root, ids, work_dir, log)
    prior = targets["train"].mean(axis=(0, 1, 2)).astype(np.float64).tolist()
    np.savez_compressed(os.path.join(out_dir, "targets.npz"), **{f"{s}_targets": targets[s] for s in dp.SPLITS},
                        **{f"{s}_ids": np.asarray(ids[s]) for s in dp.SPLITS})
    summary_path = os.path.join(out_dir, "summary.json")
    summary = {"prior": prior, "positive_cell_rate": {s: targets[s].mean(axis=(0, 1, 2)).tolist() for s in dp.SPLITS}, "results": {}}
    if os.path.exists(summary_path):
        old = _read_json(summary_path)
        if old.get("prior") == prior:
            summary["results"] = old.get("results", {})
    previous = keras.mixed_precision.global_policy().name
    try:
        keras.mixed_precision.set_global_policy("float32")
        _, reference_arrays = pl.load_reference(pretrained_path)
        for seed in seeds:
            for arm in arms:
                key = f"{arm}_seed{int(seed)}"
                saved = os.path.join(out_dir, f"probe_{key}.npz")
                checkpoint = checkpoints[(arm, int(seed))]
                if summary["results"].get(key, {}).get("weights_sha256") == checkpoint["sha256"] and os.path.exists(saved):
                    log(f"  {key}: already done -- kept")
                    continue
                log(f"  {key}: frozen features  {ep._ram()}")
                result, arrays = dp.probe_encoder(checkpoint["kind"], seed, checkpoint["path"], reference_arrays, frames, ids,
                                                  targets, prior, log)
                result = {"arm": arm, "seed": int(seed), "model_type": checkpoint["model_type"],
                          "weights_sha256": checkpoint["sha256"], **result}
                np.savez_compressed(saved, **arrays)
                summary["results"][key] = result
                behaviour = result["behaviour"]
                log(f"  {key}: test mean cell AUROC five-epoch {result['five_epoch']['test']['scores']['mean']:.4f} | converged "
                    f"{result['converged']['test']['scores']['mean']:.4f} (epoch {behaviour['converged_epoch']} of "
                    f"{behaviour['epochs_run']}, {behaviour['stopped_by']})")
                with open(summary_path + ".tmp", "w") as fh:
                    json.dump(summary, fh, indent=1, default=float)
                os.replace(summary_path + ".tmp", summary_path)
    finally:
        keras.mixed_precision.set_global_policy(previous)
    return summary


# --------------------------------------------------------------------------- the pre-declared comparison

def probed_arms(out_dir, seeds=SEEDS):
    """The arms whose test logits are saved for every seed."""
    return tuple(a for a in ARMS if all(os.path.exists(os.path.join(out_dir, f"probe_{a}_seed{int(s)}.npz")) for s in seeds))


def analyse(out_dir, seeds=SEEDS):
    """ddr_probe.analyse for the EP arms: P-EP and E1-EP (and E2-EP when probed), E1-EP - P-EP per seed and for
    the three-seed mean with the paired bootstrap of §68 (2,000 resamples, seed 20260927 -- not parameters
    here), both probe variants, and the criterion on the five-epoch variant. Writes ddr_probe_ep_result.json."""
    arms = probed_arms(out_dir, seeds)
    if any(a not in arms for a in REQUIRED_ARMS):
        raise RuntimeError(f"the probe of {REQUIRED_ARMS} is not complete in {out_dir}")
    contrasts = dict(CONTRASTS, **(E2_CONTRASTS if "e2_ep" in arms else {}))
    result = dp.analyse(out_dir, seeds, dp.N_BOOT, dp.BOOT_SEED, models=arms, contrasts=contrasts, primary=PRIMARY,
                        result_name=RESULT_NAME, criterion=CRITERION)
    if result["primary_variant"] != "five_epoch" or result["criterion_met"] != result["variants"]["five_epoch"]["criterion_met"]:
        raise RuntimeError("the criterion was not evaluated on the five-epoch variant")
    return result


def analyse_against_baseline(out_dir, baseline_dir, seeds=SEEDS):
    """Descriptive (record §69.2, Q3): P-EP against P and E1-EP against E1, from this run's logits and the saved
    logits of the §70 run (same test images, same resamples). No criterion; it decides nothing."""
    with np.load(os.path.join(out_dir, "targets.npz")) as a, np.load(os.path.join(baseline_dir, "targets.npz")) as b:
        if [str(i) for i in a["test_ids"]] != [str(i) for i in b["test_ids"]] or not np.array_equal(a["test_targets"], b["test_targets"]):
            raise RuntimeError("the two runs do not share the test images and targets")
    result = dp.analyse(out_dir, seeds, dp.N_BOOT, dp.BOOT_SEED, models=("p", "e1", "p_ep", "e1_ep"), contrasts=BASELINE_CONTRASTS,
                        primary="p_ep_minus_p", result_name=BASELINE_RESULT_NAME, sources={"p": baseline_dir, "e1": baseline_dir},
                        criterion="none (descriptive)")
    for variant in result["variants"].values():
        variant.pop("criterion_met", None)
    result.pop("criterion_met", None)
    result["role"] = "descriptive; no criterion; decides nothing"
    with open(os.path.join(out_dir, BASELINE_RESULT_NAME), "w") as fh:
        json.dump(result, fh, indent=1, default=float)
    return result


# --------------------------------------------------------------------------- the E2-EP gate

def record_e2_ep_gate(out_dir, eligibility_path, seeds=SEEDS):
    """Writes the E1-EP - P-EP criterion result (five-epoch variant) where ep_aptos_train.require_e2_ep_eligibility
    reads it. Recording a met criterion makes E2-EP ELIGIBLE; it starts nothing."""
    import ep_aptos_train as epa
    result = _read_json(os.path.join(out_dir, RESULT_NAME)) if os.path.exists(os.path.join(out_dir, RESULT_NAME)) else analyse(out_dir, seeds)
    return epa.write_e2_ep_eligibility(eligibility_path, result)


def e2_ep_gate_status(eligibility_path):
    """{"recorded", "eligible", "reason", ...} without raising (for display)."""
    import ep_aptos_train as epa
    if not eligibility_path or not os.path.exists(eligibility_path):
        return {"recorded": False, "eligible": False, "reason": "no E1-EP vs P-EP criterion result is recorded"}
    record = _read_json(eligibility_path)
    try:
        epa.require_e2_ep_eligibility(eligibility_path)
        eligible, reason = True, "criterion met"
    except RuntimeError as error:
        eligible, reason = False, str(error)
    return {"recorded": True, "eligible": eligible, "reason": reason, "per_seed": record.get("per_seed"), "mean": record.get("mean"),
            "ci": record.get("ci"), "positive_seeds": record.get("positive_seeds")}
