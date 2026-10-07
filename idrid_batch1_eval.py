"""IDRiD batch 1 -- one-time cross-dataset evaluation of the frozen E1 and E2 graders (research record §65).

Everything that defines the run is in idrid_batch1_protocol.json (six checkpoint hashes, the three stored P
tables, the comparisons, the disclosures) and, unchanged, in its parent idrid_grading_protocol.json (the 103
images and labels by hash, the three images excluded from the primary set, Stage-2 preprocessing, the 512 RGB
frame, T4 / mixed_float16 / batch 8, the parity tolerances, the bootstrap).

What is reused, not re-implemented: idrid_grading_eval (dataset reader with its hash checks, environment
record, parity-subset rule, file hashing), stage4_v2_data.stage2_rgb and stage4_v2_aptos_cache.recompute_rgb_512
(the image path), e1_model (the model), arch1_train.metrics_from_logits (the per-image table schema and the CORN
decode), arch1_posthoc (metrics, grade-stratified bootstrap, paired differences). The earlier evaluator
(idrid_grading_eval.run_idrid), its lock and its output directory are not touched.

P is NOT run: its per-image predictions are read from the tables the earlier one-time run stored, after their
hashes are checked. E1 and E2 take RGB only, so Stage 3 and Stage 4 are not run and no cache of theirs is read.

Order, enforced: `gates` (hashes, model construction and precision, APTOS parity) must be an official PASS from
this runtime before `run` reads any IDRiD image; `run` needs the confirmation token and writes a lock before any
prediction exists; a second run is refused except for a technical failure with the reason recorded.
"""
import datetime
import json
import os

import numpy as np

import arch1_posthoc as ph
import idrid_grading_eval as ig
import pathology_grader_fusion as pf

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "idrid_batch1_protocol.json")
SEEDS = (42, 123, 2026)
MODELS = ("p", "e1", "e2")
RUN_MODELS = ("e1", "e2")                         # the only models this module ever runs
CONTRASTS = {"e1_minus_p": ("e1", "p"), "e2_minus_p": ("e2", "p"), "e2_minus_e1": ("e2", "e1")}
PRIMARY_CONTRASTS = ("e1_minus_p", "e2_minus_p")
ProtocolError = ig.ProtocolError


# --------------------------------------------------------------------------- protocol and pinned files

def load_protocol(path=PROTOCOL_PATH, parent_path=None):
    """(batch protocol, its sha256, parent protocol). The parent must be the pinned locked protocol."""
    with open(path, encoding="utf-8") as fh:
        protocol = json.load(fh)
    parent_path = parent_path or os.path.join(HERE, protocol["parent_protocol"]["file"])
    parent, parent_sha = ig.load_protocol(parent_path)
    if parent_sha != protocol["parent_protocol"]["sha256"]:
        raise ProtocolError("the parent IDRiD protocol is not the pinned locked file")
    if tuple(protocol["seeds"]) != SEEDS or tuple(parent["seeds"]) != SEEDS:
        raise ProtocolError("protocol seeds are not 42, 123, 2026")
    expected = {f"{m}_seed{s}" for m in RUN_MODELS for s in SEEDS}
    if set(protocol["checkpoints"]) != expected:
        raise ProtocolError(f"the protocol must pin exactly {sorted(expected)}")
    if set(protocol["comparisons"]["primary"]) != set(PRIMARY_CONTRASTS):
        raise ProtocolError("the primary comparisons must be E1 - P and E2 - P")
    return protocol, ig.sha256_file(path), parent


def verify_files(protocol, drive_root):
    """Every pinned file exists with the pinned SHA-256: the six checkpoints (and their runs' BEST pointers
    and recorded hashes still name them) and the three stored P tables. Returns {'checkpoints', 'p_tables',
    'lesion_prior'}. Raises on the first mismatch -- nothing is substituted."""
    root = lambda rel: os.path.join(drive_root, *rel.split("/"))
    checkpoints, priors = {}, {}
    for name, entry in protocol["checkpoints"].items():
        path = root(entry["path"])
        if not os.path.exists(path):
            raise ProtocolError(f"{name}: {path} not found")
        if ig.sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"{name}: {path} is not the pinned checkpoint (SHA-256 mismatch)")
        run_dir = root(entry["run_dir"])
        with open(os.path.join(run_dir, "checkpoints", "best.json")) as fh:
            pointer = json.load(fh)
        if pointer["active"] != entry["best_slot"] or int(pointer["epoch"]) != entry["best_epoch_index"]:
            raise ProtocolError(f"{name}: the run's BEST pointer no longer names the pinned checkpoint")
        with open(os.path.join(run_dir, "result.json")) as fh:
            result = json.load(fh)
        if result["best"]["checkpoint"]["weights_sha256"] != entry["sha256"]:
            raise ProtocolError(f"{name}: the pinned checkpoint is not the one the run evaluated as BEST")
        if result["best"]["checkpoint"]["monitor"] != "val_QWK":
            raise ProtocolError(f"{name}: BEST was not selected by APTOS validation QWK")
        checkpoints[name] = path
        priors[name] = [float(p) for p in result["config"]["lesion_prior"]]
    p_tables = {}
    for seed in SEEDS:
        entry = protocol["stored_p_tables"][f"p_seed{seed}"]
        path = root(entry["path"])
        if not os.path.exists(path) or ig.sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"stored P table for seed {seed} is missing or is not the pinned file")
        p_tables[seed] = path
    return {"checkpoints": checkpoints, "p_tables": p_tables, "lesion_prior": priors}


# --------------------------------------------------------------------------- models and inputs

def load_models(files, settings):
    """{(model, seed): keras model} for E1 and E2, built under the protocol's dtype policy and loaded from the
    pinned BEST weights. The policy is set AFTER clear_session (Keras 3 resets it there) and every model is
    checked to really be in that precision."""
    import keras

    import e1_model as em
    from training import checkpointing as ckpt
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy(settings["keras_policy"])
    expected = "float16" if settings["keras_policy"] == "mixed_float16" else "float32"
    models = {}
    for model in RUN_MODELS:
        for seed in SEEDS:
            name = f"{model}_seed{seed}"
            net = em.build_e1_model(seed, files["lesion_prior"][name])
            ckpt.load_model_weights_only(net, files["checkpoints"][name])
            if str(net.outputs[0].dtype) != expected or keras.mixed_precision.global_policy().name != settings["keras_policy"]:
                raise ProtocolError(f"{name}: the model was not built under {settings['keras_policy']}")
            if tuple(net.output_names) != (em.GRADING_OUTPUT, em.LESION_OUTPUT):
                raise ProtocolError(f"{name}: unexpected outputs {net.output_names}")
            models[(model, seed)] = net
    return models


def rgb_from_raw(raw_path):
    """The locked image path of one raw file: Stage 2 (DR profile, once, native size) and the canonical 512
    frame. Returns (rgb (512, 512, 3) float32 in [0, 1], native uint8 image)."""
    import stage4_v2_aptos_cache as ac
    import stage4_v2_data as sd
    native = sd.stage2_rgb(raw_path)
    return ac.recompute_rgb_512(native), native


def predict_logits(models, frames, batch_size):
    """{(model, seed): (N, 4) float64 CORN logits} for a list of RGB frames. Only output 0 is read."""
    out = {key: [] for key in models}
    for start in range(0, len(frames), batch_size):
        rgb = np.stack(frames[start:start + batch_size]).astype(np.float32)
        for key, net in models.items():
            out[key].append(np.asarray(net.predict_on_batch({"rgb": rgb})[0], np.float64))
    return {key: np.concatenate(v, 0) for key, v in out.items()}


def table_from_logits(ids, grades, logits):
    """(frame in the project's per-image schema, arrays for the metrics) -- the same decode as every grader."""
    import pandas as pd

    import arch1_train as at
    frame = pd.DataFrame(at.metrics_from_logits(list(ids), np.asarray(grades), logits)[1])
    return frame, ph.table_arrays(frame)


# --------------------------------------------------------------------------- gates (no IDRiD image is read)

def gates(drive_root, aptos_raw_dir, bundle, out_dir, *, aptos_processed_dir=None, protocol_path=PROTOCOL_PATH,
          settings=None, official=True, models=None, log=print):
    """The pre-inference gates. `bundle`: the verified v2 bundle with the cached APTOS RGB frames. Writes
    gates.json and returns it; `official=False` marks a pre-check that can never unlock IDRiD. `models` is for
    tests only."""
    import pandas as pd

    import pathology_grader_severity_fusion as sf
    protocol, protocol_sha, parent = load_protocol(protocol_path)
    settings = dict(settings or parent["inference"])
    if official and settings != parent["inference"]:
        raise ProtocolError("the official gates run with the locked inference settings only")
    tol = parent["parity"]["tolerances"]
    failures, checks = [], {}
    ig.prepare_gpu()
    files = verify_files(protocol, drive_root)                       # gates 1, 2, 9 (raises on any mismatch)
    checks["checkpoint_sha256"] = {k: protocol["checkpoints"][k]["sha256"] for k in files["checkpoints"]}
    checks["p_table_sha256"] = {str(s): protocol["stored_p_tables"][f"p_seed{s}"]["sha256"] for s in SEEDS}
    checks["parent_protocol_sha256"] = protocol["parent_protocol"]["sha256"]
    models = models if models is not None else load_models(files, settings)      # gates 6, 7
    checks["grading_output_dtype"] = {f"{m}_seed{s}": str(net.outputs[0].dtype) for (m, s), net in models.items()}
    val_ids, val_grades = sf.authoritative_validation_ids()
    if list(bundle.val_ids) != list(val_ids):
        failures.append("the bundle's validation images are not the authoritative 730 in order")
    grade_of = dict(zip(val_ids, val_grades))
    stored = {}
    root = lambda rel: os.path.join(drive_root, *rel.split("/"))
    for (model, seed) in models:
        path = root(protocol["stored_validation_tables"][model].format(seed=seed))
        frame = pd.read_csv(path, dtype={"image_id": str})
        table = ph.table_arrays(frame)
        if list(table["ids"]) != list(val_ids):
            failures.append(f"{model}-{seed}: the stored validation table is not in the authoritative order")
        stored[(model, seed)] = {"table": table, "logits": frame[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64)}

    def compare(logits, rows, key):
        frame, table = table_from_logits([val_ids[r] for r in rows], [val_grades[r] for r in rows], logits)
        return {"logit_max_abs": float(np.abs(logits - stored[key]["logits"][rows]).max()),
                "probability_max_abs": float(np.abs(table["p_gt"] - stored[key]["table"]["p_gt"][rows]).max()),
                "grades_equal": bool(np.array_equal(table["pred"], stored[key]["table"]["pred"][rows])), "images": len(rows)}

    def judge(name, result):
        if result["logit_max_abs"] > tol["logit_max_abs"]:
            failures.append(f"{name}: logits differ by {result['logit_max_abs']:.3g} (> {tol['logit_max_abs']})")
        if result["probability_max_abs"] > tol["probability_max_abs"]:
            failures.append(f"{name}: cumulative probabilities differ by {result['probability_max_abs']:.3g}")
        if not result["grades_equal"]:
            failures.append(f"{name}: decoded grades differ")

    # (a) ten raw APTOS images through the locked image path
    subset = ig.parity_subset(val_ids, val_grades, parent["parity"]["images_per_grade"])
    index = {i: n for n, i in enumerate(val_ids)}
    raw_rows, raw_frames, raw_checks = [index[i] for i in subset], [], []
    for image_id in subset:
        rgb, native = rgb_from_raw(os.path.join(aptos_raw_dir, f"{image_id}.png"))
        row = {"image_id": image_id, "grade": int(grade_of[image_id]),
               "rgb_max_abs": float(np.abs(rgb - bundle.load_sample(image_id)["rgb"]).max())}
        processed = os.path.join(aptos_processed_dir, f"{image_id}.png") if aptos_processed_dir else None
        if processed and os.path.exists(processed):
            import stage4_v2_data as sd
            stored_native = sd.read_rgb(processed)
            row["stage2_exact"] = bool(stored_native.shape == native.shape and np.array_equal(stored_native, native))
            if not row["stage2_exact"]:
                failures.append(f"{image_id}: Stage 2 recomputed from the raw image differs from the stored Stage-2 file")
        elif official:
            failures.append(f"{image_id}: no stored Stage-2 file to compare with")
        if row["rgb_max_abs"] > tol["rgb_max_abs"]:
            failures.append(f"{image_id}: RGB frame differs from the cache by {row['rgb_max_abs']:.3g}")
        raw_checks.append(row)
        raw_frames.append(rgb)
    raw_logits = predict_logits(models, raw_frames, settings["batch_size"])
    checks["raw_subset"] = {"images": subset, "grades": [int(grade_of[i]) for i in subset], "image_checks": raw_checks,
                            "predictions": {}}
    for key, logits in raw_logits.items():
        result = compare(logits, raw_rows, key)
        checks["raw_subset"]["predictions"][f"{key[0]}_seed{key[1]}"] = result
        judge(f"{key[0]}-{key[1]} (10 raw images)", result)
    log(f"  raw subset: RGB max |diff| {max(r['rgb_max_abs'] for r in raw_checks):.3g}; Stage 2 exact "
        f"{[r.get('stage2_exact') for r in raw_checks].count(True)}/{len(raw_checks)}")
    # (b) all 730 validation images from the cached RGB frames
    cached = [bundle.load_sample(i)["rgb"] for i in val_ids]
    full_logits = predict_logits(models, cached, settings["batch_size"])
    checks["full_validation"] = {}
    for key, logits in full_logits.items():
        result = compare(logits, list(range(len(val_ids))), key)
        result["qwk"] = ph.metrics(table_from_logits(val_ids, val_grades, logits)[1])["qwk"]
        result["stored_qwk"] = ph.metrics(stored[key]["table"])["qwk"]
        checks["full_validation"][f"{key[0]}_seed{key[1]}"] = result
        judge(f"{key[0]}-{key[1]} (730 validation images)", result)
        log(f"  {key[0]}-{key[1]}: logits {result['logit_max_abs']:.3g} | probabilities {result['probability_max_abs']:.3g} | "
            f"grades equal {result['grades_equal']} | QWK {result['qwk']:.4f} (stored {result['stored_qwk']:.4f})")
    env = ig.environment(settings)
    if official and (env["gpu"] is None or env["keras_policy"] != parent["inference"]["keras_policy"]):
        failures.append("the official gates run on the GPU runtime under the locked precision policy")
    result = {"gate": "idrid batch 1 pre-inference gates", "kind": "official gate" if official else "pre-check (cannot unlock IDRiD)",
              "official": bool(official), "PASS": not failures, "failures": failures, "checks": checks, "tolerances": tol,
              "settings": settings, "environment": env, "protocol_sha256": protocol_sha, "git_commit": ig.git_commit(),
              "idrid_images_read": 0, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "gates.json"), "w") as fh:
            json.dump(ph._jsonable(result), fh, indent=1)
    log(f"IDRiD BATCH 1 GATES {'PASS' if result['PASS'] else 'FAIL'} ({result['kind']})"
        + ("" if result["PASS"] else " | " + "; ".join(failures)))
    return result


# --------------------------------------------------------------------------- metrics

def subset_table(table, mask):
    return {k: v[mask] for k, v in table.items()}


def summarise(tables, mask, n_boot=ph.N_BOOT):
    """Per-seed metrics for P, E1 and E2 on the masked images, mean +/- SD over the three seeds (no ensemble,
    no seed selection), grade-stratified bootstrap intervals for the seed mean, and the paired differences
    (same images, same seed) E1 - P, E2 - P and -- descriptive -- E2 - E1 with intervals for their seed means.
    The quantities and the bootstrap are those of idrid_grading_eval.summarise."""
    per_seed = {}
    for seed in SEEDS:
        per_seed[seed] = {}
        for model in MODELS:
            m = ph.metrics(subset_table(tables[seed][model], mask))
            per_seed[seed][model] = {**pf.headline(m), "recall_per_grade": m["recall_per_grade"],
                                     "confusion_matrix": m["confusion_matrix"]}
    stat = lambda values: {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1)), "per_seed": [float(v) for v in values]}
    aggregate = {}
    for model in MODELS:
        aggregate[model] = {k: stat([per_seed[s][model][k] for s in SEEDS]) for k in pf.HEADLINE_KEYS}
        for g in range(5):
            aggregate[model][f"recall_grade{g}"] = stat([per_seed[s][model]["recall_per_grade"][g] for s in SEEDS])
    grades = subset_table(tables[SEEDS[0]]["p"], mask)["grade"]
    indices = ph.bootstrap_indices(grades, n_boot)
    intervals = {}
    for model in MODELS:
        draws = {k: [] for k in ph.BOOT_KEYS}
        for idx in indices:
            vals = [ph.bootstrap_metrics(subset_table(tables[s][model], mask), idx) for s in SEEDS]
            for k in ph.BOOT_KEYS:
                draws[k].append(np.mean([v[k] for v in vals]))
        intervals[model] = {k: [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))] for k, d in draws.items()}
    differences = {}
    for name, (a, b) in CONTRASTS.items():
        point, draws = {k: [] for k in ph.METRICS}, {k: [] for k in ph.BOOT_KEYS}
        recall = {g: [] for g in range(5)}
        for s in SEEDS:
            d = ph.paired_delta(subset_table(tables[s][a], mask), subset_table(tables[s][b], mask), indices=indices)
            for k in ph.METRICS:
                point[k].append(d[k]["delta"])
            for k in ph.BOOT_KEYS:
                draws[k].append(d[k]["_draws"])
            for g in range(5):
                recall[g].append(per_seed[s][a]["recall_per_grade"][g] - per_seed[s][b]["recall_per_grade"][g])
        differences[name] = {k: stat(v) for k, v in point.items()}
        for k in ph.BOOT_KEYS:
            lo, hi = np.nanpercentile(np.mean(np.stack(draws[k], 0), 0), [2.5, 97.5])
            differences[name][k].update(ci_low=float(lo), ci_high=float(hi))
        for g in range(5):
            differences[name][f"recall_grade{g}"] = stat(recall[g])
        for cut in ("auroc_ge1", "auroc_ge2", "auroc_ge3", "auroc_ge4", "grade4_recall"):
            differences[name][cut] = stat([per_seed[s][a][cut] - per_seed[s][b][cut] for s in SEEDS])
        differences[name]["role"] = "primary" if name in PRIMARY_CONTRASTS else "secondary, descriptive"
        differences[name]["decoded_grades_differ"] = [int((subset_table(tables[s][a], mask)["pred"]
                                                           != subset_table(tables[s][b], mask)["pred"]).sum()) for s in SEEDS]
    return {"n": int(mask.sum()), "grade_counts": np.bincount(grades, minlength=5).tolist(), "per_seed": per_seed,
            "aggregate": aggregate, "seed_mean_interval_95": intervals, "paired_differences": differences,
            "n_boot": int(n_boot), "boot_seed": ph.BOOT_SEED, "bootstrap": "grade-stratified over the test images"}


def _f(v, signed=False):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.4f}" if signed else f"{v:.4f}")


def report_markdown(title, note, summary, protocol):
    labels = {"p": "P (stored predictions of the earlier run; not run again)", "e1": "E1 (aligned lesion supervision)",
              "e2": "E2 (shuffled lesion targets; control)"}
    rows = (("qwk", "QWK"), ("auroc_ge1", "AUROC ≥1"), ("auroc_ge2", "AUROC ≥2"), ("auroc_ge3", "AUROC ≥3"),
            ("auroc_ge4", "AUROC ≥4"), ("recall_grade0", "recall, grade 0"), ("recall_grade1", "recall, grade 1"),
            ("recall_grade2", "recall, grade 2"), ("recall_grade3", "recall, grade 3"), ("recall_grade4", "recall, grade 4"),
            ("false_urgent_rate", "false-urgent rate"))
    lines = [f"# {title}", "", note, "",
             f"Images: {summary['n']} (grades 0-4: {summary['grade_counts']}). Seeds 42, 123, 2026; mean ± SD over seeds; no "
             "ensemble, no seed selection. Decoding: grade = number of cumulative CORN probabilities above 0.5 (unchanged).", "",
             "## Disclosures", ""] + [f"- {d}" for d in protocol["disclosures"]] + [""]
    for model in MODELS:
        lines += [f"## {labels[model]}", "", "| | seed 42 | seed 123 | seed 2026 | mean | SD |", "|---|---|---|---|---|---|"]
        for key, name in rows:
            st = summary["aggregate"][model][key]
            lines.append(f"| {name} | " + " | ".join(_f(v) for v in st["per_seed"]) + f" | {_f(st['mean'])} | {_f(st['sd'])} |")
        ci = summary["seed_mean_interval_95"][model]
        lines += ["", f"95 % interval of the seed mean (bootstrap over images): QWK {_f(ci['qwk'][0])} to {_f(ci['qwk'][1])}; "
                      f"mean AUROC over the four cuts {_f(ci['mean_cut_auroc'][0])} to {_f(ci['mean_cut_auroc'][1])}.", ""]
        for seed in SEEDS:
            lines.append(f"Confusion matrix, seed {seed} (rows = true grade): {summary['per_seed'][seed][model]['confusion_matrix']}")
        lines.append("")
    names = {"e1_minus_p": "E1 − P (primary)", "e2_minus_p": "E2 − P (primary)", "e2_minus_e1": "E2 − E1 (secondary, descriptive)"}
    lines += ["## Paired differences, same images and same seed", "",
              "| | QWK (95 % interval) | per seed | mean AUROC, four cuts (95 % interval) | AUROC ≥3 | AUROC ≥4 | grade-3 recall | "
              "grade-4 recall | false-urgent |", "|---|---|---|---|---|---|---|---|---|"]
    for name, label in names.items():
        d = summary["paired_differences"][name]
        lines.append(f"| {label} | {_f(d['qwk']['mean'], True)} ({_f(d['qwk']['ci_low'], True)} to {_f(d['qwk']['ci_high'], True)}) | "
                     + " / ".join(_f(v, True) for v in d["qwk"]["per_seed"])
                     + f" | {_f(d['mean_cut_auroc']['mean'], True)} ({_f(d['mean_cut_auroc']['ci_low'], True)} to "
                       f"{_f(d['mean_cut_auroc']['ci_high'], True)}) | {_f(d['auroc_ge3']['mean'], True)} | {_f(d['auroc_ge4']['mean'], True)} | "
                       f"{_f(d['grade3_recall']['mean'], True)} | {_f(d['grade4_recall']['mean'], True)} | "
                       f"{_f(d['false_urgent_rate']['mean'], True)} |")
    lines += ["", "No equivalence or non-inferiority margin was pre-specified; none is tested.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the one-time run

def lock_path(out_root, protocol_sha):
    return os.path.join(out_root, f"idrid_batch1_lock_{protocol_sha[:12]}.json")


def require_gates(gate, protocol_sha, parent, settings):
    """The gates must be an official PASS from this environment, this protocol and this code."""
    if not gate.get("official") or not gate.get("PASS"):
        raise ProtocolError("the pre-inference gates have not passed (an official PASS is required)")
    if gate["protocol_sha256"] != protocol_sha or gate["settings"] != parent["inference"]:
        raise ProtocolError("the gates were run with another protocol or other settings")
    if gate["git_commit"] != ig.git_commit():
        raise ProtocolError("the gates were run on another commit")
    env = ig.environment(settings)
    for key in ("gpu", "tensorflow", "keras", "keras_policy"):
        if gate["environment"].get(key) != env[key]:
            raise ProtocolError(f"the gates were run in another environment ({key})")
    return True


def read_p_tables(files, data):
    """The stored P tables as metric arrays, checked to be the same 103 images, in the same order, with the
    same labels as the verified test set."""
    tables = {}
    for seed, path in files["p_tables"].items():
        table = ph._read_table(path)
        if list(table["ids"]) != list(data["ids"]) or not np.array_equal(table["grade"], data["grades"]):
            raise ProtocolError(f"stored P table for seed {seed}: not the verified test images in order with their labels")
        tables[seed] = table
    return tables


def run(drive_root, idrid_raw_dir, out_root, gate, *, confirm, technical_rerun_reason=None, protocol_path=PROTOCOL_PATH,
        command=None, models=None, log=print):
    """THE ONE-TIME EVALUATION OF E1 AND E2. Refuses without the token, without official gates passed in this
    environment, or when the lock exists (a rerun needs a written technical reason and is recorded).
    `models` is for tests only."""
    import shutil
    protocol, protocol_sha, parent = load_protocol(protocol_path)
    settings = dict(parent["inference"])
    if confirm != protocol["confirmation_token"]:
        raise ProtocolError("IDRiD batch 1 runs once: pass confirm=<the protocol's confirmation token> to run it")
    require_gates(gate, protocol_sha, parent, settings)
    ig.prepare_gpu()
    os.makedirs(out_root, exist_ok=True)
    lock = lock_path(out_root, protocol_sha)
    previous = None
    if os.path.exists(lock):
        with open(lock) as fh:
            previous = json.load(fh)
        if not (isinstance(technical_rerun_reason, str) and technical_rerun_reason.strip()):
            raise ProtocolError(f"IDRiD batch 1 has already been run ({previous['timestamp_utc']}); a rerun is allowed only "
                                "for a technical failure, with technical_rerun_reason stated")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_root, f"idrid_batch1_e1e2_{protocol_sha[:12]}_{stamp}")
    if os.path.abspath(out_dir).startswith(os.path.abspath(os.path.join(drive_root, *protocol["stored_p_tables"]["source_run"].split("/")))):
        raise ProtocolError("the output directory must not be inside the earlier IDRiD run")
    files = verify_files(protocol, drive_root)
    data = ig.read_idrid(idrid_raw_dir, parent)                      # verifies image-set and label-file hashes
    p_tables = read_p_tables(files, data)
    os.makedirs(out_dir)
    record = {"timestamp_utc": stamp, "protocol_sha256": protocol_sha, "git_commit": ig.git_commit(), "out_dir": out_dir,
              "technical_rerun_reason": technical_rerun_reason, "previous_run": previous}
    with open(lock, "w") as fh:                                      # written BEFORE any prediction exists
        json.dump(record, fh, indent=1)
    with open(os.path.join(out_dir, "lock_record.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    models = models if models is not None else load_models(files, settings)
    frames, native_shapes = [], {}
    for n, (image_id, path) in enumerate(zip(data["ids"], data["paths"]), start=1):
        rgb, native = rgb_from_raw(path)
        frames.append(rgb)
        native_shapes[image_id] = list(native.shape)
        if n % 20 == 0 or n == len(data["paths"]):
            log(f"  inputs {n}/{len(data['paths'])}")
    logits = predict_logits(models, frames, settings["batch_size"])
    tables = {seed: {"p": p_tables[seed]} for seed in SEEDS}
    for (model, seed), z in logits.items():
        frame, tables[seed][model] = table_from_logits(data["ids"], data["grades"], z)
        frame["in_primary_set"] = data["primary"]
        frame.to_csv(os.path.join(out_dir, f"per_image_{model}_seed{seed}.csv"), index=False)
    for seed, path in files["p_tables"].items():                     # copies of the locked P tables, byte for byte
        dst = os.path.join(out_dir, f"per_image_p_seed{seed}_stored.csv")
        shutil.copyfile(path, dst)
        if ig.sha256_file(dst) != protocol["stored_p_tables"][f"p_seed{seed}"]["sha256"]:
            raise ProtocolError("the copied P table differs from the pinned file")
    with open(os.path.join(out_dir, "image_manifest.json"), "w") as fh:
        json.dump({"ids": data["ids"], "sha256": data["hashes"], "grades": [int(g) for g in data["grades"]],
                   "in_primary_set": [bool(b) for b in data["primary"]], "native_shapes": native_shapes,
                   "images_manifest_sha256": parent["dataset"]["images_manifest_sha256"],
                   "label_file_sha256": parent["dataset"]["label_file_sha256"]}, fh, indent=1)
    common = {"protocol": protocol, "protocol_sha256": protocol_sha, "parent_protocol": parent,
              "parent_protocol_sha256": protocol["parent_protocol"]["sha256"], "environment": ig.environment(settings),
              "git_commit": ig.git_commit(), "command": command, "timestamp_utc": stamp, "gates": gate,
              "checkpoints_verified": {k: protocol["checkpoints"][k]["sha256"] for k in files["checkpoints"]},
              "p_tables_reused": {str(s): {"path": p, "sha256": protocol["stored_p_tables"][f"p_seed{s}"]["sha256"]}
                                  for s, p in files["p_tables"].items()},
              "models_run": [f"{m}_seed{s}" for m in RUN_MODELS for s in SEEDS], "p_rerun": False,
              "excluded_from_primary": parent["dataset"]["primary_exclusions"], "stage3_run": False, "stage4_run": False}
    with open(os.path.join(out_dir, "run_configuration.json"), "w") as fh:
        json.dump(ph._jsonable(common), fh, indent=1)
    primary = summarise(tables, data["primary"])                     # the primary result is computed and written first
    with open(os.path.join(out_dir, "primary_results_100.json"), "w") as fh:
        json.dump(ph._jsonable(primary), fh, indent=1)
    with open(os.path.join(out_dir, "primary_report_100.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown("IDRiD batch 1 (E1, E2) -- PRIMARY RESULT (100 images)",
                                 "The three test images that are copies of Stage-4 training images (IDRiD_088, IDRiD_089, "
                                 "IDRiD_091) are excluded. Evaluation only; run once.", primary, protocol))
    sensitivity = summarise(tables, np.ones(len(data["ids"]), bool))
    with open(os.path.join(out_dir, "sensitivity_results_103.json"), "w") as fh:
        json.dump(ph._jsonable(sensitivity), fh, indent=1)
    with open(os.path.join(out_dir, "sensitivity_report_103.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown("IDRiD batch 1 (E1, E2) -- SENSITIVITY ANALYSIS ONLY (103 images)",
                                 "Includes three images that are copies of Stage-4 training images. This is NOT the primary "
                                 "result and never replaces it.", sensitivity, protocol))
    log(f"IDRiD batch 1 written to {out_dir}")
    return {"out_dir": out_dir, "primary": primary, "sensitivity": sensitivity, "tables": tables}
