"""External re-test of the frozen joint graders J (direct frames) and J-crop (cropped frames) on DDR and IDRiD --
research record §96. PREPARED, NOT RUN: nothing here is executed until both joint arms are complete, the
protocol file is written from their recorded checkpoints, and the user authorises each run.

This is a documented RE-USE of both test sets (DDR: second grading use, after §86; IDRiD: fourth, after §61, §67,
§79). All settings are frozen on development data before either set is read again; nothing is selected or tuned
on them, and J is the pre-declared primary model whatever J-crop shows (§94).

Order, enforced:
  1. `make_protocol`  -- after the six joint runs are complete: pins their BEST checkpoints (from each run's
                         result.json) and the stored P / P-EP per-image tables of the earlier runs by SHA-256.
  2. `parity`         -- every joint model must reproduce its own stored APTOS validation table from the RAW
                         images through the same frame code used for the external images. No external image read.
  3. `run_ddr` / `run_idrid` -- once each, with the dataset's confirmation token; a lock is written before any
                         prediction exists.

Reused, not re-implemented: the DDR population, image, screen and official-directory checks and the statistics
(ddr_grading_eval), the IDRiD reader (idrid_grading_eval.read_idrid), the two frames of an image in one Stage-2
pass (geometry_diagnostic.two_frames), the P graph and weight loader (idrid_batch2_eval.load_models), the
per-image table and CORN decode, the metric block and the paired grade-stratified bootstrap.
"""
import datetime
import json
import os

import numpy as np

import arch1_posthoc as ph
import ddr_grading_eval as ev
import idrid_batch1_eval as b1
import idrid_batch2_eval as b2
import idrid_grading_eval as ig
import joint_train as jt

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "final_external_protocol_v1.json")
SEEDS = jt.SEEDS
ARMS = {"j": "direct", "jc": "crop"}                       # model name -> joint arm
REFERENCES = ("p", "p_ep")                                 # reused from the earlier runs' stored tables, not run again
MODELS = REFERENCES + tuple(ARMS)
CONTRASTS = {"j_minus_p": ("j", "p"), "j_minus_p_ep": ("j", "p_ep"), "jc_minus_j": ("jc", "j"), "jc_minus_p": ("jc", "p")}
PRIMARY = "j_minus_p"
TOKENS = {"ddr": "RETEST_DDR_WITH_THE_JOINT_MODELS_ONCE", "idrid": "RETEST_IDRID_WITH_THE_JOINT_MODELS_ONCE"}
OUT = {"ddr": "experiments/DDRGrading", "idrid": "experiments/IDRiDGrading"}
PREFIX = "final_external_v1_"
CHUNK = 64
#: where the earlier one-time runs stored the reference predictions (same images, same order)
STORED = {"ddr": {"p": "experiments/DDRGrading/ddr_grading_v1_38eb810f05d4_2026-10-11_12-50-12/per_image_p_seed{seed}.csv",
                  "p_ep": "experiments/DDRGrading/ddr_grading_v1_38eb810f05d4_2026-10-11_12-50-12/per_image_p_ep_seed{seed}.csv"},
          "idrid": {"p": "experiments/IDRiDGrading/idrid_grading_f93c1237eb81_2026-10-05_13-28-16/per_image_p_seed{seed}.csv",
                    "p_ep": "experiments/IDRiDGrading/idrid_batch2_ep_5ea937c4edb8_2026-10-11_07-24-10/per_image_p_ep_seed{seed}.csv"}}
ProtocolError = ig.ProtocolError


def _json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _root(drive_root, rel):
    return os.path.join(drive_root, *rel.split("/"))


# --------------------------------------------------------------------------- protocol (written once, after training)

def make_protocol(drive_root, path=PROTOCOL_PATH):
    """Writes the protocol from what the six joint runs recorded. Refuses unless every run is complete, and
    refuses to overwrite an existing protocol file."""
    if os.path.exists(path):
        raise ProtocolError(f"{path} already exists; a protocol is written once")
    experiments = _root(drive_root, "experiments")
    checkpoints = {}
    for model, arm in ARMS.items():
        for seed in SEEDS:
            run_dir = jt.run_dir_for(experiments, seed, arm)
            if jt.state(run_dir) != "complete":
                raise ProtocolError(f"{arm} arm, seed {seed}: the run is not complete")
            best = _json(os.path.join(run_dir, "result.json"))["best"]["checkpoint"]
            rel = f"experiments/{jt.EXPERIMENT}/{jt.ARMS[arm]['prefix']}_seed{seed}"
            weights = f"{rel}/checkpoints/{best['slot']}/model.weights.h5"
            if ig.sha256_file(_root(drive_root, weights)) != best["weights_sha256"]:
                raise ProtocolError(f"{arm} arm, seed {seed}: BEST is no longer the evaluated checkpoint")
            checkpoints[f"{model}_seed{seed}"] = {"arm": arm, "seed": seed, "path": weights, "sha256": best["weights_sha256"], "run_dir": rel,
                                                 "best_epoch_index": best["epoch_index"], "validation_table": f"{rel}/metrics/per_sample_best_aptos_val.csv"}
    if not _json(os.path.join(jt.run_dir_for(experiments, SEEDS[0], "direct"), "result.json"))["continuation"]["continue"]:
        raise ProtocolError("seed 42 of the direct arm did not meet its continuation rule")
    stored = {name: {f"{m}_seed{s}": {"path": STORED[name][m].format(seed=s), "sha256": ig.sha256_file(_root(drive_root, STORED[name][m].format(seed=s)))}
                     for m in REFERENCES for s in SEEDS} for name in STORED}
    ddr_protocol, ddr_sha, parent = ev.load_protocol()
    protocol = {
        "name": "External re-test of the frozen joint graders J and J-crop on DDR and IDRiD", "version": 1,
        "prepared": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), "research_record_section": "§96",
        "parent_protocols": {"ddr_grading_protocol_v1.json": ddr_sha, "idrid_grading_protocol.json": ddr_protocol["parent_protocol"]["sha256"]},
        "seeds": list(SEEDS), "checkpoints": checkpoints, "stored_reference_tables": stored,
        "models": {"p": "ImageNet ConvNeXt-Tiny + CORN, APTOS only; stored predictions reused, not run",
                   "p_ep": "EyePACS-adapted encoder, APTOS fine-tuned; stored predictions reused, not run",
                   "j": "joint APTOS + EyePACS, direct-resize frames (record 93) -- the pre-declared primary model",
                   "jc": "joint APTOS + EyePACS, fundus-cropped frames (record 94) -- ablation"},
        "pipeline": {"direct": "Stage 2 once, full-frame resize to 512", "crop": "Stage 2 once, cut to the fundus box (rule of record 89), resize to 512",
                     "inference": "the parent protocol's settings (T4, mixed_float16, batch 8); CORN decode at 0.5; no TTA, no threshold fitting, no recalibration"},
        "populations": {"ddr": "the pinned lists of ddr_grading_protocol_v1.json: primary 3,540; broader 3,725 as sensitivity",
                        "idrid": "the 103 images of idrid_grading_protocol.json: primary 100; all 103 as sensitivity"},
        "comparisons": {"primary": [PRIMARY], "secondary_descriptive": [c for c in CONTRASTS if c != PRIMARY]},
        "analysis": {"primary_estimate": "mean over the three seed-labelled pairs of QWK(J) - QWK(P), on each dataset's primary population",
                     "bootstrap": {"method": "paired, grade-stratified over images", "n_boot": ph.N_BOOT, "seed": ph.BOOT_SEED},
                     "reading": "supported if the 95 % interval lies wholly above zero; reversed if wholly below; otherwise no difference shown. "
                                "Seed-level inconsistency is flagged when any of the three differences is not positive",
                     "also_reported": "per-seed and mean QWK, per-cut AUROC, per-grade recall, referable-DR sensitivity and specificity, ECE, seed ensemble; "
                                      "on DDR the primary contrast inside each file-name family"},
        "disclosures": [
            "This is a re-use of both test sets: DDR was used once for grading (record 86), IDRiD three times (records 61, 67, 79).",
            "The earlier results on both sets were known when joint training was chosen; the audits that led to it drew on DDR's failure pattern.",
            "No setting of J or J-crop was chosen on either set; both were trained and selected on APTOS and EyePACS development data only.",
            "J is the pre-declared primary model; J-crop is reported as an ablation and is not preferred on the basis of these results.",
            "P's batch size was 2 and the joint models' 16: J - P confounds joint data with batch size.",
            "On DDR, file-name family is tied to grade; DDR has no patient identifiers; intervals are conditional on the fixed checkpoints.",
            "IDRiD's primary set has 100 images.", "No clinical-utility, equivalence or non-inferiority claim is made."],
        "rules": ["evaluation only", "no tuning", "no threshold changes", "no checkpoint or seed selection", "one run per dataset",
                  "a rerun only for a technical failure, with the reason recorded"],
        "confirmation_tokens": dict(TOKENS)}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(protocol, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    return protocol, ig.sha256_file(path)


def load_protocol(path=PROTOCOL_PATH):
    protocol = _json(path)
    expected = {f"{m}_seed{s}" for m in ARMS for s in SEEDS}
    if set(protocol["checkpoints"]) != expected or len({e["sha256"] for e in protocol["checkpoints"].values()}) != len(expected):
        raise ProtocolError("the protocol must pin exactly the six joint checkpoints")
    if protocol["comparisons"]["primary"] != [PRIMARY] or protocol["confirmation_tokens"] != TOKENS:
        raise ProtocolError("the protocol is not this evaluation's")
    ddr_protocol, ddr_sha, parent = ev.load_protocol()
    if protocol["parent_protocols"]["ddr_grading_protocol_v1.json"] != ddr_sha:
        raise ProtocolError("the DDR protocol is not the pinned file")
    return protocol, ig.sha256_file(path), ddr_protocol, parent


def verify_files(protocol, drive_root, parent):
    """The six checkpoints and the stored reference tables exist with their pinned SHA-256."""
    files = {"checkpoints": {}, "kinds": {}, "lesion_prior": {}, "pretrained": _root(drive_root, parent["checkpoints"]["convnext_pretrained"]["path"])}
    for name, entry in protocol["checkpoints"].items():
        path = _root(drive_root, entry["path"])
        if not os.path.exists(path) or ig.sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"{name}: missing or not the pinned checkpoint")
        files["checkpoints"][name], files["kinds"][name] = path, "P"
    for dataset, tables in protocol["stored_reference_tables"].items():
        for name, entry in tables.items():
            if ig.sha256_file(_root(drive_root, entry["path"])) != entry["sha256"]:
                raise ProtocolError(f"{dataset} / {name}: the stored reference table is not the pinned file")
    return files


# --------------------------------------------------------------------------- inference

def predict(models, paths, batch_size, log=print):
    """{(model, seed): logits} for raw image paths: the direct frame feeds the J models, the cropped frame the
    J-crop models; both frames come from one Stage-2 pass of each image."""
    import eyepacs_adaptation_train as eat
    import geometry_diagnostic as gd
    out = {key: [] for key in models}
    for start in range(0, len(paths), CHUNK):
        frames = [gd.two_frames(p)[:2] for p in paths[start:start + CHUNK]]
        for arm_index, model in enumerate(("j", "jc")):
            rgb_all = [f[arm_index] for f in frames]
            for s in range(0, len(rgb_all), batch_size):
                rgb = np.stack(rgb_all[s:s + batch_size]).astype(np.float32)
                for seed in SEEDS:
                    out[(model, seed)].append(np.asarray(models[(model, seed)].predict_on_batch(eat.p_inputs(rgb)), np.float64))
        log(f"  predicted {min(start + CHUNK, len(paths))}/{len(paths)}")
    return {key: np.concatenate(v, 0) for key, v in out.items()}


def parity(drive_root, aptos_raw_dir, out_dir=None, protocol_path=PROTOCOL_PATH, models=None, log=print):
    """Each joint model on the 730 APTOS validation images FROM RAW, through `predict`, against the validation
    table its own run stored. Tolerances: the parent protocol's (logits 0.05, decoded grades identical)."""
    import pandas as pd
    protocol, protocol_sha, _, parent = load_protocol(protocol_path)
    settings = dict(parent["inference"])
    ig.prepare_gpu()
    files = verify_files(protocol, drive_root, parent)
    models = models if models is not None else b2.load_models(files, settings, arms=tuple(ARMS))
    stored = {k: pd.read_csv(_root(drive_root, e["validation_table"]), dtype={"image_id": str}) for k, e in protocol["checkpoints"].items()}
    ids = list(next(iter(stored.values()))["image_id"])
    logits = predict(models, [os.path.join(aptos_raw_dir, f"{i}.png") for i in ids], settings["batch_size"], log)
    tolerance, checks, failures = parent["parity"]["tolerances"], {}, []
    for (model, seed), z in logits.items():
        frame = stored[f"{model}_seed{seed}"]
        mine, _ = b1.table_from_logits(ids, frame["true_grade"].to_numpy(int), z)
        diff = float(np.abs(z - frame[[f"logit_{k}" for k in range(4)]].to_numpy()).max())
        same = bool((mine["predicted_grade"].to_numpy() == frame["predicted_grade"].to_numpy()).all())
        checks[f"{model}_seed{seed}"] = {"logit_max_abs": diff, "grades_equal": same}
        if list(frame["image_id"]) != ids or diff > tolerance["logit_max_abs"] or not same:
            failures.append(f"{model}-{seed}")
    result = {"gate": "final external re-test: APTOS parity from raw", "official": True, "PASS": not failures, "failures": failures, "checks": checks,
              "protocol_sha256": protocol_sha, "settings": settings, "git_commit": ig.git_commit(), "environment": ig.environment(settings),
              "external_images_read": 0}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "parity.json"), "w") as fh:
            json.dump(ph._jsonable(result), fh, indent=1)
    log(f"PARITY {'PASS' if result['PASS'] else 'FAIL'} {failures}")
    return result


# --------------------------------------------------------------------------- analysis

def analyse(tables, frames, mask, families=None):
    """One population: the project's metric block, every contrast with its interval, reading and seed flag, the
    seed ensemble, referable-DR operating point and calibration; on DDR also the primary contrast per family."""
    grades = tables[SEEDS[0]][MODELS[0]]["grade"][mask]
    indices = ph.bootstrap_indices(grades)
    contrasts = {}
    for name, (a, b) in CONTRASTS.items():
        d = ev.qwk_difference(tables, a, b, mask, indices)
        d["reading"] = ev.reading(d["ci_low"], d["ci_high"])
        contrasts[name] = d
    ensemble = {m: ev.ensemble_table(tables, m) for m in MODELS}
    out = {"n": int(mask.sum()), "grade_counts": np.bincount(grades, minlength=5).tolist(), "contrasts_qwk": contrasts,
           "metric_block": b1.summarise(tables, mask, models=MODELS, contrasts=CONTRASTS, primary=(PRIMARY,)),
           "seed_ensemble_qwk": {m: ph._fast_qwk(grades, ensemble[m]["pred"][mask]) for m in MODELS},
           "referable_dr": {m: {str(s): ev.referable(ev.subset(tables[s][m], mask)) for s in SEEDS} for m in MODELS},
           "calibration": {m: {str(s): ev.calibration(frames[(m, s)][mask]) for s in SEEDS} for m in MODELS}}
    if families is not None:
        out["by_family"] = {}
        for family in ("timestamp", "007"):
            inside = mask & (families == family)
            out["by_family"][family] = {"n": int(inside.sum()),
                                        PRIMARY: ev.qwk_difference(tables, "j", "p", inside, ph.bootstrap_indices(tables[SEEDS[0]]["p"]["grade"][inside]))}
        out["by_family"]["limit"] = "grade 0 and grade 4 are never in both families: this cannot show independence of source"
    return out


def _tables(drive_root, protocol, dataset, ids, grades, logits):
    """Per-seed tables of the four models: the two references from their stored files, the two joint models from
    this run's logits. The stored tables must be the same images in the same order."""
    import pandas as pd
    tables, frames = {s: {} for s in SEEDS}, {}
    for m in REFERENCES:
        for s in SEEDS:
            frame = pd.read_csv(_root(drive_root, protocol["stored_reference_tables"][dataset][f"{m}_seed{s}"]["path"]), dtype={"image_id": str})
            if list(frame["image_id"]) != list(ids) or not np.array_equal(frame["true_grade"].to_numpy(int), np.asarray(grades)):
                raise ProtocolError(f"{dataset}: the stored {m} table of seed {s} is not the same images in the same order")
            tables[s][m], frames[(m, s)] = ph.table_arrays(frame), frame
    for (m, s), z in logits.items():
        frames[(m, s)], tables[s][m] = b1.table_from_logits(list(ids), np.asarray(grades), z)
    return tables, frames


def _start(dataset, drive_root, gate, confirm, technical_rerun_reason, protocol_path):
    protocol, protocol_sha, ddr_protocol, parent = load_protocol(protocol_path)
    settings = dict(parent["inference"])
    if confirm != protocol["confirmation_tokens"][dataset]:
        raise ProtocolError(f"the {dataset} re-test runs once: pass confirm=<its confirmation token>")
    b1.require_gates(gate, protocol_sha, parent, settings)
    ig.prepare_gpu()
    out_root = _root(drive_root, OUT[dataset])
    os.makedirs(out_root, exist_ok=True)
    previous = sorted(n for n in os.listdir(out_root) if n.startswith(PREFIX))
    if previous and not (isinstance(technical_rerun_reason, str) and technical_rerun_reason.strip()):
        raise ProtocolError(f"the {dataset} re-test has already been started ({previous}); a rerun needs technical_rerun_reason")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_root, f"{PREFIX}{protocol_sha[:12]}_{stamp}")
    files = verify_files(protocol, drive_root, parent)
    return protocol, protocol_sha, ddr_protocol, parent, settings, out_root, out_dir, files, previous, stamp


def _finish(dataset, out_dir, out_root, protocol, protocol_sha, record, tables, frames, populations, families, log):
    for (m, s) in [(m, s) for m in ARMS for s in SEEDS]:
        frames[(m, s)].to_csv(os.path.join(out_dir, f"per_image_{m}_seed{s}.csv"), index=False)
    results = {}
    for name, mask in populations.items():                           # the primary population first
        results[name] = analyse(tables, frames, mask, families)
        with open(os.path.join(out_dir, f"result_{name}.json"), "w") as fh:
            json.dump(ph._jsonable(results[name]), fh, indent=1)
    hashes = b2.hash_outputs(out_dir)
    log(f"{dataset} re-test written to {out_dir} ({len(hashes)} files hashed)")
    return {"out_dir": out_dir, "results": results, "output_hashes": hashes}


def run_ddr(drive_root, image_dir, audit_dir, gate, *, confirm, technical_rerun_reason=None, protocol_path=PROTOCOL_PATH, models=None, log=print):
    """THE DDR RE-TEST of the six joint models (P and P-EP come from the stored tables of record 86)."""
    protocol, sha, ddr_protocol, parent, settings, out_root, out_dir, files, previous, stamp = _start(
        "ddr", drive_root, gate, confirm, technical_rerun_reason, protocol_path)
    population = ev.read_population(audit_dir, ddr_protocol)
    ev.verify_images(image_dir, population, log)
    ev.check_screen(audit_dir, ddr_protocol, population)
    ev.check_official(audit_dir, ddr_protocol, population)
    os.makedirs(out_dir)
    record = {"dataset": "ddr", "timestamp_utc": stamp, "protocol_sha256": sha, "git_commit": ig.git_commit(), "previous_outputs": previous,
              "technical_rerun_reason": technical_rerun_reason, "use_of_this_test_set": "second grading use (first: record 86)"}
    for path in (os.path.join(out_root, f"{PREFIX}lock_ddr_{sha[:12]}.json"), os.path.join(out_dir, "lock_record.json")):   # before any prediction
        with open(path, "w") as fh:
            json.dump(record, fh, indent=1)
    models = models if models is not None else b2.load_models(files, settings, arms=tuple(ARMS))
    logits = predict(models, [os.path.join(image_dir, n) for n in population["ids"]], settings["batch_size"], log)
    tables, frames = _tables(drive_root, protocol, "ddr", population["ids"], population["grades"], logits)
    populations = {"primary_3540": population["primary"], "broader_3725_not_untouched": np.ones(len(population["ids"]), bool)}
    return _finish("ddr", out_dir, out_root, protocol, sha, record, tables, frames, populations, population["family"], log)


def run_idrid(drive_root, idrid_raw_dir, gate, *, confirm, technical_rerun_reason=None, protocol_path=PROTOCOL_PATH, models=None, log=print):
    """THE IDRiD RE-TEST of the six joint models (P and P-EP come from the stored tables of records 61 and 79)."""
    protocol, sha, ddr_protocol, parent, settings, out_root, out_dir, files, previous, stamp = _start(
        "idrid", drive_root, gate, confirm, technical_rerun_reason, protocol_path)
    data = ig.read_idrid(idrid_raw_dir, parent)                      # verifies the image-set and label-file hashes
    os.makedirs(out_dir)
    record = {"dataset": "idrid", "timestamp_utc": stamp, "protocol_sha256": sha, "git_commit": ig.git_commit(), "previous_outputs": previous,
              "technical_rerun_reason": technical_rerun_reason, "use_of_this_test_set": "fourth use (records 61, 67, 79)"}
    for path in (os.path.join(out_root, f"{PREFIX}lock_idrid_{sha[:12]}.json"), os.path.join(out_dir, "lock_record.json")):
        with open(path, "w") as fh:
            json.dump(record, fh, indent=1)
    models = models if models is not None else b2.load_models(files, settings, arms=tuple(ARMS))
    logits = predict(models, list(data["paths"]), settings["batch_size"], log)
    tables, frames = _tables(drive_root, protocol, "idrid", data["ids"], data["grades"], logits)
    primary = np.asarray(data["primary"], bool)
    return _finish("idrid", out_dir, out_root, protocol, sha, record, tables, frames,
                   {"primary_100": primary, "all_103_sensitivity": np.ones(len(primary), bool)}, None, log)
