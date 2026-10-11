"""DDR grading test set -- one-time restricted cross-dataset evaluation of the frozen graders P and P-EP, with
E1-EP and E2-EP as secondary arms (research record §82; protocol ddr_grading_protocol_v1.json).

Everything that defines the run is in the protocol: the two image lists by hash, the twelve checkpoint hashes,
the comparisons, the bootstrap, the analyses of source and same-visit effects, the stop conditions.

What is reused, not re-implemented: the image path (`idrid_batch1_eval.rgb_from_raw`: Stage 2 once, then the
512 frame), the per-image table and CORN decode (`table_from_logits`), the APTOS parity gates (`gates`), the gate
requirement (`require_gates`), the metric block (`summarise`), the EP checkpoint identity checks and the model
loader (`idrid_batch2_eval.verify_checkpoints`, `load_models`), and `arch1_posthoc` (QWK, bootstrap resamples).
Those modules were written for the IDRiD evaluations; only their dataset-independent functions are called here.
This module has no IDRiD code path: it never calls a dataset reader of those modules and takes no IDRiD
directory.

What is specific to DDR and lives here: the population read from the two pinned lists, the image and screen
checks, the P checkpoints run next to the EP ones (no stored P prediction on DDR exists), inference in chunks
(the frames of 3,725 images do not fit in memory at once), and the analysis -- the one primary contrast with
its reading rule and seed-consistency flag, the family-stratified and same-visit-proxy analyses, the broader
sensitivity set.

Order, enforced: `gates` must be an official PASS from this runtime before `run` decodes any DDR image; `run`
needs the confirmation token and writes a lock before any prediction exists; a second run is refused except
for a technical failure with the reason recorded. Nothing is trained and nothing is chosen on DDR.
"""
import datetime
import json
import os

import numpy as np

import arch1_posthoc as ph
import ddr_grading_data as dd
import idrid_batch1_eval as b1
import idrid_batch2_eval as b2
import idrid_grading_eval as ig

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "ddr_grading_protocol_v1.json")
SEEDS = b1.SEEDS
ARMS = {"p": "P", "p_ep": "P", "e1_ep": "E1", "e2_ep": "E1"}         # arm -> the graph its checkpoint belongs to
MODELS = tuple(ARMS)
CONTRASTS = {"p_ep_minus_p": ("p_ep", "p"), "e1_ep_minus_p_ep": ("e1_ep", "p_ep"), "e2_ep_minus_p_ep": ("e2_ep", "p_ep")}
PRIMARY = "p_ep_minus_p"
LABELS = {"p": "P (ImageNet encoder, APTOS-trained)", "p_ep": "P-EP (EyePACS-adapted encoder, APTOS-trained)",
          "e1_ep": "E1-EP (P-EP with aligned lesion supervision)", "e2_ep": "E2-EP (P-EP with deranged lesion targets; control)"}
OUTPUT_PREFIX = "ddr_grading_v1_"
RESULT_FILES = ("lock_record.json", "run_configuration.json", "image_manifest.json", "primary_result.json", "primary_report.md",
                "secondary_results.json", "sensitivity_results.json")
ProtocolError = ig.ProtocolError


def _json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def expected_outputs():
    """Every file a complete run directory holds (output_hashes.json is written last and lists the others)."""
    return sorted(RESULT_FILES + tuple(f"per_image_{arm}_seed{seed}.csv" for arm in MODELS for seed in SEEDS) + ("output_hashes.json",))


# --------------------------------------------------------------------------- protocol and pinned files

def load_protocol(path=PROTOCOL_PATH):
    """(protocol, its sha256, parent protocol). The parent and the EP protocol must be the pinned locked files;
    the twelve checkpoint pins must be THEIR pins; the comparisons, bootstrap and screen thresholds must be the
    ones this code implements."""
    protocol = _json(path)
    parent, parent_sha = ig.load_protocol(os.path.join(HERE, protocol["parent_protocol"]["file"]))
    if parent_sha != protocol["parent_protocol"]["sha256"]:
        raise ProtocolError("the parent protocol is not the pinned locked file")
    ep_path = os.path.join(HERE, protocol["ep_protocol"]["file"])
    if ig.sha256_file(ep_path) != protocol["ep_protocol"]["sha256"]:
        raise ProtocolError("the EP protocol is not the pinned locked file")
    ep = b2.load_protocol(ep_path)[0]
    if tuple(protocol["seeds"]) != SEEDS:
        raise ProtocolError("protocol seeds are not 42, 123, 2026")
    expected = {f"{arm}_seed{seed}" for arm in MODELS for seed in SEEDS}
    if set(protocol["checkpoints"]) != expected:
        raise ProtocolError(f"the protocol must pin exactly {sorted(expected)}")
    for name, entry in protocol["checkpoints"].items():
        arm, _, seed = name.rpartition("_seed")
        source = parent["checkpoints"] if arm == "p" else ep["checkpoints"]
        if entry["arm"] != arm or int(entry["seed"]) != int(seed) or entry["model_type"] != ARMS[arm]:
            raise ProtocolError(f"{name}: the pinned entry is for {entry['arm']} / seed {entry['seed']} / {entry['model_type']}")
        for key in ("path", "sha256", "best_slot", "best_epoch_index"):
            if entry[key] != source[name][key]:
                raise ProtocolError(f"{name}: {key} is not the value pinned in {entry['pinned_in']}")
    if len({e["sha256"] for e in protocol["checkpoints"].values()}) != len(expected):
        raise ProtocolError("two pinned checkpoints have the same hash")
    comparisons = protocol["comparisons"]
    if comparisons["primary"] != [PRIMARY] or set(comparisons["primary"]) | set(comparisons["secondary_descriptive"]) != set(CONTRASTS):
        raise ProtocolError("the comparisons are not the one primary and two secondary differences of this evaluation")
    boot = protocol["analysis"]["bootstrap"]
    if boot["n_boot"] != ph.N_BOOT or boot["seed"] != ph.BOOT_SEED:
        raise ProtocolError("the bootstrap is not the project's (2,000 resamples, seed 20260927)")
    screen = protocol["near_duplicate_screen"]
    if screen["copy_band_below"] != dd.COPY_BAND or screen["ambiguous_band_below"] != dd.AMBIGUOUS_BAND:
        raise ProtocolError("the screen thresholds are not the implemented ones")
    if protocol["data"]["revision"] != dd.REVISION:
        raise ProtocolError("the protocol names another revision of the mirror")
    if set(protocol["stored_validation_tables"]) != set(MODELS):
        raise ProtocolError("a stored validation table is missing for an arm")
    return protocol, ig.sha256_file(path), parent


def verify_files(protocol, drive_root, parent=None):
    """Every one of the twelve pinned checkpoints exists with its SHA-256 and is what the protocol says it is.
    P: the file's hash and the run's BEST pointer (slot and epoch). EP arms: idrid_batch2_eval.verify_checkpoints
    (run identity, start from the pinned adapted encoder, BEST by APTOS validation QWK) -- no stored prediction
    table is opened. Returns {'checkpoints', 'kinds', 'lesion_prior', 'pretrained', 'adapted_encoder_sha256'}."""
    root = lambda rel: os.path.join(drive_root, *rel.split("/"))
    if parent is None:
        parent = ig.load_protocol(os.path.join(HERE, protocol["parent_protocol"]["file"]))[0]
    ep = b2.load_protocol(os.path.join(HERE, protocol["ep_protocol"]["file"]))[0]
    files = b2.verify_checkpoints(ep, drive_root, parent)
    checkpoints, kinds = dict(files["checkpoints"]), dict(files["kinds"])
    for seed in SEEDS:
        name = f"p_seed{seed}"
        entry = protocol["checkpoints"][name]
        path = root(entry["path"])
        if not os.path.exists(path):
            raise ProtocolError(f"{name}: {path} not found")
        if ig.sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"{name}: {path} is not the pinned checkpoint (SHA-256 mismatch)")
        pointer = _json(root(entry["best_pointer"]))
        if pointer["active"] != entry["best_slot"] or int(pointer["epoch"]) != entry["best_epoch_index"]:
            raise ProtocolError(f"{name}: the run's BEST pointer no longer names the pinned checkpoint")
        checkpoints[name], kinds[name] = path, "P"
    for name, entry in protocol["checkpoints"].items():
        if name not in checkpoints or kinds[name] != entry["model_type"]:
            raise ProtocolError(f"{name}: not verified as a {entry['model_type']} checkpoint")
    return {"checkpoints": checkpoints, "kinds": kinds, "lesion_prior": files["lesion_prior"], "pretrained": files["pretrained"],
            "adapted_encoder_sha256": files["adapted_encoder_sha256"]}


# --------------------------------------------------------------------------- models

def load_models(files, settings):
    """{(arm, seed): keras model} for the twelve checkpoints (idrid_batch2_eval.load_models, with P added)."""
    return b2.load_models(files, settings, arms=MODELS)


def predict_logits(models, frames, batch_size):
    """{(arm, seed): (N, 4) float64 CORN logits}. P and P-EP are fed through P's three inputs; E1-EP and E2-EP
    take the RGB frame and only output 0 is read."""
    import eyepacs_adaptation_train as eat
    out = {key: [] for key in models}
    for start in range(0, len(frames), batch_size):
        rgb = np.stack(frames[start:start + batch_size]).astype(np.float32)
        for (arm, seed), net in models.items():
            z = net.predict_on_batch(eat.p_inputs(rgb)) if ARMS[arm] == "P" else net.predict_on_batch({"rgb": rgb})[0]
            out[(arm, seed)].append(np.asarray(z, np.float64))
    return {key: np.concatenate(v, 0) for key, v in out.items()}


def gates(drive_root, aptos_raw_dir, bundle, out_dir, *, aptos_processed_dir=None, protocol_path=PROTOCOL_PATH, settings=None,
          official=True, models=None, log=print):
    """The pre-inference gates of the IDRiD batches for this evaluation's twelve checkpoints: file and identity
    verification, model precision, and APTOS parity (ten raw images through the locked image path; all 730
    validation images against each model's stored BEST validation table). No DDR image is read."""
    result = b1.gates(drive_root, aptos_raw_dir, bundle, None, aptos_processed_dir=aptos_processed_dir, protocol_path=protocol_path,
                      settings=settings, official=official, models=models, log=log,
                      batch={"load_protocol": load_protocol, "verify_files": verify_files, "load_models": load_models,
                             "predict_logits": predict_logits, "name": "ddr grading v1 pre-inference gates", "label": "DDR GRADING V1"})
    result["kind"] = "official gate" if official else "pre-check (cannot unlock the DDR evaluation)"
    result["ddr_images_read"] = 0
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "gates.json"), "w") as fh:
            json.dump(ph._jsonable(result), fh, indent=1)
    return result


# --------------------------------------------------------------------------- population, images, screen

def read_population(audit_dir, protocol):
    """The two pinned lists, checked: file hashes, sizes, grade counts, the primary list inside the broader one,
    the broader list = primary + the lesion-subset images, unique names and contents. Returns the broader rows
    (the inference order) with the masks of the analysis sets and the label-free source signals."""
    pinned = protocol["populations"]
    rows = {}
    for name in ("primary", "broader"):
        path = os.path.join(audit_dir, pinned[name]["list"])
        if not os.path.exists(path) or ig.sha256_file(path) != pinned[name]["list_sha256"]:
            raise ProtocolError(f"the {name} image list is missing or is not the pinned file")
        rows[name] = dd.read_list(path)
        if len(rows[name]) != pinned[name]["n"] or dd.grade_counts(rows[name]) != pinned[name]["grades_0_to_4"]:
            raise ProtocolError(f"the {name} list does not hold the pinned images")
    broader = rows["broader"]
    names = [r["image"] for r in broader]
    primary_names = {r["image"] for r in rows["primary"]}
    if len(set(names)) != len(names) or len({r["sha256"] for r in broader}) != len(names):
        raise ProtocolError("the broader list repeats a file name or a file content")
    lesion = np.asarray([bool(r["lesion_subset"]) for r in broader])
    primary = np.asarray([n in primary_names for n in names])
    if int(primary.sum()) != len(primary_names) or not np.array_equal(primary, ~lesion) or any(r["lesion_subset"] for r in rows["primary"]):
        raise ProtocolError("the primary list is not the broader list without the lesion-subset images")
    if any(r["grade"] not in range(5) for r in broader):
        raise ProtocolError("an ungradable image is listed")
    return {"ids": names, "grades": np.asarray([r["grade"] for r in broader], int), "sha256": [r["sha256"] for r in broader],
            "primary": primary, "lesion_subset": lesion, "family": np.asarray([r["family"] for r in broader]),
            "units": np.asarray(dd.timestamp_chains(names))}


def verify_images(image_dir, population, log=print):
    """Every listed image exists and has its listed SHA-256 (the files are hashed, not decoded)."""
    for n, (name, digest) in enumerate(zip(population["ids"], population["sha256"]), start=1):
        path = os.path.join(image_dir, name)
        if not os.path.exists(path):
            raise ProtocolError(f"{name} is not in {image_dir}")
        if ig.sha256_file(path) != digest:
            raise ProtocolError(f"{name} does not have its listed SHA-256")
        if n % 1000 == 0:
            log(f"  image hashes {n}/{len(population['ids'])}")
    return len(population["ids"])


def check_screen(audit_dir, protocol, population):
    """The near-duplicate screen was run on the pinned broader list with the pinned thresholds and references,
    and names no image in the copy band. Returns (flagged mask, the screen's SHA-256)."""
    spec = protocol["near_duplicate_screen"]
    path = os.path.join(audit_dir, dd.SCREEN_FILE)
    if not os.path.exists(path):
        raise ProtocolError("the near-duplicate screen has not been run (no screen file)")
    report = _json(path)
    if report["list_sha256"] != protocol["populations"]["broader"]["list_sha256"] or report["images"] != len(population["ids"]):
        raise ProtocolError("the near-duplicate screen was run on another image list")
    if report["thresholds"] != {"copy_band_below": spec["copy_band_below"], "ambiguous_band_below": spec["ambiguous_band_below"]} \
            or report.get("labels_used") is not False:
        raise ProtocolError("the near-duplicate screen was not run with the pinned thresholds")
    for name, entry in spec["against"].items():
        seen = report["against"].get(name)
        if not seen or seen["signature_file_sha256"] != entry["signature_file_sha256"] or seen["images"] != entry["images"]:
            raise ProtocolError(f"the near-duplicate screen did not compare against the pinned {name} signatures")
    if report["exclude"]:
        raise ProtocolError(f"the screen names {len(report['exclude'])} image(s) in the copy band: version 1 is not run; "
                            "a new population and protocol version must be issued")
    if set(report["flagged"]) - set(population["ids"]):
        raise ProtocolError("the screen flags an image that is not listed")
    flagged = set(report["flagged"])
    return np.asarray([n in flagged for n in population["ids"]]), ig.sha256_file(path)


def check_official(audit_dir, protocol, population):
    """The downloaded images were compared with the official archive's directory (size and CRC-32) and none
    differs. Returns the check file's SHA-256."""
    path = os.path.join(audit_dir, dd.OFFICIAL_CHECK_FILE)
    if not os.path.exists(path):
        raise ProtocolError("the comparison with the official archive's directory has not been run")
    report = _json(path)
    if report["list_sha256"] != protocol["populations"]["broader"]["list_sha256"] or report["images"] != len(population["ids"]) \
            or report["official_directory_sha256"] != protocol["data"]["official_directory"]["sha256"]:
        raise ProtocolError("the official-directory comparison was run on another list or another directory file")
    if report["mismatches"] or report["not_in_official_directory"] or report["size_and_crc32_equal"] != len(population["ids"]):
        raise ProtocolError("a downloaded image differs from the official archive's directory entry of the same name")
    return ig.sha256_file(path)


# --------------------------------------------------------------------------- analysis

def subset(table, mask):
    return {k: v[mask] for k, v in table.items()}


def interval(draws):
    lo, hi = np.nanpercentile(np.asarray(draws, np.float64), [2.5, 97.5])
    return float(lo), float(hi)


def reading(lo, hi):
    """The pre-declared reading of the primary interval."""
    if lo > 0:
        return "supported: the interval lies wholly above zero"
    if hi < 0:
        return "reversed: the interval lies wholly below zero"
    return "no difference shown: the interval includes zero (no equivalence or non-inferiority statement)"


def qwk_difference(tables, a, b, mask, indices):
    """The contrast's quantity on the masked images: QWK(a) - QWK(b) per seed-labelled pair, their mean, and the
    percentile interval of that mean over `indices` (lists of row positions inside the mask; the same resample is
    applied to every table). Also the seed-consistency flag."""
    pairs = [(subset(tables[s][a], mask), subset(tables[s][b], mask)) for s in SEEDS]
    for x, y in pairs:
        if not np.array_equal(x["ids"], y["ids"]) or not np.array_equal(x["grade"], y["grade"]):
            raise ProtocolError("the two tables are not the same images in the same order")
    per_seed = [ph._fast_qwk(x["grade"], x["pred"]) - ph._fast_qwk(y["grade"], y["pred"]) for x, y in pairs]
    draws = [float(np.mean([ph._fast_qwk(x["grade"][i], x["pred"][i]) - ph._fast_qwk(y["grade"][i], y["pred"][i]) for x, y in pairs]))
             for i in indices]
    lo, hi = interval(draws)
    positive = int(sum(d > 0 for d in per_seed))
    return {"mean": float(np.mean(per_seed)), "per_seed": {str(s): float(d) for s, d in zip(SEEDS, per_seed)},
            "ci_low": lo, "ci_high": hi, "seeds_positive": positive, "seed_level_inconsistency": positive != len(SEEDS),
            "n": int(np.sum(mask)), "resamples": len(indices)}


def cluster_indices(units, strata, n_boot=ph.N_BOOT, seed=ph.BOOT_SEED):
    """Resamples of UNITS (with replacement, within each stratum); every draw is the row positions of the images
    of the drawn units. Used for the same-visit proxy: units are file-time chains, strata the file-name families."""
    rng = np.random.default_rng(seed)
    groups = []
    for stratum in sorted(set(strata.tolist())):
        members = {}
        for position in np.flatnonzero(strata == stratum):
            members.setdefault(units[position], []).append(position)
        groups.append([np.asarray(v) for v in members.values()])
    return [np.concatenate([group[k] for group in groups for k in rng.integers(0, len(group), len(group))]) for _ in range(n_boot)]


def ensemble_table(tables, arm):
    """One table per arm from the three seeds: cumulative probabilities averaged, then the same decode."""
    first = tables[SEEDS[0]][arm]
    p = np.mean([tables[s][arm]["p_gt"] for s in SEEDS], axis=0)
    return {"ids": first["ids"], "grade": first["grade"], "p_gt": p, "pred": (p > 0.5).sum(axis=1).astype(int)}


def calibration(frame):
    """Top-label ECE (15 equal-width bins), NLL and multiclass Brier from the stored class probabilities."""
    grade = frame["true_grade"].to_numpy(int)
    prob = np.stack([frame[f"class_prob_{k}"].to_numpy(np.float64) for k in range(5)], 1)
    confidence, top = prob.max(1), prob.argmax(1)
    correct = (top == grade).astype(np.float64)
    edges, ece = np.linspace(0.0, 1.0, 16), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return {"ece_top_label": float(ece), "nll": float(-np.mean(np.log(np.clip(prob[np.arange(len(grade)), grade], 1e-12, 1.0)))),
            "brier_multiclass": float(np.mean(np.sum((prob - np.eye(5)[grade]) ** 2, axis=1))),
            "mean_confidence": float(confidence.mean()), "top_label_accuracy": float(correct.mean())}


def referable(table):
    g, pred = table["grade"] >= 2, table["pred"] >= 2
    return {"sensitivity": float(pred[g].mean()) if g.any() else float("nan"), "specificity": float((~pred[~g]).mean()) if (~g).any() else float("nan")}


def family_analysis(tables, population, mask):
    """Inside each file-name family: the paired P-EP - P difference on the grades that family holds, the AUROC of
    every cumulative cut that has both sides there, and the decoded-grade distribution of every arm. Descriptive:
    grade 0 and grade 4 are never in both families, so this cannot show independence of source."""
    out = {}
    for name in ("timestamp", "007"):
        inside = mask & (population["family"] == name)
        grades = population["grades"][inside]
        entry = {"n": int(inside.sum()), "grade_counts": np.bincount(grades, minlength=5).tolist(),
                 "p_ep_minus_p": qwk_difference(tables, "p_ep", "p", inside, ph.bootstrap_indices(grades)), "arms": {}}
        for arm in MODELS:
            per_seed = [subset(tables[s][arm], inside) for s in SEEDS]
            entry["arms"][arm] = {
                "qwk_per_seed": [ph._fast_qwk(t["grade"], t["pred"]) for t in per_seed],
                "auroc_cut_mean": {f"ge{c}": float(np.mean([ph._auroc(t["grade"] >= c, t["p_gt"][:, c - 1]) for t in per_seed])) for c in ph.CUTS},
                "decoded_grade_counts_per_seed": [np.bincount(t["pred"], minlength=5).tolist() for t in per_seed]}
        out[name] = entry
    out["pooled_auroc_cut_mean"] = {arm: {f"ge{c}": float(np.mean([ph._auroc(subset(tables[s][arm], mask)["grade"] >= c,
                                                                              subset(tables[s][arm], mask)["p_gt"][:, c - 1]) for s in SEEDS]))
                                          for c in ph.CUTS} for arm in MODELS}
    out["limit"] = ("descriptive: grade 0 and grade 4 are not represented in both families, so the family-stratified analysis cannot "
                    "establish that performance is independent of source")
    return out


def analyse_primary(tables, population):
    """The primary result: P-EP - P on the primary population, with the pre-declared reading."""
    mask = population["primary"]
    grades = population["grades"][mask]
    indices = ph.bootstrap_indices(grades)
    d = qwk_difference(tables, "p_ep", "p", mask, indices)
    d["reading"] = reading(d["ci_low"], d["ci_high"])
    return {"contrast": "QWK(P-EP) - QWK(P), mean of the three seed-labelled pairs", "n": int(mask.sum()),
            "grade_counts": np.bincount(grades, minlength=5).tolist(), "result": d,
            "qwk_per_seed": {arm: {str(s): ph._fast_qwk(subset(tables[s][arm], mask)["grade"], subset(tables[s][arm], mask)["pred"])
                                   for s in SEEDS} for arm in ("p", "p_ep")},
            "bootstrap": {"method": "paired, grade-stratified over images", "n_boot": len(indices), "seed": ph.BOOT_SEED},
            "scope": "the interval is conditional on the fixed checkpoints and treats images as independent; without patient "
                     "identifiers it may be too narrow. The three P-EP checkpoints share one adapted encoder and are not independent "
                     "replicates of the adaptation."}


def analyse_secondary(tables, frames, population):
    """Everything descriptive on the primary population: the project's metric block for the four arms and three
    contrasts, the seed ensemble, referable-DR operating point, calibration, the source-family analysis and the
    same-visit proxy."""
    mask = population["primary"]
    grades = population["grades"][mask]
    indices = ph.bootstrap_indices(grades)
    block = b1.summarise(tables, mask, models=MODELS, contrasts=CONTRASTS, primary=(PRIMARY,))
    ensemble = {arm: ensemble_table(tables, arm) for arm in MODELS}
    ens_draws = [ph._fast_qwk(grades[i], ensemble["p_ep"]["pred"][mask][i]) - ph._fast_qwk(grades[i], ensemble["p"]["pred"][mask][i]) for i in indices]
    units, strata = population["units"][mask], population["family"][mask]
    cluster = qwk_difference(tables, "p_ep", "p", mask, cluster_indices(units, strata))
    return {"metric_block": block,
            "contrasts_qwk": {name: qwk_difference(tables, a, b, mask, indices) for name, (a, b) in CONTRASTS.items()},
            "seed_ensemble": {"qwk": {arm: ph._fast_qwk(grades, ensemble[arm]["pred"][mask]) for arm in MODELS},
                              "p_ep_minus_p": {"delta": ph._fast_qwk(grades, ensemble["p_ep"]["pred"][mask]) - ph._fast_qwk(grades, ensemble["p"]["pred"][mask]),
                                               "ci_95": list(interval(ens_draws))}},
            "referable_dr": {arm: {str(s): referable(subset(tables[s][arm], mask)) for s in SEEDS} for arm in MODELS},
            "calibration": {arm: {str(s): calibration(frames[(arm, s)][mask]) for s in SEEDS} for arm in MODELS},
            "source_family": family_analysis(tables, population, mask),
            "same_visit_proxy": {"p_ep_minus_p": cluster, "units": int(len(set(units.tolist()))),
                                 "method": "cluster bootstrap over file-time chains, resampled within file-name family",
                                 "limit": "a proxy sensitivity analysis, not patient-level inference"}}


def analyse_sensitivity(tables, population, flagged):
    """The primary quantity on the sets that are NOT the primary population."""
    sets = {"broader_3725_not_untouched": np.ones(len(population["ids"]), bool), "lesion_subset_185_only": population["lesion_subset"]}
    if flagged.any():
        sets["primary_without_screen_flagged"] = population["primary"] & ~flagged
    out = {}
    for name, mask in sets.items():
        grades = population["grades"][mask]
        d = qwk_difference(tables, "p_ep", "p", mask, ph.bootstrap_indices(grades))
        out[name] = {"n": int(mask.sum()), "grade_counts": np.bincount(grades, minlength=5).tolist(), "p_ep_minus_p": d}
    out["screen_flagged_images"] = int(flagged.sum())
    out["note"] = ("sensitivity only. The broader list contains 185 images that were processed in the lesion-representation experiments; "
                   "their grades were never an endpoint, but the set is not untouched and never replaces the primary result")
    return out


def report_markdown(primary, protocol):
    r = primary["result"]
    lines = ["# DDR grading v1 -- PRIMARY RESULT", "",
             f"Population: {primary['n']} images (grades 0-4: {primary['grade_counts']}). Evaluation only; run once.", "",
             f"QWK(P-EP) - QWK(P), mean of three seed-labelled pairs: **{r['mean']:+.4f}** "
             f"(95 % interval {r['ci_low']:+.4f} to {r['ci_high']:+.4f}; {primary['bootstrap']['n_boot']} paired grade-stratified resamples).", "",
             "| seed | QWK P | QWK P-EP | difference |", "|---|---|---|---|"]
    for s in SEEDS:
        lines.append(f"| {s} | {primary['qwk_per_seed']['p'][str(s)]:.4f} | {primary['qwk_per_seed']['p_ep'][str(s)]:.4f} | {r['per_seed'][str(s)]:+.4f} |")
    lines += ["", f"Reading (pre-declared): {r['reading']}.",
              f"Seeds with a positive difference: {r['seeds_positive']} of {len(SEEDS)}."
              + (" **Seed-level inconsistency: at least one difference is zero or negative.**" if r["seed_level_inconsistency"] else ""), "",
              primary["scope"], "",
              "File-name family is tied to grade in this test set (every grade-0 image is timestamp-named, every grade-4 image 007-named): "
              "the pooled QWK cannot separate grading from recognising the source. See secondary_results.json.", ""]
    return "\n".join(lines) + "\n".join(f"- {d}" for d in protocol["disclosures"]) + "\n"


# --------------------------------------------------------------------------- the one-time run

def lock_path(out_root, protocol_sha):
    return os.path.join(out_root, f"ddr_grading_v1_lock_{protocol_sha[:12]}.json")


def existing_outputs(out_root):
    """Anything under `out_root` that belongs to this evaluation (locks and run directories)."""
    return sorted(n for n in os.listdir(out_root) if n.startswith(OUTPUT_PREFIX)) if os.path.isdir(out_root) else []


def preflight(drive_root, image_dir, audit_dir, out_root, protocol_path=PROTOCOL_PATH, log=print):
    """Everything that can be checked without a model and without a prediction: the protocols, the twelve
    checkpoints and their identity, the two lists, every image's hash, the screen, and whether anything of this
    evaluation already exists. Raises on any failure; returns a report."""
    protocol, protocol_sha, parent = load_protocol(protocol_path)
    files = verify_files(protocol, drive_root, parent)
    population = read_population(audit_dir, protocol)
    verify_images(image_dir, population, log)
    flagged, screen_sha = check_screen(audit_dir, protocol, population)
    official_sha = check_official(audit_dir, protocol, population)
    found = existing_outputs(out_root)
    return {"protocol_sha256": protocol_sha, "checkpoints": {k: protocol["checkpoints"][k]["sha256"] for k in sorted(files["checkpoints"])},
            "images": len(population["ids"]), "primary_images": int(population["primary"].sum()),
            "primary_grade_counts": np.bincount(population["grades"][population["primary"]], minlength=5).tolist(),
            "screen_sha256": screen_sha, "screen_flagged": int(flagged.sum()), "official_check_sha256": official_sha, "existing_outputs": found,
            "already_run": bool(found), "lock": lock_path(out_root, protocol_sha), "confirmation_token": protocol["confirmation_token"], "PASS": True}


def run(drive_root, image_dir, audit_dir, out_root, gate, *, confirm, technical_rerun_reason=None, protocol_path=PROTOCOL_PATH,
        command=None, models=None, log=print):
    """THE ONE-TIME DDR EVALUATION. Refuses without the token, without official gates passed in this environment,
    or when anything of this evaluation already exists under `out_root` (a rerun needs a written technical reason
    and is recorded). The lock is written before any prediction exists. `models` is for tests only."""
    protocol, protocol_sha, parent = load_protocol(protocol_path)
    settings = dict(parent["inference"])
    if confirm != protocol["confirmation_token"]:
        raise ProtocolError("the DDR evaluation runs once: pass confirm=<the protocol's confirmation token> to run it")
    b1.require_gates(gate, protocol_sha, parent, settings)
    ig.prepare_gpu()
    os.makedirs(out_root, exist_ok=True)
    lock = lock_path(out_root, protocol_sha)
    previous = existing_outputs(out_root)
    if previous and not (isinstance(technical_rerun_reason, str) and technical_rerun_reason.strip()):
        raise ProtocolError(f"the DDR evaluation has already been started ({previous}); a rerun is allowed only for a technical "
                            "failure, with technical_rerun_reason stated")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_root, f"{OUTPUT_PREFIX}{protocol_sha[:12]}_{stamp}")
    if os.path.exists(out_dir):
        raise ProtocolError(f"{out_dir} already exists; an earlier run is never overwritten")
    files = verify_files(protocol, drive_root, parent)
    population = read_population(audit_dir, protocol)
    verify_images(image_dir, population, log)
    flagged, screen_sha = check_screen(audit_dir, protocol, population)
    official_sha = check_official(audit_dir, protocol, population)
    os.makedirs(out_dir)
    record = {"timestamp_utc": stamp, "protocol_sha256": protocol_sha, "git_commit": ig.git_commit(), "out_dir": out_dir,
              "technical_rerun_reason": technical_rerun_reason, "previous_outputs": previous}
    for path in (lock, os.path.join(out_dir, "lock_record.json")):   # written BEFORE any prediction exists
        with open(path, "w") as fh:
            json.dump(record, fh, indent=1)
    models = models if models is not None else load_models(files, settings)
    chunk, ids = int(protocol["inference"]["chunk_images"]), population["ids"]
    logits = {key: [] for key in models}
    for start in range(0, len(ids), chunk):                          # frames are built and dropped chunk by chunk
        frames = [b1.rgb_from_raw(os.path.join(image_dir, name))[0] for name in ids[start:start + chunk]]
        for key, z in predict_logits(models, frames, settings["batch_size"]).items():
            logits[key].append(z)
        log(f"  predicted {min(start + chunk, len(ids))}/{len(ids)}")
    tables, frames_out = {seed: {} for seed in SEEDS}, {}
    for (arm, seed), parts in logits.items():
        frame, tables[seed][arm] = b1.table_from_logits(ids, population["grades"], np.concatenate(parts, 0))
        frame["in_primary_population"], frame["lesion_subset"] = population["primary"], population["lesion_subset"]
        frame["family"], frame["screen_flagged"] = population["family"], flagged
        frame.to_csv(os.path.join(out_dir, f"per_image_{arm}_seed{seed}.csv"), index=False)
        frames_out[(arm, seed)] = frame
    with open(os.path.join(out_dir, "image_manifest.json"), "w") as fh:
        json.dump({"ids": ids, "sha256": population["sha256"], "grades": population["grades"].tolist(),
                   "in_primary_population": population["primary"].tolist(), "lesion_subset": population["lesion_subset"].tolist(),
                   "list_sha256": {k: protocol["populations"][k]["list_sha256"] for k in ("primary", "broader")},
                   "screen_sha256": screen_sha, "official_check_sha256": official_sha,
                   "screen_flagged": [n for n, f in zip(ids, flagged) if f]}, fh, indent=1)
    with open(os.path.join(out_dir, "run_configuration.json"), "w") as fh:
        json.dump(ph._jsonable({"protocol": protocol, "protocol_sha256": protocol_sha, "environment": ig.environment(settings),
                                "git_commit": ig.git_commit(), "command": command, "timestamp_utc": stamp, "gates": gate,
                                "checkpoints_verified": {k: protocol["checkpoints"][k]["sha256"] for k in sorted(files["checkpoints"])},
                                "adapted_encoder_sha256": files["adapted_encoder_sha256"], "inference_settings": settings,
                                "models_run": [f"{arm}_seed{seed}" for arm in MODELS for seed in SEEDS], "idrid_files_opened": 0}), fh, indent=1)
    primary = analyse_primary(tables, population)                    # computed and written first
    with open(os.path.join(out_dir, "primary_result.json"), "w") as fh:
        json.dump(ph._jsonable(primary), fh, indent=1)
    with open(os.path.join(out_dir, "primary_report.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown(primary, protocol))
    secondary = analyse_secondary(tables, frames_out, population)
    block = secondary["metric_block"]["paired_differences"][PRIMARY]["qwk"]
    if abs(block["mean"] - primary["result"]["mean"]) > 1e-9 or abs(block["ci_low"] - primary["result"]["ci_low"]) > 1e-9 \
            or abs(block["ci_high"] - primary["result"]["ci_high"]) > 1e-9:
        raise ProtocolError("the primary quantity and the project's metric block disagree")
    with open(os.path.join(out_dir, "secondary_results.json"), "w") as fh:
        json.dump(ph._jsonable(secondary), fh, indent=1)
    sensitivity = analyse_sensitivity(tables, population, flagged)
    with open(os.path.join(out_dir, "sensitivity_results.json"), "w") as fh:
        json.dump(ph._jsonable(sensitivity), fh, indent=1)
    hashes = b2.hash_outputs(out_dir)
    missing = sorted(set(expected_outputs()) - set(hashes) - {"output_hashes.json"})
    if missing:
        raise ProtocolError(f"the run directory is incomplete: {missing}")
    log(f"DDR grading v1 written to {out_dir} ({len(hashes)} files hashed)")
    return {"out_dir": out_dir, "primary": primary, "secondary": secondary, "sensitivity": sensitivity, "tables": tables, "output_hashes": hashes}
