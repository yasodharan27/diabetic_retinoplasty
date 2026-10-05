"""The ONE predefined severity-aware fusion of the stored RGB grader P with the stored pathology grader
(research record §58). Evaluation only: laptop, CPU, stored BEST predictions; nothing is trained or tuned.

THE LOCKED RULE, with p_k = P(grade > k), k = 0..3 (CORN cumulative probabilities):

    fused_0 = 0.5 * P_0 + 0.5 * Path_0
    fused_1 = 0.5 * P_1 + 0.5 * Path_1
    fused_2 = 0.5 * P_2 + 0.5 * Path_2
    fused_3 = min(P_3, fused_2)
    grade   = number of k with fused_k > 0.5

>=1, >=2, >=3: equal RGB + pathology fusion. >=4: the pathology branch does not vote, because Stage 4
segments MA / HE / EX / SE and none of the signs that define proliferative DR; the cap keeps the cumulative
vector ordinal. No learned gate, no validation-derived weight, no threshold other than CORN's 0.5.

Controls: P alone; the recorded 50 / 50 fusion (pathology_grader_fusion, unchanged); and this same rule with
the designated other P seed in place of the pathology branch (cycle 42 -> 123 -> 2026 -> 42; the pairs overlap
and are not independent).

The rule was designed after the stored predictions had been inspected, so its result on the APTOS validation
set is descriptive and post hoc. No other rule is evaluated by this module.

    python pathology_grader_severity_fusion.py --experiments results --out results/PathologyGrader/severity_fusion
"""
import argparse
import hashlib
import json
import os

import numpy as np

import arch1_posthoc as ph
import pathology_grader_fusion as pf

SEEDS = pf.SEEDS
RULE = {
    "name": "severity-aware fixed fusion",
    "fused_0": "0.5 * P_0 + 0.5 * Path_0", "fused_1": "0.5 * P_1 + 0.5 * Path_1",
    "fused_2": "0.5 * P_2 + 0.5 * Path_2", "fused_3": "min(P_3, fused_2)",
    "decode": "grade = number of k with fused_k > 0.5",
    "members": ["P, same seed, BEST", "pathology grader, same seed, BEST"],
    "tuned_on_validation": False, "learned": False,
    "control": "the same rule with P seed next(s) in place of the pathology branch (42 -> 123 -> 2026 -> 42)",
}
SHUFFLES = ("lesions", "vessel")
N_VALIDATION = 730
PREFIX = "pathgrader_cb5fc7a8d370"


def fuse_severity(p, other, what="severity-aware fusion"):
    """The locked rule on two aligned per-sample tables: `p` is the RGB grader, `other` the second branch."""
    pf.assert_aligned(p, other, what)
    a, b = p["p_gt"].astype(np.float64), other["p_gt"].astype(np.float64)
    fused = np.empty_like(a)
    fused[:, :3] = 0.5 * a[:, :3] + 0.5 * b[:, :3]
    fused[:, 3] = np.minimum(a[:, 3], fused[:, 2])
    return {"ids": p["ids"], "grade": p["grade"], "p_gt": fused, "pred": pf.decode(fused)}


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pathology_path(experiments_root, seed, suffix=""):
    return os.path.join(experiments_root, "PathologyGrader", f"{PREFIX}_seed{int(seed)}", "metrics",
                        f"per_sample_best{suffix}.csv")


def check_table(table, name, expected_ids, logits=None):
    """CORN semantics of one stored table; raises on the first violation."""
    p = table["p_gt"]
    if p.shape != (N_VALIDATION, 4):
        raise RuntimeError(f"{name}: expected ({N_VALIDATION}, 4) cumulative probabilities, got {p.shape}")
    if list(table["ids"]) != list(expected_ids):
        raise RuntimeError(f"{name}: not the authoritative validation images in the authoritative order")
    if p.min() < 0 or p.max() > 1 or np.any(np.diff(p, axis=1) > 1e-9):
        raise RuntimeError(f"{name}: p_gt is not a non-increasing cumulative probability vector in [0, 1]")
    if not np.array_equal(pf.decode(p), table["pred"]):
        raise RuntimeError(f"{name}: stored grades are not the CORN decode of the stored probabilities")
    if set(np.unique(table["grade"])) != {0, 1, 2, 3, 4}:
        raise RuntimeError(f"{name}: true grades are not 0..4")
    if logits is not None:
        cumulative = np.cumprod(1.0 / (1.0 + np.exp(-logits.astype(np.float64))), axis=1)
        if np.abs(cumulative - p).max() > 1e-5:
            raise RuntimeError(f"{name}: p_gt is not the cumulative product of sigmoid(logits)")
    return True


def authoritative_validation_ids():
    """The project's 730 validation images in order: the authoritative split minus the pinned empty-FOV ids."""
    import multiseed_runs as msr
    import pipeline_v2_config as v2cfg
    _, val, sha = msr.verify_split()
    if sha != v2cfg.SPLIT_SHA256:
        raise RuntimeError("the split is not the authoritative one")
    empty = set(v2cfg.APTOS_EMPTY_FOV_IDS)
    ids = [str(i) for i, _ in val if str(i) not in empty]
    grades = [int(g) for i, g in val if str(i) not in empty]
    if len(ids) != N_VALIDATION:
        raise RuntimeError(f"expected {N_VALIDATION} validation images, got {len(ids)}")
    return ids, np.asarray(grades)


def prerun_audit(experiments_root, log=print):
    """Loads every stored table and verifies the prerequisites. Raises (and so stops the run) on any failure."""
    import pandas as pd
    ids, grades = authoritative_validation_ids()
    tables, files = {"p": {}, "pathology": {}, "shuffled": {}}, {}
    for s in SEEDS:
        for kind, path in (("p", pf.p_prediction_path(experiments_root, s, "best")),
                           ("pathology", pathology_path(experiments_root, s))):
            if "per_sample_best" not in os.path.basename(path):
                raise RuntimeError(f"{path}: not a BEST prediction table")
            frame = pd.read_csv(path, dtype={"image_id": str})
            table = ph.table_arrays(frame)
            check_table(table, f"{kind}-{s}", ids, frame[[f"logit_{k}" for k in range(4)]].to_numpy())
            if not np.array_equal(table["grade"], grades):
                raise RuntimeError(f"{kind}-{s}: true grades differ from the authoritative split")
            stored = json.load(open(os.path.join(os.path.dirname(path), "metrics_best.json")))
            stored_metrics = stored.get("metrics", stored)
            ph.check_stored(ph.metrics(table), stored_metrics, f"{kind}-{s}")       # the BEST checkpoint's own metrics
            checkpoint = stored.get("checkpoint")
            which = checkpoint.get("which") if isinstance(checkpoint, dict) else checkpoint
            if which != "BEST":
                raise RuntimeError(f"{kind}-{s}: metrics file is for checkpoint {which!r}, not BEST")
            tables[kind][s], files[f"{kind}-{s}"] = table, {"path": path, "sha256": _sha256(path)}
        tables["shuffled"][s] = {}
        for name in SHUFFLES:
            path = pathology_path(experiments_root, s, f"_shuffle_{name}")
            table = ph._read_table(path)
            check_table(table, f"pathology-{s} shuffle {name}", ids)
            tables["shuffled"][s][name], files[f"pathology-{s}-shuffle-{name}"] = table, {"path": path, "sha256": _sha256(path)}
    probe = {"ids": np.array(["a", "b", "c"]), "grade": np.array([4, 4, 2]), "pred": np.zeros(3, int),
             "p_gt": np.array([[1.0, 0.9, 0.8, 0.7], [1.0, 0.9, 0.4, 0.35], [0.9, 0.8, 0.2, 0.1]])}
    other = dict(probe, p_gt=np.array([[0.9, 0.7, 0.6, 0.1], [0.8, 0.5, 0.2, 0.0], [0.7, 0.6, 0.6, 0.5]]))
    got = fuse_severity(probe, other)
    want = np.array([[0.95, 0.8, 0.7, 0.7], [0.9, 0.7, 0.3, 0.3], [0.8, 0.7, 0.4, 0.1]])
    if not np.allclose(got["p_gt"], want) or list(got["pred"]) != [4, 2, 2]:
        raise RuntimeError("the implemented rule is not the locked formula")
    log(f"PRE-RUN AUDIT OK: {N_VALIDATION} authoritative validation images, order verified for 3 P + 3 pathology BEST tables "
        f"and 6 shuffle tables; CORN cumulative semantics verified; stored BEST metrics reproduced; locked formula verified.")
    return tables, files, grades


def model_row(m):
    return pf.headline(m)


def evaluate(tables, grades, n_boot=ph.N_BOOT):
    indices = ph.bootstrap_indices(grades, n_boot)
    per_seed, draws = {}, {}
    comparisons = ("severity_minus_p", "severity_minus_5050", "severity_minus_control")
    for s in SEEDS:
        p, q, other = tables["p"][s], tables["pathology"][s], tables["p"][pf.control_seed(s)]
        models = {"p": p, "pathology": q, "fusion_5050": pf.fuse(p, q), "severity": fuse_severity(p, q),
                  "control_severity": fuse_severity(p, other, "P + P control")}
        metrics = {k: ph.metrics(v) for k, v in models.items()}
        deltas = {}
        for name, a, b in (("severity_minus_p", "severity", "p"), ("severity_minus_5050", "severity", "fusion_5050"),
                           ("severity_minus_control", "severity", "control_severity")):
            d = ph.paired_delta(models[a], models[b], indices=indices)
            for k in ph.BOOT_KEYS:
                draws.setdefault(name, {}).setdefault(k, []).append(d[k].pop("_draws"))
            d["grade4_recall"] = {"delta": metrics[a]["recall_per_grade"][4] - metrics[b]["recall_per_grade"][4]}
            deltas[name] = d
        shuffle = {}
        for name in SHUFFLES:
            shuffled = ph.metrics(fuse_severity(p, tables["shuffled"][s][name], f"severity fusion, {name} shuffled"))
            old = ph.metrics(pf.fuse(p, tables["shuffled"][s][name]))
            shuffle[name] = {"severity_dqwk": shuffled["qwk"] - metrics["severity"]["qwk"],
                             "fusion_5050_dqwk": old["qwk"] - metrics["fusion_5050"]["qwk"],
                             "pathology_dqwk": ph.metrics(tables["shuffled"][s][name])["qwk"] - metrics["pathology"]["qwk"]}
        checks = ph.threshold_check(metrics["severity"], metrics["p"])
        dep = shuffle["lesions"]["severity_dqwk"]
        per_seed[s] = {"control_seed": pf.control_seed(s), "models": {k: model_row(v) for k, v in metrics.items()},
                       "confusion_matrix": {k: v["confusion_matrix"] for k, v in metrics.items()},
                       "delta": deltas, "shuffle": shuffle,
                       "matched_seed": {"reference": f"P-{s}", "checks": checks,
                                        "performance_maintained": bool(all(c["passed"] for c in checks.values())),
                                        "lesion_shuffle_dqwk": dep, "lesion_dependence": bool(dep <= -ph.SHUFFLE_MIN_DROP + 1e-12)},
                       "grade4_cap_active": int(np.sum(models["severity"]["p_gt"][:, 3] < p["p_gt"][:, 3] - 1e-12)),
                       "grade4_calls": {k: int(np.sum(v["pred"] == 4)) for k, v in models.items()}}

    def stat(values):
        x = np.asarray(values, np.float64)
        return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "per_seed": [float(v) for v in x]}

    summary = {"models": {m: {k: stat([per_seed[s]["models"][m][k] for s in SEEDS]) for k in pf.HEADLINE_KEYS}
                          for m in ("p", "pathology", "fusion_5050", "severity", "control_severity")},
               "delta": {}, "shuffle": {n: {k: stat([per_seed[s]["shuffle"][n][k] for s in SEEDS])
                                            for k in ("severity_dqwk", "fusion_5050_dqwk", "pathology_dqwk")} for n in SHUFFLES}}
    for name in comparisons:
        summary["delta"][name] = {}
        for k in list(ph.METRICS) + ["grade4_recall"]:
            summary["delta"][name][k] = stat([per_seed[s]["delta"][name][k]["delta"] for s in SEEDS])
        for k in ph.BOOT_KEYS:
            lo, hi = np.nanpercentile(np.mean(np.stack(draws[name][k], 0), 0), [2.5, 97.5])
            summary["delta"][name][k].update(ci_low=float(lo), ci_high=float(hi))
    summary["interpretation_54_descriptive_only"] = ph.interpret(per_seed, "the severity-aware fused model")
    return {"rule": RULE, "n": int(len(grades)), "n_boot": int(n_boot), "boot_seed": ph.BOOT_SEED, "seeds": list(SEEDS),
            "per_seed": per_seed, "summary": summary,
            "statements": {
                "post_hoc": "Descriptive and post hoc: the rule was designed after the stored predictions had been inspected.",
                "control": "The P + P control pairs overlap (each P seed is in two of them); they are not independent.",
                "scope": "No superiority over P is claimed. No other fusion rule was evaluated. The IDRiD grading test is untouched."}}


def _f(v, signed=False):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:+.4f}" if signed else f"{v:.4f}")


def markdown(r):
    names = {"p": "P alone", "pathology": "pathology grader alone", "fusion_5050": "recorded 50 / 50 fusion",
             "severity": "severity-aware fusion (locked rule)", "control_severity": "P + P control, same rule"}
    rows = (("qwk", "QWK"), ("auroc_ge1", "AUROC ≥1"), ("auroc_ge2", "AUROC ≥2"), ("auroc_ge3", "AUROC ≥3"),
            ("auroc_ge4", "AUROC ≥4"), ("grade3_recall", "grade-3 recall"), ("grade4_recall", "grade-4 recall"),
            ("false_urgent_rate", "false-urgent rate"))
    s = r["summary"]
    lines = ["# Severity-aware fixed fusion of P and the pathology grader -- seeds 42, 123, 2026", "",
             "Rule: fused_k = 0.5·P_k + 0.5·Path_k for k = 0, 1, 2; fused_3 = min(P_3, fused_2); grade = #{k: fused_k > 0.5}.",
             r["statements"]["post_hoc"], r["statements"]["control"], r["statements"]["scope"], ""]
    for m, label in names.items():
        lines += [f"## {label}", "", "| | " + " | ".join(f"seed {x}" for x in SEEDS) + " | mean | SD |", "|---|" + "---|" * 5]
        for key, name in rows:
            st = s["models"][m][key]
            lines.append(f"| {name} | " + " | ".join(_f(v) for v in st["per_seed"]) + f" | {_f(st['mean'])} | {_f(st['sd'])} |")
        lines.append("")
    lines += ["## Matched-seed differences", "", "| | " + " | ".join(f"seed {x}" for x in SEEDS) + " | mean | SD | 95 % CI of the mean |",
              "|---|" + "---|" * 6]
    labels = {"severity_minus_p": "severity − P", "severity_minus_5050": "severity − recorded 50 / 50",
              "severity_minus_control": "severity − P + P control"}
    keys = (("qwk", "QWK"), ("auroc_ge3_g4_vs_g012", "AUROC ≥3 (4 vs 0–2)"), ("mean_cut_auroc", "mean AUROC, 4 cuts"),
            ("grade3_recall", "grade-3 recall"), ("grade4_recall", "grade-4 recall"), ("false_urgent_rate", "false-urgent rate"))
    for name, label in labels.items():
        for key, klabel in keys:
            st = s["delta"][name][key]
            ci = f"{_f(st['ci_low'], True)} to {_f(st['ci_high'], True)}" if "ci_low" in st else "—"
            lines.append(f"| {label}: Δ {klabel} | " + " | ".join(_f(v, True) for v in st["per_seed"])
                         + f" | {_f(st['mean'], True)} | {_f(st['sd'])} | {ci} |")
    lines += ["", "## Shuffle tests (QWK with the pathology branch's input shuffled − QWK intact)", "",
              "| input shuffled | model | " + " | ".join(f"seed {x}" for x in SEEDS) + " | mean | SD |", "|---|---|" + "---|" * 5]
    for n in SHUFFLES:
        for key, label in (("severity_dqwk", "severity-aware fusion"), ("fusion_5050_dqwk", "recorded 50 / 50"),
                           ("pathology_dqwk", "pathology grader alone")):
            st = s["shuffle"][n][key]
            lines.append(f"| {n} | {label} | " + " | ".join(_f(v, True) for v in st["per_seed"]) + f" | {_f(st['mean'], True)} | {_f(st['sd'])} |")
    i = s["interpretation_54_descriptive_only"]
    lines += ["", f"§54 category (descriptive only): {i['category']} — {i['title']}. Performance maintained in seeds "
              f"{i['performance_maintained_seeds'] or 'none'}; lesion dependence in seeds {i['lesion_dependence_seeds'] or 'none'}.",
              "", "Grade-4 calls per seed (P / 50-50 / severity): "
              + "; ".join(f"{x}: {r['per_seed'][x]['grade4_calls']['p']} / {r['per_seed'][x]['grade4_calls']['fusion_5050']} / "
                          f"{r['per_seed'][x]['grade4_calls']['severity']}" for x in SEEDS)
              + ". Images where the cap lowered P(≥4): " + ", ".join(str(r["per_seed"][x]["grade4_cap_active"]) for x in SEEDS) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", required=True, help="local copy of the Drive experiments root")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=ph.N_BOOT)
    args = parser.parse_args(argv)
    tables, files, grades = prerun_audit(args.experiments)
    result = evaluate(tables, grades, args.n_boot)
    result["files"] = files
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(ph._jsonable(result), fh, indent=1)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(markdown(result))
    print("written:", args.out)
    return result


if __name__ == "__main__":
    main()
