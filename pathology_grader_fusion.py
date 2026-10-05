"""Fixed late fusion of the pathology-only grader with the stored RGB grader P, and its reporting.

THE FUSION RULE (declared here and written into every run's config.json BEFORE training; never tuned):

    p_k(image) = P(grade > k),  k = 0..3        -- CORN's cumulative probability (the `p_gt_k` column)
    fused_k    = 0.5 * p_k[P, seed s, BEST] + 0.5 * p_k[pathology grader, seed s, BEST]
    grade      = number of k with fused_k > 0.5                     -- CORN's own decode rule

Two members, equal weights, no threshold other than CORN's 0.5, no fitting of any kind.

CONTROL (so that a gain from averaging two models is not credited to the pathology branch): the same
arithmetic on two RGB graders, P seed s and P seed next(s) in the fixed cycle 42 -> 123 -> 2026 -> 42.

Reporting follows research record §54: every comparison is matched by seed against the stored P run, on
the same validation images in the same order, with the tolerances already in the record. Nothing here
defines a new success threshold. numpy / pandas only -- no model, no GPU.
"""
import glob
import os

import numpy as np

import arch1_posthoc as ph

SEEDS = ph.SEEDS
FUSION = {
    "rule": "mean of the cumulative CORN probabilities P(grade > k), k = 0..3, then grade = #{k: fused_k > 0.5}",
    "members": ["P (RGB ConvNeXt-Tiny), same seed, BEST checkpoint", "pathology grader, same seed, BEST checkpoint"],
    "weights": [0.5, 0.5],
    "decode_threshold": 0.5,
    "tuned_on_validation": False,
    "control": "the same rule on P seed s and P seed next(s), cycle 42 -> 123 -> 2026 -> 42",
}
SHUFFLES_FOR_FUSION = ("lesions", "vessel")


def control_seed(seed):
    """The other P seed used in the ensembling control."""
    return SEEDS[(SEEDS.index(int(seed)) + 1) % len(SEEDS)]


def decode(p_gt):
    """CORN's decode rule on cumulative probabilities (corn.decode_logits: grade = sum_k [p_cum_k > 0.5])."""
    return np.sum(np.asarray(p_gt, np.float64) > FUSION["decode_threshold"], axis=-1).astype(int)


def assert_aligned(a, b, what):
    """Two per-sample tables must be the same images, in the same order, with the same grades."""
    if len(a["ids"]) != len(b["ids"]) or not np.array_equal(a["ids"], b["ids"]):
        raise RuntimeError(f"{what}: the two prediction tables are not the same images in the same order")
    if not np.array_equal(a["grade"], b["grade"]):
        raise RuntimeError(f"{what}: the two prediction tables disagree on the true grades")


def fuse(a, b, what="fusion"):
    """The fixed rule on two aligned per-sample tables (arch1_posthoc.table_arrays). Returns a table."""
    assert_aligned(a, b, what)
    w = FUSION["weights"]
    p = w[0] * a["p_gt"].astype(np.float64) + w[1] * b["p_gt"].astype(np.float64)
    return {"ids": a["ids"], "grade": a["grade"], "p_gt": p, "pred": decode(p)}


def p_prediction_path(experiments_root, seed, which="best"):
    """The stored P run's per-sample table for one seed (exactly one P experiment must exist)."""
    pattern = os.path.join(experiments_root, "PL_ConvNeXtPriors", "*", ph.ARMS["P"], f"seed_{int(seed)}", "evaluation",
                           f"per_sample_{which}.csv")
    found = sorted(glob.glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one stored P table for seed {seed}, found {found}")
    return found[0]


def headline(m):
    """The reported endpoints of one model on the validation set."""
    return {"qwk": m["qwk"], "auroc_ge1": m["auroc_cuts"]["ge1"], "auroc_ge2": m["auroc_cuts"]["ge2"],
            "auroc_ge3": m["auroc_cuts"]["ge3"], "auroc_ge4": m["auroc_cuts"]["ge4"],
            "mean_cut_auroc": m["mean_cut_auroc"], "auroc_ge3_g4_vs_g012": m["auroc_ge3_g4_vs_g012"],
            "grade3_recall": m["grade3_recall"], "grade4_recall": m["recall_per_grade"][4],
            "false_urgent_rate": m["false_urgent_rate"], "mae": m["mae"]}


HEADLINE_KEYS = ("qwk", "auroc_ge1", "auroc_ge2", "auroc_ge3", "auroc_ge4", "mean_cut_auroc", "auroc_ge3_g4_vs_g012",
                 "grade3_recall", "grade4_recall", "false_urgent_rate", "mae")


def seed_report(seed, pathology, p_tables, shuffled=None, indices=None):
    """Everything reported for one seed.

    pathology  per-sample table of the pathology grader, BEST (arch1_posthoc.table_arrays)
    p_tables   {seed: per-sample table of the stored P run, BEST} for all three seeds
    shuffled   {name: per-sample table of the pathology grader with that input shuffled}; "lesions" and
               "vessel" are fused as well, to measure whether the FUSED grade depends on them
    indices    bootstrap resamples (arch1_posthoc.bootstrap_indices) for the paired intervals
    """
    seed = int(seed)
    p_own, p_other = p_tables[seed], p_tables[control_seed(seed)]
    fused = fuse(p_own, pathology, f"seed {seed}: P + pathology")
    control = fuse(p_own, p_other, f"seed {seed}: P + P control")
    m = {"pathology": ph.metrics(pathology), "p": ph.metrics(p_own), "fused": ph.metrics(fused),
         "control": ph.metrics(control)}
    report = {
        "seed": seed, "control_seed": control_seed(seed), "n": int(len(pathology["grade"])),
        "models": {name: headline(v) for name, v in m.items()},
        "recall_per_grade": {name: v["recall_per_grade"] for name, v in m.items()},
        "confusion_matrix": {name: v["confusion_matrix"] for name, v in m.items()},
        "delta_vs_p": {"fused": ph.paired_delta(fused, p_own, indices=indices),
                       "control": ph.paired_delta(control, p_own, indices=indices),
                       "pathology": ph.paired_delta(pathology, p_own, indices=indices)},
        "delta_fused_vs_control": ph.paired_delta(fused, control, indices=indices),
    }
    for block in list(report["delta_vs_p"].values()) + [report["delta_fused_vs_control"]]:
        for value in block.values():
            value.pop("_draws", None)
    checks = ph.threshold_check(m["fused"], m["p"])
    report["matched_seed"] = {"reference": f"P-{seed}", "checks": checks,
                              "performance_maintained": bool(all(c["passed"] for c in checks.values()))}
    report["pathology_shuffle"], report["fused_shuffle"] = {}, {}
    for name, table in (shuffled or {}).items():
        assert_aligned(pathology, table, f"seed {seed}: shuffle {name}")
        sm = ph.metrics(table)
        report["pathology_shuffle"][name] = {"qwk": sm["qwk"], "dqwk": sm["qwk"] - m["pathology"]["qwk"],
                                             "d_mean_cut_auroc": sm["mean_cut_auroc"] - m["pathology"]["mean_cut_auroc"]}
        if name in SHUFFLES_FOR_FUSION:
            fm = ph.metrics(fuse(p_own, table, f"seed {seed}: P + pathology[{name} shuffled]"))
            report["fused_shuffle"][name] = {"qwk": fm["qwk"], "dqwk": fm["qwk"] - m["fused"]["qwk"]}
    if "lesions" in report["fused_shuffle"]:
        dqwk = report["fused_shuffle"]["lesions"]["dqwk"]
        report["matched_seed"].update(lesion_shuffle_dqwk=dqwk,
                                      lesion_dependence=bool(dqwk <= -ph.SHUFFLE_MIN_DROP + 1e-12))
    return report


def _stat(values):
    x = np.asarray(values, np.float64)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "per_seed": [float(v) for v in x]}


def summarise(reports):
    """Three-seed mean +/- SD of every reported quantity, and the §54 reading of the FUSED model against P
    (the two per-seed facts and the category, exactly as arch1_posthoc.interpret defines them)."""
    reports = {int(k): v for k, v in reports.items()}
    if tuple(sorted(reports)) != tuple(sorted(SEEDS)):
        raise RuntimeError(f"the summary needs all of seeds {SEEDS}; have {sorted(reports)}")
    out = {"seeds": list(SEEDS), "fusion": FUSION, "n": reports[SEEDS[0]]["n"],
           "models": {name: {k: _stat([reports[s]["models"][name][k] for s in SEEDS]) for k in HEADLINE_KEYS}
                      for name in ("pathology", "p", "fused", "control")},
           "delta_vs_p": {name: {k: _stat([reports[s]["delta_vs_p"][name][k]["delta"] for s in SEEDS]) for k in ph.METRICS}
                          for name in ("fused", "control", "pathology")},
           "delta_fused_vs_control": {k: _stat([reports[s]["delta_fused_vs_control"][k]["delta"] for s in SEEDS])
                                      for k in ph.METRICS}}
    names = sorted(set.intersection(*[set(reports[s]["pathology_shuffle"]) for s in SEEDS]))
    out["pathology_shuffle_dqwk"] = {n: _stat([reports[s]["pathology_shuffle"][n]["dqwk"] for s in SEEDS]) for n in names}
    fused_names = sorted(set.intersection(*[set(reports[s]["fused_shuffle"]) for s in SEEDS]))
    out["fused_shuffle_dqwk"] = {n: _stat([reports[s]["fused_shuffle"][n]["dqwk"] for s in SEEDS]) for n in fused_names}
    if all("lesion_dependence" in reports[s]["matched_seed"] for s in SEEDS):
        out["interpretation_fused_vs_p"] = ph.interpret({s: reports[s] for s in SEEDS}, "the fused dual-branch model")
    out["statements"] = {
        "fusion": "The fusion rule was fixed before training and is not tuned: " + FUSION["rule"] + ".",
        "control": "Any difference between the fused model and P has to be read against the P + P control, which "
                   "uses the same arithmetic without the pathology branch.",
        "scope": "Descriptive. Matched by seed against the stored P runs (record §54). No new success threshold is "
                 "defined here, no seed is singled out, and no superiority is claimed.",
    }
    return out


def _f(value, signed=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def summary_markdown(summary, reports):
    reports = {int(k): v for k, v in reports.items()}
    labels = {"pathology": "pathology grader alone", "p": "P (RGB) alone", "fused": "P + pathology (fixed fusion)",
              "control": "P + P control"}
    rows = (("qwk", "QWK"), ("auroc_ge1", "AUROC ≥1"), ("auroc_ge2", "AUROC ≥2"), ("auroc_ge3", "AUROC ≥3"),
            ("auroc_ge4", "AUROC ≥4"), ("grade3_recall", "grade-3 recall"), ("grade4_recall", "grade-4 recall"),
            ("false_urgent_rate", "false-urgent rate"))
    lines = ["# Dual-branch: pathology grader and fixed late fusion with P -- seeds 42, 123, 2026", "",
             summary["statements"]["fusion"], summary["statements"]["control"], summary["statements"]["scope"], ""]
    for name in ("pathology", "p", "fused", "control"):
        lines += [f"## {labels[name]}", "", "| | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | SD |",
                  "|---|" + "---|" * (len(SEEDS) + 2)]
        for key, label in rows:
            st = summary["models"][name][key]
            lines.append(f"| {label} | " + " | ".join(_f(v) for v in st["per_seed"]) + f" | {_f(st['mean'])} | {_f(st['sd'])} |")
        lines.append("")
    lines += ["## Matched-seed differences against P of the same seed", "",
              "| | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | SD |", "|---|" + "---|" * (len(SEEDS) + 2)]
    for name, label in (("fused", "fused − P"), ("control", "P + P control − P")):
        for key in ("qwk", "auroc_ge3_g4_vs_g012", "grade3_recall", "false_urgent_rate"):
            st = summary["delta_vs_p"][name][key]
            lines.append(f"| {label}: Δ {ph.LABELS[key]} | " + " | ".join(_f(v, True) for v in st["per_seed"])
                         + f" | {_f(st['mean'], True)} | {_f(st['sd'])} |")
    for key in ("qwk", "auroc_ge3_g4_vs_g012"):
        st = summary["delta_fused_vs_control"][key]
        lines.append(f"| fused − control: Δ {ph.LABELS[key]} | " + " | ".join(_f(v, True) for v in st["per_seed"])
                     + f" | {_f(st['mean'], True)} | {_f(st['sd'])} |")
    lines += ["", "## Shuffle tests (QWK with the input shuffled across validation images − QWK intact)", "",
              "| input shuffled | model | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | SD |",
              "|---|---|" + "---|" * (len(SEEDS) + 2)]
    for n, st in summary["pathology_shuffle_dqwk"].items():
        lines.append(f"| {n} | pathology grader | " + " | ".join(_f(v, True) for v in st["per_seed"])
                     + f" | {_f(st['mean'], True)} | {_f(st['sd'])} |")
    for n, st in summary["fused_shuffle_dqwk"].items():
        lines.append(f"| {n} | fused | " + " | ".join(_f(v, True) for v in st["per_seed"])
                     + f" | {_f(st['mean'], True)} | {_f(st['sd'])} |")
    lines += ["", "## Tolerances already in the record, fused model against P of the same seed (§54)", "",
              "| seed | Δ QWK (≥ −0.02) | Δ AUROC ≥3 (≥ −0.01) | Δ grade-3 recall (≥ −0.10) | Δ false-urgent (≤ +0.02) | "
              "performance maintained | fused lesion-shuffle ΔQWK (≤ −0.01) | lesion dependence |", "|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        m, d = reports[s]["matched_seed"], reports[s]["delta_vs_p"]["fused"]
        cell = lambda k: f"{_f(d[k]['delta'], True)} {'PASS' if m['checks'][k]['passed'] else 'FAIL'}"      # noqa: E731
        lines.append(f"| {s} | {cell('qwk')} | {cell('auroc_ge3_g4_vs_g012')} | {cell('grade3_recall')} | "
                     f"{cell('false_urgent_rate')} | {'yes' if m['performance_maintained'] else 'no'} | "
                     f"{_f(m.get('lesion_shuffle_dqwk'), True)} | "
                     f"{'n/a' if 'lesion_dependence' not in m else ('yes' if m['lesion_dependence'] else 'no')} |")
    i = summary.get("interpretation_fused_vs_p")
    if i:
        lines += ["", f"§54 category for the fused model: **{i['category']} — {i['title']}**. {i['rule']}", "", i["statement"]]
    return "\n".join(lines) + "\n"
