"""Post-run report for a three-seed run (Architecture 1, or any run evaluated the same way). Laptop, CPU,
read-only: it reads the run directories copied from Drive (no checkpoints needed) and the P / PL evaluation
files, and writes a report plus a draft research-record section. Nothing is trained, tuned or launched.

How a three-seed result is read (research record §54, fixed before seeds 123 and 2026 finished):
  1. PRIMARY: the matched-seed comparison -- each seed against the P run of the SAME seed, on the same 730
     validation images, with the same metric code: delta QWK, delta AUROC (>=3), delta grade-3 recall, delta
     false-urgent rate and the lesion-shuffle delta QWK, per seed and as mean +/- SD over the three seeds.
  2. INTERPRETATION: one of four categories (A-D) from two per-seed facts -- does the model depend on the
     lesion maps, and is matched-seed performance maintained -- using only tolerances already in the record.
     A category is a reading of the evidence. It never launches an experiment.
  3. The run's own five checks against P-42 are shown as recorded, under the label
     "P-42-derived absolute reference bounds; not a multi-seed success criterion" (P's own seeds 123 and
     2026 do not pass them).
Everything here is descriptive. No superiority over P is claimed, and no seed is singled out.

Copy the inputs first (small files only):
    rclone copy gdrive:DiabeticRetinopathy/experiments/Architecture1 results/Architecture1 --exclude "*/checkpoints/**" --exclude "*/logs/**"
    rclone copy gdrive:DiabeticRetinopathy/experiments/PL_ConvNeXtPriors/2026-09-28_05-22-44 \
        results/PL_ConvNeXtPriors/2026-09-28_05-22-44 --exclude "*/checkpoints/**" --exclude "*/logs/**"
Run:
    python arch1_posthoc.py --runs results/Architecture1 --baseline results/PL_ConvNeXtPriors/2026-09-28_05-22-44
"""
import argparse
import datetime
import glob
import json
import os

import numpy as np

SEEDS = (42, 123, 2026)
N_BOOT = 2000
BOOT_SEED = 20260927                      # the C1 / P-PL bootstrap seed (record §20, §22)
CUTS = (1, 2, 3, 4)
ARMS = {"P": "P_convnext_rgb", "PL": "PL_convnext_rgb_priors"}
#: The tolerances already in the record (arch1_train.ONE_SEED_CHECKS, pl_convnext.GUARDRAILS). No new number
#: is introduced: they are applied against P of the same seed (matched) and, as recorded by the run, against P-42.
TOLERANCE = {"qwk": ("min", -0.02), "auroc_ge3_g4_vs_g012": ("min", -0.01), "grade3_recall": ("min", -0.10),
             "false_urgent_rate": ("max", 0.02)}
SHUFFLE_MIN_DROP = 0.01
MATCHED_KEYS = ("qwk", "auroc_ge3_g4_vs_g012", "grade3_recall", "false_urgent_rate")
ABSOLUTE_LABEL = "P-42-derived absolute reference bounds; not a multi-seed success criterion"
#: The four readings of a three-seed result (record §54). "dependence" = lesion-shuffle QWK drop >= 0.01 in
#: that seed; "maintained" = all four matched-seed deltas within the tolerances in that seed.
CATEGORIES = {
    "A": ("consistent lesion dependence, matched-seed performance maintained",
          "Every seed depends on the lesion maps and stays within the tolerances of P at the same seed. "
          "{label} has evidence worth pursuing. Whether any further mechanism is scientifically necessary is a "
          "separate decision; nothing follows automatically."),
    "B": ("lesion dependence present, grading consistently worse",
          "Every seed depends on the lesion maps, and every seed falls outside the tolerances of P at the same "
          "seed. The maps carry information the model uses, but the fusion is harmful as built. Investigate or "
          "close {label} before adding any mechanism."),
    "C": ("no lesion dependence",
          "No seed depends on the lesion maps: they are not contributing meaningfully and the model is "
          "effectively P. {label} is closed as a way of using them."),
    "D": ("inconsistent across seeds",
          "The seeds disagree on lesion dependence or on matched-seed performance. This is reported as "
          "instability. No seed is singled out and no mechanism is added on this basis."),
}
METRICS = ("qwk", "auroc_ge3_g4_vs_g012", "mean_cut_auroc", "grade3_recall", "false_urgent_rate", "mae")
LABELS = {"qwk": "QWK", "auroc_ge3_g4_vs_g012": "AUROC (>=3; grade 4 vs 0-2)", "mean_cut_auroc": "mean AUROC, 4 cuts",
          "grade3_recall": "grade-3 recall", "false_urgent_rate": "false-urgent rate", "mae": "MAE",
          "lesion_shuffle": "lesion-shuffle QWK drop"}


# --------------------------------------------------------------------------- metrics on per-sample tables

def _auroc(positive, score):
    from sklearn.metrics import roc_auc_score
    positive = np.asarray(positive, bool)
    if positive.all() or not positive.any():
        return float("nan")
    return float(roc_auc_score(positive, score))


def table_arrays(frame):
    """The columns the metrics need, as arrays (multiseed per-sample schema)."""
    return {"ids": frame["image_id"].astype(str).to_numpy(), "grade": frame["true_grade"].to_numpy(int),
            "pred": frame["predicted_grade"].to_numpy(int),
            "p_gt": np.stack([frame[f"p_gt_{k}"].to_numpy(np.float64) for k in range(4)], 1)}


def metrics(arrays, idx=None):
    """The endpoints from one per-sample table (optionally on a resample `idx`). Definitions follow
    icdr_two_route_experiment.head_metrics / arch1_train.metrics_from_logits."""
    from sklearn.metrics import cohen_kappa_score
    g, pred, p = arrays["grade"], arrays["pred"], arrays["p_gt"]
    if idx is not None:
        g, pred, p = g[idx], pred[idx], p[idx]
    low, sel = g <= 2, g != 3
    cuts = [_auroc(g >= c, p[:, c - 1]) for c in CUTS]
    return {"qwk": float(cohen_kappa_score(g, pred, weights="quadratic", labels=list(range(5)))),
            "auroc_ge3_g4_vs_g012": _auroc(g[sel] == 4, p[sel, 2]),
            "mean_cut_auroc": float(np.mean(cuts)),
            "auroc_cuts": {f"ge{c}": v for c, v in zip(CUTS, cuts)},
            "grade3_recall": float(np.mean(pred[g == 3] == 3)) if np.any(g == 3) else float("nan"),
            "false_urgent_rate": float(np.mean(pred[low] >= 3)) if low.any() else float("nan"),
            "mae": float(np.mean(np.abs(pred - g))),
            "recall_per_grade": {int(k): float(np.mean(pred[g == k] == k)) if np.any(g == k) else float("nan")
                                 for k in range(5)},
            "confusion_matrix": [[int(np.sum((g == a) & (pred == b))) for b in range(5)] for a in range(5)]}


def _rank_auroc(positive, score):
    """AUROC by the rank-sum formula (ties get average ranks) -- the same value as sklearn's, without its
    per-call overhead; used inside the bootstrap."""
    from scipy.stats import rankdata
    n_pos = int(positive.sum())
    n_neg = positive.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((rankdata(score)[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


_QWK_WEIGHTS = (np.arange(5)[:, None] - np.arange(5)[None, :]) ** 2 / 16.0


def _fast_qwk(g, pred):
    observed = np.bincount(g * 5 + pred, minlength=25).reshape(5, 5).astype(np.float64)
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    denominator = float((_QWK_WEIGHTS * expected).sum())
    return 1.0 - float((_QWK_WEIGHTS * observed).sum()) / denominator if denominator else float("nan")


def bootstrap_metrics(arrays, idx):
    """The three bootstrapped endpoints on one resample (numpy only; equal to `metrics` -- tested)."""
    g, pred, p = arrays["grade"][idx], arrays["pred"][idx], arrays["p_gt"][idx]
    sel = g != 3
    return {"qwk": _fast_qwk(g, pred), "auroc_ge3_g4_vs_g012": _rank_auroc(g[sel] == 4, p[sel, 2]),
            "mean_cut_auroc": float(np.mean([_rank_auroc(g >= c, p[:, c - 1]) for c in CUTS]))}


BOOT_KEYS = ("qwk", "auroc_ge3_g4_vs_g012", "mean_cut_auroc")


def bootstrap_indices(grades, n_boot=N_BOOT, seed=BOOT_SEED):
    """Paired resamples: images drawn with replacement WITHIN every grade (as stage4_v2_c2)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(grades == g) for g in np.unique(grades)]
    return [np.concatenate([rng.choice(g, g.size) for g in groups]) for _ in range(n_boot)]


def paired_delta(a, b, keys=BOOT_KEYS, indices=None):
    """a - b on the same images: point estimates and percentile 95 % intervals. `a`, `b`: table_arrays."""
    if not np.array_equal(a["ids"], b["ids"]) or not np.array_equal(a["grade"], b["grade"]):
        raise RuntimeError("the two tables are not the same images in the same order")
    ma, mb = metrics(a), metrics(b)
    out = {k: {"delta": ma[k] - mb[k]} for k in METRICS}
    if indices is not None:
        draws = {k: [] for k in keys}
        for idx in indices:
            da, db = bootstrap_metrics(a, idx), bootstrap_metrics(b, idx)
            for k in keys:
                draws[k].append(da[k] - db[k])
        for k in keys:
            lo, hi = np.nanpercentile(draws[k], [2.5, 97.5])
            out[k].update(ci_low=float(lo), ci_high=float(hi))
            out[k]["_draws"] = np.asarray(draws[k], np.float64)
    return out


# --------------------------------------------------------------------------- loading

def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_table(path):
    import pandas as pd
    return table_arrays(pd.read_csv(path, dtype={"image_id": str}))


def find_runs(runs_root, prefix=None):
    """{seed: run_dir} and the summary dir for one experiment prefix (e.g. arch1_cb5fc7a8d370)."""
    summaries = sorted(glob.glob(os.path.join(runs_root, f"{prefix or '*'}_3seed_summary")))
    if len(summaries) != 1:
        raise RuntimeError(f"expected exactly one *_3seed_summary under {runs_root} (prefix {prefix!r}); found {summaries}")
    stem = summaries[0][:-len("_3seed_summary")]
    runs = {seed: f"{stem}_seed{seed}" for seed in SEEDS}
    missing = [d for d in runs.values() if not os.path.exists(os.path.join(d, "verdict.json"))]
    if missing:
        raise RuntimeError(f"seed runs without a verdict: {missing}")
    return runs, summaries[0]


def load_run(run_dir):
    history = [_read_json(p) for p in sorted(glob.glob(os.path.join(run_dir, "history", "epoch_*.json")))]
    return {"dir": run_dir, "config": _read_json(os.path.join(run_dir, "config.json")),
            "verdict": _read_json(os.path.join(run_dir, "verdict.json")),
            "metrics_best": _read_json(os.path.join(run_dir, "metrics", "metrics_best.json")),
            "metrics_last": _read_json(os.path.join(run_dir, "metrics", "metrics_last.json")),
            "best": _read_table(os.path.join(run_dir, "metrics", "per_sample_best.csv")),
            "last": _read_table(os.path.join(run_dir, "metrics", "per_sample_last.csv")), "history": history}


def load_baseline(root, arm, seed):
    base = os.path.join(root, ARMS[arm], f"seed_{seed}")
    history = [_read_json(p) for p in sorted(glob.glob(os.path.join(base, "history", "epoch_*.json")))]
    return {"dir": base, "stored": _read_json(os.path.join(base, "evaluation", "metrics_best.json")),
            "best": _read_table(os.path.join(base, "evaluation", "per_sample_best.csv")),
            "last": _read_table(os.path.join(base, "evaluation", "per_sample_last.csv")), "history": history}


def check_stored(recomputed, stored, where, tol=1e-9):
    """This script's metric code must reproduce what the run itself stored."""
    for key in ("qwk", "auroc_ge3_g4_vs_g012", "grade3_recall", "false_urgent_rate"):
        if abs(recomputed[key] - stored[key]) > tol:
            raise RuntimeError(f"{where}: recomputed {key} {recomputed[key]} != stored {stored[key]}")


def history_summary(history, best_epoch_index):
    """Epochs run, the BEST epoch, and the train - validation QWK gap there (over-fitting indicator)."""
    if not history:
        return None
    by_epoch = {h["epoch"]: h for h in history}
    best = by_epoch.get(None if best_epoch_index is None else best_epoch_index + 1)
    return {"epochs_run": max(by_epoch), "best_epoch_index": best_epoch_index,
            "val_qwk_at_best": None if best is None else best.get("val_QWK"),
            "train_qwk_at_best": None if best is None else best.get("QWK"),
            "train_minus_val_qwk_at_best": None if best is None or best.get("QWK") is None
            else best["QWK"] - best["val_QWK"],
            "final_learning_rate": history[-1].get("learning_rate")}


# --------------------------------------------------------------------------- analysis

def threshold_check(values, reference):
    """The four metric checks for `values` against `reference` with the pre-set tolerances."""
    out = {}
    for key, (kind, bound) in TOLERANCE.items():
        limit = reference[key] + bound
        out[key] = {"value": values[key], "limit": limit,
                    "passed": bool(values[key] >= limit - 1e-12 if kind == "min" else values[key] <= limit + 1e-12)}
    return out


def interpret(per_seed, label="Architecture 1"):
    """The §54 category from the per-seed facts. Returns the category, its statement and the evidence."""
    dependent = [s for s in SEEDS if per_seed[s]["matched_seed"]["lesion_dependence"]]
    maintained = [s for s in SEEDS if per_seed[s]["matched_seed"]["performance_maintained"]]
    n = len(SEEDS)
    if not dependent:
        category = "C"
    elif len(dependent) == n and len(maintained) == n:
        category = "A"
    elif len(dependent) == n and not maintained:
        category = "B"
    else:
        category = "D"
    title, statement = CATEGORIES[category]
    return {"category": category, "title": title, "statement": statement.format(label=label),
            "lesion_dependence_seeds": dependent, "performance_maintained_seeds": maintained,
            "rule": "C: dependence in 0/3 seeds. A: dependence 3/3 and maintained 3/3. B: dependence 3/3 and "
                    "maintained 0/3. D: anything else."}


def analyse(runs_root, baseline_root, prefix=None, n_boot=N_BOOT, log=print):
    runs, summary_dir = find_runs(runs_root, prefix)
    summary = _read_json(os.path.join(summary_dir, "summary.json"))
    label = summary.get("label", "Architecture 1")
    data = {seed: load_run(d) for seed, d in runs.items()}
    base = {arm: {seed: load_baseline(baseline_root, arm, seed) for seed in SEEDS} for arm in ARMS}
    for arm in ARMS:
        for seed in SEEDS:
            check_stored(metrics(base[arm][seed]["best"]), base[arm][seed]["stored"], f"{arm}-{seed}")
    for seed in SEEDS:
        check_stored(metrics(data[seed]["best"]), data[seed]["metrics_best"]["metrics"], f"{label} seed {seed}")
    grades = data[SEEDS[0]]["best"]["grade"]
    indices = bootstrap_indices(grades, n_boot)
    p42 = metrics(base["P"][42]["best"])

    per_seed, draws = {}, {arm: {k: [] for k in BOOT_KEYS} for arm in ARMS}
    for seed in SEEDS:
        log(f"seed {seed}: metrics and {n_boot} paired resamples")
        run = data[seed]
        best, last = metrics(run["best"]), metrics(run["last"])
        verdict = run["verdict"]
        versus = {}
        for arm in ARMS:
            d = paired_delta(run["best"], base[arm][seed]["best"], indices=indices)
            for k in draws[arm]:
                draws[arm][k].append(d[k].pop("_draws"))
            versus[arm] = d
        permutation = run["metrics_best"]["permutation"]
        shuffle_dqwk = float(permutation["pathology"]["dqwk"])
        matched = threshold_check(best, metrics(base["P"][seed]["best"]))
        per_seed[seed] = {
            "absolute": {"status": verdict["status"], "failed_criteria": verdict["failed_criteria"],
                         "criteria": {k: {kk: vv for kk, vv in c.items() if kk != "required"}
                                      for k, c in verdict["criteria"].items()}},
            "best": best, "last": last,
            "best_minus_last": {k: best[k] - last[k] for k in METRICS},
            "permutation_dqwk": {k: permutation[k]["dqwk"] for k in ("pathology", "vessel", "both")},
            "history": history_summary(run["history"], run["metrics_best"]["checkpoint"].get("best_epoch")),
            "matched_seed": {"reference": f"P-{seed}", "checks": matched,
                             "performance_maintained": bool(all(c["passed"] for c in matched.values())),
                             "lesion_shuffle_dqwk": shuffle_dqwk,
                             "lesion_dependence": bool(shuffle_dqwk <= -SHUFFLE_MIN_DROP + 1e-12)},
            "versus": versus,
        }

    reference_context = {}
    for arm in ARMS:
        for seed in SEEDS:
            m = metrics(base[arm][seed]["best"])
            checks = threshold_check(m, p42)
            reference_context[f"{arm}-{seed}"] = {
                **{k: m[k] for k in METRICS}, "checks_vs_p42": {k: c["passed"] for k, c in checks.items()},
                "all_metric_checks_pass": bool(all(c["passed"] for c in checks.values())),
                "history": history_summary(base[arm][seed]["history"], base[arm][seed]["stored"].get("best_epoch_index_0based"))}

    def stat(values):
        x = np.asarray(values, np.float64)
        return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "per_seed": [float(v) for v in x]}

    seed_mean = {}
    for arm in ARMS:
        seed_mean[arm] = {}
        for k in METRICS:
            seed_mean[arm][k] = stat([per_seed[s]["versus"][arm][k]["delta"] for s in SEEDS])
            seed_mean[arm][k]["positive_seeds"] = int(sum(per_seed[s]["versus"][arm][k]["delta"] > 0 for s in SEEDS))
        for k, d in draws[arm].items():                       # seed-mean interval: the same resample in every seed
            lo, hi = np.nanpercentile(np.mean(np.stack(d, 0), 0), [2.5, 97.5])
            seed_mean[arm][k].update(ci_low=float(lo), ci_high=float(hi))

    return {
        "label": label, "experiment": summary.get("experiment", "Architecture1"), "runs": runs, "summary_dir": summary_dir,
        "baseline_root": baseline_root, "n_validation": int(len(grades)), "n_boot": int(n_boot), "boot_seed": BOOT_SEED,
        "matched": {"delta": seed_mean["P"],
                    "lesion_shuffle_dqwk": stat([per_seed[s]["matched_seed"]["lesion_shuffle_dqwk"] for s in SEEDS])},
        "interpretation": interpret(per_seed, label),
        "absolute_reference_bounds": {
            "label": ABSOLUTE_LABEL, "tally_recorded_by_the_run": summary["route"],
            "seeds_passing_all": summary["seeds_passing_all"],
            "seeds_passing_each_criterion": summary["seeds_passing_each_criterion"], "stats": summary["stats"]},
        "statements": summary["statements"], "identity": summary["shared_identity"],
        "p42": {k: p42[k] for k in METRICS}, "per_seed": per_seed, "reference_context": reference_context,
        "arm_stats": {k: stat([per_seed[s]["best"][k] for s in SEEDS]) for k in METRICS},
        "baseline_stats": {arm: {k: stat([reference_context[f"{arm}-{s}"][k] for s in SEEDS]) for k in METRICS}
                           for arm in ARMS},
        "seed_mean_delta": seed_mean,
    }


# --------------------------------------------------------------------------- writing

def _f(value, signed=False):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _mark(ok):
    return "PASS" if ok else "FAIL"


def _ci(d):
    return f" ({_f(d['ci_low'], True)} to {_f(d['ci_high'], True)})" if "ci_low" in d else ""


def delta_table(r, arm, keys=METRICS, with_shuffle=False):
    """Matched-seed differences: per seed (with a 95 % interval where bootstrapped), mean, SD."""
    lines = [f"| {r['label']} − {arm}, same seed | " + " | ".join(f"seed {s}" for s in SEEDS)
             + " | mean | SD | 95 % CI of the mean | > 0 in |", "|---|" + "---|" * (len(SEEDS) + 4)]
    for k in keys:
        cells = [_f(r["per_seed"][s]["versus"][arm][k]["delta"], True) + _ci(r["per_seed"][s]["versus"][arm][k])
                 for s in SEEDS]
        m = r["seed_mean_delta"][arm][k]
        interval = f"{_f(m['ci_low'], True)} to {_f(m['ci_high'], True)}" if "ci_low" in m else "—"
        lines.append(f"| Δ {LABELS[k]} | " + " | ".join(cells) + f" | {_f(m['mean'], True)} | {_f(m['sd'])} | {interval} | "
                     f"{m['positive_seeds']}/{len(SEEDS)} |")
    if with_shuffle:
        m = r["matched"]["lesion_shuffle_dqwk"]
        cells = [_f(r["per_seed"][s]["matched_seed"]["lesion_shuffle_dqwk"], True) for s in SEEDS]
        lines.append("| lesion-shuffle ΔQWK (maps permuted − intact) | " + " | ".join(cells)
                     + f" | {_f(m['mean'], True)} | {_f(m['sd'])} | — | — |")
    return lines


def matched_table(r):
    lines = ["| seed | reference | Δ QWK (≥ −0.02) | Δ AUROC ≥3 (≥ −0.01) | Δ grade-3 recall (≥ −0.10) | "
             "Δ false-urgent (≤ +0.02) | performance maintained | lesion-shuffle ΔQWK (≤ −0.01) | lesion dependence |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        m, v = r["per_seed"][s]["matched_seed"], r["per_seed"][s]["versus"]["P"]
        cell = lambda k: f"{_f(v[k]['delta'], True)} {_mark(m['checks'][k]['passed'])}"          # noqa: E731
        lines.append(f"| {s} | {m['reference']} | {cell('qwk')} | {cell('auroc_ge3_g4_vs_g012')} | {cell('grade3_recall')} | "
                     f"{cell('false_urgent_rate')} | {'yes' if m['performance_maintained'] else 'no'} | "
                     f"{_f(m['lesion_shuffle_dqwk'], True)} | {'yes' if m['lesion_dependence'] else 'no'} |")
    return lines


def absolute_table(r):
    a = r["absolute_reference_bounds"]
    lines = ["| check against P-42 | " + " | ".join(f"seed {s}" for s in SEEDS) + " | mean | SD | P-42 | seeds within |",
             "|---|" + "---|" * (len(SEEDS) + 4)]
    for key in ("qwk", "auroc_ge3_g4_vs_g012", "lesion_shuffle", "grade3_recall", "false_urgent_rate"):
        cells = []
        for s in SEEDS:
            c = r["per_seed"][s]["absolute"]["criteria"][key]
            value = c["qwk_drop"] if key == "lesion_shuffle" else c["arch1"]
            cells.append(f"{_f(value, key == 'lesion_shuffle')} {_mark(c['passed'])}")
        st = a["stats"][key]
        lines.append(f"| {LABELS[key]} | " + " | ".join(cells) + f" | {_f(st['mean'])} | {_f(st['sd'])} | {_f(st.get('p42'))} | "
                     f"{a['seeds_passing_each_criterion'][key]}/{len(SEEDS)} |")
    return lines


def context_table(r):
    lines = ["| run | QWK | AUROC (>=3; 4 vs 0-2) | grade-3 recall | false-urgent | all four within |", "|---|---|---|---|---|---|"]
    for name, c in r["reference_context"].items():
        cell = lambda k: f"{_f(c[k])} {_mark(c['checks_vs_p42'][k])}"                # noqa: E731
        lines.append(f"| {name} | {cell('qwk')} | {cell('auroc_ge3_g4_vs_g012')} | {cell('grade3_recall')} | "
                     f"{cell('false_urgent_rate')} | {'yes' if c['all_metric_checks_pass'] else 'no'} |")
    return lines


def behaviour_table(r):
    lines = ["| seed | BEST epoch index | epochs run | train − val QWK at BEST | BEST − LAST QWK | BEST − LAST AUROC | "
             "shuffle lesion | shuffle vessel | shuffle both | recall by grade 0–4 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in SEEDS:
        p = r["per_seed"][s]
        h = p["history"] or {}
        recall = " / ".join(f"{p['best']['recall_per_grade'][g]:.3f}" for g in range(5))
        lines.append(f"| {s} | {h.get('best_epoch_index')} | {h.get('epochs_run')} | {_f(h.get('train_minus_val_qwk_at_best'), True)} | "
                     f"{_f(p['best_minus_last']['qwk'], True)} | {_f(p['best_minus_last']['auroc_ge3_g4_vs_g012'], True)} | "
                     f"{_f(p['permutation_dqwk']['pathology'], True)} | {_f(p['permutation_dqwk']['vessel'], True)} | "
                     f"{_f(p['permutation_dqwk']['both'], True)} | {recall} |")
    return lines


NOT_CONCLUDED = [
    "No superiority over P is claimed: there is no pre-registered superiority criterion, and a mean above zero is not one.",
    "No seed is singled out; checkpoint selection is BEST by validation QWK in every run.",
    "No further experiment follows automatically from the category. Soft targets, EMA and the PL-v2 fallback stay "
    "dormant until a decision is recorded.",
    "The IDRiD grading test is untouched. It is for the finalized downstream model, once, not for a seed.",
]


def _evidence(r):
    i = r["interpretation"]
    return (f"Lesion dependence in seeds {i['lesion_dependence_seeds'] or 'none'} "
            f"({len(i['lesion_dependence_seeds'])}/{len(SEEDS)}); matched-seed performance maintained in seeds "
            f"{i['performance_maintained_seeds'] or 'none'} ({len(i['performance_maintained_seeds'])}/{len(SEEDS)}).")


def report_markdown(r):
    i, a = r["interpretation"], r["absolute_reference_bounds"]
    lines = [f"# {r['label']} -- three-seed post-run report", "",
             f"Runs: `{os.path.dirname(r['summary_dir'])}`. Baselines: `{r['baseline_root']}`. Validation images: "
             f"{r['n_validation']} (same images, same order, same metric code for every run). Bootstrap: {r['n_boot']} "
             f"grade-stratified paired resamples, seed {r['boot_seed']}.", "",
             r["statements"]["c2"], r["statements"]["ema"], "Everything in this report is descriptive.", "",
             "## 1. Matched-seed comparison with P (primary analysis)", "",
             f"Each {r['label']} seed against the P run of the same seed, BEST checkpoints.", ""]
    lines += delta_table(r, "P", with_shuffle=True) + ["", "The same differences against the tolerances already in the record:", ""]
    lines += matched_table(r) + ["",
             f"## 2. Interpretation: {i['category']} -- {i['title']}", "",
             "Rules fixed in the research record (§54) before seeds 123 and 2026 finished. " + i["rule"], "",
             _evidence(r), "", i["statement"], "",
             f"## 3. {ABSOLUTE_LABEL}", "",
             "The five checks the run's own code applied to each seed, against P-42's values. They are kept as "
             "descriptive safety checks. Tally recorded by the run: "
             f"**{a['tally_recorded_by_the_run']}** (seeds within all five: {a['seeds_passing_all'] or 'none'}).", ""]
    lines += absolute_table(r) + ["", "P's and PL's own runs against the same bounds:", ""] + context_table(r) + ["",
             "## 4. Matched-seed comparison with PL (legacy lesion maps)", ""] + delta_table(r, "PL") + ["",
             "## 5. Checkpoints, over-fitting and what the model uses", ""] + behaviour_table(r) + ["",
             "## 6. Not concluded here", ""] + [f"- {x}" for x in NOT_CONCLUDED] + [""]
    return "\n".join(lines) + "\n"


def record_section(r, number):
    """A draft section for docs/experiments/RACAF_Gate_Initialization_And_C1_Control.md (append-only)."""
    i, a, today = r["interpretation"], r["absolute_reference_bounds"], datetime.date.today().isoformat()
    ident = r["identity"]
    pl_ = r["seed_mean_delta"]["PL"]
    failing = [name for name, c in r["reference_context"].items() if name.startswith("P-") and not c["all_metric_checks_pass"]]
    lines = [f"## {number}. {r['label']} — three seeds (42, 123, 2026): category **{i['category']}** — {i['title']} ({today})", "",
             f"**Runs.** `experiments/{r['experiment']}/` on Drive; Stage-4 `{str(ident['stage4_sha256'])[:8]}…`, Stage-3 "
             f"`{str(ident['stage3_sha256'])[:8]}…`, bundle fingerprint `{str(ident['bundle_fingerprint'])[:8]}…`, split "
             f"`{str(ident['split_sha256'])[:8]}…`. EMA: {ident['ema']}. {r['n_validation']} validation images, the P order. "
             f"{r['statements']['c2']}", "",
             "**Matched-seed comparison with P (primary analysis; BEST checkpoints; descriptive).**", ""]
    lines += delta_table(r, "P", keys=MATCHED_KEYS, with_shuffle=True) + [""] + matched_table(r) + ["",
             f"**Interpretation (rules of §54): {i['category']} — {i['title']}.** {_evidence(r)} {i['statement']}", "",
             f"**{ABSOLUTE_LABEL}.** Tally recorded by the run's code: {a['tally_recorded_by_the_run']}.", ""]
    lines += absolute_table(r) + ["",
             "- P's own seeds against the same bounds: "
             + ("all three are within them." if not failing else f"{', '.join(failing)} are not within all four metric bounds."),
             f"- {r['label']} − PL, same seed, mean ± SD: QWK {_f(pl_['qwk']['mean'], True)} ± {_f(pl_['qwk']['sd'])}; AUROC ≥3 "
             f"{_f(pl_['auroc_ge3_g4_vs_g012']['mean'], True)} ± {_f(pl_['auroc_ge3_g4_vs_g012']['sd'])}; grade-3 recall "
             f"{_f(pl_['grade3_recall']['mean'], True)} ± {_f(pl_['grade3_recall']['sd'])}.",
             "- Vessel-shuffle ΔQWK per seed: " + ", ".join(_f(r["per_seed"][s]["permutation_dqwk"]["vessel"], True) for s in SEEDS)
             + "; BEST epoch index: " + ", ".join(str((r["per_seed"][s]["history"] or {}).get("best_epoch_index")) for s in SEEDS)
             + "; train − validation QWK at BEST: "
             + ", ".join(_f((r["per_seed"][s]["history"] or {}).get("train_minus_val_qwk_at_best"), True) for s in SEEDS) + ".", "",
             "**Not concluded.**"] + [f"- {x}" for x in NOT_CONCLUDED] + [""]
    return "\n".join(lines) + "\n"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True, help="local copy of experiments/Architecture1")
    parser.add_argument("--baseline", required=True, help="local copy of the P/PL experiment directory")
    parser.add_argument("--prefix", help="run prefix, e.g. arch1_cb5fc7a8d370 (needed only if several are present)")
    parser.add_argument("--out", help="output directory (default: <runs>/posthoc_<timestamp>)")
    parser.add_argument("--section", type=int, default=55, help="number of the draft research-record section")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    args = parser.parse_args(argv)
    result = analyse(args.runs, args.baseline, args.prefix, args.n_boot)
    out = args.out or os.path.join(args.runs, "posthoc_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(result), fh, indent=1)
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report_markdown(result))
    with open(os.path.join(out, "record_section.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(record_section(result, args.section))
    i = result["interpretation"]
    print(f"category {i['category']}: {i['title']} | {_evidence(result)}")
    print(f"P-42 absolute reference bounds, tally recorded by the run: {result['absolute_reference_bounds']['tally_recorded_by_the_run']}")
    print("written:", out)
    return result


if __name__ == "__main__":
    main()
