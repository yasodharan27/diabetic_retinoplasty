"""
C1 -- ICDR-ordered two-route Stage-8 head vs a capacity-matched CORN refit, on the frozen
NO_RACAF representation E. PRE-REGISTERED, CONTROLLED EXPERIMENT (APTOS development decision).

Design record: research/Grade3vs4_Architecture_Research/FINAL_DIRECTION_DECISION.md (Addendum A)
and docs/experiments/C1_ICDR_Two_Route_Head_Report.md. Head module: icdr_two_route_head.py.

WHAT IT DOES (Colab, T4, Drive mounted -- `%run` this file or `python` it)
  1. Verifies the authoritative APTOS split (sha256) and the pinned cached population.
  2. For each frozen NO_RACAF BEST backbone (seeds 42 / 123 / 2026): validates the checkpoint
     against its sealed manifest sha256, loads it READ-ONLY, freezes it, extracts E (the exact
     input of the original CORN head) for all training and validation images, and proves
     (a) the saved validation logits are reproduced, (b) the original head applied to E
     reproduces the model logits, (c) the backbone weights are bit-for-bit unchanged.
  3. Smoke-fits H1 and H2 on a tiny training subset and checks probability sums, decoding,
     parameter parity and grade-4 routing.
  4. Fits H1 (CORN refit, Dense 256->4) and H2 (two-route, Dense 256->1 + 256->3) on TRAINING E
     only, scores the validation set once, and applies the pre-registered decision rule.

WHAT IT NEVER DOES
  trains or re-saves any backbone; regenerates Stage 02-04 caches; fits anything on validation
  data; tunes L2, class weights or the q threshold; touches DDR or IDRiD; writes anywhere except
  a new directory experiments/C1_ICDRTwoRouteHead/<timestamp>/. It refuses to run again once a
  completed run exists (no re-running until a result is liked).
"""

import datetime
import json
import os
import posixpath
import subprocess
import sys

import numpy as np

# ============================================================================================
# PRE-REGISTERED CONFIGURATION -- fixed before any result is seen. Do not edit after running.
# ============================================================================================

EXPERIMENT_VERSION = "c1-icdr-two-route-v1"
OUTPUT_GROUP = "C1_ICDRTwoRouteHead"
BACKBONE_ARM = "NO_RACAF"
BACKBONE_SEEDS = (42, 123, 2026)
BOOTSTRAP_SEED = 20260927
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95

# Decision rule (primary: Δ = AUROC_H2 − AUROC_H1 of each head's own P(grade >= 3), grade 4 vs 0–2)
SUPPORTIVE_MEAN = 0.03          # mean Δ >= +0.03 AND Δ > 0 in 3/3 seeds AND guardrails hold
NOT_SUPPORTIVE_MEAN = 0.01      # mean Δ < +0.01 OR Δ <= 0 in >= 2/3 seeds
NOT_SUPPORTIVE_MIN_NONPOSITIVE = 2
# Guardrails: 3-seed MEAN of (H2 − H1), inclusive bounds (the §12 / §16 convention).
GUARDRAILS = {"grade3_recall": ("min", -0.10), "auroc_ge3_g3_vs_g012": ("min", -0.03),
              "qwk": ("min", -0.02), "false_urgent_rate": ("max", 0.02)}

# Population. The authoritative split has 2,929 train / 733 validation entries; the population
# the frozen backbones were trained and evaluated on is that split MINUS the images with no cached
# representation (empty field of view: 11 over the whole split, 3 of them in validation). That
# yielded population is pinned in the six-run experiment manifest (`n_train_yielded`,
# `n_val_yielded`, `empty_fov_ids`) and is what C1 uses -- it is never hard-coded here.
SPLIT_TRAIN_COUNTS = (1444, 296, 799, 154, 236)      # split counts (weighted_corn pin, 2,929)
EXPECTED_N_VAL = 730                                 # 733 minus the 3 pinned empty-FOV ids
EXPECTED_PERSISTENT_FAILURES = 21                    # research record §15
PARITY_LOGIT_TOLERANCE = 0.05                        # Phase-0 tolerance (mixed_float16)
INFERENCE_BATCH = 8
SMOKE_PER_GRADE = 20

REPO_URL = "https://github.com/yasodharan27/diabetic_retinoplasty.git"
REPO_DIR = "/content/diabetic_retinoplasty"
BRANCH = "main"


def preregistration():
    import icdr_two_route_head as th
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "hypothesis": "The sequential CORN output structure gates Grade-4/PDR predictions when PDR "
                      "evidence is present but NPDR lesion burden is low; changing only the "
                      "Stage-8 output structure on the identical frozen E reduces this.",
        "backbones": {"arm": BACKBONE_ARM, "seeds": list(BACKBONE_SEEDS),
                      "source": "improved_multiseed_2026_09 BEST checkpoints (read-only)"},
        "heads": {"H0": "original jointly trained CORN head (saved per_sample_best.csv)",
                  "H1": "CORN Dense(256->4) refit on frozen training E (control)",
                  "H2": "two-route: PDR Dense(256->1) + NPDR CORN Dense(256->3) over grades 0-3 "
                        "(treatment); grade 4 excluded from NPDR supervision"},
        "fitting": th.fitting_configuration(),
        "training_population": {
            "split": "authoritative APTOS train split (2,929 entries, grades "
                     f"{list(SPLIT_TRAIN_COUNTS)}), as cached: minus the empty-field-of-view ids "
                     "pinned in the six-run manifest; n = the manifest's n_train_yielded",
            "h1_n": "all cached training images",
            "pdr_route_n": "all cached training images",
            "npdr_route_n": "cached training images of grades 0-3 (grade 4 excluded)"},
        "evaluation_population": {"split": "authoritative APTOS validation split",
                                  "n": EXPECTED_N_VAL, "used_for": "scoring once; never fitting"},
        "primary_endpoint": "per-seed Δ = AUROC(H2) − AUROC(H1), grade 4 vs grades 0–2, score = "
                            "each head's own P(grade >= 3)",
        "decision_rule": {
            "SUPPORTIVE": f"mean Δ >= +{SUPPORTIVE_MEAN} AND Δ > 0 in 3/3 seeds AND all guardrails",
            "NOT_SUPPORTIVE": f"mean Δ < +{NOT_SUPPORTIVE_MEAN} OR Δ <= 0 in >= "
                              f"{NOT_SUPPORTIVE_MIN_NONPOSITIVE}/3 seeds",
            "TRADE_OFF": "SUPPORTIVE primary criteria met but a guardrail fails",
            "INCONCLUSIVE": "anything else",
            "guardrails_on_3_seed_mean_of_H2_minus_H1": {k: list(v) for k, v in GUARDRAILS.items()},
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "n": N_BOOTSTRAP, "ci": CI_LEVEL,
                      "method": "stratified (grade 4 vs 0–2) paired resampling of validation "
                                "images; identical resamples for every seed and head"},
        "out_of_scope": ["DDR (not downloaded, not used)", "IDRiD (not used)",
                         "any Stage 05-07 change", "threshold / L2 / weight tuning"],
    }


# ============================================================================================
# PURE ANALYSIS FUNCTIONS (no Drive, no TensorFlow) -- unit-tested
# ============================================================================================

def _auroc(y, score):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64)))


def head_metrics(grades, predicted, p_ge3, p_grade4, persistent_mask):
    """Every pre-registered endpoint for one head on one validation population."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    g = np.asarray(grades, dtype=int)
    pred = np.asarray(predicted, dtype=int)
    p_ge3 = np.asarray(p_ge3, dtype=np.float64)
    p4 = np.asarray(p_grade4, dtype=np.float64)
    persistent_mask = np.asarray(persistent_mask, dtype=bool)
    low, g3, g4 = g <= 2, g == 3, g == 4
    sel_4v012, sel_3v012, sel_34 = low | g4, low | g3, g3 | g4
    called4 = pred == 4
    return {
        "auroc_ge3_g4_vs_g012": _auroc(g4[sel_4v012], p_ge3[sel_4v012]),          # PRIMARY
        "auroc_p4_g4_vs_g012": _auroc(g4[sel_4v012], p4[sel_4v012]),
        "auroc_ge3_g3_vs_g012": _auroc(g3[sel_3v012], p_ge3[sel_3v012]),
        "auroc_p4_g4_vs_rest": _auroc(g4, p4),                                     # descriptive
        "auroc_p4_g4_vs_g3": _auroc(g4[sel_34], p4[sel_34]),                       # descriptive
        "grade4_recall": float(np.mean(pred[g4] == 4)) if g4.any() else float("nan"),
        "grade4_precision": (float(np.sum(called4 & g4) / np.sum(called4))
                             if called4.any() else float("nan")),
        "grade4_predicted_count": int(called4.sum()),
        "grade3_recall": float(np.mean(pred[g3] == 3)) if g3.any() else float("nan"),
        "false_urgent_rate": float(np.mean(pred[low] >= 3)) if low.any() else float("nan"),
        "qwk": float(cohen_kappa_score(g, pred, weights="quadratic", labels=list(range(5)))),
        "mae": float(np.mean(np.abs(pred - g))),
        "grade4_decoded_counts": np.bincount(pred[g4], minlength=5).astype(int).tolist(),
        "grade4_decoded_le2": int(np.sum(pred[g4] <= 2)),
        "persistent21_decoded_le2": int(np.sum(pred[persistent_mask] <= 2)),
        "confusion_matrix": confusion_matrix(g, pred, labels=list(range(5))).astype(int).tolist(),
    }


def decide(deltas, guardrail_mean_deltas):
    """The pre-registered verdict. `deltas`: per-seed primary Δ (H2 − H1). `guardrail_mean_deltas`:
    3-seed mean of (H2 − H1) for each guardrail metric."""
    deltas = np.asarray(deltas, dtype=np.float64)
    mean = float(deltas.mean())
    positive = int(np.sum(deltas > 0))
    nonpositive = int(np.sum(deltas <= 0))
    guard = {}
    for name, (kind, bound) in GUARDRAILS.items():
        value = float(guardrail_mean_deltas[name])
        ok = value >= bound - 1e-12 if kind == "min" else value <= bound + 1e-12
        guard[name] = {"mean_delta": value, "bound": bound, "kind": kind, "holds": bool(ok)}
    guards_hold = all(v["holds"] for v in guard.values())
    improved = mean >= SUPPORTIVE_MEAN and positive == deltas.size
    if mean < NOT_SUPPORTIVE_MEAN or nonpositive >= NOT_SUPPORTIVE_MIN_NONPOSITIVE:
        verdict = "NOT_SUPPORTIVE"
    elif improved and guards_hold:
        verdict = "SUPPORTIVE"
    elif improved:
        verdict = "TRADE_OFF"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "mean_delta": mean, "sd_delta": float(deltas.std(ddof=1))
            if deltas.size > 1 else float("nan"), "seeds_positive": positive,
            "seeds_nonpositive": nonpositive, "per_seed_delta": deltas.tolist(),
            "primary_improved": bool(improved), "guardrails": guard,
            "guardrails_hold": bool(guards_hold)}


def stratified_bootstrap(y, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, pos.size, replace=True),
                            rng.choice(neg, neg.size, replace=True)]) for _ in range(n_boot)]


def paired_bootstrap_primary(grades, h1_scores, h2_scores):
    """Per-seed and seed-averaged Δ AUROC CIs on identical stratified resamples of the grade 4 vs
    0–2 population. `h1_scores`/`h2_scores`: lists (one per seed) of P(grade >= 3) arrays."""
    g = np.asarray(grades, dtype=int)
    sel = (g <= 2) | (g == 4)
    y = (g[sel] == 4).astype(int)
    a = [np.asarray(s, dtype=np.float64)[sel] for s in h1_scores]
    b = [np.asarray(s, dtype=np.float64)[sel] for s in h2_scores]
    draws = np.empty((N_BOOTSTRAP, len(a)))
    for i, idx in enumerate(stratified_bootstrap(y)):
        for s in range(len(a)):
            draws[i, s] = _auroc(y[idx], b[s][idx]) - _auroc(y[idx], a[s][idx])
    lo, hi = (1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100
    mean_draws = draws.mean(axis=1)
    return {"per_seed_ci": [[float(np.percentile(draws[:, s], lo)),
                             float(np.percentile(draws[:, s], hi))] for s in range(len(a))],
            "per_seed_fraction_positive": [float((draws[:, s] > 0).mean())
                                           for s in range(len(a))],
            "mean_delta_ci": [float(np.percentile(mean_draws, lo)),
                              float(np.percentile(mean_draws, hi))],
            "mean_delta_fraction_positive": float((mean_draws > 0).mean()),
            "n_bootstrap": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED}


STAGE_LAYER_NAMES = ("local_feature_extraction_adaptive_multi_kernel_cnn",
                     "global_feature_extraction_dual_scale_swin",
                     "feature_fusion_adaptive_cross_attention", "corn")


def forward_with_embedding(model, stage5_input, stage6_input, reliability):
    """One frozen forward pass. Returns (graph logits, eager head(E) logits, E), where E is the
    Stage-7 output -- exactly the tensor the original CORN head consumes in the NO_RACAF graph."""
    stage5, stage6, stage7, head = (model.get_layer(n) for n in STAGE_LAYER_NAMES)
    logits = np.asarray(model.predict_on_batch([stage5_input, stage6_input, reliability]),
                        np.float64)
    fused = stage7([stage5(stage5_input, training=False), stage6(stage6_input, training=False)],
                   training=False)
    return (logits, np.asarray(head(fused, training=False), np.float64),
            np.asarray(fused, np.float64))


def _safe_print(text):
    """Prints on any console encoding (Colab is UTF-8; a Windows console is not)."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(text.encode(encoding, "replace").decode(encoding, "replace"))


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def run_analysis(results, embeddings, y_train, y_val, h0_by_seed, persistent, val_ids,
                 out_dir=None, strict=True):
    """Steps 6-8, after E extraction: smoke checks, full H1/H2 fits on TRAINING E only, one
    validation scoring, contrasts, paired bootstrap, the pre-registered verdict, and the report.
    `embeddings[seed] = (E_train, E_val)`; `h0_by_seed[seed]` holds the SAVED original-head
    outputs (`predicted_grade`, `p_ge3` = p_gt_2, `p_grade4` = p_gt_3). Pure (no Drive, no
    backbone); `out_dir=None` writes nothing -- used for the offline dry run."""
    import pandas as pd

    import icdr_two_route_head as th

    # ---- [6] smoke fit on a tiny training subset ----
    smoke_idx = np.concatenate([np.flatnonzero(y_train == g)[:SMOKE_PER_GRADE] for g in range(5)])
    e_s, g_s = embeddings[BACKBONE_SEEDS[0]][0][smoke_idx], y_train[smoke_idx]
    s1, s2 = th.fit_h1_corn_refit(e_s, g_s), th.fit_h2_two_route(e_s, g_s)
    o1, o2 = s1.predict(e_s), s2.predict(e_s)
    # Routing check: with the standardised features held fixed, overwriting every grade-4 row
    # must leave each NPDR task's fit bit-for-bit identical.
    mean, sd = th.standardiser(e_s)
    x_std = (e_s - mean) / sd
    x_mod = x_std.copy()
    x_mod[g_s == 4] = 99.0
    routing_ok = True
    for name, mask, target in th.h2_tasks(g_s):
        if name.startswith("npdr"):
            w = np.asarray(th.CLASS_WEIGHTS)[g_s][mask]
            a = th.fit_weighted_logistic(x_std[mask], target[mask], w)[0]
            b = th.fit_weighted_logistic(x_mod[mask], target[mask], w)[0]
            routing_ok &= bool(np.array_equal(a, b))
    smoke = {"n": int(smoke_idx.size),
             "prob_sum_max_err_H1": float(np.max(np.abs(o1["probabilities"].sum(1) - 1))),
             "prob_sum_max_err_H2": float(np.max(np.abs(o2["probabilities"].sum(1) - 1))),
             "q_equals_p4": bool(np.array_equal(o2["q"], o2["probabilities"][:, 4])),
             "decode_rule_ok": bool(np.array_equal(
                 o2["predicted_grade"], np.where(o2["q"] > 0.5, 4, o2["npdr_grade"]))),
             "parameters": [s1.parameter_count(), s2.parameter_count()],
             "grade4_excluded_from_npdr_route": routing_ok,
             "fit_rows_are_training_rows_only": True}
    results["smoke"] = smoke
    print("Smoke checks:", smoke)
    if not (smoke["prob_sum_max_err_H1"] < 1e-9 and smoke["prob_sum_max_err_H2"] < 1e-9
            and smoke["q_equals_p4"] and smoke["decode_rule_ok"] and routing_ok
            and smoke["parameters"] == [th.EXPECTED_HEAD_PARAMETERS] * 2):
        raise RuntimeError(f"Smoke checks failed: {smoke}")

    # ---- [7] full fits (training E only) and single validation scoring ----
    per_seed, per_sample_rows, fitted_records, h1_ge3, h2_ge3 = [], [], {}, [], []
    h0_metrics = {}
    for seed in BACKBONE_SEEDS:
        e_train, e_val = embeddings[seed]
        if e_train.shape[0] != len(y_train) or e_val.shape[0] != len(y_val):
            raise RuntimeError(f"seed {seed}: E rows do not match the population.")
        if strict and len(y_val) != EXPECTED_N_VAL:
            raise RuntimeError(f"validation n {len(y_val)} != {EXPECTED_N_VAL}")
        h1 = th.fit_h1_corn_refit(e_train, y_train)
        h2 = th.fit_h2_two_route(e_train, y_train)
        for fitted in (h1, h2):
            bad = {k: v for k, v in fitted.task_info.items() if not v["converged"]}
            if bad:
                raise RuntimeError(f"seed {seed} {fitted.kind}: fit did not converge: {bad}")
        # routing invariants on the real fit: H1 and the PDR route use every training image;
        # the NPDR route uses exactly the grade 0-3 images and never a grade-4 image.
        if not (h1.task_info["corn_task_0"]["n"] == h2.task_info["pdr_route"]["n"] == len(y_train)
                and h2.task_info["npdr_task_0"]["n"] == int(np.sum(np.asarray(y_train) <= 3))
                and h1.task_info["corn_task_3"]["n"] == int(np.sum(np.asarray(y_train) >= 3))
                and all(4 not in h2.task_info[f"npdr_task_{k}"]["grades_in_task"]
                        for k in range(3))):
            raise RuntimeError(f"seed {seed}: task routing invariant violated.")
        out1, out2 = h1.predict(e_val), h2.predict(e_val)
        h0 = h0_by_seed[seed]
        m0 = head_metrics(y_val, h0["predicted_grade"], h0["p_ge3"], h0["p_grade4"], persistent)
        m1 = head_metrics(y_val, out1["predicted_grade"], out1["p_ge3"], out1["p_grade4"],
                          persistent)
        m2 = head_metrics(y_val, out2["predicted_grade"], out2["p_ge3"], out2["p_grade4"],
                          persistent)
        h0_metrics[seed] = m0
        h1_ge3.append(out1["p_ge3"])
        h2_ge3.append(out2["p_ge3"])
        for head_name, m in (("H0", m0), ("H1", m1), ("H2", m2)):
            per_seed.append({"seed": seed, "head": head_name, **m})
        fitted_records[seed] = {
            "H1": {"fingerprint": h1.fingerprint(), "parameters": h1.parameter_count(),
                   "tasks": h1.task_info},
            "H2": {"fingerprint": h2.fingerprint(), "parameters": h2.parameter_count(),
                   "tasks": h2.task_info}}
        if out_dir:
            np.savez_compressed(posixpath.join(out_dir, f"heads_seed_{seed}.npz"),
                                H1_kernel=h1.kernel, H1_bias=h1.bias, H2_kernel=h2.kernel,
                                H2_bias=h2.bias)
        for n, image_id in enumerate(val_ids):
            row = {"seed": seed, "image_id": image_id, "true_grade": int(y_val[n]),
                   "persistent21": bool(persistent[n]),
                   "H0_pred": int(h0["predicted_grade"][n]), "H0_p_ge3": h0["p_ge3"][n],
                   "H0_p4": h0["p_grade4"][n],
                   "H1_pred": int(out1["predicted_grade"][n]), "H1_p_ge3": out1["p_ge3"][n],
                   "H1_p4": out1["p_grade4"][n],
                   "H2_pred": int(out2["predicted_grade"][n]), "H2_p_ge3": out2["p_ge3"][n],
                   "H2_q": out2["q"][n], "H2_npdr_grade": int(out2["npdr_grade"][n])}
            row.update({f"H1_P{k}": out1["probabilities"][n, k] for k in range(5)})
            row.update({f"H2_P{k}": out2["probabilities"][n, k] for k in range(5)})
            per_sample_rows.append(row)
        print(f"seed {seed}: primary AUROC H0 {m0['auroc_ge3_g4_vs_g012']:.4f} | "
              f"H1 {m1['auroc_ge3_g4_vs_g012']:.4f} | H2 {m2['auroc_ge3_g4_vs_g012']:.4f}")
    results["fitted_heads"] = fitted_records

    # ---- [8] contrasts, bootstrap, verdict ----
    table = pd.DataFrame(per_seed)
    scalar_cols = [c for c in table.columns if c not in ("seed", "head", "grade4_decoded_counts",
                                                         "confusion_matrix")]
    deltas = []
    for seed in BACKBONE_SEEDS:
        r1 = table[(table["seed"] == seed) & (table["head"] == "H1")].iloc[0]
        r2 = table[(table["seed"] == seed) & (table["head"] == "H2")].iloc[0]
        deltas.append({"seed": seed, **{c: float(r2[c] - r1[c]) for c in scalar_cols}})
    delta_table = pd.DataFrame(deltas)
    primary = delta_table["auroc_ge3_g4_vs_g012"].to_numpy()
    guard_means = {k: float(delta_table[k].mean()) for k in GUARDRAILS}
    decision = decide(primary, guard_means)
    boot = paired_bootstrap_primary(y_val, h1_ge3, h2_ge3)
    decision["bootstrap"] = boot
    results["decision"] = decision
    results["per_seed_metrics"] = per_seed
    results["h2_minus_h1"] = deltas
    results["h2_minus_h1_mean"] = {c: float(delta_table[c].mean()) for c in scalar_cols}
    results["h2_minus_h1_sd"] = {c: float(delta_table[c].std(ddof=1)) for c in scalar_cols}
    results["h1_minus_h0_mean"] = {c: float(np.mean([
        table[(table["seed"] == s) & (table["head"] == "H1")].iloc[0][c]
        - table[(table["seed"] == s) & (table["head"] == "H0")].iloc[0][c] for s in BACKBONE_SEEDS]))
        for c in scalar_cols}

    samples = pd.DataFrame(per_sample_rows)
    report = build_report(results, table, samples)
    if out_dir:
        table.to_csv(posixpath.join(out_dir, "per_seed_results.csv"), index=False)
        delta_table.to_csv(posixpath.join(out_dir, "h2_minus_h1.csv"), index=False)
        samples.to_csv(posixpath.join(out_dir, "per_sample_predictions.csv"), index=False)
        with open(posixpath.join(out_dir, "REPORT.md"), "w", encoding="utf-8") as fh:
            fh.write(report)
        # results.json is written LAST and atomically: its presence marks the run COMPLETED.
        tmp = posixpath.join(out_dir, "results.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(to_jsonable(results), fh, indent=2)
        os.replace(tmp, posixpath.join(out_dir, "results.json"))
    _safe_print(report)
    if out_dir:
        print("\nAll outputs written to:", out_dir)
    return results, table, samples


# ============================================================================================
# ORCHESTRATION STEPS (Colab) -- shared by main() and colab/notebooks/stage08_icdr_two_route_head.ipynb
# ============================================================================================

SIX_RUN_SUBDIR = "ImprovedTraining"
SIX_RUN_ID = "improved_multiseed_2026_09"
LOCAL_CACHE_DIR = "/content/cache/local_feature_extraction"
LOCAL_RACAF_CACHE_DIR = "/content/cache/racaf"
LOCAL_CACHE_MARKER = "/content/cache/.multiseed_archive_extracted.json"
PREREGISTRATION_FILENAME = "PREREGISTRATION.json"
RESULTS_FILENAME = "results.json"
REPORT_FILENAME = "REPORT.md"
SOURCE_FILES = ("icdr_two_route_head.py", "icdr_two_route_experiment.py")
#: Keys stamped into the stored pre-registration that are allowed to differ between sessions.
PREREGISTRATION_SESSION_KEYS = ("repo_commit", "written")


def _bootstrap_repo():
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "origin", BRANCH], check=True)
    for path in (REPO_DIR, os.path.join(REPO_DIR, "colab", "common")):
        if path not in sys.path:
            sys.path.insert(0, path)


def _atomic_json(path, payload):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(to_jsonable(payload), fh, indent=2)
    os.replace(tmp, path)


def source_digests(repo_dir):
    """sha256 of the C1 source files (line endings normalised) -- frozen with the
    pre-registration, so the code cannot silently change between resumed sessions."""
    import hashlib
    digests = {}
    for name in SOURCE_FILES:
        with open(os.path.join(repo_dir, name), "rb") as fh:
            digests[name] = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    return digests


def resolve_experiment_dir(group_dir):
    """ONE experiment directory, created on the first session and resumed afterwards.
    Returns (out_dir, completed). A completed run is never re-run: later sessions only reprint its
    stored report. More than one pre-registered directory is refused rather than guessed."""
    os.makedirs(group_dir, exist_ok=True)
    existing = sorted(d for d in os.listdir(group_dir)
                      if os.path.exists(posixpath.join(group_dir, d, PREREGISTRATION_FILENAME)))
    if len(existing) > 1:
        raise RuntimeError(f"More than one pre-registered C1 experiment under {group_dir}: "
                           f"{existing}. Refusing to guess which one to continue.")
    if existing:
        out_dir = posixpath.join(group_dir, existing[0])
    else:
        out_dir = posixpath.join(group_dir, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(out_dir, exist_ok=False)
    completed = os.path.exists(posixpath.join(out_dir, RESULTS_FILENAME))
    return out_dir, completed


def freeze_or_verify_preregistration(out_dir, repo_dir, repo_commit):
    """First session: writes PREREGISTRATION.json before any extraction or fit. Later sessions:
    re-derives it from the code and STOPS on any difference (config, rule, endpoint or source)."""
    current = to_jsonable(preregistration())
    current["source_sha256"] = source_digests(repo_dir)
    path = posixpath.join(out_dir, PREREGISTRATION_FILENAME)
    if not os.path.exists(path):
        stored = dict(current, repo_commit=repo_commit,
                      written=datetime.datetime.now().isoformat(timespec="seconds"))
        _atomic_json(path, stored)
        return stored, "written"
    with open(path) as fh:
        stored = json.load(fh)
    comparable = {k: v for k, v in stored.items()
                  if k not in PREREGISTRATION_SESSION_KEYS + ("supersedes",)}
    if comparable != current:
        changed = sorted(k for k in set(comparable) | set(current)
                         if comparable.get(k) != current.get(k))
        artifacts = data_artifacts(out_dir)
        if artifacts:
            raise RuntimeError(f"The frozen pre-registration in {path} differs from the current "
                               f"code in {changed}, and data artifacts already exist "
                               f"({artifacts[:5]}). The protocol cannot change mid-experiment.")
        # No E has been extracted and nothing fitted or scored: the change happened before any
        # contact with the data, so the pre-registration is re-frozen. The superseded file is kept.
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        superseded = f"PREREGISTRATION.superseded_{stamp}.json"
        os.replace(path, posixpath.join(out_dir, superseded))
        stored = dict(current, repo_commit=repo_commit,
                      written=datetime.datetime.now().isoformat(timespec="seconds"),
                      supersedes=list(stored.get("supersedes", [])) + [
                          {"file": superseded, "changed_keys": changed}])
        _atomic_json(path, stored)
        return stored, (f"re-frozen before any data contact (changed {changed}; previous copy "
                        f"kept as {superseded})")
    return stored, "verified"


def data_artifacts(out_dir):
    """Files that mean the experiment has touched data (E extracted, heads fitted or scored)."""
    names = sorted(os.listdir(out_dir)) if os.path.isdir(out_dir) else []
    return [n for n in names if n.startswith(("E_seed_", "backbone_seed_", "heads_seed_",
                                              "per_seed_results", "per_sample_predictions",
                                              "h2_minus_h1", REPORT_FILENAME, RESULTS_FILENAME))]


def ensure_local_cache(config):
    """Extracts the EXISTING cache archive once per runtime. Never regenerates a cache
    (itd.complete_local_cache() is deliberately not called: it could run Stage 03/04)."""
    if os.path.exists(LOCAL_CACHE_MARKER):
        return "already_extracted"
    import joint_cache_archive as jca
    archive_dir = os.path.join(os.path.dirname(config.LOCAL_FEATURE_RESULTS_DIR), "cache_archive")
    plan = jca.plan_extraction(archive_dir=archive_dir, cache_dir=LOCAL_CACHE_DIR,
                               racaf_cache_dir=LOCAL_RACAF_CACHE_DIR)
    jca.print_extraction_plan(plan)
    if plan["drive_unreachable"] or not plan["fits"]:
        raise RuntimeError("Refusing to extract the cache archive (unreachable or no space).")
    report = jca.extract_archive(archive_dir=archive_dir, cache_dir=LOCAL_CACHE_DIR,
                                 racaf_cache_dir=LOCAL_RACAF_CACHE_DIR,
                                 min_free_bytes=plan["required_bytes"])
    jca.print_extract(report)
    if report["corrupt"] or report["drive_unreachable"]:
        raise RuntimeError("Archive extraction did not complete. Re-run.")
    with open(LOCAL_CACHE_MARKER, "w") as fh:
        json.dump({"extracted": datetime.datetime.now().isoformat(timespec="seconds")}, fh)
    return "extracted"


def prepare_population(config, itd, msr, six_run_dir):
    """Authoritative split (sha256-verified) and the pinned cached population. Any mismatch is
    fatal; nothing is regenerated."""
    import hashlib
    train_entries, val_entries, split_sha = msr.verify_split()
    assert split_sha == msr.EXPECTED_SPLIT_SHA256
    with open(posixpath.join(six_run_dir, "experiment_manifest.json")) as fh:
        six_run_population = json.load(fh)
    ensure_local_cache(config)
    cached_train = [(i, int(g)) for i, g in itd.locally_cached_entries(
        train_entries, LOCAL_CACHE_DIR, LOCAL_RACAF_CACHE_DIR)]
    cached_val = [(i, int(g)) for i, g in itd.locally_cached_entries(
        val_entries, LOCAL_CACHE_DIR, LOCAL_RACAF_CACHE_DIR)]
    train_ids = [i for i, _ in cached_train]
    val_ids = [i for i, _ in cached_val]
    y_train = np.array([g for _, g in cached_train])
    y_val = np.array([g for _, g in cached_val])
    counts = tuple(int(c) for c in np.bincount(y_train, minlength=5))
    split_counts = tuple(int(c) for c in np.bincount([int(g) for _, g in train_entries],
                                                      minlength=5))
    excluded_train = sorted({i for i, _ in train_entries} - set(train_ids))
    excluded_val = sorted({i for i, _ in val_entries} - set(val_ids))
    pinned_empty_fov = six_run_population.get("empty_fov_ids")
    population = {"split_sha256": split_sha, "n_split_train": len(train_entries),
                  "n_split_val": len(val_entries), "split_train_grade_counts": list(split_counts),
                  "n_train": len(cached_train), "n_val": len(cached_val),
                  "train_grade_counts": list(counts),
                  "val_grade_counts": np.bincount(y_val, minlength=5).tolist(),
                  "excluded_empty_fov_train": excluded_train,
                  "excluded_empty_fov_val": excluded_val,
                  "six_run_pin": {"n_train_yielded": six_run_population.get("n_train_yielded"),
                                  "n_val_yielded": six_run_population.get("n_val_yielded"),
                                  "empty_fov_ids": pinned_empty_fov},
                  "train_val_overlap": len(set(train_ids) & set(val_ids))}
    problems = []
    if split_counts != SPLIT_TRAIN_COUNTS:
        problems.append(f"split train grade counts {split_counts} != {SPLIT_TRAIN_COUNTS}")
    if [len(cached_train), len(cached_val)] != [int(six_run_population["n_train_yielded"]),
                                                int(six_run_population["n_val_yielded"])]:
        problems.append("cached population differs from the six-run manifest pin")
    if len(cached_val) != EXPECTED_N_VAL:
        problems.append(f"validation n {len(cached_val)} != {EXPECTED_N_VAL}")
    if pinned_empty_fov is not None and not set(excluded_train + excluded_val) <= set(pinned_empty_fov):
        problems.append("an excluded image is not one of the pinned empty-field-of-view ids")
    if population["train_val_overlap"] != 0:
        problems.append("train/validation overlap")
    if problems:
        raise RuntimeError(f"Population does not match the pre-registered pin ({problems}): "
                           f"{population}")
    population["train_membership_sha256"] = hashlib.sha256(
        "\n".join(f"{i},{g}" for i, g in cached_train).encode("utf-8")).hexdigest()
    return {"cached_train": cached_train, "cached_val": cached_val, "train_ids": train_ids,
            "val_ids": val_ids, "y_train": y_train, "y_val": y_val, "population": population}


def load_frozen_reference(msr, six_run_root, val_ids, y_val):
    """H0 = the SAVED original-head validation outputs, and the persistent PDR->=<2 set."""
    import pandas as pd
    saved = {}
    for seed in BACKBONE_SEEDS:
        path = posixpath.join(msr.run_dir(six_run_root, SIX_RUN_ID, BACKBONE_ARM, seed),
                              "evaluation", "per_sample_best.csv")
        frame = pd.read_csv(path, dtype={"image_id": str}).set_index("image_id").loc[val_ids]
        if not (frame["true_grade"].to_numpy() == y_val).all():
            raise RuntimeError(f"seed {seed}: saved per_sample_best grades differ from the split.")
        saved[seed] = frame
    persistent = np.ones(len(val_ids), dtype=bool) & (y_val == 4)
    for seed in BACKBONE_SEEDS:
        persistent &= saved[seed]["predicted_grade"].to_numpy() <= 2
    persistent_ids = [i for i, m in zip(val_ids, persistent) if m]
    if len(persistent_ids) != EXPECTED_PERSISTENT_FAILURES:
        raise RuntimeError(f"Persistent PDR->=<2 set has {len(persistent_ids)} images, expected "
                           f"{EXPECTED_PERSISTENT_FAILURES} (research record §15).")
    h0_by_seed = {seed: {"predicted_grade": saved[seed]["predicted_grade"].to_numpy(int),
                         "p_ge3": saved[seed]["p_gt_2"].to_numpy(np.float64),
                         "p_grade4": saved[seed]["p_gt_3"].to_numpy(np.float64)}
                  for seed in BACKBONE_SEEDS}
    return {"saved": saved, "persistent": persistent, "persistent_ids": persistent_ids,
            "h0_by_seed": h0_by_seed}


def _embedding_paths(out_dir, seed):
    return (posixpath.join(out_dir, f"E_seed_{seed}.npz"),
            posixpath.join(out_dir, f"backbone_seed_{seed}.json"))


def _embedding_sha(e_train, e_val):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(e_train).tobytes()
                          + np.ascontiguousarray(e_val).tobytes()).hexdigest()


def embedding_status(out_dir, seed):
    npz_path, record_path = _embedding_paths(out_dir, seed)
    if os.path.exists(npz_path) and os.path.exists(record_path):
        return "EXTRACTED"
    if os.path.exists(npz_path) or os.path.exists(record_path):
        return "PARTIAL"
    return "NOT_STARTED"


def extract_backbone_embeddings(seed, out_dir, data, reference, six_run_root):
    """Frozen E for one backbone, RESUMABLE. A seed already extracted in an earlier session is
    reused only if its stored record passed every parity check, its E still hashes to the recorded
    sha256, its population is identical, and its BEST weights still hash to the recorded value.
    Otherwise: validate the BEST checkpoint against its sealed manifest, load it read-only, freeze
    it, extract E, prove parity and non-mutation, and persist E + the record atomically.
    Returns (E_train, E_val, record)."""
    import gc

    import tensorflow as tf

    import corn
    import icdr_two_route_head as th
    import improved_training_data as itd
    import multiseed_runs as msr
    from training import checkpointing as ckpt

    label = f"{BACKBONE_ARM}/seed_{seed}"
    npz_path, record_path = _embedding_paths(out_dir, seed)
    run_dir = msr.run_dir(six_run_root, SIX_RUN_ID, BACKBONE_ARM, seed)
    slot_dir, pointer = msr.read_best(run_dir)   # validates sha256 against the sealed manifest
    if slot_dir is None:
        raise RuntimeError(f"{label}: no BEST checkpoint")
    weights_path = os.path.join(slot_dir, ckpt.MODEL_WEIGHTS_FILENAME)
    with open(os.path.join(slot_dir, ckpt.MANIFEST_FILENAME)) as fh:
        recorded_sha = json.load(fh)["files"][ckpt.MODEL_WEIGHTS_FILENAME]["sha256"]
    sha_before = ckpt.sha256_file(weights_path)
    if recorded_sha != sha_before:
        raise RuntimeError(f"{label}: BEST weights sha256 differs from the recorded manifest.")

    if embedding_status(out_dir, seed) == "EXTRACTED":
        with open(record_path) as fh:
            record = json.load(fh)
        with np.load(npz_path) as stored:
            e_train, e_val = stored["E_train"], stored["E_val"]
            same_population = (stored["train_ids"].tolist() == data["train_ids"]
                               and stored["val_ids"].tolist() == data["val_ids"])
        reusable = (record.get("ok") is True and record.get("weights_sha256") == sha_before
                    and record.get("E_sha256") == _embedding_sha(e_train, e_val) and same_population)
        if not reusable:
            raise RuntimeError(f"{label}: stored E from an earlier session failed re-verification "
                               f"({record_path}). It is not silently re-extracted; inspect it.")
        record["resumed"] = True
        print(f"{label}: E reused from an earlier session (sha256, population and backbone "
              "re-verified)")
        return e_train, e_val, record
    if embedding_status(out_dir, seed) == "PARTIAL":
        print(f"{label}: an incomplete earlier extraction was found; extracting again")

    def extract(model, entries):
        out = {"logits": [], "logits_eager": [], "E": []}
        for start in range(0, len(entries), INFERENCE_BATCH):
            batch = entries[start:start + INFERENCE_BATCH]
            samples = [itd.load_cached_sample(i, g, LOCAL_CACHE_DIR, LOCAL_RACAF_CACHE_DIR,
                                              False, None) for i, g in batch]
            s5 = np.stack([s["stage5_input"] for s in samples])
            s6 = np.stack([s["stage6_input"] for s in samples])
            rel = np.array([s["reliability"] for s in samples], np.float32).reshape(-1, 1)
            logits, logits_eager, fused = forward_with_embedding(model, s5, s6, rel)
            out["logits"].append(logits)
            out["logits_eager"].append(logits_eager)
            out["E"].append(fused)
            if (start // INFERENCE_BATCH) % 100 == 0:
                print(f"    {start + len(batch)}/{len(entries)}")
        return {k: np.concatenate(v, axis=0) for k, v in out.items()}

    gc.collect()
    tf.keras.backend.clear_session()
    model = msr.build_arm_model(BACKBONE_ARM, seed, verbose=0)
    ckpt.load_model_weights_only(model, weights_path)
    th.freeze(model)
    fp_before = {n: th.weights_fingerprint(model.get_layer(n)) for n in STAGE_LAYER_NAMES}
    print(f"{label}: BEST epoch {pointer['epoch']} loaded read-only; extracting E "
          f"({len(data['cached_train'])} train + {len(data['cached_val'])} validation)")
    tr = extract(model, data["cached_train"])
    va = extract(model, data["cached_val"])
    fp_after = {n: th.weights_fingerprint(model.get_layer(n)) for n in STAGE_LAYER_NAMES}
    kernel, bias = [np.asarray(w, np.float64) for w in model.get_layer("corn").get_weights()]
    del model
    sha_after = ckpt.sha256_file(weights_path)

    saved = reference["saved"][seed]
    saved_logits = saved[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64)
    head_recompute = np.concatenate([tr["E"], va["E"]]) @ kernel + bias
    graph_logits = np.concatenate([tr["logits"], va["logits"]])
    record = {
        "label": label, "best_epoch": pointer["epoch"], "weights_path": weights_path,
        "weights_sha256": sha_before, "manifest_recorded_sha256": recorded_sha,
        "weights_sha256_unchanged": sha_before == sha_after,
        "stage_fingerprints_unchanged": fp_before == fp_after,
        "stage_fingerprints": fp_before,
        "trainable_variables_after_freeze": 0,
        "E_dim": int(tr["E"].shape[1]),
        "max_abs_logit_diff_vs_saved": float(np.max(np.abs(va["logits"] - saved_logits))),
        "max_abs_logit_diff_eager_vs_graph": float(np.max(np.abs(
            np.concatenate([tr["logits_eager"], va["logits_eager"]]) - graph_logits))),
        "max_abs_logit_diff_head_on_E_vs_graph": float(np.max(np.abs(
            head_recompute - graph_logits))),
        "predicted_grade_agreement_vs_saved": float(np.mean(
            corn.decode_logits(va["logits"])["predicted_grade"]
            == saved["predicted_grade"].to_numpy())),
    }
    record["ok"] = bool(record["weights_sha256_unchanged"]
                        and record["stage_fingerprints_unchanged"]
                        and record["E_dim"] == th.D_MODEL
                        and record["max_abs_logit_diff_vs_saved"] <= PARITY_LOGIT_TOLERANCE
                        and record["max_abs_logit_diff_eager_vs_graph"] <= PARITY_LOGIT_TOLERANCE
                        and record["max_abs_logit_diff_head_on_E_vs_graph"]
                        <= PARITY_LOGIT_TOLERANCE)
    record["E_sha256"] = _embedding_sha(tr["E"], va["E"])
    record["resumed"] = False
    print(f"{label}: parity " + json.dumps(
        {k: v for k, v in record.items()
         if k.startswith(("max_", "ok", "weights_sha256_unchanged", "stage_fingerprints_unchanged",
                          "predicted_grade"))}))
    if not record["ok"]:
        raise RuntimeError(f"{label}: frozen-backbone parity failed: {record}")
    tmp = npz_path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, E_train=tr["E"], E_val=va["E"],
                            train_ids=np.array(data["train_ids"]),
                            val_ids=np.array(data["val_ids"]),
                            y_train=data["y_train"], y_val=data["y_val"])
    os.replace(tmp, npz_path)
    _atomic_json(record_path, record)        # written last: its presence marks the seed complete
    return tr["E"], va["E"], record


def base_results(out_dir, repo_commit, gpus):
    import keras
    import scipy
    import sklearn
    import tensorflow as tf

    import icdr_two_route_head as th
    return {"experiment_version": EXPERIMENT_VERSION, "out_dir": out_dir,
            "repo_commit": repo_commit,
            "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                         "sklearn": sklearn.__version__, "tensorflow": tf.__version__,
                         "keras": keras.__version__, "python": sys.version.split()[0]},
            "gpus": [g.name for g in gpus], "fitting_configuration": th.fitting_configuration()}


def print_stored_report(out_dir):
    with open(posixpath.join(out_dir, REPORT_FILENAME), encoding="utf-8") as fh:
        _safe_print(fh.read())


def main():
    """The same pipeline as the notebook, as one call (resumable; single-use)."""
    _bootstrap_repo()
    import setup as colab_setup
    colab_setup.setup()
    import colab_config
    import verify_environment
    verify_environment.verify_all(
        repo_dir=colab_config.REPO_DIR, drive_mount_point=colab_config.DRIVE_MOUNT_POINT,
        requirements_path=os.path.join(colab_config.REPO_DIR, "requirements.txt"), require_gpu=True)

    import tensorflow as tf

    import config
    import icdr_two_route_head as th
    import improved_training_data as itd
    import multiseed_runs as msr

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU: select a T4 runtime (mixed_float16 parity with the saved "
                           "predictions requires it).")
    experiments_root = colab_config.DRIVE.experiments_root
    six_run_root = posixpath.join(experiments_root, SIX_RUN_SUBDIR)
    six_run_dir = msr.experiment_root(six_run_root, SIX_RUN_ID)
    out_dir, completed = resolve_experiment_dir(posixpath.join(experiments_root, OUTPUT_GROUP))
    assert posixpath.commonpath([out_dir, six_run_dir]) != six_run_dir
    repo_commit = subprocess.run(["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
    _, how = freeze_or_verify_preregistration(out_dir, REPO_DIR, repo_commit)
    print(f"Experiment directory {out_dir}; pre-registration {how}")
    if completed:
        print("This experiment is already COMPLETED (single-use). Reprinting its stored report.")
        print_stored_report(out_dir)
        return None

    results = base_results(out_dir, repo_commit, gpus)
    results["parameter_parity"] = th.parameter_parity_report()
    data = prepare_population(config, itd, msr, six_run_dir)
    results["population"] = data["population"]
    reference = load_frozen_reference(msr, six_run_root, data["val_ids"], data["y_val"])
    results["persistent_failures"] = {"n": len(reference["persistent_ids"]),
                                      "image_ids": reference["persistent_ids"]}
    embeddings, records = {}, {}
    for seed in BACKBONE_SEEDS:
        e_train, e_val, records[seed] = extract_backbone_embeddings(
            seed, out_dir, data, reference, six_run_root)
        embeddings[seed] = (e_train, e_val)
    results["backbones"] = records
    run_analysis(results, embeddings, data["y_train"], data["y_val"], reference["h0_by_seed"],
                 reference["persistent"], data["val_ids"], out_dir)
    return results


# ============================================================================================
# REPORT
# ============================================================================================

def _f(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


_REPORT_METRICS = (
    ("auroc_ge3_g4_vs_g012", "**Primary**: AUROC P(≥3), grade 4 vs 0–2"),
    ("auroc_p4_g4_vs_g012", "AUROC P(4), grade 4 vs 0–2"),
    ("auroc_ge3_g3_vs_g012", "AUROC P(≥3), grade 3 vs 0–2"),
    ("auroc_p4_g4_vs_rest", "AUROC P(4), grade 4 vs rest"),
    ("auroc_p4_g4_vs_g3", "AUROC P(4), grade 4 vs 3 (descriptive)"),
    ("grade4_recall", "Grade-4 recall"), ("grade4_precision", "Grade-4 precision"),
    ("grade4_predicted_count", "Images called grade 4"),
    ("grade3_recall", "Grade-3 recall"),
    ("false_urgent_rate", "False urgent calls, grades 0–2"),
    ("qwk", "QWK"), ("mae", "MAE"),
    ("grade4_decoded_le2", "PDR decoded ≤ 2"),
    ("persistent21_decoded_le2", "Persistent-21 still decoded ≤ 2"))
_INTEGER_METRICS = ("grade4_decoded_le2", "persistent21_decoded_le2", "grade4_predicted_count")


def _row(table, seed, head):
    return table[(table["seed"] == seed) & (table["head"] == head)].iloc[0]


def _detail_sections(r, table, samples):
    """Everything needed to analyse the run, printed with the report (no separate reader)."""
    seeds = list(BACKBONE_SEEDS)
    out = ["", "---", "", "# Detailed analysis (for review)", ""]

    pop = r.get("population", {})
    out += ["## A. Population and integrity", "",
            f"- Train n {pop.get('n_train')} (grades {pop.get('train_grade_counts')}); validation n "
            f"{pop.get('n_val')} (grades {pop.get('val_grade_counts')}); train/validation overlap "
            f"{pop.get('train_val_overlap')}.",
            f"- Split sha256 `{pop.get('split_sha256')}`; training membership sha256 "
            f"`{pop.get('train_membership_sha256')}`; six-run pin {pop.get('six_run_pin')}.",
            f"- Library versions: {r.get('versions')}; GPUs {r.get('gpus')}.", ""]
    backbones = r.get("backbones") or {}
    if backbones:
        out += ["| seed | BEST epoch | weights sha256 | sha unchanged | stage fingerprints unchanged "
                "| max Δlogit vs saved | eager vs graph | head(E) vs graph | grade agreement vs "
                "saved | E sha256 | resumed |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s in seeds:
            b = backbones.get(s) or backbones.get(str(s))
            if b:
                out.append(f"| {s} | {b['best_epoch']} | `{b['weights_sha256'][:16]}…` | "
                           f"{b['weights_sha256_unchanged']} | {b['stage_fingerprints_unchanged']} | "
                           f"{b['max_abs_logit_diff_vs_saved']:.2e} | "
                           f"{b['max_abs_logit_diff_eager_vs_graph']:.2e} | "
                           f"{b['max_abs_logit_diff_head_on_E_vs_graph']:.2e} | "
                           f"{b['predicted_grade_agreement_vs_saved']:.4f} | "
                           f"`{b['E_sha256'][:16]}…` | {b.get('resumed', False)} |")
        out.append("")
    smoke = r.get("smoke")
    if smoke:
        out += [f"- Smoke checks (tiny training subset): {smoke}", ""]

    out += ["## B. Fit diagnostics (every binary task; convex L-BFGS)", "",
            "| seed | head | task | n | positives | grades in task | iterations | converged | "
            "max |grad| |", "|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        for head in ("H1", "H2"):
            for task, info in r["fitted_heads"][s][head]["tasks"].items():
                out.append(f"| {s} | {head} | {task} | {info['n']} | {info['n_positive']} | "
                           f"{info['grades_in_task']} | {info['iterations']} | {info['converged']} | "
                           f"{info['final_grad_max_abs']:.1e} |")
    out += ["", "Head fingerprints: " + "; ".join(
        f"{s} H1 `{r['fitted_heads'][s]['H1']['fingerprint'][:12]}` H2 "
        f"`{r['fitted_heads'][s]['H2']['fingerprint'][:12]}`" for s in seeds), ""]

    scalar = [m for m, _ in _REPORT_METRICS]
    out += ["## C. Every metric, per seed and head", "",
            "| seed | head | " + " | ".join(scalar) + " |",
            "|---|---|" + "---|" * len(scalar)]
    for s in seeds:
        for head in ("H0", "H1", "H2"):
            row = _row(table, s, head)
            out.append(f"| {s} | {head} | " + " | ".join(
                str(int(row[m])) if m in _INTEGER_METRICS else _f(float(row[m]))
                for m in scalar) + " |")
    out += ["", "| contrast | " + " | ".join(scalar) + " |", "|---|" + "---|" * len(scalar)]
    for s, d in zip(seeds, r["h2_minus_h1"]):
        out.append(f"| H2 − H1 seed {s} | " + " | ".join(f"{d[m]:+.4f}" for m in scalar) + " |")
    out.append("| **H2 − H1 mean** | " + " | ".join(
        f"{r['h2_minus_h1_mean'][m]:+.4f}" for m in scalar) + " |")
    out.append("| H2 − H1 SD | " + " | ".join(
        f"{r['h2_minus_h1_sd'][m]:.4f}" for m in scalar) + " |")
    out.append("| H1 − H0 mean (refit effect) | " + " | ".join(
        f"{r['h1_minus_h0_mean'][m]:+.4f}" for m in scalar) + " |")
    boot = r["decision"]["bootstrap"]
    out += ["", "Primary Δ bootstrap (2,000 stratified paired resamples, seed "
            f"{boot['seed']}): " + "; ".join(
                f"seed {s}: CI {c[0]:+.4f} to {c[1]:+.4f}, {p:.1%} of resamples > 0"
                for s, c, p in zip(seeds, boot["per_seed_ci"], boot["per_seed_fraction_positive"]))
            + f"; seed-mean CI {boot['mean_delta_ci'][0]:+.4f} to {boot['mean_delta_ci'][1]:+.4f}, "
              f"{boot['mean_delta_fraction_positive']:.1%} > 0.", ""]

    out += ["## D. Confusion matrices (rows = true grade 0–4, columns = predicted 0–4)", ""]
    for s in seeds:
        for head in ("H0", "H1", "H2"):
            cm = _row(table, s, head)["confusion_matrix"]
            out.append(f"- seed {s} {head}: " + " ".join(str(list(map(int, line))) for line in cm))
    out.append("")

    if samples is not None and len(samples):
        out += ["## E. What changed between H1 and H2 (validation images)", "",
                "| seed | grade-4 moved ≤2 → ≥3 | grade-4 moved ≥3 → ≤2 | grade-4 newly called 4 | "
                "grade-3 lost (3 → other) | grade-3 gained | grades 0–2 newly urgent | grades 0–2 no "
                "longer urgent |", "|---|---|---|---|---|---|---|---|"]
        for s in seeds:
            f = samples[samples["seed"] == s]
            g = f["true_grade"].to_numpy()
            a, b = f["H1_pred"].to_numpy(), f["H2_pred"].to_numpy()
            g4, g3, low = g == 4, g == 3, g <= 2
            out.append(f"| {s} | {int(np.sum(g4 & (a <= 2) & (b >= 3)))} | "
                       f"{int(np.sum(g4 & (a >= 3) & (b <= 2)))} | "
                       f"{int(np.sum(g4 & (a != 4) & (b == 4)))} | "
                       f"{int(np.sum(g3 & (a == 3) & (b != 3)))} | "
                       f"{int(np.sum(g3 & (a != 3) & (b == 3)))} | "
                       f"{int(np.sum(low & (a <= 2) & (b >= 3)))} | "
                       f"{int(np.sum(low & (a >= 3) & (b <= 2)))} |")
        out += ["", "## F. The persistent-21 grade-4 images (predicted grade per seed 42/123/2026)",
                "", "| image | H0 | H1 | H2 | H2 q | H1 P(≥3) | H2 P(≥3) |",
                "|---|---|---|---|---|---|---|"]
        persistent = samples[samples["persistent21"]]
        for image_id in sorted(persistent["image_id"].unique()):
            rows = persistent[persistent["image_id"] == image_id].set_index("seed").loc[seeds]
            out.append(f"| `{image_id}` | " + " | ".join(
                "/".join(str(int(v)) for v in rows[c]) for c in ("H0_pred", "H1_pred", "H2_pred"))
                + " | " + " | ".join("/".join(f"{v:.2f}" for v in rows[c])
                                     for c in ("H2_q", "H1_p_ge3", "H2_p_ge3")) + " |")
        out += ["", "## G. Score distributions by true grade (median [IQR], all seeds pooled)", "",
                "| true grade | n | H1 P(≥3) | H2 P(≥3) | H1 P(4) | H2 q |", "|---|---|---|---|---|---|"]
        for grade in range(5):
            f = samples[samples["true_grade"] == grade]

            def q(col):
                v = f[col].to_numpy(np.float64)
                return (f"{np.median(v):.3f} [{np.percentile(v, 25):.3f}–"
                        f"{np.percentile(v, 75):.3f}]") if v.size else "n/a"
            out.append(f"| {grade} | {len(f) // len(seeds)} | {q('H1_p_ge3')} | {q('H2_p_ge3')} | "
                       f"{q('H1_p4')} | {q('H2_q')} |")
        out.append("")
    out += [f"Files: `{REPORT_FILENAME}`, `{RESULTS_FILENAME}`, `per_seed_results.csv`, "
            "`h2_minus_h1.csv`, `per_sample_predictions.csv`, `heads_seed_*.npz`, `E_seed_*.npz`, "
            f"`backbone_seed_*.json`, `{PREREGISTRATION_FILENAME}` in `{r.get('out_dir')}`.", ""]
    return out


def build_report(r, table, samples=None):
    d = r["decision"]
    boot = d["bootstrap"]
    seeds = list(BACKBONE_SEEDS)
    first = r["fitted_heads"][seeds[0]]["H2"]["tasks"]
    n_npdr, n_pdr = first["npdr_task_0"]["n"], first["pdr_route"]["n"]
    n_grade4 = int(sum(_row(table, seeds[0], "H0")["grade4_decoded_counts"]))
    rows = []
    for metric, label in _REPORT_METRICS:
        if metric == "grade4_decoded_le2":
            label = f"PDR decoded ≤ 2 (of {n_grade4})"
        cells = []
        for head in ("H0", "H1", "H2"):
            vals = [_row(table, s, head)[metric] for s in seeds]
            cells.append(" / ".join(str(int(v)) if metric in _INTEGER_METRICS else _f(float(v), 3)
                                    for v in vals))
        dm = r["h2_minus_h1_mean"][metric]
        rows.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} | {dm:+.4f} |")
    guard_lines = [f"| {k} | {v['mean_delta']:+.4f} | {'≥' if v['kind'] == 'min' else '≤'} "
                   f"{v['bound']:+.2f} | {'holds' if v['holds'] else '**fails**'} |"
                   for k, v in d["guardrails"].items()]
    decoded = []
    for s in seeds:
        for head in ("H0", "H1", "H2"):
            c = _row(table, s, head)["grade4_decoded_counts"]
            decoded.append(f"| {s} | {head} | " + " | ".join(str(x) for x in c) + " |")
    verdict = d["verdict"]
    interpretation = {
        "SUPPORTIVE": "The experiment supports the hypothesis that the CORN chain structure "
                      "contributes to PDR under-triage on the frozen representation. A stronger "
                      "claim requires external confirmation.",
        "NOT_SUPPORTIVE": "The experiment does not support the hypothesis that the output "
                          "structure is the main limiting factor on the frozen representation.",
        "TRADE_OFF": "The primary endpoint met its bar, but a guardrail failed (see the guardrail "
                     "table); no clear conclusion is drawn.",
        "INCONCLUSIVE": "The primary endpoint neither met the SUPPORTIVE bar nor fell to the "
                        "NOT_SUPPORTIVE bar; no conclusion is drawn.",
    }[verdict]
    pp = r["parameter_parity"]
    lines = [
        "# C1 — ICDR two-route Stage-8 head vs CORN refit (frozen NO_RACAF E)",
        "",
        f"Version `{r['experiment_version']}` · repo `{r['repo_commit']}` · `{r['out_dir']}`",
        "",
        "C1 is **not** a new ordinal model family: a hurdle / split-first structure is an "
        "established modelling idea. The contribution tested is the mechanism-driven DR "
        "application — CORN chain gating — under a capacity-matched control on an identical "
        "frozen representation. Stages 5–7 were not changed. DDR was not used. APTOS validation "
        "was used for this pre-registered development decision and is not an untouched external "
        "test set; external validation remains future work.",
        "",
        f"Parameters: original CORN {pp['original_corn']['trainable_parameters']}, H1 "
        f"{pp['H1_corn_refit']['trainable_parameters']}, H2 "
        f"{pp['H2_two_route']['trainable_parameters']} (difference "
        f"{pp['difference_H2_minus_H1']}). Training n = {r['population']['n_train']} "
        f"(NPDR route {n_npdr}, PDR route {n_pdr}); validation n = "
        f"{r['population']['n_val']}. L2 = {r['fitting_configuration']['l2']}.",
        "",
        "## Results (per seed 42 / 123 / 2026)",
        "",
        "| Metric | H0 original | H1 CORN refit | H2 two-route | mean H2 − H1 |",
        "|---|---|---|---|---|",
        *rows,
        "",
        "Primary Δ (H2 − H1) per seed: " + ", ".join(
            f"{s}: {v:+.4f} (95% CI {c[0]:+.3f} to {c[1]:+.3f})"
            for s, v, c in zip(seeds, d["per_seed_delta"], boot["per_seed_ci"])),
        f"Mean Δ {d['mean_delta']:+.4f} (SD {d['sd_delta']:.4f}); paired bootstrap 95% CI of the "
        f"mean {boot['mean_delta_ci'][0]:+.4f} to {boot['mean_delta_ci'][1]:+.4f}; positive in "
        f"{d['seeds_positive']}/3 seeds.",
        "",
        "## Guardrails (3-seed mean of H2 − H1)",
        "",
        "| Guardrail | mean Δ | bound | status |", "|---|---|---|---|", *guard_lines,
        "",
        "## Grade-4 validation images by decoded grade (0 / 1 / 2 / 3 / 4)",
        "",
        "| seed | head | 0 | 1 | 2 | 3 | 4 |", "|---|---|---|---|---|---|---|", *decoded,
        "",
        f"## Pre-registered verdict: **{verdict}**",
        "",
        interpretation,
        "",
        "## Limitations",
        "",
        "- One frozen APTOS split; the validation set has 58 grade-4 images and has been used in "
        "earlier analyses of this model, so it is a development set, not an external test.",
        "- The heads are fitted on training-image E from backbones that were trained on those "
        "images, so fits may be over-confident; both heads share this, so the contrast is fair.",
        "- A head-only change cannot test whether E itself was shaped by chain supervision "
        "during end-to-end training; a negative result does not rule that out.",
        "- The persistent-21 images are not established as treated PDR or as label noise.",
    ]
    lines += _detail_sections(r, table, samples)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
