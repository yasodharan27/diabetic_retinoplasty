# ============================================================================================
# PHASE 0 -- GRADE 3 vs GRADE 4 LOCALISATION (read-only; decides GO / NO-GO for Architecture A)
#
# Design: research/Grade3vs4_Architecture_Research/RECOMMENDED_DESIGN.md §2.
#
# WHAT THIS DOES
#   1. TRAIN-FITTED LESION STATISTICS  -- logistic regression on the 10 existing lesion statistics,
#      fitted on the AUTHORITATIVE TRAIN split (grades 3/4 only), scored ONCE on the frozen
#      validation split. Primary metric: Grade-4-vs-Grade-3 AUROC. This is the GO/NO-GO gate.
#   2. FROZEN MODEL SCORE ANALYSIS    -- from the SAVED per-sample predictions of the six frozen
#      runs: is the Grade-3/4 signal specifically lost in the marginal p_gt_3 = p_gt_2 * p_cond_3?
#   3. STACKED REGRESSION BASELINE    -- frozen NO_RACAF model outputs (+/- the 10 lesion
#      statistics), fitted on the training split, scored on validation.
#   4. INTERNAL FEATURE PROBES        -- E, GAP(Stage 05 L), GAP(Stage 06 G) from the same frozen
#      forward pass, same fit-on-train / score-on-validation protocol (read-only, performed because
#      the existing sub-models are directly callable -- no production file is modified).
#
# WHAT THIS NEVER DOES
#   * trains, fine-tunes or re-saves any neural network (inference only, BEST weights read-only);
#   * fits ANYTHING on validation data (validation is scored exactly once per fitted model);
#   * regenerates Stage 02/03/04 caches (only the EXISTING cache archive is extracted; the cache
#     completion routine that could run Stage 03/04 is never called);
#   * writes anywhere except a NEW timestamped directory experiments/Grade3vs4_Phase0/<timestamp>/;
#   * reads or touches the running Grade-4 auxiliary-loss experiment.
#
# RUN IT IN ITS OWN COLAB RUNTIME, NOT the one running the auxiliary-loss experiment.
# GPU: T4 REQUIRED. Not for speed (inference is 487 images x 3 seeds): without a GPU the model
# builder falls back to float32, while the saved predictions were produced under mixed_float16,
# so the inference-parity check against per_sample_best.csv would not be like-for-like.
# Copy-paste into one Colab cell, or upload and `%run` it.
# ============================================================================================

import datetime
import json
import os
import posixpath
import re
import subprocess
import sys

import numpy as np

# ============================================================================================
# PRE-REGISTERED CONFIGURATION -- fixed before any result is seen. Do not edit after running.
# ============================================================================================

PHASE0_VERSION = "phase0-grade3vs4-v1"
ANALYSIS_SEED = 20260923
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
SEVERE_NPDR, PDR = 3, 4
FOREGROUND_THRESHOLD = 0.5                     # identical to the §6/§11 diagnostic
KAPPA_SHORT_NAMES = ["MA", "HE", "EX", "SE"]    # authoritative Stage 04 channel order
LESION_VARIABLES = ([f"lesion_{n}_mean_prob" for n in KAPPA_SHORT_NAMES]
                    + [f"lesion_{n}_foreground_fraction" for n in KAPPA_SHORT_NAMES]
                    + ["lesion_total_mean_prob", "lesion_total_foreground_fraction"])
LOG_EPSILON = 1e-4                             # transform log10(x + 1e-4), RECOMMENDED_DESIGN §3.2
LOGREG_C = 1.0                                 # fixed; no tuning
PROBE_PCA_COMPONENTS = 16                      # dimensionality control for 256/1152-d probes
STATS_REPRODUCTION_TOLERANCE = 1e-6

# GO / NO-GO for Architecture A (pre-registered; analysis 1, primary variant only):
#   GO           : AUROC >= 0.62  AND  bootstrap 95% CI lower bound > 0.50
#   INCONCLUSIVE : AUROC >= 0.62  BUT  CI lower bound <= 0.50 (signal not separable from chance)
#   NO-GO        : AUROC <  0.62
GO_AUROC = 0.62
GO_CI_LOWER = 0.50

# Analysis 2 classification (descriptive qualifier, not the gate):
#   SPECIFIC_LOSS_IN_MARGINAL : mean AUROC(p_cond_3) - AUROC(p_gt_3) >= +0.05 and > 0 in >= 5/6 runs
#   NO_SPECIFIC_LOSS          : mean difference < +0.02
#   PARTIAL                   : otherwise
MARGINAL_LOSS_MEAN = 0.05
MARGINAL_LOSS_MIN_RUNS = 5
MARGINAL_NO_LOSS_MEAN = 0.02

# Analysis 3 qualifier: lesion statistics "improve" the frozen model score if the stacked model
# beats BOTH the frozen p_gt_3 score and the train-fitted model-only regression by a mean of
# >= +0.03 AUROC, positive in 3/3 NO_RACAF seeds (same scale as the §12 and Architecture A rules).
STACK_IMPROVE_MEAN = 0.03
STACK_IMPROVE_MIN_SEEDS = 3

# Analysis 4 (descriptive) localisation margins, RECOMMENDED_DESIGN §2 table.
PROBE_MARGIN = 0.05

INFERENCE_ARM = "NO_RACAF"                     # Architecture A is specified on the NO_RACAF baseline
INFERENCE_BATCH = 8
PARITY_LOGIT_TOLERANCE = 0.05                  # inference vs saved per_sample_best.csv logits

REPO_URL = "https://github.com/yasodharan27/diabetic_retinoplasty.git"
REPO_DIR = "/content/diabetic_retinoplasty"
BRANCH = "main"


# ============================================================================================
# PURE ANALYSIS FUNCTIONS (no Drive, no TensorFlow) -- unit-testable in isolation
# ============================================================================================

def lesion_statistics(lesion):
    """The exact 10 variables of the §6/§11 diagnostic, from one (H, W, 4) lesion map."""
    lesion = np.asarray(lesion, dtype=np.float32)
    record = {}
    for channel, name in enumerate(KAPPA_SHORT_NAMES):
        channel_map = lesion[..., channel]
        record[f"lesion_{name}_mean_prob"] = float(channel_map.mean())
        record[f"lesion_{name}_foreground_fraction"] = float(
            (channel_map > FOREGROUND_THRESHOLD).mean())
    record["lesion_total_mean_prob"] = float(lesion.mean())
    record["lesion_total_foreground_fraction"] = float((lesion > FOREGROUND_THRESHOLD).mean())
    return record


def log_transform(x):
    return np.log10(np.asarray(x, dtype=np.float64) + LOG_EPSILON)


def make_logreg(pca_components=None):
    """StandardScaler -> [PCA] -> L2 logistic regression, C fixed. Every step is fitted on the
    training rows only by `fit_and_score`."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    steps = [("scale", StandardScaler())]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=ANALYSIS_SEED)))
    steps.append(("logreg", LogisticRegression(C=LOGREG_C, max_iter=10000)))
    return Pipeline(steps)


def fit_and_score(x_train, y_train, x_val, pca_components=None):
    """Fits on TRAIN only; returns (validation scores, fitted pipeline)."""
    x_train = np.asarray(x_train, dtype=np.float64)
    x_val = np.asarray(x_val, dtype=np.float64)
    if pca_components is not None:
        pca_components = int(min(pca_components, x_train.shape[1], x_train.shape[0] - 1))
    model = make_logreg(pca_components)
    model.fit(x_train, np.asarray(y_train, dtype=int))
    return model.predict_proba(x_val)[:, 1], model


def hanley_mcneil_se(auc, n_pos, n_neg):
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) \
        / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def stratified_bootstrap_indices(y, n_boot=N_BOOTSTRAP, seed=ANALYSIS_SEED):
    """One fixed list of stratified resamples, reused for every score, so paired differences
    are computed on identical resamples."""
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    return [np.concatenate([rng.choice(pos, pos.size, replace=True),
                            rng.choice(neg, neg.size, replace=True)]) for _ in range(n_boot)]


def auroc_auprc(y, score, boot):
    """Point AUROC/AUPRC plus percentile bootstrap CIs and the Hanley-McNeil SE."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=np.float64)
    auc = float(roc_auc_score(y, score))
    ap = float(average_precision_score(y, score))
    aucs = np.array([roc_auc_score(y[i], score[i]) for i in boot])
    aps = np.array([average_precision_score(y[i], score[i]) for i in boot])
    lo, hi = (1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100
    return {
        "n": int(y.size), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "auroc": auc, "auroc_ci": [float(np.percentile(aucs, lo)), float(np.percentile(aucs, hi))],
        "auroc_hanley_mcneil_se": hanley_mcneil_se(auc, int(y.sum()), int((1 - y).sum())),
        "auprc": ap, "auprc_ci": [float(np.percentile(aps, lo)), float(np.percentile(aps, hi))],
        "auprc_baseline_prevalence": float(y.mean()),
    }


def paired_auroc_difference(y, score_a, score_b, boot):
    """AUROC(a) - AUROC(b) on identical resamples."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, dtype=int)
    a = np.asarray(score_a, dtype=np.float64)
    b = np.asarray(score_b, dtype=np.float64)
    point = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    diffs = np.array([roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]) for i in boot])
    lo, hi = (1 - CI_LEVEL) / 2 * 100, (1 + CI_LEVEL) / 2 * 100
    return {"difference": point,
            "ci": [float(np.percentile(diffs, lo)), float(np.percentile(diffs, hi))],
            "fraction_of_resamples_positive": float((diffs > 0).mean())}


def gate_verdict(auroc, ci_lower):
    if auroc < GO_AUROC:
        return "NO-GO"
    if ci_lower <= GO_CI_LOWER:
        return "INCONCLUSIVE"
    return "GO"


def classify_marginal_loss(differences):
    differences = np.asarray(differences, dtype=np.float64)
    mean = float(differences.mean())
    positive = int((differences > 0).sum())
    if mean >= MARGINAL_LOSS_MEAN and positive >= MARGINAL_LOSS_MIN_RUNS:
        label = "SPECIFIC_LOSS_IN_MARGINAL"
    elif mean < MARGINAL_NO_LOSS_MEAN:
        label = "NO_SPECIFIC_LOSS"
    else:
        label = "PARTIAL"
    return {"label": label, "mean_difference": mean, "runs_positive": positive,
            "runs": int(differences.size)}


def classify_stack_improvement(vs_frozen, vs_model_only):
    vs_frozen = np.asarray(vs_frozen, dtype=np.float64)
    vs_model_only = np.asarray(vs_model_only, dtype=np.float64)
    ok_frozen = (vs_frozen.mean() >= STACK_IMPROVE_MEAN
                 and int((vs_frozen > 0).sum()) >= STACK_IMPROVE_MIN_SEEDS)
    ok_model = (vs_model_only.mean() >= STACK_IMPROVE_MEAN
                and int((vs_model_only > 0).sum()) >= STACK_IMPROVE_MIN_SEEDS)
    return {"improves": bool(ok_frozen and ok_model),
            "mean_gain_over_frozen_p_gt_3": float(vs_frozen.mean()),
            "seeds_positive_over_frozen": int((vs_frozen > 0).sum()),
            "mean_gain_over_model_only_regression": float(vs_model_only.mean()),
            "seeds_positive_over_model_only": int((vs_model_only > 0).sum())}


def localise(lesion_probe, probe_l, probe_e, frozen_p_gt_3):
    """RECOMMENDED_DESIGN §2 interpretation table, per seed. Descriptive only."""
    labels = []
    if probe_l - probe_e >= PROBE_MARGIN:
        labels.append("lost_at_stage07_aggregation")
    if probe_l <= lesion_probe - PROBE_MARGIN:
        labels.append("weak_already_in_stage05_features")
    if probe_e >= lesion_probe - 0.02 and frozen_p_gt_3 <= probe_e - PROBE_MARGIN:
        labels.append("present_in_E_not_used_by_head_or_decode")
    return labels or ["no_pattern_meets_margin"]


def parse_duplicate_val_ids(markdown_text):
    """Validation ids from the pinned table in RACAF_Duplicate_Contamination_Audit.md
    (`| `val_id` | grade | `train twin` [g] | ...`)."""
    ids, seen = [], set()
    for line in markdown_text.splitlines():
        match = re.match(r"^\|\s*`([0-9a-f]{12})`\s*\|\s*(\d)\s*\|(.*)$", line.strip())
        if match and match.group(1) not in seen:
            seen.add(match.group(1))
            twins = re.findall(r"`([0-9a-f]{12})`\s*\[(\d)\]", match.group(3))
            ids.append({"val_id": match.group(1), "val_grade": int(match.group(2)),
                        "train_twins": [{"train_id": t, "grade": int(g)} for t, g in twins]})
    return ids


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


# ============================================================================================
# MAIN (Colab): bootstrap, locate read-only artifacts, run the four analyses, write outputs
# ============================================================================================

def main():
    # ---- [0] BOOTSTRAP (project standard) ----
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        subprocess.run(["git", "clone", "--branch", BRANCH, REPO_URL, REPO_DIR], check=True)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "origin", BRANCH], check=True)
    for _p in (REPO_DIR, os.path.join(REPO_DIR, "colab", "common")):
        if _p not in sys.path:
            sys.path.insert(0, _p)

    import setup as colab_setup
    colab_setup.setup()
    import colab_config
    import verify_environment
    env_report = verify_environment.verify_all(
        repo_dir=colab_config.REPO_DIR, drive_mount_point=colab_config.DRIVE_MOUNT_POINT,
        requirements_path=os.path.join(colab_config.REPO_DIR, "requirements.txt"),
        require_gpu=True)

    import gc
    import shutil

    import pandas as pd
    import tensorflow as tf

    import config
    import corn
    import improved_training_data as itd
    import joint_training_dataset as jtd
    import local_feature_extraction_dataset as lfed
    import multiseed_runs as msr
    from lesion_segmentation_dataset import LESION_CLASSES
    from training import checkpointing as ckpt

    assert [c for c in LESION_CLASSES] and len(LESION_CLASSES) == 4, LESION_CLASSES
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise RuntimeError("No GPU: select a T4 runtime (mixed_float16 parity with the saved "
                           "predictions requires it).")
    print(f"multiseed_runs protocol {msr.PROTOCOL_VERSION}; GPUs: {gpus}")

    # ---- [1] PATHS: read-only inputs, one new output directory ----
    experiments_root = colab_config.DRIVE.experiments_root
    six_run_root = posixpath.join(experiments_root, "ImprovedTraining")
    six_run_id = "improved_multiseed_2026_09"
    six_run_dir = msr.experiment_root(six_run_root, six_run_id)
    pdr_diag_csv = posixpath.join(experiments_root, "PDR_Grade4_Diagnostic",
                                  "2026-09-22_13-30-45", "per_image_variables.csv")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = posixpath.join(experiments_root, "Grade3vs4_Phase0", timestamp)

    protected = (six_run_dir,
                 posixpath.join(experiments_root, "Grade4AuxLoss"),
                 posixpath.join(experiments_root, "ImprovedTrainingC1Control"),
                 posixpath.join(experiments_root, "ImprovedTrainingC3Kappa"),
                 posixpath.join(experiments_root, "PDR_Grade4_Diagnostic"),
                 colab_config.DRIVE.experiment_dir("FinalClassification"))
    for root in protected:
        assert posixpath.commonpath([posixpath.normpath(out_dir), posixpath.normpath(root)]) \
            != posixpath.normpath(root), f"{out_dir} is under read-only {root}; refusing."
    os.makedirs(out_dir, exist_ok=False)
    print("Writes only to:", out_dir)

    results = {"phase0_version": PHASE0_VERSION, "timestamp": timestamp, "out_dir": out_dir,
               "preregistered": {
                   "analysis_seed": ANALYSIS_SEED, "n_bootstrap": N_BOOTSTRAP,
                   "go_auroc": GO_AUROC, "go_ci_lower": GO_CI_LOWER, "logreg_C": LOGREG_C,
                   "log_transform": f"log10(x + {LOG_EPSILON})",
                   "probe_pca_components": PROBE_PCA_COMPONENTS,
                   "marginal_loss_rule": [MARGINAL_LOSS_MEAN, MARGINAL_LOSS_MIN_RUNS,
                                          MARGINAL_NO_LOSS_MEAN],
                   "stack_improve_rule": [STACK_IMPROVE_MEAN, STACK_IMPROVE_MIN_SEEDS],
                   "inference_arm": INFERENCE_ARM},
               "environment": {"gpus": [g.name for g in gpus],
                               "repo_commit": subprocess.run(
                                   ["git", "-C", REPO_DIR, "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip()}}

    # ---- [2] SPLIT + POPULATION (authoritative, verified) ----
    train_entries, val_entries, split_sha = msr.verify_split()
    assert split_sha == msr.EXPECTED_SPLIT_SHA256
    with open(posixpath.join(six_run_dir, "experiment_manifest.json")) as fh:
        six_run_population = json.load(fh)

    local_cache_dir = "/content/cache/local_feature_extraction"
    local_racaf_cache_dir = "/content/cache/racaf"
    marker = "/content/cache/.multiseed_archive_extracted.json"
    archive_dir = os.path.join(os.path.dirname(config.LOCAL_FEATURE_RESULTS_DIR), "cache_archive")
    if not os.path.exists(marker):
        import joint_cache_archive as jca
        plan = jca.plan_extraction(archive_dir=archive_dir, cache_dir=local_cache_dir,
                                   racaf_cache_dir=local_racaf_cache_dir)
        jca.print_extraction_plan(plan)
        if plan["drive_unreachable"] or not plan["fits"]:
            raise RuntimeError("Refusing to extract the cache archive (unreachable or no space).")
        extract = jca.extract_archive(archive_dir=archive_dir, cache_dir=local_cache_dir,
                                      racaf_cache_dir=local_racaf_cache_dir,
                                      min_free_bytes=plan["required_bytes"])
        jca.print_extract(extract)
        if extract["corrupt"] or extract["drive_unreachable"]:
            raise RuntimeError("Archive extraction did not complete. Re-run.")
        with open(marker, "w") as fh:
            json.dump({"extracted": datetime.datetime.now().isoformat(timespec="seconds")}, fh)
    # Deliberately NOT itd.complete_local_cache(): that routine may run Stage 03/04 for a missing
    # entry. Phase 0 must never regenerate caches, so a population mismatch is fatal instead.
    cached_train = itd.locally_cached_entries(train_entries, local_cache_dir, local_racaf_cache_dir)
    cached_val = itd.locally_cached_entries(val_entries, local_cache_dir, local_racaf_cache_dir)
    if (len(cached_train), len(cached_val)) != (int(six_run_population["n_train_yielded"]),
                                                 int(six_run_population["n_val_yielded"])):
        raise RuntimeError(
            f"Extracted cache yields {len(cached_train)} train / {len(cached_val)} val, but the "
            f"six-run pin is {six_run_population['n_train_yielded']} / "
            f"{six_run_population['n_val_yielded']}. Phase 0 does not regenerate caches.")

    train34 = [(i, int(g)) for i, g in cached_train if int(g) in (SEVERE_NPDR, PDR)]
    val34 = [(i, int(g)) for i, g in cached_val if int(g) in (SEVERE_NPDR, PDR)]
    y_train = np.array([int(g == PDR) for _, g in train34])
    y_val = np.array([int(g == PDR) for _, g in val34])
    val_ids = [i for i, _ in val34]
    results["population"] = {
        "split_sha256": split_sha, "n_train_cached": len(cached_train),
        "n_val_cached": len(cached_val),
        "train_grade3": int((y_train == 0).sum()), "train_grade4": int(y_train.sum()),
        "val_grade3": int((y_val == 0).sum()), "val_grade4": int(y_val.sum())}
    print("Population:", results["population"])

    # duplicates (pinned list, parsed from the committed audit report)
    with open(os.path.join(REPO_DIR, "docs", "experiments",
                           "RACAF_Duplicate_Contamination_Audit.md"), encoding="utf-8") as fh:
        duplicates = parse_duplicate_val_ids(fh.read())
    duplicate_val_ids = {d["val_id"] for d in duplicates}
    dup34 = [d for d in duplicates if d["val_id"] in set(val_ids)]
    results["duplicates"] = {"pinned_pairs": len(duplicates),
                             "in_val_grade34": len(dup34), "val_grade34_entries": dup34}
    keep_nodup = np.array([i not in duplicate_val_ids for i in val_ids])

    boot = stratified_bootstrap_indices(y_val)
    boot_nodup = stratified_bootstrap_indices(y_val[keep_nodup])

    # ---- [3] LESION STATISTICS (from the extracted cache; verified against §6's saved CSV) ----
    def stats_for(entries, split_name):
        rows = []
        for index, (id_code, grade) in enumerate(entries):
            lesion = np.load(lfed._cache_path(local_cache_dir, id_code, "lesion",
                                              jtd.STAGE5_IMAGE_SIZE))
            row = {"image_id": id_code, "split": split_name, "true_grade": grade}
            row.update(lesion_statistics(lesion))
            rows.append(row)
            if (index + 1) % 100 == 0:
                print(f"  {split_name}: {index + 1}/{len(entries)} lesion maps read")
        return pd.DataFrame(rows)

    stats_train = stats_for(train34, "train")
    stats_val = stats_for(val34, "val")
    stats_all = pd.concat([stats_train, stats_val], ignore_index=True)
    stats_all.to_csv(posixpath.join(out_dir, "lesion_statistics_grade34.csv"), index=False)

    reproduction = {"reference": pdr_diag_csv, "available": os.path.exists(pdr_diag_csv)}
    if reproduction["available"]:
        reference = pd.read_csv(pdr_diag_csv, dtype={"image_id": str})
        merged = stats_val.merge(reference[["image_id"] + LESION_VARIABLES], on="image_id",
                                 suffixes=("", "_ref"), how="inner")
        max_diff = float(max(np.abs(merged[v] - merged[f"{v}_ref"]).max()
                             for v in LESION_VARIABLES))
        reproduction.update(n_matched=int(len(merged)), max_abs_diff=max_diff,
                            reproduced=bool(len(merged) == len(stats_val)
                                            and max_diff <= STATS_REPRODUCTION_TOLERANCE))
        if not reproduction["reproduced"]:
            raise RuntimeError(f"Lesion statistics do not reproduce §6's saved CSV: {reproduction}")
        # direction check across grades 2/3/4 on the full validation population (saved CSV)
        medians_val = reference.groupby("true_grade")[LESION_VARIABLES].median()
    else:
        medians_val = None
    results["lesion_statistics_reproduction"] = reproduction
    print("Statistics reproduction:", reproduction)

    grade_medians = []
    for split_name, frame in (("train", stats_train), ("val", stats_val)):
        for grade, group in frame.groupby("true_grade"):
            grade_medians.append({"split": split_name, "true_grade": int(grade), "n": len(group),
                                  **{v: float(group[v].median()) for v in LESION_VARIABLES}})
    if medians_val is not None:
        for grade, row in medians_val.iterrows():
            grade_medians.append({"split": "val_all_grades_saved_csv", "true_grade": int(grade),
                                  **{v: float(row[v]) for v in LESION_VARIABLES}})
    pd.DataFrame(grade_medians).to_csv(posixpath.join(out_dir, "lesion_grade_medians.csv"),
                                       index=False)

    x_train_raw = stats_train[LESION_VARIABLES].to_numpy(np.float64)
    x_val_raw = stats_val[LESION_VARIABLES].to_numpy(np.float64)
    assert stats_val["image_id"].tolist() == val_ids

    # ============================================================================================
    # ANALYSIS 1 -- TRAIN-FITTED LESION STATISTICS (the gate)
    # ============================================================================================
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    a1 = {}
    variants = {"primary_log10_standardised": (log_transform(x_train_raw), log_transform(x_val_raw)),
                "sensitivity_untransformed_standardised": (x_train_raw, x_val_raw)}
    lesion_val_scores = {}
    for name, (xtr, xva) in variants.items():
        scores, fitted = fit_and_score(xtr, y_train, xva)
        lesion_val_scores[name] = scores
        entry = auroc_auprc(y_val, scores, boot)
        entry["duplicate_excluded"] = auroc_auprc(y_val[keep_nodup], scores[keep_nodup],
                                                  boot_nodup)
        cv = cross_val_predict(make_logreg(), xtr, y_train, method="predict_proba",
                               cv=StratifiedKFold(5, shuffle=True, random_state=ANALYSIS_SEED))[:, 1]
        entry["train_internal_5fold_auroc"] = float(roc_auc_score(y_train, cv))
        entry["coefficients_standardised"] = dict(zip(
            LESION_VARIABLES, fitted.named_steps["logreg"].coef_[0].tolist()))
        a1[name] = entry
    primary = a1["primary_log10_standardised"]
    verdict = gate_verdict(primary["auroc"], primary["auroc_ci"][0])
    a1["gate"] = {"variant": "primary_log10_standardised", "auroc": primary["auroc"],
                  "ci_lower": primary["auroc_ci"][0], "verdict": verdict}

    univariate = []
    for j, v in enumerate(LESION_VARIABLES):
        univariate.append({"variable": v,
                           "train_auroc_g4_vs_g3": float(roc_auc_score(y_train, x_train_raw[:, j])),
                           "val_auroc_g4_vs_g3": float(roc_auc_score(y_val, x_val_raw[:, j])),
                           "train_median_g3": float(np.median(x_train_raw[y_train == 0, j])),
                           "train_median_g4": float(np.median(x_train_raw[y_train == 1, j])),
                           "coef_primary": a1["primary_log10_standardised"][
                               "coefficients_standardised"][v]})
    pd.DataFrame(univariate).to_csv(posixpath.join(out_dir, "analysis1_univariate.csv"),
                                    index=False)
    pd.DataFrame([{"variant": k, "auroc": e["auroc"], "auroc_ci_low": e["auroc_ci"][0],
                   "auroc_ci_high": e["auroc_ci"][1], "auprc": e["auprc"],
                   "auprc_ci_low": e["auprc_ci"][0], "auprc_ci_high": e["auprc_ci"][1],
                   "auprc_baseline": e["auprc_baseline_prevalence"],
                   "hanley_mcneil_se": e["auroc_hanley_mcneil_se"],
                   "dup_excluded_auroc": e["duplicate_excluded"]["auroc"],
                   "train_internal_5fold_auroc": e["train_internal_5fold_auroc"]}
                  for k, e in a1.items() if k != "gate"]).to_csv(
        posixpath.join(out_dir, "analysis1_lesion_probe.csv"), index=False)
    results["analysis1_train_fitted_lesion_statistics"] = a1
    print(f"ANALYSIS 1: val AUROC {primary['auroc']:.4f} CI {primary['auroc_ci']} -> {verdict}")

    # ============================================================================================
    # ANALYSIS 2 -- FROZEN MODEL SCORES FROM SAVED PREDICTIONS (six runs, validation, n = 97)
    # ============================================================================================
    a2_rows, frozen_frames = [], {}
    for seed in msr.RUN_SEEDS:
        for arm in msr.ARMS:
            label = f"{arm}/seed_{seed}"
            path = posixpath.join(msr.run_dir(six_run_root, six_run_id, arm, seed), "evaluation",
                                  "per_sample_best.csv")
            frame = pd.read_csv(path, dtype={"image_id": str})
            frozen_frames[label] = frame
            sub = frame.set_index("image_id").loc[val_ids]
            assert (sub["true_grade"].to_numpy() == np.array([g for _, g in val34])).all()
            p2 = sub["p_gt_2"].to_numpy(np.float64)
            p3 = sub["p_gt_3"].to_numpy(np.float64)
            pc3 = (sub["p_cond_3"].to_numpy(np.float64) if "p_cond_3" in sub.columns
                   else p3 / np.clip(p2, 1e-12, None))
            if "p_cond_3" in sub.columns:
                recon = float(np.max(np.abs(p2 * pc3 - p3)))
                assert recon < 1e-5, f"{label}: p_gt_3 != p_gt_2 * p_cond_3 (max {recon})"
            expected_grade = sum(sub[f"p_gt_{k}"].to_numpy(np.float64) for k in range(4))
            scores = {"p_gt_3_marginal": p3, "p_cond_3_conditional": pc3, "p_gt_2": p2,
                      "p_gt_1": sub["p_gt_1"].to_numpy(np.float64),
                      "logit_3": sub["logit_3"].to_numpy(np.float64),
                      "class_prob_4": sub["class_prob_4"].to_numpy(np.float64),
                      "expected_grade_sum_p_gt": expected_grade}
            row = {"run": label, "arm": arm, "seed": seed}
            for name, s in scores.items():
                row[f"auroc_{name}"] = float(roc_auc_score(y_val, s))
            diff = paired_auroc_difference(y_val, pc3, p3, boot)
            row["conditional_minus_marginal"] = diff["difference"]
            row["conditional_minus_marginal_ci_low"] = diff["ci"][0]
            row["conditional_minus_marginal_ci_high"] = diff["ci"][1]
            row["median_p_gt_2_grade3"] = float(np.median(p2[y_val == 0]))
            row["median_p_gt_2_grade4"] = float(np.median(p2[y_val == 1]))
            a2_rows.append(row)
    a2 = pd.DataFrame(a2_rows)
    a2.to_csv(posixpath.join(out_dir, "analysis2_frozen_score_components.csv"), index=False)
    a2_class = classify_marginal_loss(a2["conditional_minus_marginal"].to_numpy())
    results["analysis2_frozen_model_scores"] = {
        "per_run": a2_rows, "classification": a2_class,
        "mean_auroc": {c: float(a2[c].mean()) for c in a2.columns if c.startswith("auroc_")}}
    print("ANALYSIS 2:", a2_class)

    # ============================================================================================
    # ANALYSES 3 + 4 -- FROZEN NO_RACAF INFERENCE (BEST weights, read-only) -> stacking + probes
    # ============================================================================================
    def extract(model, entries):
        stage5 = model.get_layer("local_feature_extraction_adaptive_multi_kernel_cnn")
        stage6 = model.get_layer("global_feature_extraction_dual_scale_swin")
        stage7 = model.get_layer("feature_fusion_adaptive_cross_attention")
        head = model.get_layer("corn")
        out = {"logits": [], "logits_eager": [], "E": [], "gap_L": [], "gap_G": []}
        for start in range(0, len(entries), INFERENCE_BATCH):
            batch = entries[start:start + INFERENCE_BATCH]
            samples = [itd.load_cached_sample(i, g, local_cache_dir, local_racaf_cache_dir,
                                              False, None) for i, g in batch]
            s5 = np.stack([s["stage5_input"] for s in samples])
            s6 = np.stack([s["stage6_input"] for s in samples])
            rel = np.array([s["reliability"] for s in samples], np.float32).reshape(-1, 1)
            out["logits"].append(np.asarray(model.predict_on_batch([s5, s6, rel]), np.float64))
            local = stage5(s5, training=False)
            global_ = stage6(s6, training=False)
            fused = stage7([local, global_], training=False)
            out["logits_eager"].append(np.asarray(head(fused, training=False), np.float64))
            out["E"].append(np.asarray(fused, np.float64))
            out["gap_L"].append(np.asarray(tf.reduce_mean(tf.cast(local, tf.float32), axis=[1, 2])))
            out["gap_G"].append(np.asarray(tf.reduce_mean(tf.cast(global_, tf.float32), axis=1)))
        return {k: np.concatenate(v, axis=0) for k, v in out.items()}

    a3_rows, probe_rows, parity_rows, output_rows = [], [], [], []
    lesion_primary_val = lesion_val_scores["primary_log10_standardised"]
    lt_train, lt_val = log_transform(x_train_raw), log_transform(x_val_raw)
    for seed in msr.RUN_SEEDS:
        label = f"{INFERENCE_ARM}/seed_{seed}"
        run_dir = msr.run_dir(six_run_root, six_run_id, INFERENCE_ARM, seed)
        slot_dir, pointer = msr.read_best(run_dir)
        assert slot_dir is not None, f"{label}: no BEST checkpoint"
        gc.collect()
        tf.keras.backend.clear_session()
        model = msr.build_arm_model(INFERENCE_ARM, seed, verbose=0)
        ckpt.load_model_weights_only(model, os.path.join(slot_dir, ckpt.MODEL_WEIGHTS_FILENAME))
        print(f"{label}: BEST epoch {pointer['epoch']} loaded (read-only); running inference")
        tr = extract(model, train34)
        va = extract(model, val34)
        del model

        saved = frozen_frames[label].set_index("image_id").loc[val_ids]
        saved_logits = saved[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64)
        parity = {"run": label,
                  "max_abs_logit_diff_vs_saved": float(np.max(np.abs(va["logits"] - saved_logits))),
                  "max_abs_logit_diff_eager_vs_graph": float(max(
                      np.max(np.abs(tr["logits_eager"] - tr["logits"])),
                      np.max(np.abs(va["logits_eager"] - va["logits"])))),
                  "predicted_grade_agreement_vs_saved": float(np.mean(
                      corn.decode_logits(va["logits"])["predicted_grade"]
                      == saved["predicted_grade"].to_numpy()))}
        parity["ok"] = bool(parity["max_abs_logit_diff_vs_saved"] <= PARITY_LOGIT_TOLERANCE
                            and parity["max_abs_logit_diff_eager_vs_graph"] <= PARITY_LOGIT_TOLERANCE)
        parity_rows.append(parity)
        if not parity["ok"]:
            raise RuntimeError(f"Inference does not reproduce the saved predictions: {parity}")

        for split_name, ids_, grades_, arr in (("train", [i for i, _ in train34],
                                                [g for _, g in train34], tr),
                                               ("val", val_ids, [g for _, g in val34], va)):
            for n, (i, g) in enumerate(zip(ids_, grades_)):
                output_rows.append({"run": label, "split": split_name, "image_id": i,
                                    "true_grade": g,
                                    **{f"logit_{k}": float(arr["logits"][n, k]) for k in range(4)}})

        # --- analysis 3: stacking (train-fitted; validation scored once) ---
        frozen_val = corn.decode_logits(va["logits"])["p_cum"][:, 3].astype(np.float64)
        model_only, _ = fit_and_score(tr["logits"], y_train, va["logits"])
        stacked, _ = fit_and_score(np.hstack([tr["logits"], lt_train]), y_train,
                                   np.hstack([va["logits"], lt_val]))
        row = {"run": label, "seed": seed,
               "auroc_frozen_p_gt_3": float(roc_auc_score(y_val, frozen_val)),
               "auroc_model_only_logreg_on_logits": float(roc_auc_score(y_val, model_only)),
               "auroc_lesion_only_logreg": float(roc_auc_score(y_val, lesion_primary_val)),
               "auroc_stacked_logits_plus_lesion": float(roc_auc_score(y_val, stacked))}
        for tag, other in (("vs_frozen_p_gt_3", frozen_val), ("vs_model_only", model_only),
                           ("vs_lesion_only", lesion_primary_val)):
            d = paired_auroc_difference(y_val, stacked, other, boot)
            row[f"stacked_gain_{tag}"] = d["difference"]
            row[f"stacked_gain_{tag}_ci_low"], row[f"stacked_gain_{tag}_ci_high"] = d["ci"]
        stacked_ci = auroc_auprc(y_val, stacked, boot)
        row["stacked_auprc"] = stacked_ci["auprc"]
        row["stacked_auroc_ci_low"], row["stacked_auroc_ci_high"] = stacked_ci["auroc_ci"]
        a3_rows.append(row)

        # --- analysis 4: internal feature probes (same protocol) ---
        probes = {}
        for rep in ("E", "gap_L", "gap_G"):
            scores, _ = fit_and_score(tr[rep], y_train, va[rep],
                                      pca_components=PROBE_PCA_COMPONENTS)
            probes[rep] = float(roc_auc_score(y_val, scores))
        probe_rows.append({"run": label, "seed": seed,
                           "probe_lesion_statistics_10d": row["auroc_lesion_only_logreg"],
                           "probe_gap_L_stage05_pca16": probes["gap_L"],
                           "probe_E_stage07_pca16": probes["E"],
                           "probe_gap_G_stage06_pca16": probes["gap_G"],
                           "probe_logits_4d": row["auroc_model_only_logreg_on_logits"],
                           "frozen_p_gt_3": row["auroc_frozen_p_gt_3"],
                           "localisation": ";".join(localise(
                               row["auroc_lesion_only_logreg"], probes["gap_L"], probes["E"],
                               row["auroc_frozen_p_gt_3"]))})
        print(f"{label}: stacked {row['auroc_stacked_logits_plus_lesion']:.4f} vs frozen "
              f"{row['auroc_frozen_p_gt_3']:.4f}; probes L={probes['gap_L']:.3f} "
              f"E={probes['E']:.3f} G={probes['gap_G']:.3f}")

    a3 = pd.DataFrame(a3_rows)
    a3.to_csv(posixpath.join(out_dir, "analysis3_stacked_regression.csv"), index=False)
    probes_df = pd.DataFrame(probe_rows)
    probes_df.to_csv(posixpath.join(out_dir, "analysis4_feature_probes.csv"), index=False)
    pd.DataFrame(parity_rows).to_csv(posixpath.join(out_dir, "inference_parity.csv"), index=False)
    pd.DataFrame(output_rows).to_csv(posixpath.join(out_dir, "frozen_logits_grade34.csv"),
                                     index=False)
    a3_class = classify_stack_improvement(a3["stacked_gain_vs_frozen_p_gt_3"].to_numpy(),
                                          a3["stacked_gain_vs_model_only"].to_numpy())
    results["analysis3_stacked_regression"] = {"per_seed": a3_rows, "classification": a3_class}
    results["analysis4_feature_probes"] = {
        "per_seed": probe_rows,
        "mean": {c: float(probes_df[c].mean()) for c in probes_df.columns if c.startswith(("probe_", "frozen_"))}}
    results["inference_parity"] = parity_rows

    # ============================================================================================
    # REPORT
    # ============================================================================================
    results["final_verdict_architecture_A"] = verdict
    with open(posixpath.join(out_dir, "phase0_results.json"), "w") as fh:
        json.dump(to_jsonable(results), fh, indent=2)
    report = build_report(results, a2, a3, probes_df)
    with open(posixpath.join(out_dir, "PHASE0_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    print(report)
    print("\nAll outputs written to:", out_dir)
    return results


def build_report(r, a2, a3, probes):
    a1 = r["analysis1_train_fitted_lesion_statistics"]
    p = a1["primary_log10_standardised"]
    s = a1["sensitivity_untransformed_standardised"]
    pop = r["population"]
    c2 = r["analysis2_frozen_model_scores"]["classification"]
    c3 = r["analysis3_stacked_regression"]["classification"]
    verdict = r["final_verdict_architecture_A"]
    lines = [
        "# Phase 0 — Grade 3 vs Grade 4 localisation (read-only)",
        "",
        f"Version `{r['phase0_version']}` · run `{r['timestamp']}` · repo commit "
        f"`{r['environment']['repo_commit']}` · outputs `{r['out_dir']}`",
        "",
        "No neural network was trained. Nothing was fitted on validation data. No cache was "
        "regenerated. Frozen BEST weights were only read.",
        "",
        f"Population: train {pop['train_grade3']} grade-3 / {pop['train_grade4']} grade-4; "
        f"validation {pop['val_grade3']} grade-3 / {pop['val_grade4']} grade-4. Split sha256 "
        f"`{pop['split_sha256'][:12]}…`. Lesion statistics vs §6's saved CSV: "
        + (f"reproduced on {r['lesion_statistics_reproduction']['n_matched']} validation images, "
           f"max |Δ| {r['lesion_statistics_reproduction']['max_abs_diff']:.1e}."
           if r["lesion_statistics_reproduction"].get("available")
           else "reference CSV not found, so reproduction was not checked."),
        "",
        "## 1. Train-fitted lesion statistics (GO/NO-GO gate)",
        "",
        "| variant | val AUROC (95% CI) | HM SE | val AUPRC (95% CI; prevalence) | dup-excluded AUROC | train 5-fold AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for name, e in (("primary: log10, standardised", p), ("sensitivity: untransformed", s)):
        lines.append(
            f"| {name} | {fmt(e['auroc'])} ({fmt(e['auroc_ci'][0])}–{fmt(e['auroc_ci'][1])}) | "
            f"{fmt(e['auroc_hanley_mcneil_se'])} | {fmt(e['auprc'])} ({fmt(e['auprc_ci'][0])}–"
            f"{fmt(e['auprc_ci'][1])}; {fmt(e['auprc_baseline_prevalence'], 3)}) | "
            f"{fmt(e['duplicate_excluded']['auroc'])} | {fmt(e['train_internal_5fold_auroc'])} |")
    lines += [
        "",
        f"Reference (§11): the in-validation cross-validated lesion probe gave 0.6700. Gate "
        f"(pre-registered): AUROC ≥ {GO_AUROC} and CI lower bound > {GO_CI_LOWER}. "
        f"Result: **{a1['gate']['verdict']}**.",
        "",
        "## 2. Frozen model scores (saved predictions, six runs, validation grade 3 vs 4)",
        "",
        "| run | p_gt_3 (marginal) | p_cond_3 (conditional) | p_gt_2 | conditional − marginal (95% CI) |",
        "|---|---|---|---|---|",
    ]
    for _, row in a2.iterrows():
        lines.append(
            f"| {row['run']} | {fmt(row['auroc_p_gt_3_marginal'])} | "
            f"{fmt(row['auroc_p_cond_3_conditional'])} | {fmt(row['auroc_p_gt_2'])} | "
            f"{row['conditional_minus_marginal']:+.4f} ({row['conditional_minus_marginal_ci_low']:+.3f}"
            f" to {row['conditional_minus_marginal_ci_high']:+.3f}) |")
    lines += [
        f"| **mean** | {fmt(a2['auroc_p_gt_3_marginal'].mean())} | "
        f"{fmt(a2['auroc_p_cond_3_conditional'].mean())} | {fmt(a2['auroc_p_gt_2'].mean())} | "
        f"{a2['conditional_minus_marginal'].mean():+.4f} |",
        "",
        f"Classification: **{c2['label']}** (mean difference {c2['mean_difference']:+.4f}, positive in "
        f"{c2['runs_positive']}/{c2['runs']} runs). Rule: SPECIFIC_LOSS_IN_MARGINAL if mean ≥ "
        f"+{MARGINAL_LOSS_MEAN} and > 0 in ≥ {MARGINAL_LOSS_MIN_RUNS}/6; NO_SPECIFIC_LOSS if mean < "
        f"+{MARGINAL_NO_LOSS_MEAN}. Other saved scores (logit_3, class_prob_4, expected grade, "
        "p_gt_1) are in `analysis2_frozen_score_components.csv`.",
        "",
        f"## 3. Stacked regression baseline ({INFERENCE_ARM}, train-fitted, validation scored once)",
        "",
        "| run | frozen p_gt_3 | model-only LR (4 logits) | lesion-only LR | stacked (logits + lesion) | gain vs frozen (95% CI) | gain vs model-only (95% CI) |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in a3.iterrows():
        lines.append(
            f"| {row['run']} | {fmt(row['auroc_frozen_p_gt_3'])} | "
            f"{fmt(row['auroc_model_only_logreg_on_logits'])} | {fmt(row['auroc_lesion_only_logreg'])} | "
            f"{fmt(row['auroc_stacked_logits_plus_lesion'])} | "
            f"{row['stacked_gain_vs_frozen_p_gt_3']:+.4f} ({row['stacked_gain_vs_frozen_p_gt_3_ci_low']:+.3f}"
            f" to {row['stacked_gain_vs_frozen_p_gt_3_ci_high']:+.3f}) | "
            f"{row['stacked_gain_vs_model_only']:+.4f} ({row['stacked_gain_vs_model_only_ci_low']:+.3f}"
            f" to {row['stacked_gain_vs_model_only_ci_high']:+.3f}) |")
    lines += [
        "",
        f"Lesion statistics improve on the frozen model score: **{'YES' if c3['improves'] else 'NO'}** "
        f"(mean gain over frozen p_gt_3 {c3['mean_gain_over_frozen_p_gt_3']:+.4f}, "
        f"{c3['seeds_positive_over_frozen']}/3 seeds positive; over the model-only regression "
        f"{c3['mean_gain_over_model_only_regression']:+.4f}, "
        f"{c3['seeds_positive_over_model_only']}/3). Rule: mean ≥ +{STACK_IMPROVE_MEAN} and "
        f"positive in {STACK_IMPROVE_MIN_SEEDS}/3 against both references.",
        "",
        "## 4. Internal feature probes (read-only; PCA-16 + logistic regression, train-fitted)",
        "",
        "| run | lesion stats (10-d) | GAP(L) Stage 05 | E Stage 07 | GAP(G) Stage 06 | logits (4-d) | frozen p_gt_3 | localisation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in probes.iterrows():
        lines.append(
            f"| {row['run']} | {fmt(row['probe_lesion_statistics_10d'])} | "
            f"{fmt(row['probe_gap_L_stage05_pca16'])} | {fmt(row['probe_E_stage07_pca16'])} | "
            f"{fmt(row['probe_gap_G_stage06_pca16'])} | {fmt(row['probe_logits_4d'])} | "
            f"{fmt(row['frozen_p_gt_3'])} | {row['localisation']} |")
    parity = r["inference_parity"]
    lines += [
        "",
        "Inference parity against saved predictions (max |Δlogit|): "
        + ", ".join(f"{q['run']} {q['max_abs_logit_diff_vs_saved']:.2e}" for q in parity)
        + f". Tolerance {PARITY_LOGIT_TOLERANCE}.",
        "",
        "Localisation labels are descriptive (RECOMMENDED_DESIGN.md §2, margin "
        f"{PROBE_MARGIN}). With 39 vs 58 validation cases the single-AUROC SE is about 0.058, "
        "so differences below ~0.05 are within noise.",
        "",
        "## Limitations",
        "",
        "- One frozen split. Validation has 97 grade-3/4 images, so every AUROC has a wide CI.",
        "- Probes are linear, fitted on 390 training images with fixed hyperparameters. A low probe "
        "score means the signal is not *linearly* recoverable at that sample size, not that it is absent.",
        "- Stage 04 maps are weak (IDRiD Dice HE 0.127) and have no PDR-defining lesion channel. "
        "The possible laser-scar (EX) confound is not assessed here.",
        "- Known train/val duplicates are handled by the duplicate-excluded row in analysis 1. "
        "Some duplicate pairs carry different grades (see `phase0_results.json` → duplicates).",
        "",
        "## Verdict for Architecture A",
        "",
    ]
    if verdict == "GO":
        lines.append("Architecture A may proceed to implementation. The lesion-burden signal is "
                     "learnable from the training split at or above the pre-registered bar. "
                     "Analyses 2–4 above qualify how Phase 1 should be read.")
    elif verdict == "NO-GO":
        lines.append("Architecture A should not be implemented. A lesion-statistics model fitted "
                     "on the training split does not reach the pre-registered AUROC on validation.")
    else:
        lines.append("The point AUROC clears the bar, but its 95% CI does not exclude chance. "
                     "Architecture A is not cleared for implementation on this evidence.")
    lines += ["", f"**{verdict}**", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
