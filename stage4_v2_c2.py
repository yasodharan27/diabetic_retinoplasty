"""C2 information screen (research record §40 §10, §46): does the new Stage-4 v2 pathology representation carry
DR-grading information beyond a frozen ImageNet ConvNeXt? CPU only, laptop; frozen features; no training of
any network. An information screen, NOT a final architecture.

Features (APTOS training split only, 2,921 images; validation never read; no IDRiD test, no DDR):
  R = frozen ImageNet ConvNeXt-T pooled_final at 512 (experiments/HR_Screen_ImageNet/v1, SHA pinned)
  Q = fixed spatial pyramid (1x1 + 2x2 + 4x4; mean + max) of the 2K = 8 Stage-4 cache channels -> 336
  V = the same pyramid of the Stage-3 vessel map -> 42
Probe (§34 protocol, capacity-matched): every arm has k = 64 dimensions -- R_64, Q_64 = Scale->PCA(64);
R_32+Q_32 and R_32+V_32 = block-wise Scale->PCA(32) per block; then Scale -> LogisticRegressionCV
(C in {0.01, 0.1, 1, 10}, inner stratified 5-fold, AUROC). Every scaler / PCA is fitted on the training folds
only (inside the pipeline). Outer stratified 5-fold x 10 repeats with identical folds for every arm;
out-of-fold scores averaged over repeats.
Endpoint: mean AUROC over the cumulative cuts grade >= 1, 2, 3, 4 (a binary probe per cut).
Uncertainty: paired, grade-stratified bootstrap (2,000) of images, shared by all arms and cuts.
PRE-REGISTERED criterion (pipeline_v2_config; not to be changed after results):
  PASS iff  mean(R+Q) - mean(R) >= +0.005  AND  bootstrap 95% CI lower bound of that difference > 0
            AND  mean(R+Q) >= mean(Q).
Descriptive only: grade >= 3 vs <= 2 (cut 3), V increment (R+V - R), leave-one-class-out R+Q increments.

Usage:  python stage4_v2_c2.py --r <hr_screen_features.npz> --q <c2_q_pyramid.npz> --v <c2_v_pyramid.npz>
                               --stage4-sha <64-hex> [--out-root results/C2] [--no-loco] [--jobs -1]
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

import pipeline_v2_config as v2cfg

ARMS = ("R", "Q", "RQ", "RV")


class C2Error(RuntimeError):
    pass


# --------------------------------------------------------------------------- inputs

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_r(path, expected_sha256=v2cfg.C2_R_FEATURES_SHA256):
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise C2Error(f"{path}: R feature file sha differs from the pinned {expected_sha256}")
    with np.load(path, allow_pickle=False) as z:
        return [str(i) for i in z["train_ids"]], z["y_train"].astype(int), z["f512"].astype(np.float64)


def load_pyramid(path, expected):
    import stage4_v2_aptos_cache as ac
    ids, x, names, prov = ac.load_pyramid_features(path, expected=expected)
    return ids, x, names, prov


def align(train_ids, ids, x, label, val_ids=()):
    """Rows of `x` for exactly `train_ids`, in that order. Refuses validation / unknown ids."""
    leak = set(train_ids) & set(val_ids)
    if leak:
        raise C2Error(f"{label}: validation ids in the training probe set: {sorted(leak)[:5]}")
    index = {i: k for k, i in enumerate(ids)}
    missing = [i for i in train_ids if i not in index]
    if missing:
        raise C2Error(f"{label}: {len(missing)} training ids have no features, e.g. {missing[:5]}")
    return x[[index[i] for i in train_ids]]


def loco_columns(names, cls):
    """Q columns that do NOT belong to class `cls` (both its mean and max channels)."""
    return [k for k, n in enumerate(names) if not n.startswith(f"{cls}:")]


# --------------------------------------------------------------------------- probe

def make_probe(blocks, seed, cs=v2cfg.C2_CS, n_jobs=None):
    """`blocks`: list of (column slice/list, n_components). One block -> Scale->PCA; several -> block-wise.
    Followed by Scale -> LogisticRegressionCV. Everything is fitted on the training fold only."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def block(n):
        return Pipeline([("s", StandardScaler()), ("p", PCA(n_components=n, random_state=seed))])
    red = (block(blocks[0][1]) if len(blocks) == 1 else
           ColumnTransformer([(f"b{k}", block(n), cols) for k, (cols, n) in enumerate(blocks)]))
    return Pipeline([("red", red), ("s2", StandardScaler()),
                     ("lr", LogisticRegressionCV(Cs=list(cs), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                                                 scoring="roc_auc", max_iter=5000, n_jobs=n_jobs))])


def arm_design(arm, dims, k=v2cfg.C2_K_TOTAL):
    """(feature matrix key order, blocks). dims: {"R": dR, "Q": dQ, "V": dV}."""
    if arm == "R":
        return ["R"], [(slice(0, dims["R"]), k)]
    if arm == "Q":
        return ["Q"], [(slice(0, dims["Q"]), k)]
    second = "Q" if arm.startswith("RQ") else "V"
    return ["R", second], [(slice(0, dims["R"]), k // 2), (slice(dims["R"], dims["R"] + dims[second]), k // 2)]


def oof_scores(x, y, blocks, n_repeats=v2cfg.C2_N_REPEATS, n_folds=v2cfg.C2_N_FOLDS, n_jobs=None):
    """Repeat-averaged out-of-fold probabilities. Fold assignment depends only on (y, repeat): identical for
    every arm."""
    from sklearn.model_selection import StratifiedKFold
    total = np.zeros(len(y))
    for r in range(n_repeats):
        out = np.zeros(len(y))
        for a, b in StratifiedKFold(n_folds, shuffle=True, random_state=r).split(x, y):
            out[b] = make_probe(blocks, r, n_jobs=n_jobs).fit(x[a], y[a]).predict_proba(x[b])[:, 1]
        total += out
    return total / n_repeats


def auroc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, bool)
    r = rankdata(np.asarray(s, np.float64))
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def stratified_bootstrap_indices(grades, n_boot=v2cfg.C2_N_BOOT, seed=v2cfg.C2_SEED):
    """n_boot index arrays; each resamples images with replacement WITHIN every grade (paired across arms)."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(grades == g) for g in np.unique(grades)]
    return [np.concatenate([rng.choice(g, g.size) for g in groups]) for _ in range(n_boot)]


def mean_cut_auroc(grades, scores_by_cut, idx=None, cuts=v2cfg.C2_CUTS):
    g = grades if idx is None else grades[idx]
    vals = [auroc(g >= c, scores_by_cut[c] if idx is None else scores_by_cut[c][idx]) for c in cuts]
    return float(np.mean(vals)), vals


def decide(delta, ci_low, rq_mean, q_mean, min_delta=v2cfg.C2_MIN_DELTA):
    checks = {"delta_ge_min": bool(delta >= min_delta), "ci_lower_gt_0": bool(ci_low > 0),
              "rq_ge_q": bool(rq_mean >= q_mean)}
    return {"PASS": all(checks.values()), "checks": checks,
            "criterion": f"R+Q - R >= +{min_delta} AND bootstrap CI lower > 0 AND R+Q >= Q (pre-registered)"}


def run_probe(R, Q, V, grades, q_names, loco=True, n_repeats=v2cfg.C2_N_REPEATS, n_boot=v2cfg.C2_N_BOOT,
              n_jobs=None, log=print):
    """The full screen on aligned matrices (rows = training images). Returns the results dict."""
    t0 = time.time()
    grades = np.asarray(grades, int)
    mats = {"R": R, "Q": Q, "V": V}
    dims = {k: m.shape[1] for k, m in mats.items()}
    arms = {a: arm_design(a, dims) for a in ARMS}
    if loco:
        for c in v2cfg.STAGE4_V2A_CLASSES:
            cols = loco_columns(q_names, c)
            mats[f"Q-{c}"] = Q[:, cols]
            dims[f"Q-{c}"] = len(cols)
            arms[f"RQ-{c}"] = (["R", f"Q-{c}"], [(slice(0, dims["R"]), v2cfg.C2_K_TOTAL // 2),
                                                 (slice(dims["R"], dims["R"] + len(cols)), v2cfg.C2_K_TOTAL // 2)])
    scores = {}
    for name, (keys, blocks) in arms.items():
        x = np.concatenate([mats[k] for k in keys], axis=1)
        scores[name] = {c: oof_scores(x, (grades >= c).astype(int), blocks, n_repeats, n_jobs=n_jobs)
                        for c in v2cfg.C2_CUTS}
        log(f"  arm {name} done ({time.time() - t0:.0f}s)")
    point = {a: mean_cut_auroc(grades, scores[a]) for a in scores}
    boots = stratified_bootstrap_indices(grades, n_boot)
    boot = {a: np.array([mean_cut_auroc(grades, scores[a], i)[0] for i in boots]) for a in scores}

    def diff(a, b):
        d = boot[a] - boot[b]
        return {"delta": point[a][0] - point[b][0],
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}

    primary = diff("RQ", "R")
    res = {"arms": {a: {"mean_auroc": point[a][0],
                        "per_cut": {f">={c}": v for c, v in zip(v2cfg.C2_CUTS, point[a][1])},
                        "mean_auroc_ci95": [float(np.percentile(boot[a], 2.5)), float(np.percentile(boot[a], 97.5))]}
                    for a in scores},
           "primary_RQ_minus_R": primary, "RQ_minus_Q": diff("RQ", "Q"), "RV_minus_R": diff("RV", "R"),
           "secondary_ge3_auroc": {a: point[a][1][v2cfg.C2_CUTS.index(3)] for a in scores},
           "loco_RQ_minus_RQc": {c: diff("RQ", f"RQ-{c}") for c in v2cfg.STAGE4_V2A_CLASSES} if loco else None,
           "n_train": int(len(grades)), "grade_counts": np.bincount(grades, minlength=5).tolist(),
           "dims": {k: int(v) for k, v in dims.items()}, "runtime_s": round(time.time() - t0)}
    res["decision"] = decide(primary["delta"], primary["ci95"][0], point["RQ"][0], point["Q"][0])
    return res


# --------------------------------------------------------------------------- report

def report_markdown(res, cfg):
    a = res["arms"]
    lines = ["# C2 information screen — Stage-4 v2 pathology vs frozen RGB (APTOS training split)", "",
             f"- Stage-4 model {cfg['stage4_sha256']} ({cfg['stage4_generation']}); Stage-3 {cfg['stage3_sha256'][:12]}",
             f"- n = {res['n_train']} (grades {res['grade_counts']}); k = {v2cfg.C2_K_TOTAL}; "
             f"{cfg['n_repeats']} x {v2cfg.C2_N_FOLDS}-fold; bootstrap {cfg['n_boot']}", "",
             "| arm | mean AUROC (95% CI) | >=1 | >=2 | >=3 | >=4 |", "|---|---|---|---|---|---|"]
    for name in [x for x in a if not x.startswith("RQ-")] + [x for x in a if x.startswith("RQ-")]:
        r = a[name]
        lines.append(f"| {name} | {r['mean_auroc']:.4f} ({r['mean_auroc_ci95'][0]:.4f}–{r['mean_auroc_ci95'][1]:.4f}) | "
                     + " | ".join(f"{r['per_cut'][f'>={c}']:.4f}" for c in v2cfg.C2_CUTS) + " |")
    p, d = res["primary_RQ_minus_R"], res["decision"]
    lines += ["", f"**Primary: R+Q − R = {p['delta']:+.4f} (95% CI {p['ci95'][0]:+.4f} to {p['ci95'][1]:+.4f}); "
              f"R+Q − Q = {res['RQ_minus_Q']['delta']:+.4f}**",
              f"- V increment (descriptive): R+V − R = {res['RV_minus_R']['delta']:+.4f} "
              f"({res['RV_minus_R']['ci95'][0]:+.4f} to {res['RV_minus_R']['ci95'][1]:+.4f})"]
    if res["loco_RQ_minus_RQc"]:
        lines.append("- Leave-one-class-out (descriptive, R+Q − R+Q without the class): " + "; ".join(
            f"{c} {v['delta']:+.4f}" for c, v in res["loco_RQ_minus_RQc"].items()))
    lines += ["", f"Criterion: {d['criterion']}", f"Checks: {d['checks']}", "",
              f"**C2 {'PASS' if d['PASS'] else 'FAIL'}**"]
    return "\n".join(lines)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                              cwd=os.path.dirname(os.path.abspath(__file__))).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--r", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--v", required=True)
    ap.add_argument("--stage4-sha", required=True)
    ap.add_argument("--out-root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "C2"))
    ap.add_argument("--no-loco", action="store_true")
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args(argv)

    import multiseed_runs as msr
    import stage4_v2_aptos_cache as ac
    train_entries, val_entries, split_sha = msr.verify_split()
    pop = ac.aptos_population(train_entries, val_entries, split_sha)
    train_pop = [i for i, _ in pop["train"]]
    val_pop = [i for i, _ in pop["val"]]
    r_ids, y, R = load_r(args.r)
    if sorted(r_ids) != sorted(train_pop):
        raise C2Error("R feature ids are not exactly the 2,921 training ids of the pinned population")
    grade_of = dict(pop["train"])
    if any(grade_of[i] != int(g) for i, g in zip(r_ids, y)):
        raise C2Error("R feature grades disagree with the split")
    q_ids, Qall, q_names, q_prov = load_pyramid(args.q, {"stage4_sha256": args.stage4_sha})
    v_ids, Vall, _, v_prov = load_pyramid(args.v, {"stage3_sha256": v2cfg.STAGE3_LWNET_SHA256})
    Q = align(r_ids, q_ids, Qall, "Q", val_pop)
    V = align(r_ids, v_ids, Vall, "V", val_pop)
    if Q.shape[1] != 2 * len(v2cfg.STAGE4_V2A_CLASSES) * 2 * 21 or V.shape[1] != 42:
        raise C2Error(f"unexpected pyramid dims Q {Q.shape[1]} V {V.shape[1]}")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(args.out_root, f"c2_{args.stage4_sha[:12]}_{stamp}")
    if os.path.exists(out_dir):
        raise C2Error(f"{out_dir} exists; experiment directories are never overwritten")
    os.makedirs(out_dir)
    cfg = {"stage4_sha256": args.stage4_sha, "stage4_generation": q_prov.get("generation_id"),
           "stage3_sha256": v_prov.get("stage3_sha256"), "population_sha256": pop["population_sha256"],
           "split_sha256": split_sha, "inputs": {"r": [args.r, sha256_file(args.r)], "q": [args.q, sha256_file(args.q)],
                                                 "v": [args.v, sha256_file(args.v)]},
           "k_total": v2cfg.C2_K_TOTAL, "n_repeats": v2cfg.C2_N_REPEATS, "n_boot": v2cfg.C2_N_BOOT,
           "seed": v2cfg.C2_SEED, "cs": list(v2cfg.C2_CS), "cuts": list(v2cfg.C2_CUTS),
           "criterion_min_delta": v2cfg.C2_MIN_DELTA, "git_commit": git_commit(), "started": stamp}
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=1)
    res = run_probe(R, Q, V, y, q_names, loco=not args.no_loco, n_jobs=args.jobs)
    md = report_markdown(res, cfg)
    with open(os.path.join(out_dir, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(md + "\n")
    print(md)
    print(f"\nwritten to {out_dir}")
    return res


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
