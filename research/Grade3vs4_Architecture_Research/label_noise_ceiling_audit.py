# ============================================================================================
# READ-ONLY LABEL-NOISE AND GRADE-3-vs-4 CEILING AUDIT  (APTOS 2019, authoritative split)
#
# Design: research/Grade3vs4_Architecture_Research/NEXT_STEP_RECOMMENDATION.md, Step LN.
#
# WHAT IT DOES
#   1. SHA-256 of every original APTOS image in the split manifest (duplicates by content only).
#   2. Train/validation byte-identical duplicates, with both grades.
#   3. What the duplicate disagreement could imply for the observable Grade-3-vs-4 AUROC, under
#      explicitly stated noise models (a sensitivity analysis, not an empirical ceiling).
#   4. Compares that with the already-recorded Grade-3-vs-4 AUROCs (no model is run or refit).
#   5. Review sheet: validation Grade-4 images the frozen NO_RACAF models consistently decode <= 2.
#   6. How many of those are connected to a known duplicate / label conflict.
#   7. A rule-based verdict, printed.
#
# WHAT IT NEVER DOES
#   no training, no inference, no model, no label change, no cache/checkpoint write, no fitting on
#   validation, no notebook, no TensorFlow. It reads files and writes ONE JSON under
#   experiments/Grade3vs4_LabelNoiseAudit/<timestamp>/. Everything that matters is printed.
#   Predictions from the running task-3-vs-placebo experiment are deliberately NOT read.
#
# RUN: a fresh Colab runtime, CPU is enough. Paste into one cell (or %run). Hashing ~3,662 PNGs
# over Drive takes a while; progress is printed.
# ============================================================================================

import collections
import datetime
import glob
import hashlib
import json
import math
import os
import posixpath
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ---- Pre-stated constants (fixed before any output is seen) ----
EXPECTED_SPLIT_SHA256 = "bc80fd450340b09307fbd80a1b00553e70e34d64a3cdf94635162b6c1e99aca5"
EXPECTED_COUNTS = {"train": 2929, "val": 733}
SEEDS = (42, 123, 2026)
AUX_LAMBDAS = ("0.05", "0.10", "0.20")
SEVERE, PDR = 3, 4
HASH_WORKERS = 16
SENSITIVITY_FLIPS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
ASYMMETRIC_GRID = (0.05, 0.15, 0.30)
TRUE_AUROC_LEVELS = (1.0, 0.9, 0.8)
# Audit-7 thresholds (stated in advance):
#   "clearly plausible"   : point-estimate ceiling for a PERFECT model <= 0.85
#   "possible, insufficient evidence": only the CI-upper / worst-case ceiling <= 0.85
#   "unlikely"            : even the CI-upper / worst-case ceiling > 0.85
# 0.85 = the best readout seen so far (~0.76, Stage-05 probe) + ~0.09: a noise level that keeps
# even a perfect model near the observed plateau.
MATERIAL_CEILING = 0.85
# Recorded Grade-3-vs-4 AUROCs (research record section 13) -- used only if the Phase-0 JSON is
# not found on Drive.
RECORDED_S13 = {"Stage 05 GAP(L) probe": 0.7576, "Stage 07 E probe": 0.7098,
                "CORN 4-logit readout": 0.6599, "p_cond_3 (NO_RACAF, unfitted)": 0.6394,
                "p_gt_3 (NO_RACAF, unfitted)": 0.5775, "lesion statistics (train-fitted)": 0.6202}

REPO_URL = "https://github.com/yasodharan27/diabetic_retinoplasty.git"
REPO_DIR = os.environ.get("DR_REPO_DIR", "/content/diabetic_retinoplasty")


# ============================================================================================
# Pure helpers
# ============================================================================================

def clopper_pearson(k, n, level=0.95):
    from scipy.stats import beta
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf((1 - level) / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - (1 - level) / 2, k + 1, n - k)
    return float(lo), float(hi)


def rate(k, n):
    return (k / n) if n else float("nan")


def flip_from_disagreement(d):
    """Two independent gradings, each flipping between two classes with probability e, disagree with
    probability 2e(1-e). Inverse, capped at 0.5."""
    if d is None or not np.isfinite(d):
        return float("nan")
    d = min(max(d, 0.0), 0.5)
    return (1.0 - math.sqrt(1.0 - 2.0 * d)) / 2.0


def observed_auroc_flip(true_auroc, a, b):
    """Observed AUROC when a fraction a of '4' labels are truly 3 and b of '3' labels are truly 4,
    ASSUMING a mislabelled image's score is distributed like its true class (label errors independent
    of image features given the true class)."""
    return ((1 - a) * (1 - b) * true_auroc + 0.5 * ((1 - a) * b + a * (1 - b))
            + a * b * (1 - true_auroc))


def observed_auroc_contamination(auroc_34, c, s):
    """A fraction c of '4' labels are images whose true grade is <= 2; s = P(score of such an image
    exceeds a true grade-3 image). s = 0 is the worst case, 0.5 is neutral."""
    return (1 - c) * auroc_34 + c * s


def implied_true_auroc(observed, e):
    """Inverse of observed_auroc_flip for symmetric e (valid e < 0.5)."""
    if e >= 0.5:
        return float("nan")
    return (observed - e * (1 - e) - e * e) / (1 - 2 * e)


def auroc(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=np.float64)
    pos, neg = s[y == 1], s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def parse_pinned_pairs(markdown_text):
    ids = {}
    for line in markdown_text.splitlines():
        m = re.match(r"^\|\s*`([0-9a-f]{12})`\s*\|\s*(\d)\s*\|(.*)$", line.strip())
        if m and m.group(1) not in ids:
            ids[m.group(1)] = [t for t, _ in re.findall(r"`([0-9a-f]{12})`\s*\[(\d)\]", m.group(3))]
    return ids


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(x, d=4):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def pct_ci(k, n):
    lo, hi = clopper_pearson(k, n)
    return f"{k}/{n} = {fmt(rate(k, n), 3)} (95% CI {fmt(lo, 3)}-{fmt(hi, 3)})"


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


# ============================================================================================
# Environment
# ============================================================================================

def setup_environment():
    on_colab = os.path.isdir("/content") and "DR_EXPERIMENTS_ROOT" not in os.environ
    if on_colab and not os.path.isdir("/content/drive/MyDrive"):
        from google.colab import drive
        drive.mount("/content/drive")
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)
    sys.path.insert(0, os.path.join(REPO_DIR, "colab", "common"))
    import colab_config
    experiments_root = os.environ.get("DR_EXPERIMENTS_ROOT", colab_config.DRIVE.experiments_root)
    image_dir = os.environ.get("DR_APTOS_IMAGE_DIR",
                               os.path.join(colab_config.APTOS2019_RAW_DIR, "train_images"))
    return experiments_root, image_dir


def load_split():
    import pandas as pd
    path = os.path.join(REPO_DIR, "dataset_splits", "aptos2019_train_val_split.csv")
    with open(path, "rb") as fh:
        raw = fh.read()
    sha = hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest()
    if sha != EXPECTED_SPLIT_SHA256:
        raise RuntimeError(f"split manifest sha256 {sha} != pinned {EXPECTED_SPLIT_SHA256}")
    split = pd.read_csv(path, dtype={"id_code": str})
    counts = split["split"].value_counts().to_dict()
    if {k: counts.get(k) for k in EXPECTED_COUNTS} != EXPECTED_COUNTS:
        raise RuntimeError(f"split counts {counts} != {EXPECTED_COUNTS}")
    return split, sha


def load_per_sample(path):
    import pandas as pd
    if not os.path.exists(path):
        return None
    f = pd.read_csv(path, dtype={"image_id": str})
    if "p_cond_3" not in f.columns:
        f["p_cond_3"] = f["p_gt_3"] / f["p_gt_2"].clip(lower=1e-12)
    return f.set_index("image_id")


# ============================================================================================
# Main
# ============================================================================================

def main():
    import pandas as pd

    experiments_root, image_dir = setup_environment()
    split, split_sha = load_split()
    grade = dict(zip(split["id_code"], split["diagnosis"].astype(int)))
    part = dict(zip(split["id_code"], split["split"]))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = posixpath.join(experiments_root, "Grade3vs4_LabelNoiseAudit", timestamp)
    result = {"timestamp": timestamp, "split_sha256": split_sha, "image_dir": image_dir,
              "assumptions_and_thresholds": {"material_ceiling": MATERIAL_CEILING}}

    print("READ-ONLY LABEL-NOISE AND GRADE-3-vs-4 CEILING AUDIT")
    print(f"run {timestamp} | split sha256 {split_sha[:12]}... verified | "
          f"train {EXPECTED_COUNTS['train']} / val {EXPECTED_COUNTS['val']} | images: {image_dir}")

    # ---------------------------------------------------------------- AUDIT 1
    section("AUDIT 1 - SHA-256 OF EVERY APTOS IMAGE IN THE SPLIT MANIFEST")
    ids = list(split["id_code"])
    paths = {i: os.path.join(image_dir, f"{i}.png") for i in ids}
    missing = [i for i in ids if not os.path.exists(paths[i])]
    present = [i for i in ids if i not in set(missing)]
    digests = {}

    def work(i):
        return i, sha256_file(paths[i])

    with ThreadPoolExecutor(max_workers=HASH_WORKERS) as pool:
        for n, (i, d) in enumerate(pool.map(work, present), start=1):
            digests[i] = d
            if n % 500 == 0 or n == len(present):
                print(f"  hashed {n}/{len(present)}")
    groups_by_hash = collections.defaultdict(list)
    for i, d in digests.items():
        groups_by_hash[d].append(i)
    dup_groups = [sorted(g) for g in groups_by_hash.values() if len(g) > 1]
    size_dist = collections.Counter(len(g) for g in dup_groups)
    print(f"images in manifest        : {len(ids)}")
    print(f"images hashed             : {len(digests)}   (missing on Drive: {len(missing)})")
    print(f"unique hashes             : {len(groups_by_hash)}")
    print(f"duplicate groups          : {len(dup_groups)}")
    print(f"images in duplicate groups: {sum(len(g) for g in dup_groups)}")
    print(f"group size distribution   : {dict(sorted(size_dist.items()))}")
    if missing:
        print(f"missing ids (first 20)    : {missing[:20]}")
    kind = collections.Counter(
        "train+val" if len({part[i] for i in g}) > 1 else f"{part[g[0]]} only" for g in dup_groups)
    print(f"groups by split membership: {dict(kind)}")
    result["audit1"] = {"n_manifest": len(ids), "n_hashed": len(digests), "n_missing": len(missing),
                        "missing_ids": missing, "n_unique": len(groups_by_hash),
                        "n_duplicate_groups": len(dup_groups),
                        "n_images_in_groups": sum(len(g) for g in dup_groups),
                        "size_distribution": dict(size_dist), "membership": dict(kind)}

    # ---------------------------------------------------------------- AUDIT 2
    section("AUDIT 2 - TRAIN/VALIDATION BYTE-IDENTICAL DUPLICATES")
    cross = [g for g in dup_groups if len({part[i] for i in g}) > 1]
    rows, pair_records = [], []
    for g in cross:
        tr = [i for i in g if part[i] == "train"]
        va = [i for i in g if part[i] == "val"]
        grades_all = {grade[i] for i in g}
        agree = len(grades_all) == 1
        touches = bool(grades_all & {SEVERE, PDR})
        rows.append({"train_ids": tr, "val_ids": va, "train_grades": [grade[i] for i in tr],
                     "val_grades": [grade[i] for i in va], "agree": agree, "touches_3_4": touches})
        for v in va:
            for t in tr:
                pair_records.append({"val_id": v, "train_id": t, "val_grade": grade[v],
                                     "train_grade": grade[t]})
    print(f"{'#':>3} {'validation id(s)':<16} {'vg':>3} {'train id(s)':<16} {'tg':>3}  agree  3/4")
    for n, r in enumerate(sorted(rows, key=lambda r: (r["agree"], r["val_ids"])), start=1):
        print(f"{n:>3} {','.join(r['val_ids']):<16} {','.join(map(str, r['val_grades'])):>3} "
              f"{','.join(r['train_ids']):<16} {','.join(map(str, r['train_grades'])):>3}  "
              f"{'yes' if r['agree'] else 'NO ':<5}  {'yes' if r['touches_3_4'] else '-'}")
    n_pairs = len(pair_records)
    dis = [p for p in pair_records if p["val_grade"] != p["train_grade"]]
    t34 = [p for p in pair_records if {p["val_grade"], p["train_grade"]} & {SEVERE, PDR}]
    t34_dis = [p for p in t34 if p["val_grade"] != p["train_grade"]]
    within34 = [p for p in pair_records if {p["val_grade"], p["train_grade"]} <= {SEVERE, PDR}]
    within34_dis = [p for p in within34 if p["val_grade"] != p["train_grade"]]
    transitions = collections.Counter(f"val {p['val_grade']} -> train {p['train_grade']}" for p in dis)
    img_in_cross = {i for g in cross for i in g}
    g4_dup_le2 = sorted(i for g in dup_groups for i in g
                        if grade[i] == PDR and any(grade[j] <= 2 for j in g if j != i))
    g3_dup_diff = sorted(i for g in dup_groups for i in g
                         if grade[i] == SEVERE and any(grade[j] != SEVERE for j in g if j != i))
    print(f"\ncross-split groups: {len(cross)} | cross-split pairs: {n_pairs}")
    print(f"agreeing groups   : {sum(r['agree'] for r in rows)} | disagreeing groups: "
          f"{sum(not r['agree'] for r in rows)}")
    print(f"pair disagreement, all pairs              : {pct_ci(len(dis), n_pairs)}")
    print(f"pair disagreement, pairs touching 3 or 4  : {pct_ci(len(t34_dis), len(t34))}")
    print(f"pair disagreement, both labels in {{3,4}}   : {pct_ci(len(within34_dis), len(within34))}")
    print(f"grade-4 images whose duplicate is <= 2    : {len(g4_dup_le2)} {g4_dup_le2}")
    print(f"grade-3 images whose duplicate differs    : {len(g3_dup_diff)} {g3_dup_diff}")
    print(f"conflicting transitions (val -> train)    : {dict(transitions)}")

    # all duplicate groups (incl. within-split) -- secondary, label-level
    all_pairs = [(grade[a], grade[b]) for g in dup_groups for x, a in enumerate(g) for b in g[x + 1:]]
    all_dis = sum(1 for a, b in all_pairs if a != b)
    print(f"[secondary] all duplicate pairs incl. within-split: {pct_ci(all_dis, len(all_pairs))}")

    pinned = {}
    pinned_path = os.path.join(REPO_DIR, "docs", "experiments", "RACAF_Duplicate_Contamination_Audit.md")
    if os.path.exists(pinned_path):
        with open(pinned_path, encoding="utf-8") as fh:
            pinned = parse_pinned_pairs(fh.read())
        found_val = {v for r in rows for v in r["val_ids"]}
        print(f"pinned 41-pair list reproduced: {set(pinned) == found_val} "
              f"(pinned {len(pinned)}, found {len(found_val)}, "
              f"only-pinned {sorted(set(pinned) - found_val)}, only-found {sorted(found_val - set(pinned))})")

    # label-level contamination rate: grade-4 labels (in duplicate groups) whose partner is <= 2
    g4_labels = [(i, j) for g in dup_groups for i in g for j in g if i != j and grade[i] == PDR]
    g4_label_ids = sorted({i for i, _ in g4_labels})
    g4_with_le2 = sorted({i for i, j in g4_labels if grade[j] <= 2})
    result["audit2"] = {"groups": rows, "n_pairs": n_pairs, "n_disagree": len(dis),
                        "n_touch34": len(t34), "n_touch34_disagree": len(t34_dis),
                        "n_within34": len(within34), "n_within34_disagree": len(within34_dis),
                        "transitions": dict(transitions), "grade4_dup_le2": g4_dup_le2,
                        "grade3_dup_differs": g3_dup_diff, "all_pairs": len(all_pairs),
                        "all_pairs_disagree": all_dis,
                        "pinned_reproduced": set(pinned) == {v for r in rows for v in r["val_ids"]}}

    # ---------------------------------------------------------------- AUDIT 3
    section("AUDIT 3 - WHAT THE DISAGREEMENT COULD IMPLY FOR GRADE-3-vs-4 AUROC (sensitivity analysis)")
    print("NOISE MODELS AND ASSUMPTIONS")
    print(" M1 (3<->4 flips). Among images labelled 3 or 4, a fraction a of '4' labels are truly 3 and b of")
    print("    '3' labels truly 4. Observed AUROC = (1-a)(1-b)A + 0.5[(1-a)b + a(1-b)] + ab(1-A).")
    print("    Assumes a mislabelled image's score is distributed like its TRUE class. Real grading")
    print("    disagreements concentrate on borderline images, so the real attenuation may differ")
    print("    in either direction.")
    print(" M2 (contamination). A fraction c of '4' labels are images whose true grade is <= 2, scoring")
    print("    above a true grade 3 with probability s (s = 0.5 neutral, s = 0 worst case).")
    print("    Observed = (1-c) * A_34 + c * s.")
    print(" Duplicate disagreement measures DISAGREEMENT BETWEEN TWO GRADINGS, not error against a")
    print(" reference standard. Mapping it to a per-label flip rate needs the assumption that both")
    print(" copies were graded independently with equal error: d = 2e(1-e) => e = (1 - sqrt(1-2d)) / 2.")
    print(" For M2, which copy is correct is unknown: c ranges from 0 (the '4' is always right) to the")
    print(" label-level rate r4 (the '<=2' copy is always right); the midpoint r4/2 is the neutral point.")

    d_all = rate(len(dis), n_pairs)
    d_t34 = rate(len(t34_dis), len(t34))
    d_w34 = rate(len(within34_dis), len(within34))
    d_w34_hi = clopper_pearson(len(within34_dis), len(within34))[1]
    r4 = rate(len(g4_with_le2), len(g4_label_ids))
    r4_hi = clopper_pearson(len(g4_with_le2), len(g4_label_ids))[1]
    e_all, e_t34, e_w34, e_w34_hi = (flip_from_disagreement(x) for x in (d_all, d_t34, d_w34, d_w34_hi))
    print("\nRATES FROM THE DATA")
    print(f" d(all cross-split pairs)          = {fmt(d_all, 3)} -> symmetric flip e = {fmt(e_all, 3)}")
    print(f" d(pairs touching 3/4)             = {fmt(d_t34, 3)} -> e = {fmt(e_t34, 3)}   "
          "(mixes 3<->4 with 4<->(<=2); an upper-bound-style figure)")
    print(f" d(both labels in {{3,4}})           = {fmt(d_w34, 3)} -> e = {fmt(e_w34, 3)}   "
          f"(CI upper {fmt(d_w34_hi, 3)} -> e = {fmt(e_w34_hi, 3)})  <- the M1-relevant rate")
    print(f" r4 = grade-4 duplicate labels whose partner is <= 2 = {len(g4_with_le2)}/{len(g4_label_ids)}"
          f" = {fmt(r4, 3)} (CI upper {fmt(r4_hi, 3)})  <- the M2-relevant rate")

    def combined(A, e, c, s):
        return observed_auroc_contamination(observed_auroc_flip(A, e, e), c, s)

    scenarios = {
        "M1 at d(all)": combined(1.0, e_all, 0.0, 0.5),
        "M1 at d(touching 3/4)": combined(1.0, e_t34, 0.0, 0.5),
        "M1 at d({3,4}) point": combined(1.0, e_w34, 0.0, 0.5),
        "M1+M2 point (e{3,4}, c=r4/2, s=0.5)": combined(1.0, e_w34, r4 / 2, 0.5),
        "M1+M2 worst (e{3,4} CI-up, c=r4 CI-up, s=0)": combined(1.0, e_w34_hi, r4_hi, 0.0),
    }
    print("\nCEILING FOR A PERFECT MODEL (true 3-vs-4 AUROC = 1) UNDER EACH ASSUMPTION")
    for k, v in scenarios.items():
        print(f"  {k:<48} observed AUROC <= {fmt(v, 3)}")

    print("\nSENSITIVITY - symmetric 3<->4 flip e (rows) x true AUROC A (cols); M2 off")
    print("   e    " + "  ".join(f"A={A:.1f}" for A in TRUE_AUROC_LEVELS))
    for e in SENSITIVITY_FLIPS:
        print(f"  {e:.2f}  " + "  ".join(f"{observed_auroc_flip(A, e, e):.3f}" for A in TRUE_AUROC_LEVELS))
    print("SENSITIVITY - asymmetric (a = P(true 3 | label 4), b = P(true 4 | label 3)), A = 1")
    print("   a\\b  " + "  ".join(f"{b:.2f} " for b in ASYMMETRIC_GRID))
    for a in ASYMMETRIC_GRID:
        print(f"  {a:.2f}  " + "  ".join(f"{observed_auroc_flip(1.0, a, b):.3f}" for b in ASYMMETRIC_GRID))
    print("SENSITIVITY - contamination c (rows) x s, with A_34 = 1")
    for c in (0.05, 0.10, 0.20, 0.30):
        print(f"  c={c:.2f}  s=0.5 -> {observed_auroc_contamination(1.0, c, 0.5):.3f}   "
              f"s=0 -> {observed_auroc_contamination(1.0, c, 0.0):.3f}")
    print("\nSTATUS OF THESE NUMBERS: a model-dependent SENSITIVITY ANALYSIS. They are not a demonstrated")
    print("empirical ceiling: the true labels are unknown, the duplicate sample is small, and M1/M2 rest on")
    print("independence assumptions stated above.")
    result["audit3"] = {"d_all": d_all, "d_touch34": d_t34, "d_within34": d_w34,
                        "d_within34_ci_upper": d_w34_hi, "r4": r4, "r4_ci_upper": r4_hi,
                        "e_all": e_all, "e_touch34": e_t34, "e_within34": e_w34,
                        "e_within34_ci_upper": e_w34_hi, "perfect_model_ceilings": scenarios}

    # ---------------------------------------------------------------- AUDIT 4
    section("AUDIT 4 - NOISE BURDEN vs THE RECORDED GRADE-3-vs-4 AUROCs (no model run, no correction)")
    recorded, source = dict(RECORDED_S13), "research record section 13 (constants)"
    phase0 = sorted(glob.glob(posixpath.join(experiments_root, "Grade3vs4_Phase0", "*",
                                             "phase0_results.json")))
    if phase0:
        with open(phase0[-1]) as fh:
            p0 = json.load(fh)
        mean = p0["analysis4_feature_probes"]["mean"]
        recorded.update({"Stage 05 GAP(L) probe": mean["probe_gap_L_stage05_pca16"],
                         "Stage 07 E probe": mean["probe_E_stage07_pca16"],
                         "CORN 4-logit readout": mean["probe_logits_4d"],
                         "p_gt_3 (NO_RACAF, unfitted)": mean["frozen_p_gt_3"],
                         "lesion statistics (train-fitted)": mean["probe_lesion_statistics_10d"]})
        source = phase0[-1]
    six_run = posixpath.join(experiments_root, "ImprovedTraining", "improved_multiseed_2026_09")
    base = {s: load_per_sample(posixpath.join(six_run, "NO_RACAF", f"seed_{s}", "evaluation",
                                              "per_sample_best.csv")) for s in SEEDS}
    if all(f is not None for f in base.values()):
        vals = []
        for f in base.values():
            m = f["true_grade"].isin([SEVERE, PDR])
            vals.append(auroc((f.loc[m, "true_grade"] == PDR).astype(int), f.loc[m, "p_cond_3"]))
        recorded["p_cond_3 (NO_RACAF, unfitted)"] = float(np.mean(vals))
    print(f"source: {source}; p_cond_3 recomputed from the saved NO_RACAF per-sample files: "
          f"{all(f is not None for f in base.values())}")
    val34 = split[(split["split"] == "val") & split["diagnosis"].isin([SEVERE, PDR])]
    n3 = int((val34["diagnosis"] == SEVERE).sum())
    n4 = int((val34["diagnosis"] == PDR).sum())
    disputed_val34 = sorted(i for i in val34["id_code"] if i in img_in_cross and len(
        {grade[j] for g in cross if i in g for j in g}) > 1)
    swing = sum((n3 if grade[i] == PDR else n4) for i in disputed_val34) / (n3 * n4) if n3 and n4 else 0
    print(f"validation 3/4 population: {n3} grade-3, {n4} grade-4 (manifest; evaluated set excludes "
          "3 empty-FOV images)")
    print(f"validation 3/4 images with a conflicting byte-identical twin: {len(disputed_val34)} "
          f"{disputed_val34} -> they can move any AUROC by at most +/-{swing:.3f}")
    print(f"\n{'readout':<36} {'observed':>8} | implied true AUROC if the observed value were produced by"
          " M1 at e =")
    print(f"{'':<36} {'':>8} | {fmt(e_w34, 3)} (point) | {fmt(e_w34_hi, 3)} (CI up) | 0.20 (reference)")
    for name, v in recorded.items():
        print(f"{name:<36} {fmt(v):>8} | {fmt(implied_true_auroc(v, e_w34), 3):>13} | "
              f"{fmt(implied_true_auroc(v, e_w34_hi), 3):>13} | {fmt(implied_true_auroc(v, 0.20), 3)}")
    point_ceiling = scenarios["M1+M2 point (e{3,4}, c=r4/2, s=0.5)"]
    worst_ceiling = scenarios["M1+M2 worst (e{3,4} CI-up, c=r4 CI-up, s=0)"]
    if point_ceiling <= MATERIAL_CEILING:
        plateau = "CLEARLY PLAUSIBLE"
    elif worst_ceiling <= MATERIAL_CEILING:
        plateau = "POSSIBLE BUT INSUFFICIENT EVIDENCE"
    else:
        plateau = "UNLIKELY"
    print(f"\nperfect-model ceiling at the point estimates: {fmt(point_ceiling, 3)}; "
          f"at the CI-upper / worst case: {fmt(worst_ceiling, 3)}; material threshold {MATERIAL_CEILING}")
    print(f"Could label noise explain a substantial part of the plateau?  -> {plateau}")
    result["audit4"] = {"recorded": recorded, "source": source, "disputed_val34": disputed_val34,
                        "max_swing": swing, "point_ceiling": point_ceiling,
                        "worst_ceiling": worst_ceiling, "plateau_verdict": plateau}

    # ---------------------------------------------------------------- AUDIT 5
    section("AUDIT 5 - PDR REVIEW SHEET: validation grade-4 images the frozen NO_RACAF models decode <= 2")
    if not all(f is not None for f in base.values()):
        raise RuntimeError("NO_RACAF per_sample_best.csv files not found; cannot build the review sheet")
    ref = base[SEEDS[0]]
    g4_ids = [i for i in ref.index if int(ref.loc[i, "true_grade"]) == PDR]
    primary = [i for i in g4_ids if all(int(base[s].loc[i, "predicted_grade"]) <= 2 for s in SEEDS)]
    aux_runs = {}
    aux_root = posixpath.join(experiments_root, "Grade4AuxLoss", "grade4_aux_loss_2026_09")
    for lam in AUX_LAMBDAS:
        for s in SEEDS:
            f = load_per_sample(posixpath.join(aux_root, f"aux_lambda_{lam}", f"seed_{s}",
                                               "evaluation", "per_sample_best.csv"))
            if f is not None and len(f) == len(ref):
                aux_runs[f"aux{lam}/s{s}"] = f
    robust = [i for i in primary if all(int(f.loc[i, "predicted_grade"]) <= 2 for f in aux_runs.values())]
    print("criterion (primary): predicted grade <= 2 in ALL THREE frozen NO_RACAF seeds (BEST).")
    print(f"stronger criterion : also <= 2 in all {len(aux_runs)} available NO_RACAF-architecture "
          "aux-loss runs. Task-3-vs-placebo runs are deliberately NOT read (result pending).")
    print(f"validation grade-4 evaluated: {len(g4_ids)} | consistently <= 2 (primary): {len(primary)} "
          f"({fmt(rate(len(primary), len(g4_ids)), 3)}) | also robust across aux runs: {len(robust)}")
    group_of = {i: g for g in dup_groups for i in g}
    sheet = []
    print(f"\n{'image_id':<13} {'pred s42/123/2026':<17} {'mean p_gt_2':>11} {'p_gt_3':>7} "
          f"{'p_cond_3':>8} {'robust':>6}  duplicate (grades)")
    for i in sorted(primary, key=lambda i: np.mean([base[s].loc[i, "p_gt_2"] for s in SEEDS])):
        preds = "/".join(str(int(base[s].loc[i, "predicted_grade"])) for s in SEEDS)
        m = {k: float(np.mean([base[s].loc[i, k] for s in SEEDS])) for k in ("p_gt_2", "p_gt_3", "p_cond_3")}
        g = group_of.get(i)
        dup = "-" if g is None else ", ".join(f"{j}[{part[j]} g{grade[j]}]" for j in g if j != i)
        conflict = g is not None and len({grade[j] for j in g}) > 1
        sheet.append({"image_id": i, "true_grade": PDR, "preds": preds, **m, "robust": i in robust,
                      "duplicate_of": [j for j in (g or []) if j != i], "duplicate_conflict": conflict,
                      "image_path": paths.get(i)})
        print(f"{i:<13} {preds:<17} {m['p_gt_2']:>11.3f} {m['p_gt_3']:>7.3f} {m['p_cond_3']:>8.3f} "
              f"{'yes' if i in robust else 'no':>6}  {dup}{'  <- CONFLICT' if conflict else ''}")
    print(f"\nimages for visual review are at: {image_dir}/<image_id>.png (look for laser scars, "
          "regressed/fibrotic PDR, vitreous haemorrhage, poor quality). No automated interpretation.")
    result["audit5"] = {"criterion": "pred <= 2 in all three NO_RACAF seeds", "n_grade4": len(g4_ids),
                        "n_primary": len(primary), "n_robust": len(robust),
                        "aux_runs_used": sorted(aux_runs), "sheet": sheet}

    # ---------------------------------------------------------------- AUDIT 6
    section("AUDIT 6 - PDR <= 2 GROUP vs KNOWN DUPLICATE / LABEL CONFLICT")
    with_dup = [r for r in sheet if r["duplicate_of"]]
    with_conflict = [r for r in sheet if r["duplicate_conflict"]]
    with_le2 = [r for r in sheet if any(grade[j] <= 2 for j in r["duplicate_of"])]
    n = len(sheet)
    print(f"consistently <= 2 PDR images: {n}")
    print(f"  with a byte-identical duplicate      : {len(with_dup)} ({fmt(rate(len(with_dup), n), 3)})")
    print(f"  whose duplicate has another grade    : {len(with_conflict)} ({fmt(rate(len(with_conflict), n), 3)})")
    print(f"  whose duplicate is graded <= 2       : {len(with_le2)} ({fmt(rate(len(with_le2), n), 3)})")
    g4_not_le2 = [i for i in g4_ids if i not in set(primary)]
    other_conf = sum(1 for i in g4_not_le2 if i in group_of and len({grade[j] for j in group_of[i]}) > 1)
    print(f"  comparison: grade-4 images NOT consistently <= 2 with a conflicting duplicate: "
          f"{other_conf}/{len(g4_not_le2)}")
    print("  (only images that happen to have a byte-identical twin can reveal a conflict; the absence of")
    print("   a twin says nothing about the label's correctness)")
    result["audit6"] = {"n": n, "with_duplicate": len(with_dup), "with_conflict": len(with_conflict),
                        "duplicate_le2": len(with_le2), "other_grade4_conflicts": other_conf,
                        "other_grade4_n": len(g4_not_le2)}

    # ---------------------------------------------------------------- AUDIT 7
    section("LABEL-NOISE AUDIT VERDICT")
    from scipy.stats import fisher_exact
    other_pairs = n_pairs - len(t34)
    other_dis = len(dis) - len(t34_dis)
    fisher_p = (fisher_exact([[len(t34_dis), len(t34) - len(t34_dis)],
                              [other_dis, other_pairs - other_dis]], alternative="greater")[1]
                if len(t34) and other_pairs else float("nan"))
    q = {}
    q[1] = ("YES" if len(dis) else "NO") + f" - {len(dis)} of {n_pairs} byte-identical train/validation " \
        "pairs carry different grades."
    if d_t34 > rate(other_dis, other_pairs) and fisher_p < 0.05:
        q[2] = f"YES - disagreement {fmt(d_t34, 3)} in pairs touching 3/4 vs " \
               f"{fmt(rate(other_dis, other_pairs), 3)} elsewhere (one-sided Fisher p = {fmt(fisher_p, 3)})."
    elif d_t34 > rate(other_dis, other_pairs):
        q[2] = f"HIGHER BUT NOT STATISTICALLY DISTINGUISHABLE - {fmt(d_t34, 3)} vs " \
               f"{fmt(rate(other_dis, other_pairs), 3)} (Fisher p = {fmt(fisher_p, 3)}). The 3/4 " \
               f"conflicts are mostly 4 vs <=2, not 3 vs 4 ({len(within34_dis)}/{len(within34)} pairs " \
               "with both labels in {3,4} disagree)."
    else:
        q[2] = f"NO - {fmt(d_t34, 3)} vs {fmt(rate(other_dis, other_pairs), 3)}."
    q[3] = f"{plateau} - perfect-model ceiling {fmt(point_ceiling, 3)} at the point estimates, " \
           f"{fmt(worst_ceiling, 3)} in the worst case (threshold {MATERIAL_CEILING})."
    q[4] = "NO - only a model-dependent sensitivity analysis. True labels are unknown, the duplicate " \
           "sample is small, and the models assume label errors independent of image features."
    q[5] = ("NO - not until the pending task-3-vs-placebo result is in and the noise question is " \
            "resolved; the audit gives no reason to add training."
            if plateau != "UNLIKELY" else
            "NOT BLOCKED BY NOISE - the duplicate evidence does not cap the AUROC; any further training "
            "still depends on the pending pre-registered result.")
    q[6] = f"NO - first establish whether the {n} consistently-<=2 PDR images are genuine untreated " \
           f"PDR. {len(with_conflict)} of them carry a known conflicting duplicate grade, and the rest " \
           "have not been visually reviewed."
    q[7] = ("ANOTHER READ-ONLY STEP FIRST - visually review the Audit-5 sheet (laser scars, regressed PDR, " \
            "quality). External IDRiD evaluation stays tied to the pending pre-registered outcome; this " \
            "audit does not by itself trigger it.")
    for k in range(1, 8):
        print(f"{k}. {q[k]}")
    result["verdict"] = q

    os.makedirs(out_dir, exist_ok=True)
    json_path = posixpath.join(out_dir, "label_noise_audit.json")
    with open(json_path, "w") as fh:
        json.dump(result, fh, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---------------------------------------------------------------- SUMMARY
    print("\n" + "#" * 100 + "\nAUDIT COMPLETE\n" + "#" * 100)
    print("KEY NUMBERS")
    print(f"  hashed {len(digests)}/{len(ids)} images; {len(dup_groups)} duplicate groups; "
          f"{len(cross)} cross-split groups / {n_pairs} pairs")
    print(f"  cross-split disagreement: {pct_ci(len(dis), n_pairs)}")
    print(f"  touching 3/4: {pct_ci(len(t34_dis), len(t34))}; both in {{3,4}}: "
          f"{pct_ci(len(within34_dis), len(within34))}; grade-4 labels with a <=2 twin: "
          f"{len(g4_with_le2)}/{len(g4_label_ids)}")
    print("KEY CONFLICTING DUPLICATE PAIRS (val id [grade] <-> train id [grade])")
    for p in sorted(dis, key=lambda p: (-(p["val_grade"] in (3, 4) or p["train_grade"] in (3, 4)),
                                        p["val_id"])):
        print(f"  {p['val_id']} [{p['val_grade']}] <-> {p['train_id']} [{p['train_grade']}]")
    print("NOISE / CEILING INTERPRETATION")
    print(f"  3<->4 flip estimate e = {fmt(e_w34, 3)} (CI-up {fmt(e_w34_hi, 3)}); 4-vs-<=2 label rate "
          f"r4 = {fmt(r4, 3)}; perfect-model ceiling {fmt(point_ceiling, 3)} (point) / "
          f"{fmt(worst_ceiling, 3)} (worst) -> sensitivity analysis, not an empirical ceiling")
    print(f"  plateau explanation: {plateau}")
    print(f"PDR <= 2 REVIEW POPULATION: {len(primary)} of {len(g4_ids)} validation grade-4 images "
          f"({len(robust)} also <= 2 in every aux run); {len(with_conflict)} with a conflicting duplicate")
    print("FINAL SCIENTIFIC ASSESSMENT")
    for k in range(1, 8):
        print(f"  {k}. {q[k]}")
    print("RECOMMENDED NEXT ACTION")
    print("  Visually review the Audit-5 images (read-only), then act on the pending task-3-vs-placebo "
          "verdict per NEXT_STEP_DECISION_TREE.md. No training, relabelling or new annotation now.")
    print(f"\nJSON: {json_path}")


if __name__ == "__main__":
    main()
