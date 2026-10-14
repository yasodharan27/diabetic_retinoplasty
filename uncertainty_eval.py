"""Stage 10 -- uncertainty of the final grader from its three-seed ensemble (research record §96). PREPARED, NOT RUN.

No model is loaded here: everything is computed from per-image tables that already exist (one per seed, P's
schema: `logit_0..3`, `true_grade`). For every image:

  * the ENSEMBLE prediction: the three seeds' cumulative probabilities averaged, decoded with the fixed 0.5 rule;
  * three uncertainty scores -- the predictive entropy of the ensemble's class distribution, the seed
    disagreement (standard deviation of the three expected grades), and the margin to the nearest 0.5 threshold.

Whether a score is useful is tested the standard way, by selective prediction: images are ranked by the score,
the most uncertain are referred, and agreement (QWK, accuracy) is measured on the rest at each coverage. A score
is informative when the retained set improves as coverage falls, and more than under random referral.

Honest framing: ConvNeXt has no dropout, so this is deep-ensemble uncertainty from three seeds, not Monte-Carlo
dropout. The earlier attempt to use a segmentation-derived reliability score for selective prediction was
rejected (D6, 2026-09-22); this is a different signal and is evaluated on its own evidence.
"""
import json
import os

import numpy as np
import pandas as pd

import arch1_posthoc as ph

SCORES = ("entropy", "seed_disagreement", "threshold_margin")
COVERAGES = (1.0, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5)
RANDOM_REPEATS, RANDOM_SEED = 200, 20261012


def cumulative(logits):
    return np.cumprod(1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64))), axis=1)


def class_probabilities(p_gt):
    full = np.concatenate([np.ones((len(p_gt), 1)), p_gt, np.zeros((len(p_gt), 1))], axis=1)
    return np.clip(full[:, :-1] - full[:, 1:], 1e-12, 1.0)


def read_seed_tables(paths):
    """{seed: path} -> (ids, grades, logits (seeds, N, 4)); the tables must be the same images in the same order."""
    frames = [pd.read_csv(paths[s], dtype={"image_id": str}) for s in sorted(paths)]
    ids, grades = frames[0]["image_id"].to_numpy(), frames[0]["true_grade"].to_numpy(int)
    for f in frames[1:]:
        if not np.array_equal(f["image_id"].to_numpy(), ids) or not np.array_equal(f["true_grade"].to_numpy(int), grades):
            raise ValueError("the seed tables are not the same images in the same order")
    return ids, grades, np.stack([f[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64) for f in frames])


def ensemble(logits):
    """Per image: ensemble grade, class distribution and the three uncertainty scores (higher = less certain)."""
    p_seed = np.stack([cumulative(z) for z in logits])               # (seeds, N, 4)
    p = p_seed.mean(axis=0)
    prob = class_probabilities(p)
    prob = prob / prob.sum(axis=1, keepdims=True)
    return {"pred": (p > 0.5).sum(axis=1).astype(int), "p_gt": p, "class_prob": prob,
            "entropy": -(prob * np.log(prob)).sum(axis=1),
            "seed_disagreement": p_seed.sum(axis=2).std(axis=0),
            "threshold_margin": -np.abs(p - 0.5).min(axis=1),          # negated: a small margin is an uncertain image
            "seed_pred": (p_seed > 0.5).sum(axis=2).astype(int)}


def selective(grades, pred, score, coverages=COVERAGES):
    """Agreement on the retained images when the most uncertain are referred, at each coverage."""
    order = np.argsort(score, kind="stable")                         # most certain first
    rows = []
    for c in coverages:
        keep = order[:max(1, int(round(c * len(order))))]
        rows.append({"coverage": c, "n": int(len(keep)), "qwk": ph._fast_qwk(grades[keep], pred[keep]),
                     "accuracy": float((pred[keep] == grades[keep]).mean()),
                     "referable_retained": float((grades[keep] >= 2).mean())})
    return rows


def random_baseline(grades, pred, coverages=COVERAGES, repeats=RANDOM_REPEATS, seed=RANDOM_SEED):
    """The same curve under random referral: mean and 95 % range over `repeats` random orders."""
    rng = np.random.default_rng(seed)
    out = {c: [] for c in coverages}
    for _ in range(repeats):
        order = rng.permutation(len(grades))
        for c in coverages:
            keep = order[:max(1, int(round(c * len(order))))]
            out[c].append(ph._fast_qwk(grades[keep], pred[keep]))
    return [{"coverage": c, "qwk_mean": float(np.mean(v)), "qwk_low": float(np.percentile(v, 2.5)), "qwk_high": float(np.percentile(v, 97.5))}
            for c, v in out.items()]


def error_detection_auroc(grades, pred, score):
    """How well the score ranks wrong predictions above right ones (0.5 = uninformative)."""
    wrong = pred != grades
    return ph._auroc(wrong, score)


def evaluate(paths, name):
    """Everything for one image set: the ensemble's agreement, each score's error-detection AUROC and selective
    curve, the random-referral baseline, and how often the three seeds disagree on the decoded grade."""
    ids, grades, logits = read_seed_tables(paths)
    e = ensemble(logits)
    result = {"set": name, "images": int(len(grades)), "ensemble_qwk": ph._fast_qwk(grades, e["pred"]),
              "per_seed_qwk": [ph._fast_qwk(grades, e["seed_pred"][k]) for k in range(len(logits))],
              "seeds_disagree_on_grade": float((e["seed_pred"].min(axis=0) != e["seed_pred"].max(axis=0)).mean()),
              "random_referral": random_baseline(grades, e["pred"]), "scores": {}}
    for s in SCORES:
        result["scores"][s] = {"error_detection_auroc": error_detection_auroc(grades, e["pred"], e[s]),
                               "selective": selective(grades, e["pred"], e[s])}
    table = pd.DataFrame({"image_id": ids, "true_grade": grades, "ensemble_grade": e["pred"], **{s: e[s] for s in SCORES},
                          **{f"class_prob_{k}": e["class_prob"][:, k] for k in range(5)}})
    return result, table


def run(sets, out_dir):
    """`sets`: {name: {seed: per-image table path}}. Writes one JSON and one per-image CSV per set."""
    os.makedirs(out_dir, exist_ok=True)
    results = {}
    for name, paths in sets.items():
        results[name], table = evaluate(paths, name)
        table.to_csv(os.path.join(out_dir, f"uncertainty_{name}.csv"), index=False)
    with open(os.path.join(out_dir, "uncertainty_results.json"), "w") as fh:
        json.dump(ph._jsonable(results), fh, indent=1)
    return results


# --------------------------------------------------------------------------- test-time augmentation (a separate uncertainty method)

def dihedral(rgb, k):
    """The k-th of the 8 flips / quarter turns (k = 0 is the identity) -- the transformations seen in training."""
    x = np.rot90(np.asarray(rgb), k % 4, axes=(-3, -2))
    return np.flip(x, axis=-2) if k >= 4 else x


def tta_logits(model_fn, rgb):
    """`model_fn(rgb (N, H, W, 3)) -> logits (N, 4)` on each of the 8 transformations: logits (8, N, 4). The grade is
    orientation-free, so the transformations should agree; their spread is the uncertainty."""
    return np.stack([np.asarray(model_fn(np.ascontiguousarray(dihedral(rgb, k))), np.float64) for k in range(8)])


def tta_summary(logits_tta):
    """For logits (T, N, 4) of one model under T transformations: the TTA-mean grade, the standard deviation of
    the expected grade across transformations, and how often the decoded grade differs between them."""
    p = np.stack([cumulative(z) for z in logits_tta])
    grades = (p > 0.5).sum(axis=2)
    return {"pred": (p.mean(axis=0) > 0.5).sum(axis=1).astype(int), "expected_grade_sd": p.sum(axis=2).std(axis=0),
            "grade_changes": grades.min(axis=0) != grades.max(axis=0)}
