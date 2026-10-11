"""Operating-point and calibration diagnostic v1 for the frozen P graders (research record §91). NO INFERENCE.

Reads only per-image outputs that already exist for the three P checkpoints on direct-resize frames: the stored
APTOS validation tables (730 images) and the EyePACS validation tables written by the geometry diagnostic (3,512
images, direct arm). Seven decoding rules are compared:

  A  the current rule: grade = number of cumulative probabilities above 0.5              (no parameter)
  B  expected grade: round(sum of the four cumulative probabilities)                     (no parameter)
  C  most probable class of the implied class distribution                              (no parameter)
  D  temperature on the CORN logits (fitted by NLL), then rule A                         (1, APTOS validation)
  E  one threshold per cut, chosen for QWK                                               (4, APTOS validation)
  F  prior-shift correction: class priors re-estimated by EM on the target's UNLABELLED
     outputs, applied to the temperature-scaled class probabilities, then rule A         (exploratory)
  G  oracle thresholds fitted on EyePACS validation LABELS, cross-fitted by patient      (exploratory; never
                                                                                          generalisation evidence)

D and E are five-fold cross-fitted on APTOS validation (every APTOS image is decoded with parameters fitted
without it) and, for EyePACS, fitted on all of APTOS validation -- EyePACS labels never enter A-E. Nothing is
selected: no rule is adopted, no weight, preprocessing or stored result changes.
"""
import json
import os

import numpy as np
import pandas as pd

import arch1_posthoc as ph

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (42, 123, 2026)
FOLDS, FOLD_SEED = 5, 20261011
GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)            # threshold grid of rules E and G
SWEEPS = 3
APTOS_TABLE = os.path.join(HERE, "results", "PL_ConvNeXtPriors", "2026-09-28_05-22-44", "P_convnext_rgb", "seed_{seed}", "evaluation", "per_sample_best.csv")
EYEPACS_DIR = os.path.join(HERE, "results", "GeometryDiagnostic", "geometry_diag_v1_2026-10-11_17-34-10")
EYEPACS_TABLE = os.path.join(EYEPACS_DIR, "per_image_eyepacs_val_direct_p_seed{seed}.csv")
SPLIT_FILE = os.path.join(HERE, "dataset_splits", "aptos2019_train_val_split.csv")
OUT_DIR = os.path.join(HERE, "results", "OperatingPointDiagnostic")
RULES = ("A", "B", "C", "D", "E", "F", "G")


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cumulative(logits, temperature=1.0):
    """P(grade > k), k = 0..3, from CORN logits (conditional logits; cumulative product of their sigmoids)."""
    return np.cumprod(sigmoid(logits / temperature), axis=1)


def class_probabilities(p_gt):
    full = np.concatenate([np.ones((len(p_gt), 1)), p_gt, np.zeros((len(p_gt), 1))], axis=1)
    return np.clip(full[:, :-1] - full[:, 1:], 1e-12, 1.0)


def decode(p_gt, thresholds=(0.5, 0.5, 0.5, 0.5)):
    return (p_gt > np.asarray(thresholds)[None, :]).sum(axis=1)


def fit_temperature(logits, grades):
    """The temperature that minimises the NLL of the implied class distribution."""
    from scipy.optimize import minimize_scalar
    nll = lambda log_t: -np.mean(np.log(class_probabilities(cumulative(logits, np.exp(log_t)))[np.arange(len(grades)), grades]))   # noqa: E731
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded").x))


def fit_thresholds(p_gt, grades):
    """Four thresholds by coordinate ascent on QWK over a fixed grid, from 0.5; ties keep the value nearest 0.5."""
    tau = np.full(4, 0.5)
    best = ph._fast_qwk(grades, decode(p_gt, tau))
    for _ in range(SWEEPS):
        for k in range(4):
            for value in sorted(GRID, key=lambda v: abs(v - 0.5)):
                trial = tau.copy()
                trial[k] = value
                q = ph._fast_qwk(grades, decode(p_gt, trial))
                if q > best + 1e-12:
                    best, tau = q, trial
    return tau


def em_priors(prob, source_prior, iterations=2000, tol=1e-10):
    """Saerens-Latinne-Decaestecker EM: target class priors from unlabelled outputs under the label-shift
    assumption, and the re-weighted posteriors."""
    prior = source_prior.copy()
    for _ in range(iterations):
        adjusted = prob * (prior / source_prior)[None, :]
        adjusted /= adjusted.sum(axis=1, keepdims=True)
        new = adjusted.mean(axis=0)
        if np.abs(new - prior).max() < tol:
            prior = new
            break
        prior = new
    adjusted = prob * (prior / source_prior)[None, :]
    return prior, adjusted / adjusted.sum(axis=1, keepdims=True)


def ece(confidence, correct, bins=15):
    edges, total = np.linspace(0.0, 1.0, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confidence > lo) & (confidence <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def describe(grades, pred, prob):
    """The reported quantities of one rule on one table. ECE: confidence = the probability the rule's distribution
    gives to the grade it decodes."""
    referable, called = grades >= 2, pred >= 2
    return {"qwk": ph._fast_qwk(grades, pred),
            "recall": [float(np.mean(pred[grades == g] == g)) for g in range(5)],
            "sensitivity": float(called[referable].mean()), "specificity": float((~called[~referable]).mean()),
            "ece": ece(prob[np.arange(len(grades)), pred], (pred == grades).astype(float)),
            "decoded_counts": np.bincount(pred, minlength=5).tolist()}


def stratified_folds(grades, groups=None):
    """Fold index per image. Stratified by grade; with `groups`, whole groups (patients) stay in one fold and are
    stratified by their highest grade."""
    rng = np.random.default_rng(FOLD_SEED)
    fold = np.zeros(len(grades), int)
    if groups is None:
        for g in np.unique(grades):
            idx = rng.permutation(np.flatnonzero(grades == g))
            fold[idx] = np.arange(len(idx)) % FOLDS
        return fold
    frame = pd.DataFrame({"group": groups, "grade": grades})
    top = frame.groupby("group")["grade"].max()
    assignment = {}
    for g in sorted(top.unique()):
        names = rng.permutation(np.asarray(sorted(top.index[top == g])))
        assignment.update({n: k % FOLDS for k, n in enumerate(names)})
    return frame["group"].map(assignment).to_numpy()


def load(path):
    frame = pd.read_csv(path, dtype={"image_id": str})
    return {"ids": frame["image_id"].to_numpy(), "grades": frame["true_grade"].to_numpy(int),
            "logits": frame[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64),
            "stored_pred": frame["predicted_grade"].to_numpy(int)}


def run(log=print):
    aptos = {s: load(APTOS_TABLE.format(seed=s)) for s in SEEDS}
    eyepacs = {s: load(EYEPACS_TABLE.format(seed=s)) for s in SEEDS}
    manifest = pd.read_csv(os.path.join(HERE, "dataset_splits", "eyepacs_adaptation_split_v1.csv"), dtype=str).set_index("image")
    patients = manifest.loc[eyepacs[SEEDS[0]]["ids"], "patient"].to_numpy()
    split = pd.read_csv(SPLIT_FILE, dtype={"id_code": str})
    source_prior = np.bincount(split.loc[split["split"] == "train", "diagnosis"].astype(int), minlength=5) / float((split["split"] == "train").sum())
    for s in SEEDS:                                                    # the tables are the same images in the same order, and rule A is what was stored
        for data in (aptos, eyepacs):
            assert np.array_equal(data[s]["ids"], data[SEEDS[0]]["ids"]) and np.array_equal(data[s]["grades"], data[SEEDS[0]]["grades"])
            assert np.array_equal(decode(cumulative(data[s]["logits"])), data[s]["stored_pred"]), "rule A does not reproduce the stored grades"
    a_grades, e_grades = aptos[SEEDS[0]]["grades"], eyepacs[SEEDS[0]]["grades"]
    a_fold, e_fold = stratified_folds(a_grades), stratified_folds(e_grades, patients)
    out = {"aptos_val": {r: [] for r in RULES}, "eyepacs_val": {r: [] for r in RULES}}
    fitted = {"temperature_all_aptos": [], "temperature_folds": [], "thresholds_all_aptos": [], "thresholds_folds": [],
              "oracle_thresholds_folds": [], "oracle_thresholds_all_eyepacs": [], "em_priors": {"aptos_val": [], "eyepacs_val": []},
              "fold_qwk": {"D_aptos": [], "E_aptos": [], "G_eyepacs": []}}
    preds = {"aptos_val": {r: {} for r in RULES}, "eyepacs_val": {r: {} for r in RULES}}
    for s in SEEDS:
        za, ze = aptos[s]["logits"], eyepacs[s]["logits"]
        pa, pe = cumulative(za), cumulative(ze)
        temperature, tau = fit_temperature(za, a_grades), fit_thresholds(pa, a_grades)
        fitted["temperature_all_aptos"].append(temperature), fitted["thresholds_all_aptos"].append(tau.tolist())
        for name, grades, z, p in (("aptos_val", a_grades, za, pa), ("eyepacs_val", e_grades, ze, pe)):
            cp = class_probabilities(p)
            rules = {"A": (decode(p), cp), "B": (np.clip(np.floor(p.sum(axis=1) + 0.5), 0, 4).astype(int), cp), "C": (cp.argmax(axis=1), cp)}
            if name == "eyepacs_val":                                  # D, E: parameters from ALL of APTOS validation
                pt = cumulative(z, temperature)
                rules["D"], rules["E"] = (decode(pt), class_probabilities(pt)), (decode(p, tau), cp)
            else:                                                      # D, E: five-fold cross-fitted on APTOS validation
                pred_d, pred_e, prob_d = np.zeros(len(grades), int), np.zeros(len(grades), int), np.zeros((len(grades), 5))
                temps, taus, qd, qe = [], [], [], []
                for k in range(FOLDS):
                    fit, held = a_fold != k, a_fold == k
                    t_k, tau_k = fit_temperature(z[fit], grades[fit]), fit_thresholds(p[fit], grades[fit])
                    pt = cumulative(z[held], t_k)
                    pred_d[held], prob_d[held], pred_e[held] = decode(pt), class_probabilities(pt), decode(p[held], tau_k)
                    temps.append(t_k), taus.append(tau_k.tolist())
                    qd.append(ph._fast_qwk(grades[held], pred_d[held])), qe.append(ph._fast_qwk(grades[held], pred_e[held]))
                rules["D"], rules["E"] = (pred_d, prob_d), (pred_e, cp)
                fitted["temperature_folds"].append(temps), fitted["thresholds_folds"].append(taus)
                fitted["fold_qwk"]["D_aptos"].append(qd), fitted["fold_qwk"]["E_aptos"].append(qe)
            # F: EM prior re-estimation on this set's unlabelled, temperature-scaled class probabilities
            prior, adjusted = em_priors(class_probabilities(cumulative(z, temperature)), source_prior)
            tail = np.cumsum(adjusted[:, ::-1], axis=1)[:, ::-1][:, 1:]                    # P(grade > k) of the adjusted distribution
            rules["F"] = (decode(tail), adjusted)
            fitted["em_priors"][name].append(prior.tolist())
            # G: oracle thresholds from EyePACS validation labels (cross-fitted by patient on EyePACS; all of it for APTOS)
            if name == "eyepacs_val":
                pred_g, taus, qg = np.zeros(len(grades), int), [], []
                for k in range(FOLDS):
                    fit, held = e_fold != k, e_fold == k
                    tau_k = fit_thresholds(p[fit], grades[fit])
                    pred_g[held] = decode(p[held], tau_k)
                    taus.append(tau_k.tolist()), qg.append(ph._fast_qwk(grades[held], pred_g[held]))
                rules["G"] = (pred_g, cp)
                fitted["oracle_thresholds_folds"].append(taus), fitted["fold_qwk"]["G_eyepacs"].append(qg)
            else:
                oracle = fit_thresholds(pe, e_grades)
                fitted["oracle_thresholds_all_eyepacs"].append(oracle.tolist())
                rules["G"] = (decode(p, oracle), cp)
            for r in RULES:
                out[name][r].append(describe(grades, rules[r][0], rules[r][1]))
                preds[name][r][s] = rules[r][0]
        log(f"  seed {s}: temperature {temperature:.3f} | thresholds {tau.tolist()}")
    mean = lambda rows, key: np.mean([row[key] for row in rows], axis=0)       # noqa: E731
    summary = {}
    for name, grades in (("aptos_val", a_grades), ("eyepacs_val", e_grades)):
        indices = ph.bootstrap_indices(grades)
        summary[name] = {}
        for r in RULES:
            rows = out[name][r]
            entry = {"qwk_mean": float(mean(rows, "qwk")), "qwk_per_seed": [row["qwk"] for row in rows], "recall_mean": mean(rows, "recall").tolist(),
                     "sensitivity_mean": float(mean(rows, "sensitivity")), "specificity_mean": float(mean(rows, "specificity")),
                     "ece_mean": float(mean(rows, "ece")), "decoded_counts_mean": mean(rows, "decoded_counts").round(0).astype(int).tolist()}
            if r != "A":
                draws = [np.mean([ph._fast_qwk(grades[i], preds[name][r][s][i]) - ph._fast_qwk(grades[i], preds[name]["A"][s][i]) for s in SEEDS]) for i in indices]
                entry["qwk_minus_A"] = {"mean": entry["qwk_mean"] - float(mean(out[name]["A"], "qwk")),
                                        "ci_95": [float(x) for x in np.percentile(draws, [2.5, 97.5])],
                                        "per_seed": [a["qwk"] - b["qwk"] for a, b in zip(rows, out[name]["A"])]}
            summary[name][r] = entry
        summary[name]["auroc_cut_mean"] = {f"ge{c}": float(np.mean([ph._auroc(grades >= c, cumulative(d[s]["logits"])[:, c - 1]) for s in SEEDS]))
                                           for c, d in ((c, aptos if name == "aptos_val" else eyepacs) for c in ph.CUTS)}
        summary[name]["true_prior"] = (np.bincount(grades, minlength=5) / len(grades)).tolist()
    result = {"protocol": "operating-point diagnostic v1 (research record 91)", "folds": FOLDS, "fold_seed": FOLD_SEED,
              "source_prior_aptos_train": source_prior.tolist(), "summary": summary, "fitted": fitted,
              "status": {"A-C": "parameter-free", "D-E": "APTOS validation only; cross-fitted there, EyePACS labels never used",
                         "F": "exploratory: unsupervised, uses target inputs; assumes label shift",
                         "G": "exploratory oracle: fitted on EyePACS validation labels; not generalisation evidence"}}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "operating_point_v1.json"), "w") as fh:
        json.dump(ph._jsonable(result), fh, indent=1)
    return result


if __name__ == "__main__":
    r = run()
    f3 = lambda v: [round(float(x), 3) for x in v]                              # noqa: E731
    for name in ("aptos_val", "eyepacs_val"):
        print("==", name, "| AUROC per cut", {k: round(v, 3) for k, v in r["summary"][name]["auroc_cut_mean"].items()}, "| true prior", f3(r["summary"][name]["true_prior"]))
        for rule in RULES:
            e = r["summary"][name][rule]
            d = e.get("qwk_minus_A")
            print(f"  {rule}  QWK {e['qwk_mean']:.4f} {f3(e['qwk_per_seed'])} | recall {f3(e['recall_mean'])} | sens {e['sensitivity_mean']:.3f} spec {e['specificity_mean']:.3f}"
                  f" | ECE {e['ece_mean']:.3f} | counts {e['decoded_counts_mean']}" + (f" | vs A {d['mean']:+.4f} [{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}]" if d else ""))
    ft = r["fitted"]
    print("temperature (all APTOS val)", f3(ft["temperature_all_aptos"]), "| fold range", [f3([min(t), max(t)]) for t in ft["temperature_folds"]])
    print("thresholds (all APTOS val)", ft["thresholds_all_aptos"])
    print("threshold fold ranges", [[f3([min(c), max(c)]) for c in zip(*seed)] for seed in ft["thresholds_folds"]])
    print("oracle thresholds (EyePACS folds, range per cut)", [[f3([min(c), max(c)]) for c in zip(*seed)] for seed in ft["oracle_thresholds_folds"]])
    print("fold QWK SD: D", f3([np.std(q, ddof=1) for q in ft["fold_qwk"]["D_aptos"]]), "E", f3([np.std(q, ddof=1) for q in ft["fold_qwk"]["E_aptos"]]),
          "G", f3([np.std(q, ddof=1) for q in ft["fold_qwk"]["G_eyepacs"]]))
    print("source prior", f3(r["source_prior_aptos_train"]), "| EM priors APTOS", [f3(p) for p in ft["em_priors"]["aptos_val"]], "| EyePACS", [f3(p) for p in ft["em_priors"]["eyepacs_val"]])
