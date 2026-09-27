"""
ICDR-ordered two-route ordinal head (research direction "C1") -- an ADDITIVE Stage-8 module.

`corn.py`, `weighted_corn.py`, and every Stage 05-07 module are imported unmodified; nothing here
changes the frozen representation E. Named `icdr_two_route_*` rather than `c1_*` because
`racaf_c1_control_model.py` already uses "C1" for the unrelated RACAF gate control.

WHAT IS BEING TESTED
    CORN is a continuation-ratio chain: task k is fitted on grades >= k (`corn.py:151`), so every
    grade-4 label is a POSITIVE for NPDR tasks 0-2, and a grade-4 prediction is reachable only
    through tasks 0-2. ICDR defines PDR disjunctively (neovascularisation or vitreous/preretinal
    haemorrhage), independent of how many NPDR lesions are present. The hypothesis is that the
    chain structure itself gates PDR images with low intraretinal lesion burden down to <= 2.

    This head is NOT a new ordinal model family: splitting off a category first (a hurdle / a
    sequential split tree) is an established statistical idea. The contribution being tested is
    the mechanism-driven DR application under a controlled, capacity-matched comparison.

HEADS (both read the SAME frozen 256-d Stage-7 output E, both have 1,028 parameters)
    H1  CORN refit    Dense(256 -> 4)                        4 CORN tasks, grades 0-4
    H2  two-route     Dense(256 -> 1)  PDR route      q = sigmoid(z_pdr) = P(grade 4)
                      Dense(256 -> 3)  NPDR route     3 CORN tasks over grades 0-3 ONLY
                      P(4) = q ;  P(k) = (1 - q) * P_NPDR(k),  k = 0..3
                      decode: 4 if q > 0.5, else the CORN decode over grades 0-3

FITTING (identical for H1 and H2; pre-registered, no tuning)
    Every binary task -- H1's four CORN tasks, H2's three NPDR tasks and its PDR task -- is fitted
    by the SAME function, `fit_weighted_logistic`: an L2-regularised, class-weighted logistic
    regression, loss = sum_i w_i * BCE_i / n_task + (L2 / 2) * ||beta||^2, bias unpenalised, full-
    batch L-BFGS from a zero start (convex, deterministic). With a linear head the CORN tasks share
    no parameters, so fitting them one by one IS the CORN likelihood decomposition. (The pooled
    denominator of `weighted_corn` would only rescale each task's effective L2; per-task
    normalisation was chosen in advance so H1 and H2 run through one identical fitter.)
    Per-sample weights: `weighted_corn.PREREGISTERED_CLASS_WEIGHTS[grade]` (square-root inverse
    frequency, training counts only) -- the same values for every task of both heads.
    Features: standardised with the TRAINING-split mean/SD only, then folded back into the Dense
    kernel/bias, so the exported heads act on raw E exactly like the original CORN head.

    The ONLY difference between H1 and H2 is the (subset, target) of each task:
        H1 task k (k = 0..3):  grades >= k,             target grade > k
        H2 NPDR task k (0..2): grades >= k AND <= 3,    target grade > k   (grade 4 EXCLUDED)
        H2 PDR task:           all grades,              target grade == 4
"""

import hashlib
import json

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

import corn
import weighted_corn

# ---------------------------------------------------------------------------------------------
# Pre-registered constants -- fixed before any fit on real data. Do not edit after a run.
# ---------------------------------------------------------------------------------------------

HEAD_VERSION = "icdr-two-route-head-v1"
D_MODEL = corn.D_MODEL                      # 256, the frozen Stage-7 embedding width
NUM_GRADES = corn.NUM_GRADES                # 5
PDR_GRADE = NUM_GRADES - 1                  # 4
NPDR_NUM_GRADES = NUM_GRADES - 1            # 4 (grades 0-3)
NPDR_NUM_THRESHOLDS = NPDR_NUM_GRADES - 1   # 3

#: L2 strength, fixed in advance (NEXT_STEP_RECOMMENDATION.md, Step cRT: "L2 = 1e-4 fixed in
#: advance"). Never selected on validation data.
L2 = 1e-4
#: Per-sample class weights: the project's pre-registered square-root inverse-frequency weights.
CLASS_WEIGHTS = tuple(float(w) for w in weighted_corn.PREREGISTERED_CLASS_WEIGHTS)
#: The PDR decision threshold for decoding. Fixed at exactly 0.5; not a tunable parameter.
PDR_DECISION_THRESHOLD = 0.5
#: L-BFGS configuration (deterministic; zero initialisation).
LBFGS_MAXITER = 20000
LBFGS_GTOL = 1e-9
LBFGS_FTOL = 1e-15
SD_FLOOR = 1e-12

#: Expected trainable-parameter count of BOTH heads: 256 * 4 + 4.
EXPECTED_HEAD_PARAMETERS = D_MODEL * corn.NUM_THRESHOLDS + corn.NUM_THRESHOLDS  # 1,028

if EXPECTED_HEAD_PARAMETERS != 1028:
    raise RuntimeError(f"Expected 1,028 head parameters, got {EXPECTED_HEAD_PARAMETERS}.")


def fitting_configuration():
    """Everything that defines how H1 and H2 are fitted -- recorded with every result."""
    return {
        "head_version": HEAD_VERSION,
        "fitter": "per-task L2-regularised class-weighted logistic regression, scipy L-BFGS-B, "
                  "zero initialisation, bias unpenalised, loss normalised by task sample count",
        "l2": L2,
        "class_weights": list(CLASS_WEIGHTS),
        "class_weight_policy": "weighted_corn.PREREGISTERED_CLASS_WEIGHTS (sqrt inverse "
                               "frequency, training counts 1444/296/799/154/236)",
        "standardisation": "training-split mean/SD only; folded into Dense kernel/bias",
        "lbfgs": {"maxiter": LBFGS_MAXITER, "gtol": LBFGS_GTOL, "ftol": LBFGS_FTOL},
        "pdr_decision_threshold": PDR_DECISION_THRESHOLD,
        "expected_head_parameters": EXPECTED_HEAD_PARAMETERS,
    }


# ---------------------------------------------------------------------------------------------
# 1. Task definitions -- the ONLY place H1 and H2 differ
# ---------------------------------------------------------------------------------------------

def h1_tasks(grades):
    """H1 (CORN refit): four tasks. Returns [(name, mask, target)], each over all input rows."""
    grades = np.asarray(grades, dtype=np.int64)
    tasks = []
    for k in range(corn.NUM_THRESHOLDS):
        mask = grades >= k
        tasks.append((f"corn_task_{k}", mask, (grades > k).astype(np.float64)))
    return tasks


def h2_tasks(grades):
    """H2 (two-route): the PDR task over all rows, then three NPDR CORN tasks from which grade-4
    rows are EXCLUDED. Returns [(name, mask, target)] in parameter order [pdr, npdr_0..2]."""
    grades = np.asarray(grades, dtype=np.int64)
    tasks = [("pdr_route", np.ones_like(grades, dtype=bool),
              (grades == PDR_GRADE).astype(np.float64))]
    npdr = grades <= PDR_GRADE - 1
    for k in range(NPDR_NUM_THRESHOLDS):
        mask = (grades >= k) & npdr
        tasks.append((f"npdr_task_{k}", mask, (grades > k).astype(np.float64)))
    return tasks


# ---------------------------------------------------------------------------------------------
# 2. The single fitter shared by every task of both heads
# ---------------------------------------------------------------------------------------------

def standardiser(e_train):
    """Training-only mean/SD. A zero-variance column gets SD 1 (it then carries no signal)."""
    e_train = np.asarray(e_train, dtype=np.float64)
    mean = e_train.mean(axis=0)
    sd = e_train.std(axis=0)
    sd = np.where(sd < SD_FLOOR, 1.0, sd)
    return mean, sd


def fit_weighted_logistic(x, y, w, l2=L2):
    """Minimises  sum_i w_i * BCE(y_i, x_i.beta + b) / n  +  (l2 / 2) * ||beta||^2.
    `x`: (n, d) standardised features; `y`: (n,) in {0, 1}; `w`: (n,) positive weights.
    Convex; deterministic (zero start, full batch). Returns (beta, b, info)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n, d = x.shape
    if n == 0 or y.min() == y.max():
        raise ValueError(f"A task needs both classes present; got n={n}, "
                         f"positives={int(y.sum()) if n else 0}.")
    if np.any(w <= 0):
        raise ValueError("Sample weights must be positive.")

    def objective(theta):
        beta, b = theta[:d], theta[d]
        z = x @ beta + b
        loss = np.logaddexp(0.0, z) - y * z
        residual = w * (expit(z) - y) / n
        value = float(np.sum(w * loss) / n + 0.5 * l2 * beta @ beta)
        grad = np.empty(d + 1)
        grad[:d] = x.T @ residual + l2 * beta
        grad[d] = residual.sum()
        return value, grad

    result = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                      options={"maxiter": LBFGS_MAXITER, "gtol": LBFGS_GTOL,
                               "ftol": LBFGS_FTOL, "maxfun": LBFGS_MAXITER * 2})
    _, grad = objective(result.x)
    info = {"converged": bool(result.success), "message": str(result.message),
            "iterations": int(result.nit), "final_objective": float(result.fun),
            "final_grad_max_abs": float(np.max(np.abs(grad))), "n": int(n),
            "n_positive": int(y.sum())}
    return result.x[:d].copy(), float(result.x[d]), info


def _fit_tasks(e_train, grades_train, tasks, l2):
    e_train = np.asarray(e_train, dtype=np.float64)
    grades_train = np.asarray(grades_train, dtype=np.int64)
    if e_train.ndim != 2 or e_train.shape[0] != grades_train.shape[0]:
        raise ValueError(f"E {e_train.shape} and grades {grades_train.shape} do not align.")
    mean, sd = standardiser(e_train)
    x = (e_train - mean) / sd
    weights = np.asarray(CLASS_WEIGHTS, dtype=np.float64)[grades_train]
    kernel, bias, infos = [], [], {}
    for name, mask, target in tasks:
        beta, b, info = fit_weighted_logistic(x[mask], target[mask], weights[mask], l2)
        info["grades_in_task"] = {int(g): int(c) for g, c in
                                  zip(*np.unique(grades_train[mask], return_counts=True))}
        infos[name] = info
        # fold the standardisation back so the head acts on raw E: z = E.(beta/sd) + b - mu.(beta/sd)
        raw_beta = beta / sd
        kernel.append(raw_beta)
        bias.append(b - float(mean @ raw_beta))
    return np.stack(kernel, axis=1), np.asarray(bias), infos


class FittedHead:
    """A fitted linear head on raw E. `kind` is "H1_corn_refit" or "H2_two_route".
    `kernel`: (256, 4); `bias`: (4,). For H2 the column order is [pdr, npdr_0, npdr_1, npdr_2]."""

    def __init__(self, kind, kernel, bias, task_info, l2):
        self.kind = kind
        self.kernel = np.asarray(kernel, dtype=np.float64)
        self.bias = np.asarray(bias, dtype=np.float64)
        self.task_info = task_info
        self.l2 = l2

    def logits(self, e):
        return np.asarray(e, dtype=np.float64) @ self.kernel + self.bias

    def predict(self, e):
        z = self.logits(e)
        if self.kind == "H1_corn_refit":
            return corn_outputs(z)
        return two_route_outputs(z[:, 0], z[:, 1:])

    def parameter_count(self):
        return int(self.kernel.size + self.bias.size)

    def fingerprint(self):
        payload = self.kernel.tobytes() + self.bias.tobytes()
        return hashlib.sha256(payload).hexdigest()


def fit_h1_corn_refit(e_train, grades_train, l2=L2):
    kernel, bias, info = _fit_tasks(e_train, grades_train, h1_tasks(grades_train), l2)
    return FittedHead("H1_corn_refit", kernel, bias, info, l2)


def fit_h2_two_route(e_train, grades_train, l2=L2):
    kernel, bias, info = _fit_tasks(e_train, grades_train, h2_tasks(grades_train), l2)
    return FittedHead("H2_two_route", kernel, bias, info, l2)


# ---------------------------------------------------------------------------------------------
# 3. Probability construction and decoding
# ---------------------------------------------------------------------------------------------

def corn_outputs(logits):
    """H0/H1: the project's own CORN decode (`corn.decode_logits`), in float64, plus the two
    scores the experiment compares. P(grade >= 3) = p_gt_2; P(grade 4) = p_gt_3."""
    logits = np.asarray(logits, dtype=np.float64)
    decoded = corn.decode_logits(logits)
    p_cond = expit(logits)
    p_cum = np.cumprod(p_cond, axis=1)
    probabilities = np.concatenate([1.0 - p_cum[:, :1], p_cum[:, :-1] - p_cum[:, 1:],
                                    p_cum[:, -1:]], axis=1)
    return {"probabilities": probabilities,
            "predicted_grade": decoded["predicted_grade"].astype(np.int64),
            "p_ge3": p_cum[:, 2], "p_grade4": p_cum[:, 3]}


def two_route_outputs(pdr_logit, npdr_logits):
    """H2: q = sigmoid(z_pdr); P_NPDR from the 3-task CORN chain; P(4) = q,
    P(k) = (1 - q) * P_NPDR(k). Decode: 4 if q > 0.5, else the CORN decode over grades 0-3."""
    pdr_logit = np.asarray(pdr_logit, dtype=np.float64).reshape(-1)
    npdr_logits = np.asarray(npdr_logits, dtype=np.float64)
    if npdr_logits.shape != (pdr_logit.shape[0], NPDR_NUM_THRESHOLDS):
        raise ValueError(f"npdr_logits must be (n, {NPDR_NUM_THRESHOLDS}), got {npdr_logits.shape}.")
    q = expit(pdr_logit)
    p_cum = np.cumprod(expit(npdr_logits), axis=1)
    p_npdr = np.concatenate([1.0 - p_cum[:, :1], p_cum[:, :-1] - p_cum[:, 1:], p_cum[:, -1:]],
                            axis=1)
    probabilities = np.concatenate([(1.0 - q)[:, None] * p_npdr, q[:, None]], axis=1)
    npdr_grade = np.sum(p_cum > 0.5, axis=1).astype(np.int64)
    predicted = np.where(q > PDR_DECISION_THRESHOLD, PDR_GRADE, npdr_grade).astype(np.int64)
    return {"probabilities": probabilities, "predicted_grade": predicted,
            "p_ge3": probabilities[:, 3] + probabilities[:, 4], "p_grade4": q,
            "q": q, "p_npdr": p_npdr, "npdr_grade": npdr_grade}


# ---------------------------------------------------------------------------------------------
# 4. Keras heads -- parameter verification and the deployable Stage-8 interface
# ---------------------------------------------------------------------------------------------

def build_corn_refit_head():
    """H1's architecture: exactly the project's CORN head, `corn.build_corn_model()`."""
    return corn.build_corn_model()


def build_two_route_head(d_model=D_MODEL):
    """H2's architecture: Dense(d -> 1) PDR route + Dense(d -> 3) NPDR CORN route, output
    logits concatenated as [z_pdr, n_0, n_1, n_2]."""
    from keras import Input, Model, layers
    e_input = Input(shape=(d_model,), name="E")
    pdr = layers.Dense(1, activation=None, name="pdr_route")(e_input)
    npdr = layers.Dense(NPDR_NUM_THRESHOLDS, activation=None, name="npdr_corn_route")(e_input)
    logits = layers.Concatenate(axis=-1, name="two_route_logits")([pdr, npdr])
    return Model(inputs=e_input, outputs=logits, name="icdr_two_route_head")


def trainable_counts(model):
    variables = model.trainable_variables
    return {"trainable_parameters": int(sum(int(np.prod(v.shape)) for v in variables)),
            "trainable_tensors": len(variables)}


def parameter_parity_report():
    """Builds both Keras heads and verifies they have exactly the same parameter count as the
    original CORN head. Raises if the difference is not zero."""
    original = trainable_counts(corn.build_corn_model())
    h1 = trainable_counts(build_corn_refit_head())
    h2 = trainable_counts(build_two_route_head())
    report = {"original_corn": original, "H1_corn_refit": h1, "H2_two_route": h2,
              "difference_H2_minus_H1": h2["trainable_parameters"] - h1["trainable_parameters"],
              "difference_H2_minus_original": (h2["trainable_parameters"]
                                               - original["trainable_parameters"])}
    for name in ("original_corn", "H1_corn_refit", "H2_two_route"):
        if report[name]["trainable_parameters"] != EXPECTED_HEAD_PARAMETERS:
            raise RuntimeError(f"{name} has {report[name]['trainable_parameters']} parameters, "
                               f"expected {EXPECTED_HEAD_PARAMETERS}: {json.dumps(report)}")
    return report


def to_keras(fitted):
    """Loads a FittedHead into its Keras architecture (for deployment / parity checks)."""
    if fitted.kind == "H1_corn_refit":
        model = build_corn_refit_head()
        model.get_layer("corn_logits").set_weights(
            [fitted.kernel.astype(np.float32), fitted.bias.astype(np.float32)])
    else:
        model = build_two_route_head()
        model.get_layer("pdr_route").set_weights(
            [fitted.kernel[:, :1].astype(np.float32), fitted.bias[:1].astype(np.float32)])
        model.get_layer("npdr_corn_route").set_weights(
            [fitted.kernel[:, 1:].astype(np.float32), fitted.bias[1:].astype(np.float32)])
    return model


# ---------------------------------------------------------------------------------------------
# 5. Frozen-backbone guards
# ---------------------------------------------------------------------------------------------

def weights_fingerprint(model):
    """SHA-256 over every weight's name, shape and bytes -- trainable or not. Used to prove a
    frozen backbone is bit-for-bit unchanged after E extraction."""
    digest = hashlib.sha256()
    for variable in model.weights:
        value = np.asarray(variable.numpy())
        digest.update(variable.path.encode("utf-8") if hasattr(variable, "path")
                      else variable.name.encode("utf-8"))
        digest.update(str(value.shape).encode("utf-8"))
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def freeze(model):
    """Marks a loaded backbone non-trainable and verifies it exposes no trainable variable."""
    model.trainable = False
    if model.trainable_variables:
        raise RuntimeError(f"{model.name} still exposes {len(model.trainable_variables)} "
                           "trainable variables after freezing.")
    return model
