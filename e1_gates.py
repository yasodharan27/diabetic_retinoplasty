"""The pre-training gates of the E1 multi-task grader. None of them trains E1 or reads a grade label for a
decision; a failed gate stops everything (tolerances are never loosened).

  Gate 1  p_parity      E1's grading path, given a P checkpoint's weights, IS P: logits / cumulative
                        probabilities / decoded grades on the 730 validation images, float32 (tier 1) and
                        mixed_float16 against the stored P BEST tables (tier 2); plus equality at initialisation.
  Gate 2  log_alias     On the running Keras version: the actual log keys of the two-output model, the alias
                        callback, and that the checkpoint state and the stored history hold val_QWK. Runs one
                        epoch on the synthetic test bundle with random weights (mechanics only, no result).
  Gate 3  targets       Statistics and hard checks of all training targets (label-free).
  (also)  gradients     Each loss alone sends a finite, non-zero gradient into every encoder variable.
"""
import datetime
import json
import os

import numpy as np

import e1_data as ed
import e1_model as em
import e1_train as et

FLOAT32_TOL = {"logit_max_abs": 1e-4, "probability_max_abs": 1e-4}          # tier 1
MIXED_TOL = {"logit_max_abs": 0.05, "probability_max_abs": 0.01}            # tier 2 (record §60)
INIT_TOL = 1e-5                                                              # pl_convnext.PL_P_EQUIVALENCE_TOL
INIT_IMAGES = 16


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def _write(out_dir, name, payload):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, name), "w") as fh:
            json.dump(payload, fh, indent=1, default=float)
    return payload


def cumulative(logits):
    """CORN cumulative probabilities P(grade > k) from logits (float64)."""
    return np.cumprod(1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64))), axis=1)


def decode(logits):
    return (cumulative(logits) > 0.5).sum(axis=1).astype(int)


def p_inputs(rgb):
    """P's three inputs for a batch of RGB frames: RGB in channels 0-2 of the 8-channel input (P reads nothing
    else), and the two inert auxiliary inputs as zeros."""
    import joint_training_model as jtm
    rgb = np.asarray(rgb, np.float32)
    x8 = np.concatenate([rgb, np.zeros(rgb.shape[:3] + (5,), np.float32)], axis=-1)
    return [x8, np.zeros((len(rgb),) + tuple(jtm.STAGE6_INPUT_SHAPE), np.float32), np.zeros((len(rgb), 1), np.float32)]


def paired_logits(p_model, e1_model, bundle, image_ids, batch_size=8):
    """(P logits, E1 grading logits, E1 lesion logits) on the same unaugmented RGB frames."""
    p_out, e_out, lesion = [], [], []
    ids = [str(i) for i in image_ids]
    for s in range(0, len(ids), batch_size):
        rgb = np.stack([ed.load_inputs(bundle, i)["rgb"] for i in ids[s:s + batch_size]])
        p_out.append(np.asarray(p_model.predict_on_batch(p_inputs(rgb)), np.float64))
        grading, z = e1_model.predict_on_batch({"rgb": rgb})
        e_out.append(np.asarray(grading, np.float64))
        lesion.append(np.asarray(z, np.float32))
    return np.concatenate(p_out, 0), np.concatenate(e_out, 0), np.concatenate(lesion, 0)


def compare(a_logits, b_logits):
    return {"logit_max_abs": float(np.abs(a_logits - b_logits).max()),
            "probability_max_abs": float(np.abs(cumulative(a_logits) - cumulative(b_logits)).max()),
            "grades_equal": bool(np.array_equal(decode(a_logits), decode(b_logits))),
            "images": int(len(a_logits))}


def _check(name, result, tol, failures):
    if result["logit_max_abs"] > tol["logit_max_abs"]:
        failures.append(f"{name}: grading logits differ by {result['logit_max_abs']:.3g} (> {tol['logit_max_abs']})")
    if result["probability_max_abs"] > tol["probability_max_abs"]:
        failures.append(f"{name}: cumulative probabilities differ by {result['probability_max_abs']:.3g} "
                        f"(> {tol['probability_max_abs']})")
    if not result["grades_equal"]:
        failures.append(f"{name}: decoded grades differ")


def p_parity(bundle, p_weight_paths, convnext_weights_path, lesion_prior, *, policies=("float32", "mixed_float16"),
             stored_root=None, expected_sha256=None, image_ids=None, out_dir=None, official=True, log=print):
    """Gate 1. `p_weight_paths`: {seed: pinned P BEST model.weights.h5}. `expected_sha256`: {seed: sha256} --
    required for the official gate. `stored_root`: the experiments root with the stored P BEST per-sample
    tables (needed for the mixed_float16 tier). `image_ids`: default the bundle's 730 validation images.
    `official=False` marks a pre-check on a subset or another machine; it cannot make E1 ready for training."""
    import keras
    import pandas as pd

    import arch1_posthoc as ph
    import pathology_grader_fusion as pf
    import pl_convnext as pl
    import stage34_cache_v2 as cache
    from training import checkpointing as ckpt
    ids = [str(i) for i in (image_ids if image_ids is not None else bundle.val_ids)]
    failures, tiers, shas = [], {}, {}
    if official:
        if list(ids) != list(bundle.val_ids):
            failures.append("the official gate uses the bundle's validation images, all of them, in order")
        if not expected_sha256:
            failures.append("the official gate needs the pinned P checkpoint hashes")
        import tensorflow as tf
        if not tf.config.list_physical_devices("GPU"):
            failures.append("the official gate runs on the GPU runtime that will train (mixed_float16 tier)")
    for seed, path in p_weight_paths.items():
        shas[int(seed)] = cache.sha256_file(path)
        if expected_sha256 and shas[int(seed)] != expected_sha256[int(seed)]:
            failures.append(f"P-{seed}: checkpoint sha256 {shas[int(seed)]} is not the pinned {expected_sha256[int(seed)]}")
    previous = keras.mixed_precision.global_policy().name
    try:
        keras.mixed_precision.set_global_policy("float32")
        reference, reference_arrays = pl.load_reference(convnext_weights_path)          # arrays taken in float32
        # Initialisation parity: E1 from the ImageNet weights and the seed == P from the same, before training.
        init = {}
        for seed in p_weight_paths:
            p = pl.build_pl_model("P", int(seed), reference_arrays)
            e1 = em.build_e1_model(int(seed), lesion_prior, reference)
            a, b, _ = paired_logits(p, e1, bundle, ids[:INIT_IMAGES])
            init[int(seed)] = dict(compare(a, b), tolerance=INIT_TOL)
            if init[int(seed)]["logit_max_abs"] > INIT_TOL:
                failures.append(f"initialisation, seed {seed}: grading logits differ from P by "
                                f"{init[int(seed)]['logit_max_abs']:.3g} (> {INIT_TOL})")
            log(f"  init parity seed {seed}: logits {init[int(seed)]['logit_max_abs']:.3g}")
            del p, e1
        for policy in policies:
            tiers[policy] = {}
            expected_dtype = "float16" if policy == "mixed_float16" else "float32"
            for seed, path in p_weight_paths.items():
                seed = int(seed)
                # Keras 3: clear_session() resets the global dtype policy to float32, so the policy is set
                # AFTER it, for every model, and the built models are checked to really be in that precision.
                keras.backend.clear_session()
                keras.mixed_precision.set_global_policy(policy)
                p = pl.build_pl_model("P", seed, reference_arrays)
                ckpt.load_model_weights_only(p, path)
                e1 = em.build_e1_model(seed, lesion_prior)
                dtypes = {"p": str(p.outputs[0].dtype), "e1_grading": str(e1.outputs[0].dtype)}
                if set(dtypes.values()) != {expected_dtype}:
                    raise RuntimeError(f"{policy}, seed {seed}: the models were not built in this precision "
                                       f"(grading outputs {dtypes}, expected {expected_dtype})")
                report = em.copy_from_p(e1, p)
                p_logits, e_logits, lesion = paired_logits(p, e1, bundle, ids)
                row = {"copy_report": report, "e1_vs_live_p": compare(p_logits, e_logits),
                       "grading_output_dtype": dtypes, "policy_in_force": keras.mixed_precision.global_policy().name,
                       "lesion_output_finite": bool(np.isfinite(lesion).all()),
                       "lesion_output_dtype": str(lesion.dtype), "lesion_output_shape": list(lesion.shape[1:])}
                if not row["lesion_output_finite"]:
                    failures.append(f"{policy}, seed {seed}: non-finite auxiliary output")
                if policy == "float32":
                    _check(f"float32, seed {seed} (E1 vs P, same weights)", row["e1_vs_live_p"], FLOAT32_TOL, failures)
                else:
                    if stored_root is None:
                        failures.append("the mixed_float16 tier needs the stored P BEST tables (stored_root)")
                    else:
                        table = pf.p_prediction_path(stored_root, seed, "best")
                        frame = pd.read_csv(table)
                        stored = ph._read_table(table)
                        index = {str(i): n for n, i in enumerate(stored["ids"])}
                        rows = [index[i] for i in ids]
                        stored_logits = frame[[f"logit_{k}" for k in range(4)]].to_numpy()[rows]
                        row["e1_vs_stored_p_best"] = {
                            "logit_max_abs": float(np.abs(e_logits - stored_logits).max()),
                            "probability_max_abs": float(np.abs(cumulative(e_logits) - stored["p_gt"][rows]).max()),
                            "grades_equal": bool(np.array_equal(decode(e_logits), stored["pred"][rows])),
                            "images": len(ids), "table": table}
                        _check(f"mixed_float16, seed {seed} (E1 vs stored P BEST)", row["e1_vs_stored_p_best"],
                               MIXED_TOL, failures)
                tiers[policy][seed] = row
                shown = row.get("e1_vs_stored_p_best", row["e1_vs_live_p"])
                log(f"  {policy} seed {seed}: logits {shown['logit_max_abs']:.3g} | probabilities "
                    f"{shown['probability_max_abs']:.3g} | grades equal {shown['grades_equal']}")
                del p, e1
    finally:
        keras.mixed_precision.set_global_policy(previous)
    if official and set(policies) != {"float32", "mixed_float16"}:
        failures.append("the official gate runs both tiers")
    result = {"gate": "p_parity", "kind": "official gate" if official else "pre-check (cannot make E1 ready)",
              "official": bool(official), "PASS": not failures, "failures": failures, "images": len(ids),
              "tolerances": {"float32": FLOAT32_TOL, "mixed_float16": MIXED_TOL, "initialisation": INIT_TOL},
              "initialisation": init, "tiers": tiers, "p_checkpoint_sha256": shas, "environment": environment(),
              "timestamp_utc": _now()}
    log(f"GATE 1 P PARITY {'PASS' if result['PASS'] else 'FAIL'} ({result['kind']})"
        + ("" if result["PASS"] else " | " + "; ".join(failures)))
    return _write(out_dir, "gate1_p_parity.json", result)


def environment():
    import keras
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    name = None
    if gpus:
        try:
            name = tf.config.experimental.get_device_details(gpus[0]).get("device_name")
        except Exception:  # noqa: BLE001
            name = "GPU"
    return {"keras": keras.__version__, "tensorflow": tf.__version__, "gpu": name,
            "policy": keras.mixed_precision.global_policy().name}


# --------------------------------------------------------------------------- gate 2: log aliases

def log_alias(work_dir, bundle, grade_of, reference, class_weights, *, lesion_prior=(0.1, 0.1, 0.1, 0.1),
              mixed_precision=True, repo_dir=None, out_dir=None, official=True, log=print):
    """Gate 2. One epoch of e1_train.train_seed on `bundle` -- the SYNTHETIC test bundle with a random-weight
    reference -- in `work_dir`, to observe on this Keras version: the raw log keys, the aliases, the callback
    order, the checkpoint state's monitor value and the stored history. Mechanics only; the numbers mean
    nothing and are not a result. `official`: run on the Colab runtime that will train."""
    import keras
    import tensorflow as tf

    import multiseed_runs as msr
    from training import checkpointing as ckpt
    failures = []
    run_dir = os.path.join(work_dir, "alias_gate_run")
    captured = {}
    original = et.with_aliases

    def recording(callbacks, strict=True):
        wrapped = original(callbacks, strict)
        captured["order"] = [type(c).__name__ for c in wrapped]
        captured["alias"] = wrapped[0]

        class _After(keras.callbacks.Callback):                 # last: what the callbacks after the alias saw
            def on_epoch_end(self, epoch, logs=None):
                captured["final_keys"] = sorted(logs or {})

        return wrapped + [_After()]

    et.with_aliases = recording
    try:
        et.train_seed(run_dir, bundle, 42, list(lesion_prior), reference, class_weights, repo_dir=repo_dir or os.getcwd(),
                      staging_dir=os.path.join(work_dir, "alias_gate_staging"), max_epochs=1, grade_of=grade_of,
                      mixed_precision=mixed_precision, log=lambda *a: None)
    finally:
        et.with_aliases = original
    alias = captured.get("alias")
    raw = list(getattr(alias, "observed_keys", []))
    for key in et.REQUIRED_TRAIN_KEYS + et.REQUIRED_VAL_KEYS:
        if key not in raw:
            failures.append(f"Keras did not log {key!r}")
    if captured.get("order", [None])[0] != "LogAliases":
        failures.append(f"the alias callback is not first: {captured.get('order')}")
    for name in et.ALIASES.values():
        if name not in captured.get("final_keys", []):
            failures.append(f"alias {name!r} was not visible to the later callbacks")
    history = msr.read_history(run_dir)
    if len(history) != 1:
        failures.append(f"expected one stored history row, found {len(history)}")
    else:
        for name in et.ALIASES.values():
            if history[0].get(name) is None:
                failures.append(f"stored history has no {name!r}")
    generation = ckpt.find_resumable_generation(os.path.join(run_dir, "checkpoints"), verbose=False)
    state = ckpt.read_state(generation) if generation else None
    if state is None or state.monitor != "val_QWK" or state.best_metric is None:
        failures.append("the checkpoint state did not record val_QWK as its monitored value")
    best_dir, _ = msr.read_best(run_dir)
    if best_dir is None:
        failures.append("no BEST checkpoint was published from the aliased monitor")
    result = {"gate": "log_alias", "kind": "official gate" if official else "pre-check (cannot make E1 ready)",
              "official": bool(official), "PASS": not failures, "failures": failures, "raw_keras_log_keys": raw,
              "keys_after_alias": captured.get("final_keys"), "aliases_created": list(getattr(alias, "created", [])),
              "callback_order": captured.get("order"), "stored_history_row": history[0] if history else None,
              "checkpoint_monitor": getattr(state, "monitor", None),
              "checkpoint_best_metric": getattr(state, "best_metric", None),
              "mixed_precision_requested": bool(mixed_precision),
              "gpu_present": bool(tf.config.list_physical_devices("GPU")), "environment": environment(),
              "note": "synthetic bundle, random weights, one epoch: mechanics only", "timestamp_utc": _now()}
    log(f"GATE 2 LOG ALIAS {'PASS' if result['PASS'] else 'FAIL'} ({result['kind']}): raw keys {raw}"
        + ("" if result["PASS"] else " | " + "; ".join(failures)))
    return _write(out_dir, "gate2_log_alias.json", result)


# --------------------------------------------------------------------------- gate 3: targets

def targets(bundle, *, out_dir=None, image_ids=None, bruteforce_every=1, official=True, log=print):
    """Gate 3. All training targets (default: the bundle's 2,921 training images). Sparse classes do not fail
    the gate; only invalid values, wrong shapes, collapse, a pooling disagreement or a misaligned file do."""
    ids = list(image_ids if image_ids is not None else bundle.train_ids)
    report = ed.target_statistics(bundle, ids, bruteforce_every=bruteforce_every, log=log)
    failures = list(report["failures"])
    if official and ids != list(bundle.train_ids):
        failures.append("the official gate inspects every training image of the bundle")
    result = {"gate": "targets", "kind": "official gate" if official else "pre-check (cannot make E1 ready)",
              "official": bool(official), "PASS": not failures, "failures": failures,
              "statistics": report["statistics"], "lesion_prior": report["lesion_prior"],
              "lesion_prior_logits": [float(v) for v in em.prior_logits(report["lesion_prior"])],
              "classes": list(em.LESION_CLASSES), "bundle_fingerprint": bundle.fingerprint,
              "stage4_sha256": bundle.stage4_sha256, "timestamp_utc": _now()}
    log(f"GATE 3 TARGETS {'PASS' if result['PASS'] else 'FAIL'} ({result['kind']}): {len(ids)} images"
        + ("" if result["PASS"] else " | " + "; ".join(failures[:5])))
    return _write(out_dir, "gate3_targets.json", result)


# --------------------------------------------------------------------------- gradient check

def gradient_report(model, rgb, grades, lesion_targets, class_weights):
    """Gradients of each loss ALONE with respect to every encoder variable (the ConvNeXt stages; not the head
    LayerNorm, which the lesion head cannot reach). PASS: all finite and none identically zero, for both."""
    import tensorflow as tf

    import weighted_corn
    variables = [v for layer in em.encoder_layers(model) for v in layer.trainable_weights]
    corn_loss = weighted_corn.make_weighted_corn_loss(list(class_weights))
    with tf.GradientTape(persistent=True) as tape:
        grading, lesion = model({"rgb": tf.convert_to_tensor(rgb, tf.float32)}, training=True)
        l_corn = corn_loss(tf.convert_to_tensor(grades), grading)
        l_lesion = et.lesion_loss(tf.convert_to_tensor(lesion_targets, tf.float32), lesion)
    out = {"encoder_variables": len(variables), "corn_loss": float(l_corn), "lesion_loss": float(l_lesion)}
    failures = []
    for name, loss in (("corn", l_corn), ("lesion", l_lesion)):
        grads = tape.gradient(loss, variables)
        missing = sum(g is None for g in grads)
        norms = [float(tf.norm(tf.cast(g, tf.float32))) for g in grads if g is not None]
        finite = bool(np.all(np.isfinite(norms)))
        zero = int(sum(n == 0.0 for n in norms))
        out[name] = {"variables_without_gradient": int(missing), "all_finite": finite, "zero_gradients": zero,
                     "min_norm": float(min(norms)) if norms else None, "max_norm": float(max(norms)) if norms else None}
        if missing or not finite or zero:
            failures.append(f"{name} loss: {missing} variables without gradient, {zero} zero, finite {finite}")
    head = model.get_layer(em.LESION_CONV_NAME).trainable_weights
    out["lesion_head_gradient_from_corn_loss"] = [g is not None for g in tape.gradient(l_corn, head)]
    del tape
    out["PASS"], out["failures"] = not failures, failures
    return out


# --------------------------------------------------------------------------- gate 1 diagnostic (no pass / fail)

LOGIT_BINS = (0.0, 2.0, 4.0, 8.0, 16.0, np.inf)
DIFF_COUNTS = (0.05, 0.1, 0.5)


def _distribution(values, thresholds=DIFF_COUNTS):
    values = np.asarray(values, np.float64).ravel()
    out = {"n": int(values.size), "max": float(values.max()), "mean": float(values.mean()),
           "median": float(np.median(values))}
    out.update({f"p{q}": float(np.percentile(values, q)) for q in (90, 95, 99)})
    out.update({f"n_above_{t}": int((values > t).sum()) for t in thresholds})
    return out


def logit_difference_report(image_ids, reference_logits, recomputed_logits):
    """How two sets of CORN logits for the same images differ: distributions of the absolute differences
    (per element and per-image maximum), the same for cumulative probabilities and decoded grades, the worst
    image, where the differences sit relative to the logit magnitude (float16 resolution grows with it), and
    whether they are a shift or scatter. Returns (summary, per-image frame)."""
    import pandas as pd
    ids = [str(i) for i in image_ids]
    a, b = np.asarray(reference_logits, np.float64), np.asarray(recomputed_logits, np.float64)
    diff, signed = np.abs(b - a), b - a
    pa, pb = cumulative(a), cumulative(b)
    pdiff = np.abs(pb - pa)
    ga, gb = decode(a), decode(b)
    magnitude = np.abs(a)
    step = np.spacing(np.abs(a).astype(np.float16)).astype(np.float64)       # float16 step at each reference logit
    in_steps = diff / step
    per_image = diff.max(axis=1)
    worst = int(np.argmax(per_image))
    by_magnitude = []
    for lo, hi in zip(LOGIT_BINS[:-1], LOGIT_BINS[1:]):
        sel = (magnitude >= lo) & (magnitude < hi)
        by_magnitude.append({"abs_logit_from": lo, "abs_logit_to": None if np.isinf(hi) else hi, "n": int(sel.sum()),
                             "mean_abs_diff": float(diff[sel].mean()) if sel.any() else None,
                             "max_abs_diff": float(diff[sel].max()) if sel.any() else None,
                             "n_above_0.05": int((diff[sel] > 0.05).sum()),
                             "max_diff_in_float16_steps": float(in_steps[sel].max()) if sel.any() else None})
    big = diff > 0.05
    rank_m, rank_d = np.argsort(np.argsort(magnitude.ravel())), np.argsort(np.argsort(diff.ravel()))
    summary = {
        "images": len(ids),
        "logit_abs_diff": _distribution(diff),
        "logit_abs_diff_per_image_max": _distribution(per_image),
        "probability_abs_diff": _distribution(pdiff, thresholds=(0.001, 0.005, 0.01)),
        "probability_abs_diff_per_image_max": _distribution(pdiff.max(axis=1), thresholds=(0.001, 0.005, 0.01)),
        "decoded_grades_differ": int((ga != gb).sum()),
        "worst_image": {"image_id": ids[worst], "index": worst, "max_abs_logit_diff": float(per_image[worst]),
                        "threshold_index": int(np.argmax(diff[worst])),
                        "reference_logits": a[worst].tolist(), "recomputed_logits": b[worst].tolist(),
                        "abs_logit_diff": diff[worst].tolist(),
                        "reference_cumulative_probabilities": pa[worst].tolist(),
                        "recomputed_cumulative_probabilities": pb[worst].tolist(),
                        "max_abs_probability_diff": float(pdiff[worst].max()),
                        "reference_grade": int(ga[worst]), "recomputed_grade": int(gb[worst])},
        "saturation": {
            "by_abs_reference_logit": by_magnitude,
            "rank_correlation_abs_logit_vs_abs_diff": float(np.corrcoef(rank_m, rank_d)[0, 1]),
            "elements_above_0.05": int(big.sum()),
            "abs_reference_logit_of_elements_above_0.05": (
                {"min": float(magnitude[big].min()), "median": float(np.median(magnitude[big])),
                 "max": float(magnitude[big].max())} if big.any() else None),
            "share_of_elements_above_0.05_with_abs_logit_ge_8": float((magnitude[big] >= 8).mean()) if big.any() else None,
            "max_probability_diff_among_elements_above_0.05": float(pdiff[big].max()) if big.any() else None,
            "max_diff_in_float16_steps": float(in_steps.max()), "median_diff_in_float16_steps": float(np.median(in_steps))},
        "shift": {"mean_signed_diff": float(signed.mean()), "mean_signed_diff_per_threshold": signed.mean(axis=0).tolist(),
                  "share_positive": float((signed > 0).mean()), "share_zero": float((signed == 0).mean()),
                  "mean_abs_diff": float(diff.mean()),
                  "images_with_max_diff_above_0.05": int((per_image > 0.05).sum())}}
    frame = pd.DataFrame({"index": np.arange(len(ids)), "image_id": ids, "max_abs_logit_diff": per_image,
                          "max_abs_probability_diff": pdiff.max(axis=1), "reference_grade": ga, "recomputed_grade": gb})
    for k in range(a.shape[1]):
        frame[f"reference_logit_{k}"], frame[f"recomputed_logit_{k}"] = a[:, k], b[:, k]
        frame[f"abs_logit_diff_{k}"] = diff[:, k]
        frame[f"reference_p_gt_{k}"], frame[f"recomputed_p_gt_{k}"] = pa[:, k], pb[:, k]
    return summary, frame


def p_parity_diagnostic(bundle, p_weight_paths, convnext_weights_path, lesion_prior, stored_root, *, expected_sha256=None,
                        policy="mixed_float16", image_ids=None, out_dir=None, batch_size=8, alt_batch_size=2, log=print):
    """DIAGNOSTIC of the gate-1 tier-2 result; it has no pass / fail and changes no rule or tolerance.

    Per seed, under `policy` on this runtime, on the validation images: P (the pinned BEST checkpoint) is
    recomputed and compared with P's STORED validation logits -- element by element, with the worst image, the
    distributions, the dependence on logit magnitude and the sign pattern -- and, independently of the stored
    table, E1 with the copied weights is compared with that recomputed P. Two more recomputations of P place
    the stored-table difference in context: the same policy at another batch size (within-runtime float16
    variation) and float32. Per-image CSVs and one JSON are written to `out_dir`."""
    import keras
    import pandas as pd

    import arch1_posthoc as ph
    import pathology_grader_fusion as pf
    import pl_convnext as pl
    import stage34_cache_v2 as cache
    from training import checkpointing as ckpt
    ids = [str(i) for i in (image_ids if image_ids is not None else bundle.val_ids)]
    previous = keras.mixed_precision.global_policy().name
    seeds = {}
    try:
        keras.mixed_precision.set_global_policy("float32")
        _, reference_arrays = pl.load_reference(convnext_weights_path)
        for seed, path in p_weight_paths.items():
            seed = int(seed)
            sha = cache.sha256_file(path)
            if expected_sha256 and sha != expected_sha256[seed]:
                raise RuntimeError(f"P-{seed}: checkpoint sha256 {sha} is not the pinned {expected_sha256[seed]}")
            table = pf.p_prediction_path(stored_root, seed, "best")
            frame = pd.read_csv(table)
            stored = ph._read_table(table)
            index = {str(i): n for n, i in enumerate(stored["ids"])}
            rows = [index[i] for i in ids]
            stored_logits = frame[[f"logit_{k}" for k in range(4)]].to_numpy(np.float64)[rows]
            stored_table_consistent = bool(np.abs(cumulative(stored_logits) - stored["p_gt"][rows]).max() < 1e-6
                                           and np.array_equal(decode(stored_logits), stored["pred"][rows]))

            def run(policy_name, batch, with_e1):
                keras.backend.clear_session()
                keras.mixed_precision.set_global_policy(policy_name)
                p = pl.build_pl_model("P", seed, reference_arrays)
                ckpt.load_model_weights_only(p, path)
                if not with_e1:
                    out = []
                    for s in range(0, len(ids), batch):
                        rgb = np.stack([ed.load_inputs(bundle, i)["rgb"] for i in ids[s:s + batch]])
                        out.append(np.asarray(p.predict_on_batch(p_inputs(rgb)), np.float64))
                    return np.concatenate(out, 0), None
                e1 = em.build_e1_model(seed, lesion_prior)
                em.copy_from_p(e1, p)
                p_logits, e_logits, _ = paired_logits(p, e1, bundle, ids, batch_size=batch)
                return p_logits, e_logits

            p_logits, e_logits = run(policy, batch_size, True)
            summary, per_image = logit_difference_report(ids, stored_logits, p_logits)
            for k in range(4):
                per_image[f"e1_logit_{k}"] = e_logits[:, k]
            per_image["e1_equals_recomputed_p"] = np.all(e_logits == p_logits, axis=1)
            p_alt, _ = run(policy, alt_batch_size, False)
            p_f32, _ = run("float32", batch_size, False)
            alt_summary, _ = logit_difference_report(ids, p_logits, p_alt)
            f32_vs_stored, _ = logit_difference_report(ids, stored_logits, p_f32)
            f32_vs_policy, _ = logit_difference_report(ids, p_f32, p_logits)
            keep = ("logit_abs_diff", "probability_abs_diff", "decoded_grades_differ")
            seeds[seed] = {
                "p_checkpoint_sha256": sha, "stored_table": table,
                "stored_table_internally_consistent": stored_table_consistent,
                "stored_logit_range": [float(stored_logits.min()), float(stored_logits.max())],
                "recomputed_p_vs_stored_p": summary,
                "e1_vs_recomputed_p_same_runtime": dict(compare(p_logits, e_logits),
                                                        bitwise_equal=bool(np.array_equal(p_logits, e_logits))),
                "e1_vs_stored_p": compare(stored_logits, e_logits),
                "recomputed_p_other_batch_size_vs_this_one_same_policy": {k: alt_summary[k] for k in keep},
                "recomputed_p_float32_vs_stored_p": {k: f32_vs_stored[k] for k in keep},
                "recomputed_p_policy_vs_recomputed_p_float32": {k: f32_vs_policy[k] for k in keep}}
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                per_image.to_csv(os.path.join(out_dir, f"gate1_diagnostic_per_image_seed{seed}.csv"), index=False)
            d, w = summary["logit_abs_diff"], summary["worst_image"]
            log(f"  seed {seed}: P recomputed vs stored: logits max {d['max']:.4f} mean {d['mean']:.4f} median "
                f"{d['median']:.4f} p99 {d['p99']:.4f} | >0.05: {d['n_above_0.05']} | worst {w['image_id']} | "
                f"probabilities max {summary['probability_abs_diff']['max']:.4f} | grades differ "
                f"{summary['decoded_grades_differ']} | E1 == P same runtime: "
                f"{seeds[seed]['e1_vs_recomputed_p_same_runtime']['bitwise_equal']}")
    finally:
        keras.mixed_precision.set_global_policy(previous)
    result = {"diagnostic": "gate 1, tier 2 (no pass / fail; no rule or tolerance changed)", "policy": policy,
              "batch_size": batch_size, "alt_batch_size": alt_batch_size, "images": len(ids), "seeds": seeds,
              "environment": environment(), "timestamp_utc": _now()}
    return _write(out_dir, "gate1_diagnostic.json", result)
