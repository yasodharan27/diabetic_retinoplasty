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
            keras.mixed_precision.set_global_policy(policy)
            tiers[policy] = {}
            for seed, path in p_weight_paths.items():
                seed = int(seed)
                keras.backend.clear_session()
                p = pl.build_pl_model("P", seed, reference_arrays)
                ckpt.load_model_weights_only(p, path)
                e1 = em.build_e1_model(seed, lesion_prior)
                report = em.copy_from_p(e1, p)
                p_logits, e_logits, lesion = paired_logits(p, e1, bundle, ids)
                row = {"copy_report": report, "e1_vs_live_p": compare(p_logits, e_logits),
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
