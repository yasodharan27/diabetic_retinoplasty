"""IDRiD batch 2 -- one-time cross-dataset evaluation of the frozen EyePACS-adapted graders P-EP, E1-EP and E2-EP
(research record §78; declared in §65 as the last planned use of this test set).

An extension of the batch-1 evaluator, not a second pipeline. From idrid_batch1_eval, unchanged: the image path
(`rgb_from_raw`: Stage 2 once, then the 512 frame), the per-image table and CORN decode (`table_from_logits`), the
pre-inference gates (`gates`: ten raw APTOS images and all 730 validation images against each model's stored BEST
validation table, with the parent protocol's tolerances), the metrics, grade-stratified bootstrap and paired
differences (`summarise`), the report (`report_markdown`), the gate requirement (`require_gates`) and the stored
P tables (`read_p_tables`). From the parent protocol idrid_grading_protocol.json, unchanged: the 103 images and
labels by hash, the three images excluded from the primary set, T4 / mixed_float16 / batch 8, the tolerances.

What this module adds is what differs: the nine pinned EP checkpoints and their identity checks (each must be the
pinned BEST of a run of the EP experiment for that arm and seed, started from the one pinned adapted encoder with
a fresh head, selected by APTOS validation QWK), the P graph for P-EP next to E1's graph for E1-EP / E2-EP, the
four paired differences of this batch, its own confirmation token, lock and output directory, and a hash of
every output file.

P is NOT run: its stored per-image tables are reused after their hashes are checked. No model is trained, no
threshold, decoding rule, checkpoint or seed is chosen on IDRiD. One run; a second is refused except for a
technical failure with the reason recorded.
"""
import datetime
import json
import os

import numpy as np

import arch1_posthoc as ph
import idrid_batch1_eval as b1
import idrid_grading_eval as ig

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOL_PATH = os.path.join(HERE, "idrid_batch2_protocol.json")
SEEDS = b1.SEEDS
ARMS = {"p_ep": "P", "e1_ep": "E1", "e2_ep": "E1"}                # arm -> the graph its checkpoint belongs to
RUN_MODELS = tuple(ARMS)                                          # the only models this module ever runs
MODELS = ("p",) + RUN_MODELS
CONTRASTS = {"p_ep_minus_p": ("p_ep", "p"), "e1_ep_minus_p_ep": ("e1_ep", "p_ep"), "e2_ep_minus_p_ep": ("e2_ep", "p_ep"),
             "e1_ep_minus_e2_ep": ("e1_ep", "e2_ep")}
PRIMARY_CONTRASTS = ("p_ep_minus_p", "e1_ep_minus_p_ep")
LABELS = {"p": "P (stored predictions of the earlier run; not run again)",
          "p_ep": "P-EP (EyePACS-adapted encoder, APTOS-trained)",
          "e1_ep": "E1-EP (the same, with aligned lesion supervision)",
          "e2_ep": "E2-EP (the same, with deranged lesion targets; control)"}
CONTRAST_NAMES = {"p_ep_minus_p": "P-EP − P (primary)", "e1_ep_minus_p_ep": "E1-EP − P-EP (primary)",
                  "e2_ep_minus_p_ep": "E2-EP − P-EP (secondary, descriptive)",
                  "e1_ep_minus_e2_ep": "E1-EP − E2-EP (secondary, descriptive)"}
ProtocolError = ig.ProtocolError


def _json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- protocol and pinned files

def load_protocol(path=PROTOCOL_PATH, parent_path=None):
    """(batch protocol, its sha256, parent protocol). The parent must be the pinned locked protocol -- the same
    file batch 1 used -- and the batch must pin exactly the nine EP checkpoints and this batch's comparisons."""
    protocol = _json(path)
    parent_path = parent_path or os.path.join(HERE, protocol["parent_protocol"]["file"])
    parent, parent_sha = ig.load_protocol(parent_path)
    if parent_sha != protocol["parent_protocol"]["sha256"]:
        raise ProtocolError("the parent IDRiD protocol is not the pinned locked file")
    if tuple(protocol["seeds"]) != SEEDS or tuple(parent["seeds"]) != SEEDS:
        raise ProtocolError("protocol seeds are not 42, 123, 2026")
    expected = {f"{m}_seed{s}" for m in RUN_MODELS for s in SEEDS}
    if set(protocol["checkpoints"]) != expected:
        raise ProtocolError(f"the protocol must pin exactly {sorted(expected)}")
    for name, entry in protocol["checkpoints"].items():
        arm, _, seed = name.rpartition("_seed")
        if entry.get("arm") != arm or int(entry.get("seed", -1)) != int(seed) or entry.get("model_type") != ARMS[arm]:
            raise ProtocolError(f"{name}: the pinned entry is for {entry.get('arm')} / seed {entry.get('seed')} / {entry.get('model_type')}")
    if len({e["sha256"] for e in protocol["checkpoints"].values()}) != len(expected):
        raise ProtocolError("two pinned checkpoints have the same hash")
    if set(protocol["comparisons"]["primary"]) != set(PRIMARY_CONTRASTS) or \
            set(protocol["comparisons"]["primary"]) | set(protocol["comparisons"]["secondary_descriptive"]) != set(CONTRASTS):
        raise ProtocolError("the comparisons are not this batch's four paired differences")
    return protocol, ig.sha256_file(path), parent


def baseline_hashes(parent):
    """SHA-256 of every checkpoint pinned for P / E1 / E2 and the earlier pipeline: none of them is an EP checkpoint."""
    hashes = {entry["sha256"] for entry in parent["checkpoints"].values()}
    hashes |= {entry["sha256"] for entry in _json(b1.PROTOCOL_PATH)["checkpoints"].values()}
    return hashes


def verify_files(protocol, drive_root, parent=None):
    """Every pinned file exists with the pinned SHA-256, and every checkpoint is what the protocol says it is.

    For each of the nine: the weights file's SHA-256; the run's pinned-checkpoint record names this file and this
    hash (it is frozen and was not replaced); the run is the EP experiment's run for this arm, seed and model
    type; it started from the pinned adapted encoder -- not ImageNet's -- with a fresh head; its BEST pointer
    still names the pinned slot and epoch; the run evaluated this file as BEST, selected by APTOS validation QWK;
    and the hash is not one of the P / E1 / E2 checkpoints. Also: the adapted encoder's own pin, the ImageNet file
    the P graph is built from, and the three stored P tables. Raises on the first mismatch; nothing is substituted."""
    root = lambda rel: os.path.join(drive_root, *rel.split("/"))
    parent = parent or ig.load_protocol(os.path.join(HERE, protocol["parent_protocol"]["file"]))[0]
    forbidden = baseline_hashes(parent)
    adapted = protocol["adapted_encoder"]
    adapted_record = _json(os.path.join(root(adapted["run_dir"]), "frozen_checkpoint.json"))
    if adapted_record["sha256"] != adapted["sha256"] or not os.path.exists(root(adapted["path"])) \
            or ig.sha256_file(root(adapted["path"])) != adapted["sha256"]:
        raise ProtocolError("the EyePACS-adapted encoder is missing or is not the pinned checkpoint")
    checkpoints, priors, kinds = {}, {}, {}
    for name, entry in protocol["checkpoints"].items():
        arm, seed = entry["arm"], int(entry["seed"])
        path, run_dir = root(entry["path"]), root(entry["run_dir"])
        if not os.path.exists(path):
            raise ProtocolError(f"{name}: {path} not found")
        if ig.sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"{name}: {path} is not the pinned checkpoint (SHA-256 mismatch)")
        if entry["sha256"] in forbidden or entry["sha256"] == adapted["sha256"]:
            raise ProtocolError(f"{name}: this is a P / E1 / E2 / adaptation checkpoint, not an EP checkpoint")
        for required in ("config.json", "frozen_checkpoint.json", "initialization.json", "result.json"):
            if not os.path.exists(os.path.join(run_dir, required)):
                raise ProtocolError(f"{name}: {entry['run_dir']} is not a finished, pinned EP run (no {required})")
        config, frozen = _json(os.path.join(run_dir, "config.json")), _json(os.path.join(run_dir, "frozen_checkpoint.json"))
        started, result = _json(os.path.join(run_dir, "initialization.json")), _json(os.path.join(run_dir, "result.json"))
        if config.get("experiment") != protocol["experiment"] or config.get("arm") != arm or int(config.get("seed", -1)) != seed \
                or config.get("model_type") != ARMS[arm]:
            raise ProtocolError(f"{name}: the run is {config.get('experiment')} / {config.get('arm')} / seed {config.get('seed')} / "
                                f"{config.get('model_type')}")
        if config.get("adapted_encoder_sha256") != adapted["sha256"] or started.get("adapted_encoder_sha256") != adapted["sha256"] \
                or not started.get("encoder_is_adapted") or started.get("encoder_is_imagenet") or not started.get("fresh_corn_head"):
            raise ProtocolError(f"{name}: the run did not start from the pinned adapted encoder with a fresh head")
        if frozen["sha256"] != entry["sha256"] or f"{entry['run_dir']}/{frozen['weights']}" != entry["path"]:
            raise ProtocolError(f"{name}: the pinned entry is not the run's frozen checkpoint")
        pointer = _json(os.path.join(run_dir, "checkpoints", "best.json"))
        if pointer["active"] != entry["best_slot"] or int(pointer["epoch"]) != entry["best_epoch_index"] \
                or int(frozen["best_epoch_index"]) != entry["best_epoch_index"]:
            raise ProtocolError(f"{name}: the run's BEST pointer no longer names the pinned checkpoint")
        if result["best"]["checkpoint"]["weights_sha256"] != entry["sha256"]:
            raise ProtocolError(f"{name}: the pinned checkpoint is not the one the run evaluated as BEST")
        if result["best"]["checkpoint"]["monitor"] != "val_QWK" or "APTOS validation QWK" not in str(config.get("checkpoint_selection")):
            raise ProtocolError(f"{name}: BEST was not selected by APTOS validation QWK")
        checkpoints[name], kinds[name] = path, ARMS[arm]
        if ARMS[arm] == "E1":
            priors[name] = [float(p) for p in config["lesion_prior"]]
    pretrained = parent["checkpoints"]["convnext_pretrained"]
    if not os.path.exists(root(pretrained["path"])) or ig.sha256_file(root(pretrained["path"])) != pretrained["sha256"]:
        raise ProtocolError("the ImageNet ConvNeXt file the P graph is built from is missing or is not the pinned file")
    p_tables = {}
    for seed in SEEDS:
        entry = protocol["stored_p_tables"][f"p_seed{seed}"]
        path = root(entry["path"])
        if not os.path.exists(path) or ig.sha256_file(path) != entry["sha256"]:
            raise ProtocolError(f"stored P table for seed {seed} is missing or is not the pinned file")
        p_tables[seed] = path
    return {"checkpoints": checkpoints, "p_tables": p_tables, "lesion_prior": priors, "kinds": kinds,
            "pretrained": root(pretrained["path"]), "adapted_encoder_sha256": adapted["sha256"]}


# --------------------------------------------------------------------------- models

def load_models(files, settings):
    """{(arm, seed): keras model}: P-EP in P's graph, E1-EP and E2-EP in E1's graph, built under the protocol's
    dtype policy (set AFTER clear_session) and loaded from the pinned BEST weights; every model is checked to
    really be in that precision."""
    import keras

    import e1_model as em
    import pl_convnext as pl
    from training import checkpointing as ckpt
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy("float32")
    _, reference_arrays = pl.load_reference(files["pretrained"])     # build dependency of the P graph only
    keras.backend.clear_session()
    keras.mixed_precision.set_global_policy(settings["keras_policy"])
    expected = "float16" if settings["keras_policy"] == "mixed_float16" else "float32"
    models = {}
    for arm in RUN_MODELS:
        for seed in SEEDS:
            name = f"{arm}_seed{seed}"
            if files["kinds"][name] == "P":
                net = pl.build_pl_model("P", seed, reference_arrays)
            else:
                net = em.build_e1_model(seed, files["lesion_prior"][name])
                if tuple(net.output_names) != (em.GRADING_OUTPUT, em.LESION_OUTPUT):
                    raise ProtocolError(f"{name}: unexpected outputs {net.output_names}")
            ckpt.load_model_weights_only(net, files["checkpoints"][name])     # every weight comes from the checkpoint
            net.trainable = False
            if str(net.outputs[0].dtype) != expected or keras.mixed_precision.global_policy().name != settings["keras_policy"]:
                raise ProtocolError(f"{name}: the model was not built under {settings['keras_policy']}")
            models[(arm, seed)] = net
    return models


def predict_logits(models, frames, batch_size):
    """{(arm, seed): (N, 4) float64 CORN logits}. P-EP is fed through P's three inputs (the RGB frame in the first
    three channels, everything else zero); E1-EP / E2-EP take the RGB frame and only output 0 is read."""
    import eyepacs_adaptation_train as eat
    out = {key: [] for key in models}
    for start in range(0, len(frames), batch_size):
        rgb = np.stack(frames[start:start + batch_size]).astype(np.float32)
        for (arm, seed), net in models.items():
            if ARMS[arm] == "P":
                z = net.predict_on_batch(eat.p_inputs(rgb))
            else:
                z = net.predict_on_batch({"rgb": rgb})[0]
            out[(arm, seed)].append(np.asarray(z, np.float64))
    return {key: np.concatenate(v, 0) for key, v in out.items()}


# --------------------------------------------------------------------------- gates, summary, report (batch 1's)

def gates(drive_root, aptos_raw_dir, bundle, out_dir, *, aptos_processed_dir=None, protocol_path=PROTOCOL_PATH,
          settings=None, official=True, models=None, log=print):
    """idrid_batch1_eval.gates for this batch's nine checkpoints: file and identity verification, model precision,
    and APTOS parity (ten raw images through the locked image path; all 730 validation images against each
    model's stored BEST validation table). No IDRiD image is read."""
    return b1.gates(drive_root, aptos_raw_dir, bundle, out_dir, aptos_processed_dir=aptos_processed_dir,
                    protocol_path=protocol_path, settings=settings, official=official, models=models, log=log,
                    batch={"load_protocol": load_protocol, "verify_files": verify_files, "load_models": load_models,
                           "predict_logits": predict_logits, "name": "idrid batch 2 pre-inference gates", "label": "IDRiD BATCH 2"})


def summarise(tables, mask, n_boot=ph.N_BOOT):
    """idrid_batch1_eval.summarise for P, P-EP, E1-EP and E2-EP and this batch's four paired differences."""
    return b1.summarise(tables, mask, n_boot, models=MODELS, contrasts=CONTRASTS, primary=PRIMARY_CONTRASTS)


def report_markdown(title, note, summary, protocol):
    return b1.report_markdown(title, note, summary, protocol, labels=LABELS, names=CONTRAST_NAMES)


# --------------------------------------------------------------------------- the one-time run

def lock_path(out_root, protocol_sha):
    return os.path.join(out_root, f"idrid_batch2_lock_{protocol_sha[:12]}.json")


def preflight(drive_root, idrid_raw_dir, out_root, protocol_path=PROTOCOL_PATH):
    """Everything that can be checked without a model and without a prediction: the protocols, the nine
    checkpoints and their identity, the stored P tables, the IDRiD image-set and label hashes and the primary /
    sensitivity subsets, and whether the batch has already been run. Raises on any failure; returns a report."""
    protocol, protocol_sha, parent = load_protocol(protocol_path)
    files = verify_files(protocol, drive_root, parent)
    data = ig.read_idrid(idrid_raw_dir, parent)                      # image-set and label-file hashes; no image is decoded
    b1.read_p_tables(files, data)
    primary = np.asarray(data["primary"], bool)
    excluded = [i for i, keep in zip(data["ids"], primary) if not keep]
    if sorted(excluded) != sorted(parent["dataset"]["primary_exclusions"]) or int(primary.sum()) != 100 or len(primary) != 103:
        raise ProtocolError("the primary / sensitivity subsets are not the locked 100 / 103 images")
    lock = lock_path(out_root, protocol_sha)
    return {"protocol_sha256": protocol_sha, "parent_protocol_sha256": protocol["parent_protocol"]["sha256"],
            "checkpoints": {k: protocol["checkpoints"][k]["sha256"] for k in sorted(files["checkpoints"])},
            "adapted_encoder_sha256": files["adapted_encoder_sha256"], "images": len(data["ids"]),
            "primary_images": int(primary.sum()), "excluded_from_primary": excluded,
            "primary_grade_counts": np.bincount(np.asarray(data["grades"])[primary], minlength=5).tolist(),
            "already_run": os.path.exists(lock), "lock": lock, "confirmation_token": protocol["confirmation_token"], "PASS": True}


def hash_outputs(out_dir):
    """output_hashes.json: the SHA-256 of every file the run wrote (written last)."""
    hashes = {name: ig.sha256_file(os.path.join(out_dir, name)) for name in sorted(os.listdir(out_dir)) if name != "output_hashes.json"}
    with open(os.path.join(out_dir, "output_hashes.json"), "w") as fh:
        json.dump(hashes, fh, indent=1)
    return hashes


def run(drive_root, idrid_raw_dir, out_root, gate, *, confirm, technical_rerun_reason=None, protocol_path=PROTOCOL_PATH,
        command=None, models=None, log=print):
    """THE ONE-TIME EVALUATION OF P-EP, E1-EP AND E2-EP. Refuses without the token, without official gates passed
    in this environment, or when the lock exists (a rerun needs a written technical reason and is recorded).
    The lock is written before any prediction exists. `models` is for tests only."""
    import shutil
    protocol, protocol_sha, parent = load_protocol(protocol_path)
    settings = dict(parent["inference"])
    if confirm != protocol["confirmation_token"]:
        raise ProtocolError("IDRiD batch 2 runs once: pass confirm=<the protocol's confirmation token> to run it")
    b1.require_gates(gate, protocol_sha, parent, settings)
    ig.prepare_gpu()
    os.makedirs(out_root, exist_ok=True)
    lock = lock_path(out_root, protocol_sha)
    previous = None
    if os.path.exists(lock):
        previous = _json(lock)
        if not (isinstance(technical_rerun_reason, str) and technical_rerun_reason.strip()):
            raise ProtocolError(f"IDRiD batch 2 has already been run ({previous['timestamp_utc']}); a rerun is allowed only "
                                "for a technical failure, with technical_rerun_reason stated")
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join(out_root, f"idrid_batch2_ep_{protocol_sha[:12]}_{stamp}")
    if os.path.exists(out_dir):
        raise ProtocolError(f"{out_dir} already exists; an earlier run is never overwritten")
    files = verify_files(protocol, drive_root, parent)
    data = ig.read_idrid(idrid_raw_dir, parent)                      # verifies image-set and label-file hashes
    p_tables = b1.read_p_tables(files, data)
    os.makedirs(out_dir)
    record = {"timestamp_utc": stamp, "protocol_sha256": protocol_sha, "git_commit": ig.git_commit(), "out_dir": out_dir,
              "technical_rerun_reason": technical_rerun_reason, "previous_run": previous}
    with open(lock, "w") as fh:                                      # written BEFORE any prediction exists
        json.dump(record, fh, indent=1)
    with open(os.path.join(out_dir, "lock_record.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    models = models if models is not None else load_models(files, settings)
    frames, native_shapes = [], {}
    for n, (image_id, path) in enumerate(zip(data["ids"], data["paths"]), start=1):
        rgb, native = b1.rgb_from_raw(path)
        frames.append(rgb)
        native_shapes[image_id] = list(native.shape)
        if n % 20 == 0 or n == len(data["paths"]):
            log(f"  inputs {n}/{len(data['paths'])}")
    logits = predict_logits(models, frames, settings["batch_size"])
    tables = {seed: {"p": p_tables[seed]} for seed in SEEDS}
    for (arm, seed), z in logits.items():
        frame, tables[seed][arm] = b1.table_from_logits(data["ids"], data["grades"], z)
        frame["in_primary_set"] = data["primary"]
        frame.to_csv(os.path.join(out_dir, f"per_image_{arm}_seed{seed}.csv"), index=False)      # logits, probabilities, grades
    for seed, path in files["p_tables"].items():                     # copies of the locked P tables, byte for byte
        dst = os.path.join(out_dir, f"per_image_p_seed{seed}_stored.csv")
        shutil.copyfile(path, dst)
        if ig.sha256_file(dst) != protocol["stored_p_tables"][f"p_seed{seed}"]["sha256"]:
            raise ProtocolError("the copied P table differs from the pinned file")
    with open(os.path.join(out_dir, "image_manifest.json"), "w") as fh:
        json.dump({"ids": data["ids"], "sha256": data["hashes"], "grades": [int(g) for g in data["grades"]],
                   "in_primary_set": [bool(b) for b in data["primary"]], "native_shapes": native_shapes,
                   "images_manifest_sha256": parent["dataset"]["images_manifest_sha256"],
                   "label_file_sha256": parent["dataset"]["label_file_sha256"]}, fh, indent=1)
    common = {"protocol": protocol, "protocol_sha256": protocol_sha, "parent_protocol": parent,
              "parent_protocol_sha256": protocol["parent_protocol"]["sha256"], "environment": ig.environment(settings),
              "git_commit": ig.git_commit(), "command": command, "timestamp_utc": stamp, "gates": gate,
              "checkpoints_verified": {k: protocol["checkpoints"][k]["sha256"] for k in sorted(files["checkpoints"])},
              "adapted_encoder_sha256": files["adapted_encoder_sha256"],
              "p_tables_reused": {str(s): {"path": p, "sha256": protocol["stored_p_tables"][f"p_seed{s}"]["sha256"]}
                                  for s, p in files["p_tables"].items()},
              "models_run": [f"{m}_seed{s}" for m in RUN_MODELS for s in SEEDS], "p_rerun": False,
              "excluded_from_primary": parent["dataset"]["primary_exclusions"], "stage3_run": False, "stage4_run": False}
    with open(os.path.join(out_dir, "run_configuration.json"), "w") as fh:
        json.dump(ph._jsonable(common), fh, indent=1)
    primary = summarise(tables, np.asarray(data["primary"], bool))   # the primary result is computed and written first
    with open(os.path.join(out_dir, "primary_results_100.json"), "w") as fh:
        json.dump(ph._jsonable(primary), fh, indent=1)
    with open(os.path.join(out_dir, "primary_report_100.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown("IDRiD batch 2 (P-EP, E1-EP, E2-EP) -- PRIMARY RESULT (100 images)",
                                 "The three test images that are copies of Stage-4 training images (IDRiD_088, IDRiD_089, "
                                 "IDRiD_091) are excluded. Evaluation only; run once.", primary, protocol))
    sensitivity = summarise(tables, np.ones(len(data["ids"]), bool))
    with open(os.path.join(out_dir, "sensitivity_results_103.json"), "w") as fh:
        json.dump(ph._jsonable(sensitivity), fh, indent=1)
    with open(os.path.join(out_dir, "sensitivity_report_103.md"), "w", encoding="utf-8") as fh:
        fh.write(report_markdown("IDRiD batch 2 (P-EP, E1-EP, E2-EP) -- SENSITIVITY ANALYSIS ONLY (103 images)",
                                 "Includes three images that are copies of Stage-4 training images. This is NOT the primary "
                                 "result and never replaces it.", sensitivity, protocol))
    hashes = hash_outputs(out_dir)
    log(f"IDRiD batch 2 written to {out_dir} ({len(hashes)} files hashed)")
    return {"out_dir": out_dir, "primary": primary, "sensitivity": sensitivity, "tables": tables, "output_hashes": hashes}
