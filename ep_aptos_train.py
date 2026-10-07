"""APTOS runs of the EyePACS-adapted arms (research record §69.2): P-EP, E1-EP and -- only if eligible -- E2-EP.

Every run starts from the SAME pinned EyePACS-adapted encoder (eyepacs_adaptation_train.pin_checkpoint) with a
freshly initialised CORN head, and is otherwise the existing APTOS experiment it mirrors:

  p_ep   P    -- pl_convnext.build_pl_model("P", seed, adapted arrays) + compile_pl_model
  e1_ep  E1   -- e1_model.build_e1_model + e1_train.compile_e1_model (1x1 lesion head, lambda = 1, aligned targets)
  e2_ep  E2   -- E1-EP with e2_control's fixed derangement for the TRAINING targets (validation stays aligned)

Unchanged from P / E1 / E2: arch1_train.P_PROTOCOL (batch 2, AdamW 1e-4, weight decay 0.05, <= 50 epochs, early
stopping 12, LR reduction), the pre-registered APTOS class weights, the verified v2 bundle and its 2,921 / 730
split, P's epoch order and per-image augmentation RNG, BEST by APTOS validation QWK, seeds 42 -> 123 -> 2026.
The only difference is the encoder the run starts from. The loop is eyepacs_adaptation_train.training_loop.

Order and conditions (enforced here, not left to the notebook):
  * no APTOS run starts before the adaptation checkpoint is pinned AND its held-out EyePACS test evaluation is
    recorded (the evaluation's numbers are not read -- only that the declared order was kept);
  * E1-EP runs after P-EP is complete; both are mandatory;
  * E2-EP runs only with a recorded E1-EP-vs-P-EP probe result whose pre-declared criterion was met.

`assert_initialisation` proves, for every fresh run, that the encoder is the pinned adapted one (and not the
ImageNet one) and that the CORN head is the seed's fresh initialisation; the facts go to initialization.json.
"""
import hashlib
import json
import os
import posixpath

import numpy as np

import arch1_train as at
import eyepacs_adaptation_train as eat

EXPERIMENT = "EyePACSAdaptedAPTOS"
SEEDS = at.SEEDS
PROTOCOL = at.P_PROTOCOL
ARMS = {"p_ep": {"model": "P", "mirrors": "P", "mandatory": True, "requires": ()},
        "e1_ep": {"model": "E1", "mirrors": "E1", "mandatory": True, "requires": ("p_ep",)},
        "e2_ep": {"model": "E1", "mirrors": "E2", "mandatory": False, "requires": ("p_ep", "e1_ep")}}
SELECTION = "APTOS validation QWK (as P)"
#: The pre-declared E2-EP condition (record §69.2 C): the §68 criterion with the EP arms.
E2_EP_CONDITION = {"contrast": "e1_ep_minus_p_ep", "primary_variant": "five_epoch",
                   "criterion": "E1-EP - P-EP mean cell AUROC positive in 3/3 seeds AND the 95% paired bootstrap interval "
                                "of the three-seed mean excludes zero (2,000 resamples, seed 20260927)"}
ELIGIBILITY_NAME = "e1_ep_probe_criterion.json"
#: Eligibility does not start E2-EP: the run also needs this phrase, typed by a person.
E2_EP_CONFIRMATION = "TRAIN E2-EP ON APTOS"


def _array_digest(array):
    array = np.ascontiguousarray(np.asarray(array, np.float32))
    return hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def encoder_digest(arrays):
    """Order-independent content hash of a set of encoder arrays (the two graphs order them differently)."""
    return hashlib.sha256("".join(sorted(_array_digest(a) for a in arrays)).encode()).hexdigest()


# --------------------------------------------------------------------------- the pinned adapted encoder

def load_adapted_encoder(adaptation_run_dir, imagenet_arrays, *, expected_sha256=None, require_heldout_test=True,
                         image_size=None):
    """The pinned EyePACS-adapted encoder: {"arrays", "sha256", "digest", "imagenet_digest", "record"}.
    Refuses a run that is not pinned, whose weights are not the pinned file, whose hash is not `expected_sha256`
    (when given), whose held-out test evaluation is not yet recorded, or whose encoder equals the ImageNet one."""
    weights_path, record = eat.read_frozen(adaptation_run_dir)
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise RuntimeError("the pinned adaptation checkpoint is not the expected one")
    if require_heldout_test:
        heldout = eat.heldout_result(adaptation_run_dir)
        if heldout is None or heldout["checkpoint_sha256"] != record["sha256"]:
            raise RuntimeError("the held-out EyePACS test evaluation of the pinned checkpoint is not recorded yet "
                               "(record §69.2 D: it precedes every APTOS run)")
    model, _ = eat.load_frozen(adaptation_run_dir, imagenet_arrays, mixed_precision=False, image_size=image_size)
    arrays = eat.adapted_backbone_arrays(model)
    del model
    if len(arrays) != len(imagenet_arrays):
        raise RuntimeError("the adapted encoder does not have the ConvNeXt-Tiny arrays")
    digest, imagenet_digest = encoder_digest(arrays), encoder_digest(imagenet_arrays)
    if digest == imagenet_digest:
        raise RuntimeError("the 'adapted' encoder is identical to the ImageNet encoder")
    return {"arrays": arrays, "sha256": record["sha256"], "digest": digest, "imagenet_digest": imagenet_digest,
            "record": record}


def adapted_backbone(adapted, image_size=None):
    """The adapted arrays as a Keras ConvNeXt (the form arch1_model.copy_backbone_weights copies from by name)."""
    import pl_convnext as pl
    backbone = pl.build_backbone("P", image_size or pl.IMAGE_SIZE)
    pl.copy_pretrained(backbone, adapted["arrays"], "P")
    return backbone


# --------------------------------------------------------------------------- model

def _encoder_arrays(arm, model):
    import arch1_model as a1
    import pl_convnext as pl
    if ARMS[arm]["model"] == "P":
        return [v.numpy() for v in model.get_layer(pl.BACKBONE_NAME).weights]
    return [w for layer in a1.backbone_layers(model) for w in layer.get_weights()]


def assert_initialisation(arm, model, seed, adapted, lesion_prior=None):
    """Raises unless the fresh model's encoder is exactly the pinned adapted encoder (hence not ImageNet's), its
    CORN head is the seed's fresh initialisation, and (E1 / E2) the lesion head is E1's seeded initialisation.
    Returns the facts recorded in initialization.json."""
    import pl_convnext as pl
    arrays = _encoder_arrays(arm, model)
    digest = encoder_digest(arrays)
    if len(arrays) != len(adapted["arrays"]) or digest != adapted["digest"]:
        raise RuntimeError(f"{arm}-{seed}: the model's encoder is not the pinned EyePACS-adapted encoder")
    if digest == adapted["imagenet_digest"]:
        raise RuntimeError(f"{arm}-{seed}: the model's encoder is the ImageNet encoder")
    kernel, bias = pl.corn_head_initial_weights(seed)
    head = model.get_layer("corn").get_layer("corn_logits").get_weights()
    if not (np.array_equal(head[0], kernel) and np.array_equal(head[1], bias)):
        raise RuntimeError(f"{arm}-{seed}: the CORN head is not the seed's fresh initialisation")
    facts = {"arm": arm, "model_type": ARMS[arm]["model"], "encoder_is_adapted": True, "encoder_is_imagenet": False,
             "adapted_encoder_sha256": adapted["sha256"], "adapted_encoder_digest": digest,
             "encoder_arrays": len(arrays), "fresh_corn_head": True}
    if ARMS[arm]["model"] == "E1":
        import e1_model as em
        from keras import initializers
        lesion_kernel, lesion_bias = model.get_layer(em.LESION_CONV_NAME).get_weights()
        expected = np.asarray(initializers.GlorotUniform(seed=int(seed))(lesion_kernel.shape))
        if not (np.array_equal(lesion_kernel, expected) and np.allclose(lesion_bias, em.prior_logits(lesion_prior), atol=1e-6)):
            raise RuntimeError(f"{arm}-{seed}: the lesion head is not E1's initialisation")
        facts["fresh_lesion_head"] = True
    return facts


def build_model(arm, seed, adapted, class_weights, lesion_prior=None, protocol=PROTOCOL, mixed_precision=True,
                image_size=None):
    """The compiled model of one arm and seed, started from the pinned adapted encoder with a fresh CORN head."""
    import pl_convnext as pl
    from training import enable_mixed_precision
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
    enable_mixed_precision(bool(mixed_precision))
    size = image_size or pl.IMAGE_SIZE
    if ARMS[arm]["model"] == "P":
        model = pl.build_pl_model("P", int(seed), adapted["arrays"], image_size=size)
        return pl.compile_pl_model(model, list(class_weights), protocol["learning_rate"], protocol["weight_decay"])
    import arch1_model as a1
    import e1_model as em
    import e1_train as et
    if lesion_prior is None:
        raise ValueError(f"{arm} needs E1's lesion prior")
    model = em.build_e1_model(int(seed), lesion_prior, image_size=size)          # random encoder, seeded heads
    backbone = adapted_backbone(adapted, size)
    report = a1.copy_backbone_weights(model, backbone)                           # every array, by layer name, once
    if report["copied_arrays"] != len(adapted["arrays"]):
        raise RuntimeError(f"copied {report['copied_arrays']} arrays, the adapted encoder has {len(adapted['arrays'])}")
    return et.compile_e1_model(model, list(class_weights), protocol["learning_rate"], protocol["weight_decay"])


# --------------------------------------------------------------------------- data (the verified v2 bundle)

class BundleFrames:
    """The bundle's Stage-2 RGB frames behind the `frame(image)` interface of the P-type sequence and predictor."""

    def __init__(self, bundle):
        self.bundle = bundle

    def frame(self, image_id):
        import stage34_cache_v2 as cache
        return cache.read_stage2_rgb(self.bundle._path("stage2_rgb_v2", image_id),
                                     expected_file_sha256=self.bundle._sha("stage2_rgb_v2", image_id))


def entries(bundle, grade_of=None):
    """(training entries, validation entries) of the authoritative APTOS split (arch1_train._grades verifies the
    bundle's split hash against it)."""
    train = list(zip(bundle.train_ids, at._grades(bundle, bundle.train_ids, grade_of)))
    val = list(zip(bundle.val_ids, at._grades(bundle, bundle.val_ids, grade_of)))
    if set(i for i, _ in train) & set(i for i, _ in val):
        raise RuntimeError("an APTOS image is in both parts of the split")
    return train, val


# --------------------------------------------------------------------------- conditions

def arm_complete(experiments_root, arm, bundle, adapted):
    return all(os.path.exists(posixpath.join(run_dir_for(experiments_root, arm, adapted["sha256"], seed), "result.json"))
               for seed in SEEDS)


def write_e2_ep_eligibility(path, probe_result):
    """Records the E1-EP-vs-P-EP probe criterion for the E2-EP decision. `probe_result`: the analysis output of
    the DDR ground-truth probe on the EP arms ({'variants': {'five_epoch': {'mean': {'e1_ep_minus_p_ep': ...}}}})."""
    primary = probe_result["variants"][E2_EP_CONDITION["primary_variant"]]["mean"][E2_EP_CONDITION["contrast"]]
    met = bool(primary["positive_seeds"] == len(SEEDS) and primary["ci"][0] > 0)
    record = dict(E2_EP_CONDITION, per_seed=primary["per_seed"], mean=primary["mean"], ci=primary["ci"],
                  positive_seeds=primary["positive_seeds"], criterion_met=met)
    if os.path.exists(path):
        with open(path) as fh:
            if json.load(fh) != json.loads(json.dumps(record)):
                raise RuntimeError(f"{path} already records another criterion result")
        return record
    with open(path + ".tmp", "w") as fh:
        json.dump(record, fh, indent=1)
    os.replace(path + ".tmp", path)
    return record


def require_e2_ep_eligibility(path):
    """E2-EP may run only with a recorded probe result of E1-EP against P-EP whose criterion was met."""
    if not path or not os.path.exists(path):
        raise RuntimeError("E2-EP is conditional: no recorded E1-EP probe criterion result")
    with open(path) as fh:
        record = json.load(fh)
    if any(record.get(k) != v for k, v in E2_EP_CONDITION.items()):
        raise RuntimeError("the recorded result is not the pre-declared E1-EP vs P-EP criterion")
    if record.get("criterion_met") is not True or record.get("positive_seeds") != len(SEEDS) or not record["ci"][0] > 0:
        raise RuntimeError("E2-EP is not eligible: the E1-EP probe criterion was not met")
    return record


# --------------------------------------------------------------------------- identity

def run_dir_for(experiments_root, arm, adapted_sha256, seed):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")
    return posixpath.join(experiments_root, EXPERIMENT, f"{arm}_{adapted_sha256[:12]}_seed{int(seed)}")


def run_mapping(arm, bundle, seed, adapted, class_weights, lesion_prior=None, derangement=None, protocol=PROTOCOL,
                repo_dir=None):
    """Everything that identifies a run (written to config.json and hashed into the checkpoint config)."""
    import pl_convnext as pl
    mapping = {"experiment": EXPERIMENT, "arm": arm, "mirrors": ARMS[arm]["mirrors"], "model_type": ARMS[arm]["model"],
               "seed": int(seed), "encoder_initialisation": "pinned EyePACS-adapted encoder (not ImageNet)",
               "adapted_encoder_sha256": adapted["sha256"], "adapted_encoder_digest": adapted["digest"],
               "imagenet_weights_sha256": pl.WEIGHTS_SHA256, "corn_head": "fresh, pl_convnext.corn_head_initial_weights(seed)",
               "split_sha256": bundle.bundle["split_sha256"], "population_sha256": bundle.bundle.get("population_sha256"),
               "bundle_id": bundle.bundle["bundle_id"], "bundle_fingerprint": bundle.fingerprint,
               "class_weights": [float(w) for w in class_weights], "optimizer": "AdamW (no decay on 1-D)",
               "checkpoint_selection": SELECTION, **protocol, "mixed_precision": True, "ema": at.EMA,
               "git_commit": at.git_commit(repo_dir)}
    if ARMS[arm]["model"] == "E1":
        import e1_data as ed
        import e1_model as em
        import e1_train as et
        mapping.update({"stage4_sha256": bundle.stage4_sha256, "stage4_generation": bundle.stage4_generation,
                        "lesion_classes": list(em.LESION_CLASSES),
                        "lesion_target": {"source_channels": [f"{c}:{ed.POOLING}" for c in em.LESION_CLASSES],
                                          "grid": ed.GRID, "pooling": "block maximum", "soft": True},
                        "lesion_loss_weight": et.LAMBDA, "lesion_prior": [float(p) for p in lesion_prior],
                        "added_parameters": em.ADDED_PARAMETERS,
                        "training_targets": "aligned" if arm == "e1_ep" else "fixed derangement (e2_control)"})
        if arm == "e2_ep":
            mapping["derangement"] = derangement
    return mapping


# --------------------------------------------------------------------------- one run

def train_seed(arm, run_dir, bundle, seed, adapted, class_weights, *, repo_dir, staging_dir, lesion_prior=None,
               partner=None, eligibility=None, protocol=PROTOCOL, log=print, max_epochs=None, grade_of=None,
               mixed_precision=True, image_size=None):
    """One resumable APTOS run of an EP arm. Checks before anything is built: the seed, the arm, the APTOS split
    (bundle hash against the authoritative split), and for E2-EP the eligibility record and the derangement."""
    if int(seed) not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}, got {seed!r}")
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
    train_entries, val_entries = entries(bundle, grade_of)
    derangement = None
    if arm == "e2_ep":
        import e2_control as e2
        require_e2_ep_eligibility(eligibility)
        derangement = e2.verify_derangement(partner, bundle.train_ids, bundle.val_ids)
    batch = protocol["batch_size"]
    if ARMS[arm]["model"] == "P":
        frames = BundleFrames(bundle)
        train_sequence = lambda epoch: eat.make_epoch_sequence(frames, train_entries, epoch, seed, batch, True)   # noqa: E731
        val_sequence = lambda: eat.make_epoch_sequence(frames, val_entries, 0, seed, batch, False)               # noqa: E731
        wrap = None
    else:
        import e1_data as ed
        import e1_train as et
        if arm == "e1_ep":
            train_sequence = lambda epoch: ed.make_epoch_sequence(bundle, train_entries, epoch, seed, batch, augment=True)   # noqa: E731
        else:
            train_sequence = lambda epoch: e2.make_epoch_sequence(bundle, train_entries, epoch, seed, batch, partner)       # noqa: E731
        val_sequence = lambda: ed.make_epoch_sequence(bundle, val_entries, 0, seed, batch, augment=False)                   # noqa: E731
        wrap = et.with_aliases
    return eat.training_loop(
        run_dir, run_mapping(arm, bundle, seed, adapted, class_weights, lesion_prior, derangement, protocol, repo_dir),
        lambda: build_model(arm, seed, adapted, class_weights, lesion_prior, protocol, mixed_precision, image_size),
        train_sequence, val_sequence, experiment_id=f"{EXPERIMENT}/{arm}/seed_{seed}",
        dataset_version=f"bundle:{bundle.fingerprint}", protocol=protocol, repo_dir=repo_dir, staging_dir=staging_dir,
        log=log, max_epochs=max_epochs, mixed_precision=mixed_precision, wrap_callbacks=wrap,
        initialisation=lambda model: assert_initialisation(arm, model, seed, adapted, lesion_prior))


def evaluate_run(arm, run_dir, bundle, seed, adapted, class_weights, *, lesion_prior=None, protocol=PROTOCOL,
                 grade_of=None, mixed_precision=True, image_size=None):
    """BEST and LAST on the APTOS validation part, in the form of the mirrored experiment: P's per-sample schema
    and metrics; for E1-EP / E2-EP also the lesion head's agreement with the teacher (e1_train.evaluate_run)."""
    if ARMS[arm]["model"] == "P":
        _, val_entries = entries(bundle, grade_of)
        return eat.evaluate_checkpoints(
            run_dir, lambda: build_model(arm, seed, adapted, class_weights, None, protocol, mixed_precision, image_size),
            BundleFrames(bundle), val_entries, "APTOS validation (also the selection set, as for P)")
    import e1_train as et
    return et.evaluate_run(run_dir, bundle, seed, lesion_prior, None, class_weights, protocol, grade_of, mixed_precision)


def seed_state(run_dir):
    return eat.state(run_dir)


def run_arm(arm, experiments_root, bundle, adapted, class_weights, *, repo_dir, staging_root, lesion_prior=None,
            partner=None, eligibility=None, confirmation=None, log=print, train_fn=None, evaluate_fn=None, grade_of=None):
    """Seeds 42 -> 123 -> 2026 of one arm, each from the same pinned adapted encoder with its own fresh head:
    train (or resume), pin BEST, evaluate, write result.json. Refuses to start an arm whose prerequisites are
    not complete (E1-EP after P-EP; E2-EP after both, with the eligibility record AND the explicit confirmation
    phrase -- eligibility alone starts nothing). A completed seed is kept."""
    import gc

    import multiseed_runs as msr
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {tuple(ARMS)}, got {arm!r}")
    missing = [a for a in ARMS[arm]["requires"] if not arm_complete(experiments_root, a, bundle, adapted)]
    if missing:
        raise RuntimeError(f"{arm} cannot start: {missing} not complete for seeds {SEEDS}")
    if arm == "e2_ep":
        require_e2_ep_eligibility(eligibility)
        if confirmation != E2_EP_CONFIRMATION:
            raise RuntimeError(f"E2-EP is eligible but not confirmed: pass confirmation={E2_EP_CONFIRMATION!r}")
    results = {}
    for n, seed in enumerate(SEEDS, start=1):
        run_dir = run_dir_for(experiments_root, arm, adapted["sha256"], seed)
        log(f"=== {arm} [{n}/{len(SEEDS)}] seed {seed}: {seed_state(run_dir)} | {run_dir}")
        if seed_state(run_dir) == "complete":
            eat.read_frozen(run_dir)
            with open(posixpath.join(run_dir, "result.json")) as fh:
                results[seed] = json.load(fh)
            continue
        (train_fn or train_seed)(arm, run_dir, bundle, seed, adapted, class_weights, repo_dir=repo_dir,
                                 staging_dir=posixpath.join(staging_root, arm, f"seed_{seed}"), lesion_prior=lesion_prior,
                                 partner=partner, eligibility=eligibility, log=log, grade_of=grade_of)
        if msr.read_stop_decision(run_dir) is None:
            raise RuntimeError(f"{run_dir}: seed {seed} returned without a stop decision -- training is not finished")
        with open(posixpath.join(run_dir, "initialization.json")) as fh:
            started = json.load(fh)
        if not started.get("encoder_is_adapted") or started.get("adapted_encoder_sha256") != adapted["sha256"] \
                or not started.get("fresh_corn_head"):
            raise RuntimeError(f"{run_dir}: the run did not start from the pinned adapted encoder with a fresh head")
        pinned = eat.pin_checkpoint(run_dir, repo_dir)
        evaluated = (evaluate_fn or evaluate_run)(arm, run_dir, bundle, seed, adapted, class_weights,
                                                  lesion_prior=lesion_prior, grade_of=grade_of)
        if evaluated["best"]["checkpoint"]["weights_sha256"] != pinned["sha256"]:
            raise RuntimeError(f"{run_dir}: the evaluated BEST is not the pinned checkpoint")
        results[seed] = eat.write_result(run_dir, evaluated, pinned)
        log(f"=== {arm} [{n}/{len(SEEDS)}] seed {seed} DONE: APTOS validation QWK {evaluated['best']['metrics']['qwk']:.4f} "
            f"| pinned {pinned['sha256']}")
        gc.collect()
    return results
