"""One-time IDRiD test gate for a trained Stage-4 v2 model (research record §40 step 3, §44).

The ONLY module that reads the 27 IDRiD lesion test images. It runs only when called explicitly with
`confirm=pipeline_v2_config.STAGE4_GATE_CONFIRM_TOKEN`, and only once per exported model: a lock file
records the first run and every later call is refused. Nothing here feeds training or model
selection (the training cache holds no test images; stage4_v2_data refuses test ids).

Pass criterion (pre-registered):
  * per-class Dice beats the documented old Stage-4 values under the OLD protocol, i.e. dataset-pooled
    soft Dice (training.metrics.dice_coefficient, smooth 1) at 512^2, with targets resized by the old
    `> 0` rule (lesion_segmentation_dataset._resize_target) and predictions = the exact 3x3 block mean
    of the 1536 probabilities (the cache's mean channel);
    reference MA 0.0165, HE 0.1273, EX 0.3574, SE 0.0244;
  * and mean pixel AUPR over the 4 classes (1536 frame, current mask rule) >= 0.55.
"""
import datetime
import json
import os

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import stage4_v2 as s4
import stage4_v2_data as data
import stage4_v2_train as train

CLASSES = v2cfg.STAGE4_V2A_CLASSES


class GateRefused(RuntimeError):
    pass


def old_protocol_target(masks_native, size=v2cfg.CACHE_SIZE):
    """lesion_segmentation_dataset._resize_target, reproduced exactly: order-1 anti-aliased resize with
    mode='constant', re-binarised at > 0."""
    from skimage.transform import resize as sk_resize
    t = np.moveaxis(masks_native.astype(np.float32), 0, -1)
    r = sk_resize(t, (size, size, t.shape[-1]), order=1, mode="constant", cval=0.0,
                  anti_aliasing=True, preserve_range=True)
    return np.moveaxis((r > 0).astype(np.float32), -1, 0)


def decide(result):
    dice = {c: result["old_protocol_512"][c]["dice_pooled_soft"] for c in CLASSES}
    beats = {c: dice[c] > v2cfg.STAGE4_GATE_REFERENCE_DICE[c] for c in CLASSES}
    mean_aupr = result["frame_1536"]["mean_aupr"]
    return {"dice_beats_reference": beats, "all_classes_beat": all(beats.values()),
            "mean_aupr": mean_aupr, "mean_aupr_ok": bool(mean_aupr >= v2cfg.STAGE4_GATE_MIN_MEAN_AUPR),
            "PASS": bool(all(beats.values()) and mean_aupr >= v2cfg.STAGE4_GATE_MIN_MEAN_AUPR),
            "reference_dice": dict(v2cfg.STAGE4_GATE_REFERENCE_DICE),
            "min_mean_aupr": v2cfg.STAGE4_GATE_MIN_MEAN_AUPR}


def _lock_path(model_sha256, lock_root=None):
    """One lock per MODEL SHA (not per export folder): re-exporting the same weights cannot re-open the gate."""
    root = lock_root or os.path.join(v2cfg.STAGE4_V2_MODEL_ROOT, "idrid_test_gate_locks")
    return os.path.join(root, f"{model_sha256}.json")


def run_gate(model_path, *, expected_sha256, confirm, device="cuda", raw_dir=None, processed_dir=None,
             report_dir=None, lock_root=None):
    """The one-time gate. Writes its report (and the lock) next to the exported model, and a copy into
    `report_dir` (the training run's gate/ folder) when given."""
    if confirm != v2cfg.STAGE4_GATE_CONFIRM_TOKEN:
        raise GateRefused("The IDRiD test gate runs only with the explicit confirmation token.")
    lock = _lock_path(expected_sha256, lock_root)
    if os.path.exists(lock):
        raise GateRefused(f"The one-time IDRiD test gate was already run for this model ({lock}).")
    cache.assert_not_legacy_path(lock)
    model = s4.load_stage4_v2(model_path, expected_sha256=expected_sha256, classes=CLASSES).to(device)
    os.makedirs(os.path.dirname(lock), exist_ok=True)
    with open(lock, "w") as fh:          # written BEFORE evaluation: a crashed run still counts as used
        json.dump({"model": model_path, "sha256": expected_sha256,
                   "started_utc": datetime.datetime.utcnow().isoformat()}, fh)
    proc = processed_dir or data.idrid_processed_dir()
    old, frame = train.MetricAccumulator(), train.MetricAccumulator()
    for image_id in v2cfg.IDRID_SEG_TEST_IDS:
        rgb = data.read_rgb(os.path.join(proc, "1. Original Images", "b. Testing Set", f"{image_id}.jpg"))
        masks = data.read_idrid_masks(image_id, "b. Testing Set", raw_dir)
        probs = s4.predict_full_frame(model, s4.resize_full_frame(rgb), device, amp=True).transpose(2, 0, 1)
        mean512 = np.stack([cache.block_pool_mean_max(p[..., None])[0][..., 0] for p in probs])
        old.update(mean512, old_protocol_target(masks))
        frame.update(probs, np.stack([s4.resize_mask_full_frame(m) for m in masks]))
    result = {"old_protocol_512": old.result(), "frame_1536": frame.result(), "images": len(v2cfg.IDRID_SEG_TEST_IDS),
              "model_path": model_path, "model_sha256": expected_sha256,
              "finished_utc": datetime.datetime.utcnow().isoformat()}
    result["decision"] = decide(result)
    for folder in [os.path.dirname(model_path)] + ([report_dir] if report_dir else []):
        cache.assert_not_legacy_path(folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "idrid_test_gate.json"), "w") as fh:
            json.dump(result, fh, indent=1)
    return result
