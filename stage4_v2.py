"""Stage 4 v2 -- multi-label pathology segmentation (PyTorch), research record §40,
`research/Grade3vs4_Architecture_Research/IMPLEMENTATION_SPEC_STAGE3_8_V2.md` §3-§7.

A NEW model; the legacy Stage-4 code, checkpoint and caches are untouched and never loaded.

  * smp 0.5.0 `Unet`, encoder SE-ResNet-101 with the pinned ImageNet weights
    (HF `smp-hub/se_resnet101.imagenet` @ a fixed revision, SHA-256 verified). The weights are
    fetched by this module, not by smp, so smp's own "try another URL" path can never run. If the
    pinned file cannot be obtained or does not verify, `Stage4WeightsUnavailableError` is raised --
    there is NO fallback encoder (user decision, 2026-10-01).
  * K sigmoid outputs, K taken from the ordered class list (v2-a: MA, HE, EX, SE).
  * Partial-label loss: masked BCE + batch-pooled soft Dice, each class only over the images whose
    dataset annotates it.
  * Geometry: the full native Stage-2 image directly resized to 1536^2 (the canonical 512 frame x3);
    train on 512^2 patches (class-aware), infer on the full 1536^2 image, pool exactly 3x3 to 512^2.
"""
import copy
import dataclasses
import types
import hashlib
import json
import os

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache


class Stage4WeightsUnavailableError(RuntimeError):
    """The pinned SE-ResNet-101 ImageNet weights could not be obtained or did not verify."""


# --------------------------------------------------------------------------- datasets (A_d)

@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """One Stage-4 training source. `annotated_classes` is A_d: the classes this dataset labels
    completely (a missing mask = a true negative). Classes outside A_d are UNANNOTATED for its
    images and never contribute to the loss. `verified` is False until the counts, splits, classes,
    resolutions and completeness have been checked on the downloaded data."""
    name: str
    annotated_classes: tuple
    verified: bool
    notes: str = ""
    metadata: types.MappingProxyType = dataclasses.field(default_factory=lambda: types.MappingProxyType({}))

    def annotated_vector(self, classes):
        return np.array([c in self.annotated_classes for c in classes], dtype=bool)


IDRID_SEG = DatasetSpec("IDRiD-seg", ("MA", "HE", "EX", "SE"), verified=True,
                        notes="54 train (10 held out), 27 test used once for the gate; a missing "
                              "mask file means absent (IDRiD annotates all four lesions).")
TJDR = DatasetSpec(
    "TJDR", ("MA", "HE", "EX", "SE"), verified=True,
    notes="Verified on download (record §42) and by tjdr_dataset.verify_tjdr (§43): palette-index "
          "masks, codes MA 3 / HE 2 / EX 1 / SE 4, all four classes annotated per image (0 = negative); "
          "only tjdr_dataset.usable_ids may be used (pinned exclusions).",
    metadata=types.MappingProxyType({
        "mask_mode": v2cfg.TJDR_MASK_MODE, "mask_values": v2cfg.TJDR_MASK_VALUES,
        "mask_codes": dict(v2cfg.TJDR_MASK_CODES), "official_counts": dict(v2cfg.TJDR_OFFICIAL_COUNTS),
        "excluded": {k: tuple(v) for k, v in v2cfg.TJDR_EXCLUDED.items()},
        "usable_counts": dict(v2cfg.TJDR_USABLE_COUNTS),
        "class_image_counts_official": v2cfg.TJDR_CLASS_IMAGE_COUNTS_OFFICIAL,
        "class_image_counts_usable": v2cfg.TJDR_CLASS_IMAGE_COUNTS_USABLE,
        "source": v2cfg.TJDR_SOURCE, "source_listing_sha256": v2cfg.TJDR_SOURCE_LISTING_SHA256,
        "drive_raw_dir": v2cfg.TJDR_DRIVE_RAW_DIR}))
DATASET_SPECS = {s.name: s for s in (IDRID_SEG, TJDR)}


def require_verified(specs):
    unverified = [s.name for s in specs if not s.verified]
    if unverified:
        raise RuntimeError(f"Stage-4 datasets not verified on download: {unverified}")
    return True


def annotated_matrix(dataset_names, classes):
    """(B, K) bool: A_{d(i)} for each image of a batch."""
    return np.stack([DATASET_SPECS[n].annotated_vector(classes) for n in dataset_names])


# --------------------------------------------------------------------------- pinned encoder weights

def _default_downloader(repo_id, filename, revision, cache_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id, filename=filename, revision=revision, cache_dir=cache_dir)


def fetch_pinned_encoder_weights(cache_dir=None, downloader=None):
    """Download (or reuse from the HF cache) the pinned file and verify size + SHA-256.
    Any failure raises `Stage4WeightsUnavailableError` carrying the exact underlying error."""
    downloader = downloader or _default_downloader
    source = (f"{v2cfg.STAGE4_ENCODER_HF_REPO}@{v2cfg.STAGE4_ENCODER_HF_REVISION}/"
              f"{v2cfg.STAGE4_ENCODER_HF_FILE}")
    try:
        path = downloader(v2cfg.STAGE4_ENCODER_HF_REPO, v2cfg.STAGE4_ENCODER_HF_FILE,
                          v2cfg.STAGE4_ENCODER_HF_REVISION, cache_dir)
    except Exception as exc:  # noqa: BLE001 -- reported verbatim, never swallowed
        raise Stage4WeightsUnavailableError(
            f"SE-ResNet-101 ImageNet weights unavailable from {source}: "
            f"{type(exc).__name__}: {exc}. No fallback encoder is permitted; STOP and report.") from exc
    size = os.path.getsize(path)
    digest = cache.sha256_file(path)
    if size != v2cfg.STAGE4_ENCODER_BYTES or digest != v2cfg.STAGE4_ENCODER_SHA256:
        raise Stage4WeightsUnavailableError(
            f"{source} at {path} does not verify: {size} bytes / sha256 {digest}, expected "
            f"{v2cfg.STAGE4_ENCODER_BYTES} / {v2cfg.STAGE4_ENCODER_SHA256}. No fallback; STOP.")
    return {"source": source, "path": path, "sha256": digest, "bytes": size,
            "original": "Cadene pretrainedmodels se_resnet101-7e38fcc6.pth (ImageNet-1k)"}


def build_stage4_model(classes, *, pretrained, weights_cache_dir=None, downloader=None):
    """smp.Unet(se_resnet101) with K = len(classes) raw-logit outputs. `pretrained=True` loads the
    pinned, verified ImageNet encoder (strict); `pretrained=False` is for tests and for loading a
    trained v2 checkpoint. Returns (model, provenance)."""
    import segmentation_models_pytorch as smp
    if smp.__version__ != v2cfg.SMP_VERSION:
        raise RuntimeError(f"segmentation_models_pytorch {smp.__version__} != pinned {v2cfg.SMP_VERSION}")
    classes = tuple(classes)
    cache.channel_names(classes)            # validates names / uniqueness
    model = smp.Unet(encoder_name=v2cfg.STAGE4_ENCODER, encoder_weights=None,
                     decoder_channels=v2cfg.STAGE4_DECODER_CHANNELS, in_channels=3,
                     classes=len(classes), activation=None)
    provenance = {"encoder": v2cfg.STAGE4_ENCODER, "classes": list(classes), "pretrained": None}
    if pretrained:
        from safetensors.torch import load_file
        weights = fetch_pinned_encoder_weights(weights_cache_dir, downloader)
        state = load_file(weights["path"], device="cpu")
        model.encoder.load_state_dict(state)            # strict; drops only last_linear.*
        provenance["pretrained"] = weights
    return model, provenance


def parameter_groups(model, learning_rate, encoder_lr_factor=0.1):
    encoder = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder}
    rest = [p for p in model.parameters() if id(p) not in encoder_ids]
    return [{"params": encoder, "lr": learning_rate * encoder_lr_factor, "name": "encoder"},
            {"params": rest, "lr": learning_rate, "name": "decoder_head"}]


# --------------------------------------------------------------------------- loss

def positive_weights(pos_pixels, neg_pixels, low=1.0, high=20.0):
    """w+_k = clip(sqrt(neg_k / pos_k), 1, 20) from training-pixel counts (FOV only)."""
    pos = np.asarray(pos_pixels, dtype=np.float64)
    neg = np.asarray(neg_pixels, dtype=np.float64)
    if np.any(pos <= 0):
        raise ValueError(f"every class needs positive training pixels, got {pos.tolist()}")
    return np.clip(np.sqrt(neg / pos), low, high).astype(np.float32)


def class_loss_weights(classes):
    """lambda_k = 1 for the four core classes, 0.5 for any later-admitted sparse class."""
    return np.array([1.0 if c in v2cfg.STAGE4_V2A_CLASSES else 0.5 for c in classes],
                    dtype=np.float32)


def partial_label_loss(logits, targets, fov, annotated, pos_weight, lam, eps=1.0):
    """Spec §6, exactly. logits/targets (B, K, H, W); fov (B, 1, H, W) in {0, 1};
    annotated (B, K) bool = A_{d(i)}; pos_weight, lam (K,).

      BCE_ik = sum_p f [-w+_k y log s(z) - (1-y) log(1-s(z))] / sum_p f
      D_k    = 1 - (2 sum_{i in B_k,p} f s y + eps) / (sum_{i in B_k,p} f (s + y) + eps)
      L      = sum_k lam_k [mean_{i in B_k} BCE_ik + D_k] / #{k : |B_k| > 0}

    Images that do not annotate class k are EXCLUDED by index selection, so they contribute neither
    negatives nor gradient for k. Returns (loss, per-class dict)."""
    import torch
    import torch.nn.functional as F

    annotated = torch.as_tensor(annotated, dtype=torch.bool, device=logits.device)
    pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32, device=logits.device)
    lam = torch.as_tensor(lam, dtype=torch.float32, device=logits.device)
    b, k = logits.shape[:2]
    if targets.shape != logits.shape or annotated.shape != (b, k) or fov.shape != (b, 1, *logits.shape[2:]):
        raise ValueError(f"shape mismatch: logits {tuple(logits.shape)}, targets "
                         f"{tuple(targets.shape)}, fov {tuple(fov.shape)}, annotated {tuple(annotated.shape)}")
    total = logits.sum() * 0.0
    active = 0
    per_class = {}
    for c in range(k):
        idx = torch.nonzero(annotated[:, c], as_tuple=False).flatten()
        if idx.numel() == 0:
            per_class[c] = None
            continue
        z = logits[idx, c].float()
        y = targets[idx, c].float()
        f = fov[idx, 0].float()
        area = f.sum(dim=(1, 2)).clamp_min(1.0)
        bce_map = -pos_weight[c] * y * F.logsigmoid(z) - (1.0 - y) * F.logsigmoid(-z)
        bce = ((f * bce_map).sum(dim=(1, 2)) / area).mean()
        s = torch.sigmoid(z)
        dice = 1.0 - (2.0 * (f * s * y).sum() + eps) / ((f * (s + y)).sum() + eps)
        total = total + lam[c] * (bce + dice)
        active += 1
        per_class[c] = {"bce": float(bce.detach()), "dice_loss": float(dice.detach()),
                        "images": int(idx.numel())}
    if active:
        total = total / active
    return total, per_class


# --------------------------------------------------------------------------- geometry

def resize_full_frame(rgb_native, size=v2cfg.STAGE4_INFERENCE_SIZE):
    """Full native Stage-2 image -> direct (squashing) resize to size^2, float32 [0, 1]. The same
    operation as `joint_training_dataset._resize_rgb_01` (the canonical 512 frame), at 3x."""
    from skimage.transform import resize as sk_resize
    rgb = np.asarray(rgb_native, dtype=np.float32)
    if np.asarray(rgb_native).dtype == np.uint8 or rgb.max() > 1.0:
        rgb = rgb / 255.0
    out = sk_resize(rgb, (size, size, rgb.shape[-1]), order=1, mode="reflect", anti_aliasing=True,
                    preserve_range=True)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def resize_mask_full_frame(mask_native, size=v2cfg.STAGE4_INFERENCE_SIZE, threshold=0.5):
    """Binary native mask -> the same direct resize (area-weighted), re-binarised at `threshold`.
    Returns uint8 {0, 1}."""
    from skimage.transform import resize as sk_resize
    m = np.asarray(mask_native)
    if m.ndim != 2:
        raise ValueError(f"resize_mask_full_frame takes ONE 2-D class mask, got shape {m.shape}")
    if m.dtype != bool:
        values = np.unique(m)
        if not (set(values.tolist()) <= {0, 1} or set(values.tolist()) <= {0, 255}):
            raise ValueError(f"resize_mask_full_frame takes a binary mask ({{0,1}} or {{0,255}}), got values "
                             f"{values[:8].tolist()} -- an indexed multi-class mask (e.g. TJDR 0-4) must be "
                             "split per class first (tjdr_dataset.split_index_mask); `> 0` would merge classes")
    m = (m > 0).astype(np.float32)
    out = sk_resize(m, (size, size), order=1, mode="reflect", anti_aliasing=True, preserve_range=True)
    return (out >= threshold).astype(np.uint8)


def fov_mask(rgb01, threshold=10.0 / 255.0):
    """Field of view: pixels whose brightest channel exceeds the black-border level."""
    return (np.asarray(rgb01, dtype=np.float32).max(axis=-1) > threshold).astype(np.uint8)


def sample_patch_origin(rng, masks, annotated, patch=v2cfg.STAGE4_TRAIN_PATCH, p_lesion=0.5):
    """Class-aware patch sampling. `masks` (K, H, W) {0, 1}; `annotated` (K,) bool. With probability
    `p_lesion` the patch is centred (clamped to the image) on a random positive pixel of a random
    annotated class that has positives; otherwise it is uniform. Returns (y0, x0, class_or_None)."""
    k, h, w = masks.shape
    if h < patch or w < patch:
        raise ValueError(f"image {h}x{w} smaller than the patch {patch}")
    candidates = [c for c in range(k) if annotated[c] and masks[c].any()]
    if candidates and rng.random() < p_lesion:
        c = int(rng.choice(candidates))
        ys, xs = np.nonzero(masks[c])
        j = int(rng.integers(len(ys)))
        y0 = int(np.clip(ys[j] - patch // 2, 0, h - patch))
        x0 = int(np.clip(xs[j] - patch // 2, 0, w - patch))
        return y0, x0, c
    return int(rng.integers(0, h - patch + 1)), int(rng.integers(0, w - patch + 1)), None


def to_model_input(rgb01):
    """(H, W, 3) [0, 1] -> (1, 3, H, W) float32 tensor, ImageNet-normalised (the encoder's
    pretraining convention: input range [0, 1], ImageNet mean/std)."""
    import torch
    mean = np.asarray(v2cfg.IMAGENET_MEAN, np.float32)
    std = np.asarray(v2cfg.IMAGENET_STD, np.float32)
    x = (np.asarray(rgb01, np.float32) - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


def predict_full_frame(model, rgb01, device="cpu", amp=False):
    """Full-image inference (no tiling): (H, W, 3) [0, 1] -> (H, W, K) float32 probabilities."""
    import torch
    model.eval()
    x = to_model_input(rgb01).to(device)
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=bool(amp)):
        probs = torch.sigmoid(model(x).float())
    return probs[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)


def pathology_cache_maps(model, rgb_native, device="cpu", amp=False,
                         size=v2cfg.STAGE4_INFERENCE_SIZE, cache_size=v2cfg.CACHE_SIZE):
    """Native Stage-2 image -> uint8 (512, 512, 2K) cache maps ([c:mean, c:max])."""
    if size % cache_size:
        raise ValueError(f"inference size {size} is not a multiple of the cache size {cache_size}")
    probs = predict_full_frame(model, resize_full_frame(rgb_native, size), device, amp)
    return cache.pack_pathology_maps(probs, size // cache_size)


# --------------------------------------------------------------------------- EMA

class ModelEMA:
    """Exponential moving average of all parameters (decay 0.999); buffers (BatchNorm running
    statistics) are copied from the live model. The EMA model is the exported Stage-4 model."""

    def __init__(self, model, decay=0.999):
        import torch
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self._torch = torch

    def update(self, model):
        with self._torch.no_grad():
            for ema_p, p in zip(self.module.parameters(), model.parameters()):
                ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            for ema_b, b in zip(self.module.buffers(), model.buffers()):
                ema_b.copy_(b)


# --------------------------------------------------------------------------- save / load

def save_stage4_v2(model, directory, manifest_extra):
    """Writes `model.pt` (state_dict) + `stage4_model_manifest.json` under a v2 directory. Returns
    the model SHA-256 (which names the cache generation)."""
    import torch
    cache.assert_not_legacy_path(directory)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "model.pt")
    tmp = path + ".part"
    torch.save(model.state_dict(), tmp)
    os.replace(tmp, path)
    sha = cache.sha256_file(path)
    cache.assert_not_deny_listed(sha)
    manifest = dict(manifest_extra, sha256=sha, encoder=v2cfg.STAGE4_ENCODER,
                    smp_version=v2cfg.SMP_VERSION,
                    geometry=f"full-frame direct resize -> {v2cfg.STAGE4_INFERENCE_SIZE}^2",
                    deny_list=sorted(cache.DENY_LISTED_SHA256))
    cache.write_manifest(os.path.join(directory, "stage4_model_manifest.json"), manifest)
    return sha


def load_stage4_v2(path, *, expected_sha256, classes):
    """Loads a trained v2 checkpoint. `expected_sha256` is REQUIRED and must match the file; the
    deny-listed legacy model, legacy locations and non-`.pt` files are refused."""
    import torch
    cache.assert_not_legacy_path(path)
    cache.assert_not_deny_listed(expected_sha256, "requested Stage-4 model")
    if not str(path).endswith(".pt"):
        raise cache.LegacyArtifactError(f"{path}: a v2 Stage-4 model is a `.pt` state_dict")
    digest = cache.sha256_file(path)
    cache.assert_not_deny_listed(digest, str(path))
    if digest != expected_sha256:
        raise cache.ManifestMismatchError(f"{path}: sha256 {digest} != expected {expected_sha256}")
    model, _ = build_stage4_model(classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def loss_spec():
    """Recorded verbatim in the model manifest."""
    return {"bce": "FOV-masked, per image, w+ = clip(sqrt(neg/pos), 1, 20)",
            "dice": "soft, batch-pooled over images annotating the class, eps = 1",
            "total": "sum_k lambda_k (mean BCE + Dice) / #active classes; unannotated excluded",
            "lambda": "1 core (MA, HE, EX, SE), 0.5 sparse", "grad_clip": 1.0, "ema_decay": 0.999,
            "encoder_lr_factor": 0.1}


def spec_digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()
