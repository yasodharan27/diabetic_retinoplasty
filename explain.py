"""Explanations of the frozen ConvNeXt-Tiny + CORN grader, beside the Grad-CAM of `gradcam.py` (research record §100).
PREPARED; tested on synthetic inputs and randomly initialised weights only.

Every method explains the SAME scalar, a function of the CORN output: by default the expected grade
E[grade] = sum_k P(grade > k) (the sum of the four cumulative probabilities), or one cumulative probability
P(grade > k) (k = 0..3; k = 1 is "referable or worse"). `score` below is the single definition all methods use.

  integrated_gradients   path integral of input gradients from a black image (the fundus background); pixel-level.
  occlusion              the score drop when a square patch is replaced by black; model-agnostic, no gradients.
  superpixel_shap        Kernel SHAP over SLIC superpixels, implemented here (no `shap` package needed) -- costly,
                         for a few figures only.
  localisation           how a map overlaps an expert lesion mask (pointing game, top-k precision, AUROC of the map
                         against the mask, and the lesion share a uniform map would score). Overlap is not proof that
                         the model detects lesions.

Uncertainty (ensemble, test-time augmentation) lives in `uncertainty_eval.py`; these are explanations only.
"""
import numpy as np


# --------------------------------------------------------------------------- the explained scalar

def score(logits, cut=None):
    """numpy: the explained scalar per image from CORN logits (N, 4)."""
    p = np.cumprod(1.0 / (1.0 + np.exp(-np.asarray(logits, np.float64))), axis=1)
    return p.sum(axis=1) if cut is None else p[:, int(cut)]


def _tf_score(logits, cut=None):
    import gradcam
    return gradcam.score_from_logits(logits, cut)


def grader_scores(model_fn, rgb_batch, cut=None):
    """Scores of a batch through `model_fn(rgb) -> logits` (numpy)."""
    return score(model_fn(np.asarray(rgb_batch, np.float32)), cut)


def graph_model_fn(grad_model, batch_size=16):
    """`model_fn` for an explainer graph (`gradcam.load_explainer`): its second output is the grader's logits."""
    def run(rgb):
        out = [np.asarray(grad_model.predict_on_batch({"rgb": np.asarray(rgb[s:s + batch_size], np.float32)})[1], np.float64)
               for s in range(0, len(rgb), batch_size)]
        return np.concatenate(out, 0)
    return run


def p_model_fn(p_model, batch_size=16):
    """`model_fn` for a P-graph grader (P's three inputs)."""
    import eyepacs_adaptation_train as eat

    def run(rgb):
        out = [np.asarray(p_model.predict_on_batch(eat.p_inputs(rgb[s:s + batch_size])), np.float64) for s in range(0, len(rgb), batch_size)]
        return np.concatenate(out, 0)
    return run


# --------------------------------------------------------------------------- Integrated Gradients

def integrated_gradients(grad_model, rgb, cut=None, steps=32, baseline=None, batch=8):
    """Integrated Gradients (Sundararajan et al., 2017) from `baseline` (default: black) with a midpoint Riemann sum
    of `steps` points. Returns (attribution (H, W, 3) for ONE image, completeness gap = |sum(attr) - (s(x) - s(x0))|,
    map (H, W) = sum over channels of |attr|, scaled to [0, 1])."""
    import tensorflow as tf
    x = np.asarray(rgb, np.float32)
    if x.ndim != 3:
        raise ValueError("one image (H, W, 3) at a time")
    x0 = np.zeros_like(x) if baseline is None else np.asarray(baseline, np.float32)
    alphas = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    total = np.zeros_like(x, dtype=np.float64)
    for s in range(0, steps, batch):
        a = alphas[s:s + batch][:, None, None, None]
        path = tf.convert_to_tensor(x0[None] + a * (x - x0)[None])
        with tf.GradientTape() as tape:
            tape.watch(path)
            _, logits = grad_model({"rgb": path}, training=False)
            value = tf.reduce_sum(_tf_score(logits, cut))
        total += np.asarray(tape.gradient(value, path), np.float64).sum(axis=0)
    attribution = (x - x0).astype(np.float64) * total / steps
    ends = np.stack([x0, x])
    _, end_logits = grad_model({"rgb": tf.convert_to_tensor(ends)}, training=False)
    s0, s1 = score(np.asarray(end_logits, np.float64), cut)
    gap = float(abs(attribution.sum() - (s1 - s0)))
    heat = np.abs(attribution).sum(axis=-1)
    return attribution.astype(np.float32), gap, (heat / heat.max() if heat.max() > 0 else heat).astype(np.float32)


# --------------------------------------------------------------------------- Occlusion

def occlusion(model_fn, rgb, cut=None, patch=64, stride=32, fill=0.0):
    """Occlusion sensitivity (Zeiler & Fergus, 2014): the score drop when a `patch` x `patch` square is set to
    `fill`, at every `stride`; overlapping drops are averaged. Positive = the region raised the score. Returns
    (map (H, W) of raw drops, map scaled to [0, 1] after clipping at zero)."""
    x = np.asarray(rgb, np.float32)
    h, w = x.shape[:2]
    base = grader_scores(model_fn, x[None], cut)[0]
    tops = list(range(0, max(h - patch, 0) + 1, stride))
    lefts = list(range(0, max(w - patch, 0) + 1, stride))
    boxes = [(t, l) for t in tops for l in lefts]
    occluded = np.repeat(x[None], len(boxes), axis=0)
    for k, (t, l) in enumerate(boxes):
        occluded[k, t:t + patch, l:l + patch] = fill
    drops = base - grader_scores(model_fn, occluded, cut)
    total, count = np.zeros((h, w)), np.zeros((h, w))
    for (t, l), d in zip(boxes, drops):
        total[t:t + patch, l:l + patch] += d
        count[t:t + patch, l:l + patch] += 1
    raw = np.where(count > 0, total / np.maximum(count, 1), 0.0)
    pos = np.clip(raw, 0, None)
    return raw.astype(np.float32), (pos / pos.max() if pos.max() > 0 else pos).astype(np.float32)


# --------------------------------------------------------------------------- Kernel SHAP over superpixels

def superpixels(rgb, n_segments=50, compactness=10.0, seed=0):
    from skimage.segmentation import slic
    return slic(np.asarray(rgb, np.float64), n_segments=n_segments, compactness=compactness, start_label=0, channel_axis=-1)


def kernel_shap(value_fn, n_features, n_samples=512, seed=0):
    """Kernel SHAP (Lundberg & Lee, 2017) with the empty and full coalitions enforced exactly (efficiency). When every
    coalition fits in `n_samples` they are all enumerated and solved by weighted least squares with the Shapley kernel,
    which gives the exact Shapley values; otherwise coalition sizes are drawn in proportion to their total kernel weight
    and each drawn coalition is paired with its complement (paired sampling, which lowers the variance), then solved by
    least squares. `value_fn(masks (S, n_features) of 0/1) -> values (S,)`. Returns (phi (n_features,), v(empty))."""
    import itertools
    from math import comb
    m = n_features
    ends = value_fn(np.stack([np.zeros(m), np.ones(m)]))
    v0, total = float(ends[0]), float(ends[1] - ends[0])
    if 2 ** m - 2 <= n_samples:
        masks = np.array([[1.0 if j in c else 0.0 for j in range(m)] for r in range(1, m) for c in itertools.combinations(range(m), r)])
        weights = np.array([(m - 1) / (comb(m, int(k)) * k * (m - k)) for k in masks.sum(axis=1)])
    else:
        rng = np.random.default_rng(seed)
        sizes = np.arange(1, m)
        p = (m - 1) / (sizes * (m - sizes))
        half = max(1, n_samples // 2)
        drawn = np.zeros((half, m))
        for i, k in enumerate(rng.choice(sizes, half, p=p / p.sum())):
            drawn[i, rng.choice(m, int(k), replace=False)] = 1.0
        masks, weights = np.concatenate([drawn, 1.0 - drawn]), np.ones(2 * half)
    y = value_fn(masks) - v0
    a = (masks[:, :-1] - masks[:, -1:]) * np.sqrt(weights)[:, None]           # the last feature eliminated through efficiency
    b = (y - masks[:, -1] * total) * np.sqrt(weights)
    phi_rest, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.append(phi_rest, total - phi_rest.sum()), v0


def superpixel_shap(model_fn, rgb, cut=None, n_segments=50, n_samples=512, fill=0.0, seed=0, batch=32):
    """Kernel SHAP of the explained score over SLIC superpixels of one frame. Returns (map (H, W) of each pixel's
    superpixel value, segments, phi, base value). Costly: n_samples forward passes per image."""
    x = np.asarray(rgb, np.float32)
    segments = superpixels(x, n_segments, seed=seed)
    n = int(segments.max()) + 1

    def value_fn(masks):
        out = []
        for s in range(0, len(masks), batch):
            images = np.repeat(x[None], len(masks[s:s + batch]), axis=0)
            for k, mask in enumerate(masks[s:s + batch]):
                images[k][~mask.astype(bool)[segments]] = fill
            out.append(grader_scores(model_fn, images, cut))
        return np.concatenate(out)
    phi, base = kernel_shap(value_fn, n, n_samples, seed)
    return phi[segments].astype(np.float32), segments, phi, base


# --------------------------------------------------------------------------- localisation against expert masks

def localisation(heat, lesion_mask, top_fraction=0.1):
    """Overlap of one map (H, W) with a binary lesion mask of the same size. Returns the pointing-game hit, the
    lesion share of the map's top `top_fraction` pixels, the AUROC of the map as a lesion score, and the lesion
    share of the image (what a uniform map would get). Describes overlap only."""
    import arch1_posthoc as ph
    heat, mask = np.asarray(heat, np.float64), np.asarray(lesion_mask, bool)
    if heat.shape != mask.shape:
        raise ValueError("map and mask must have the same shape")
    peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
    k = max(1, int(round(top_fraction * heat.size)))
    top = np.argsort(heat, axis=None, kind="stable")[-k:]
    return {"pointing_hit": bool(mask[peak]), "top_precision": float(mask.reshape(-1)[top].mean()),
            "auroc": ph._auroc(mask.reshape(-1), heat.reshape(-1)), "lesion_fraction": float(mask.mean())}


def frame_mask(mask_native, box=None, size=512):
    """An expert mask in the grader's frame: the same cut (crop arm) and the same full-frame resize as the image
    (nearest neighbour, so it stays binary)."""
    import cv2
    m = np.asarray(mask_native) > 0
    if box is not None:
        top, bottom, left, right = box
        m = m[top:bottom, left:right]
    return cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


# --------------------------------------------------------------------------- the localisation sample (DDR lesion-segmentation TRAINING split)

def localisation_sample(lesion_root, audit_dir, n, seed=20261014):
    """A fixed random sample of DDR lesion-segmentation TRAINING images with expert masks, excluding every image the
    lesion manifest records as also being in the DDR grading TEST list (those belong to the external re-test).
    Returns [(name, image path, {class: mask path})]."""
    import csv

    import ddr_probe as dp
    names = [name for name, _ in dp.read_manifest(audit_dir, lesion_root, verify_images=False)["train"]]
    with open(f"{audit_dir}/ddr_lesion_manifest.csv", newline="", encoding="utf-8") as fh:
        in_grading_test = {r["image"] for r in csv.DictReader(fh) if r["grading_split"] == "test"}
    pool = sorted(n_ for n_ in names if n_ not in in_grading_test)
    pick = sorted(np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False))
    return [(pool[k], dp.image_path(lesion_root, "train", pool[k]), {c: dp.mask_path(lesion_root, "train", pool[k], c) for c in dp.CLASSES})
            for k in pick]


def union_mask(mask_paths):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    masks = [np.asarray(Image.open(p)) > 0 for p in mask_paths.values()]
    return np.any(np.stack(masks), axis=0)


def localisation_study(items, arm, grad_models, model_fns, methods=("grad_cam", "integrated_gradients"),
                       occlusion_images=20, ig_steps=32, log=print):
    """Each method's map (the mean over the seeds' maps) against the union of the expert lesion masks, per image,
    in the grader's frame for `arm`. Occlusion runs on the first `occlusion_images` images only."""
    import cv2

    import geometry_diagnostic as gd
    import gradcam
    rows = []
    for k, (name, image_path, masks) in enumerate(items):
        direct, crop, box, native = gd.two_frames(image_path)
        rgb = direct if arm == "direct" else crop
        mask = frame_mask(union_mask(masks), None if arm == "direct" else box)
        maps = {}
        if "grad_cam" in methods:
            maps["grad_cam"] = np.mean([gradcam.grad_cam(g, rgb[None])[0][0] for g in grad_models], axis=0)
        if "integrated_gradients" in methods:
            maps["integrated_gradients"] = np.mean([integrated_gradients(g, rgb, steps=ig_steps)[2] for g in grad_models], axis=0)
        if "occlusion" in methods and k < occlusion_images:
            maps["occlusion"] = np.mean([occlusion(f, rgb)[1] for f in model_fns], axis=0)
        for method, heat in maps.items():
            rows.append({"image": name, "arm": arm, "method": method, **localisation(heat, mask)})
        if (k + 1) % 10 == 0:
            log(f"  {arm}: {k + 1}/{len(items)} images")
    return rows
