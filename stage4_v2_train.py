"""Stage-4 v2 K=4 training (research record §40, §44). Colab-ready and resumable. Running it IS training:
it is only invoked explicitly (notebook `stage4_v2_training.ipynb`, or `python stage4_v2_train.py --run`).

What is fixed by §40 (and asserted here):
  * smp.Unet + pinned ImageNet SE-ResNet-101 (stage4_v2.build_stage4_model(pretrained=True): no fallback);
  * K = 4 (MA, HE, EX, SE); partial-label loss (stage4_v2.partial_label_loss); w+ = clip(sqrt(neg/pos), 1, 20)
    from training-pixel counts; lambda = 1 for the core classes;
  * encoder LR = 0.1 x decoder LR; gradient clipping 1.0; EMA 0.999 (the EMA model is validated/exported);
  * 512 patches, class-aware with p = 0.5; dataset-balanced batches (IDRiD / TJDR halves);
  * mixed precision (torch.autocast + GradScaler) on CUDA.
Validation: per class, per dataset, per camera (never silently pooled); pooled soft Dice (the project's
`training.metrics.dice_coefficient` definition, smooth 1) and pixel AUPR, at the 1536 Stage-4 frame.
"""
import argparse
import dataclasses
import datetime
import hashlib
import io
import json
import os
import random
import subprocess

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import stage4_v2 as s4
import stage4_v2_data as data

CLASSES = v2cfg.STAGE4_V2A_CLASSES


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)


@dataclasses.dataclass
class TrainConfig:
    seed: int = 42
    iterations: int = 12000
    warmup_steps: int = 500
    batch_size: int = 8
    patch: int = v2cfg.STAGE4_TRAIN_PATCH
    p_lesion: float = 0.5
    decoder_lr: float = 3e-4
    encoder_lr_factor: float = 0.1
    weight_decay: float = 1e-4
    min_lr_factor: float = 0.01
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp: bool = True
    flip_augmentation: bool = True
    val_every: int = 1000
    checkpoint_every: int = 500
    num_workers: int = 2

    @classmethod
    def from_defaults(cls, **overrides):
        d = {k: v for k, v in v2cfg.STAGE4_TRAIN_DEFAULTS.items() if k in {f.name for f in dataclasses.fields(cls)}}
        return cls(**{**d, **overrides})

    def assert_fixed_design(self, check_patch=True):
        """§40 values that must not drift (the patch size is exempt only for the toy-model loop test)."""
        assert self.encoder_lr_factor == 0.1 and self.grad_clip == 1.0 and self.ema_decay == 0.999
        assert self.p_lesion == 0.5
        assert not check_patch or self.patch == v2cfg.STAGE4_TRAIN_PATCH
        assert self.batch_size % 2 == 0, "dataset-balanced batches need an even batch size"
        return True


# --------------------------------------------------------------------------- reproducibility

def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def environment_record():
    import torch
    import segmentation_models_pytorch as smp
    return {"torch": torch.__version__, "smp": smp.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": git_commit()}


# --------------------------------------------------------------------------- loss weights

def pixel_counts(training_cache, role="train"):
    """FOV positive / negative pixel counts per class over the training images whose dataset annotates
    the class (all four for IDRiD and TJDR)."""
    pos = np.zeros(len(CLASSES), np.float64)
    neg = np.zeros(len(CLASSES), np.float64)
    for name in training_cache.names(role):
        rgb, masks, meta = training_cache.load(name)
        fov = s4.fov_mask(rgb.astype(np.float32) / 255.0).astype(bool)
        ann = np.asarray(meta["annotated"], bool)
        for k in range(len(CLASSES)):
            if ann[k]:
                p = int((masks[k].astype(bool) & fov).sum())
                pos[k] += p
                neg[k] += int(fov.sum()) - p
    return pos, neg


def loss_weights(pos, neg, classes=CLASSES):
    return {"pos_weight": s4.positive_weights(pos, neg).tolist(),
            "lambda": s4.class_loss_weights(classes).tolist(),
            "pos_pixels": [int(x) for x in pos], "neg_pixels": [int(x) for x in neg]}


# --------------------------------------------------------------------------- optimiser / schedule

def build_optimizer(model, cfg):
    import torch
    groups = s4.parameter_groups(model, cfg.decoder_lr, cfg.encoder_lr_factor)
    out = []
    for g in groups:
        decay = [p for p in g["params"] if p.ndim > 1]
        no_decay = [p for p in g["params"] if p.ndim <= 1]
        out += [{"params": decay, "lr": g["lr"], "base_lr": g["lr"], "weight_decay": cfg.weight_decay, "name": g["name"]},
                {"params": no_decay, "lr": g["lr"], "base_lr": g["lr"], "weight_decay": 0.0, "name": g["name"] + "_no_decay"}]
    return torch.optim.AdamW(out)


def lr_factor(step, cfg):
    if step < cfg.warmup_steps:
        return (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, cfg.iterations - cfg.warmup_steps)
    return cfg.min_lr_factor + (1 - cfg.min_lr_factor) * 0.5 * (1 + np.cos(np.pi * min(1.0, t)))


def build_scheduler(optimizer, cfg):
    import torch
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, cfg))


# --------------------------------------------------------------------------- one step

def train_step(model, ema, optimizer, scheduler, scaler, batch, weights, cfg, device):
    import torch
    model.train()
    x = batch["x"].to(device, non_blocking=True)
    y = batch["y"].to(device, non_blocking=True)
    fov = batch["fov"].to(device, non_blocking=True)
    with torch.autocast(device_type=torch.device(device).type, enabled=bool(cfg.amp and torch.device(device).type == "cuda")):
        logits = model(x)
    loss, per_class = s4.partial_label_loss(logits.float(), y, fov, batch["annotated"], weights["pos_weight"], weights["lambda"])
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip))
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
    ema.update(model)
    return {"loss": float(loss.detach()), "grad_norm": grad_norm,
            "per_class": {CLASSES[k]: v for k, v in per_class.items() if v is not None}}


# --------------------------------------------------------------------------- metrics

class MetricAccumulator:
    """Streaming per-class metrics for one reporting group:
      * pooled soft Dice = (2 sum p*y + 1) / (sum p + sum y + 1) -- training.metrics.dice_coefficient
        applied per class over all pixels of all images of the group (the definition behind the
        documented Stage-4 reference values);
      * pixel AUPR (average precision) from exact-count histograms with STAGE4_AP_BINS bins."""

    def __init__(self, n_classes=len(CLASSES), bins=v2cfg.STAGE4_AP_BINS):
        self.bins = bins
        self.inter = np.zeros(n_classes)
        self.psum = np.zeros(n_classes)
        self.ysum = np.zeros(n_classes)
        self.pos_hist = np.zeros((n_classes, bins), np.int64)
        self.neg_hist = np.zeros((n_classes, bins), np.int64)
        self.images = 0

    def update(self, probs, targets):
        """probs (K, H, W) float in [0,1]; targets (K, H, W) {0,1}."""
        self.images += 1
        for k in range(probs.shape[0]):
            p = probs[k].astype(np.float64).ravel()
            y = targets[k].astype(bool).ravel()
            self.inter[k] += p[y].sum()
            self.psum[k] += p.sum()
            self.ysum[k] += y.sum()
            idx = np.minimum((p * self.bins).astype(np.int64), self.bins - 1)
            self.pos_hist[k] += np.bincount(idx[y], minlength=self.bins)
            self.neg_hist[k] += np.bincount(idx[~y], minlength=self.bins)

    @staticmethod
    def average_precision(pos_hist, neg_hist):
        """AP = sum_n (R_n - R_{n-1}) P_n over thresholds from high to low score (ties share a bin)."""
        tp = np.cumsum(pos_hist[::-1])
        fp = np.cumsum(neg_hist[::-1])
        total = tp[-1]
        if total == 0:
            return float("nan")
        precision = tp / np.maximum(tp + fp, 1)
        recall = tp / total
        return float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))

    def result(self, classes=CLASSES):
        out = {"images": self.images}
        for k, c in enumerate(classes):
            out[c] = {"dice_pooled_soft": float((2 * self.inter[k] + 1) / (self.psum[k] + self.ysum[k] + 1)),
                      "aupr": self.average_precision(self.pos_hist[k], self.neg_hist[k]),
                      "positive_pixels": int(self.ysum[k])}
        auprs = [out[c]["aupr"] for c in classes if not np.isnan(out[c]["aupr"])]
        out["mean_dice"] = float(np.mean([out[c]["dice_pooled_soft"] for c in classes]))
        out["mean_aupr"] = float(np.mean(auprs)) if auprs else float("nan")
        out["aupr_classes_defined"] = len(auprs)
        return out


def reporting_groups(meta):
    """Every group an image contributes to. Cameras are never pooled silently: TJDR is reported per camera,
    plus an explicitly labelled both-camera row."""
    if meta["dataset"] == "IDRiD":
        return ["IDRiD-val"]
    return [f"TJDR-val-{meta['camera'].replace('TJDR-', '')}", "TJDR-val-all(TRC50DX+CLARUS500)"]


SELECTION_GROUPS = ("IDRiD-val", "TJDR-val-TRC50DX", "TJDR-val-CLARUS500")


def validate(model, training_cache, device, amp=True):
    """Full-frame (1536^2) inference with the given (EMA) model on every validation image."""
    groups = {}
    for name in training_cache.names("val"):
        rgb, masks, meta = training_cache.load(name)
        probs = s4.predict_full_frame(model, rgb.astype(np.float32) / 255.0, device, amp).transpose(2, 0, 1)
        for g in reporting_groups(meta):
            groups.setdefault(g, MetricAccumulator()).update(probs, masks)
    report = {g: acc.result() for g, acc in sorted(groups.items())}
    sel = [report[g]["mean_aupr"] for g in SELECTION_GROUPS if g in report]
    report["selection_score"] = float(np.mean(sel)) if sel else float("nan")
    return report


# --------------------------------------------------------------------------- checkpoints

def state_dict_sha256(state_dict):
    import torch
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def fingerprints(training_cache):
    return {"training_cache_generation": v2cfg.STAGE4_TRAIN_CACHE_GENERATION,
            "training_cache_fingerprint": training_cache.fingerprint,
            "idrid_split_sha256": v2cfg.IDRID_V2_SPLIT_SHA256,
            "idrid_val_ids": list(v2cfg.IDRID_V2_VAL_IDS),
            "tjdr_excluded": {k: list(v) for k, v in v2cfg.TJDR_EXCLUDED.items()},
            "tjdr_source_listing_sha256": v2cfg.TJDR_SOURCE_LISTING_SHA256,
            "encoder_weights_sha256": v2cfg.STAGE4_ENCODER_SHA256,
            "encoder_weights_source": f"{v2cfg.STAGE4_ENCODER_HF_REPO}@{v2cfg.STAGE4_ENCODER_HF_REVISION}"}


def save_checkpoint(path, *, step, model, ema, optimizer, scheduler, scaler, cfg, weights, fps, history, best):
    import torch
    cache.assert_not_legacy_path(path)
    ema_state = ema.module.state_dict()
    payload = {"step": step, "model": model.state_dict(), "ema": ema_state,
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict(), "config": dataclasses.asdict(cfg), "classes": list(CLASSES),
               "loss_weights": weights, "fingerprints": fps, "history": history, "best": best,
               "model_weights_sha256": {"model": state_dict_sha256(model.state_dict()), "ema": state_dict_sha256(ema_state)},
               "rng": {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
               "environment": environment_record(), "saved_utc": datetime.datetime.utcnow().isoformat()}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return payload["model_weights_sha256"]


def load_checkpoint(path):
    import torch
    cache.assert_not_legacy_path(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt["model_weights_sha256"]["ema"] != state_dict_sha256(ckpt["ema"]):
        raise RuntimeError(f"{path}: EMA weights do not match their recorded sha")
    return ckpt


# --------------------------------------------------------------------------- the run

def run(run_dir, cache_dir, cfg=None, device="cuda", model_factory=None, max_steps=None, log=print):
    """Train (or resume) one run under `run_dir`. `model_factory` exists for tests only; the real run
    uses the pinned pretrained smp U-Net and nothing else."""
    import torch
    cfg = cfg or TrainConfig.from_defaults()
    cfg.assert_fixed_design(check_patch=model_factory is None)
    cache.assert_not_legacy_path(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    tc = data.TrainingCache(cache_dir, verify_files=True)
    fps = fingerprints(tc)
    seed_everything(cfg.seed)
    if model_factory is None:
        model, provenance = s4.build_stage4_model(CLASSES, pretrained=True)      # pinned, no fallback
    else:
        model, provenance = model_factory(), {"test_factory": True}
    model.to(device)
    ema = s4.ModelEMA(model, cfg.ema_decay)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg.amp and torch.device(device).type == "cuda"))

    weights_path = os.path.join(run_dir, "loss_weights.json")
    if os.path.exists(weights_path):
        weights = _read_json(weights_path)
    else:
        weights = loss_weights(*pixel_counts(tc))
        _write_json(weights_path, weights)
    latest = os.path.join(run_dir, "checkpoints", "latest.pt")
    step, history, best = 0, [], {"score": -1.0, "step": None}
    if os.path.exists(latest):
        ck = load_checkpoint(latest)
        if ck["fingerprints"] != fps or ck["config"] != dataclasses.asdict(cfg):
            raise RuntimeError("resume refused: data fingerprints or config differ from the checkpoint")
        model.load_state_dict(ck["model"])
        ema.module.load_state_dict(ck["ema"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        step, history, best = ck["step"], ck["history"], ck["best"]
        log(f"resumed at step {step}")
    else:
        _write_json(os.path.join(run_dir, "run_config.json"),
                    {"config": dataclasses.asdict(cfg), "fingerprints": fps, "loss_weights": weights,
                     "provenance": provenance, "environment": environment_record(),
                     "selection": v2cfg.STAGE4_TRAIN_DEFAULTS["selection_metric"]})

    ds = data.PatchDataset(tc, cfg.patch, cfg.p_lesion, cfg.flip_augmentation)
    names = {d: tc.names("train", d) for d in ("IDRiD", "TJDR")}
    end = cfg.iterations if max_steps is None else min(cfg.iterations, step + max_steps)
    loader = torch.utils.data.DataLoader(
        ds, batch_sampler=[data.balanced_batch_plan(names, cfg.batch_size, s, cfg.seed) for s in range(step, end)],
        num_workers=cfg.num_workers, collate_fn=data.collate, pin_memory=torch.device(device).type == "cuda")
    for batch in loader:
        stats = train_step(model, ema, optimizer, scheduler, scaler, batch, weights, cfg, device)
        step += 1
        if step % 50 == 0:
            log(f"step {step}: loss {stats['loss']:.4f} grad {stats['grad_norm']:.3f} "
                f"lr {optimizer.param_groups[0]['lr']:.2e}")
        if step % cfg.val_every == 0 or step == cfg.iterations:
            report = validate(ema.module, tc, device, cfg.amp)
            history.append({"step": step, "validation": report})
            _write_json(os.path.join(run_dir, "validation_history.json"), history)
            if report["selection_score"] > best["score"]:
                best = {"score": report["selection_score"], "step": step}
                save_checkpoint(os.path.join(run_dir, "checkpoints", "best_ema.pt"), step=step, model=model, ema=ema,
                                optimizer=optimizer, scheduler=scheduler, scaler=scaler, cfg=cfg, weights=weights,
                                fps=fps, history=history, best=best)
            log(f"validation @ {step}: selection {report['selection_score']:.4f} (best {best['score']:.4f} @ {best['step']})")
        if step % cfg.checkpoint_every == 0 or step == end:
            save_checkpoint(latest, step=step, model=model, ema=ema, optimizer=optimizer, scheduler=scheduler,
                            scaler=scaler, cfg=cfg, weights=weights, fps=fps, history=history, best=best)
    return {"step": step, "best": best, "run_dir": run_dir}


def export_best(run_dir, export_root=v2cfg.STAGE4_V2_MODEL_ROOT):
    """Exports the best EMA weights as the Stage-4 v2 model (stage4_v2.save_stage4_v2 -> model.pt + manifest).
    The returned SHA names the Stage-4 cache generation."""
    ck = load_checkpoint(os.path.join(run_dir, "checkpoints", "best_ema.pt"))
    model, _ = s4.build_stage4_model(CLASSES, pretrained=False)
    model.load_state_dict(ck["ema"])
    stamp = datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    return s4.save_stage4_v2(model, os.path.join(export_root, stamp), {
        "classes": list(CLASSES), "source_run": run_dir, "best": ck["best"], "config": ck["config"],
        "fingerprints": ck["fingerprints"], "loss": s4.loss_spec(), "loss_weights": ck["loss_weights"],
        "validation": ck["history"][-1] if ck["history"] else None, "idrid_test_gate": "NOT RUN"})


def main():
    ap = argparse.ArgumentParser(description="Stage-4 v2 K=4 training (explicit --run required).")
    ap.add_argument("--run", action="store_true", help="actually train (nothing happens without it)")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--cache-dir", default=data.cache_dir())
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()
    if not args.run:
        print("Refusing to train without --run.")
        return
    run(args.run_dir, args.cache_dir, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
