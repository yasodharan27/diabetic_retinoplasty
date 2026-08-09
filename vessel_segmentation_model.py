"""
Vessel Segmentation model architecture (pipeline Stage 03).

Unlike every other trainable module in this repository, this stage does not
train its own weights: it loads a pretrained, externally-sourced checkpoint
("LWNet" -- Adrian Galdran et al., "The Little W-Net That Could: State-of-
the-Art Retinal Vessel Segmentation with Minimalistic Models",
https://arxiv.org/abs/2009.01907, MIT License, https://github.com/agaldran/lwnet)
for inference only. See SEGMENTATION_ARCHITECTURE.md Sec 1.2/2 for the full
design and Appendix A.1 for why this reverses an earlier, since-superseded
"Baseline U-Net trained within this project" design.

This module vendors only the minimum architecture needed to load and run
`models/vessel_segmentation/best_model.pth` (originally
`external/lwnet/experiments/wnet_drive/model_checkpoint.pth`) -- ported and
renamed from `external/lwnet/models/get_model.py` and
`external/lwnet/models/res_unet_adrian.py`. The checkpoint's `wnet`
architecture is the composition of two chained U-Nets (`unet1`, `unet2`);
`res_unet_adrian.py`'s separate, single-network `WNet` class is dead code in
the upstream repo (commented out in its own `get_model.py`) and is not
ported here. Every hyperparameter below is fixed to what the vendored
checkpoint actually requires -- independently verified against the
checkpoint's own tensor shapes during integration, not just its `config.cfg`
-- rather than reproducing the upstream `get_arch()` dispatcher's support
for architectures (`unet`, `big_unet`, `big_wnet`) this project does not use.

Training-only code (loss functions, optimizers, augmentation transforms) is
intentionally not ported -- this stage has no training workflow in this
project (SEGMENTATION_ARCHITECTURE.md Sec 7.0).
"""

import torch
import torch.nn as nn

# Fixed by the vendored `wnet_drive` checkpoint -- not configurable, since
# changing any of these would make the checkpoint's weights fail to load.
IN_CHANNELS = 3
NUM_CLASSES = 1
LAYERS = (8, 16, 32)
CONV_BRIDGE = True
SHORTCUT = True


def _conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class ConvBlock(nn.Module):
    """Two 3x3 conv+ReLU+BatchNorm layers, optional max-pool before and a
    1x1-conv+BatchNorm residual shortcut after. Ported from
    `res_unet_adrian.py`'s `ConvBlock` (unchanged behavior)."""

    def __init__(self, in_c, out_c, k_sz=3, shortcut=False, pool=True):
        super().__init__()
        self.shortcut = nn.Sequential(_conv1x1(in_c, out_c), nn.BatchNorm2d(out_c)) if shortcut else False
        pad = (k_sz - 1) // 2
        self.pool = nn.MaxPool2d(kernel_size=2) if pool else False

        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k_sz, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
            nn.Conv2d(out_c, out_c, kernel_size=k_sz, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(out_c),
        )

    def forward(self, x):
        if self.pool:
            x = self.pool(x)
        out = self.block(x)
        if self.shortcut:
            return out + self.shortcut(x)
        return out


class UpsampleBlock(nn.Module):
    """Transposed-conv upsampling. Ported from `res_unet_adrian.py`'s
    `UpsampleBlock`; only the `transp_conv` mode is kept -- the vendored
    checkpoint's `UNet` is always built with `up_mode='transp_conv'`
    (`get_model.py`'s default), so the alternative `up_conv` branch is
    unreachable dead code for this checkpoint and is not ported."""

    def __init__(self, in_c, out_c):
        super().__init__()
        # Wrapped in a 1-element nn.Sequential to match the vendored
        # checkpoint's state_dict key names exactly (`up_layer.block.0.weight`,
        # not `up_layer.block.weight`) -- the upstream code builds `block` as
        # a list and does `nn.Sequential(*block)` even for this single-layer
        # case; unwrapping it here would silently break checkpoint loading.
        self.block = nn.Sequential(nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2))

    def forward(self, x):
        return self.block(x)


class ConvBridgeBlock(nn.Module):
    """One 3x3 conv+ReLU+BatchNorm applied to a skip connection before
    concatenation. Ported from `res_unet_adrian.py`'s `ConvBridgeBlock`."""

    def __init__(self, channels, k_sz=3):
        super().__init__()
        pad = (k_sz - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=k_sz, padding=pad),
            nn.ReLU(),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    """One decoder step: upsample, optionally refine the skip connection
    through a `ConvBridgeBlock`, concatenate, then a `ConvBlock`. Ported
    from `res_unet_adrian.py`'s `UpConvBlock`."""

    def __init__(self, in_c, out_c, k_sz=3, conv_bridge=False, shortcut=False):
        super().__init__()
        self.conv_bridge = conv_bridge
        self.up_layer = UpsampleBlock(in_c, out_c)
        self.conv_layer = ConvBlock(2 * out_c, out_c, k_sz=k_sz, shortcut=shortcut, pool=False)
        if self.conv_bridge:
            self.conv_bridge_layer = ConvBridgeBlock(out_c, k_sz=k_sz)

    def forward(self, x, skip):
        up = self.up_layer(x)
        if self.conv_bridge:
            out = torch.cat([up, self.conv_bridge_layer(skip)], dim=1)
        else:
            out = torch.cat([up, skip], dim=1)
        return self.conv_layer(out)


class UNet(nn.Module):
    """Encoder/decoder U-Net with residual shortcuts and conv-bridge skip
    connections. Ported from `res_unet_adrian.py`'s `UNet` (unchanged
    behavior, including its `kaiming_normal_` / BatchNorm init, which only
    matters for training from scratch -- irrelevant here since weights are
    always loaded from the vendored checkpoint, never randomly initialized
    and trained)."""

    def __init__(self, in_c, n_classes, layers, k_sz=3, conv_bridge=True, shortcut=True):
        super().__init__()
        self.n_classes = n_classes
        self.first = ConvBlock(in_c=in_c, out_c=layers[0], k_sz=k_sz, shortcut=shortcut, pool=False)

        self.down_path = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.down_path.append(
                ConvBlock(in_c=layers[i], out_c=layers[i + 1], k_sz=k_sz, shortcut=shortcut, pool=True)
            )

        self.up_path = nn.ModuleList()
        reversed_layers = list(reversed(layers))
        for i in range(len(layers) - 1):
            self.up_path.append(
                UpConvBlock(
                    in_c=reversed_layers[i], out_c=reversed_layers[i + 1],
                    k_sz=k_sz, conv_bridge=conv_bridge, shortcut=shortcut,
                )
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.final = nn.Conv2d(layers[0], n_classes, kernel_size=1)

    def forward(self, x):
        x = self.first(x)
        down_activations = []
        for down in self.down_path:
            down_activations.append(x)
            x = down(x)
        down_activations.reverse()
        for i, up in enumerate(self.up_path):
            x = up(x, down_activations[i])
        return self.final(x)


class WNet(nn.Module):
    """Two chained `UNet`s: `unet1` predicts an initial map (raw logits, no
    activation) from the RGB image; `unet2` refines it, taking the RGB
    image concatenated with `unet1`'s *raw logits* (not a sigmoid of them)
    as input. This is the exact composition
    `external/lwnet/models/get_model.py`'s `wnet` class defines (renamed
    here to `WNet` -- the attribute names `unet1`/`unet2`, which is what
    the checkpoint's `state_dict` keys are actually keyed on, are preserved
    unchanged so the vendored weights load without remapping).

    Deliberately does NOT apply `sigmoid` to `x1` before concatenating --
    that would match `res_unet_adrian.py`'s separate, unused `WNet` class
    (dead code in the upstream repo, commented out in its own
    `get_model.py`; not what the vendored checkpoint was trained with) and
    silently break `unet2`'s input distribution relative to what its
    weights were trained on. Confirmed against `get_model.py`'s exact
    source (`x2 = self.unet2(torch.cat([x, x1], dim=1))`) after an earlier
    draft of this file introduced exactly that sigmoid-conflation bug.

    `mode='eval'` must be set (a plain instance attribute, unrelated to
    `nn.Module.eval()`) before calling `forward()` for inference -- with the
    default `mode='train'`, `forward()` returns `(x1, x2)` instead of just
    the final prediction `x2`. `build_vessel_segmentation_model()` below
    sets this automatically."""

    def __init__(self, in_c=IN_CHANNELS, n_classes=NUM_CLASSES, layers=LAYERS,
                 conv_bridge=CONV_BRIDGE, shortcut=SHORTCUT, mode="train"):
        super().__init__()
        self.unet1 = UNet(in_c=in_c, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.unet2 = UNet(in_c=in_c + n_classes, n_classes=n_classes, layers=layers, conv_bridge=conv_bridge, shortcut=shortcut)
        self.n_classes = n_classes
        self.mode = mode

    def forward(self, x):
        x1 = self.unet1(x)
        x2 = self.unet2(torch.cat([x, x1], dim=1))
        if self.mode != "train":
            return x2
        return x1, x2


def build_vessel_segmentation_model():
    """Build the exact `WNet` configuration the vendored `wnet_drive`
    checkpoint requires (in_c=3, n_classes=1, layers=(8,16,32),
    conv_bridge=True, shortcut=True), with `mode` already set to `"eval"`
    so a freshly loaded model returns a single prediction tensor from
    `forward()`, not a `(x1, x2)` tuple. Weights are NOT loaded here -- see
    `load_state_dict_from_checkpoint()`; this function only builds the
    (randomly initialized) architecture shape."""
    model = WNet(
        in_c=IN_CHANNELS, n_classes=NUM_CLASSES, layers=LAYERS,
        conv_bridge=CONV_BRIDGE, shortcut=SHORTCUT, mode="eval",
    )
    return model


def load_state_dict_from_checkpoint(model, checkpoint_path, device="cpu"):
    """Load the vendored checkpoint's `model_state_dict` into `model`
    in-place and return it, with `model.eval()` (the real `nn.Module`
    train/eval-mode switch -- distinct from, and in addition to, the
    `model.mode` attribute already set by `build_vessel_segmentation_model()`)
    called before returning, since this stage is inference-only.

    Only `model_state_dict` is read -- the checkpoint's `optimizer_state_dict`
    and `stats` entries (present because the upstream repo's `save_model()`
    always writes them) are training-run bookkeeping this project never uses.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
