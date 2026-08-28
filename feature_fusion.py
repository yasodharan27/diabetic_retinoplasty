"""
Adaptive Cross-Attention model + inference for Feature Fusion (pipeline
Stage 07).

Implements exactly the architecture approved and documented in
`PROJECT_STRUCTURE.md` Sec 7, `IMPLEMENTATION_PLAN.md`'s Step 7, and the
Stage 07 design-resolution record produced before this module existed. No
architectural decision is made here that was not already fixed by that
review -- this module is a direct translation of the approved design into
Keras/TensorFlow, not a redesign.

Consumes Stage 05's local features `L_spatial = (32, 32, 256)` and Stage
06's global features `G = (64, 1152)` (Stage 06's real, already-flattened
output -- see the DEFAULT_GLOBAL_SHAPE comment below for why this differs
from the design's conceptual "G_spatial = (8, 8, 1152)" framing), both
unmodified, and produces a single fused embedding `E = (256,)` via one-way,
Perceiver-style
cross-attention: Global's 64 tokens query Local's 1024 tokens (`Q` from
Global, `K`/`V` from Local), never the reverse and never bidirectional
(RACAF_ARCHITECTURE.md Sec 5 consumes exactly one such vector; a second,
Local-queries-Global stream has nowhere in that formula to go). This is not
independently trained -- like `LocalFeatureExtractionStage`/
`GlobalFeatureExtractionStage`, `train()`/`evaluate()` raise
`NotImplementedError`: Stage 07 has no standalone ground truth of its own
and only receives a learning signal by participating in a future, not-yet-
implemented joint Stage 05-08 + RACAF training graph.

RACAF boundary (see RACAF_ARCHITECTURE.md Sec 3/14): this module computes
only `E`. It never reads Stage 04's output (TTA or otherwise), never
computes disagreement/reliability/uncertainty of any kind, never gates or
weights `E` by a confidence signal, and never reads a ground-truth label.
RACAF wraps `E` and separately, independently reads the same raw `G` this
module also receives -- RACAF's own `GAP(G)` is not derived from anything
computed here, and is not implemented in this module.

Positional embeddings: "Global: 8x8 grid, dim 256" (per the approved
design) is only dimensionally addable to Global's *projected* 256-d
representation, not its raw 1152-d one -- so, for both branches, the
factorized row+column positional embedding is added after each branch's
own projection into the shared d_model=256 attention space, immediately
before the cross-attention call. This is the only placement consistent
with the approved design's own stated embedding dimension; see this
module's `build_adaptive_cross_attention` docstring for the exact order.

Serialization: every layer used here -- `keras.layers.MultiHeadAttention`,
`Dense`, `LayerNormalization`, `Dropout`, `Add`, `Reshape`,
`GlobalAveragePooling1D` -- is a built-in Keras layer with existing
`get_config()` support. The one custom layer this module introduces,
`Factorized2DPositionalEmbedding`, implements `get_config()` and is
registered via `@register_keras_serializable()`, so the whole model
round-trips through a full `model.save()`/`load_model()` `.keras` archive
-- deliberately avoiding Stage 06's weights-only fallback, which was only
required there because Stage 06 reuses pre-existing Swin classes that
predate this project's serialization conventions and were out of scope to
retrofit. No such legacy classes are reused here, so there is no reason to
repeat that compromise.
"""

import os

import numpy as np
import tensorflow as tf
from keras import Input, Model, layers
from keras.saving import register_keras_serializable

import config
from pipeline.inference import InferenceStage
from pipeline.trainable import TrainableStage

# Fixed by the approved Stage 05/06 designs -- not free parameters of this module.
#
# Local: Stage 05's real, implemented model output (`local_feature_extraction_model.py`,
# verified via `build_local_feature_extractor().output_shape`) is genuinely spatial,
# (32, 32, 256) -- flattened to 1024 tokens inside this module.
#
# Global: the approved design describes Global conceptually as an 8x8 grid of
# 1152-d tokens, "G_spatial = (8, 8, 1152)". Stage 06's real, implemented model
# (`swin_transformer.py`'s `create_dual_scale_swin_model()`, verified via
# `.output_shape`) already performs this exact flatten internally and hands off
# `(64, 1152)` directly -- there is no separate spatial (8,8,1152) tensor to
# consume. Stage 06 is frozen and not modified here, so this module's Global
# input matches Stage 06's real output shape, `(64, 1152)`, not the spatial
# form -- the same row-major 8x8 grid `create_dual_scale_swin_model()` itself
# flattens from (`layers.Reshape((grid*grid, channels))`), so the factorized
# 2D positional embedding below (grid_h=8, grid_w=8) still applies correctly to
# this already-flattened tensor's token order.
DEFAULT_LOCAL_SHAPE = (32, 32, 256)
DEFAULT_GLOBAL_SHAPE = (64, 1152)
GLOBAL_GRID = (8, 8)

# Fixed by the approved Stage 07 design.
D_MODEL = 256
NUM_HEADS = 8
HEAD_DIM = D_MODEL // NUM_HEADS  # 32
FFN_DIM = 1024
DROPOUT_RATE = 0.1
ATTENTION_DROPOUT_RATE = 0.1

DEFAULT_MODEL_PATH = os.path.join(config.FEATURE_FUSION_MODEL_DIR, "best_model.keras")


@register_keras_serializable(package="feature_fusion")
class Factorized2DPositionalEmbedding(layers.Layer):
    """Learned, factorized (row + column) 2D absolute positional embedding.

    Neither Stage 05's CNN (implicit locality) nor Stage 06's Swin
    (relative window bias) encodes position relative to the *other*
    branch's coordinate frame -- this layer supplies that, so Stage 07's
    cross-attention has an explicit signal for the fixed 4:1 grid
    correspondence between Local's 32x32 grid and Global's 8x8 grid
    (32 / 8 = 4), rather than relying on content alone to discover it.

    Holds two small embedding tables (`grid_h` row vectors, `grid_w`
    column vectors, each `dim`-wide) instead of one full `grid_h*grid_w`
    table, following established ViT/DETR-style absolute positional
    embedding practice -- a project-specific engineering adaptation, not a
    novel mechanism (see `PROJECT_STRUCTURE.md` Sec 7).

    Expects `inputs` already flattened to `(batch, grid_h*grid_w, dim)` in
    row-major order (i.e. token `i = row*grid_w + col`), matching
    `keras.layers.Reshape`'s default flattening of an `(H, W, C)` tensor.
    """

    def __init__(self, grid_h, grid_w, dim, **kwargs):
        super().__init__(**kwargs)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.dim = dim

    def build(self, input_shape):
        self.row_embed = self.add_weight(
            name="row_embed",
            shape=(self.grid_h, self.dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.col_embed = self.add_weight(
            name="col_embed",
            shape=(self.grid_w, self.dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # (grid_h, 1, dim) + (1, grid_w, dim) -> (grid_h, grid_w, dim), then
        # flattened to (grid_h*grid_w, dim) in the same row-major order the
        # upstream Reshape used, and broadcast-added over the batch axis.
        pos_grid = self.row_embed[:, None, :] + self.col_embed[None, :, :]
        pos_seq = tf.reshape(pos_grid, (self.grid_h * self.grid_w, self.dim))
        return inputs + pos_seq[None, :, :]

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"grid_h": self.grid_h, "grid_w": self.grid_w, "dim": self.dim})
        return cfg


def build_adaptive_cross_attention(
    local_shape=DEFAULT_LOCAL_SHAPE,
    global_shape=DEFAULT_GLOBAL_SHAPE,
    global_grid=GLOBAL_GRID,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    ffn_dim=FFN_DIM,
    dropout_rate=DROPOUT_RATE,
    attention_dropout_rate=ATTENTION_DROPOUT_RATE,
):
    """Builds the Adaptive Cross-Attention model: one-way, pre-LayerNorm
    Transformer-style cross-attention, Global queries Local, followed by a
    residual FFN block and global average pooling. Returns an
    **uncompiled** `keras.Model` -- see this module's docstring for why no
    loss is attached here.

    `local_shape` is spatial `(H, W, C)`, matching Stage 05's real output
    (flattened to tokens here). `global_shape` is already-flattened
    `(N, C)`, matching Stage 06's real output directly; `global_grid`
    records the row-major `(H, W)` grid those `N` tokens correspond to
    (Stage 06's own construction), used only for the positional embedding.

    Exact order (per the approved design's cross-attention block diagram):
        1. Flatten Local to a token sequence (row-major); Global is already
           a token sequence (Stage 06's own output).
        2. Pre-LayerNorm each branch's *raw* tokens.
        3. Project each branch into the shared `d_model` space -- Local's
           K/V projection (`256 -> d_model`), Global's Q projection
           (`1152 -> d_model`).
        4. Add the factorized 2D positional embedding to each branch's
           *projected* tokens (dimensionally, this is the only point they
           can be added -- Global's raw 1152-d tokens cannot receive a
           `d_model`-wide positional vector before projection).
        5. Cross-attention: `Q` = Global's projected+positioned tokens,
           `K`,`V` = Local's projected+positioned tokens.
        6. Residual connection around the cross-attention.
        7. Pre-LayerNorm -> FFN (`d_model -> ffn_dim -> d_model`, GELU,
           dropout) -> residual connection.
        8. Global average pooling over the 64 output tokens -> `E`.

    `d_model` must be divisible by `num_heads` (256 / 8 = 32 dims/head,
    matching Stage 06 Branch A's own documented per-head-dimension
    convention -- see `PROJECT_STRUCTURE.md` Sec 7).
    """
    if d_model % num_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")

    local_h, local_w, local_c = local_shape
    global_tokens_n, global_c = global_shape
    global_h, global_w = global_grid
    if global_h * global_w != global_tokens_n:
        raise ValueError(
            f"global_grid {global_grid} does not match global_shape's token count "
            f"({global_tokens_n})."
        )
    local_tokens_n = local_h * local_w

    local_input = Input(shape=local_shape, name="local_features")
    global_input = Input(shape=global_shape, name="global_features")

    local_tokens = layers.Reshape((local_tokens_n, local_c), name="local_flatten")(local_input)
    global_tokens = global_input

    local_norm = layers.LayerNormalization(name="local_pre_ln")(local_tokens)
    global_norm = layers.LayerNormalization(name="global_pre_ln")(global_tokens)

    local_proj = layers.Dense(d_model, name="local_kv_projection")(local_norm)
    global_proj = layers.Dense(d_model, name="global_q_projection")(global_norm)

    local_proj = Factorized2DPositionalEmbedding(
        local_h, local_w, d_model, name="local_pos_embed"
    )(local_proj)
    global_proj = Factorized2DPositionalEmbedding(
        global_h, global_w, d_model, name="global_pos_embed"
    )(global_proj)

    attn_out = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=attention_dropout_rate,
        name="cross_attention",
    )(query=global_proj, value=local_proj, key=local_proj)

    h = layers.Add(name="cross_attn_residual")([global_proj, attn_out])

    h_norm = layers.LayerNormalization(name="ffn_pre_ln")(h)
    ffn = layers.Dense(ffn_dim, activation="gelu", name="ffn_expand")(h_norm)
    ffn = layers.Dropout(dropout_rate, name="ffn_dropout")(ffn)
    ffn = layers.Dense(d_model, name="ffn_project")(ffn)

    out_tokens = layers.Add(name="ffn_residual")([h, ffn])

    pooled = layers.GlobalAveragePooling1D(name="global_average_pool")(out_tokens)

    # Explicit float32 output, matching every other stage's final-layer
    # convention in this project (see local_feature_extraction_model.py /
    # swin_transformer.py's identical rationale).
    outputs = layers.Activation("linear", dtype="float32", name="fused_embedding")(pooled)

    return Model(
        inputs=[local_input, global_input],
        outputs=outputs,
        name="feature_fusion_adaptive_cross_attention",
    )


# --- pipeline.TrainableStage / pipeline.InferenceStage implementation ---
#
# Not `pipeline.FeatureExtractionStage`: that class's own docstring fixes
# its `predict`/`predict_batch` contract to "a single spatial feature map
# ... never a globally pooled vector" for a *single* input -- exactly the
# opposite of what Stage 07 does (two inputs, fused into a pooled vector).
# `TrainableStage`/`InferenceStage` are the correct, narrower contracts
# this stage actually satisfies.

class AdaptiveCrossAttentionStage(TrainableStage, InferenceStage):
    """`pipeline.TrainableStage`/`pipeline.InferenceStage` implementation
    for Stage 07. `predict`/`predict_batch`/`save`/`load`/`build` are
    fully functional; `train`/`evaluate` raise `NotImplementedError` --
    Stage 07 has no standalone training or evaluation procedure of its own
    (see this module's docstring), mirroring
    `GlobalFeatureExtractionStage.train()`'s exact pattern."""

    def __init__(self, local_shape=DEFAULT_LOCAL_SHAPE, global_shape=DEFAULT_GLOBAL_SHAPE):
        self.local_shape = local_shape
        self.global_shape = global_shape
        self.model = None

    def train(self, train_data, val_data=None, **kwargs):
        raise NotImplementedError(
            "Adaptive Cross-Attention has no standalone training procedure in this project -- "
            "it has no fusion-quality ground truth of its own and is only ever trained jointly "
            "with Stages 05-06, 08, and RACAF, through the downstream CORN ordinal loss (see "
            "PROJECT_STRUCTURE.md Sec 7 and RACAF_ARCHITECTURE.md Sec 7). That joint training "
            "script does not exist yet."
        )

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError(
            "Adaptive Cross-Attention has no standalone evaluation metric in this project -- "
            "its usefulness is only observable through the joint downstream pipeline's "
            "classification metrics, not reproduced here."
        )

    def save(self, path):
        if self.model is None:
            raise RuntimeError("No model to save -- call build() (or assign self.model) first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        return path

    def load(self, path=DEFAULT_MODEL_PATH):
        self.model = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects={"Factorized2DPositionalEmbedding": Factorized2DPositionalEmbedding},
        )
        return self

    def build(self):
        """Builds a fresh, uncompiled model from `self.local_shape`/
        `self.global_shape` and assigns it to `self.model`. Separate from
        `__init__`, mirroring `LocalFeatureExtractionStage`'s/
        `GlobalFeatureExtractionStage`'s identical lazy-build pattern."""
        self.model = build_adaptive_cross_attention(
            local_shape=self.local_shape, global_shape=self.global_shape,
        )
        return self.model

    def predict(self, input_data):
        """`input_data`: a `(local, global)` pair, each a single
        (un-batched) spatial feature map -- Stage 05's `(32,32,256)`
        output and Stage 06's `(8,8,1152)` output respectively. Returns
        the fused embedding `E`, shape `(256,)`."""
        if self.model is None:
            raise RuntimeError(
                "AdaptiveCrossAttentionStage.build() or .load() must be called before predict()."
            )
        local, global_ = input_data
        local = np.asarray(local, dtype="float32")[None, ...]
        global_ = np.asarray(global_, dtype="float32")[None, ...]
        return self.model.predict([local, global_], verbose=0)[0]

    def predict_batch(self, inputs):
        """`inputs`: a list of `(local, global)` pairs. Returns one fused
        embedding per pair, in the same order."""
        if self.model is None:
            raise RuntimeError(
                "AdaptiveCrossAttentionStage.build() or .load() must be called before "
                "predict_batch()."
            )
        locals_ = np.stack([np.asarray(l, dtype="float32") for l, _ in inputs], axis=0)
        globals_ = np.stack([np.asarray(g, dtype="float32") for _, g in inputs], axis=0)
        predictions = self.model.predict([locals_, globals_], verbose=0)
        return [predictions[i] for i in range(predictions.shape[0])]
