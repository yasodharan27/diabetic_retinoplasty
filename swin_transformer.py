import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
import os
import config
from pipeline import FeatureExtractionStage

class PatchEmbed(layers.Layer):
    def __init__(self, patch_size=4, embed_dim=96, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size,
                                  padding='valid', name='proj')
        self.norm = norm_layer(epsilon=1e-5, name='norm') if norm_layer else None
    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        x = self.proj(x)
        if self.norm:
            x = self.norm(x)
        return x

def window_partition(x, window_size):
    B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
    if isinstance(window_size, int):
        window_h = window_w = window_size
    else:
        window_h, window_w = window_size
    x = tf.reshape(x, [B, H // window_h, window_h, W // window_w, window_w, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, [-1, window_h, window_w, C])
    return windows

def window_reverse(windows, window_size, H, W, C):
    if isinstance(window_size, int):
        window_h = window_w = window_size
    else:
        window_h, window_w = window_size
    B = tf.shape(windows)[0] // (H * W // window_h // window_w)
    x = tf.reshape(windows, [B, H // window_h, W // window_w, window_h, window_w, C])
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    x = tf.reshape(x, [B, H, W, C])
    return x

class WindowAttention(layers.Layer):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, dropout_rate=0.0, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name='qkv')
        self.attn_drop = layers.Dropout(dropout_rate, name='attn_drop')
        self.proj = layers.Dense(dim, name='proj')
        self.proj_drop = layers.Dropout(dropout_rate, name='proj_drop')
        if isinstance(window_size, tuple):
            window_h, window_w = window_size
        else:
            window_h = window_w = window_size
        self.relative_position_bias_table = self.add_weight(
            shape=((2 * window_h - 1) * (2 * window_w - 1), num_heads),
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name='relative_position_bias_table')
        coords_h = tf.range(window_h)
        coords_w = tf.range(window_w)
        coords = tf.stack(tf.meshgrid(coords_h, coords_w, indexing='ij'))
        coords = tf.reshape(coords, [2, -1])
        coords = tf.transpose(coords, [1, 0])
        coords_1 = tf.expand_dims(coords, 1)
        coords_2 = tf.expand_dims(coords, 0)
        relative_coords = coords_1 - coords_2
        relative_coords = tf.cast(relative_coords, tf.int32)
        relative_coords = relative_coords + tf.constant([window_h - 1, window_w - 1], dtype=tf.int32)
        relative_coords = relative_coords[..., 0] * (2 * window_w - 1) + relative_coords[..., 1]
        # Device-placement fix: this used to be a bare `tf.Variable(..., trainable=False)`. A
        # `tf.Variable` is a RESOURCE -- it has a persistent handle pinned to one device forever,
        # and TensorFlow's own placement policy for int32 tensors defaults that device to CPU
        # regardless of GPU availability (this is a well-known, TF-wide convention: most int32
        # index-like ops/kernels are CPU-only or CPU-preferred, so the placer keeps int32
        # Variables on CPU). Reading a resource variable requires the reading op to run on the
        # SAME device as the variable, so once the whole joint model executes end-to-end on a
        # GPU, the GPU-placed `tf.gather`/`tf.reshape` in `call()` below cannot read this
        # CPU-pinned resource -- exactly `InvalidArgumentError: Trying to access resource
        # relative_position_index ... located on device CPU:0 from device GPU:0`. The sibling
        # `relative_position_bias_table` above never had this problem: it is float32 (freely
        # GPU-placeable) -- dtype, not `add_weight` vs. bare `tf.Variable`, is what matters here.
        #
        # `relative_position_index` is not a learned parameter -- it is a fixed, deterministic
        # function of `window_h`/`window_w` alone, recomputed identically every time this layer
        # is constructed (including after a `create_dual_scale_swin_model()` rebuild + weights
        # reload -- Stage 06's own established "rebuild architecture, then load_weights()"
        # checkpoint strategy, see `GlobalFeatureExtractionStage.load()`). So it does not need
        # `tf.Variable`'s mutability/checkpoint semantics at all -- a plain `tf.constant` is the
        # correct representation: constants are NOT resources, carry no persistent device-pinned
        # handle, and are implicitly copied by TensorFlow's runtime to whatever device the
        # consuming op (the `tf.gather` in `call()`) actually executes on, CPU or GPU alike. This
        # also keeps `relative_position_index` out of `layer.weights`/`model.count_params()`,
        # exactly matching its behavior before this fix (a bare `tf.Variable` attribute is not
        # tracked by Keras's own weight-tracking either) -- Stage 06's total/trainable parameter
        # count is therefore unchanged by this fix. `relative_coords` above is computed with the
        # EXACT SAME TensorFlow ops as before (an eager tensor -- window_h/window_w are static
        # Python ints, so this has no dependency on any runtime/graph tensor); only how the
        # resulting values are turned into a persistent buffer changes.
        self.relative_position_index = tf.constant(
            relative_coords.numpy(), dtype=tf.int32, name='relative_position_index')

    def call(self, x, mask=None, training=None):
        B_, N, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, [B_, N, 3, self.num_heads, C // self.num_heads])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = tf.matmul(q, k, transpose_b=True)
        rel_pos_bias = tf.gather(
            self.relative_position_bias_table,
            tf.reshape(self.relative_position_index, [-1]))
        if isinstance(self.window_size, tuple):
            window_h, window_w = self.window_size
            window_area = window_h * window_w
        else:
            window_area = self.window_size * self.window_size
        rel_pos_bias = tf.reshape(
            rel_pos_bias, [window_area, window_area, -1])
        rel_pos_bias = tf.transpose(rel_pos_bias, [2, 0, 1])
        attn = attn + tf.expand_dims(rel_pos_bias, axis=0)
        if mask is not None:
            nW = tf.shape(mask)[0]
            mask = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=0)
            attn = tf.reshape(attn, [B_ // nW, nW, self.num_heads, N, N])
            attn = attn + mask
            attn = tf.reshape(attn, [-1, self.num_heads, N, N])
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, training=training)
        x = tf.matmul(attn, v)
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reshape(x, [B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x, training=training)
        return x

class MLP(layers.Layer):
    def __init__(self, hidden_features=None, out_features=None, dropout_rate=0., name=None):
        super().__init__(name=name)
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.fc1 = layers.Dense(hidden_features, name='fc1')
        self.act = layers.Activation('gelu')
        self.fc2 = layers.Dense(out_features if out_features is not None else hidden_features, name='fc2')
        self.drop = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        original_dims = tf.shape(x)
        original_shape = x.shape
        if len(original_shape) == 4:
            B, H, W, C = original_dims[0], original_dims[1], original_dims[2], original_dims[3]
            x = tf.reshape(x, [-1, C])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x, training=training)
        x = self.fc2(x)
        x = self.drop(x, training=training)
        if len(original_shape) == 4:
            x = tf.reshape(x, [B, H, W, -1])
        return x

class SwinTransformerBlock(layers.Layer):
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, dropout_rate=0.,
                 attention_dropout_rate=0., trainable=True, dtype=None, name=None):
        super().__init__(name=name, trainable=trainable, dtype=dtype)  # Pass trainable and dtype to parent class
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if isinstance(window_size, int):
            self.window_size_h = self.window_size_w = window_size
        else:
            self.window_size_h, self.window_size_w = window_size

        self.norm1 = layers.LayerNormalization(epsilon=1e-5, name='norm1')
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, dropout_rate=attention_dropout_rate, name='attn')
        self.norm2 = layers.LayerNormalization(epsilon=1e-5, name='norm2')
        self.mlp = MLP(
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            dropout_rate=dropout_rate,
            name='mlp'
        )
        if isinstance(window_size, int):
            min_size = window_size
        else:
            min_size = min(window_size)
        if min_size > 0 and shift_size > 0:
            self.shift_size = min_size // 2
        else:
            self.shift_size = 0

    def build(self, input_shape):
        if self.shift_size > 0:
            H, W = input_shape[1], input_shape[2]
            # Two pre-existing defects in this block, unrelated to Stage 06,
            # fixed here as the minimal, behavior-preserving changes required
            # for any SwinTransformerBlock with shift_size > 0 (every
            # BasicLayer of depth >= 2, including every stage of both Stage
            # 06 branches) to build at all under the currently-installed
            # Keras version. create_hybrid_model() is unaffected by either
            # fix (it passes shift_size=0 explicitly, never entering this
            # branch):
            #   1. dtype=np.float32 (not np.zeros' default float64) -- the
            #      attn_mask built from this array is later combined with a
            #      Python float32 literal via tf.where, which otherwise
            #      raises a float64-vs-float32 TypeError.
            #   2. `with tf.init_scope()` around the mask computation -- H
            #      and W are static Python ints here (known at build time),
            #      so this mask has no dependency on any runtime tensor, but
            #      without init_scope, constructing the tf.Variable below
            #      from a tensor produced by graph ops traced during Keras
            #      3's symbolic build pass raises "could not be lifted out
            #      of a tf.function". init_scope is TensorFlow's own
            #      documented mechanism for exactly this situation --
            #      escaping the current tracing context to compute a
            #      Variable's initial value eagerly -- and changes nothing
            #      about the mask's resulting values.
            with tf.init_scope():
                img_mask = np.zeros([1, H, W, 1], dtype=np.float32)
                h_slices = (slice(0, -self.window_size_h),
                            slice(-self.window_size_h, -self.shift_size),
                            slice(-self.shift_size, None))
                w_slices = (slice(0, -self.window_size_w),
                            slice(-self.window_size_w, -self.shift_size),
                            slice(-self.shift_size, None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, h, w, :] = cnt
                        cnt += 1
                if isinstance(self.window_size, int):
                    window_size = self.window_size
                else:
                    window_size = self.window_size_h
                mask_windows = window_partition(
                    tf.convert_to_tensor(img_mask), window_size)
                if isinstance(self.window_size, tuple):
                    window_h, window_w = self.window_size
                    window_area = window_h * window_w
                else:
                    window_area = self.window_size * self.window_size
                mask_windows = tf.reshape(
                    mask_windows, [-1, window_area])
                attn_mask = tf.expand_dims(mask_windows, 1) - tf.expand_dims(mask_windows, 2)
                attn_mask = tf.where(attn_mask != 0, -100.0, attn_mask)
                attn_mask = tf.where(attn_mask == 0, 0.0, attn_mask)
                self.attn_mask = tf.Variable(
                    initial_value=attn_mask,
                    trainable=False,
                    name='attn_mask')
        else:
            self.attn_mask = None
        super().build(input_shape)

    def call(self, x, training=None):
        H, W, C = tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        B = tf.shape(x)[0]
        shortcut = x
        x = self.norm1(x)
        x_reshaped = tf.reshape(x, [B, H, W, C])
        if self.shift_size > 0:
            shifted_x = tf.roll(
                x_reshaped, shift=[-self.shift_size, -self.shift_size], axis=[1, 2])
        else:
            shifted_x = x_reshaped
        x_windows = window_partition(shifted_x, self.window_size)
        if isinstance(self.window_size, tuple):
            window_h, window_w = self.window_size
            window_area = window_h * window_w
        else:
            window_area = self.window_size * self.window_size
        x_windows = tf.reshape(
            x_windows, [-1, window_area, C])
        attn_windows = self.attn(
            x_windows, mask=self.attn_mask, training=training)
        if isinstance(self.window_size, tuple):
            window_h, window_w = self.window_size
            attn_windows = tf.reshape(
                attn_windows, [-1, window_h, window_w, C])
        else:
            attn_windows = tf.reshape(
                attn_windows, [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, H, W, C)
        if self.shift_size > 0:
            x = tf.roll(
                shifted_x, shift=[self.shift_size, self.shift_size], axis=[1, 2])
        else:
            x = shifted_x
        x = shortcut + x
        x = x + self.mlp(self.norm2(x), training=training)
        return x

class PatchMerging(layers.Layer):
    def __init__(self, dim, norm_layer=layers.LayerNormalization, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.reduction = layers.Dense(2 * dim, use_bias=False, name='reduction')
        self.norm = norm_layer(epsilon=1e-5, name='norm')

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        # Pre-existing defect, unrelated to Stage 06, fixed here as the
        # minimal change required for this layer to build at all under the
        # currently-installed Keras version: `assert H % 2 == 0` on a
        # `tf.shape(x)` result tries to use a symbolic tensor as a Python
        # bool during Keras 3's graph tracing, which is disallowed. None of
        # H/W/B/C (all dynamic via tf.shape) are otherwise used anywhere in
        # this method -- the actual merge below operates on `x` directly via
        # tensor slicing -- so this sanity check is switched to the static
        # shape (always known for this project's fixed-resolution inputs),
        # skipped only if genuinely unknown, rather than removed.
        static_h, static_w = x.shape[1], x.shape[2]
        if static_h is not None and static_w is not None:
            assert static_h % 2 == 0 and static_w % 2 == 0, \
                f"H and W ({static_h}, {static_w}) are not even."
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = tf.concat([x0, x1, x2, x3], axis=-1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class BasicLayer(layers.Layer):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, dropout_rate=0.,
                 attention_dropout_rate=0., drop_path_rate=0.,
                 downsample=None, use_checkpoint=False, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = []
        for i in range(depth):
            self.blocks.append(
                SwinTransformerBlock(
                    dim=dim, num_heads=num_heads, window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    dropout_rate=dropout_rate, attention_dropout_rate=attention_dropout_rate,
                    name=f'blocks_{i}'))
        if downsample is not None:
            self.downsample = downsample(dim=dim, name='downsample')
        else:
            self.downsample = None

    def call(self, x, training=None):
        for block in self.blocks:
            if self.use_checkpoint:
                x = tf.keras.utils.tf_utils.call_with_conditional_update(block, x, training=training)
            else:
                x = block(x, training=training)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class SwinTransformer(Model):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=5,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 dropout_rate=0.1, attention_dropout_rate=0.1,
                 drop_path_rate=0.1, norm_layer=layers.LayerNormalization,
                 patch_norm=True, use_checkpoint=False, name="swin_transformer"):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]
        self.layers = []
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                dropout_rate=dropout_rate,
                attention_dropout_rate=attention_dropout_rate,
                drop_path_rate=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                name=f'layers_{i_layer}'
            )
            self.layers.append(layer)
        self.norm = norm_layer(epsilon=1e-5, name='norm')
        self.head = layers.Dense(num_classes, name='head')

    def call(self, x, training=None):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.norm(x)
        x = tf.reduce_mean(x, axis=[1, 2])
        x = self.head(x)
        return x

def create_swin_tiny_model(input_shape=(224, 224, 1), num_classes=5):
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        drop_path_rate=0.1,
        norm_layer=layers.LayerNormalization,
        patch_norm=True,
        use_checkpoint=False
    )
    outputs = model(x)
    return Model(inputs=inputs, outputs=outputs)

def create_hybrid_model(input_shape=(224, 224, 1), num_classes=5):
    from tensorflow.keras.applications import EfficientNetB0
    inputs = Input(shape=input_shape)
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])
    base_model = EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(224, 224, 3)
    )
    for layer in base_model.layers[:100]:
        layer.trainable = False
    features = base_model(x)
    feature_dim = features.shape[-1]
    swin_block = SwinTransformerBlock(
        dim=feature_dim,
        num_heads=8,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.,
        qkv_bias=True,
        dropout_rate=0.1,
        attention_dropout_rate=0.1,
        name='swin_refine'
    )
    refined_features = swin_block(features)
    x = layers.GlobalAveragePooling2D()(refined_features)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC()]
    )
    return model


# --- Stage 06: Global Feature Extraction (Dual-Scale Swin Transformer) ---
#
# Approved architecture (see PROJECT_CODE.md's "Approved Research Innovation"
# section and the Stage 06 design-resolution record) -- two parallel Swin
# encoder branches at different patch granularities, fused by channel-wise
# concatenation only. Assembled entirely from the existing PatchEmbed /
# BasicLayer / PatchMerging classes above, unmodified. create_swin_tiny_model()
# and create_hybrid_model() are untouched by everything below, per
# IMPLEMENTATION_PLAN.md's explicit file-scope instruction for this stage.
#
# Every parameter is a named, inspectable module-level constant -- none of
# the shapes below are hardcoded magic numbers; they are derived from the
# two branch configs (see _dual_scale_branch_geometry()).

DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE = (256, 256, 3)

# Branch A -- fine scale: patch_size=4, full 4-stage Swin-Tiny hierarchy.
# Per-head dimension is 32 at every stage (96/3, 192/6, 384/12, 768/24),
# following the original Swin Transformer paper's own design rule
# (Liu et al., ICCV 2021, arXiv:2103.14030).
DUAL_SCALE_BRANCH_A_CONFIG = {
    "patch_size": 4,
    "embed_dim": 96,
    "depths": (2, 2, 6, 2),
    "num_heads": (3, 6, 12, 24),
    "window_size": 8,
}

# Branch B -- coarse scale: patch_size=8, truncated to 3 stages (one fewer
# than Branch A). This asymmetry is a project-specific engineering choice,
# not taken from any cited paper: Branch A's total downsampling factor is
# patch_size * 2**(num_stages-1) = 4 * 2**3 = 32; Branch B's is
# 8 * 2**2 = 32 -- identical. Matching total downsampling (rather than
# matching stage count) is what makes both branches converge on the same,
# position-aligned final spatial grid, so channel-wise concatenation needs
# no resize/interpolation step. DS-TransUNet (Lin et al., arXiv:2106.06716)
# is the literature precedent for the (4, 8) patch-size pairing itself --
# its own branches serve a decoder needing matched-depth skip connections,
# which Stage 06 has no equivalent of, hence the depth truncation here.
DUAL_SCALE_BRANCH_B_CONFIG = {
    "patch_size": 8,
    "embed_dim": 96,
    "depths": (2, 2, 6),
    "num_heads": (3, 6, 12),
    "window_size": 8,
}


def _dual_scale_branch_geometry(config_dict):
    """Derives (total_downsample_factor, final_channels) from a branch
    config -- never hardcoded, always recomputed from patch_size/embed_dim/
    depths, so the module-level output-shape constants below cannot drift
    out of sync with the branch configs themselves."""
    num_stages = len(config_dict["depths"])
    total_downsample = config_dict["patch_size"] * (2 ** (num_stages - 1))
    final_channels = config_dict["embed_dim"] * (2 ** (num_stages - 1))
    return total_downsample, final_channels


_BRANCH_A_DOWNSAMPLE, _BRANCH_A_FINAL_CHANNELS = _dual_scale_branch_geometry(DUAL_SCALE_BRANCH_A_CONFIG)
_BRANCH_B_DOWNSAMPLE, _BRANCH_B_FINAL_CHANNELS = _dual_scale_branch_geometry(DUAL_SCALE_BRANCH_B_CONFIG)

# Derived, not chosen: both branches share the same total downsampling
# factor (32), so at the default 256x256 input they both reach an 8x8 final
# grid; concatenating their channels gives 768+384=1152. No post-
# concatenation projection is applied -- RACAF's own formula
# (RACAF_ARCHITECTURE.md Sec 5: Ghat = W_r * GAP(G) + b_r) already projects
# G's raw channel count to whatever dimensionality the downstream needs, so
# a second projection here would be redundant.
DUAL_SCALE_OUTPUT_GRID = DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE[0] // _BRANCH_A_DOWNSAMPLE
DUAL_SCALE_OUTPUT_TOKENS = DUAL_SCALE_OUTPUT_GRID ** 2
DUAL_SCALE_OUTPUT_CHANNELS = _BRANCH_A_FINAL_CHANNELS + _BRANCH_B_FINAL_CHANNELS

# .weights.h5, not .keras: see GlobalFeatureExtractionStage.save()'s docstring --
# this stage's custom Swin layer classes have no get_config() (a pre-existing
# gap, not fixed here), so it persists weights only, not a full architecture file.
DEFAULT_GLOBAL_FEATURE_MODEL_PATH = os.path.join(config.GLOBAL_FEATURE_MODEL_DIR, "best_model.weights.h5")


def _validate_dual_scale_input_shape(input_shape):
    """Raises ValueError with a specific message if `input_shape` does not
    satisfy every geometric constraint the approved dual-scale design
    depends on: 3-channel RGB, square, exact window-size divisibility at
    each branch's final stage (which -- since every earlier stage's grid is
    a power-of-2 multiple of the final one -- also guarantees divisibility
    at every other stage), and identical final grid size across both
    branches (required for a resize-free channel concatenation)."""
    if len(input_shape) != 3 or input_shape[2] != 3:
        raise ValueError(
            f"create_dual_scale_swin_model requires a 3-channel RGB input_shape "
            f"(H, W, 3); got {input_shape}."
        )
    height, width, _ = input_shape
    if height != width:
        raise ValueError(
            f"create_dual_scale_swin_model requires a square input_shape; got {height}x{width}."
        )

    grids = {}
    for branch_name, branch_config in (("Branch A", DUAL_SCALE_BRANCH_A_CONFIG),
                                        ("Branch B", DUAL_SCALE_BRANCH_B_CONFIG)):
        total_downsample, _ = _dual_scale_branch_geometry(branch_config)
        if height % total_downsample != 0:
            raise ValueError(
                f"{branch_name}: input resolution {height} is not divisible by its total "
                f"downsampling factor {total_downsample} (patch_size={branch_config['patch_size']} "
                f"* 2**{len(branch_config['depths']) - 1})."
            )
        final_grid = height // total_downsample
        if final_grid % branch_config["window_size"] != 0:
            raise ValueError(
                f"{branch_name}: final grid size {final_grid}x{final_grid} is not divisible by "
                f"window_size={branch_config['window_size']} -- window partitioning would fail."
            )
        grids[branch_name] = final_grid

    if grids["Branch A"] != grids["Branch B"]:
        raise ValueError(
            f"Branch A's final grid ({grids['Branch A']}x{grids['Branch A']}) does not match "
            f"Branch B's final grid ({grids['Branch B']}x{grids['Branch B']}) at input resolution "
            f"{height} -- channel-wise concatenation requires identical, position-aligned grids. "
            "This input resolution breaks the approved architecture's alignment property."
        )


def _build_dual_scale_branch(inputs, branch_config, name_prefix):
    """Assembles one Swin encoder branch directly from this module's
    existing PatchEmbed / BasicLayer / PatchMerging classes -- no
    modification to any of them. Returns the branch's final (B, H', W', C')
    feature map after a closing LayerNormalization (matching
    SwinTransformer's own final-norm convention, Sec `SwinTransformer.norm`
    above) -- no pooling, no head. PatchEmbed has no `name` parameter in
    this module's existing implementation, so it relies on Keras' automatic
    unique-name assignment; BasicLayer/PatchMerging both accept an explicit
    `name` and are named per-branch below."""
    x = PatchEmbed(
        patch_size=branch_config["patch_size"],
        embed_dim=branch_config["embed_dim"],
        norm_layer=layers.LayerNormalization,
    )(inputs)

    num_stages = len(branch_config["depths"])
    for stage_index in range(num_stages):
        x = BasicLayer(
            dim=int(branch_config["embed_dim"] * 2 ** stage_index),
            depth=branch_config["depths"][stage_index],
            num_heads=branch_config["num_heads"][stage_index],
            window_size=branch_config["window_size"],
            downsample=PatchMerging if stage_index < num_stages - 1 else None,
            name=f"{name_prefix}_stage{stage_index}",
        )(x)

    return layers.LayerNormalization(epsilon=1e-5, name=f"{name_prefix}_final_norm")(x)


def create_dual_scale_swin_model(input_shape=DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE):
    """
    Builds Stage 06's Dual-Scale Swin Transformer for Global Feature
    Extraction: two parallel Swin encoder branches (Branch A: patch_size=4,
    4 stages; Branch B: patch_size=8, 3 stages -- see
    DUAL_SCALE_BRANCH_A_CONFIG/DUAL_SCALE_BRANCH_B_CONFIG above) over the
    same RGB input, fused by channel-wise concatenation only. There is no
    post-concatenation projection, no cross-branch or cross-attention
    interaction, no pooling, and no classification head -- Stage 7/RACAF
    are responsible for any further interaction or projection.

    Consumes Stage 02 processed RGB directly -- never Stage 5's
    local_features, never vessel or lesion probability maps, and (unlike
    Stage 05) has no frozen upstream dependency to guard with
    `stop_gradient`, since it never touches Stage 03/04 at all.

    Random initialization only -- no pretrained weights are loaded or
    referenced anywhere in this function. Returns an **uncompiled**
    `keras.Model`: Stage 06, like Stage 05, has no standalone loss (see
    `GlobalFeatureExtractionStage` below) -- it is trained only as part of
    a future joint Stage 05-08 + RACAF graph, through CORN's eventual
    ordinal loss.

    Returns a tensor of shape (B, DUAL_SCALE_OUTPUT_TOKENS,
    DUAL_SCALE_OUTPUT_CHANNELS) -- (B, 64, 1152) at the default input shape.
    """
    _validate_dual_scale_input_shape(input_shape)

    inputs = Input(shape=input_shape, name="global_feature_input")

    branch_a = _build_dual_scale_branch(inputs, DUAL_SCALE_BRANCH_A_CONFIG, "branch_a")
    branch_b = _build_dual_scale_branch(inputs, DUAL_SCALE_BRANCH_B_CONFIG, "branch_b")

    fused = layers.concatenate([branch_a, branch_b], name="dual_scale_concat")
    grid = fused.shape[1]
    channels = fused.shape[3]
    flattened = layers.Reshape((grid * grid, channels), name="dual_scale_flatten")(fused)

    # Explicit float32 output: keeps this feature map numerically stable if
    # this model is ever run under a mixed_float16 policy during the future
    # joint Stage 05-08 training graph, matching local_feature_extraction_model.py's
    # / lesion_segmentation_model.py's identical convention for their own
    # final layers.
    outputs = layers.Activation("linear", dtype="float32", name="global_features")(flattened)

    return Model(inputs=inputs, outputs=outputs, name="global_feature_extraction_dual_scale_swin")


class GlobalFeatureExtractionStage(FeatureExtractionStage):
    """`pipeline.FeatureExtractionStage` implementation for the Dual-Scale
    Swin Transformer. `predict`/`predict_batch`/`save`/`load` are fully
    functional; `train`/`evaluate` raise `NotImplementedError` -- Stage 06,
    like Stage 05, has no standalone training or evaluation procedure of
    its own (see `create_dual_scale_swin_model`'s docstring), mirroring
    `LocalFeatureExtractionStage`'s exact pattern."""

    def __init__(self, input_shape=DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE):
        self.input_shape = input_shape
        self.model = None

    def train(self, train_data, val_data=None, **kwargs):
        raise NotImplementedError(
            "Global Feature Extraction has no standalone training procedure in this project -- "
            "it has no global-feature ground truth of its own and is only ever trained jointly "
            "with Stages 05, 07, 08 and RACAF, through the downstream CORN ordinal loss (see the "
            "Stage 06 design resolution and RACAF_ARCHITECTURE.md Sec 7). That joint training "
            "script does not exist yet."
        )

    def evaluate(self, eval_data, **kwargs):
        raise NotImplementedError(
            "Global Feature Extraction has no standalone evaluation metric in this project -- "
            "its usefulness is only observable through the joint downstream pipeline's "
            "classification metrics, not reproduced here."
        )

    def save(self, path):
        """Saves weights only (`model.save_weights`), not a full-architecture
        `.keras` file. Two independent reasons, not one workaround:

        1. The custom Swin layer classes this model is built from
           (PatchEmbed, WindowAttention, SwinTransformerBlock, PatchMerging,
           BasicLayer -- all pre-existing, none modified here) have no
           `get_config()` implementation, so a full-architecture `.keras`
           save is not currently possible without adding serialization
           support to five existing classes -- judged out of scope for what
           this stage actually needs.
        2. This is not a deviation from project convention -- it IS the
           project's shared training framework's own default.
           `training/callbacks.py`'s `build_callbacks()` defaults to
           `save_weights_only=True`, documented there as producing
           `best.weights.h5`/`last.weights.h5`; `training/trainer.py`'s
           `TrainingConfig.save_weights_only` also defaults to `True`.
           `LesionSegmentationStage` (Stage 04) is the one that explicitly
           *opts into* `save_weights_only=False` for full `.keras`
           checkpoints (`lesion_segmentation_model.py`'s own
           `# complete .keras checkpoints, not weights-only` comment) --
           it is the deliberate exception, not the baseline this stage
           deviates from.

        Weights-only save/load (rebuild via `create_dual_scale_swin_model()`,
        then load weights into the fresh instance) is the standard, safe
        Keras pattern for this situation, requires no change to any existing
        Swin class, and matches how the eventual joint Stage 05-08 + RACAF
        training run will itself checkpoint by default unless a future
        implementer deliberately opts into `save_weights_only=False` for
        that joint model too -- at which point the same custom-layer
        `get_config()` gap would need addressing for the *whole* joint
        model, not just Stage 06 in isolation. `path` should end in
        `.weights.h5` (`DEFAULT_GLOBAL_FEATURE_MODEL_PATH`'s own extension)."""
        if self.model is None:
            raise RuntimeError("No model to save -- call build() (or assign self.model) first.")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save_weights(path)
        return path

    def load(self, path=DEFAULT_GLOBAL_FEATURE_MODEL_PATH, input_shape=None):
        """Rebuilds the architecture via `create_dual_scale_swin_model()`
        (random-initialized) and then loads saved weights into it -- see
        `save()`'s docstring for why this is weights-only rather than a
        full-architecture `.keras` load. `input_shape` defaults to
        `self.input_shape` (set at construction); pass it explicitly only
        if loading weights saved from a differently-shaped instance."""
        self.model = create_dual_scale_swin_model(input_shape=input_shape or self.input_shape)
        self.model.load_weights(path)
        return self

    def build(self):
        """Builds a fresh, uncompiled model from `self.input_shape` and
        assigns it to `self.model` -- mirrors
        `LocalFeatureExtractionStage.build()`'s lazy-build-on-first-use
        pattern."""
        self.model = create_dual_scale_swin_model(input_shape=self.input_shape)
        return self.model

    def predict(self, input_data):
        if self.model is None:
            raise RuntimeError(
                "GlobalFeatureExtractionStage.build() or .load() must be called before predict()."
            )
        batch = np.asarray(input_data, dtype="float32")
        if batch.ndim == len(self.input_shape):
            batch = batch[None, ...]
        return self.model.predict(batch, verbose=0)[0]

    def predict_batch(self, inputs):
        if self.model is None:
            raise RuntimeError(
                "GlobalFeatureExtractionStage.build() or .load() must be called before predict_batch()."
            )
        batch = np.stack([np.asarray(x, dtype="float32") for x in inputs], axis=0)
        predictions = self.model.predict(batch, verbose=0)
        return [predictions[i] for i in range(predictions.shape[0])]


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Memory growth enabled for GPU")
        except:
            print("Invalid device or cannot modify virtual devices once initialized")
    from tensorflow.keras.utils import plot_model
    import matplotlib.pyplot as plt
    model = create_hybrid_model(input_shape=(224, 224, 1), num_classes=5)
    model.summary()
    try:
        plot_model(model, to_file='hybrid_model.png', show_shapes=True)
        print("Model diagram saved to hybrid_model.png")
    except:
        print("Could not generate model diagram. Install pydot and graphviz for visualization.")
    test_batch = np.random.random((2, 224, 224, 1))
    with tf.device('/CPU:0'):
        outputs = model.predict(test_batch)
    print(f"Output shape: {outputs.shape}")
    print("Memory test successful!")
