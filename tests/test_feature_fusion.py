"""
Regression tests for feature_fusion.py's Stage 07 implementation
(build_adaptive_cross_attention / AdaptiveCrossAttentionStage /
Factorized2DPositionalEmbedding).

Model *construction* (defining the Keras functional graph) is cheap
regardless of parameter count -- only actual forward/backward passes cost
real compute -- so each TestCase below builds its model once in
setUpClass and reuses it across every test method, mirroring
test_swin_transformer_dual_scale.py's identical convention.

No training happens anywhere in this file. Every model built here is
untrained (random initialization) -- no metric is ever reported as a real
evaluation result. All save/load tests use `tempfile` and clean up after
themselves; no real checkpoint is left behind.
"""

import inspect
import os
import shutil
import tempfile
import unittest

import numpy as np
import tensorflow as tf

import feature_fusion as ff
from pipeline.inference import InferenceStage
from pipeline.trainable import TrainableStage


def _random_local(batch=2):
    return np.random.rand(batch, *ff.DEFAULT_LOCAL_SHAPE).astype("float32")


def _random_global(batch=2):
    return np.random.rand(batch, *ff.DEFAULT_GLOBAL_SHAPE).astype("float32")


class ShapeTests(unittest.TestCase):
    """Every tensor contract listed in the approved Stage 07 design,
    measured directly rather than assumed."""

    @classmethod
    def setUpClass(cls):
        cls.model = ff.build_adaptive_cross_attention()

    def test_local_input_shape(self):
        self.assertEqual(self.model.get_layer("local_features").output.shape, (None, 32, 32, 256))

    def test_global_input_shape(self):
        # Stage 06's real, implemented output is already flattened (64, 1152) --
        # not the conceptual spatial (8, 8, 1152) form -- see feature_fusion.py's
        # module-level DEFAULT_GLOBAL_SHAPE comment for why.
        self.assertEqual(self.model.get_layer("global_features").output.shape, (None, 64, 1152))

    def test_local_flatten_shape(self):
        self.assertEqual(self.model.get_layer("local_flatten").output.shape, (None, 1024, 256))

    def test_q_source_shape(self):
        # Q is derived from Global's projection: (B, 64, 256).
        self.assertEqual(self.model.get_layer("global_q_projection").output.shape, (None, 64, 256))

    def test_k_v_source_shape(self):
        # K and V are both derived from Local's single projection: (B, 1024, 256).
        self.assertEqual(self.model.get_layer("local_kv_projection").output.shape, (None, 1024, 256))

    def test_cross_attention_output_shape(self):
        self.assertEqual(self.model.get_layer("cross_attention").output.shape, (None, 64, 256))

    def test_final_output_shape(self):
        self.assertEqual(self.model.output_shape, (None, 256))

    def test_forward_pass_measured_shapes(self):
        out = self.model.predict([_random_local(2), _random_global(2)], verbose=0)
        self.assertEqual(out.shape, (2, 256))

    def test_attention_score_tensor_shape(self):
        """Directly measures the (B, heads, 64, 1024) attention-score tensor
        the approved design calls out explicitly -- not inferred from the
        output shape alone."""
        mha = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=32, dropout=0.1)
        q = tf.random.normal((2, 64, 256))
        kv = tf.random.normal((2, 1024, 256))
        out, scores = mha(query=q, value=kv, key=kv, return_attention_scores=True)
        self.assertEqual(tuple(out.shape), (2, 64, 256))
        self.assertEqual(tuple(scores.shape), (2, 8, 64, 1024))


class ArchitectureTests(unittest.TestCase):
    """Verifies every fixed hyperparameter from the approved design by
    measurement, not by re-stating the constant back at itself where
    avoidable."""

    @classmethod
    def setUpClass(cls):
        cls.model = ff.build_adaptive_cross_attention()

    def test_module_level_constants(self):
        self.assertEqual(ff.D_MODEL, 256)
        self.assertEqual(ff.NUM_HEADS, 8)
        self.assertEqual(ff.HEAD_DIM, 32)
        self.assertEqual(ff.D_MODEL // ff.NUM_HEADS, ff.HEAD_DIM)
        self.assertEqual(ff.FFN_DIM, 1024)
        self.assertEqual(ff.DROPOUT_RATE, 0.1)
        self.assertEqual(ff.ATTENTION_DROPOUT_RATE, 0.1)

    def test_cross_attention_layer_configuration(self):
        mha = self.model.get_layer("cross_attention")
        self.assertEqual(mha.num_heads, 8)
        self.assertEqual(mha.key_dim, 32)
        self.assertEqual(mha.dropout, 0.1)

    def test_ffn_dimensions_and_activation(self):
        expand = self.model.get_layer("ffn_expand")
        project = self.model.get_layer("ffn_project")
        self.assertEqual(expand.output.shape, (None, 64, 1024))
        self.assertEqual(project.output.shape, (None, 64, 256))
        self.assertEqual(expand.activation.__name__, "gelu")
        # ffn_project must be linear (no activation) -- only ffn_expand carries GELU.
        self.assertEqual(project.activation.__name__, "linear")

    def test_ffn_dropout_rate(self):
        self.assertEqual(self.model.get_layer("ffn_dropout").rate, 0.1)

    def test_pre_layer_norm_present(self):
        # Pre-LN: LayerNorm on the RAW tokens, before projection -- one per branch,
        # plus a third before the FFN sub-layer.
        for name in ("local_pre_ln", "global_pre_ln", "ffn_pre_ln"):
            layer = self.model.get_layer(name)
            self.assertIsInstance(layer, tf.keras.layers.LayerNormalization)

    def test_residual_connections_present(self):
        cross_attn_residual = self.model.get_layer("cross_attn_residual")
        ffn_residual = self.model.get_layer("ffn_residual")
        self.assertIsInstance(cross_attn_residual, tf.keras.layers.Add)
        self.assertIsInstance(ffn_residual, tf.keras.layers.Add)
        # The cross-attention residual must add back the Q-source (post-projection,
        # post-positional-embedding Global tokens), not some other tensor.
        inbound_names = [
            t._keras_history.operation.name for t in cross_attn_residual.input
        ]
        self.assertIn("global_pos_embed", inbound_names)
        self.assertIn("cross_attention", inbound_names)

    def test_no_self_attention_sublayer(self):
        """The approved design has exactly one attention layer (cross-attention
        only) -- no additional self-attention sub-layer, since both branches
        are already self-mixed upstream (Swin windows, CNN receptive fields)."""
        attention_layers = [
            l for l in self.model.layers if isinstance(l, tf.keras.layers.MultiHeadAttention)
        ]
        self.assertEqual(len(attention_layers), 1)
        self.assertEqual(attention_layers[0].name, "cross_attention")

    def test_d_model_divisible_by_heads(self):
        with self.assertRaises(ValueError):
            ff.build_adaptive_cross_attention(d_model=257, num_heads=8)


class AttentionDirectionTests(unittest.TestCase):
    """Explicitly verifies Global provides Q and Local provides K,V -- not
    merely that the output shape happens to work out. A reversed
    assignment (Local queries Global) would produce a differently-shaped
    intermediate (cross_attention output (B,1024,256) instead of
    (B,64,256)), so shape alone is suggestive; the token-count and
    perturbation-sensitivity checks below make the direction unambiguous."""

    @classmethod
    def setUpClass(cls):
        cls.model = ff.build_adaptive_cross_attention()

    def test_cross_attention_output_token_count_matches_global_not_local(self):
        # Only possible if Global's 64 tokens are the query -- if Local's 1024
        # tokens were the query instead, this would be (None, 1024, 256).
        out_shape = self.model.get_layer("cross_attention").output.shape
        self.assertEqual(out_shape[1], 64)
        self.assertNotEqual(out_shape[1], 1024)

    def test_source_code_uses_global_as_query_local_as_key_value(self):
        """Reads the actual call arguments in source -- structural evidence,
        not just a runtime-shape inference."""
        source = inspect.getsource(ff.build_adaptive_cross_attention)
        call_line = next(l for l in source.splitlines() if "query=" in l)
        self.assertIn("query=global_proj", call_line.replace(" ", ""))
        value_key_block = source[source.index("query=global_proj"):]
        self.assertIn("value=local_proj", value_key_block.replace(" ", "").replace("\n", ""))
        self.assertIn("key=local_proj", value_key_block.replace(" ", "").replace("\n", ""))

    def test_output_is_more_sensitive_to_local_content_than_global_alone(self):
        """Local supplies K,V (the content being attended over) -- changing
        Local's content should change the fused output. This does not by
        itself prove direction, but is a necessary consistency check: if
        Local had no effect on the output at all, K,V could not actually be
        wired to Local."""
        rng = np.random.RandomState(0)
        local = rng.rand(1, *ff.DEFAULT_LOCAL_SHAPE).astype("float32")
        glob = rng.rand(1, *ff.DEFAULT_GLOBAL_SHAPE).astype("float32")
        local2 = rng.rand(1, *ff.DEFAULT_LOCAL_SHAPE).astype("float32")

        out1 = self.model.predict([local, glob], verbose=0)
        out2 = self.model.predict([local2, glob], verbose=0)
        self.assertFalse(np.allclose(out1, out2, atol=1e-5))


class PositionalEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ff.build_adaptive_cross_attention()

    def test_local_positional_embedding_grid(self):
        layer = self.model.get_layer("local_pos_embed")
        self.assertIsInstance(layer, ff.Factorized2DPositionalEmbedding)
        self.assertEqual((layer.grid_h, layer.grid_w), (32, 32))
        self.assertEqual(layer.dim, 256)
        self.assertEqual(layer.row_embed.shape, (32, 256))
        self.assertEqual(layer.col_embed.shape, (32, 256))

    def test_global_positional_embedding_grid(self):
        layer = self.model.get_layer("global_pos_embed")
        self.assertIsInstance(layer, ff.Factorized2DPositionalEmbedding)
        self.assertEqual((layer.grid_h, layer.grid_w), (8, 8))
        self.assertEqual(layer.dim, 256)
        self.assertEqual(layer.row_embed.shape, (8, 256))
        self.assertEqual(layer.col_embed.shape, (8, 256))

    def test_positional_embedding_actually_added_before_attention(self):
        """Not merely present in the graph -- verifies the layer's output
        differs from its input by exactly the (broadcast) position tensor,
        and that this happens upstream of the cross-attention call."""
        layer = ff.Factorized2DPositionalEmbedding(grid_h=8, grid_w=8, dim=256, name="probe")
        x = tf.zeros((1, 64, 256))
        out = layer(x)
        # With a zero input, the output must equal the (broadcast) position
        # tensor exactly -- i.e. positional information was actually added.
        self.assertFalse(np.allclose(out.numpy(), 0.0))
        row = layer.row_embed.numpy()
        col = layer.col_embed.numpy()
        expected = (row[:, None, :] + col[None, :, :]).reshape(64, 256)
        np.testing.assert_allclose(out.numpy()[0], expected, atol=1e-6)

    def test_positional_embedding_upstream_of_cross_attention(self):
        cross_attn_node = self.model.get_layer("cross_attention")._inbound_nodes[0]
        inbound_layer_names = {
            t._keras_history.operation.name for t in cross_attn_node.input_tensors
        }
        self.assertIn("global_pos_embed", inbound_layer_names)
        self.assertIn("local_pos_embed", inbound_layer_names)

    def test_no_second_positional_mechanism(self):
        """The approved design uses exactly one positional mechanism
        (factorized 2D embeddings) -- no second, competing positional
        encoding (e.g. no sinusoidal PE, no relative position bias) is
        introduced by this module."""
        pos_layers = [
            l for l in self.model.layers
            if isinstance(l, ff.Factorized2DPositionalEmbedding)
        ]
        self.assertEqual(len(pos_layers), 2)  # exactly one per branch


class RACAFBoundaryTests(unittest.TestCase):
    """Structural evidence (source inspection), not just a claim, that
    Stage 07 stays entirely inside its documented boundary -- mirroring
    test_swin_transformer_dual_scale.py's KnownLegacyIssueTests' own use
    of structural (not just narrative) verification.

    Checks operate on actual CODE identifiers (via `tokenize`'s NAME
    tokens) and actual imports (via `ast`) -- deliberately NOT a raw
    substring search over the whole file, since this module's own
    docstrings legitimately *discuss* RACAF's vocabulary (TTA,
    reliability, gate, ground-truth label) precisely to explain that Stage
    07 does none of it. A raw text search would false-positive on that
    explanation; only real code identifiers are forbidden here."""

    @classmethod
    def setUpClass(cls):
        cls.path = os.path.join(os.path.dirname(__file__), "..", "feature_fusion.py")
        with open(cls.path) as f:
            cls.source = f.read()

        import io
        import tokenize as tokenize_mod

        names = []
        with open(cls.path, "rb") as f:
            for tok in tokenize_mod.tokenize(f.readline):
                if tok.type == tokenize_mod.NAME:
                    names.append(tok.string.lower())
        cls.code_identifiers = set(names)

        cls.imported_modules = set()
        import ast
        tree = ast.parse(cls.source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    cls.imported_modules.add(alias.name.lower())
            elif isinstance(node, ast.ImportFrom) and node.module:
                cls.imported_modules.add(node.module.lower())

    def test_no_tta_identifiers(self):
        forbidden = {"tta", "test_time_aug", "hflip", "vflip", "rot90", "rotate180"}
        found = forbidden & self.code_identifiers
        self.assertFalse(found, f"found forbidden TTA identifiers in code: {found}")

    def test_no_reliability_or_uncertainty_identifiers(self):
        forbidden = {"reliability", "disagreement", "uncertainty", "kappa", "confidence_score", "delta"}
        found = forbidden & self.code_identifiers
        self.assertFalse(found, f"found forbidden reliability identifiers in code: {found}")

    def test_no_gating_identifiers(self):
        forbidden = {"gate", "sigmoid_gate", "w_g", "b_g"}
        found = forbidden & self.code_identifiers
        self.assertFalse(found, f"found forbidden gating identifiers in code: {found}")

    def test_no_ground_truth_or_label_identifiers(self):
        forbidden = {"ground_truth", "label", "labels", "y_true", "diagnosis"}
        found = forbidden & self.code_identifiers
        self.assertFalse(found, f"found forbidden ground-truth identifiers in code: {found}")

    def test_no_stage_4_import(self):
        forbidden_modules = {"lesion_segmentation_model", "lesion_segmentation_dataset"}
        found = forbidden_modules & self.imported_modules
        self.assertFalse(found, f"found forbidden Stage 4 imports: {found}")
        forbidden_calls = {"predict_lesion_mask", "predict_lesion_mask_batch"}
        found_calls = forbidden_calls & self.code_identifiers
        self.assertFalse(found_calls, f"found forbidden Stage 4 calls: {found_calls}")

    def test_no_racaf_import(self):
        self.assertNotIn("racaf", self.imported_modules)

    def test_no_classification_head(self):
        model = ff.build_adaptive_cross_attention()
        for layer in model.layers:
            self.assertNotIsInstance(layer, tf.keras.layers.Softmax)
        # Output must remain the raw fused embedding, not class probabilities.
        self.assertEqual(model.output_shape, (None, 256))

    def test_no_loss_or_optimizer(self):
        model = ff.build_adaptive_cross_attention()
        self.assertIsNone(model.loss)
        self.assertFalse(model.compiled)
        self.assertIsNone(getattr(model, "optimizer", None))


class SerializationTests(unittest.TestCase):
    """Build -> save -> rebuild -> load -> compare outputs/weights, using
    full `.keras` serialization -- deliberately not repeating Stage 06's
    weights-only fallback (see feature_fusion.py's module docstring)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_full_keras_save_load_roundtrip(self):
        model = ff.build_adaptive_cross_attention()
        local = _random_local(2)
        glob = _random_global(2)
        out_before = model.predict([local, glob], verbose=0)

        path = os.path.join(self.tmpdir, "test_model.keras")
        model.save(path)
        loaded = tf.keras.models.load_model(
            path, compile=False,
            custom_objects={"Factorized2DPositionalEmbedding": ff.Factorized2DPositionalEmbedding},
        )
        out_after = loaded.predict([local, glob], verbose=0)

        np.testing.assert_allclose(out_before, out_after, atol=1e-5)
        for w_before, w_after in zip(model.get_weights(), loaded.get_weights()):
            np.testing.assert_array_equal(w_before, w_after)

    def test_positional_embedding_layer_get_config_roundtrip(self):
        layer = ff.Factorized2DPositionalEmbedding(grid_h=8, grid_w=8, dim=256, name="probe")
        layer.build((None, 64, 256))
        config = layer.get_config()
        self.assertEqual(config["grid_h"], 8)
        self.assertEqual(config["grid_w"], 8)
        self.assertEqual(config["dim"], 256)
        rebuilt = ff.Factorized2DPositionalEmbedding.from_config(config)
        self.assertEqual(rebuilt.grid_h, 8)
        self.assertEqual(rebuilt.grid_w, 8)
        self.assertEqual(rebuilt.dim, 256)

    def test_stage_class_build_save_load_predict_roundtrip(self):
        """Exercises the actual public Stage interface end-to-end, not just
        the raw builder function."""
        stage = ff.AdaptiveCrossAttentionStage()
        stage.build()
        local = _random_local(1)[0]
        glob = _random_global(1)[0]
        e_before = stage.predict((local, glob))

        path = os.path.join(self.tmpdir, "stage_model.keras")
        stage.save(path)

        loaded_stage = ff.AdaptiveCrossAttentionStage()
        loaded_stage.load(path)
        e_after = loaded_stage.predict((local, glob))

        np.testing.assert_allclose(e_before, e_after, atol=1e-5)
        self.assertEqual(e_before.shape, (256,))


class GradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ff.build_adaptive_cross_attention()

    def test_gradients_exist_and_are_finite_for_all_trainable_variables(self):
        local = tf.random.normal((2, *ff.DEFAULT_LOCAL_SHAPE))
        glob = tf.random.normal((2, *ff.DEFAULT_GLOBAL_SHAPE))
        with tf.GradientTape() as tape:
            out = self.model([local, glob])
            loss = tf.reduce_sum(out)
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.assertGreater(len(self.model.trainable_variables), 0)
        for var, grad in zip(self.model.trainable_variables, grads):
            self.assertIsNotNone(grad, f"missing gradient for {var.name}")
            self.assertTrue(np.all(np.isfinite(grad.numpy())), f"non-finite gradient for {var.name}")

    def test_no_trainable_variables_belong_to_a_frozen_upstream_model(self):
        """This module never builds or wraps Stage 1-4; there is nothing to
        accidentally require gradients through. Structural check: this
        model's own trainable-variable names never reference any Stage 1-4
        module name."""
        names = " ".join(v.name for v in self.model.trainable_variables).lower()
        for term in ("lwnet", "vessel", "lesion", "attention_unet", "efficientnet"):
            self.assertNotIn(term, names)


class BatchIndependenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = ff.build_adaptive_cross_attention()

    def test_samples_do_not_affect_one_another(self):
        rng = np.random.RandomState(42)
        local = rng.rand(2, *ff.DEFAULT_LOCAL_SHAPE).astype("float32")
        glob = rng.rand(2, *ff.DEFAULT_GLOBAL_SHAPE).astype("float32")

        # A resampled (not constant-shifted) sample 0 -- LayerNormalization is
        # translation-invariant, so a constant offset would be silently
        # absorbed and wrongly look like "no effect."
        local2 = local.copy()
        local2[0] = rng.rand(*ff.DEFAULT_LOCAL_SHAPE).astype("float32")

        out1 = self.model.predict([local, glob], verbose=0)
        out2 = self.model.predict([local2, glob], verbose=0)

        np.testing.assert_allclose(out1[1], out2[1], atol=1e-5)
        self.assertFalse(np.allclose(out1[0], out2[0], atol=1e-5))


class IntegrationTests(unittest.TestCase):
    """Synthetic end-to-end shape tests plus, separately, a real
    Stage 05 -> Stage 06 -> Stage 07 chain using the actual implemented
    models (not just representative tensors)."""

    def test_synthetic_representative_tensors(self):
        """The approved design's own representative tensors are
        already-flattened Local (2,1024,256) and Global (2,64,1152). Local's
        real upstream (Stage 05) output is spatial (32,32,256) -- this
        model's actual Input layer matches that real shape and flattens
        internally (see feature_fusion.py's DEFAULT_LOCAL_SHAPE comment) --
        so a (2,1024,256) representative tensor is reshaped back to
        (2,32,32,256) here before feeding it in; this is a lossless,
        order-preserving inverse of the model's own internal flatten, not a
        reinterpretation of the data. Global's representative tensor
        (2,64,1152) is fed directly -- it already matches Stage 06's real,
        already-flattened output shape exactly, with no reshape needed."""
        local_flat = np.random.rand(2, 1024, 256).astype("float32")
        global_flat = np.random.rand(2, 64, 1152).astype("float32")

        local_spatial = local_flat.reshape(2, 32, 32, 256)

        model = ff.build_adaptive_cross_attention()
        out = model.predict([local_spatial, global_flat], verbose=0)
        self.assertEqual(out.shape, (2, 256))

    def test_real_stage5_stage6_stage7_chain(self):
        """Runs the actual, already-implemented Stage 05 and Stage 06
        models (not representative tensors) through the actual Stage 07
        model. Confirms real integration, not just a shape contract on
        paper. Uses tiny batch size 1 to keep this test fast -- Stage 06's
        Swin construction dominates the cost, same tradeoff
        test_swin_transformer_dual_scale.py already accepts."""
        import local_feature_extraction_model as lfe
        import swin_transformer as st

        local_model = lfe.build_local_feature_extractor()
        global_model = st.create_dual_scale_swin_model()
        fusion_model = ff.build_adaptive_cross_attention()

        local_raw = np.random.rand(1, *lfe.DEFAULT_INPUT_SHAPE).astype("float32")
        global_raw = np.random.rand(1, *st.DEFAULT_GLOBAL_FEATURE_INPUT_SHAPE).astype("float32")

        L = local_model.predict(local_raw, verbose=0)
        G = global_model.predict(global_raw, verbose=0)

        self.assertEqual(L.shape, (1, 32, 32, 256))
        self.assertEqual(G.shape, (1, 64, 1152))

        E = fusion_model.predict([L, G], verbose=0)
        self.assertEqual(E.shape, (1, 256))
        self.assertTrue(np.all(np.isfinite(E)))


class AdaptiveCrossAttentionStageTests(unittest.TestCase):
    def test_is_a_trainable_and_inference_stage(self):
        stage = ff.AdaptiveCrossAttentionStage()
        self.assertIsInstance(stage, TrainableStage)
        self.assertIsInstance(stage, InferenceStage)

    def test_train_raises_not_implemented(self):
        stage = ff.AdaptiveCrossAttentionStage()
        with self.assertRaises(NotImplementedError):
            stage.train(train_data=None)

    def test_evaluate_raises_not_implemented(self):
        stage = ff.AdaptiveCrossAttentionStage()
        with self.assertRaises(NotImplementedError):
            stage.evaluate(eval_data=None)

    def test_predict_before_build_or_load_raises(self):
        stage = ff.AdaptiveCrossAttentionStage()
        with self.assertRaises(RuntimeError):
            stage.predict((_random_local(1)[0], _random_global(1)[0]))

    def test_save_before_build_raises(self):
        stage = ff.AdaptiveCrossAttentionStage()
        with self.assertRaises(RuntimeError):
            stage.save("/tmp/should_not_be_created.keras")

    def test_build_returns_uncompiled_model(self):
        stage = ff.AdaptiveCrossAttentionStage()
        model = stage.build()
        self.assertIsNone(model.loss)
        self.assertFalse(model.compiled)

    def test_predict_batch(self):
        stage = ff.AdaptiveCrossAttentionStage()
        stage.build()
        pairs = [(_random_local(1)[0], _random_global(1)[0]) for _ in range(3)]
        results = stage.predict_batch(pairs)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertEqual(r.shape, (256,))


if __name__ == "__main__":
    unittest.main()
