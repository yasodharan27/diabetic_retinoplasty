"""CPU tests for stage4_v2_aptos_cache.py (post-training APTOS cache generation), on a tiny synthetic
population built through the real pipeline functions (tests/v2_bundle_fixture.py)."""
import json
import os
import tempfile
import unittest

import numpy as np

import pipeline_v2_config as v2cfg
import stage34_cache_v2 as cache
import stage4_v2_aptos_cache as ac

try:
    from tests import v2_bundle_fixture as fx
except ImportError:
    import v2_bundle_fixture as fx


class PopulationTests(unittest.TestCase):
    def test_empty_fov_removed_counts_and_ids(self):
        pop = ac.aptos_population(fx.TRAIN + [fx.EMPTY_FOV_ONE], fx.VAL, v2cfg.SPLIT_SHA256,
                                  expected_counts={"train": 3, "val": 2})
        self.assertNotIn(fx.EMPTY_FOV_ONE[0], ac.population_ids(pop))
        self.assertEqual(len(pop["population_sha256"]), 64)
        with self.assertRaises(ac.AptosCacheError):
            ac.aptos_population(fx.TRAIN, fx.VAL, "0" * 64, expected_counts={"train": 3, "val": 2})
        with self.assertRaises(ac.AptosCacheError):
            ac.aptos_population(fx.TRAIN, fx.VAL, v2cfg.SPLIT_SHA256)            # real counts 2921/730 required
        with self.assertRaises(ac.AptosCacheError):
            ac.aptos_population(fx.TRAIN + [("IDRiD_55", 2)], fx.VAL, v2cfg.SPLIT_SHA256,
                                expected_counts={"train": 4, "val": 2})
        self.assertEqual(len(v2cfg.APTOS_EMPTY_FOV_IDS), 11)
        self.assertEqual(v2cfg.APTOS_POPULATION, sum(v2cfg.APTOS_EXPECTED_COUNTS.values()))

    def test_parity_ids_fixed_and_order_independent(self):
        ids = [f"{k:012x}" for k in range(100)]
        a = ac.parity_ids(ids, 25)
        self.assertEqual(a, ac.parity_ids(list(reversed(ids)), 25))
        self.assertEqual(len(set(a)), 25)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.b = fx.build(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_namespace_dims_dtype_channels(self):
        b = self.b
        self.assertEqual(b["gen4"], f"s4v2-{fx.MODEL_SHA[:12]}-K4")
        self.assertTrue(os.path.isdir(os.path.join(b["roots"]["stage4_cache_v2"], b["gen4"], "pathology")))
        i = b["ids"][0]
        with np.load(os.path.join(b["roots"]["stage4_cache_v2"], b["gen4"], "pathology", cache.pathology_filename(i))) as z:
            self.assertEqual((z["maps"].shape, z["maps"].dtype), ((512, 512, 8), np.uint8))
            self.assertEqual(tuple(z["channels"]), ("MA:mean", "MA:max", "HE:mean", "HE:max", "EX:mean", "EX:max",
                                                    "SE:mean", "SE:max"))
            self.assertEqual(str(z["stage4_sha256"]), fx.MODEL_SHA)
            self.assertEqual(str(z["stage3_sha256"]), v2cfg.STAGE3_LWNET_SHA256)
            np.testing.assert_array_equal(z["maps"], fx.fake_maps(fx.fake_native(i)))   # exact 3x3 mean/max path

    def test_completeness_and_bundle_contract(self):
        b = self.b["bundle"]
        self.assertEqual(sorted(b["population"]), self.b["ids"])
        self.assertEqual(b["stage4_sha256"], fx.MODEL_SHA)
        self.assertEqual(b["stage4_generation"], self.b["gen4"])
        self.assertEqual(b["k"], 4)
        self.assertEqual(b["cache_dims"]["pathology"], [512, 512, 8])
        self.assertIn("3x3 block mean+max", b["pooling"])
        self.assertTrue(b["bundle_id"].endswith(self.b["gen4"]))
        self.assertEqual(len(b["fingerprint"]), 64)
        self.assertTrue(self.b["s4"]["canary"]["passed"])

    def test_c2_features_have_the_preregistered_dimensions(self):
        q = next(iter(self.b["qfeat"].values()))
        v = next(iter(self.b["vfeat"].values()))
        self.assertEqual(q.shape, (8 * 2 * 21,))        # 2K channels x {mean,max} x (1+4+16) cells
        self.assertEqual(v.shape, (1 * 2 * 21,))

    def test_completed_generation_is_immutable(self):
        b = self.b
        with self.assertRaises(ac.AptosCacheError):
            ac.generate_stage4_maps(None, fx.MODEL_SHA, b["ids"], fx.fake_native, v2cfg.STAGE3_LWNET_SHA256,
                                    roots=b["roots"], compute_maps=fx.fake_maps, log=lambda *a: None)
        changed = dict(b["s4"], files={**b["s4"]["files"], b["ids"][0]: "f" * 64})
        with self.assertRaises(ac.AptosCacheError):
            ac.write_generation_manifest("stage4_cache_v2", b["gen4"], changed, b["roots"])
        with self.assertRaises(ac.AptosCacheError):
            ac.write_bundle(dict(b["bundle"], fingerprint="0" * 64), b["roots"])

    def test_bundle_staging_to_local_disk_survives_a_drive_drop(self):
        import errno
        import sys
        from unittest import mock
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "colab", "common"))
        import arch1_data as ad
        import dataset_staging
        import stage4_v2_setup
        b = self.b
        with tempfile.TemporaryDirectory() as local:
            local_roots = {k: os.path.join(local, os.path.basename(v)) for k, v in b["roots"].items()}
            real, calls = dataset_staging._copy_one, {"n": 0}

            def flaky(src, dst, *a, **k):
                calls["n"] += 1
                if calls["n"] == 4:
                    raise OSError(errno.ENOTCONN, "Transport endpoint is not connected")
                return real(src, dst, *a, **k)
            with mock.patch.object(dataset_staging, "_copy_one", side_effect=flaky):
                res = stage4_v2_setup.stage_bundle(b["bundle"]["bundle_id"], b["roots"], local_roots,
                                                   remount=lambda: None, log=lambda *a: None)
            self.assertEqual(sum(r["remounts"] for r in res.values()), 1)
            bundle = ad.Arch1Bundle(expected_bundle_id=b["bundle"]["bundle_id"], expected_split_sha256=v2cfg.SPLIT_SHA256,
                                    expected_stage4_sha256=fx.MODEL_SHA, roots=local_roots, expected_population=5)
            bundle.require(bundle.population)
            again = stage4_v2_setup.stage_bundle(b["bundle"]["bundle_id"], b["roots"], local_roots,
                                                 remount=lambda: None, log=lambda *a: None)
            self.assertEqual(sum(r["copied"] for r in again.values()), 0)          # resume: all verified, none recopied

    def test_stale_or_mismatched_files_rejected(self):
        b = self.b
        bad_sha = dict(b["sha4"], **{b["ids"][1]: "e" * 64})
        with self.assertRaises(cache.ManifestMismatchError):
            ac.verify_stage4_generation(b["ids"], b["s4"]["stage4_sha256"], v2cfg.STAGE3_LWNET_SHA256, bad_sha,
                                        roots=b["roots"], workers=1)
        with self.assertRaises(cache.IncompleteCacheError):           # no generation for another model
            ac.verify_stage4_generation(b["ids"], "c" * 64, v2cfg.STAGE3_LWNET_SHA256, b["sha4"], roots=b["roots"])
        with self.assertRaises(cache.ManifestMismatchError):          # wrong Stage-3 lineage inside the files
            ac.verify_stage4_generation(b["ids"], fx.MODEL_SHA, "c" * 64, b["sha4"], roots=b["roots"], workers=1)


class BoundedPrefetchTests(unittest.TestCase):
    """The OOM fix: at most `max_ahead` images may be loaded (or loading) ahead of consumption."""

    def _instrumented(self, delay=0.0, fail_at=None):
        import threading
        import time
        state = {"started": 0, "consumed": 0, "max_ahead_seen": 0}
        lock = threading.Lock()

        def load(key):
            with lock:
                state["started"] += 1
                state["max_ahead_seen"] = max(state["max_ahead_seen"], state["started"] - state["consumed"])
            if fail_at is not None and key == fail_at:
                raise OSError(f"cannot read {key}")
            time.sleep(delay)
            return f"image-{key}"

        def consumed():
            with lock:
                state["consumed"] += 1
        return load, consumed, state

    def test_outstanding_items_are_bounded_with_a_slow_consumer(self):
        import time
        load, consumed, state = self._instrumented()
        for key, item in ac.bounded_prefetch(range(200), load, max_ahead=3, readers=2):
            time.sleep(0.002)                        # GPU slower than the readers
            consumed()
        self.assertEqual(state["started"], 200)
        self.assertLessEqual(state["max_ahead_seen"], 3)

    def test_order_and_ids_preserved_when_loads_finish_out_of_order(self):
        import random
        import time
        rnd = random.Random(0)

        def load(key):
            time.sleep(rnd.random() * 0.005)
            return f"image-{key}"
        out = list(ac.bounded_prefetch([f"{k:012x}" for k in range(60)], load, max_ahead=4, readers=4))
        self.assertEqual([k for k, _ in out], [f"{k:012x}" for k in range(60)])
        self.assertTrue(all(v == f"image-{k}" for k, v in out))

    def test_reader_exception_propagates_in_order_and_stops_reading(self):
        load, consumed, state = self._instrumented(delay=0.001, fail_at=10)
        seen = []
        with self.assertRaises(OSError):
            for key, _ in ac.bounded_prefetch(range(100), load, max_ahead=4, readers=2):
                seen.append(key)
                consumed()
        self.assertEqual(seen, list(range(10)))       # everything before the failing key, in order
        self.assertLessEqual(state["started"], 10 + 1 + 4)

    def test_early_exit_cancels_outstanding_reads(self):
        load, consumed, state = self._instrumented(delay=0.002)
        gen = ac.bounded_prefetch(range(100), load, max_ahead=3, readers=1)
        for n, _ in enumerate(gen):
            consumed()
            if n == 4:
                break
        gen.close()
        self.assertLessEqual(state["started"], 5 + 3)
        with self.assertRaises(ValueError):
            list(ac.bounded_prefetch([1], load, max_ahead=0))

    def test_defaults_are_conservative(self):
        self.assertLessEqual(v2cfg.STAGE4_CACHE_PREFETCH, 8)
        self.assertLessEqual(v2cfg.STAGE4_CACHE_READERS, v2cfg.STAGE4_CACHE_PREFETCH)
        self.assertLessEqual(v2cfg.STAGE4_CACHE_MAX_PENDING_WRITES, 16)

    def test_generation_bounds_reads_and_writes_and_keeps_ids(self):
        import threading
        import time
        with tempfile.TemporaryDirectory() as tmp:
            roots = {k: tmp for k in ("stage2_rgb_v2", "stage3_cache_v2", "stage4_cache_v2", "bundle_v2")}
            ids = [f"{k:012x}" for k in range(1, 25)]
            lock = threading.Lock()
            st = {"read": 0, "computed": 0, "written": 0, "max_ahead": 0, "max_pending_writes": 0}
            real_write = cache.write_pathology_npz

            def load(i):
                with lock:
                    st["read"] += 1
                    st["max_ahead"] = max(st["max_ahead"], st["read"] - st["computed"])
                return fx.fake_native(i)

            def compute(rgb):
                time.sleep(0.003)
                with lock:
                    st["computed"] += 1
                return fx.fake_maps(rgb)

            def slow_write(*a, **k):
                time.sleep(0.01)                     # Drive slower than inference
                out = real_write(*a, **k)
                with lock:
                    st["written"] += 1
                return out

            from unittest import mock
            with mock.patch.object(cache, "write_pathology_npz", side_effect=slow_write) as w:
                def tracking_compute(rgb):
                    with lock:
                        st["max_pending_writes"] = max(st["max_pending_writes"], st["computed"] - st["written"])
                    return compute(rgb)
                shas = ac.generate_stage4_maps(None, fx.MODEL_SHA, ids, load, v2cfg.STAGE3_LWNET_SHA256, roots=roots,
                                               prefetch=3, readers=2, writers=2, max_pending_writes=4,
                                               compute_maps=tracking_compute, log=lambda *a: None)
                self.assertEqual(w.call_count, len(ids))
            self.assertLessEqual(st["max_ahead"], 3)
            self.assertLessEqual(st["max_pending_writes"], 4 + 1)
            gen = cache.stage4_generation_id(fx.MODEL_SHA, 4)
            for i in ids:                             # every file holds the maps of ITS image
                with np.load(os.path.join(tmp, gen, "pathology", cache.pathology_filename(i))) as z:
                    np.testing.assert_array_equal(z["maps"], fx.fake_maps(fx.fake_native(i)))
                    self.assertEqual(str(z["image_id"]), i)
            self.assertEqual(sorted(shas), sorted(ids))

    def test_reader_failure_fails_generation_and_resume_completes(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = {k: tmp for k in ("stage2_rgb_v2", "stage3_cache_v2", "stage4_cache_v2", "bundle_v2")}
            ids = [f"{k:012x}" for k in range(1, 9)]
            bad = ids[5]

            def flaky_load(i):
                if i == bad:
                    raise OSError("[Errno 107] Transport endpoint is not connected")
                return fx.fake_native(i)
            with self.assertRaises(OSError):
                ac.generate_stage4_maps(None, fx.MODEL_SHA, ids, flaky_load, v2cfg.STAGE3_LWNET_SHA256, roots=roots,
                                        prefetch=2, readers=1, compute_maps=fx.fake_maps, log=lambda *a: None)
            gen = cache.stage4_generation_id(fx.MODEL_SHA, 4)
            with open(os.path.join(tmp, gen, "progress.json"), encoding="utf-8") as fh:
                done = json.load(fh)["files"]
            self.assertEqual(sorted(done), sorted(ids[:5]))           # completed writes recorded, none for bad
            shas = ac.generate_stage4_maps(None, fx.MODEL_SHA, ids, fx.fake_native, v2cfg.STAGE3_LWNET_SHA256,
                                           roots=roots, compute_maps=fx.fake_maps, log=lambda *a: None)
            self.assertEqual(sorted(shas), sorted(ids))
            for i in ids:
                self.assertEqual(shas[i], done.get(i, shas[i]))       # resumed files were not rewritten


class GuardTests(unittest.TestCase):
    def test_parity_enforced(self):
        with self.assertRaises(cache.StaleCacheError):
            ac.run_parity(["x"], lambda i: np.zeros((4, 4)), lambda i: np.full((4, 4), 2e-4), 1e-4, "stage3")
        ok = ac.run_parity(["x"], lambda i: np.zeros((4, 4)), lambda i: np.full((4, 4), 5e-5), 1e-4, "stage3")
        self.assertTrue(ok["passed"])

    def test_bundle_refuses_failed_parity_and_missing_model_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = fx.build(tmp)
            for broken in (dict(b["s3"], parity={"passed": False}), ):
                with self.assertRaises(ac.AptosCacheError):
                    ac.bundle_manifest_v2(b["s2"], broken, b["s4"], b["population"], expected_population=5)
            with self.assertRaises(ac.AptosCacheError):
                ac.bundle_manifest_v2(dict(b["s2"], parity={"passed": False}), b["s3"], b["s4"], b["population"],
                                      expected_population=5)
            with self.assertRaises(cache.IncompleteCacheError):
                ac.bundle_manifest_v2(b["s2"], b["s3"], b["s4"], b["population"])     # 5 != 3651

    def test_partial_generation_of_another_model_refused_and_resume_same_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            roots = {"stage2_rgb_v2": tmp, "stage3_cache_v2": tmp, "stage4_cache_v2": tmp, "bundle_v2": tmp}
            ids = [i for i, _ in fx.TRAIN]
            gen = cache.stage4_generation_id(fx.MODEL_SHA, 4)
            calls = []

            def counting(rgb):
                calls.append(1)
                if len(calls) == 2:
                    raise RuntimeError("runtime died")
                return fx.fake_maps(rgb)
            with self.assertRaises(RuntimeError):
                ac.generate_stage4_maps(None, fx.MODEL_SHA, ids, fx.fake_native, v2cfg.STAGE3_LWNET_SHA256,
                                        roots=roots, prefetch=1, writers=1, compute_maps=counting, log=lambda *a: None)
            progress = os.path.join(tmp, gen, "progress.json")
            with open(progress, encoding="utf-8") as fh:
                p = json.load(fh)
            p["stage4_sha256"] = "d" * 64
            with open(progress, "w", encoding="utf-8") as fh:
                json.dump(p, fh)
            with self.assertRaises(ac.AptosCacheError):
                ac.generate_stage4_maps(None, fx.MODEL_SHA, ids, fx.fake_native, v2cfg.STAGE3_LWNET_SHA256,
                                        roots=roots, compute_maps=fx.fake_maps, log=lambda *a: None)
            p["stage4_sha256"] = fx.MODEL_SHA
            with open(progress, "w", encoding="utf-8") as fh:
                json.dump(p, fh)
            shas = ac.generate_stage4_maps(None, fx.MODEL_SHA, ids, fx.fake_native, v2cfg.STAGE3_LWNET_SHA256,
                                           roots=roots, prefetch=1, writers=1, compute_maps=fx.fake_maps,
                                           log=lambda *a: None)
            self.assertEqual(sorted(shas), sorted(ids))

    def test_legacy_sources_and_targets(self):
        with self.assertRaises(cache.LegacyArtifactError):
            cache.assert_legacy_rgb_source("/x/APTOS_00000000000a_lesion_512x512.npy")
        cache.assert_legacy_rgb_source("/x/APTOS_00000000000a_rgb_512x512.npy")
        with self.assertRaises(cache.DenyListedModelError):
            ac.generate_stage4_maps(None, v2cfg.LEGACY_STAGE4_SHA256, ["00000000000a"], fx.fake_native, "a" * 64,
                                    compute_maps=fx.fake_maps)
        with self.assertRaises(ac.AptosCacheError):
            ac.copy_reused_generation("stage4_cache_v2", ["00000000000a"], "/x", "/y")    # Stage 4 is never copied
        with tempfile.TemporaryDirectory() as tmp:
            dest = os.path.join(v2cfg.LEGACY_DIRS[0], "x.json")
            with self.assertRaises(cache.LegacyArtifactError):
                ac._write_json(dest, {})

    def test_gate_requirement(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ac.AptosCacheError):
                ac.require_gate(os.path.join(tmp, "none.json"), fx.MODEL_SHA)
            path = os.path.join(tmp, "idrid_test_gate.json")
            for passed in (True, False):
                with open(path, "w") as fh:
                    json.dump({"model_sha256": fx.MODEL_SHA, "decision": {"PASS": passed}}, fh)
                if passed:
                    self.assertTrue(ac.require_gate(path, fx.MODEL_SHA)["PASS"])
                else:
                    with self.assertRaises(ac.AptosCacheError):
                        ac.require_gate(path, fx.MODEL_SHA)
                    self.assertEqual(ac.require_gate(path, fx.MODEL_SHA, v2cfg.STAGE4_GATE_OVERRIDE_TOKEN)["override"],
                                     v2cfg.STAGE4_GATE_OVERRIDE_TOKEN)
            with self.assertRaises(ac.AptosCacheError):
                ac.require_gate(path, "c" * 64)                     # report of another model


if __name__ == "__main__":
    unittest.main()
