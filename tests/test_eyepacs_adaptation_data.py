"""CPU tests for eyepacs_adaptation_data. Synthetic label files and fake frames only: no EyePACS image is
decoded, no APTOS file and no EyeQ label is read."""
import csv
import inspect
import json
import os
import tempfile
import unittest

import numpy as np

import eyepacs_adaptation_data as ea


def synthetic_labels(path, n_patients=400, seed=0, with_blank=True):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(1, n_patients + 1):
        base = rng.choice(5, p=[0.73, 0.07, 0.15, 0.03, 0.02])
        for eye in ("left", "right"):
            grade = int(np.clip(base + rng.choice([-1, 0, 0, 0, 1]), 0, 4))
            rows.append((f"{p}_{eye}", grade))
    if with_blank:                                                        # the four recorded blank images must exist
        blank_patients = {name.split("_")[0] for name in ea.BLANK_TRAINING_IMAGES}
        rows = [r for r in rows if r[0].split("_")[0] not in blank_patients]
        for name in ea.BLANK_TRAINING_IMAGES:
            stem = name[:-5]
            patient, eye = stem.split("_")
            other = "right" if eye == "left" else "left"
            rows += [(stem, 0), (f"{patient}_{other}", 0)]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "level"])
        w.writerows(rows)
    return rows


class InventoryTests(unittest.TestCase):
    def test_names_are_parsed_strictly(self):
        self.assertEqual(ea.parse_name("10_left.jpeg"), ("10", "left"))
        self.assertEqual(ea.parse_name("44349_right"), ("44349", "right"))
        for bad in ("10_centre.jpeg", "abc_left.jpeg", "007-2809-100.jpg", "0a1b2c3d4e5f.png"):
            with self.assertRaises(ValueError):
                ea.parse_name(bad)

    def test_exactly_the_four_recorded_blank_images_are_excluded(self):
        self.assertEqual(ea.BLANK_TRAINING_IMAGES, ("1986_left.jpeg", "32253_right.jpeg", "34689_left.jpeg", "43457_left.jpeg"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trainLabels.csv")
            rows = synthetic_labels(path)
            usable, report = ea.inventory(path)
            self.assertEqual(report["labelled"], len(rows))
            self.assertEqual(report["usable"], len(rows) - 4)
            self.assertEqual(report["excluded_blank"], list(ea.BLANK_TRAINING_IMAGES))
            self.assertFalse({u["image"] for u in usable} & set(ea.BLANK_TRAINING_IMAGES))
            self.assertEqual(report["patients_usable"], report["patients_labelled"])   # their fellow eyes stay
            synthetic_labels(path, with_blank=False)
            with self.assertRaises(ValueError):
                ea.inventory(path)                                         # the recorded images must be present

    def test_label_file_and_folder_must_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trainLabels.csv")
            rows = synthetic_labels(path, n_patients=6)
            folder = os.path.join(tmp, "train")
            os.makedirs(folder)
            for name, _ in rows:
                open(os.path.join(folder, name + ".jpeg"), "w").close()
            ea.inventory(path, folder)
            os.remove(os.path.join(folder, rows[0][0] + ".jpeg"))
            with self.assertRaises(ValueError):
                ea.inventory(path, folder)

    def test_no_quality_label_or_stage1_dependency(self):
        source = inspect.getsource(ea)
        for forbidden in ("Label_EyeQ", "quality_assessment", "iqa", "import pl_convnext", "DOWNSTREAM_SPLIT", "load_split"):
            self.assertNotIn(forbidden, source)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(self.tmp.name, "trainLabels.csv")
        synthetic_labels(path, n_patients=3000)
        self.usable, _ = ea.inventory(path)
        self.rows = ea.build_split(self.usable)

    def tearDown(self):
        self.tmp.cleanup()

    def test_patient_level_stratified_and_deterministic(self):
        rows = self.rows
        split_of = {}
        for r in rows:
            split_of.setdefault(r["patient"], set()).add(r["split"])
        self.assertTrue(all(len(v) == 1 for v in split_of.values()))       # both eyes together, always
        report = ea.split_report(rows)
        self.assertEqual(report["patients_in_both_splits"], 0)
        self.assertEqual(report["images"]["train"] + report["images"]["val"], len(self.usable))
        for g in range(5):                                                 # 10 % of the patients of every stratum
            tr, va = report["patient_max_grade_counts"]["train"][g], report["patient_max_grade_counts"]["val"][g]
            self.assertEqual(va, int(round(0.1 * (tr + va))))
            self.assertGreater(va, 0)
        for r in rows:                                                     # the stratum is the higher of the two eyes
            self.assertEqual(r["patient_max_grade"], max(x["grade"] for x in self.usable if x["patient"] == r["patient"]))
            break
        self.assertEqual({r["image"]: r["split"] for r in ea.build_split(list(reversed(self.usable)))},
                         {r["image"]: r["split"] for r in rows})           # independent of row order
        self.assertEqual({r["image"]: r["split"] for r in ea.build_split(self.usable)}, {r["image"]: r["split"] for r in rows})
        other = ea.build_split(self.usable, seed=1)
        self.assertNotEqual([r["split"] for r in other], [r["split"] for r in rows])
        self.assertEqual((ea.SPLIT_SEED, ea.VAL_FRACTION), (20261008, 0.10))

    def test_a_patient_in_both_splits_is_refused(self):
        broken = [dict(r) for r in self.rows]
        first = broken[0]
        partner = next(r for r in broken if r["patient"] == first["patient"] and r is not first)
        partner["split"] = "val" if first["split"] == "train" else "train"
        with self.assertRaises(ValueError):
            ea.split_report(broken)

    def test_manifest_is_written_once_and_is_immutable(self):
        out = os.path.join(self.tmp.name, "splits")
        path, summary = ea.write_split(self.rows, out)
        self.assertEqual(len(summary["manifest_sha256"]), 64)
        self.assertEqual(summary["quality_filter"], "none (no EyeQ label and no Stage-1 prediction is used)")
        again_path, again = ea.write_split(self.rows, out)                 # identical content: accepted
        self.assertEqual(again["manifest_sha256"], summary["manifest_sha256"])
        with self.assertRaises(RuntimeError):
            ea.write_split(ea.build_split(self.usable, seed=1), out)       # another split: refused
        self.assertEqual(ea.sha256_file(path), summary["manifest_sha256"]) # and the file is unchanged
        back = ea.read_split(out, summary["manifest_sha256"])
        self.assertEqual({r["image"]: r["split"] for r in back}, {r["image"]: r["split"] for r in self.rows})
        with self.assertRaises(RuntimeError):
            ea.read_split(out, "0" * 64)
        counts, weights = ea.class_weights(self.rows)
        self.assertEqual(counts, summary["grade_counts"]["train"])          # training part only
        self.assertEqual(len(weights), 5)
        self.assertAlmostEqual(sum(c / sum(counts) * w for c, w in zip(counts, weights)), 1.0, places=9)


class CacheTests(unittest.TestCase):
    def test_only_eyepacs_training_files_may_enter_the_cache(self):
        self.assertTrue(ea.assert_not_aptos(os.path.join("datasets", "EyePACS", "raw", "train", "10_left.jpeg")))
        for bad in (os.path.join("datasets", "APTOS2019", "raw", "train_images", "000c1434d8d7.png"),
                    os.path.join("datasets", "APTOS2019", "raw", "10_left.jpeg"),
                    os.path.join("datasets", "EyePACS", "raw", "train", "000c1434d8d7.png")):
            with self.assertRaises((RuntimeError, ValueError)):
                ea.assert_not_aptos(bad)

    def test_shards_are_resumable_and_bound_to_the_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = os.path.join(tmp, "trainLabels.csv")
            synthetic_labels(labels, n_patients=30)
            usable, _ = ea.inventory(labels)
            rows = ea.build_split(usable)
            calls = []
            original = ea.frame_from_raw

            def fake(path):
                calls.append(os.path.basename(path))
                patient, _ = ea.parse_name(os.path.basename(path))
                return np.full((512, 512, 3), (int(patient) % 97) / 97.0, np.float32)
            ea.frame_from_raw = fake
            os.makedirs(os.path.join(tmp, "train"))
            for r in rows:                                                 # source files: hashed into the index
                with open(os.path.join(tmp, "train", r["image"]), "w") as fh:
                    fh.write(r["image"])
            try:
                cache = os.path.join(tmp, "cache")
                index = ea.build_cache(rows, os.path.join(tmp, "train"), cache, "a" * 64, workers=2, shard_size=16, log=lambda *a: None)
                plan = ea.shard_plan(rows, 16)
                self.assertEqual(len(index["shards"]), len(plan["train"]) + len(plan["val"]))
                self.assertEqual(sorted(calls), sorted(r["image"] for r in rows))  # every usable image once, no blank one
                name = "frames_train_000.npy"
                block = np.load(os.path.join(cache, name))
                self.assertEqual((block.dtype, block.shape), (np.float16, (16, 512, 512, 3)))
                self.assertEqual(index["shards"][name]["images"], plan["train"][0])
                self.assertEqual(ea.sha256_file(os.path.join(cache, name)), index["shards"][name]["sha256"])
                patient, _ = ea.parse_name(plan["train"][0][3])
                self.assertAlmostEqual(float(block[3, 0, 0, 0]), (int(patient) % 97) / 97.0, places=3)
                n = len(calls)
                ea.build_cache(rows, os.path.join(tmp, "train"), cache, "a" * 64, workers=2, shard_size=16, log=lambda *a: None)
                self.assertEqual(len(calls), n)                            # nothing recomputed
                with self.assertRaises(RuntimeError):
                    ea.build_cache(rows, os.path.join(tmp, "train"), cache, "b" * 64, workers=2, shard_size=16, log=lambda *a: None)
                train_names = [x for s in plan["train"] for x in s]
                val_names = [x for s in plan["val"] for x in s]
                self.assertFalse(set(train_names) & set(val_names))
                self.assertEqual(index["shards"][name]["source_sha256"][3],
                                 ea.sha256_file(os.path.join(tmp, "train", plan["train"][0][3])))
                self._verification(tmp, cache, rows, plan)
            finally:
                ea.frame_from_raw = original

    def _verification(self, tmp, cache, rows, plan):
        """verify_cache / FrameCache accept exactly the complete cache of this split and identity."""
        index, report = ea.verify_cache(cache, rows, "a" * 64, shard_size=16)
        self.assertEqual(report["frames"], {"train": sum(map(len, plan["train"])), "val": sum(map(len, plan["val"]))})
        self.assertEqual(report["identity"]["preproc_version"], "stage2-DR-profile/full-frame-direct-resize/v1")
        self.assertEqual((report["identity"]["dtype"], report["hashes_checked"]), ("float16", True))
        frames = ea.FrameCache(cache, rows, "a" * 64, shard_size=16)
        self.assertEqual([i for i, _ in frames.entries("train")], [x for s in plan["train"] for x in s])
        grade_of = {r["image"]: r["grade"] for r in rows}
        self.assertTrue(all(grade_of[i] == g for split in ("train", "val") for i, g in frames.entries(split)))
        image = plan["val"][0][1]
        frame = frames.frame(image)
        self.assertEqual((frame.dtype, frame.shape), (np.float32, (512, 512, 3)))
        self.assertAlmostEqual(float(frame[0, 0, 0]), (int(ea.parse_name(image)[0]) % 97) / 97.0, places=3)
        frames.close()
        for wrong in (dict(manifest_sha256="b" * 64), dict(shard_size=8)):   # another manifest / another layout
            with self.assertRaises(RuntimeError):
                ea.verify_cache(cache, rows, wrong.get("manifest_sha256", "a" * 64), shard_size=wrong.get("shard_size", 16))
        original_version = ea.CACHE_VERSION
        try:                                                               # another preprocessing version
            ea.CACHE_VERSION = "eyepacs-frames-v2"
            with self.assertRaises(RuntimeError):
                ea.verify_cache(cache, rows, "a" * 64, shard_size=16)
        finally:
            ea.CACHE_VERSION = original_version
        moved = [dict(r, split="val" if r["split"] == "train" else "train") for r in rows]
        with self.assertRaises(RuntimeError):                              # another split of the same images
            ea.verify_cache(cache, moved, "a" * 64, shard_size=16)
        regraded = [dict(r, grade=(r["grade"] + 1) % 5) for r in rows]
        with self.assertRaises(RuntimeError):                              # other labels
            ea.verify_cache(cache, regraded, "a" * 64, shard_size=16)
        shard = os.path.join(cache, "frames_val_000.npy")
        block = np.load(shard)
        block[0, 0, 0, 0] += 0.5
        np.save(shard, block)                                              # same size, other content
        ea.verify_cache(cache, rows, "a" * 64, shard_size=16, check_hashes=False)
        with self.assertRaises(RuntimeError):
            ea.verify_cache(cache, rows, "a" * 64, shard_size=16)
        with self.assertRaises(RuntimeError):
            ea.FrameCache(cache, rows, "a" * 64, shard_size=16)
        os.remove(shard)
        with self.assertRaises(RuntimeError):                              # incomplete cache
            ea.verify_cache(cache, rows, "a" * 64, shard_size=16, check_hashes=False)
        with self.assertRaises(RuntimeError):
            ea.verify_cache(os.path.join(tmp, "no_cache"), rows, "a" * 64)

    def test_staging_copies_every_shard_by_its_recorded_hash_and_the_index_last(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "drive"), os.path.join(tmp, "local")
            os.makedirs(src)
            index = {"shards": {"frames_train_000.npy": {"sha256": "1" * 64}, "frames_val_000.npy": {"sha256": "2" * 64}}}
            with open(os.path.join(src, "index.json"), "w") as fh:
                json.dump(index, fh)
            seen = {}

            def stage_files(pairs, log=None, label=None):
                seen["pairs"] = pairs
                seen["index_present"] = os.path.exists(os.path.join(dst, "index.json"))
                return {"copied": len(pairs), "skipped": 0}
            self.assertEqual(ea.stage_cache(src, dst, log=lambda *a: None, stage_files=stage_files)["copied"], 2)
            self.assertEqual([(os.path.basename(a), os.path.basename(b), c) for a, b, c in seen["pairs"]],
                             [("frames_train_000.npy", "frames_train_000.npy", "1" * 64), ("frames_val_000.npy", "frames_val_000.npy", "2" * 64)])
            self.assertFalse(seen["index_present"])
            self.assertTrue(os.path.exists(os.path.join(dst, "index.json")))


class PreprocessingParityTests(unittest.TestCase):
    def test_frame_is_the_locked_stage2_and_512_path_used_for_aptos_idrid_and_ddr(self):
        import cv2

        import ddr_probe
        import stage4_v2_aptos_cache as ac
        import stage4_v2_data as sd
        with tempfile.TemporaryDirectory() as tmp:
            yy, xx = np.mgrid[:600, :800]
            disc = ((yy - 300) ** 2 + (xx - 400) ** 2) < 280 ** 2
            image = np.zeros((600, 800, 3), np.uint8)
            image[disc] = (40, 90, 170)
            image[280:320, 380:420] = (90, 200, 240)
            image = np.clip(image + np.random.default_rng(0).integers(0, 12, image.shape), 0, 255).astype(np.uint8)
            path = os.path.join(tmp, "10_left.jpeg")
            cv2.imwrite(path, image)
            frame = ea.frame_from_raw(path)
            self.assertEqual((frame.dtype, frame.shape), (np.float32, (512, 512, 3)))
            self.assertTrue(0.0 <= frame.min() and frame.max() <= 1.0)
            np.testing.assert_array_equal(frame, ac.recompute_rgb_512(sd.stage2_rgb(path)))   # P's path, bit for bit
            np.testing.assert_array_equal(frame, ddr_probe.frame_from_raw(path))              # and the DDR probe's
            np.testing.assert_array_equal(frame, ea.frame_from_raw(path))                     # deterministic
            self.assertLessEqual(float(np.abs(frame.astype(np.float16).astype(np.float32) - frame).max()), 2.5e-4)
        self.assertEqual(inspect.getsource(ea.frame_from_raw).count("recompute_rgb_512(sd.stage2_rgb(raw_path))"), 1)


if __name__ == "__main__":
    unittest.main()
