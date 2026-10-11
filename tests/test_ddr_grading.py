"""CPU tests for ddr_grading_data and ddr_grading_eval. Synthetic lists, fabricated images, synthetic tables and fake
models only: no grader is loaded, no DDR grading image is decoded, no IDRiD file is opened and nothing here is a
DDR result. The tests that read the real label list, lesion manifest and mirror metadata (text files) are skipped
when those git-ignored files are absent."""
import copy
import inspect
import io
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pandas as pd

import arch1_posthoc as ph
import arch1_train as at
import ddr_grading_data as dd
import ddr_grading_eval as ev
import idrid_batch1_eval as b1
import idrid_batch2_eval as b2
import idrid_grading_eval as ig

HAVE_SOURCES = all(os.path.exists(p) for p in (dd.LABEL_FILE, dd.LESION_MANIFEST, dd.FILE_METADATA,
                                               os.path.join(dd.AUDIT_DIR, dd.PRIMARY_LIST), os.path.join(dd.AUDIT_DIR, dd.BROADER_LIST)))
LOCKED = {"idrid_grading_protocol.json": "f93c1237eb81ab20397dea8bc6d9079871d6308578304f793dfabee17938587e",
          "idrid_batch1_protocol.json": "656dde8932ebaa2c42b39ce9fb54d6f245f7d12f643a4d811492216c5f0b3ac0",
          "idrid_batch2_protocol.json": "5ea937c4edb8baef9458ef996beef92f80c7eb969a0a45a8f0a0aaa636cc073a"}


def h(text):
    return __import__("hashlib").sha256(text.encode()).hexdigest()


def toy_sources():
    """A small test list with every case the rules name. Content is identified by the hash of a label string."""
    labels = {"20170413100000000.jpg": 0, "20170413100030000.jpg": 0, "20170413100500000.jpg": 0, "20170414090000000.jpg": 2,
              "20170414090010000.jpg": 5, "007-0001-100.jpg": 1, "007-0002-300.jpg": 2, "007-0003-400.jpg": 4, "007-0004-400.jpg": 4,
              "007-0005-300.jpg": 3, "007-0006-300.jpg": 2, "20170415080000000.jpg": 0}
    content = {n: n for n in labels}
    content["20170413100500000.jpg"] = "20170413100000000.jpg"       # a byte-identical copy of the first image (grade 0)
    content["20170415080000000.jpg"] = "20170413100000000.jpg"       # and a third member of the same group
    content["007-0006-300.jpg"] = "lesion-content"                   # the content of a lesion image under another name
    metadata = {n: {"sha256": h(content[n]), "bytes": 1000 + i} for i, n in enumerate(sorted(labels))}
    lesion = {"007-0002-300.jpg": {"sha256": h("007-0002-300.jpg"), "split": "test"},
              "007-9999-300.jpg": {"sha256": h("lesion-content"), "split": "train"}}
    return labels, lesion, metadata


class PopulationTests(unittest.TestCase):
    def test_exclusions_in_the_protocols_order(self):
        labels, lesion, metadata = toy_sources()
        population = dd.build_population(labels, lesion, metadata)
        primary = [r["image"] for r in population["primary"]]
        self.assertEqual(primary, sorted(["20170413100000000.jpg", "20170413100030000.jpg", "20170414090000000.jpg", "007-0001-100.jpg",
                                          "007-0003-400.jpg", "007-0004-400.jpg", "007-0005-300.jpg"]))
        self.assertNotIn("20170414090010000.jpg", primary)             # ungradable
        self.assertNotIn("007-0002-300.jpg", primary)                  # lesion subset, by name
        self.assertNotIn("007-0006-300.jpg", primary)                  # lesion subset, by content under another name
        x = population["exclusions"]
        self.assertEqual((x["test_list"], x["ungradable"], x["gradable"], x["lesion_subset_by_name"], x["lesion_subset_by_content"],
                          x["after_lesion_exclusion"], x["duplicate_groups"], x["duplicate_extras_removed"]), (12, 1, 11, 1, 1, 9, 1, 2))
        self.assertEqual([r["image"] for r in population["broader"]], sorted(primary + ["007-0002-300.jpg"]))
        self.assertEqual([r["lesion_subset"] for r in population["broader"] if r["image"] == "007-0002-300.jpg"], [1])
        self.assertTrue(all(r["grade"] in range(5) for r in population["broader"]))

    def test_the_smallest_file_name_of_a_duplicate_group_is_kept(self):
        labels, lesion, metadata = toy_sources()
        population = dd.build_population(labels, lesion, metadata)
        self.assertEqual(population["duplicate_groups"], {"20170413100000000.jpg": ["20170413100500000.jpg", "20170415080000000.jpg"]})
        shuffled = dict(reversed(list(labels.items())))                # the rule does not depend on the order of the list
        self.assertEqual(dd.list_text(dd.build_population(shuffled, lesion, metadata)["primary"]), dd.list_text(population["primary"]))
        hashes = [r["sha256"] for r in population["broader"]]
        self.assertEqual(len(set(hashes)), len(hashes))

    def test_inconsistent_sources_are_refused(self):
        labels, lesion, metadata = toy_sources()
        with self.assertRaises(dd.PopulationError):                    # copies with different grades
            dd.build_population(dict(labels, **{"20170413100500000.jpg": 2}), lesion, metadata)
        with self.assertRaises(dd.PopulationError):                    # a listed image the mirror does not hold
            dd.build_population(dict(labels, **{"007-0007-100.jpg": 1}), lesion, metadata)
        with self.assertRaises(dd.PopulationError):                    # a lesion image that is not the grading file of its name
            dd.build_population(labels, dict(lesion, **{"007-0002-300.jpg": {"sha256": h("other"), "split": "test"}}), metadata)
        with self.assertRaises(dd.PopulationError):
            dd.build_population(dict(labels, **{"007-0001-100.jpg": 7}), lesion, metadata)
        with self.assertRaises(dd.PopulationError):
            dd.build_population(labels, lesion, metadata, extra_exclusions={"not-listed.jpg": "x"})

    def test_a_later_version_can_remove_named_images_without_touching_version_one(self):
        labels, lesion, metadata = toy_sources()
        v1 = dd.build_population(labels, lesion, metadata)
        v2 = dd.build_population(labels, lesion, metadata, extra_exclusions={"007-0003-400.jpg": "copy band"})
        self.assertEqual(len(v2["primary"]), len(v1["primary"]) - 1)
        self.assertEqual(v2["exclusions"]["near_duplicate_or_other"], ["007-0003-400.jpg"])
        self.assertEqual(v1["exclusions"]["near_duplicate_or_other"], [])
        self.assertNotEqual(dd.list_text(v1["primary"]), dd.list_text(v2["primary"]))

    def test_lists_round_trip_and_hash_deterministically(self):
        population = dd.build_population(*toy_sources())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.csv")
            first = dd.write_list(path, population["primary"])
            self.assertEqual(dd.read_list(path), population["primary"])
            self.assertEqual(dd.write_list(path, population["primary"]), first)
            self.assertNotIn(b"\r", open(path, "rb").read())

    def test_families_and_file_time_chains(self):
        self.assertEqual((dd.family("007-0001-100.jpg"), dd.family("20170413100000000.jpg")), ("007", "timestamp"))
        with self.assertRaises(dd.PopulationError):
            dd.family("IDRiD_001.jpg")
        names = ["20170413100000000.jpg", "007-0001-100.jpg", "20170413100059000.jpg", "20170413100205000.jpg", "20170413100200000.jpg",
                 "007-0002-100.jpg"]
        units = dd.timestamp_chains(names)
        self.assertEqual(units[0], units[2])                           # 59 seconds apart
        self.assertNotEqual(units[2], units[4])                        # 61 seconds after the previous file time
        self.assertEqual(units[3], units[4])
        self.assertEqual(len({units[1], units[5]}), 2)                 # 007 images are single units
        self.assertEqual(dd.CHAIN_GAP_SECONDS, 60)


@unittest.skipUnless(HAVE_SOURCES, "the DDR label list, lesion manifest, mirror metadata or image lists are not present locally")
class RealPopulationTests(unittest.TestCase):
    """The approved population, rebuilt from the real (text) sources. No image is read."""

    @classmethod
    def setUpClass(cls):
        cls.protocol = json.load(open(ev.PROTOCOL_PATH, encoding="utf-8"))
        cls.labels, cls.lesion, cls.metadata = dd.read_labels(), dd.read_lesion_manifest(), dd.read_file_metadata()
        cls.population = dd.build_population(cls.labels, cls.lesion, cls.metadata)

    def test_the_pinned_population_is_reproduced_from_its_sources(self):
        found = dd.verify_population(self.protocol)
        self.assertEqual((found["primary"]["n"], found["primary"]["grades"]), (3540, [1846, 159, 1224, 57, 254]))
        self.assertEqual((found["broader"]["n"], found["broader"]["grades"], found["broader"]["lesion_subset"]), (3725, [1846, 189, 1344, 71, 275], 185))
        self.assertEqual(found["primary"]["by_family"], {"timestamp": {"n": 2765, "grades": [1846, 21, 867, 31, 0]},
                                                         "007": {"n": 775, "grades": [0, 138, 357, 26, 254]}})
        x = found["exclusions"]
        self.assertEqual((x["test_list"], x["ungradable"], x["gradable"], x["lesion_subset_by_name"], x["lesion_subset_by_content"],
                          x["after_lesion_exclusion"], x["duplicate_groups"], x["duplicate_extras_removed"]), (4105, 346, 3759, 185, 0, 3574, 32, 34))
        self.assertEqual(found["duplicate_group_sizes"], {2: 31, 4: 1})
        self.assertEqual(x["lesion_roles"], {"test": 59, "val": 118, "train": 8})

    def test_membership_and_exclusions(self):
        primary = {r["image"] for r in self.population["primary"]}
        broader = {r["image"] for r in self.population["broader"]}
        lesion_hashes = {e["sha256"] for e in self.lesion.values()}
        self.assertFalse(primary & set(self.lesion))
        self.assertFalse({self.metadata[n]["sha256"] for n in primary} & lesion_hashes)
        self.assertFalse([n for n in broader if self.labels[n] == dd.UNGRADABLE])
        self.assertEqual(broader - primary, {n for n in self.lesion if self.labels.get(n, dd.UNGRADABLE) != dd.UNGRADABLE})
        self.assertEqual(len(broader - primary), 185)
        self.assertTrue(primary < broader)
        self.assertEqual(len({self.metadata[n]["sha256"] for n in broader}), len(broader))

    def test_duplicate_groups_keep_the_smallest_name_and_lose_only_grade_zero_copies(self):
        by_hash = {}
        eligible = [n for n, g in self.labels.items() if g != dd.UNGRADABLE and n not in self.lesion]
        for n in eligible:
            by_hash.setdefault(self.metadata[n]["sha256"], []).append(n)
        groups = {min(v): sorted(v)[1:] for v in by_hash.values() if len(v) > 1}
        self.assertEqual(groups, self.population["duplicate_groups"])
        removed = [n for others in groups.values() for n in others]
        primary = {r["image"] for r in self.population["primary"]}
        self.assertEqual((len(groups), len(removed)), (32, 34))
        self.assertFalse(primary & set(removed))
        self.assertTrue(set(groups) <= primary)
        self.assertEqual({self.labels[n] for n in removed}, {0})
        self.assertEqual({dd.family(n) for n in removed}, {"timestamp"})

    def test_a_changed_pin_or_source_is_refused(self):
        for section, change in (("populations", lambda p: p["primary"].update(n=3574)),
                                ("populations", lambda p: p["primary"].update(list_sha256="0" * 64)),
                                ("populations", lambda p: p["exclusion_counts"].update(duplicate_extras_removed=0)),
                                ("populations", lambda p: p["broader"].update(grades_0_to_4=[1880, 189, 1344, 71, 275])),
                                ("data", lambda d: d["label_file"].update(sha256="0" * 64)),
                                ("data", lambda d: d["file_metadata"].update(sha256="0" * 64))):
            spec = copy.deepcopy(self.protocol)
            change(spec[section])
            with self.assertRaises(dd.PopulationError):
                dd.verify_population(spec)

    def test_the_same_visit_proxy_counts_are_the_recorded_ones(self):
        self.assertEqual(dd.describe(self.population)["same_visit_proxy"], self.protocol["populations"]["same_visit_proxy"])


# --------------------------------------------------------------------------- near-duplicate screen and download

def fundus(seed, size=400):
    """A synthetic fundus-like picture: a bright disc on black with smooth structure that depends on `seed`."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(yy - size / 2, xx - size / 2)
    field = sum(a * np.sin(xx / fx + p) * np.cos(yy / fy + q) for a, fx, fy, p, q in rng.uniform([20, 15, 15, 0, 0], [50, 70, 70, 6, 6], (5, 5)))
    image = np.clip(120 + field, 30, 255) * (r < size * 0.45)
    return Image.fromarray(np.stack([image, image * 0.6, image * 0.3], -1).astype(np.uint8))


class ScreenTests(unittest.TestCase):
    def test_the_thresholds_and_the_decision_rule_are_frozen(self):
        self.assertEqual((dd.COPY_BAND, dd.AMBIGUOUS_BAND, dd.SIGNATURE_SIZE), (0.01, 0.06, 32))
        self.assertEqual([dd.band(d) for d in (0.0, 0.0099, 0.01, 0.0599, 0.06, 1.2)],
                         ["copy", "copy", "ambiguous", "ambiguous", "distinct", "distinct"])

    def test_a_resized_recompressed_copy_falls_inside_the_screened_bands_and_another_picture_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = fundus(1), fundus(2)
            a.save(os.path.join(tmp, "a.png"))
            a.resize((300, 300)).save(os.path.join(tmp, "a_copy.jpg"), quality=75)
            b.save(os.path.join(tmp, "b.jpg"), quality=95)
            sig, ok = dd.signatures([os.path.join(tmp, n) for n in ("a.png", "a_copy.jpg", "b.jpg")], workers=1)
            self.assertTrue(ok.all())
            d, j = dd.nearest(sig[:1], sig[1:])
            self.assertLess(d[0], dd.AMBIGUOUS_BAND)                    # a resized, recompressed copy: inside the screened bands,
            self.assertEqual(j[0], 0)                                   # though not always inside the copy band (see positive_control)
            self.assertEqual(dd.nearest(sig[:1], sig[:1])[0][0], 0.0)   # the same file is at distance zero
            self.assertGreater(dd.nearest(sig[:1], sig[2:])[0][0], dd.AMBIGUOUS_BAND)
            self.assertIsNone(dd.signature(os.path.join(tmp, "missing.jpg")))

    def test_the_screen_names_copies_for_exclusion_and_uses_no_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for k, name in enumerate(("007-0001-100.jpg", "007-0002-300.jpg", "20170413100000000.jpg")):
                fundus(10 + k).save(os.path.join(tmp, name), quality=95)
                rows.append({"image": name, "grade": k, "sha256": dd.sha256_file(os.path.join(tmp, name)), "bytes": 1, "family": dd.family(name),
                             "lesion_subset": 0})
            fundus(10).resize((256, 256)).save(os.path.join(tmp, "dev_copy.png"))
            fundus(99).save(os.path.join(tmp, "dev_other.png"))
            ref, _ = dd.signatures([os.path.join(tmp, "dev_copy.png"), os.path.join(tmp, "dev_other.png")], workers=1)
            report = dd.screen(rows, tmp, {"development": (ref, ["dev_copy.png", "dev_other.png"], "f" * 64)}, "a" * 64)
            self.assertEqual(report["exclude"], ["007-0001-100.jpg"])
            self.assertEqual(report["closest"][0]["nearest"], "development:dev_copy.png")
            self.assertIs(report["labels_used"], False)
            self.assertEqual(report["thresholds"], {"copy_band_below": 0.01, "ambiguous_band_below": 0.06})
            self.assertEqual((report["list_sha256"], report["images"], report["against"]["development"]["copy_band"]), ("a" * 64, 3, 1))
            relabelled = [dict(r, grade=4 - r["grade"]) for r in rows]     # the grades play no part
            self.assertEqual(dd.screen(relabelled, tmp, {"development": (ref, ["dev_copy.png", "dev_other.png"], "f" * 64)}, "a" * 64)["closest"],
                             report["closest"])
            with self.assertRaises(dd.PopulationError):                   # a file that is not the listed one
                dd.screen([dict(rows[0], sha256="0" * 64)], tmp, {}, "a" * 64)

    @unittest.skipUnless(os.path.exists(os.path.join(dd.SIGNATURE_CACHE, "sig_aptos_train.npz")) and os.path.exists(dd.ADAPTATION_MANIFEST),
                         "the stored development signatures are not present locally")
    def test_the_reference_sets_are_the_pinned_development_images(self):
        protocol = json.load(open(ev.PROTOCOL_PATH, encoding="utf-8"))
        references = dd.reference_signatures()
        self.assertEqual(set(references), set(protocol["near_duplicate_screen"]["against"]))
        for name, (sig, names, digest) in references.items():
            pinned = protocol["near_duplicate_screen"]["against"][name]
            self.assertEqual((len(names), sig.shape, digest), (pinned["images"], (pinned["images"], 1024), pinned["signature_file_sha256"]))
        self.assertEqual({k: len(v[1]) for k, v in references.items()}, {"aptos_train_and_validation": 3662, "eyepacs_adaptation": 35122})


class OfficialDirectoryTests(unittest.TestCase):
    def test_the_directory_is_read_from_the_last_bytes_of_a_stored_archive(self):
        import zipfile
        import zlib
        files = {"DDR-dataset/DR_grading/test/007-0001-100.jpg": b"a" * 5000, "DDR-dataset/DR_grading/test/20170413100000000.jpg": b"b" * 7000,
                 "DDR-dataset/DR_grading/test.txt": b"007-0001-100.jpg 1\n"}
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            archive.writestr("DDR-dataset/", b"")
            for name, content in files.items():
                archive.writestr(name, content)
        whole = buffer.getvalue()
        tail = whole[-600:]                                              # the images themselves are not in these bytes
        self.assertNotIn(b"a" * 5000, tail)
        entries = dd.parse_zip_directory(tail, len(whole))
        self.assertEqual({e[0]: (e[1], e[2]) for e in entries}, {n: (len(c), f"{zlib.crc32(c) & 0xFFFFFFFF:08x}") for n, c in files.items()})
        self.assertEqual({e[3] for e in entries}, {0})
        with self.assertRaises(dd.PopulationError):                      # too few bytes to hold the directory
            dd.parse_zip_directory(whole[-60:], len(whole))
        with self.assertRaises(dd.PopulationError):
            dd.parse_zip_directory(b"not a zip archive", 17)

    def test_downloaded_files_are_compared_by_size_and_crc_and_the_result_is_not_called_identity(self):
        import zlib
        with tempfile.TemporaryDirectory() as tmp:
            rows, official = [], {}
            for name, content in (("007-0001-100.jpg", b"one"), ("007-0002-300.jpg", b"two"), ("007-0003-400.jpg", b"three")):
                with open(os.path.join(tmp, name), "wb") as fh:
                    fh.write(content)
                rows.append({"image": name, "sha256": dd.sha256_file(os.path.join(tmp, name))})
                official[name] = (len(content), f"{zlib.crc32(content) & 0xFFFFFFFF:08x}")
            self.assertEqual(dd.crc32_file(os.path.join(tmp, "007-0001-100.jpg")), official["007-0001-100.jpg"][1])
            clean = dd.official_check(rows, tmp, official, "a" * 64, "b" * 64)
            self.assertEqual((clean["images"], clean["size_and_crc32_equal"], clean["mismatches"], clean["not_in_official_directory"]), (3, 3, [], []))
            self.assertIn("not a proof of byte identity", clean["meaning"])
            official["007-0002-300.jpg"] = (3, "00000000")
            del official["007-0003-400.jpg"]
            report = dd.official_check(rows, tmp, official, "a" * 64, "b" * 64)
            self.assertEqual((report["size_and_crc32_equal"], report["mismatches"], report["not_in_official_directory"]),
                             (1, ["007-0002-300.jpg"], ["007-0003-400.jpg"]))
            with self.assertRaises(dd.PopulationError):                  # the file on disk is not the listed one
                dd.official_check([dict(rows[0], sha256="0" * 64)], tmp, official, "a" * 64, "b" * 64)

    @unittest.skipUnless(os.path.exists(dd.OFFICIAL_DIRECTORY) and HAVE_SOURCES, "the official archive directory is not present locally")
    def test_the_official_directory_names_every_test_file_with_the_mirrors_size(self):
        protocol = json.load(open(ev.PROTOCOL_PATH, encoding="utf-8"))
        self.assertEqual(dd.sha256_file(dd.OFFICIAL_DIRECTORY), protocol["data"]["official_directory"]["sha256"])
        official, mirror = dd.read_official_directory(), dd.read_file_metadata()
        self.assertEqual(set(official), set(mirror))
        self.assertEqual({n: v[0] for n, v in official.items()}, {n: v["bytes"] for n, v in mirror.items()})
        self.assertEqual(len(official), 4105)


class FakeResponse(io.BytesIO):
    headers = {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class DownloadTests(unittest.TestCase):
    def test_nothing_is_fetched_without_the_token(self):
        calls = []
        for confirm in (None, "yes", "RUN_DDR_GRADING_V1_ONCE"):
            with self.assertRaises(dd.PopulationError):
                dd.download([{"image": "a.jpg", "sha256": "0" * 64}], "unused", confirm=confirm, opener=lambda *a, **k: calls.append(a))
        self.assertEqual(calls, [])
        self.assertIs(inspect.signature(dd.download).parameters["confirm"].default, inspect.Parameter.empty)

    def test_only_listed_files_with_the_listed_hash_are_kept(self):
        payload = {"007-0001-100.jpg": b"one", "007-0002-300.jpg": b"two"}
        rows = [{"image": n, "sha256": __import__("hashlib").sha256(v).hexdigest()} for n, v in payload.items()]
        urls = []

        def opener(request, timeout=None):
            urls.append(request.full_url)
            return FakeResponse(payload[request.full_url.rsplit("/", 1)[1]])
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(dd.download(rows, tmp, confirm=dd.DOWNLOAD_TOKEN, opener=opener, log=lambda *a: None), 2)
            self.assertEqual(sorted(os.listdir(tmp)), sorted(payload))
            self.assertTrue(all(f"/resolve/{dd.REVISION}/DR_grading/test/" in u for u in urls))
            self.assertEqual(dd.download(rows, tmp, confirm=dd.DOWNLOAD_TOKEN, opener=opener, log=lambda *a: None), 0)      # nothing fetched twice
            bad = [{"image": "007-0003-400.jpg", "sha256": "0" * 64}]
            payload["007-0003-400.jpg"] = b"another file"
            with self.assertRaises(dd.PopulationError):
                dd.download(bad, tmp, confirm=dd.DOWNLOAD_TOKEN, opener=opener, log=lambda *a: None)
            self.assertEqual(sorted(os.listdir(tmp)), ["007-0001-100.jpg", "007-0002-300.jpg"])                         # the wrong file is not kept


# --------------------------------------------------------------------------- protocol

class ProtocolTests(unittest.TestCase):
    def test_existing_locked_protocols_are_untouched(self):
        for name, digest in LOCKED.items():
            self.assertEqual(ig.sha256_file(os.path.join(ev.HERE, name)), digest)

    def test_protocol_pins_the_approved_design(self):
        protocol, sha, parent = ev.load_protocol()
        ep = b2.load_protocol()[0]
        self.assertEqual(protocol["version"], 1)
        self.assertEqual({k: protocol[k]["sha256"] for k in ("parent_protocol", "ep_protocol")},
                         {"parent_protocol": LOCKED["idrid_grading_protocol.json"], "ep_protocol": LOCKED["idrid_batch2_protocol.json"]})
        self.assertEqual(sorted(protocol["checkpoints"]), sorted(f"{a}_seed{s}" for a in ("p", "p_ep", "e1_ep", "e2_ep") for s in (42, 123, 2026)))
        self.assertEqual({k: v["sha256"] for k, v in protocol["checkpoints"].items() if v["arm"] != "p"}, {k: v["sha256"] for k, v in ep["checkpoints"].items()})
        self.assertEqual([protocol["checkpoints"][f"p_seed{s}"]["sha256"] for s in ev.SEEDS], [parent["checkpoints"][f"p_seed{s}"]["sha256"] for s in ev.SEEDS])
        self.assertEqual([protocol["checkpoints"][f"p_seed{s}"]["sha256"][:8] for s in ev.SEEDS], ["7d6f5498", "e5f4d386", "d7d7d95f"])
        self.assertEqual([protocol["checkpoints"][f"p_ep_seed{s}"]["sha256"][:8] for s in ev.SEEDS], ["8828eacb", "6bdc68a0", "2bcd1830"])
        self.assertEqual(len({e["sha256"] for e in protocol["checkpoints"].values()}), 12)
        self.assertFalse([k for k in protocol["checkpoints"] if k.startswith(("e1_seed", "e2_seed"))])       # the original E1 / E2 are not included
        self.assertEqual(protocol["comparisons"]["primary"], ["p_ep_minus_p"])
        self.assertEqual(protocol["comparisons"]["secondary_descriptive"], ["e1_ep_minus_p_ep", "e2_ep_minus_p_ep"])
        self.assertEqual((protocol["populations"]["primary"]["n"], protocol["populations"]["primary"]["grades_0_to_4"]), (3540, [1846, 159, 1224, 57, 254]))
        self.assertEqual((protocol["populations"]["broader"]["n"], protocol["populations"]["broader"]["grades_0_to_4"]), (3725, [1846, 189, 1344, 71, 275]))
        self.assertEqual(protocol["populations"]["primary"]["list_sha256"], "f5091c4a6e16c95d80f44af5498bc0a12f9118a446038787a90d8cea1f08fd8b")
        self.assertEqual(protocol["populations"]["broader"]["list_sha256"], "52d57fcce3a2d791d6063c9aa436d12354904909e0502b3c878114a5b19645fa")
        self.assertEqual((protocol["analysis"]["bootstrap"]["n_boot"], protocol["analysis"]["bootstrap"]["seed"]), (2000, 20260927))
        self.assertEqual((protocol["pipeline"]["stage2"], protocol["pipeline"]["rgb_frame"], protocol["pipeline"]["decode"]),
                         (ep["pipeline"]["stage2"], ep["pipeline"]["rgb_frame"], ep["pipeline"]["decode"]))
        self.assertEqual((parent["inference"]["keras_policy"], parent["inference"]["batch_size"]), ("mixed_float16", 8))
        self.assertEqual(protocol["data"]["revision"], "50541319cfcdc963e3edc8b15508bc822d6b3ab5")
        self.assertIn("CC BY-NC-SA 4.0", protocol["data"]["licence_basis"])
        self.assertIn("NOT verified", protocol["data"]["provenance"])
        self.assertNotIn(protocol["confirmation_token"], (ep["confirmation_token"], b1.load_protocol()[0]["confirmation_token"], dd.DOWNLOAD_TOKEN))
        self.assertEqual(len(sha), 64)

    def test_the_protocol_states_its_limits(self):
        protocol = ev.load_protocol()[0]
        analysis, text = protocol["analysis"], json.dumps(protocol)
        self.assertIn("conditional on these fixed checkpoints", analysis["interval_scope"])
        self.assertIn("too narrow", analysis["interval_scope"])
        self.assertIn("not independent replicates", analysis["seed_aggregation"])
        self.assertIn("never represented in both families", analysis["source_family"]["confound"])
        self.assertIn("cannot establish that performance is independent of source", analysis["source_family"]["limit"])
        self.assertIn("not patient-level inference", analysis["same_visit_proxy"]["limit"])
        self.assertIn("NOT an untouched set", protocol["populations"]["broader"]["status"])
        self.assertIn("never an endpoint", protocol["populations"]["broader"]["status"])
        self.assertIn("seed-level inconsistency is flagged", analysis["reading"]["seed_consistency"])
        self.assertTrue(any("clinical" in d for d in protocol["disclosures"]))
        self.assertNotIn("non-inferior to", text)
        self.assertIn("a new protocol file", protocol["near_duplicate_screen"]["rule"]["copy_band"])
        self.assertIn("does not prove that the test images are independent", protocol["near_duplicate_screen"]["limit"])
        self.assertIn("FLAGGED rather than excluded", protocol["near_duplicate_screen"]["positive_control"]["reading"])
        self.assertIn("evidence, not proof, of byte identity", protocol["data"]["provenance"])

    def test_a_changed_protocol_is_refused(self):
        real = json.load(open(ev.PROTOCOL_PATH, encoding="utf-8"))
        changes = [lambda p: p["checkpoints"]["p_ep_seed42"].update(sha256="0" * 64),
                   lambda p: p["checkpoints"]["p_seed123"].update(sha256=p["checkpoints"]["p_seed42"]["sha256"]),
                   lambda p: p["checkpoints"]["p_seed42"].update(best_slot="best_b"),
                   lambda p: p["checkpoints"].pop("e2_ep_seed2026"),
                   lambda p: p["checkpoints"].update(e1_seed42=dict(p["checkpoints"]["e1_ep_seed42"])),
                   lambda p: p["comparisons"].update(primary=["p_ep_minus_p", "e1_ep_minus_p_ep"]),
                   lambda p: p["analysis"]["bootstrap"].update(n_boot=500),
                   lambda p: p["analysis"]["bootstrap"].update(seed=1),
                   lambda p: p["near_duplicate_screen"].update(copy_band_below=0.001),
                   lambda p: p["near_duplicate_screen"].update(ambiguous_band_below=0.02),
                   lambda p: p["parent_protocol"].update(sha256="0" * 64),
                   lambda p: p["ep_protocol"].update(sha256="0" * 64),
                   lambda p: p["data"].update(revision="main"),
                   lambda p: p.update(seeds=[42, 123, 7])]
        with tempfile.TemporaryDirectory() as tmp:
            for n, change in enumerate(changes):
                protocol = copy.deepcopy(real)
                change(protocol)
                path = os.path.join(tmp, f"p{n}.json")
                json.dump(protocol, open(path, "w", encoding="utf-8"))
                with self.assertRaises(ev.ProtocolError, msg=f"change {n}"):
                    ev.load_protocol(path)


# --------------------------------------------------------------------------- fixtures for the evaluator

class FakeNet:
    """Stands in for a model: deterministic logits from the mean of each RGB frame."""

    def __init__(self, kind, shift):
        self.kind, self.shift, self.calls = kind, float(shift), 0

    def predict_on_batch(self, x):
        self.calls += 1
        rgb = x["stage5_input"][..., :3] if self.kind == "P" else x["rgb"]
        m = rgb.reshape(len(rgb), -1).mean(axis=1, dtype=np.float64)
        base = (m - 0.5) * 12.0 + self.shift
        z = np.stack([base + 3, base + 1, base - 1, base - 3], axis=1).astype(np.float32)
        return z if self.kind == "P" else [z, None]


def fake_models():
    return {(arm, seed): FakeNet(ev.ARMS[arm], 0.2 * i) for i, (arm, seed) in enumerate((a, s) for a in ev.MODELS for s in ev.SEEDS)}


def toy_population(tmp):
    """A fabricated audit folder, image folder and protocol: 90 listed 'images' (bytes, never decoded), ten of
    them in the lesion subset, with the pins of a copy of the real protocol rewritten to them."""
    rng = np.random.default_rng(0)
    rows = []
    for k in range(90):
        stamped = k % 3 != 0
        grade = int(rng.choice([0, 1, 2, 3]) if stamped else rng.choice([1, 2, 3, 4]))
        name = f"201704{13 + k // 30:02d}10{k % 60:02d}{(k * 7) % 60:02d}000.jpg" if stamped else f"007-{k:04d}-{grade}00.jpg"
        content = f"image {k}".encode()
        rows.append({"image": name, "grade": grade, "sha256": __import__("hashlib").sha256(content).hexdigest(), "bytes": len(content),
                     "family": dd.family(name), "lesion_subset": int((not stamped) and k % 9 == 0), "_content": content})
    rows.sort(key=lambda r: r["image"])
    audit, images = os.path.join(tmp, "audit"), os.path.join(tmp, "images")
    os.makedirs(audit), os.makedirs(images)
    for r in rows:
        with open(os.path.join(images, r["image"]), "wb") as fh:
            fh.write(r.pop("_content"))
    primary = [r for r in rows if not r["lesion_subset"]]
    protocol = json.load(open(ev.PROTOCOL_PATH, encoding="utf-8"))
    for name, listed in (("primary", primary), ("broader", rows)):
        digest = dd.write_list(os.path.join(audit, protocol["populations"][name]["list"]), listed)
        protocol["populations"][name].update(n=len(listed), grades_0_to_4=dd.grade_counts(listed), list_sha256=digest)
    protocol["inference"]["chunk_images"] = 32
    path = os.path.join(tmp, "ddr_grading_protocol_v1.json")
    json.dump(protocol, open(path, "w", encoding="utf-8"))
    return protocol, path, audit, images, rows


def write_screen(audit, protocol, rows, **changes):
    spec = protocol["near_duplicate_screen"]
    report = {"list_sha256": protocol["populations"]["broader"]["list_sha256"], "images": len(rows), "labels_used": False,
              "thresholds": {"copy_band_below": spec["copy_band_below"], "ambiguous_band_below": spec["ambiguous_band_below"]},
              "against": {k: {"images": v["images"], "signature_file_sha256": v["signature_file_sha256"]} for k, v in spec["against"].items()},
              "exclude": [], "flagged": []}
    report.update(changes)
    json.dump(report, open(os.path.join(audit, dd.SCREEN_FILE), "w"))
    return report


def write_official(audit, protocol, rows, **changes):
    report = {"list_sha256": protocol["populations"]["broader"]["list_sha256"], "images": len(rows), "size_and_crc32_equal": len(rows),
              "official_directory_sha256": protocol["data"]["official_directory"]["sha256"], "mismatches": [], "not_in_official_directory": []}
    report.update(changes)
    json.dump(report, open(os.path.join(audit, dd.OFFICIAL_CHECK_FILE), "w"))


def synthetic_tables(population, shifts=None, seed=0):
    rng = np.random.default_rng(seed)
    grades, ids = population["grades"], population["ids"]
    tables, frames = {}, {}
    for s in ev.SEEDS:
        tables[s] = {}
        base = (grades[:, None] - np.arange(4)[None, :] - 0.5) * 2.0
        for arm in ev.MODELS:
            z = base + rng.normal(0, 1.6, base.shape) + (shifts or {}).get((arm, s), 0.0)
            frame, tables[s][arm] = b1.table_from_logits(ids, grades, z)
            frames[(arm, s)] = frame
    return tables, frames


class PopulationAndScreenGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.protocol, self.path, self.audit, self.images, self.rows = toy_population(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_lists_are_read_as_pinned(self):
        population = ev.read_population(self.audit, self.protocol)
        self.assertEqual(population["ids"], [r["image"] for r in self.rows])
        self.assertEqual(int(population["primary"].sum()), self.protocol["populations"]["primary"]["n"])
        self.assertTrue(np.array_equal(population["primary"], ~population["lesion_subset"]))
        self.assertEqual(set(population["family"].tolist()), {"timestamp", "007"})
        self.assertEqual(len(population["units"]), len(self.rows))

    def test_changed_lists_are_refused(self):
        primary_path = os.path.join(self.audit, self.protocol["populations"]["primary"]["list"])
        broader_path = os.path.join(self.audit, self.protocol["populations"]["broader"]["list"])
        with open(primary_path, "a") as fh:                              # one more row than pinned
            fh.write("007-9999-100.jpg,1," + "0" * 64 + ",5,007,0\n")
        with self.assertRaises(ev.ProtocolError):
            ev.read_population(self.audit, self.protocol)
        os.remove(primary_path)
        with self.assertRaises(ev.ProtocolError):
            ev.read_population(self.audit, self.protocol)

        def repin(name, rows):
            self.protocol["populations"][name].update(n=len(rows), grades_0_to_4=dd.grade_counts([r for r in rows if r["grade"] < 5]),
                                                      list_sha256=dd.write_list({"primary": primary_path, "broader": broader_path}[name], rows))
        primary = [r for r in self.rows if not r["lesion_subset"]]
        lesion = [r for r in self.rows if r["lesion_subset"]]
        repin("primary", primary + lesion[:1])                           # a lesion-subset image inside the primary list
        with self.assertRaises(ev.ProtocolError):
            ev.read_population(self.audit, self.protocol)
        repin("primary", primary[:-1])                                   # a non-lesion image missing from the primary list
        with self.assertRaises(ev.ProtocolError):
            ev.read_population(self.audit, self.protocol)
        repin("primary", primary)
        repin("broader", self.rows[:-1] + [dict(self.rows[-1], sha256=self.rows[0]["sha256"])])      # two listed names, one content
        with self.assertRaises(ev.ProtocolError):
            ev.read_population(self.audit, self.protocol)

    def test_missing_or_altered_images_are_refused(self):
        population = ev.read_population(self.audit, self.protocol)
        self.assertEqual(ev.verify_images(self.images, population, lambda *a: None), 90)
        victim = os.path.join(self.images, population["ids"][3])
        with open(victim, "ab") as fh:
            fh.write(b"x")
        with self.assertRaises(ev.ProtocolError):
            ev.verify_images(self.images, population, lambda *a: None)
        os.remove(victim)
        with self.assertRaises(ev.ProtocolError):
            ev.verify_images(self.images, population, lambda *a: None)

    def test_the_screen_must_have_been_run_as_pinned_and_be_clean(self):
        population = ev.read_population(self.audit, self.protocol)
        with self.assertRaises(ev.ProtocolError):                        # not run
            ev.check_screen(self.audit, self.protocol, population)
        name = next(iter(self.protocol["near_duplicate_screen"]["against"]))
        for change in ({"list_sha256": "0" * 64}, {"images": 5}, {"labels_used": True},
                       {"thresholds": {"copy_band_below": 0.001, "ambiguous_band_below": 0.06}},
                       {"against": {}}, {"against": {name: {"images": 1, "signature_file_sha256": "0" * 64}}},
                       {"exclude": [population["ids"][0]]}, {"flagged": ["not-listed.jpg"]}):
            write_screen(self.audit, self.protocol, self.rows, **change)
            with self.assertRaises(ev.ProtocolError, msg=str(change)):
                ev.check_screen(self.audit, self.protocol, population)
        write_screen(self.audit, self.protocol, self.rows, flagged=[population["ids"][2], population["ids"][7]])
        flagged, digest = ev.check_screen(self.audit, self.protocol, population)
        self.assertEqual((np.flatnonzero(flagged).tolist(), len(digest)), ([2, 7], 64))


    def test_the_official_directory_comparison_must_have_been_run_and_be_clean(self):
        population = ev.read_population(self.audit, self.protocol)
        with self.assertRaises(ev.ProtocolError):                        # not run
            ev.check_official(self.audit, self.protocol, population)
        for change in ({"list_sha256": "0" * 64}, {"images": 5}, {"official_directory_sha256": "0" * 64}, {"mismatches": [population["ids"][0]]},
                       {"not_in_official_directory": [population["ids"][1]]}, {"size_and_crc32_equal": 89}):
            write_official(self.audit, self.protocol, self.rows, **change)
            with self.assertRaises(ev.ProtocolError, msg=str(change)):
                ev.check_official(self.audit, self.protocol, population)
        write_official(self.audit, self.protocol, self.rows)
        self.assertEqual(len(ev.check_official(self.audit, self.protocol, population)), 64)


class VerifyFilesTests(unittest.TestCase):
    """Fabricated run directories (the stand-in drive of the batch-2 tests, with P's three files added)."""

    def setUp(self):
        import test_idrid_batch2_eval as t2
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        ep, _, parent = b2.load_protocol()
        self.ep, self.parent = t2.fake_drive(self.tmp.name, copy.deepcopy(ep), copy.deepcopy(parent))
        self.protocol = json.load(open(ev.PROTOCOL_PATH, encoding="utf-8"))
        root = lambda rel: os.path.join(self.tmp.name, *rel.split("/"))
        for name, entry in self.protocol["checkpoints"].items():
            if entry["arm"] == "p":
                os.makedirs(os.path.dirname(root(entry["path"])), exist_ok=True)
                with open(root(entry["path"]), "wb") as fh:
                    fh.write(name.encode())
                entry["sha256"] = ig.sha256_file(root(entry["path"]))
                json.dump({"active": entry["best_slot"], "epoch": entry["best_epoch_index"]}, open(root(entry["best_pointer"]), "w"))
            else:
                entry["sha256"] = self.ep["checkpoints"][name]["sha256"]
        self.root = root
        self._orig = b2.load_protocol
        b2.load_protocol = lambda path=None, parent_path=None: (self.ep, "unused", self.parent)

    def tearDown(self):
        b2.load_protocol = self._orig
        self.tmp.cleanup()

    def test_the_twelve_pinned_checkpoints_are_accepted_and_no_prediction_table_is_needed(self):
        for s in ev.SEEDS:                                               # the stored IDRiD P tables are not part of this evaluation
            os.remove(self.root(self.ep["stored_p_tables"][f"p_seed{s}"]["path"]))
        files = ev.verify_files(self.protocol, self.tmp.name, self.parent)
        self.assertEqual(sorted(files["checkpoints"]), sorted(self.protocol["checkpoints"]))
        self.assertEqual({k: v for k, v in files["kinds"].items()}, {k: v["model_type"] for k, v in self.protocol["checkpoints"].items()})
        self.assertNotIn("p_tables", files)
        self.assertEqual(sorted(files["lesion_prior"]), sorted(k for k, v in self.protocol["checkpoints"].items() if v["model_type"] == "E1"))

    def test_a_wrong_missing_or_moved_checkpoint_is_refused(self):
        entry = self.protocol["checkpoints"]["p_seed123"]
        with open(self.root(entry["path"]), "ab") as fh:
            fh.write(b"changed")
        with self.assertRaises(ev.ProtocolError):
            ev.verify_files(self.protocol, self.tmp.name, self.parent)
        os.remove(self.root(entry["path"]))
        with self.assertRaises(ev.ProtocolError):
            ev.verify_files(self.protocol, self.tmp.name, self.parent)
        with open(self.root(entry["path"]), "wb") as fh:
            fh.write(b"p_seed123")
        ev.verify_files(self.protocol, self.tmp.name, self.parent)
        json.dump({"active": "best_b", "epoch": 99}, open(self.root(entry["best_pointer"]), "w"))      # BEST moved after pinning
        with self.assertRaises(ev.ProtocolError):
            ev.verify_files(self.protocol, self.tmp.name, self.parent)
        json.dump({"active": entry["best_slot"], "epoch": entry["best_epoch_index"]}, open(self.root(entry["best_pointer"]), "w"))
        ep_entry = self.ep["checkpoints"]["e1_ep_seed42"]
        with open(self.root(ep_entry["path"]), "ab") as fh:
            fh.write(b"changed")
        with self.assertRaises(ev.ProtocolError):
            ev.verify_files(self.protocol, self.tmp.name, self.parent)


# --------------------------------------------------------------------------- analysis

class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.protocol, _, audit, _, cls.rows = toy_population(cls.tmp.name)
        cls.population = ev.read_population(audit, cls.protocol)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_reading_rule(self):
        self.assertTrue(ev.reading(0.001, 0.05).startswith("supported"))
        self.assertTrue(ev.reading(-0.05, -0.001).startswith("reversed"))
        for lo, hi in ((-0.01, 0.03), (0.0, 0.03), (-0.03, 0.0)):
            self.assertTrue(ev.reading(lo, hi).startswith("no difference shown"))
        self.assertIn("no equivalence", ev.reading(-0.01, 0.01))

    def test_the_primary_quantity_is_the_paired_seed_mean_and_matches_the_projects_block(self):
        from sklearn.metrics import cohen_kappa_score
        tables, _ = synthetic_tables(self.population, {("p_ep", s): 0.5 for s in ev.SEEDS})
        primary = ev.analyse_primary(tables, self.population)
        mask = self.population["primary"]
        direct = [cohen_kappa_score(tables[s]["p_ep"]["grade"][mask], tables[s]["p_ep"]["pred"][mask], weights="quadratic", labels=list(range(5)))
                  - cohen_kappa_score(tables[s]["p"]["grade"][mask], tables[s]["p"]["pred"][mask], weights="quadratic", labels=list(range(5)))
                  for s in ev.SEEDS]
        result = primary["result"]
        self.assertTrue(np.allclose([result["per_seed"][str(s)] for s in ev.SEEDS], direct))
        self.assertAlmostEqual(result["mean"], float(np.mean(direct)))
        self.assertEqual((primary["n"], result["resamples"], primary["bootstrap"]["seed"]), (int(mask.sum()), 2000, 20260927))
        block = b1.summarise(tables, mask, models=ev.MODELS, contrasts=ev.CONTRASTS, primary=(ev.PRIMARY,))["paired_differences"]["p_ep_minus_p"]["qwk"]
        self.assertAlmostEqual(block["mean"], result["mean"], places=10)
        self.assertAlmostEqual(block["ci_low"], result["ci_low"], places=10)
        self.assertAlmostEqual(block["ci_high"], result["ci_high"], places=10)
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertIn("not independent", primary["scope"])
        self.assertIn("too narrow", primary["scope"])

    def test_seed_level_inconsistency_is_flagged_whatever_the_interval(self):
        tables, _ = synthetic_tables(self.population)
        for s in ev.SEEDS:
            tables[s]["p_ep"] = {k: v.copy() for k, v in tables[s]["p"].items()}
        mask = self.population["primary"]
        same = ev.qwk_difference(tables, "p_ep", "p", mask, ph.bootstrap_indices(self.population["grades"][mask], 50))
        self.assertEqual((same["mean"], same["seeds_positive"], same["seed_level_inconsistency"]), (0.0, 0, True))
        grades = tables[42]["p"]["grade"]
        for s in ev.SEEDS[:2]:                                           # two seeds made perfect, the third left equal to P
            tables[s]["p_ep"]["pred"] = grades.copy()
        mixed = ev.qwk_difference(tables, "p_ep", "p", mask, ph.bootstrap_indices(self.population["grades"][mask], 50))
        self.assertEqual((mixed["seeds_positive"], mixed["seed_level_inconsistency"]), (2, True))
        tables[ev.SEEDS[2]]["p_ep"]["pred"] = grades.copy()
        self.assertEqual(ev.qwk_difference(tables, "p_ep", "p", mask, ph.bootstrap_indices(self.population["grades"][mask], 50))["seed_level_inconsistency"], False)
        swapped = {s: {"p": tables[s]["p"], "p_ep": dict(tables[s]["p_ep"], ids=tables[s]["p_ep"]["ids"][::-1])} for s in ev.SEEDS}
        with self.assertRaises(ev.ProtocolError):                        # not the same images in the same order
            ev.qwk_difference(swapped, "p_ep", "p", np.ones(len(grades), bool), [np.arange(len(grades))])

    def test_cluster_resamples_keep_units_whole_and_stay_inside_their_family(self):
        units = np.array(["a", "a", "b", "c", "c", "c", "x", "y"])
        strata = np.array(["t"] * 6 + ["s"] * 2)
        draws = ev.cluster_indices(units, strata, n_boot=200, seed=1)
        self.assertEqual(len(draws), 200)
        for idx in draws:
            self.assertEqual(int((strata[idx] == "s").sum()), 2)        # two units of one image each
            counts = np.bincount(idx, minlength=8)
            self.assertEqual((counts[0], counts[3], counts[3]), (counts[1], counts[4], counts[5]))     # members travel together
        self.assertTrue(any(len(idx) != 8 for idx in draws))
        again = ev.cluster_indices(units, strata, n_boot=200, seed=1)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(draws, again)))

    def test_family_analysis_reports_only_what_each_family_can_show(self):
        tables, _ = synthetic_tables(self.population)
        out = ev.family_analysis(tables, self.population, self.population["primary"])
        self.assertEqual(out["timestamp"]["grade_counts"][4], 0)
        self.assertEqual(out["007"]["grade_counts"][0], 0)
        self.assertTrue(np.isnan(out["timestamp"]["arms"]["p"]["auroc_cut_mean"]["ge4"]))             # no grade 4 among timestamp names
        self.assertTrue(np.isnan(out["007"]["arms"]["p"]["auroc_cut_mean"]["ge1"]))                   # no grade 0 among 007 names
        self.assertFalse(np.isnan(out["007"]["arms"]["p"]["auroc_cut_mean"]["ge3"]))
        self.assertEqual(out["timestamp"]["n"] + out["007"]["n"], int(self.population["primary"].sum()))
        self.assertIn("cannot establish that performance is independent of source", out["limit"])
        self.assertEqual(sorted(out["timestamp"]["arms"]), sorted(ev.MODELS))

    def test_secondary_and_sensitivity_results_are_complete_and_labelled(self):
        tables, frames = synthetic_tables(self.population)
        secondary = ev.analyse_secondary(tables, frames, self.population)
        self.assertEqual(sorted(secondary["contrasts_qwk"]), sorted(ev.CONTRASTS))
        self.assertEqual(sorted(secondary["seed_ensemble"]["qwk"]), sorted(ev.MODELS))
        self.assertEqual(secondary["metric_block"]["paired_differences"]["p_ep_minus_p"]["role"], "primary")
        self.assertEqual(secondary["metric_block"]["paired_differences"]["e2_ep_minus_p_ep"]["role"], "secondary, descriptive")
        self.assertIn("not patient-level inference", secondary["same_visit_proxy"]["limit"])
        self.assertLess(secondary["same_visit_proxy"]["units"], int(self.population["primary"].sum()) + 1)
        for arm in ev.MODELS:
            c = secondary["calibration"][arm]["42"]
            self.assertTrue(0 <= c["ece_top_label"] <= 1 and c["nll"] > 0)
            self.assertTrue(0 <= secondary["referable_dr"][arm]["42"]["sensitivity"] <= 1)
        none = ev.analyse_sensitivity(tables, self.population, np.zeros(len(self.rows), bool))
        self.assertEqual(sorted(k for k in none if k.endswith(("untouched", "only", "flagged"))), ["broader_3725_not_untouched", "lesion_subset_185_only"])
        self.assertEqual(none["broader_3725_not_untouched"]["n"], len(self.rows))
        self.assertEqual(none["lesion_subset_185_only"]["n"], int(self.population["lesion_subset"].sum()))
        self.assertIn("never replaces the primary result", none["note"])
        flagged = np.zeros(len(self.rows), bool)
        flagged[np.flatnonzero(self.population["primary"])[:4]] = True
        some = ev.analyse_sensitivity(tables, self.population, flagged)
        self.assertEqual(some["primary_without_screen_flagged"]["n"], int(self.population["primary"].sum()) - 4)

    def test_the_ensemble_averages_probabilities_and_decodes_as_every_grader(self):
        tables, _ = synthetic_tables(self.population)
        table = ev.ensemble_table(tables, "p")
        mean = np.mean([tables[s]["p"]["p_gt"] for s in ev.SEEDS], axis=0)
        self.assertTrue(np.array_equal(table["pred"], (mean > 0.5).sum(axis=1)))


# --------------------------------------------------------------------------- the one-time run (mechanics only)

class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.protocol, self.path, self.audit, self.images, self.rows = toy_population(self.tmp.name)
        self.sha = ig.sha256_file(self.path)
        write_screen(self.audit, self.protocol, self.rows, flagged=[self.rows[1]["image"]])
        write_official(self.audit, self.protocol, self.rows)
        self.decoded = []
        self._orig = (ev.verify_files, b1.rgb_from_raw, ev.load_models)
        ev.verify_files = lambda protocol, root, parent=None: {
            "checkpoints": {k: "unused" for k in protocol["checkpoints"]}, "kinds": {k: v["model_type"] for k, v in protocol["checkpoints"].items()},
            "lesion_prior": {}, "pretrained": "unused", "adapted_encoder_sha256": "e" * 64}

        def fake_rgb(path):
            self.decoded.append(os.path.basename(path))
            value = (sum(open(path, "rb").read()) % 97) / 97.0
            return np.full((512, 512, 3), value, np.float32), np.zeros((6, 8, 3), np.uint8)
        b1.rgb_from_raw = fake_rgb

        def no_real_models(*a, **k):
            raise AssertionError("a test tried to load a real model")
        ev.load_models = no_real_models
        self.out_root = os.path.join(self.tmp.name, "drive", "experiments", "DDRGrading")
        settings = dict(ev.load_protocol(self.path)[2]["inference"])
        self.gate = {"official": True, "PASS": True, "protocol_sha256": self.sha, "settings": settings, "git_commit": ig.git_commit(),
                     "environment": ig.environment(settings)}

    def tearDown(self):
        ev.verify_files, b1.rgb_from_raw, ev.load_models = self._orig
        self.tmp.cleanup()

    def _run(self, **kw):
        args = dict(confirm=self.protocol["confirmation_token"], models=fake_models(), protocol_path=self.path, log=lambda *a: None)
        args.update(kw)
        return ev.run("unused_drive", self.images, self.audit, self.out_root, self.gate, **args)

    def test_refuses_without_token_gate_screen_or_images(self):
        for confirm in (None, "yes", dd.DOWNLOAD_TOKEN, b2.load_protocol()[0]["confirmation_token"]):
            with self.assertRaises(ev.ProtocolError):
                self._run(confirm=confirm)
        for change in ({"PASS": False}, {"official": False}, {"protocol_sha256": "0" * 64}, {"git_commit": "another commit"},
                       {"settings": dict(self.gate["settings"], batch_size=4)}):
            gate, self.gate = self.gate, dict(self.gate, **change)
            with self.assertRaises(ev.ProtocolError):
                self._run()
            self.gate = gate
        write_screen(self.audit, self.protocol, self.rows, exclude=[self.rows[0]["image"]])          # a copy-band image stops version 1
        with self.assertRaises(ev.ProtocolError):
            self._run()
        os.remove(os.path.join(self.audit, dd.SCREEN_FILE))
        with self.assertRaises(ev.ProtocolError):
            self._run()
        write_screen(self.audit, self.protocol, self.rows)
        write_official(self.audit, self.protocol, self.rows, mismatches=[self.rows[2]["image"]])      # a file that differs from the official entry
        with self.assertRaises(ev.ProtocolError):
            self._run()
        os.remove(os.path.join(self.audit, dd.OFFICIAL_CHECK_FILE))
        with self.assertRaises(ev.ProtocolError):
            self._run()
        write_official(self.audit, self.protocol, self.rows)
        os.remove(os.path.join(self.images, self.rows[5]["image"]))
        with self.assertRaises(ev.ProtocolError):
            self._run()
        self.assertEqual(self.decoded, [])                               # no image was decoded on any refusal
        self.assertEqual(ev.existing_outputs(self.out_root), [])         # and no lock or directory was left behind

    def test_one_run_writes_every_output_once_and_then_locks(self):
        models = fake_models()
        out = self._run(models=models)
        names = sorted(os.listdir(out["out_dir"]))
        self.assertEqual(names, ev.expected_outputs())
        self.assertEqual(len([n for n in names if n.startswith("per_image_")]), 12)
        self.assertEqual(self.decoded, [r["image"] for r in self.rows])   # every listed image once, in list order
        self.assertTrue(all(net.calls == -(-32 // 8) * 2 + -(-26 // 8) for net in models.values()))      # chunks of 32, batches of 8
        hashes = json.load(open(os.path.join(out["out_dir"], "output_hashes.json")))
        self.assertEqual(sorted(hashes), [n for n in names if n != "output_hashes.json"])
        self.assertTrue(all(ig.sha256_file(os.path.join(out["out_dir"], n)) == h_ for n, h_ in hashes.items()))
        frame = pd.read_csv(os.path.join(out["out_dir"], "per_image_p_ep_seed42.csv"))
        self.assertEqual((len(frame), int(frame["in_primary_population"].sum()), int(frame["screen_flagged"].sum())),
                         (90, self.protocol["populations"]["primary"]["n"], 1))
        self.assertEqual(list(frame["image_id"]), [r["image"] for r in self.rows])
        primary = json.load(open(os.path.join(out["out_dir"], "primary_result.json")))
        self.assertEqual((primary["n"], sorted(primary["result"]["per_seed"])), (self.protocol["populations"]["primary"]["n"], ["123", "2026", "42"]))
        self.assertIn(primary["result"]["reading"].split(":")[0], ("supported", "reversed", "no difference shown"))
        report = open(os.path.join(out["out_dir"], "primary_report.md"), encoding="utf-8").read()
        self.assertIn("Reading (pre-declared)", report)
        self.assertIn("cannot separate grading from recognising the source", report)
        sensitivity = json.load(open(os.path.join(out["out_dir"], "sensitivity_results.json")))
        self.assertIn("primary_without_screen_flagged", sensitivity)
        configuration = json.load(open(os.path.join(out["out_dir"], "run_configuration.json")))
        self.assertEqual((configuration["idrid_files_opened"], len(configuration["models_run"]), configuration["protocol_sha256"]), (0, 12, self.sha))
        self.assertTrue(os.path.exists(ev.lock_path(self.out_root, self.sha)))
        with self.assertRaises(ev.ProtocolError):                        # a second run is refused
            self._run()
        with self.assertRaises(ev.ProtocolError):
            self._run(technical_rerun_reason="   ")
        self.assertEqual(len(self.decoded), 90)
        again = self._run(technical_rerun_reason="runtime disconnected before the tables were written")
        record = json.load(open(os.path.join(again["out_dir"], "lock_record.json")))
        self.assertEqual(record["technical_rerun_reason"], "runtime disconnected before the tables were written")
        self.assertEqual(len(record["previous_outputs"]), 2)             # the first lock and the first directory are kept and named
        self.assertTrue(os.path.isdir(out["out_dir"]))

    def test_an_unfinished_earlier_attempt_cannot_be_mistaken_for_a_fresh_run(self):
        os.makedirs(os.path.join(self.out_root, ev.OUTPUT_PREFIX + "leftover"))
        with self.assertRaises(ev.ProtocolError):
            self._run()
        self.assertEqual(self.decoded, [])

    def test_preflight_checks_everything_that_needs_no_model(self):
        report = ev.preflight("unused_drive", self.images, self.audit, self.out_root, self.path, log=lambda *a: None)
        self.assertEqual((report["PASS"], report["images"], report["already_run"], report["screen_flagged"], len(report["checkpoints"])),
                         (True, 90, False, 1, 12))
        self.assertEqual(self.decoded, [])
        self._run()
        self.assertTrue(ev.preflight("unused_drive", self.images, self.audit, self.out_root, self.path, log=lambda *a: None)["already_run"])


class ModuleTests(unittest.TestCase):
    def test_no_idrid_code_path_and_no_training(self):
        source = open(os.path.join(ev.HERE, "ddr_grading_eval.py"), encoding="utf-8").read()
        code = source.split('"""', 2)[2]                                 # the module docstring names IDRiD; the code must not use it
        for forbidden in ("read_idrid", "idrid_raw_dir", "read_p_tables", "stored_p_tables", "verify_files(ep", ".fit(", "train_on_batch",
                          "run_idrid", "b2.run(", "b1.run("):
            self.assertNotIn(forbidden, code, forbidden)
        data_source = open(os.path.join(ev.HERE, "ddr_grading_data.py"), encoding="utf-8").read()
        for forbidden in ("import keras", "import tensorflow", "import torch", "idrid_"):
            self.assertNotIn(forbidden, data_source, forbidden)
        for function in (ev.run, ev.preflight, ev.gates):
            self.assertFalse([p for p in inspect.signature(function).parameters if "idrid" in p.lower()])
        self.assertIs(inspect.signature(ev.run).parameters["confirm"].default, inspect.Parameter.empty)
        self.assertEqual(ev.MODELS, ("p", "p_ep", "e1_ep", "e2_ep"))
        self.assertEqual(ev.CONTRASTS, {"p_ep_minus_p": ("p_ep", "p"), "e1_ep_minus_p_ep": ("e1_ep", "p_ep"), "e2_ep_minus_p_ep": ("e2_ep", "p_ep")})

    def test_the_image_path_and_decode_are_the_locked_ones(self):
        source = open(os.path.join(ev.HERE, "ddr_grading_eval.py"), encoding="utf-8").read()
        for reused in ("b1.rgb_from_raw(", "b1.table_from_logits(", "b1.gates(", "b1.require_gates(", "b1.summarise(", "b2.verify_checkpoints(",
                       "b2.load_models(", "ph.bootstrap_indices("):
            self.assertIn(reused, source, reused)

    def test_expected_outputs(self):
        names = ev.expected_outputs()
        self.assertEqual(len(names), 12 + len(ev.RESULT_FILES) + 1)
        self.assertIn("primary_result.json", names)
        self.assertIn("per_image_p_seed2026.csv", names)


if __name__ == "__main__":
    unittest.main()
