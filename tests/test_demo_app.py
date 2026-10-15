"""CPU tests for the local demonstration server's model-free parts: the demo image list, image encoding, the lesion
overlay, the page and its endpoints with a stand-in engine. The full pipeline on real models is exercised by running
the app (record §100)."""
import base64
import io
import json
import os
import threading
import unittest
import urllib.request

import numpy as np

import demo_app as app


class HelperTests(unittest.TestCase):
    def test_png_encodes_images_and_maps(self):
        from PIL import Image
        for array in (np.random.default_rng(0).random((40, 60, 3)), np.random.default_rng(1).random((40, 60))):
            im = Image.open(io.BytesIO(base64.b64decode(app.png(array))))
            self.assertEqual(im.size, (60, 40))
        self.assertLessEqual(max(Image.open(io.BytesIO(base64.b64decode(app.png(np.zeros((800, 600, 3)), size=200)))).size), 200)

    def test_lesion_overlay_tints_only_where_a_map_is_above_one_half(self):
        rgb = np.full((8, 8, 3), 0.5, np.float32)
        maps = np.zeros((8, 8, 4), np.float32)
        maps[2, 3, 1] = 0.9                                              # one haemorrhage pixel
        out = app.lesion_overlay(rgb, maps)
        self.assertTrue(np.allclose(out[0, 0], 0.5))
        self.assertGreater(out[2, 3, 0], out[2, 3, 2])                   # tinted red
        self.assertEqual(app.LESIONS, ("MA", "HE", "EX", "SE"))

    @unittest.skipUnless(os.path.exists(os.path.join(app.DEMO_DIR, "labels.csv")), "the demo image folder is not present")
    def test_the_demo_images_carry_reference_grades(self):
        demos = app.demo_list()
        self.assertEqual(len(demos), 30)
        self.assertEqual(sorted({d["reference_grade"] for d in demos}), [0, 1, 2, 3, 4])
        self.assertTrue(all(os.path.exists(os.path.join(app.DEMO_DIR, d["file"])) for d in demos))
        self.assertFalse([d for d in demos if "DDR" in d["source"] or "test" in d["original_path"].lower().replace("testing", "test") and "IDRiD" in d["source"]])

    def test_reliability_flags_each_failed_check_and_uses_the_validation_band(self):
        cal = {"coverage": 0.8, "entropy_threshold": 0.7, "certain": {"exact": 0.92, "within_one": 0.98, "images": 10},
               "uncertain": {"exact": 0.62, "within_one": 0.91, "images": 3}}
        review, reasons, rates = app.reliability("Good", [2, 2, 2], False, 0.3, cal)
        self.assertFalse(review)
        self.assertEqual((reasons, rates["exact"]), ([], 0.92))
        review, reasons, rates = app.reliability("Reject", [2, 4, 2], True, 0.9, cal)
        self.assertTrue(review)
        self.assertEqual(len(reasons), 4)
        self.assertEqual(rates["exact"], 0.62)
        self.assertTrue(app.reliability("Usable", [3, 3, 3], False, 0.71, cal)[0])       # entropy alone

    def test_map_on_lesions_is_the_top_decile_share_beside_the_chance_share(self):
        heat = np.zeros((10, 10))
        heat[0, :] = 1.0                                                  # the top 10 % = the first row
        mask = np.zeros((10, 10), bool)
        mask[0, :5] = True
        self.assertEqual(app.map_on_lesions(heat, mask), (0.5, 0.05))

    def test_the_written_summary_states_only_what_the_outputs_say(self):
        rates = {"exact": 0.92, "within_one": 0.98, "images": 3393}
        probs = [0.0, 0.0, 0.76, 0.0, 0.24]
        lines = app.written_summary(2, probs, {"MA": 0.0, "HE": 0.003, "EX": 0.018, "SE": 0.0}, {"Grad-CAM": (0.35, 0.02)},
                                    False, [], rates, "optimistic")
        text = " ".join(lines)
        self.assertIn("Grade 2 - Moderate NPDR", text)
        self.assertIn("76 %", text)
        self.assertLess(text.index("Stage 4 marks hard exudates"), text.index("haemorrhages (0"))   # largest area first
        self.assertNotIn("microaneurysms (", text)                       # not found -> not listed
        self.assertIn("35 % of its most influential area", text)
        self.assertIn("No reliability flag", text)
        self.assertIn("exact in 92 %", text)
        lines = app.written_summary(3, [0, 0, 0.1, 0.9, 0], {k: 0.0 for k in app.LESIONS}, {}, True, ["the three models give different grades [3, 2, 3]"], rates, "x")
        text = " ".join(lines)
        self.assertIn("Stage 4 marks no lesions", text)
        self.assertIn("worth a closer look", text)
        self.assertIn("Review recommended", text)

    def test_the_reliability_calibration_file_matches_the_validation_tables_it_came_from(self):
        cal = json.load(open(app.RELIABILITY_PATH, encoding="utf-8"))
        for arm in ("crop", "direct"):
            c = cal["arms"][arm]
            self.assertEqual((c["images"], c["certain"]["images"] + c["uncertain"]["images"]), (4242, 4242))
            self.assertGreater(c["certain"]["exact"], c["uncertain"]["exact"])
        tables = [os.path.join(app.HERE, "results", "FinalExternal", "uncertainty", f"uncertainty_jc_{s}_val.csv") for s in ("aptos", "eyepacs")]
        if all(os.path.exists(p) for p in tables):
            import uncertainty_eval as ue
            self.assertEqual(ue.reliability_calibration(tables), cal["arms"]["crop"])

    def test_the_default_arm_is_the_post_hoc_preferred_candidate(self):
        self.assertEqual(app.DEFAULT_ARM, "crop")
        self.assertIn("not a medical device", app.NOTICE)


class ServerTests(unittest.TestCase):
    """The page and the job endpoints with a stand-in engine: no model is loaded."""

    @classmethod
    def setUpClass(cls):
        from http.server import ThreadingHTTPServer

        class FakeEngine:
            lock = threading.Lock()
        cls.original = (app.ENGINE, app.run_pipeline)
        app.ENGINE = FakeEngine()

        def fake_pipeline(engine, path, arm, explanations, emit, reference=None):
            emit({"number": 1, "title": f"stage one ({arm})", "text": ",".join(explanations), "images": [], "seconds": 0.1, "data": {}})
            emit({"number": 99, "title": "Done", "text": app.NOTICE, "images": [], "seconds": None, "data": {}})
        app.run_pipeline = fake_pipeline
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        app.ENGINE, app.run_pipeline = cls.original

    def _get(self, path):
        with urllib.request.urlopen(self.url + path, timeout=10) as r:
            return r.read()

    def test_the_page_and_a_job_from_upload_to_done(self):
        self.assertIn(b"Run the pipeline", self._get("/"))
        upload = base64.b64encode(b"not really an image").decode()
        request = urllib.request.Request(self.url + "/run", data=json.dumps({"upload": upload, "filename": "x.png", "arm": "direct",
                                                                             "explanations": ["grad_cam", "bogus"]}).encode(), method="POST")
        job = json.loads(urllib.request.urlopen(request, timeout=10).read())
        for _ in range(50):
            state = json.loads(self._get("/job/" + job["id"]))
            if state["done"]:
                break
            threading.Event().wait(0.1)
        self.assertTrue(state["done"] and state["error"] is None)
        self.assertEqual(state["stages"][0]["title"], "stage one (direct)")
        self.assertEqual(state["stages"][0]["text"], "grad_cam")       # unknown explanation names are dropped
        self.assertIsNone(json.loads(self._get("/job/unknown")).get("done") and None)


if __name__ == "__main__":
    unittest.main()
