"""Local web demonstration of the full pipeline (research record §100) -- http://localhost:8501

    python demo_app.py                    # then open http://localhost:8501 in a browser

Upload a fundus image (or pick one of the demo images with known reference grades) and watch the stages run one
after another: Stage 1 image quality, Stage 2 preprocessing, the grader's input frame, Stage 3 vessels, Stage 4
lesions, the grade from the three-seed joint grader, its uncertainty (seed ensemble and test-time augmentation),
and explanation maps (Grad-CAM, Integrated Gradients, Occlusion).

Only Python's standard library serves the page: no web framework is needed. Every model is frozen and is checked
against its pinned SHA-256 when loaded from `trained_models/demo/`. The grade comes from the RGB grader alone; the
vessel and lesion maps are supporting views. Runs on CPU: about 4 minutes per image with Grad-CAM (the
default); Integrated Gradients adds about 7 minutes and Occlusion about 3 (measured on the project laptop).
Research prototype -- not a medical device, not a diagnosis.
"""
import base64
import csv
import hashlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL_ROOT = os.path.join(HERE, "trained_models", "demo")
DEMO_DIR = os.path.join(HERE, "datasets", "demo_images_v1")
PORT = 8501
GRADES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR")
LESIONS = ("MA", "HE", "EX", "SE")
LESION_NAMES = {"MA": "microaneurysms", "HE": "haemorrhages", "EX": "hard exudates", "SE": "soft exudates"}
NOTICE = "Research prototype - not a medical device; its output is not a diagnosis."
#: The current preferred candidate is the cropped joint arm -- a choice made AFTER the external re-test (record 100.1),
#: so its external results are not an unbiased estimate for a pre-selected model. The direct arm stays selectable.
DEFAULT_ARM = "crop"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def png(array, cmap=None, vmin=0.0, vmax=1.0, size=None):
    """A base64 PNG of an image (H, W, 3 in [0, 1] or uint8) or a map (H, W) through a colormap."""
    import matplotlib
    matplotlib.use("Agg")
    from PIL import Image
    a = np.asarray(array)
    if a.ndim == 2:
        a = matplotlib.colormaps[cmap or "inferno"](np.clip((a.astype(np.float64) - vmin) / max(vmax - vmin, 1e-12), 0, 1))[..., :3]
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    if size:
        im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def lesion_overlay(rgb, lesion_max):
    """The frame with each lesion class tinted where its map exceeds 0.5."""
    colours = {"MA": (1.0, 1.0, 0.0), "HE": (1.0, 0.0, 0.0), "EX": (0.0, 1.0, 1.0), "SE": (1.0, 0.0, 1.0)}
    out = np.asarray(rgb, np.float32).copy()
    for i, name in enumerate(LESIONS):
        m = lesion_max[..., i] > 0.5
        out[m] = 0.35 * out[m] + 0.65 * np.asarray(colours[name])
    return out


# --------------------------------------------------------------------------- models (loaded once, hash-checked)

class Engine:
    def __init__(self, log=print):
        import gradcam
        import idrid_grading_eval as ig
        import image_quality_inference as iqa
        import pl_convnext as pl
        parent = ig.load_protocol()[0]
        protocol = json.load(open(os.path.join(HERE, "ddr_idrid_retest_protocol_v1.json"), encoding="utf-8"))

        def local(entry):
            path = os.path.join(MODEL_ROOT, *entry["path"].split("/"))
            if not os.path.exists(path):
                raise FileNotFoundError(f"{path} is missing: download the demo models first")
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"{path} is not the pinned model (SHA-256 mismatch)")
            return path
        self.settings = {"device": "cpu", "stage4_amp": False}
        log("loading Stage 3 (LWNet) and Stage 4 (lesion U-Net) ...")
        self.vessel, self.stage4 = ig.load_segmentation_models(
            {"stage3_lwnet": local(parent["checkpoints"]["stage3_lwnet"]), "stage4_unet": local(parent["checkpoints"]["stage4_unet"])}, self.settings)
        log("loading Stage 1 (image quality) ...")
        self.iqa = iqa.load_iqa_model(os.path.join(MODEL_ROOT, "exported_models", "IQA", "best_model.keras"))
        self._local, self._protocol = local, protocol
        self._arrays = pl.load_reference(local(parent["checkpoints"]["convnext_pretrained"]))[1]
        self.arm, self.graders = None, {}
        self.load_arm(DEFAULT_ARM, log)
        self.lock = threading.Lock()
        log("models ready")

    def load_arm(self, arm, log=print):
        """Only ONE arm's three graders are held at a time (memory); switching arm releases the other."""
        import gc

        import gradcam
        if arm == self.arm:
            return
        self.graders.clear()
        gc.collect()
        model = {"direct": "j", "crop": "jc"}[arm]
        log(f"loading the three {arm} joint graders ...")
        self.graders[arm] = [gradcam.load_explainer(seed, self._local(self._protocol["checkpoints"][f"{model}_seed{seed}"]), self._arrays)[0]
                             for seed in (42, 123, 2026)]
        self.arm = arm


# --------------------------------------------------------------------------- one image, stage by stage

def run_pipeline(engine, image_path, arm, explanations, emit, reference=None):
    """Runs every stage on one image and calls `emit(stage)` as soon as each stage is done."""
    import cv2

    import explain
    import geometry_diagnostic as gd
    import gradcam
    import image_quality_inference as iqa
    import stage4_v2 as s4
    import stage4_v2_aptos_cache as ac
    import stage4_v2_data as sd
    import uncertainty_eval as ue

    def stage(number, title, text, images=(), started=None, data=None):
        emit({"number": number, "title": title, "text": text, "images": list(images), "seconds": round(time.time() - started, 1) if started else None,
              "data": data or {}})

    engine.load_arm(arm)
    t = time.time()
    raw = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    q = iqa.predict_quality(image_path, model=engine.iqa)
    stage(1, "Stage 1 - Image quality (EfficientNet-B0, EyeQ)",
          f"Quality: {q['label']} (confidence {q['confidence']:.2f}). " + ", ".join(f"{k} {v:.2f}" for k, v in q["probabilities"].items())
          + (" - a Reject image would normally be retaken; the remaining stages run anyway for the demonstration." if q["label"] == "Reject" else ""),
          [{"caption": "Uploaded image", "png": png(raw, size=420)}], t, {"quality": q})

    t = time.time()
    native = sd.stage2_rgb(image_path)
    stage(2, "Stage 2 - Preprocessing (gamma 1.2, then CLAHE on the L channel)",
          f"Native size {native.shape[1]} x {native.shape[0]}. Gamma correction brightens mid-tones; CLAHE raises local contrast so small lesions and vessels stand out.",
          [{"caption": "Before (raw)", "png": png(raw, size=420)}, {"caption": "After Stage 2", "png": png(native, size=420)}], t)

    t = time.time()
    box = gd.fundus_box(cv2.imread(image_path))
    frame = ac.recompute_rgb_512(native) if arm == "direct" else ac.recompute_rgb_512(np.ascontiguousarray(native[box[0]:box[1], box[2]:box[3]]))
    stage(3, f"Grader input frame ({arm} arm)",
          ("The whole Stage-2 image resized to 512 x 512." if arm == "direct" else
           f"The Stage-2 image cut to the fundus box (rows {box[0]}-{box[1]}, columns {box[2]}-{box[3]}) and resized to 512 x 512."),
          [{"caption": "Grader frame, 512 x 512", "png": png(frame, size=420)}], t)

    t = time.time()
    vessel = ac.recompute_vessel_512(native, engine.vessel)
    stage(4, "Stage 3 - Vessel segmentation (pretrained LWNet)", f"Vessel pixels above 0.5: {100 * float((vessel > 0.5).mean()):.1f} % of the frame. A supporting view: the grade does not use it.",
          [{"caption": "Vessel probability", "png": png(vessel, "gray", size=420)}], t)

    t = time.time()
    maps = np.asarray(s4.pathology_cache_maps(engine.stage4, native, device="cpu", amp=False))
    lesion_max = maps[..., 1::2].astype(np.float32) / 255.0             # channels: [class:mean, class:max]
    rgb512 = ac.recompute_rgb_512(native)
    burden = {name: float((lesion_max[..., i] > 0.5).mean()) for i, name in enumerate(LESIONS)}
    stage(5, "Stage 4 - Lesion segmentation (SE-ResNet-101 U-Net: MA, HE, EX, SE)",
          "Area above 0.5: " + ", ".join(f"{LESION_NAMES[k]} {100 * v:.2f} %" for k, v in burden.items()) + ". A supporting view: the grade does not use it.",
          [{"caption": "Lesions over the frame (yellow MA, red HE, cyan EX, magenta SE)", "png": png(lesion_overlay(rgb512, lesion_max), size=420)}]
          + [{"caption": f"{LESION_NAMES[n]}", "png": png(lesion_max[..., i], "magma", size=200)} for i, n in enumerate(LESIONS)], t, {"lesion_area": burden})

    t = time.time()
    graders = engine.graders[arm]
    fns = [explain.graph_model_fn(g, batch_size=2) for g in graders]
    logits = np.stack([f(frame[None]) for f in fns])                      # (3, 1, 4)
    e = ue.ensemble(logits)
    grade = int(e["pred"][0])
    probs = e["class_prob"][0]
    seeds = [int(x) for x in e["seed_pred"][:, 0]]
    ref = "" if reference is None else f" Reference grade: {reference['reference_grade']} ({GRADES[int(reference['reference_grade'])]}) - {reference['source']}, {reference['role']}."
    stage(6, "Grade - three-seed joint grader (ConvNeXt-Tiny + CORN), seed ensemble",
          f"Predicted grade {grade}: {GRADES[grade]}. Referable (grade 2 or worse): {'yes' if grade >= 2 else 'no'}. Per-seed grades {seeds}.{ref}",
          [], t, {"grade": grade, "grade_name": GRADES[grade], "probabilities": [float(p) for p in probs], "per_seed": seeds,
                  "expected_grade": float(e["p_gt"][0].sum()), "reference": reference})

    t = time.time()
    tta = [ue.tta_summary(ue.tta_logits(f, frame[None])) for f in fns]
    tta_sd = float(np.mean([x["expected_grade_sd"][0] for x in tta]))
    tta_change = bool(any(x["grade_changes"][0] for x in tta))
    stage(7, "Uncertainty - seed ensemble and test-time augmentation",
          f"Predictive entropy {float(e['entropy'][0]):.2f} (0 = certain, {np.log(5):.2f} = uniform). Seed disagreement (SD of expected grade) {float(e['seed_disagreement'][0]):.2f}; "
          f"the three seeds {'agree' if len(set(seeds)) == 1 else 'DISAGREE'}. Under 8 flips/rotations the expected grade varies by SD {tta_sd:.2f}"
          f"{' and the decoded grade CHANGES for at least one seed' if tta_change else '; the decoded grade does not change'}.",
          [], t, {"entropy": float(e["entropy"][0]), "seed_disagreement": float(e["seed_disagreement"][0]), "tta_sd": tta_sd, "tta_grade_changes": tta_change})

    titles = {"grad_cam": "Grad-CAM (3 seeds)","integrated_gradients": "Integrated Gradients (seed 42, 16 steps)",
              "occlusion": "Occlusion sensitivity (seed 42, 64-pixel patches)"}
    for k, name in enumerate(explanations):
        t = time.time()
        if name == "grad_cam":
            heat = np.mean([gradcam.grad_cam(g, frame[None])[0][0] for g in graders], axis=0)
        elif name == "integrated_gradients":
            heat = explain.integrated_gradients(graders[0], frame, steps=16, batch=2)[2]
        elif name == "occlusion":
            heat = explain.occlusion(fns[0], frame, patch=64, stride=64)[1]
        else:
            continue
        stage(8 + k / 10, f"Why this grade - {titles[name]}",
              "Where the model's expected grade is most sensitive (red = high). This shows sensitivity, not that the model detected lesions.",
              [{"caption": titles[name], "png": png(gradcam.overlay(frame, heat), size=420)}, {"caption": "map", "png": png(heat, "inferno", size=200)}], t)
    emit({"number": 99, "title": "Done", "text": NOTICE, "images": [], "seconds": None, "data": {}})


# --------------------------------------------------------------------------- the web page and its endpoints

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>DR grading pipeline - demo</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f4f6f8;color:#1d2733}
header{background:#14324a;color:#fff;padding:14px 24px}header h1{margin:0;font-size:20px}header p{margin:4px 0 0;font-size:13px;opacity:.85}
main{max-width:1180px;margin:0 auto;padding:18px}
.panel{background:#fff;border-radius:8px;padding:14px 18px;margin-bottom:14px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
label{margin-right:14px;font-size:14px}select,input,button{font-size:14px}button{background:#14324a;color:#fff;border:0;padding:8px 18px;border-radius:5px;cursor:pointer}
button:disabled{background:#8a9aa8}.stage h3{margin:0 0 6px;font-size:16px}.stage .t{color:#5b6b78;font-size:12px}
.imgs{display:flex;flex-wrap:wrap;gap:12px;margin-top:8px}.imgs figure{margin:0;text-align:center;font-size:12px;color:#44525e}
.grade{font-size:22px;font-weight:600}.bar{display:flex;align-items:center;gap:8px;font-size:13px;margin:2px 0}.bar span{display:inline-block;height:12px;background:#2e7bbd;border-radius:3px}
.notice{font-size:12px;color:#7a4b00;background:#fff4e0;padding:8px 12px;border-radius:6px}
</style></head><body>
<header><h1>Diabetic-retinopathy grading pipeline - live demonstration</h1>
<p>Stage 1 quality - Stage 2 preprocessing - Stage 3 vessels - Stage 4 lesions - joint ConvNeXt-Tiny + CORN grader (3 seeds) - uncertainty - explanations</p></header>
<main>
<div class="panel">
<div><label>Demo image (known grade): <select id="demo"><option value="">- choose -</option></select></label>
<label>or upload: <input type="file" id="file" accept="image/*"></label></div>
<div style="margin-top:10px"><label>Grader arm: <select id="arm"><option value="crop">fundus-cropped (preferred, chosen after the external re-test)</option><option value="direct">direct resize</option></select></label>
<label><input type="checkbox" class="ex" value="grad_cam" checked> Grad-CAM</label>
<label><input type="checkbox" class="ex" value="integrated_gradients"> Integrated Gradients (about 7 min on CPU)</label>
<label><input type="checkbox" class="ex" value="occlusion"> Occlusion (about 3 min on CPU)</label>
<button id="run">Run the pipeline</button> <span id="status"></span></div>
<p class="notice">Research prototype - not a medical device; its output is not a diagnosis. Explanation maps show where the score is sensitive; they do not prove that lesions were detected.</p>
</div>
<div id="out"></div>
</main>
<script>
const GR=["No DR","Mild NPDR","Moderate NPDR","Severe NPDR","Proliferative DR"];
fetch('/demo-list').then(r=>r.json()).then(list=>{const s=document.getElementById('demo');
 list.forEach(d=>{const o=document.createElement('option');o.value=d.file;o.textContent=`${d.source} - reference grade ${d.reference_grade} (${GR[d.reference_grade]}) - ${d.file}`;s.appendChild(o);});});
function card(st){const d=document.createElement('div');d.className='panel stage';
 let h=`<h3>${st.title}</h3>`;if(st.seconds!==null)h+=`<div class="t">${st.seconds} s</div>`;
 if(st.data&&st.data.grade!==undefined){h+=`<div class="grade">Grade ${st.data.grade}: ${st.data.grade_name}</div>`;
  st.data.probabilities.forEach((p,i)=>{h+=`<div class="bar">${GR[i].padEnd(16)} <span style="width:${Math.round(p*300)}px"></span> ${(p*100).toFixed(1)} %</div>`});}
 h+=`<p>${st.text}</p><div class="imgs">`;st.images.forEach(im=>{h+=`<figure><img src="data:image/png;base64,${im.png}"><figcaption>${im.caption}</figcaption></figure>`});
 d.innerHTML=h+'</div>';return d;}
document.getElementById('run').onclick=async()=>{
 const btn=document.getElementById('run'),out=document.getElementById('out'),st=document.getElementById('status');
 const body={arm:document.getElementById('arm').value,explanations:[...document.querySelectorAll('.ex:checked')].map(x=>x.value)};
 const f=document.getElementById('file').files[0],demo=document.getElementById('demo').value;
 if(f){body.filename=f.name;body.upload=await new Promise(res=>{const r=new FileReader();r.onload=()=>res(r.result.split(',')[1]);r.readAsDataURL(f)});}
 else if(demo){body.demo=demo}else{alert('Choose a demo image or upload one');return}
 btn.disabled=true;out.innerHTML='';st.textContent='running ...';
 const job=await (await fetch('/run',{method:'POST',body:JSON.stringify(body)})).json();
 if(job.error){st.textContent=job.error;btn.disabled=false;return}
 let shown=0;const poll=async()=>{const r=await (await fetch('/job/'+job.id)).json();
  while(shown<r.stages.length){out.appendChild(card(r.stages[shown]));shown++}
  if(r.error){st.textContent='error: '+r.error;btn.disabled=false;return}
  if(r.done){st.textContent='finished';btn.disabled=false}else{st.textContent=`running stage ${shown+1} ...`;setTimeout(poll,800)}};poll();};
</script></body></html>"""

ENGINE, JOBS = None, {}


def demo_list():
    path = os.path.join(DEMO_DIR, "labels.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return [{k: (int(v) if k == "reference_grade" else v) for k, v in r.items()} for r in csv.DictReader(fh)]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload, kind="application/json"):
        data = payload if isinstance(payload, bytes) else (payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8"))
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/":
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path == "/demo-list":
            return self._send(200, [{k: d[k] for k in ("file", "source", "reference_grade", "role")} for d in demo_list()])
        if self.path.startswith("/job/"):
            job = JOBS.get(self.path.split("/")[-1])
            return self._send(200, job if job else {"error": "unknown job", "stages": [], "done": True})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/run":
            return self._send(404, {"error": "not found"})
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if not ENGINE.lock.acquire(blocking=False):
            return self._send(200, {"error": "another image is still running; wait for it to finish"})
        reference = None
        if body.get("demo"):
            entry = next((d for d in demo_list() if d["file"] == body["demo"]), None)
            if entry is None:
                ENGINE.lock.release()
                return self._send(200, {"error": "unknown demo image"})
            path, reference = os.path.join(DEMO_DIR, entry["file"]), entry
        else:
            suffix = os.path.splitext(body.get("filename", "upload.png"))[1] or ".png"
            fd, path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as fh:
                fh.write(base64.b64decode(body["upload"]))
            digest = sha256_file(path)                                    # an uploaded copy of a demo image keeps its reference grade
            reference = next((d for d in demo_list() if d.get("sha256") == digest), None)
        job_id = uuid.uuid4().hex[:12]
        JOBS[job_id] = {"stages": [], "done": False, "error": None}
        arm = body.get("arm") if body.get("arm") in ("direct", "crop") else DEFAULT_ARM
        explanations = [x for x in body.get("explanations", []) if x in ("grad_cam", "integrated_gradients", "occlusion")]

        def work():
            try:
                run_pipeline(ENGINE, path, arm, explanations, lambda s: JOBS[job_id]["stages"].append(s), reference)
            except Exception as error:  # noqa: BLE001 -- shown on the page, and printed in the terminal
                traceback.print_exc()
                JOBS[job_id]["error"] = f"{type(error).__name__}: {error}"
            finally:
                JOBS[job_id]["done"] = True
                ENGINE.lock.release()
        threading.Thread(target=work, daemon=True).start()
        return self._send(200, {"id": job_id})


def main():
    global ENGINE
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    ENGINE = Engine()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"open http://localhost:{PORT}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
