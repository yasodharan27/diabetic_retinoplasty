"""End-to-end demonstration of the pipeline on ONE fundus image (research record §96). PREPARED, NOT RUN.

    image -> Stage 1 quality check -> Stage 2 preprocessing -> Stage 3 vessel map -> Stage 4 lesion maps
          -> the grader (three seeds) -> ensemble grade and probabilities -> uncertainty (seed ensemble and
          test-time augmentation) -> explanation maps (Grad-CAM, Integrated Gradients, Occlusion)

Every stage is the frozen component already used in the experiments, called through its existing function:
`image_quality_inference.predict_quality`, `idrid_grading_eval.image_inputs` (Stage 2, LWNet, the Stage-4
lesion model), the joint grader's checkpoints through `gradcam.load_explainer`, `uncertainty_eval.ensemble`.
Nothing is trained. The vessel and lesion maps are shown as supporting views; the grade comes from the RGB
grader alone (the project found no grading gain from feeding or supervising with those maps).

This is a research prototype. It is not a medical device and its output is not a diagnosis.
"""
import json
import os

import numpy as np

GRADES = ("No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR")
EXPLANATIONS = ("grad_cam", "integrated_gradients", "occlusion")
EXPLANATION_TITLES = {"grad_cam": "Grad-CAM", "integrated_gradients": "Integrated Gradients",
                      "occlusion": "Occlusion sensitivity"}
LESIONS = ("MA", "HE", "EX", "SE")                 # the Stage-4 channel order of the project
NOTICE = "Research prototype - not a medical device; not a diagnosis."


class Pipeline:
    """Loads every frozen model once; `run(image_path)` returns the result of one image."""

    def __init__(self, grader_checkpoints, pretrained_path, stage3_path, stage4_path, iqa_path=None, arm="direct",
                 device=None, uncertainty_threshold=None):
        """`grader_checkpoints`: {seed: weights path} of the final grader (one joint arm, three seeds).
        `arm`: 'direct' or 'crop' -- the framing those checkpoints were trained with.
        `uncertainty_threshold`: entropy above which the image is flagged for review; set it from the Stage-10
        validation results (None: the scores are reported without a flag)."""
        import pl_convnext as pl
        import torch

        import gradcam
        import idrid_grading_eval as ig
        self.arm, self.uncertainty_threshold = arm, uncertainty_threshold
        self.settings = {"device": device or ("cuda" if torch.cuda.is_available() else "cpu"), "stage4_amp": bool(torch.cuda.is_available())}
        self.vessel, self.stage4 = ig.load_segmentation_models({"stage3_lwnet": stage3_path, "stage4_unet": stage4_path}, self.settings)
        self.iqa = None
        if iqa_path:
            import image_quality_inference as iqa
            self.iqa = iqa.load_iqa_model(iqa_path)
        _, arrays = pl.load_reference(pretrained_path)
        self.seeds = sorted(grader_checkpoints)
        self.explainers = [gradcam.load_explainer(seed, grader_checkpoints[seed], arrays)[0] for seed in self.seeds]

    def frame(self, image_path, native):
        """The grader's input frame for this arm, from the Stage-2 image already computed."""
        import stage4_v2_aptos_cache as ac
        if self.arm == "direct":
            return ac.recompute_rgb_512(native)
        import cv2

        import geometry_diagnostic as gd
        top, bottom, left, right = gd.fundus_box(cv2.imread(image_path))
        return ac.recompute_rgb_512(np.ascontiguousarray(native[top:bottom, left:right]))

    def run(self, image_path, explanations=EXPLANATIONS, ig_steps=32):
        """One image. `explanations`: which maps to compute (each the mean of the three seeds' maps), all of the
        expected grade."""
        import explain
        import gradcam
        import idrid_grading_eval as ig
        import uncertainty_eval as ue
        result = {"image": os.path.basename(image_path), "notice": NOTICE}
        if self.iqa is not None:
            import image_quality_inference as iqa
            result["quality"] = iqa.predict_quality(image_path, model=self.iqa)
        inputs, native = ig.image_inputs(image_path, self.vessel, self.stage4, self.settings)
        rgb = self.frame(image_path, native)
        cam, logits = gradcam.ensemble_cam(self.explainers, rgb[None])
        e = ue.ensemble(logits)                                          # logits: (seeds, 1, 4)
        fns = [explain.graph_model_fn(g) for g in self.explainers]
        tta = [ue.tta_summary(ue.tta_logits(f, rgb[None])) for f in fns]
        maps = {"grad_cam": cam[0]}
        if "integrated_gradients" in explanations:
            maps["integrated_gradients"] = np.mean([explain.integrated_gradients(g, rgb, steps=ig_steps)[2] for g in self.explainers], axis=0)
        if "occlusion" in explanations:
            maps["occlusion"] = np.mean([explain.occlusion(f, rgb)[1] for f in fns], axis=0)
        maps = {k: v for k, v in maps.items() if k in explanations}
        grade = int(e["pred"][0])
        k = len(LESIONS)
        lesion_maps = np.asarray(inputs["maps"])
        result.update({
            "grade": grade, "grade_name": GRADES[grade], "class_probabilities": [float(p) for p in e["class_prob"][0]],
            "referable": bool(grade >= 2), "per_seed_grades": {str(s): int(g) for s, g in zip(self.seeds, e["seed_pred"][:, 0])},
            "uncertainty": {"entropy": float(e["entropy"][0]), "seed_disagreement": float(e["seed_disagreement"][0]),
                            "seeds_agree": bool(len(set(e["seed_pred"][:, 0].tolist())) == 1),
                            "tta_expected_grade_sd": float(np.mean([t["expected_grade_sd"][0] for t in tta])),
                            "tta_grade_changes": bool(any(t["grade_changes"][0] for t in tta))},
            "explanations": list(maps),
            "lesion_burden": {name: float(lesion_maps[..., 2 * i].mean() / 255.0) for i, name in enumerate(LESIONS)}})   # channels: [c:mean, c:max] per class
        if self.uncertainty_threshold is not None:
            result["flag_for_review"] = bool(result["uncertainty"]["entropy"] > self.uncertainty_threshold or not result["uncertainty"]["seeds_agree"])
        lesion = np.asarray(inputs["maps"])
        views = {"frame": rgb, "vessel": np.asarray(inputs["vessel"])[..., 0], "lesion_max": lesion[..., 1::2].astype(np.float32) / 255.0,
                 "explanations": {name: gradcam.overlay(rgb, m) for name, m in maps.items()}, "maps": maps}
        return result, views


def figure(result, views, path):
    """One summary picture: the frame, the vessel map, the lesion maps, the Grad-CAM overlay and the reading."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(18, 13.5))
    panels = [("Input frame (Stage 2)", views["frame"], None), ("Vessel map (Stage 3)", views["vessel"], "gray")]
    panels += [(f"Lesion map: {name} (Stage 4)", views["lesion_max"][..., i], "magma") for i, name in enumerate(LESIONS)]
    panels += [(f"{EXPLANATION_TITLES[name]} (expected grade)", picture, None) for name, picture in views["explanations"].items()]
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, image, cmap) in zip(axes.ravel(), panels):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None)
        ax.set_title(title, fontsize=11)
    quality = result.get("quality", {}).get("label", "not assessed")
    u = result["uncertainty"]
    flag = "" if "flag_for_review" not in result else ("  |  FLAGGED FOR REVIEW" if result["flag_for_review"] else "  |  not flagged")
    probabilities = "  ".join(f"{g.split()[0]} {p:.2f}" for g, p in zip(GRADES, result["class_probabilities"]))
    fig.suptitle(f"{result['image']}   |   quality: {quality}   |   grade {result['grade']}: {result['grade_name']}"
                 f"   |   referable: {'yes' if result['referable'] else 'no'}\n{probabilities}\n"
                 f"ensemble entropy {u['entropy']:.2f}   seed grades {list(result['per_seed_grades'].values())}   "
                 f"TTA spread {u.get('tta_expected_grade_sd', float('nan')):.2f}{flag}\n"
                 f"Explanation maps show where the score is sensitive; they do not show that lesions were detected.   {NOTICE}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run_folder(pipeline, image_paths, out_dir, log=print):
    """Runs the pipeline on several images; writes one picture and one JSON per image and a summary table."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for path in image_paths:
        result, views = pipeline.run(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        figure(result, views, os.path.join(out_dir, f"{stem}.png"))
        with open(os.path.join(out_dir, f"{stem}.json"), "w") as fh:
            json.dump(result, fh, indent=1)
        rows.append({"image": result["image"], "quality": result.get("quality", {}).get("label"), "grade": result["grade"],
                     "grade_name": result["grade_name"], "entropy": result["uncertainty"]["entropy"],
                     "seeds_agree": result["uncertainty"]["seeds_agree"], "flag_for_review": result.get("flag_for_review")})
        log(f"  {result['image']}: grade {result['grade']} ({result['grade_name']})")
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return rows
