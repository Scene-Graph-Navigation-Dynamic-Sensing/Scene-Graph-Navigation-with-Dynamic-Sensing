"""Draws the two architecture diagrams (the repo had no source for the 2026-09-24 versions):

  architecture_pipeline.png  the static scene-graph pipeline (scripts/run_static_scene_graph.py), stage by stage
  architecture_fusion.png    track-level instance fusion (object_fusion/instance_fusion.py): where a detection goes

Edit the stage lists below and re-run to update them:

    python scripts/draw_architecture.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
BG = "#F3F3F1"
BAND = "#E9EAE6"
INK = "#2B2F33"
SUBTLE = "#5F666D"
BLUE_EDGE = "#3A55B4"
BLUE_FILL = "#E4E9F7"
BLUE_INK = "#1F2F73"
ORANGE = "#B85A12"
GREEN = "#1F7A5C"
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"


class Canvas:
    def __init__(self, w, h):
        self.fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(h, 0)
        self.ax.axis("off")
        self.fig.patch.set_facecolor(BG)

    def text(self, x, y, s, size=13, color=INK, weight="normal", family=SANS, ha="left", va="center", **kw):
        self.ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, ha=ha, va=va, **kw)

    def box(self, x, y, w, h, title, sub="", dashed=False, edge=INK, fill="white", title_color=INK, lw=1.8):
        self.ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=7", lw=lw,
                                         ec=edge, fc=fill, ls=(0, (5, 3)) if dashed else "-"))
        if sub:
            self.text(x + 22, y + h * 0.36, title, 14.5, title_color, "bold")
            self.text(x + 22, y + h * 0.70, sub, 12, SUBTLE if title_color == INK else BLUE_INK, family=MONO)
        else:
            self.text(x + 22, y + h / 2, title, 14.5, title_color, "bold")

    def arrow(self, x0, y0, x1, y1, color=INK, dashed=False, lw=1.8):
        self.ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=lw,
                                          color=color, ls=(0, (5, 3)) if dashed else "-", shrinkA=0, shrinkB=0))

    def line(self, xs, ys, color=INK, dashed=False, lw=1.8):
        self.ax.plot(xs, ys, color=color, lw=lw, ls=(0, (5, 3)) if dashed else "-", solid_capstyle="butt")

    def save(self, path):
        self.fig.savefig(path, facecolor=BG)
        plt.close(self.fig)


# ─── pipeline ────────────────────────────────────────────────────────────

# (band, number, title, subtitle, dashed (DoVSG code), written-to-disk lines, model (title, subtitle) or None,
#  label on the arrow INTO this stage)
PIPELINE = [
    ("CAPTURE", "", "Frame source", "stream · video file · dataset folder", False,
     ["data/raw/<name>/", "rgb · depth · poses · camera.yaml"], None, None),
    ("CAPTURE", "2", "Depth + capture", "resize to target width · back-project", False,
     ["rgb/ point/ mask/", "capture.json"], ("Depth Anything V3 metric", "or dataset depth (reference)"),
     "RGB frames · stride, max_frames"),
    ("CAPTURE", "3", "Camera pose", "one 4×4 camera-to-world per frame", False,
     ["poses/*.txt", "<scene>_pose_eval.json"], ("DROID-SLAM (subprocess)", "or Open3D ICP · or dataset poses"),
     "metric depth (m) + intrinsics K"),
    ("MAP", "4a", "Gravity alignment", "RANSAC floor plane → z up, floor z = 0", False,
     ["gravity_alignment.json"], None, "camera-to-world poses"),
    ("MAP", "4", "Voxel map (worker processes)", "every depth pixel → 1 cm voxel + colour", True,
     [], None, "z-up poses"),
    ("STREAMING · detection, embedding and fusion run as parallel threads", "5", "Panoptic segment",
     "one segment per pixel · wall/floor/ceiling dropped", False, ["viz/detection/"],
     ("Mask2Former Swin-L panoptic", "ADE20K · score threshold 0.5"),
     "frames (optional gamma brightness for dark frames)"),
    ("STREAMING · detection, embedding and fusion run as parallel threads", "5a", "Small-object union",
     "YOLOE masks ≤ 2% of frame, off panoptic things", False, [],
     ("YOLOE-26x-seg-pf", "1024 px · stuff + group checks"), "panoptic thing segments (disjoint)"),
    ("STREAMING · detection, embedding and fusion run as parallel threads", "5b", "Track",
     "BoT-SORT ids on the disjoint masks", False, [], None, "panoptic + small YOLOE masks"),
    ("STREAMING · detection, embedding and fusion run as parallel threads", "5c", "Crop + embed",
     "padded box crop → embedding, every detection", False, [],
     ("NOVIC image embedder", "CLIP DFN5B ViT-H-14 · 1+ GPUs"), "tracked detections"),
    ("STREAMING · detection, embedding and fusion run as parallel threads", "6", "Track accumulation",
     "same track id → one pending object", False, ["track_commits.json"], None,
     "embedding per detection · mask → voxels"),
    ("STREAMING · detection, embedding and fusion run as parallel threads", "6a", "Thing/stuff check",
     "track's avg embedding: p(stuff) > 0.72 → reject", False, [],
     ("Thing/stuff probe", "logistic · all Replica + 3RScan"), "pending track ends"),
    ("STREAMING · detection, embedding and fusion run as parallel threads", "6", "Track commit (unseen 10 frames)",
     "0.3 overlap + 0.7 cosine ≥ 0.55 → merge / new", True, ["track_names.json"],
     ("NOVIC decoder", "names the object at track end"), "kept track, voxels + avg embedding"),
    ("3D INSTANCES", "6", "Final fusion", "re-match · DBSCAN · overlap merge · voxel owner", True,
     [], None, "map objects"),
    ("3D INSTANCES", "6b", "Cleanup", "drop floor slabs · merge fragments · fill tops", False,
     [], None, "fused 3D instances"),
    ("3D INSTANCES", "7", "Instance naming", "track-end name · label mapper · confidence", False,
     ["viz/semantic_decoding/"], ("NOVIC decoder", "only changed / untracked objects"), "cleaned instances"),
    ("SCENE GRAPH", "9", "Spatial relations", "on · belong · inside (DoVSG rules)", False,
     [], None, "class_name · class_id · embedding"),
    ("SCENE GRAPH", "", "Save scene graph", "nodes + embeddings + voxel map", False,
     ["scene_graphs/<scene>.pkl", "scene_graphs/<scene>.json"], None, "floor-rooted tree"),
    ("QUERY · SEPARATE SCRIPTS", "", "Task planning · scene update", "run_task_planning.py · update_scene_graph.py",
     False, ["navigation target (x, y, z)", "updated scene graph"],
     ("Gemini 2.5 Flash", "+ NOVIC text-similarity tool"), "saved .pkl"),
]


def draw_pipeline(path):
    W, BOX_H, GAP, BAND_PAD = 1920, 100, 58, 44
    X_STAGE, W_STAGE, X_MODEL, W_MODEL, X_DISK = 720, 560, 1380, 510, 680
    bands = []
    y = 150
    rows = []
    for i, (band, *_rest) in enumerate(PIPELINE):
        if not bands or bands[-1][0] != band:
            if bands:
                y += BAND_PAD
            bands.append([band, y - 20, None])
            y += 40
        rows.append(y)
        y += BOX_H + GAP
        bands[-1][2] = y - GAP + 24
    H = y + 120
    c = Canvas(W, H)
    c.text(32, 44, "Static scene graph pipeline: run_static_scene_graph.py", 22, INK, "bold")
    c.text(X_DISK - 155, 118, "WRITTEN TO DISK", 12, SUBTLE, "bold", ha="center")
    c.text(X_STAGE + W_STAGE / 2, 118, "PIPELINE STAGE", 12, SUBTLE, "bold", ha="center")
    c.text(X_MODEL + 40, 118, "MODEL", 12, SUBTLE, "bold")
    for name, y0, y1 in bands:
        c.ax.add_patch(FancyBboxPatch((16, y0), W - 32, y1 - y0, boxstyle="round,pad=0,rounding_size=10",
                                      fc=BAND, ec="none"))
        c.text(44, y0 + 26, name, 12, SUBTLE, "bold")

    commit_y = name_y = None
    for (band, num, title, sub, dashed, disk, model, label), yy in zip(PIPELINE, rows):
        c.box(X_STAGE, yy, W_STAGE, BOX_H, f"{num} · {title}" if num else title, sub, dashed)
        for k, line in enumerate(disk):
            c.text(X_DISK, yy + 34 + 32 * k, line, 12.5, INK, family=MONO, ha="right")
        if model:
            c.box(X_MODEL, yy, W_MODEL, BOX_H, model[0], model[1], edge=BLUE_EDGE, fill=BLUE_FILL, title_color=BLUE_INK)
            c.line([X_STAGE + W_STAGE, X_MODEL], [yy + BOX_H / 2] * 2, BLUE_EDGE)
        if label:
            c.arrow(X_STAGE + W_STAGE / 2, yy - GAP, X_STAGE + W_STAGE / 2, yy)
            c.text(X_STAGE + W_STAGE / 2 + 16, yy - GAP / 2, label, 12, SUBTLE)
        if title.startswith("Track commit"):
            commit_y = yy
        if title == "Instance naming":
            name_y = yy

    # side channel: names decided at track end reach the naming stage directly
    xs = X_STAGE + W_STAGE + 52
    c.line([X_STAGE + W_STAGE, xs, xs], [commit_y + 78, commit_y + 78, name_y + 70], ORANGE, dashed=True)
    c.arrow(xs, name_y + 70, X_STAGE + W_STAGE + 2, name_y + 70, ORANGE, dashed=True)
    mid = (commit_y + name_y) / 2 + 40
    for k, s in enumerate(["NOVIC name of each object,", "decoded when a track ends;", "re-decoded only if the object",
                           "changed afterwards"]):
        c.text(xs + 22, mid - 30 + 30 * k, s, 12.5, ORANGE)

    # legend
    ly = H - 52
    c.ax.add_patch(FancyBboxPatch((40, ly - 12), 52, 24, boxstyle="round,pad=0,rounding_size=5", fc="white", ec=INK, lw=1.8))
    c.text(104, ly, "Code in this repo", 13)
    c.ax.add_patch(FancyBboxPatch((360, ly - 12), 52, 24, boxstyle="round,pad=0,rounding_size=5", fc="white", ec=INK,
                                  lw=1.8, ls=(0, (5, 3))))
    c.text(424, ly, "DoVSG code reused (Scene-Graph-Generation), adapted", 13)
    c.ax.add_patch(FancyBboxPatch((1010, ly - 12), 52, 24, boxstyle="round,pad=0,rounding_size=5", fc=BLUE_FILL,
                                  ec=BLUE_EDGE, lw=1.8))
    c.text(1074, ly, "Learned model or external service", 13)
    c.line([1440, 1490], [ly, ly], ORANGE, dashed=True)
    c.text(1502, ly, "Side channel that skips stages", 13)
    c.save(path)


# ─── fusion ──────────────────────────────────────────────────────────────

def diamond(c, cx, cy, w, h, lines):
    c.ax.add_patch(Polygon([(cx, cy - h / 2), (cx + w / 2, cy), (cx, cy + h / 2), (cx - w / 2, cy)], closed=True,
                           fc="white", ec=INK, lw=1.8))
    for k, s in enumerate(lines):
        c.text(cx, cy + (k - (len(lines) - 1) / 2) * 26, s, 14, ha="center")


def draw_fusion(path):
    W, H = 2200, 1180
    c = Canvas(W, H)
    c.text(32, 44, "Instance fusion: where each detection goes (track level)", 22, INK, "bold")
    c.text(32, 84, "object_fusion/instance_fusion.py · per frame on the fusion thread, then once at the end of the video",
           13, SUBTLE)

    # per-frame part
    c.box(40, 300, 330, 110, "Detection, frame t", "mask voxels · embedding · id")
    diamond(c, 520, 355, 240, 150, ["track id?"])
    c.arrow(370, 355, 400, 355)
    # untracked
    c.line([520, 520], [280, 200], INK)
    c.arrow(520, 200, 760, 200)
    c.text(505, 222, "none (−1):", 12.5, SUBTLE, ha="right")
    c.text(505, 246, "one-detection track", 12.5, SUBTLE, ha="right")
    # same track id
    c.arrow(640, 355, 760, 355)
    c.text(700, 385, "known /", 12.5, SUBTLE, ha="center")
    c.text(700, 409, "new id", 12.5, SUBTLE, ha="center")
    c.box(760, 300, 470, 110, "Pending track (one per id)", "join voxels · running-avg embedding")
    c.text(780, 440, "optional: split the track when a detection's", 12, SUBTLE)
    c.text(780, 464, "cosine to it drops below track_split_cosine", 12, SUBTLE)
    diamond(c, 1400, 355, 250, 150, ["unseen for", "10 frames?"])
    c.arrow(1230, 355, 1275, 355)
    c.text(1400, 450, "no · wait for more frames", 12.5, SUBTLE, ha="center")

    # commit
    c.box(760, 150, 470, 100, "Commit now", "no tracker identity to wait for")
    c.line([1230, 1660, 1660], [200, 200, 355], INK)
    c.line([1525, 1660], [355, 355], INK)
    c.text(1540, 335, "yes", 12.5, SUBTLE)
    c.arrow(1660, 355, 1660, 560)
    c.text(1680, 500, "first: thing/stuff probe on the avg embedding,", 12, SUBTLE)
    c.text(1680, 524, "p(stuff) > 0.72 → track rejected (wall / floor)", 12, SUBTLE)
    c.box(1430, 560, 700, 110, "Commit: score against every map object",
          "DBSCAN clean · 0.3·overlap + 0.7·cosine")
    diamond(c, 1780, 790, 380, 170, ["agg ≥ 0.55 and one", "≥ 30% inside the other?", "(saved objects: ≥ 0.75)"])
    c.arrow(1780, 670, 1780, 705)
    c.text(1990, 735, "optional gates:", 12, SUBTLE)
    c.text(1990, 760, "min_merge_cosine", 12, SUBTLE, family=MONO)
    c.text(1990, 785, "small_merge_ratio", 12, SUBTLE, family=MONO)
    c.box(1180, 920, 520, 110, "Merge into best object", "count-weighted avg embedding", edge=GREEN, lw=2.4)
    c.box(1760, 920, 400, 110, "New map object", "added to the map")
    c.line([1590, 1440, 1440], [790, 790, 920], INK)
    c.ax.add_patch(FancyArrowPatch((1440, 900), (1440, 920), arrowstyle="-|>", mutation_scale=18, color=INK, lw=1.8))
    c.text(1450, 770, "yes · best match", 12.5, SUBTLE)
    c.arrow(1780, 875, 1960, 920)
    c.text(1880, 885, "no", 12.5, SUBTLE)
    c.box(700, 1060, 760, 90, "NOVIC decodes the object's running average → track_names", "",
          edge=BLUE_EDGE, fill=BLUE_FILL, title_color=BLUE_INK)
    c.arrow(1440, 1030, 1440, 1060, BLUE_EDGE)
    c.line([1960, 1960, 1460], [1030, 1105, 1105], BLUE_EDGE)

    # end of video
    c.ax.add_patch(FancyBboxPatch((40, 560), 1080, 410, boxstyle="round,pad=0,rounding_size=10", fc=BAND, ec="none"))
    c.text(64, 592, "END OF VIDEO (stage 6, final fusion)", 12, SUBTLE, "bold")
    steps = [("1", "Commit every track still open", "same rule as above"),
             ("2", "Re-match pass, smallest object first", "merge if it now matches (optional: ≥ 70% inside)"),
             ("3", "DBSCAN over all objects · drop < 3 detections", "obj_min_detections"),
             ("4", "Overlap merge (merge_objects)", "objects that overlap and look alike"),
             ("5", "Voxel ownership (indexes_align_objects)", "a voxel claimed twice goes to one object")]
    for k, (n, t, s) in enumerate(steps):
        yy = 620 + 66 * k
        c.text(70, yy + 14, f"{n}.", 14, INK, "bold")
        c.text(100, yy + 14, t, 14, INK, "bold")
        c.text(100, yy + 40, s, 12, SUBTLE, family=MONO)
    c.text(64, 1010, "Every 20 frames during streaming: DBSCAN clean-up of all objects built so far.", 12.5, SUBTLE)
    c.save(path)


def main():
    draw_pipeline(REPO / "docs" / "architecture" / "architecture_pipeline.png")
    draw_fusion(REPO / "docs" / "architecture" / "architecture_fusion.png")
    print("wrote docs/architecture/architecture_pipeline.png, architecture_fusion.png")


if __name__ == "__main__":
    main()
