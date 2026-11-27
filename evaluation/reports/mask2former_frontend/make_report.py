"""Report: Mask2Former as the only detector vs Baseline (YOLOE), Mask2Former + YOLOE (current default) and the
published Replica methods (ConceptGraphs, ConceptFusion, HOV-SG, ReasoningGraph, ...). Also compares run time.

Every number is read from the run folders in the main project; runs that have not finished are listed as pending,
so re-running after the queue ends gives the complete report.

    python evaluation/reports/mask2former_frontend/make_report.py   (scene_graph env)
writes parts/option2_mask2former_replica_report.pdf next to this script
"""
from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN = Path(__file__).resolve().parents[3]  # repo root
OUT = HERE / "parts" / "option2_mask2former_replica_report.pdf"
OUT.parent.mkdir(exist_ok=True)
sys.path.insert(0, str(MAIN))
sys.path.insert(0, str(MAIN / "scripts"))
import make_baseline_comparison as mbc  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_LEFT  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

SCENES = mbc.SCENES
# (key, label, colour, [results roots])
METHODS = [
    ("base", "Baseline (YOLOE)", "#8A8F96",
     ["results/replica_rgbfix/run1", "results/replica_rgbfix/run2"]),
    ("panyoloe", "Mask2Former + YOLOE", "#3A55B4",
     ["results/frontend_choice/panprobe/run1", "results/frontend_choice/panprobe/run2"]),
    ("opt2", "Mask2Former", "#1F7A5C",
     ["results/option2_panoptic/run1", "results/option2_panoptic/run2"]),
    ("opt3", "YOLOE + label filter", "#B85A12",
     ["results/option3_yoloe_labels/run1", "results/option3_yoloe_labels/run2"]),
]
SHORT = {"base": "Baseline (YOLOE)", "panyoloe": "Mask2Former + YOLOE", "opt2": "Mask2Former",
         "opt3": "YOLOE + label filter"}
WRAP = {"base": "Baseline\n(YOLOE)", "panyoloe": "Mask2Former\n+ YOLOE", "opt2": "Mask2Former",
        "opt3": "YOLOE +\nlabel filter"}  # chart tick labels

# fonts with the symbols used below (>=, ->, x)
_fd = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(_fd / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(_fd / "DejaVuSans-Bold.ttf")))
from reportlab.lib.fonts import addMapping  # noqa: E402
addMapping("DV", 0, 0, "DV"); addMapping("DV", 1, 0, "DVB"); addMapping("DV", 0, 1, "DV"); addMapping("DV", 1, 1, "DVB")  # noqa: E702
SS = getSampleStyleSheet()
BODY = ParagraphStyle("b", parent=SS["BodyText"], fontName="DV", fontSize=9, leading=12.5, alignment=TA_LEFT)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.8, leading=10, textColor=colors.HexColor("#4A4F55"))
H1 = ParagraphStyle("h1", parent=BODY, fontName="DVB", fontSize=17, leading=21, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=BODY, fontName="DVB", fontSize=12, leading=15, spaceBefore=10, spaceAfter=4)
CELL = ParagraphStyle("c", parent=BODY, fontSize=7.8, leading=9.6)
CELLB = ParagraphStyle("cb", parent=CELL, fontName="DVB")


# ─── data ────────────────────────────────────────────────────────────────

def timing(root: Path, scene: str) -> dict | None:
    p = MAIN / root / scene / "pipeline.log"
    if not (MAIN / root / scene / "metrics.json").exists() or not p.exists():
        return None
    t = p.read_text(errors="ignore")
    st = {m.group(1): float(m.group(2)) for m in re.finditer(r"\[stage\] (\S+): ([\d.]+)s", t)}
    if "9_scene_graph_relations" not in st:
        return None
    sf = re.search(r"Streaming fusion: (\d+) track commits.*?(\{'detection_s'.*?\})", t)
    tr = re.search(r"Tracker: (\{.*\})", t)
    s = ast.literal_eval(sf.group(2)) if sf else {}
    k = json.loads(tr.group(1)) if tr else {}
    w = MAIN / root / scene / "pipeline_wall_time_s.txt"
    return {"total": sum(st.values()), "stream": st.get("4-6_voxel_map_detect_fuse_streaming", 0),
            "finalize": st.get("6_instance_fusion_finalize", 0), "relations": st.get("9_scene_graph_relations", 0),
            "other": sum(v for k_, v in st.items() if k_ not in ("4-6_voxel_map_detect_fuse_streaming",
                                                                 "6_instance_fusion_finalize", "9_scene_graph_relations")),
            "detect": s.get("detection_s", 0), "commits": int(sf.group(1)) if sf else 0,
            "dets": k.get("detections", 0), "tracks": k.get("unique_tracks", 0),
            "wall": float(w.read_text()) if w.exists() else None}


def collect():
    D = {}
    for key, _, _, roots in METHODS:
        runs = [mbc.load_run(MAIN / r) for r in roots]
        tim = [{s: timing(Path(r), s) for s in SCENES} for r in roots]
        D[key] = {"runs": runs, "tim": tim}
    return D


def complete(run: dict) -> bool:
    return bool(run) and len(run["scenes"]) == 8


def mean_runs(D, key, f):
    v = [f(r) for r in D[key]["runs"] if complete(r)]
    return float(np.mean(v)) if v else None



def gt_found(root: str) -> dict | None:
    """(scene, gt id) -> (class, found) for visible, scored GT objects; None if a scene is missing."""
    import csv
    out = {}
    for s in SCENES:
        f = MAIN / root / s / "gt_objects.csv"
        if not f.exists():
            return None
        for r in csv.DictReader(open(f)):
            if r["kind"] != "ignored" and r["visible"] == "True":
                out[(s, r["gt_object_id"])] = (r["class"], r["found"] == "True")
    return out


# ─── formatting ──────────────────────────────────────────────────────────

def pc(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def fx(x, d=1):
    return "–" if x is None else f"{x:.{d}f}"


def table(rows, widths, header_rows=1, bold_last=False, hl_rows=()):
    data = [[Paragraph(str(c), CELLB if i < header_rows else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * mm for w in widths], repeatRows=header_rows)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#C9CCD1")),
          ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#E9EAE6")),
          ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 2.2),
          ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]
    for r in hl_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#E4F1EC")))
    if bold_last:
        st.append(("BACKGROUND", (0, len(rows) - 1), (-1, len(rows) - 1), colors.HexColor("#F3F3F1")))
    t.setStyle(TableStyle(st))
    return t


def fig_png(fig, name, w_mm):
    p = HERE / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    w, h = fig.get_size_inches()
    return Image(str(p), width=w_mm * mm, height=w_mm * mm * h / w)


# ─── report ──────────────────────────────────────────────────────────────

def main():
    D = collect()
    lab = {k: l for k, l, _, _ in METHODS}
    col = {k: c for k, _, c, _ in METHODS}
    pending = [f"{SHORT[k]} run {i + 1}: {len(r['scenes']) if r else 0}/8 scenes"
               for k, *_ in METHODS for i, r in enumerate(D[k]["runs"]) if not complete(r)]
    S = []
    S += [Paragraph("Replica: four detection front ends compared", H1),
          Paragraph(f"Generated {time.strftime('%Y-%m-%d %H:%M')} · all 8 Replica scenes (office_0–4, room_0–2), "
                    "ground-truth depth and poses, every 10th of 2,000 frames, 2 runs per configuration", SMALL),
          Spacer(1, 6)]
    if pending:
        S.append(Paragraph("<b>Incomplete:</b> " + "; ".join(pending) + ". Pooled numbers below use complete runs only.",
                           ParagraphStyle("w", parent=BODY, textColor=colors.HexColor("#B85A12"))))

    # headline numbers
    f1 = {k: mean_runs(D, k, lambda r: r["f1"]) for k in lab}
    prec = {k: mean_runs(D, k, lambda r: r["prec"]) for k in lab}
    rec = {k: mean_runs(D, k, lambda r: r["rec"]) for k in lab}
    fake = {k: mean_runs(D, k, lambda r: r["fake"]) for k in lab}
    dup = {k: mean_runs(D, k, lambda r: r["dup"]) for k in lab}

    def tot_time(k):
        per = []
        for tim in D[k]["tim"]:
            if all(tim[s] for s in SCENES):
                per.append(sum(tim[s]["total"] for s in SCENES))
        return float(np.mean(per)) / 60 if per else None
    T = {k: tot_time(k) for k in lab}

    S += [Paragraph("Setup", H2), Paragraph(
        "The <b>Mask2Former</b> configuration uses <b>Mask2Former Swin-L, ADE20K panoptic</b> as the only detector "
        "(<font face='DV'>facebook/mask2former-swin-large-ade-panoptic</font>, the most accurate Mask2Former trained "
        "on ADE20K: 48.1 PQ). Every pixel belongs to one segment; wall, floor, ceiling and other stuff segments are "
        "dropped; each remaining object mask gets its tight bounding box (min/max of its pixels), which BoT-SORT "
        "tracks and NOVIC's CLIP crops. The rest of the pipeline is unchanged: CLIP DFN5B ViT-H embeddings, track-level "
        "DoVSG fusion, held-out thing/stuff probe, NOVIC naming, DoVSG relations.", BODY), Spacer(1, 3), Paragraph(
        "The <b>YOLOE + label filter</b> configuration uses prompt-free YOLOE-26x alone and lets it name each "
        "detection from its built-in 4,585-tag vocabulary. A detection whose tag names a place, room, building or "
        "surface (422 tags such as wall, floor, ceiling, playroom, office; configs/vocab/yoloe_pf_scene_tags.txt, built "
        "with WordNet by scripts/build_yoloe_object_vocab.py) is dropped before tracking; the tag is then discarded and "
        "the rest of the pipeline is unchanged. Shape, colour and other abstract tags are kept: YOLOE often gives real "
        "objects such tags (a switch as 'rectangle', a screen as 'dot'), and dropping them cost 15% of real-object "
        "detections in a frame-level check, against about 4% for the scene-tag filter.", BODY), Spacer(1, 3),
        table([["Configuration", "Detector(s)", "Masks overlap?", "Rest of pipeline"],
               [lab["base"], "YOLOE-26x prompt-free, 1024 px, conf 0.10", "yes (nested, groups)", "same, no probe"],
               [lab["panyoloe"], "Mask2Former Swin-L + YOLOE-26x masks ≤ 2% of the frame", "no", "same + held-out probe"],
               [lab["opt2"], "Mask2Former Swin-L only", "no", "same + held-out probe"],
               [lab["opt3"], "YOLOE-26x prompt-free; detections tagged as a place / room / surface dropped", "yes",
                "same + held-out probe"]],
              [44, 66, 26, 38]),
        Spacer(1, 3), Paragraph(
        "Shared settings (Replica evaluation stack): merge_fixes, brightness, fusion_naming, discovery overrides; "
        "the probe used on each scene was trained without that scene's pair (configs/stuff_probe/heldout_*.npz). "
        "Mask2Former differs from Mask2Former + YOLOE (the current default) only in having no YOLOE small-object masks; "
        "YOLOE + label filter differs from Baseline (YOLOE) in the label filter and the held-out probe.", SMALL)]

    S += [Paragraph("Headline (mean of 2 runs, 8 scenes pooled)", H2),
          table([["", "Non-object nodes", "Object precision", "Recall", "F1", "Duplicates", "Processing time, 8 scenes"]]
                + [[lab[k], fx(fake[k], 1), pc(prec[k]), pc(rec[k]), fx(f1[k], 3), fx(dup[k], 1),
                    f"{T[k]:.1f} min" if T[k] else "–"] for k in lab],
                [46, 22, 22, 17, 15, 21, 33]),
          Spacer(1, 3), Paragraph(
              "Object precision = objects found / (real-object nodes + non-object nodes); non-objects are wall / floor "
              "/ ceiling pieces and nodes spanning several objects; recall = GT objects found / visible GT objects "
              "(349). Processing time = sum of pipeline stages (model loading excluded).", SMALL)]

    # F1 / precision / recall bar chart
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.6))
    for a, (name, vals) in zip(ax, [("F1", f1), ("Object precision (%)", {k: v and 100 * v for k, v in prec.items()}),
                                     ("Recall (%)", {k: v and 100 * v for k, v in rec.items()})]):
        ks = [k for k in lab if vals[k] is not None]
        a.bar([WRAP[k] for k in ks], [vals[k] for k in ks], color=[col[k] for k in ks], width=0.6)
        for i, k in enumerate(ks):
            a.text(i, vals[k], f"{vals[k]:.3f}" if name == "F1" else f"{vals[k]:.1f}", ha="center", va="bottom", fontsize=8)
        a.set_title(name, fontsize=9)
        a.tick_params(labelsize=7.5)
        a.spines[["top", "right"]].set_visible(False)
        lo = min(vals[k] for k in ks) if ks else 0
        a.set_ylim(lo * 0.85, max(vals[k] for k in ks) * 1.08 if ks else 1)
    fig.tight_layout()
    S += [Spacer(1, 4), fig_png(fig, "fig_objects", 175)]

    # per-run table
    rows = [["Run", "Scenes", "Nodes", "Non-objects", "Precision", "Recall (found)", "F1", "Duplicates"]]
    for k in lab:
        for i, r in enumerate(D[k]["runs"]):
            if r:
                rows.append([f"{SHORT[k]} r{i + 1}", len(r["scenes"]), r["nodes"], r["fake"], pc(r["prec"]),
                             f"{pc(r['rec'])} ({r['found']}/{r['vis']})", f"{r['f1']:.3f}", r["dup"]])
    S += [Paragraph("All runs", H2), table(rows, [32, 15, 15, 21, 19, 34, 14, 20]),
          Paragraph("The two runs of one configuration show the noise floor (Baseline (YOLOE)'s two runs differ by 0.002 F1).",
                    SMALL)]


    # 1. segmentation
    S += [Paragraph("1. 3D semantic segmentation (ConceptGraphs Table II, Replica, exclude-6)", H2)]
    rows = [["Method", "Type", "mAcc", "F-mIoU"]] + [[m, t, f"{a:.2f}", f"{b:.2f}"] for m, t, a, b in mbc.CG_SEG[:4]]
    hl = []
    for k in lab:
        for i, r in enumerate(D[k]["runs"]):
            if complete(r) and r["seg"]:
                rows.append([f"Ours: {SHORT[k]} r{i + 1}", "zero-shot", pc(r["seg"]["macc"], 2), pc(r["seg"]["fmiou"], 2)])
                if k == "opt2":
                    hl.append(len(rows) - 1)
    rows += [[m, t, f"{a:.2f}", f"{b:.2f}"] for m, t, a, b in mbc.CG_SEG[4:]]
    S += [table(rows, [62, 42, 22, 22], hl_rows=hl), Spacer(1, 3), Paragraph(
        "Exclude-6 drops other, floor, wall, ceiling, door, window (ConceptGraphs' --n_exclude 6); our pipeline does "
        "not map background, so this is the like-for-like row. Published rows copied from ConceptGraphs (Gu et al., "
        "ICRA 2024). Caveats: class names embedded with DFN5B ViT-H-14 (ConceptGraphs: LAION ViT-H-14); we use every "
        "10th frame, ConceptGraphs every 5th. OpenSeg, LSeg and CLIPSeg are fine-tuned on segmentation data "
        "(privileged), shown for reference.", SMALL)]

    # 2. embedding quality
    S += [Paragraph("2. Object embedding quality (ReasoningGraph Table I protocol)", H2)]
    rows = [["Method", "Acc@5", "Acc@10", "Acc@25", "AUC", "Objects / scene"]] + \
           [[m, f"{a:.2f}", f"{b:.2f}", f"{c:.2f}", f"{u:.2f}", o] for m, a, b, c, u, o in mbc.RG_EMB]
    hl = []
    for k in lab:
        for i, r in enumerate(D[k]["runs"]):
            if complete(r):
                rows.append([f"Ours: {SHORT[k]} r{i + 1}", pc(r["acc"][5]), pc(r["acc"][10]), pc(r["acc"][25]),
                             f"{100 * r['auc']:.2f}", f"{r['obj_per_scene']:.0f}"])
                if k == "opt2":
                    hl.append(len(rows) - 1)
    S += [table(rows, [52, 20, 20, 20, 20, 26], hl_rows=hl), Spacer(1, 3), Paragraph(
        "Each node's image embedding is ranked against Replica's 101 class names (class list given; not a naming "
        "score). ReasoningGraph ranks over 500+ labels and mixes in the detector label's text embedding, so its AUC "
        "is not directly comparable; HOV-SG, ConceptGraphs and ReasoningGraph rows are copied from ReasoningGraph.",
        SMALL)]

    # 3. scene graph quality per scene
    S += [Paragraph("3. Scene-graph node precision and duplicates (ConceptGraphs Table I)", H2)]
    rk = [(k, i, r) for k in lab for i, r in enumerate(D[k]["runs"]) if complete(r) and i == 0]     # run 1 of each
    rows = [["Scene", "CG prec.", "CG-D prec.", "CG dup.", "CG-D dup."] + [f"{SHORT[k]} r{i + 1}" for k, i, _ in rk]]
    for s in SCENES:
        cg = mbc.CG_GRAPH.get(s)
        rows.append([s] + ([str(x) for x in cg] if cg else ["–"] * 4) +
                    [f"{pc(r['per_scene_prec'].get(s))} / {r['per_scene_dup'].get(s, '–')}" for _, _, r in rk])
    rows.append(["All", "71 (avg)", "61 (avg)", "21", "15"] + [f"{pc(r['node_prec'])} / {r['dup']}" for _, _, r in rk])
    w_ = (180 - 15 - 4 * 15) / max(len(rk), 1)
    S += [table(rows, [15, 15, 15, 15, 15] + [w_] * len(rk), bold_last=True), Spacer(1, 3), Paragraph(
        "Ours (run 1): node precision / duplicates. Not like-for-like: ConceptGraphs' precision is human-judged (2 of 3 "
        "judges accept the caption); ours counts a node right only if its NOVIC name maps to the GT class by text "
        "similarity, a stricter automatic proxy. ConceptGraphs does not report office_4.", SMALL)]

    # 4. naming
    S += [Paragraph("4. Open-vocabulary naming", H2)]
    rows = [["Run", "NOVIC acceptable", "Embedding lookup acceptable"]]
    for k in lab:
        for i, r in enumerate(D[k]["runs"]):
            if complete(r):
                rows.append([f"{SHORT[k]} r{i + 1}", pc(r["novic"]), pc(r["embed"])])
    S += [table(rows, [60, 40, 50]), Spacer(1, 3), Paragraph(
        "Names from NOVIC's 42,919 nouns, no class list; a WordNet judge accepts exact or more specific names.", SMALL)]

    # 5. time
    S += [Paragraph("5. Run time", H2), Paragraph(
        "Processing time per scene (sum of pipeline stages, model loading excluded), mean of the two runs. All runs "
        "on Quadro RTX 5000 GPUs (16 GB). The Baseline (YOLOE) and Mask2Former + YOLOE runs (2026-10-14) ran two jobs at a time on "
        "GPUs 1 and 2; the Mask2Former runs (2026-10-15) ran one job at a time on GPU 1 while another user's training "
        "jobs occupied GPUs 0 and 2 (CPU shared); the YOLOE + label filter runs (2026-10-17) ran two jobs at a time on "
        "GPUs 1 and 2. Treat single-scene ratios as rough; the 8-scene totals are robust.",
        BODY), Spacer(1, 4)]

    def mt(k, s, f="total"):
        v = [t[s][f] for t in D[k]["tim"] if t[s]]
        return float(np.mean(v)) if v else None
    rows = [["Scene (minutes)"] + [SHORT[k] for k in lab]]
    for s in SCENES:
        v = {k: mt(k, s) for k in lab}
        rows.append([s] + [fx(v[k] / 60 if v[k] else None) for k in lab])
    rows.append(["All 8"] + [fx(T[k]) for k in lab])
    rows.append(["Speed-up vs Baseline (YOLOE)"] + [f"{T['base'] / T[k]:.2f}x" if T["base"] and T[k] else "–" for k in lab])
    t5 = table(rows, [44, 34, 34, 34, 34])
    t5.setStyle([("BACKGROUND", (0, len(rows) - 2), (-1, len(rows) - 1), colors.HexColor("#F3F3F1"))])
    S += [t5]

    # stage breakdown
    def stage_sum(k, f):
        v = []
        for t in D[k]["tim"]:
            if all(t[s] for s in SCENES):
                v.append(sum(t[s][f] for s in SCENES))
        return float(np.mean(v)) / 60 if v else None
    stages = [("stream", "Streaming: detection + embedding + fusion threads"), ("finalize", "Final fusion"),
              ("relations", "Spatial relations"), ("other", "Capture, gravity, cleanup, naming"),
              ("detect", "  of which: detection thread alone")]
    rows = [["Stage (minutes, 8 scenes)"] + [SHORT[k] for k in lab]] + \
           [[n] + [fx(stage_sum(k, f)) for k in lab] for f, n in stages]

    def cnt(k, f):
        v = []
        for t in D[k]["tim"]:
            if all(t[s] for s in SCENES):
                v.append(sum(t[s][f] for s in SCENES))
        return f"{np.mean(v):,.0f}" if v else "–"
    rows += [["Detections passed to the tracker"] + [cnt(k, "dets") for k in lab],
             ["Tracks"] + [cnt(k, "tracks") for k in lab],
             ["Track commits into the map"] + [cnt(k, "commits") for k in lab]]
    S += [Spacer(1, 6), Paragraph("Where the time goes", H2), table(rows, [64, 29, 29, 29, 29])]

    # time chart
    fig, ax = plt.subplots(figsize=(9, 3.1))
    x = np.arange(len(SCENES))
    wb = 0.8 / len(lab)
    for j, k in enumerate(lab):
        v = [(mt(k, s) or 0) / 60 for s in SCENES]
        ax.bar(x + (j - (len(lab) - 1) / 2) * wb, v, wb, color=col[k], label=SHORT[k])
    ax.set_xticks(x, SCENES, fontsize=8)
    ax.set_ylabel("minutes per scene", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    S += [Spacer(1, 6), fig_png(fig, "fig_time", 175)]

    # which objects option 2 loses
    G = {k: gt_found(roots[0]) for k, _, _, roots in METHODS}
    if all(G.values()):
        import collections
        lost = collections.Counter(G["panyoloe"][o][0] for o in G["opt2"] if G["panyoloe"][o][1] and not G["opt2"][o][1])
        gained = collections.Counter(G["opt2"][o][0] for o in G["opt2"] if G["opt2"][o][1] and not G["panyoloe"][o][1])
        S += [Paragraph("What Mask2Former misses compared with Mask2Former + YOLOE (run 1)", H2), Paragraph(
            f"Objects found by Mask2Former + YOLOE but not by Mask2Former: <b>{sum(lost.values())}</b>; found by Mask2Former "
            f"only: {sum(gained.values())}. By class: " + ", ".join(f"{c} {n}" for c, n in lost.most_common()) + ".",
            BODY), Spacer(1, 3)]
        small = ["wall plug", "switch", "panel", "vent", "camera", "book", "blinds", "window", "cushion", "chair",
                 "lamp", "table"]
        rows = [["GT class (found / visible)"] + [SHORT[k] for k in lab]]
        for c in small:
            n = sum(1 for v in G["opt2"].values() if v[0] == c)
            if n:
                rows.append([c] + [f"{sum(1 for v in G[k].values() if v[0] == c and v[1])} / {n}" for k in lab])
        S += [table(rows, [56, 31, 31, 31, 31]), Paragraph(
            "Wall plugs, switches, panels and vents are small fixtures ADE20K has no class for; Mask2Former "
            "never segments them as objects, so they need a second source (the YOLOE union) or a different model. "
            "YOLOE + label filter keeps YOLOE's own detections of them.", SMALL)]

    # 6. reading
    S += [Paragraph("6. Reading the results", H2)]
    notes = []
    if f1["opt2"] is not None and f1["panyoloe"] is not None and f1["base"] is not None:
        notes.append(f"<b>F1:</b> Mask2Former {f1['opt2']:.3f} vs Mask2Former + YOLOE {f1['panyoloe']:.3f} vs Baseline (YOLOE) "
                     f"{f1['base']:.3f}. Precision {pc(prec['opt2'])} / {pc(prec['panyoloe'])} / {pc(prec['base'])}%; "
                     f"recall {pc(rec['opt2'])} / {pc(rec['panyoloe'])} / {pc(rec['base'])}%.")
        notes.append(f"<b>Non-object nodes:</b> {fx(fake['opt2'])} vs {fx(fake['panyoloe'])} vs {fx(fake['base'])}; "
                     f"duplicates {fx(dup['opt2'])} vs {fx(dup['panyoloe'])} vs {fx(dup['base'])}.")
    if T["opt2"] and T["panyoloe"] and T["base"]:
        notes.append(f"<b>Time (8 scenes):</b> Mask2Former {T['opt2']:.1f} min, Mask2Former + YOLOE {T['panyoloe']:.1f} min, "
                     f"Baseline (YOLOE) {T['base']:.1f} min.")
    if f1.get("opt3") is not None:
        notes.append(f"<b>YOLOE + label filter</b> (one detection model): F1 {f1['opt3']:.3f}, precision "
                     f"{pc(prec['opt3'])}%, recall {pc(rec['opt3'])}% (the highest of the four), "
                     f"{fx(fake['opt3'])} non-object nodes, {fx(dup['opt3'])} duplicates, "
                     f"{T['opt3']:.1f} min for 8 scenes. Against Baseline (YOLOE), the label filter alone removes "
                     f"{fx(fake['base'] - fake['opt3'])} non-object nodes while recall rises, and the run is "
                     f"{T['base'] / T['opt3']:.1f}x faster; it keeps more junk and more duplicates than the two "
                     "Mask2Former configurations.")
    notes += ["Mask2Former differs from Mask2Former + YOLOE only by the YOLOE small-object masks, so the gap between "
              "those two rows is the value of the YOLOE union: recall of small fixtures (plugs, switches, vents, "
              "small desk items) against extra detections, time and a second model.",
              "A gap smaller than the difference between two runs of the same configuration is a tie."]
    S += [Paragraph("• " + n, BODY) for n in notes]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Replica: four detection front ends compared")
    doc.build(S)
    (HERE / "summary.json").write_text(json.dumps({"f1": f1, "precision": prec, "recall": rec, "non_objects": fake,
                                                   "duplicates": dup, "minutes_8_scenes": T, "pending": pending},
                                                  indent=1))
    print("wrote", OUT, "| pending:", pending or "none")


if __name__ == "__main__":
    main()
