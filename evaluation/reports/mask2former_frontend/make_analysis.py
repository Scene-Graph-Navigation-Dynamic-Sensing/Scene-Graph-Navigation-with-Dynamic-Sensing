"""Deeper analysis of the Mask2Former Replica runs against Baseline (YOLOE) and Mask2Former + YOLOE:
what the nodes are, what each front end finds and misses (by size and class), the junk that remains, duplicates,
naming errors, per-scene numbers and side-by-side detection frames.

    python evaluation/reports/mask2former_frontend/make_analysis.py   (scene_graph env)
writes parts/option2_mask2former_analysis.pdf next to this script
"""
from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

import make_report as R  # noqa: E402  (fonts, styles, table helper, METHODS, timing)
from make_report import BODY, CELL, H1, H2, MAIN, SCENES, SHORT, SMALL, fx, mm, pc, plt, table  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "parts" / "option2_mask2former_analysis.pdf"
OUT.parent.mkdir(exist_ok=True)
IMG = HERE / "analysis_img"
IMG.mkdir(exist_ok=True)
KEYS = [k for k, *_ in R.METHODS]
ROOTS = {k: roots for k, _, _, roots in R.METHODS}
COL = {k: c for k, _, c, _ in R.METHODS}
LAB = {k: l for k, l, _, _ in R.METHODS}
STATUS = [("correct", "real object, named right", "#1F7A5C"), ("wrong_name", "real object, wrong name", "#7FB8A4"),
          ("structural", "wall / floor / ceiling", "#B85A12"), ("mixed", "spans several objects", "#E0A458"),
          ("unlabelled", "on unlabelled GT (not scored)", "#C9CCD1")]


# ─── loading ─────────────────────────────────────────────────────────────

def nodes(root):
    out = []
    for s in SCENES:
        for r in csv.DictReader(open(MAIN / root / s / "nodes.csv")):
            r["scene"] = s
            out.append(r)
    return out


def gt(root):
    out = {}
    for s in SCENES:
        for r in csv.DictReader(open(MAIN / root / s / "gt_objects.csv")):
            if r["kind"] == "countable" and r["visible"] == "True":
                out[(s, r["gt_object_id"])] = {"cls": r["class"], "found": r["found"] == "True",
                                               "size": int(r["visible_samples"]),
                                               "n": len([x for x in r["matched_nodes"].split("|") if x.strip()])
                                               if r["matched_nodes"] else 0}
    return out


def naming(root):
    out = []
    for s in SCENES:
        for r in csv.DictReader(open(MAIN / root / s / "naming.csv")):
            r["scene"] = s
            out.append(r)
    return out


def metrics(root):
    return {s: json.loads((MAIN / root / s / "metrics.json").read_text()) for s in SCENES}


def save(fig, name):
    p = IMG / f"{name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return p


def img(path, w_mm, max_h_mm=None):
    w, h = PILImage.open(path).size
    hh = w_mm * h / w
    if max_h_mm and hh > max_h_mm:
        w_mm, hh = max_h_mm * w / h, max_h_mm
    return Image(str(path), width=w_mm * mm, height=hh * mm)


def crop_path(root, scene, node_id):
    return MAIN / root / scene / "node_crops" / f"{node_id.replace(' ', '_')}.jpg"


def square(p):
    """Pad a crop to a square on a light background (cached in IMG) so thumbnail grids line up."""
    q = IMG / ("sq_" + "_".join(Path(p).parts[-4:]).replace(" ", "_"))
    if not q.exists():
        im = PILImage.open(p).convert("RGB")
        n = max(im.size)
        bg = PILImage.new("RGB", (n, n), (243, 243, 241))
        bg.paste(im, ((n - im.size[0]) // 2, (n - im.size[1]) // 2))
        bg.save(q, quality=90)
    return q


def thumb_grid(items, cols, cell_mm, caption_style):
    """items: [(image path or None, caption)] -> Table of square thumbnails with captions."""
    cells, row = [], []
    for p, cap in items:
        parts = []
        if p and Path(p).exists():
            parts.append(img(square(p), cell_mm - 3))
        else:
            parts.append(Paragraph("(no crop saved)", caption_style))
        parts.append(Paragraph(cap, caption_style))
        row.append(parts)
        if len(row) == cols:
            cells.append(row)
            row = []
    if row:
        cells.append(row + [""] * (cols - len(row)))
    t = Table(cells, colWidths=[cell_mm * mm] * cols)
    t.setStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4)])
    return t


# ─── analysis ────────────────────────────────────────────────────────────

def main():
    N = {k: [nodes(r) for r in ROOTS[k]] for k in KEYS}
    G = {k: gt(ROOTS[k][0]) for k in KEYS}
    NM = {k: naming(ROOTS[k][0]) for k in KEYS}
    M = {k: metrics(ROOTS[k][0]) for k in KEYS}
    CAP = R.ParagraphStyle("cap", parent=CELL, fontSize=6.6, leading=8)
    S = [Paragraph("Analysis of the results", H1),
         Paragraph("What the maps contain, what each front end finds and misses, and why. Four configurations, "
                   "8 Replica scenes, 2 runs each; per-node analyses use run 1 (the two runs differ by ≤ 0.005 F1). "
                   "Headline tables, published baselines and timing are in the first part of this document.", SMALL), Spacer(1, 4)]

    # 0. configurations
    DESC = {"base": "YOLOE-26x prompt-free (1024 px, conf 0.10) as the detector",
            "panyoloe": "Mask2Former Swin-L ADE20K panoptic + YOLOE-26x masks <= 2% of the frame (current default)",
            "opt2": "Mask2Former Swin-L ADE20K panoptic as the only detector",
            "opt3": "YOLOE-26x prompt-free; detections whose own YOLOE tag is a place, room, building or surface "
                    "are dropped before tracking (configs/vocab/yoloe_pf_scene_tags.txt)"}
    S += [table([["Configuration", "Detector"]] + [[SHORT[k], DESC[k]] for k in KEYS], [40, 140])]

    # ── 1. what the nodes are ──
    cnt = {k: {s: np.mean([sum(1 for n in run if n["status"] == s) for run in N[k]]) for s, *_ in STATUS} for k in KEYS}
    tot = {k: sum(cnt[k].values()) for k in KEYS}
    fig, ax = plt.subplots(figsize=(8.5, 2.7))
    for i, k in enumerate(KEYS):
        left = 0
        for s, lab, c in STATUS:
            v = cnt[k][s]
            ax.barh(i, v, left=left, color=c, label=lab if i == 0 else None, height=0.6)
            if v >= 12:
                ax.text(left + v / 2, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white")
            left += v
    ax.set_yticks(range(len(KEYS)), [SHORT[k] for k in KEYS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("nodes in the 8 maps (mean of 2 runs)", fontsize=8)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.3))
    ax.spines[["top", "right"]].set_visible(False)
    rows = [["Nodes (mean of 2 runs)"] + [SHORT[k] for k in KEYS]]
    for s, lab, _ in STATUS:
        rows.append([lab] + [f"{cnt[k][s]:.1f} ({100 * cnt[k][s] / tot[k]:.0f}%)" for k in KEYS])
    rows.append(["Total"] + [f"{tot[k]:.1f}" for k in KEYS])
    real = {k: cnt[k]["correct"] + cnt[k]["wrong_name"] for k in KEYS}
    S += [Paragraph("1. What the map nodes are", H2), Paragraph(
        f"Every node is graded against Replica's ground truth. Mask2Former's map is {100 * real['opt2'] / tot['opt2']:.0f}% "
        f"real objects, against {100 * real['panyoloe'] / tot['panyoloe']:.0f}% for Mask2Former + YOLOE and "
        f"{100 * real['base'] / tot['base']:.0f}% for Baseline (YOLOE). Baseline (YOLOE)'s {cnt['base']['structural']:.0f} wall / "
        f"floor / ceiling nodes fall to {cnt['opt2']['structural']:.0f}: Mask2Former labels those surfaces as stuff "
        f"and they never enter the pipeline. The remaining non-objects in Mask2Former are mostly nodes spanning several "
        f"objects (section 4). YOLOE + label filter's map is {100 * real['opt3'] / tot['opt3']:.0f}% real objects: "
        f"dropping YOLOE's own place / surface tags cuts the wall / floor / ceiling nodes to "
        f"{cnt['opt3']['structural']:.0f}.", BODY), Spacer(1, 3), img(save(fig, "status"), 170), Spacer(1, 2),
        table(rows, [48, 33, 33, 33, 33], bold_last=True)]

    # ── 2. recall by size ──
    sizes = np.array([v["size"] for v in G["opt2"].values()])
    edges = np.quantile(sizes, [0, 0.25, 0.5, 0.75, 1.0])
    bins = [(edges[i], edges[i + 1]) for i in range(4)]
    names = ["smallest 25%", "25–50%", "50–75%", "largest 25%"]
    rec = {k: [] for k in KEYS}
    for lo, hi in bins:
        for k in KEYS:
            sel = [v for v in G[k].values() if lo <= v["size"] <= hi]
            rec[k].append(100 * np.mean([v["found"] for v in sel]))
    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    x = np.arange(4)
    for j, k in enumerate(KEYS):
        off, wb = (j - (len(KEYS) - 1) / 2) * 0.8 / len(KEYS), 0.8 / len(KEYS)
        ax.bar(x + off, rec[k], wb, color=COL[k], label=SHORT[k])
        for i, v in enumerate(rec[k]):
            ax.text(x[i] + off, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [f"{n}\n({int(lo):,}–{int(hi):,} samples)" for n, (lo, hi) in zip(names, bins)], fontsize=7)
    ax.set_ylabel("GT objects found (%)", fontsize=8)
    ax.set_ylim(0, 105)
    ax.tick_params(labelsize=7.5)
    ax.legend(fontsize=7.5, frameon=False, ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    S += [Paragraph("2. Which objects are found: by size", H2), Paragraph(
        "The 349 visible countable GT objects are split into quarters by how much of each is visible (mesh samples "
        "seen in the 200 frames; a size proxy). Recall = share found by at least one node.", BODY),
        img(save(fig, "recall_size"), 170), Paragraph(
        f"Mask2Former finds {rec['opt2'][0]:.0f}% of the smallest quarter against {rec['panyoloe'][0]:.0f}% for "
        f"Mask2Former + YOLOE, {rec['base'][0]:.0f}% for Baseline (YOLOE) and {rec['opt3'][0]:.0f}% for YOLOE + label "
        f"filter; for the largest quarter the four are within "
        f"{max(r[3] for r in rec.values()) - min(r[3] for r in rec.values()):.0f} points. The recall Mask2Former loses is "
        "almost entirely small objects.", BODY)]

    # ── 3. by class ──
    cls_n = collections.Counter(v["cls"] for v in G["opt2"].values())
    top = [c for c, _ in cls_n.most_common(30)]
    rows = [["GT class", "Visible"] + [SHORT[k] for k in KEYS] + ["Mask2Former − Mask2Former + YOLOE"]]
    hl = []
    for c in top:
        f = {k: sum(1 for v in G[k].values() if v["cls"] == c and v["found"]) for k in KEYS}
        d = f["opt2"] - f["panyoloe"]
        rows.append([c, cls_n[c]] + [f"{f[k]} ({100 * f[k] / cls_n[c]:.0f}%)" for k in KEYS] + [f"{d:+d}" if d else "0"])
        if f["opt2"] == 0:
            hl.append(len(rows) - 1)
    never = sorted({v["cls"] for o, v in G["opt2"].items() if not any(G[k][o]["found"] for k in KEYS)})
    never_n = sum(1 for o in G["opt2"] if not any(G[k][o]["found"] for k in KEYS))
    t3 = table(rows, [26, 14, 26, 27, 26, 27, 34])
    t3.setStyle([("BACKGROUND", (0, r), (-1, r), colors.HexColor("#F7E3D3")) for r in hl])
    S += [PageBreak(), Paragraph("3. Which objects are found: by class (30 most common classes)", H2),
          t3, Spacer(1, 3), Paragraph(
              "Highlighted rows: classes Mask2Former never finds. "
              f"<b>Found by none of the four:</b> {never_n} of 349 GT objects, from these classes: " + ", ".join(never) + ".", SMALL)]

    # ── 4. remaining non-objects in option 2 ──
    from composition import composition
    bad = [n for n in N["opt2"][0] if n["status"] in ("structural", "mixed")]
    tag = "panonly_r1"                                   # scene-graph name suffix of results/option2_panoptic/run1
    comp = {}
    for s in sorted({n["scene"] for n in bad}):
        for nid, c in composition(tag, ROOTS["opt2"][0], s, {n["node_id"] for n in bad if n["scene"] == s}).items():
            comp[(s, nid)] = c

    def on(n):
        c = comp.get((n["scene"], n["node_id"]), [])
        return ", ".join(f"{lab} {100 * f:.0f}%" for lab, _, f in c[:3] if f >= 0.03) or "–"
    items = []
    for n in bad:
        p = crop_path(ROOTS["opt2"][0], n["scene"], n["node_id"])
        items.append((p, f"<b>{n['scene']}</b> · {n['node_id']} · NOVIC: {n['novic_name']}<br/>"
                         f"<b>{n['status']}</b> · points on: {on(n)}"))
    mix_cls = collections.Counter(n["gt_class"] for n in bad if n["status"] == "mixed")
    # sort the mixed nodes by what their second-largest share is
    kinds = collections.defaultdict(list)
    for n in bad:
        c = comp.get((n["scene"], n["node_id"]), [])
        if n["status"] != "mixed" or len(c) < 2:
            continue
        if c[1][1] == "structural":
            kinds["behind"].append(n)
        elif c[1][0] == c[0][0]:
            kinds["same"].append(n)
        else:
            kinds["different"].append(n)

    def names(ns):
        return ", ".join(f"{n['scene']} · {n['node_id']}" for n in ns) or "none"
    shelf = comp.get(("room_2", "shelf_0"))
    S += [PageBreak(), Paragraph(f"4. The {len(bad)} non-object nodes left in Mask2Former (run 1)", H2), Paragraph(
        "How the evaluation grades a node: every 3D point (1 cm voxel) of the node takes the ground-truth object of "
        "the nearest GT mesh point within 5 cm. The object with the most points is the node's GT object, and its "
        "share of the points is the node's <b>purity</b>. <i>structural</i> = that object is a wall, floor or "
        "ceiling; <i>mixed</i> = purity below 50%, i.e. no single object holds most of the node's points. The "
        "grade is about the node's 3D footprint, not its name or its crop: a node can look like one object, carry "
        "the right name, and still be graded mixed.", BODY), Spacer(1, 3)]
    if shelf:
        rows = [["What room_2 · shelf_0's points lie on", "GT kind", "Share of points"]] + \
               [[lab, kind, f"{100 * f:.1f}%"] for lab, kind, f in shelf[:6]]
        S += [Paragraph(
            "<b>Example: room_2 · shelf_0.</b> The crop is clearly a shelf and NOVIC names it 'shelf', yet it is graded "
            f"mixed. Only {100 * shelf[0][2]:.0f}% of its points lie on the GT shelf; {100 * shelf[1][2]:.0f}% lie on "
            "the <b>wall behind it</b>, and the rest on the plates, sculptures, boxes and vases standing on it. It is "
            "an open wire shelf: Mask2Former's 'shelf' segment covers the gaps between the shelves, so the wall seen "
            "through those gaps is lifted into 3D as part of the shelf. The same node appears in both Mask2Former "
            "runs and in Mask2Former + YOLOE (purity 0.46 each time); Baseline (YOLOE) has no node whose majority is "
            "the shelf.", BODY), Spacer(1, 2), table(rows, [70, 30, 30]), Spacer(1, 3)]
    nm = sum(mix_cls.values())
    S += [Paragraph(f"The {nm} mixed nodes fall into three groups, by what holds their second-largest share of "
                    "points:", BODY), Spacer(1, 2),
          table([["Group", "#", "What happened", "Which nodes"],
                 ["Object + surface behind it", len(kinds["behind"]),
                  "One real object; the wall / floor seen through or around it was lifted with it (the shelf above)",
                  names(kinds["behind"])],
                 ["Several objects of the same class", len(kinds["same"]),
                  "Neighbouring GT instances of one class (adjacent blind panels, two cushions, a stack of boxes, "
                  "two stools) fused into one node. Replica annotates each panel / item separately; Mask2Former "
                  "and the fusion treat the touching group as one object",
                  names(kinds["same"])],
                 ["Two different touching objects", len(kinds["different"]),
                  "Different objects in contact (blinds and a door, a comforter on a bed) fused into one node",
                  names(kinds["different"])]],
                [36, 12, 76, 56]), Spacer(1, 3),
          Paragraph(
              "So 'non-object' in this section means 'not one clean GT object in 3D', not 'not an object': every "
              "mixed node lies mostly on real objects, and most carry a sensible name. The 'same class' group is "
              "partly a question of granularity (is a row of blind panels one object or several?), which the "
              "automatic grading cannot settle. Each crop below lists the node's three largest shares.", BODY),
          Spacer(1, 3), thumb_grid(items, 5, 35, CAP)]

    # ── 5. duplicates ──
    dup = {}
    for k in KEYS:
        c = collections.Counter()
        for s in SCENES:
            for lab, n in M[k][s]["nodes"].get("duplicated_objects", {}).items():
                c[lab.split("#")[0]] += n - 1
        dup[k] = c
    allc = sorted(set().union(*[set(c) for c in dup.values()]), key=lambda c: -sum(dup[k][c] for k in KEYS))[:15]
    rows = [["GT class (extra nodes)"] + [SHORT[k] for k in KEYS]] + \
           [[c] + [dup[k][c] or "" for k in KEYS] for c in allc] + \
           [["Total"] + [sum(dup[k].values()) for k in KEYS]]
    S += [Paragraph("5. Duplicates: one GT object split into several nodes (run 1)", H2),
          table(rows, [52, 32, 32, 32, 32], bold_last=True), Paragraph(
              "Extra nodes on an already-found GT object. Most come from " +
              ", ".join(f"{c} ({dup['base'][c]} / {dup['panyoloe'][c]} / {dup['opt2'][c]} / {dup['opt3'][c]})"
                        for c in allc[:4]) +
              " (Baseline (YOLOE) / Mask2Former + YOLOE / Mask2Former / YOLOE + label filter). The label filter does "
              "not reduce duplicates: YOLOE's overlapping masks of a whole and its parts all carry object tags. Panoptic masks never overlap, so one object is one "
              "segment per frame rather than a whole plus its parts; Mask2Former also has no YOLOE masks that can "
              "start a second track on part of an object.", SMALL)]

    # ── 6. naming ──
    grades = ["exact", "specific", "generic", "wrong", "no_name"]
    gl = {"exact": "exact / synonym", "specific": "more specific", "generic": "too generic", "wrong": "wrong",
          "no_name": "no name"}
    gcnt = {k: collections.Counter(r["grade_novic"] for r in NM[k]) for k in KEYS}
    rows = [["NOVIC grade (nodes on one real object)"] + [SHORT[k] for k in KEYS]]
    for g in grades:
        rows.append([gl[g]] + [f"{gcnt[k][g]} ({100 * gcnt[k][g] / max(sum(gcnt[k].values()), 1):.0f}%)" for k in KEYS])
    rows.append(["acceptable (exact + more specific)"] +
                [f"{100 * (gcnt[k]['exact'] + gcnt[k]['specific']) / max(sum(gcnt[k].values()), 1):.1f}%" for k in KEYS])
    wrong = collections.Counter((r["gt_class"], r["novic"]) for r in NM["opt2"] if r["grade_novic"] == "wrong")
    conf = collections.Counter(r["gt_class"] for r in NM["opt2"] if r["grade_novic"] == "wrong")
    S += [PageBreak(), Paragraph("6. Naming", H2), Paragraph(
        "NOVIC names each node from its 42,919 nouns with no class list; a WordNet judge grades the name against "
        "the GT class. Only nodes matched cleanly to one countable object are graded, so the share depends on "
        "which objects each front end found.", BODY), Spacer(1, 3),
        table(rows, [58, 30, 30, 30, 30], bold_last=True), Spacer(1, 4), Paragraph(
        "Mask2Former's acceptable share is the highest. A likely reason is selection: it no longer finds the small "
        "fixtures (wall plugs, switches, vents) that NOVIC rarely names right, so the graded set is easier. Many "
        "'wrong' names are strict-judge cases rather than errors: Replica's ceiling lamps are named 'panel light', "
        "'fluorescent fixture' or 'room light', which the WordNet judge does not accept for 'lamp'. "
        "GT classes Mask2Former's nodes are most often misnamed: " +
        ", ".join(f"{c} {n}" for c, n in conf.most_common(10)) + ".", BODY), Spacer(1, 3)]
    rows = [["GT class", "NOVIC said", "Nodes"]] + [[g, n, k] for (g, n), k in wrong.most_common(8)]
    S += [Paragraph("Most frequent wrong names, Mask2Former run 1", H2), table(rows, [55, 80, 20])]
    ex, seen = [], set()
    for (g, n_), _ in wrong.most_common():
        r = next(x for x in NM["opt2"] if x["grade_novic"] == "wrong" and x["gt_class"] == g and x["novic"] == n_)
        if g not in seen or len(ex) >= 5:
            ex.append(r)
            seen.add(g)
        if len(ex) == 10:
            break
    items = [(crop_path(ROOTS["opt2"][0], r["scene"], r["node_id"]),
              f"GT <b>{r['gt_class']}</b><br/>NOVIC: {r['novic']}") for r in ex]
    S += [Spacer(1, 4), Paragraph("Examples of wrong names (crops)", H2), thumb_grid(items, 5, 33, CAP)]

    # ── 7. per scene ──
    rows = [["Scene"] + [f"{SHORT[k]} P / R / F1" for k in KEYS] + ["Mask2Former non-obj / dup"]]
    for s in SCENES:
        cells = []
        for k in KEYS:
            n = M[k][s]["nodes"]
            sc = n["status_counts"]
            realn = sc.get("correct", 0) + sc.get("wrong_name", 0)
            fake = sc.get("structural", 0) + sc.get("mixed", 0) + sc.get("phantom", 0)
            p = n["valid_objects"] / max(realn + fake, 1)
            r = n["valid_objects"] / max(n["gt_objects_visible"], 1)
            f1 = 2 * p * r / (p + r) if p + r else 0
            cells.append(f"{100 * p:.0f} / {100 * r:.0f} / {f1:.2f}")
        n = M["opt2"][s]["nodes"]
        sc = n["status_counts"]
        rows.append([s] + cells + [f"{sc.get('structural', 0) + sc.get('mixed', 0)} / {n['n_dup']}"])
    S += [PageBreak(), Paragraph("7. Per scene (run 1)", H2), table(rows, [18, 34, 34, 34, 34, 26]), Paragraph(
        "P = object precision, R = recall of visible countable GT objects, F1 from both (%).", SMALL)]

    # ── 8. side-by-side frames ──
    S += [Paragraph("8. What each front end segments: same frames, four front ends", H2), Paragraph(
        "All masks the detector passed to the tracker in that frame (pipeline_visualizations/detection/"
        "*/02_masks_combined.jpg). Baseline (YOLOE) masks overlap and include walls and groups; panoptic masks tile the "
        "objects without overlap; the YOLOE union adds small masks (fixtures, small items) on top; YOLOE + label "
        "filter shows YOLOE's masks after wall / floor / room-tagged ones are removed.", BODY)]
    for scene, frame in [("office_0", "000080"), ("room_0", "000100"), ("office_3", "000060")]:
        cells = []
        for k in KEYS:
            p = MAIN / ROOTS[k][0] / scene / "pipeline_visualizations" / "detection" / frame / "02_masks_combined.jpg"
            cells.append([img(p, 43) if p.exists() else Paragraph("(missing)", CELL),
                          Paragraph(f"{SHORT[k]}", CAP)])
        t = Table([cells], colWidths=[45 * mm] * len(KEYS))
        t.setStyle([("VALIGN", (0, 0), (-1, -1), "TOP")])
        S += [Spacer(1, 3), Paragraph(f"<b>{scene}</b>, frame {int(frame)}", CELL), t]

    # ── 9. top-down maps ──
    S += [PageBreak(), Paragraph("9. Top-down map vs ground truth (office_0, run 1)", H2)]
    for k in ("panyoloe", "opt2", "opt3"):
        p = MAIN / ROOTS[k][0] / "office_0" / "topdown_prediction_vs_gt.png"
        if p.exists():
            S += [Paragraph(LAB[k], CELL), img(p, 175, 70), Spacer(1, 2)]

    # ── 10. Mask2Former vs YOLOE: what is different ──
    S += [PageBreak(), Paragraph("10. YOLOE prompt-free vs Mask2Former panoptic: what is different", H2),
          table([["", "YOLOE prompt-free", "Mask2Former panoptic"],
                 ["1. Vocabulary",
                  "4,585 tags, including plug, light switch, vent, camera, but also room, office, wall, academy",
                  "150 classes, each marked thing or stuff. No plug, switch, vent or camera."],
                 ["2. What our pipeline does with the label",
                  "Nothing. Every detection goes on, even one labelled wall or room.",
                  "Filters on it: segments labelled stuff (wall, floor, ceiling) are dropped before tracking."],
                 ["3. How the image is divided",
                  "Separate, overlapping boxes and masks. A wall, a desk, the desk plus its clutter, and a laptop "
                  "can all be detected at once.",
                  "Every pixel belongs to exactly one segment. Walls are taken up by stuff segments, and objects "
                  "can't overlap or nest."]],
                [40, 70, 70]), Spacer(1, 4),
          Paragraph("<b>Difference 2</b> is the main reason the Baseline (YOLOE) map has so much junk. YOLOE often "
                    "does label walls <i>wall</i> and rooms <i>room</i>, but the pipeline ignored those labels, so "
                    "they went through. Mask2Former's walls are dropped because the pipeline acts on the label.",
                    BODY), Spacer(1, 3),
          Paragraph("<b>Difference 3</b> is something YOLOE can't copy by filtering labels. Mask2Former was trained "
                    "on images where every pixel is annotated as either a thing or stuff, so it learned where "
                    "objects end and surfaces begin. Its masks can't overlap. YOLOE was trained only to detect "
                    "nameable regions, so it also produces group boxes (a desk with everything on it) and nested "
                    "masks. Those have valid object names (<i>desk</i>), so a label filter keeps them.", BODY),
          Spacer(1, 3),
          Paragraph("<b>Tested:</b> YOLOE + label filter applies difference 2 to YOLOE: its own place, room and "
                    "surface tags now decide what is dropped. Section 1 shows the effect on wall / floor nodes; "
                    "sections 1 and 5 show what difference 3 leaves behind (nodes spanning several objects, "
                    "duplicates).", BODY)]

    # ── 11. findings ──
    f = json.loads((HERE / "summary.json").read_text())
    S += [Paragraph("11. Findings", H2)]
    for t in [
        f"<b>Mask2Former removes the junk at the source.</b> Wall / floor / ceiling nodes fall from "
        f"{cnt['base']['structural']:.0f} (Baseline (YOLOE)) and {cnt['panyoloe']['structural']:.0f} (Mask2Former + YOLOE) to "
        f"{cnt['opt2']['structural']:.0f}; object precision is {pc(f['precision']['opt2'])}%.",
        f"<b>Its recall loss is the small objects.</b> Smallest quarter: {rec['opt2'][0]:.0f}% found vs "
        f"{rec['panyoloe'][0]:.0f}% (Mask2Former + YOLOE); classes with no ADE20K counterpart (wall plug, switch, "
        "panel, vent) are never found. Large objects are found as often as by the other front ends.",
        f"<b>Duplicates drop to {sum(dup['opt2'].values())}</b> (run 1) because panoptic masks never overlap: one "
        "object is one segment per frame.",
        "<b>Naming scores higher</b> on the objects it does find, likely because the hard small fixtures are no "
        "longer in the graded set.",
        f"<b>It is the fastest</b> ({fx(f['minutes_8_scenes']['opt2'])} min for 8 scenes vs "
        f"{fx(f['minutes_8_scenes']['panyoloe'])} and {fx(f['minutes_8_scenes']['base'])}): fewer detections, "
        "fewer tracks, much cheaper fusion.",
        f"<b>YOLOE + label filter matches Mask2Former + YOLOE with one model.</b> F1 "
        f"{fx(f['f1']['opt3'], 3)} vs {fx(f['f1']['panyoloe'], 3)}, with the highest recall of the four "
        f"({pc(f['recall']['opt3'])}%) and {fx(f['minutes_8_scenes']['opt3'])} min for 8 scenes. Using YOLOE's own "
        "tags to drop places and surfaces (difference 2 in section 10) removes most wall / floor nodes; what it "
        "cannot remove are group boxes and overlapping part masks (difference 3), so it keeps more non-objects "
        f"({fx(f['non_objects']['opt3'])}) and duplicates ({fx(f['duplicates']['opt3'])}) than the Mask2Former "
        "configurations.",
        "<b>For the paper:</b> Mask2Former alone is a strong high-precision, fast baseline whose failure mode is clear and "
        "measurable: object types outside ADE20K's vocabulary. A front end that keeps its precision but recovers "
        "those (the YOLOE-embedding plan in yoloe_unified_frontend/plan.md) is the natural next contribution."]:
        S.append(Paragraph("• " + t, BODY))

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title="Replica front ends: analysis")
    doc.build(S)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
