#!/usr/bin/env python3
"""Scene graph report as a PDF: the graph with every relation, the objects with their names, and the relation list.

Reads a scene graph saved by the pipeline (scene_graph.json, from scene_graph/storage/serializer.py) and, when the
run was evaluated on Replica, its nodes.csv (scripts/eval_replica_scene.py) for the ground-truth class and status
of each node.

    python scripts/render_scene_graph_report.py results/<run>/<scene>
    python scripts/render_scene_graph_report.py path/to/scene_graph.json --nodes path/to/nodes.csv --out report.pdf

Page 1: the graph. Arrows are tree relations (child -> parent: "on", "inside", "part of"); dashed grey lines are
"next to". Nodes show the predicted name and, with nodes.csv, the GT class; colour = status.
Page 2+: object table and relation list.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

STATUS_COLOUR = {          # fill colour per eval status
    "correct": "#7BC67E",
    "wrong_name": "#F5B062",
    "mixed": "#E07A7A",
    "invalid": "#E07A7A",
    "unlabelled": "#C9CDD2",
    None: "#BFD7EA",       # no GT available
}
STATUS_TEXT = {"correct": "correct name", "wrong_name": "real object, wrong name", "mixed": "spans several objects",
               "invalid": "not an object (wall / floor / ...)", "unlabelled": "on unlabelled GT (not scored)"}
REL_TEXT = {"on": "on", "inside": "inside", "belongs": "part of", "belong": "part of", "none": "in room (no support found)"}
REL_STYLE = {"on": 'color="#333333"', "inside": 'color="#1F5FA8" penwidth=1.6',
             "belongs": 'color="#7A4FA0" style=dotted penwidth=1.6',
             "none": 'color="#C0C4C8" style=dotted'}


def flatten(root: dict) -> tuple[list[dict], list[tuple[str, str, str]], list[tuple[str, str, float]]]:
    nodes, tree, nxt, seen = [], [], [], set()

    def walk(n, parent):
        nodes.append(n)
        if parent is not None:
            tree.append((n["node_id"], n.get("relation") or "none", parent["node_id"]))
        for o in n.get("next_to") or []:
            key = tuple(sorted((n["node_id"], o["node_id"])))
            if key not in seen:
                seen.add(key)
                nxt.append((key[0], key[1], float(o.get("gap_m") or 0.0)))
        for c in n.get("children") or []:
            walk(c, n)

    walk(root, None)
    return nodes, tree, nxt


def read_nodes_csv(path: Path | None) -> dict[str, dict]:
    if not path or not path.exists():
        return {}
    with open(path, newline="") as f:
        return {r["node_id"]: r for r in csv.DictReader(f)}


def esc(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_graph(nodes, tree, nxt, ev: dict, out_png: Path, out_pdf: Path) -> None:
    lines = ['graph G {', '  graph [rankdir=BT, nodesep=0.35, ranksep=0.7, fontname="Helvetica", splines=true];',
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, penwidth=0];',
             '  edge [fontname="Helvetica", fontsize=9];']
    for n in nodes:
        nid = n["node_id"]
        e = ev.get(nid)
        name = n.get("node_class") or nid
        if n.get("relation") is None and not n.get("has_embedding"):          # the root (floor)
            label, fill = name, "#E8ECEF"
        else:
            label = f"{nid}"
            if e:
                gt = e.get("gt_class") or "-"
                label += f"\nGT: {gt}"
            fill = STATUS_COLOUR.get(e.get("status") if e else None, STATUS_COLOUR[None])
        lines.append(f'  "{esc(nid)}" [label="{esc(label)}", fillcolor="{fill}"];')
    for child, rel, parent in tree:
        st = REL_STYLE.get(rel, REL_STYLE["on"])
        lines.append(f'  "{esc(child)}" -- "{esc(parent)}" [dir=forward, arrowsize=0.7, label=" {"" if rel == "none" else REL_TEXT.get(rel, rel)}", {st}];')
    for a, b, gap in nxt:
        lines.append(f'  "{esc(a)}" -- "{esc(b)}" [style=dashed, color="#9AA0A6", constraint=false, '
                     f'label=" next to", fontcolor="#80868B"];')
    lines.append("}")
    dot_src = "\n".join(lines)
    with tempfile.NamedTemporaryFile("w", suffix=".dot", delete=False) as f:
        f.write(dot_src)
    # many objects share one parent (the floor): stagger them over several rows so the page stays readable
    src = f.name
    if shutil.which("unflatten"):
        src = f.name + ".u.dot"
        with open(src, "w") as g:
            subprocess.run(["unflatten", "-f", "-l", "7", "-c", "5", f.name], stdout=g, check=True)
    subprocess.run(["dot", "-Tpng", "-Gdpi=200", src, "-o", str(out_png)], check=True)
    subprocess.run(["dot", "-Tpdf", src, "-o", str(out_pdf)], check=True)
    out_pdf.with_suffix(".dot").write_text(dot_src)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="scene_graph.json, or an eval scene folder (results/<run>/<scene>)")
    ap.add_argument("--nodes", type=Path, default=None, help="nodes.csv from eval_replica_scene.py (GT and status)")
    ap.add_argument("--out", type=Path, default=None, help="output PDF (default: next to the input)")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    if args.src.is_dir():
        sg_json = args.src / "scene_graph" / "scene_graph.json"
        nodes_csv = args.nodes or args.src / "nodes.csv"
        out = args.out or args.src / "scene_graph_report.pdf"
        title = args.title or f"Scene graph: {args.src.name}"
    else:
        sg_json, nodes_csv = args.src, args.nodes
        out = args.out or args.src.with_name("scene_graph_report.pdf")
        title = args.title or "Scene graph"

    root = json.loads(sg_json.read_text())
    nodes, tree, nxt = flatten(root)
    ev = read_nodes_csv(nodes_csv)
    objs = [n for n in nodes if not (n.get("relation") is None and not n.get("has_embedding"))]

    tmp = Path(tempfile.mkdtemp())
    png, gpdf = tmp / "graph.png", out.with_name(out.stem + "_graph.pdf")
    render_graph(nodes, tree, nxt, ev, png, gpdf)

    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=4)
    H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=6, spaceAfter=4)
    B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=8.5, leading=11)
    C = ParagraphStyle("c", parent=B, fontSize=7.5, leading=9)
    page = landscape(A4)
    doc = SimpleDocTemplate(str(out), pagesize=page, leftMargin=12 * mm, rightMargin=12 * mm,
                            topMargin=10 * mm, bottomMargin=10 * mm, title=title)
    W, Hh = page[0] - 24 * mm, page[1] - 20 * mm
    story = []

    # summary line
    counts = {}
    for n in objs:
        s = (ev.get(n["node_id"]) or {}).get("status")
        counts[s] = counts.get(s, 0) + 1
    rel_counts = {}
    for _, r, _ in tree:
        rel_counts[REL_TEXT.get(r, r)] = rel_counts.get(REL_TEXT.get(r, r), 0) + 1
    summ = f"{len(objs)} objects; relations: " + ", ".join(f"{v} {k}" for k, v in rel_counts.items())
    summ += f", {len(nxt)} next to."
    if ev:
        summ += " Ground truth: " + ", ".join(f"{counts.get(k, 0)} {STATUS_TEXT[k]}" for k in STATUS_TEXT if counts.get(k))
    story += [Paragraph(title, H1), Paragraph(summ, B)]

    # legend
    if ev:
        leg = [[Paragraph(f'<font color="{STATUS_COLOUR[k]}">&#9632;</font> {v}', C) for k, v in STATUS_TEXT.items()]]
        story.append(Table(leg, hAlign="LEFT"))
    story.append(Paragraph("Arrows: child &#8594; parent (<b>on</b> black, <b>inside</b> blue, <b>part of</b> purple "
                           "dotted). Dashed grey: <b>next to</b>. Node text: node id (predicted name)"
                           + (" and GT class." if ev else "."), C))
    story.append(Spacer(1, 2 * mm))
    from PIL import Image as PILImage
    iw, ih = PILImage.open(png).size
    avail_h = Hh - 32 * mm
    s = min(W / iw, avail_h / ih)
    story += [Image(str(png), width=iw * s, height=ih * s), PageBreak()]

    # objects table
    story.append(Paragraph("Objects", H2))
    parent_of = {c: (r, p) for c, r, p in tree}
    nb = {}
    for a, b, _ in nxt:
        nb.setdefault(a, []).append(b)
        nb.setdefault(b, []).append(a)
    hdr = ["Node", "Predicted name", "Conf.", "Other candidates", "Relations"] + (["GT class", "Status"] if ev else [])
    rows = [hdr]
    styles = []
    for i, n in enumerate(sorted(objs, key=lambda n: n["node_id"]), start=1):
        nid = n["node_id"]
        e = ev.get(nid) or {}
        conf = n.get("name_confidence")
        cands = ", ".join(t["label"] for t in (n.get("top_k") or [])[:4])
        rel = []
        if nid in parent_of:
            r, p = parent_of[nid]
            rel.append(f"{REL_TEXT.get(r, r)} {p}")
        kids = [f"{c} ({REL_TEXT.get(r, r)})" for c, r, p in tree if p == nid]
        if kids:
            rel.append("supports/contains: " + ", ".join(kids))
        if nb.get(nid):
            rel.append("next to " + ", ".join(sorted(nb[nid])))
        row = [Paragraph(nid, C), Paragraph(n.get("node_class") or "", C),
               f"{conf:.2f}" if isinstance(conf, (int, float)) else "-", Paragraph(cands, C),
               Paragraph("; ".join(rel) or "-", C)]
        if ev:
            st = e.get("status")
            row += [Paragraph(e.get("gt_class") or "-", C), Paragraph(STATUS_TEXT.get(st, st or "-"), C)]
            styles.append(("BACKGROUND", (len(row) - 1, i), (len(row) - 1, i), colors.HexColor(STATUS_COLOUR.get(st, "#FFFFFF"))))
        rows.append(row)
    cw = [32, 36, 12, 62, 82] + ([30, 32] if ev else [])
    scale = (W / mm) / sum(cw)
    t = Table(rows, colWidths=[c * scale * mm for c in cw], repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.5), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAED")),
                           ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C0C4C8")), ("VALIGN", (0, 0), (-1, -1), "TOP")]
                          + styles))
    story.append(t)

    # relation list
    story += [Spacer(1, 5 * mm), Paragraph("Inter-object relations", H2)]
    rrows = [["Subject", "Relation", "Object", "Note"]]
    for c, r, p in sorted(tree, key=lambda x: (x[2], x[0])):
        rrows.append([c, REL_TEXT.get(r, r), p, ""])
    for a, b, gap in sorted(nxt):
        rrows.append([a, "next to", b, f"gap {gap:.2f} m"])
    t2 = Table(rrows, colWidths=[W * 0.3, W * 0.12, W * 0.3, W * 0.15], repeatRows=1, hAlign="LEFT")
    t2.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EAED")),
                            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#C0C4C8"))]))
    story.append(t2)
    doc.build(story)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"wrote {out}\nwrote {gpdf} (graph only, vector)")


if __name__ == "__main__":
    main()
