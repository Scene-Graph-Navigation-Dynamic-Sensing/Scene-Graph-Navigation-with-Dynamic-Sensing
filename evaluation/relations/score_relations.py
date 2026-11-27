#!/usr/bin/env python3
"""Stage 3 of the relation comparison: voxel geometry vs RelateAnything vs hybrids, against 3DSSG.

Both methods get the same ground-truth objects (stage 1), so this measures relation prediction only.

(A) Support parent. Subjects: non-structural objects that 3DSSG says stand / lie on (standing on, lying on,
    supported by) or sit in (inside, lying in, standing in) something other than a wall or ceiling.
    parent acc  the method's single parent for the subject is one of its GT parents (any floor segment = floor)
    full acc    ... and the type matches (on-group vs in-group)
    Voxel: the tree parent from scene_graph/builder/dovsg_relations.py ("support unknown" counts as wrong).
    RA: pooled over frames (mean calibrated score per predicate), parent = argmax over objects of the best
    on/in-group score.
(B) Semantic relations 3DSSG annotates that the voxel tree cannot express (hanging on, attached to, leaning
    against, connected to, cover, hanging in, build in) plus part of. RA correct when its best predicate for
    the pair (over the whole vocabulary) is the GT one; top-3 also reported. Voxel scores only "part of" (belong).

    python evaluation/relations/score_relations.py --root ../results/relations_3dssg
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ON = {"standing on", "lying on", "supported by"}
IN = {"inside", "lying in", "standing in"}
RA_ON = ["standing on", "lying on", "supported by", "on"]
RA_IN = ["inside", "lying in", "standing in"]
SEM = {"hanging on": "hanging on", "hanging in": "hanging in", "attached to": "attached to",
       "leaning against": "leaning against", "part of": "part of", "build in": "built into",
       "connected to": "connected to", "cover": "covering"}
STRUCT_NOT_FLOOR = {"wall", "ceiling"}


def pooled(npz):
    """{(sub, obj): (mean scores per predicate, n_frames)} with object ids as str."""
    d = np.load(npz)
    acc, cnt = defaultdict(lambda: 0.0), defaultdict(int)
    for s, o, sc in zip(d["sub"], d["obj"], d["scores"].astype(np.float32)):
        acc[(str(s), str(o))] = acc[(str(s), str(o))] + sc
        cnt[(str(s), str(o))] += 1
    return {k: (acc[k] / cnt[k], cnt[k]) for k in acc}, [str(v) for v in d["vocab"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--model", default="relsgg-vits16plus")
    ap.add_argument("--hybrid_tau", type=float, default=0.5, help="RA overrides geometry above this score")
    args = ap.parse_args()

    A = defaultdict(int)                  # counters for task A
    B = defaultdict(lambda: defaultdict(int))
    examples, scans = [], 0
    for sdir in sorted(p for p in args.root.iterdir() if (p / f"ra_{args.model}.npz").exists()):
        meta = json.loads((sdir / "meta.json").read_text())
        lab = meta["objects"]
        canon = lambda i: "floor" if lab.get(i) == "floor" else i      # noqa: E731
        P, vocab = pooled(sdir / f"ra_{args.model}.npz")
        vi = {p: i for i, p in enumerate(vocab)}
        on_i, in_i = [vi[p] for p in RA_ON], [vi[p] for p in RA_IN]
        scans += 1

        # RA pooled over floor segments: (s, "floor") keeps the best segment
        Pc = {}
        for (s, o), (sc, n) in P.items():
            k = (canon(s), canon(o))
            if k not in Pc or sc.max() > Pc[k][0].max():
                Pc[k] = (sc, n)

        # ── A: support parent ──
        gt_par = defaultdict(dict)
        for s, o, p in meta["gt"]:
            if lab.get(s) in STRUCT_NOT_FLOOR or lab.get(s) == "floor" or lab.get(o) in STRUCT_NOT_FLOOR:
                continue
            if p in ON or p in IN:
                gt_par[s][canon(o)] = "on" if p in ON else "in"
        vox = {s: (p, r) for s, p, r in meta["voxel_tree"]}
        for s, parents in gt_par.items():
            A["subjects"] += 1
            # voxel
            vp, vr = vox.get(s, (None, None))
            vtype = {"on": "on", "inside": "in"}.get(vr)
            v_ok = vtype is not None and vp in parents
            A["vox_parent"] += v_ok
            A["vox_full"] += v_ok and parents[vp] == vtype
            A["vox_unknown"] += vr is None
            # RA
            cands = [(max(sc[on_i].max(), sc[in_i].max()), o, "on" if sc[on_i].max() >= sc[in_i].max() else "in")
                     for (ss, o), (sc, n) in Pc.items() if ss == s and lab.get(o) not in STRUCT_NOT_FLOOR]
            A["ra_covered"] += any(o in parents for _, o, _ in cands)
            if cands:
                best, rp, rtype = max(cands)
            else:
                best, rp, rtype = 0.0, None, None
            r_ok = rp in parents
            A["ra_parent"] += r_ok
            A["ra_full"] += r_ok and parents[rp] == rtype
            # hybrid 1: geometry, but RA where geometry says "support unknown"
            h1p, h1t = (vp, vtype) if vr is not None else (rp, rtype)
            A["h1_parent"] += h1p in parents
            A["h1_full"] += h1p in parents and parents.get(h1p) == h1t
            # hybrid 2: geometry, but RA wherever RA is confident
            h2p, h2t = (rp, rtype) if (best >= args.hybrid_tau or vr is None) and rp else (vp, vtype)
            A["h2_parent"] += h2p in parents
            A["h2_full"] += h2p in parents and parents.get(h2p) == h2t
            if len(examples) < 25 and (v_ok != r_ok):
                examples.append(f"{lab.get(s)}: GT {[lab.get(p, p) + '/' + t for p, t in parents.items()]} | "
                                f"voxel {lab.get(vp, vp)}/{vr} | RA {lab.get(rp, rp)}/{rtype} ({best:.2f})")

        # ── B: semantic relations ──
        for s, o, p in meta["gt"]:
            if p not in SEM:
                continue
            b = B[p]
            b["n"] += 1
            sc_n = Pc.get((canon(s), canon(o)))
            if sc_n is not None:
                sc = sc_n[0]
                b["covered"] += 1
                top = np.argsort(-sc)
                b["ra_top1"] += vocab[top[0]] == SEM[p]
                b["ra_top3"] += SEM[p] in [vocab[i] for i in top[:3]]
            if p == "part of":
                b["vox"] += vox.get(s, (None, None)) == (canon(o), "belong")

    n = max(A["subjects"], 1)
    pct = lambda k: f"{100 * A[k] / n:.1f}%"   # noqa: E731
    lines = [f"# Relations on 3RScan with ground-truth objects: voxel geometry vs RelateAnything ({args.model})", "",
             f"{scans} scans, scored against 3DSSG. Both methods get the same GT objects.", "",
             f"## A. Support parent ({A['subjects']} objects that stand / lie on, or sit in, something)", "",
             "| Method | Right parent | Right parent and type (on / inside) |", "|---|---:|---:|",
             f"| Voxel geometry (current) | {pct('vox_parent')} | {pct('vox_full')} |",
             f"| RelateAnything (pooled over frames) | {pct('ra_parent')} | {pct('ra_full')} |",
             f"| Hybrid 1: geometry, RA where geometry says 'support unknown' | {pct('h1_parent')} | {pct('h1_full')} |",
             f"| Hybrid 2: geometry, RA where RA score >= {args.hybrid_tau} | {pct('h2_parent')} | {pct('h2_full')} |",
             "", f"Geometry left {A['vox_unknown']} objects as 'support unknown'. RA saw the subject with its GT "
             f"parent in at least one frame for {pct('ra_covered')} of subjects.", "",
             "## B. Relations only RelateAnything can express", "",
             "| GT relation | Pairs | Co-visible | RA top-1 | RA top-3 | Voxel |", "|---|---:|---:|---:|---:|---:|"]
    tot = defaultdict(int)
    for p, b in sorted(B.items(), key=lambda kv: -kv[1]["n"]):
        for k, v in b.items():
            tot[k] += v
        cv = max(b["covered"], 1)
        lines.append(f"| {p} | {b['n']} | {100 * b['covered'] / b['n']:.0f}% | {100 * b['ra_top1'] / cv:.0f}% | "
                     f"{100 * b['ra_top3'] / cv:.0f}% | " + (f"{100 * b['vox'] / b['n']:.0f}%" if p == "part of" else "n/a") + " |")
    cv = max(tot["covered"], 1)
    lines.append(f"| **all** | {tot['n']} | {100 * tot['covered'] / max(tot['n'], 1):.0f}% | "
                 f"{100 * tot['ra_top1'] / cv:.0f}% | {100 * tot['ra_top3'] / cv:.0f}% | |")
    lines += ["", "RA top-1 / top-3 are over co-visible pairs: its best (or top three) of the 20 predicates is the GT one.",
              "", "## Where the two disagree (first 25)", ""] + [f"- {e}" for e in examples]
    out = args.root / f"report_{args.model}.md"
    out.write_text("\n".join(lines) + "\n")
    (args.root / f"summary_{args.model}.json").write_text(json.dumps({"A": A, "B": B}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
