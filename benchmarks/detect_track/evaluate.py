#!/usr/bin/env python3
"""
Score every outputs/<pipeline>/predictions.jsonl against gt/gt.json and summaries.

Accuracy (on the GT keyframes, IoU >= 0.5, one-to-one Hungarian matching on IoU):
  det_recall      matched GT objects / GT objects                     (did it find a box?)
  label_acc       matched objects whose name is acceptable / matched  (is the open-vocab name right?)
  correct_recall  found AND correctly named / GT objects
  precision       matched / (matched + false boxes); a prediction is not a false box if it has
                  IoU >= 0.5 with an 'ignore' region (desks, truncated / tiny objects)
  correct_prec    found AND correctly named / (matched + false boxes)
  f1_correct      harmonic mean of correct_recall and correct_prec
Tracking:
  id_switches     over consecutive keyframes (1 s apart) where a GT object is matched both times,
                  how often the predicted track id changed
  id_consistency  1 - id_switches / consecutive matched pairs
  label_flicker   over all frames, fraction of (track, frame) steps where a track's displayed name
                  differs from its name on the previous frame it was seen
Name matching: lower-cased; a name is acceptable if it equals an accepted term for the object's
class or ends with " <term>" (so "black office chair" counts as chair).

  'ignore_zones' (optional per frame): predictions whose box centre falls inside are ignored.

Usage: python evaluate.py [--iou 0.5] [--gt gt/gt.json] [--out-root outputs]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
ORDER = ["p1_dtd_every_frame", "p4_dtd_confirm_then_track", "p4b_dtd_confirm_strict", "p2_ram_yoloe_botsort",
         "p3a_ram_gdino_sam2_botsort", "p3b_ram_gdino_sam2_videotrack"]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def name_ok(label: str, accepted: list[str]) -> bool:
    lab = label.lower().replace("_", " ").strip()
    return any(lab == t or lab.endswith(" " + t) for t in accepted)


def _in_zone(box, zones) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return any(z[0] <= cx <= z[2] and z[1] <= cy <= z[3] for z in zones)


def evaluate(name: str, gt: dict, iou_thr: float) -> dict | None:
    pred_path = os.path.join(OUT, name, "predictions.jsonl")
    if not os.path.exists(pred_path):
        return None
    rows = {}
    with open(pred_path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["frame"]] = r
    scale = gt["scale_to_source"]

    n_gt = n_match = n_correct = n_fp = 0
    per_class: dict[str, list[int]] = {}
    matches_by_frame: dict[int, dict[str, int]] = {}
    wrong_names: dict[str, dict[str, int]] = {}
    missed: dict[str, int] = {}

    for fk in sorted(gt["frames"], key=int):
        fi = int(fk)
        fr = gt["frames"][fk]
        gids = list(fr["gt"])
        gboxes = np.array([fr["gt"][g] for g in gids], float) * scale
        ign = np.array(fr["ignore"], float).reshape(-1, 4) * scale
        zones = [[v * scale for v in z] for z in fr.get("ignore_zones", [])]
        preds = [p for p in rows.get(fi, {"objects": []})["objects"] if not _in_zone(p["box"], zones)]
        pboxes = np.array([p["box"] for p in preds], float).reshape(-1, 4)

        m = iou_matrix(gboxes, pboxes)
        gi, pj = linear_sum_assignment(-m) if m.size else (np.array([], int), np.array([], int))
        pairs = [(g, p) for g, p in zip(gi, pj) if m[g, p] >= iou_thr]
        matched_p = {p for _, p in pairs}
        matched_g = {g for g, _ in pairs}

        n_gt += len(gids)
        n_match += len(pairs)
        frame_ids = {}
        for g, p in pairs:
            cls = gt["objects"][gids[g]]
            ok = name_ok(preds[p]["label"], gt["classes"][cls])
            n_correct += ok
            per_class.setdefault(cls, [0, 0, 0])
            per_class[cls][1] += 1
            per_class[cls][2] += ok
            frame_ids[gids[g]] = preds[p]["id"]
            if not ok:
                wrong_names.setdefault(cls, {})
                lab = preds[p]["label"] or "<no name yet>"
                wrong_names[cls][lab] = wrong_names[cls].get(lab, 0) + 1
        for g, gid in enumerate(gids):
            cls = gt["objects"][gid]
            per_class.setdefault(cls, [0, 0, 0])
            per_class[cls][0] += 1
            if g not in matched_g:
                missed[gid] = missed.get(gid, 0) + 1
        matches_by_frame[fi] = frame_ids

        unmatched = [p for p in range(len(preds)) if p not in matched_p]
        if unmatched:
            ig = iou_matrix(pboxes[unmatched], ign)
            n_fp += int(sum(1 for k in range(len(unmatched)) if ig.shape[1] == 0 or ig[k].max() < iou_thr))

    # ID consistency across consecutive keyframes
    keys = sorted(matches_by_frame)
    pairs_n = switches = 0
    for a, b in zip(keys, keys[1:]):
        for gid, pid in matches_by_frame[a].items():
            if gid in matches_by_frame[b]:
                pairs_n += 1
                switches += matches_by_frame[b][gid] != pid

    # label flicker over all frames
    last: dict[int, str] = {}
    steps = changes = 0
    for fi in sorted(rows):
        for o in rows[fi]["objects"]:
            tid, lab = o["id"], o["label"]
            if not lab:
                continue
            if tid in last:
                steps += 1
                changes += last[tid] != lab
            last[tid] = lab

    rec = n_match / n_gt
    crec = n_correct / n_gt
    prec = n_match / max(1, n_match + n_fp)
    cprec = n_correct / max(1, n_match + n_fp)
    f1 = 2 * crec * cprec / max(1e-9, crec + cprec)
    return {
        "gt_objects": n_gt, "matched": n_match, "correct": n_correct, "false_boxes": n_fp,
        "det_recall": round(rec, 3), "label_acc": round(n_correct / max(1, n_match), 3),
        "correct_recall": round(crec, 3), "precision": round(prec, 3), "correct_prec": round(cprec, 3),
        "f1_correct": round(f1, 3),
        "id_pairs": pairs_n, "id_switches": switches,
        "id_consistency": round(1 - switches / max(1, pairs_n), 3),
        "label_flicker": round(changes / max(1, steps), 4),
        "per_class": {c: {"gt": v[0], "found": v[1], "named_ok": v[2]} for c, v in sorted(per_class.items())},
        "wrong_names": wrong_names,
        "missed_objects": missed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--gt", default=os.path.join(HERE, "gt", "gt.json"))
    ap.add_argument("--out-root", default=os.path.join(HERE, "outputs"))
    args = ap.parse_args()
    global OUT
    OUT = os.path.abspath(args.out_root)
    with open(args.gt) as f:
        gt = json.load(f)

    names = [n for n in ORDER if os.path.isdir(os.path.join(OUT, n))]
    names += sorted(n for n in os.listdir(OUT) if n not in names and os.path.exists(os.path.join(OUT, n, "summary.json")))
    results = {}
    for n in names:
        acc = evaluate(n, gt, args.iou)
        spath = os.path.join(OUT, n, "summary.json")
        if acc is None or not os.path.exists(spath):
            continue
        with open(spath) as f:
            speed = json.load(f)
        results[n] = {"accuracy": acc, "speed": speed}

    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    hdr = ("| pipeline | mean ms/frame | FPS | p95 ms | load s | GPU GB | det recall | name acc | correct recall "
           "| precision | F1 (correct) | ID switches | label flicker | tracks |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for n, r in results.items():
        a, s = r["accuracy"], r["speed"]
        print(f"| {n} | {s['mean_ms']:.0f} | {s['fps_mean']:.2f} | {s['p95_ms']:.0f} | {s['model_load_s']:.0f} "
              f"| {s['peak_gpu_mem_gb']:.1f} | {a['det_recall']:.2f} | {a['label_acc']:.2f} | {a['correct_recall']:.2f} "
              f"| {a['precision']:.2f} | {a['f1_correct']:.2f} | {a['id_switches']}/{a['id_pairs']} "
              f"| {a['label_flicker']:.3f} | {s['unique_track_ids']} |")
    print()
    for n, r in results.items():
        print(f"{n}: stages {r['speed']['stages_mean_ms']}")


if __name__ == "__main__":
    main()
