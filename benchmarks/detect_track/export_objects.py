#!/usr/bin/env python3
"""
List every tracked object per pipeline for manual checking.

Writes, for each outputs/<pipeline>/:
  objects.csv         one row per track: id, final name, all names seen, frames seen, first/last frame, crop file,
                      plus empty 'correct (y/n)' and 'true name' columns to fill in by hand
  object_crops/       one crop per track, taken from the middle frame of the track (padded 10 %)
and outputs/objects_review.html: all pipelines' crops with their names, for side-by-side checking.
"""

import argparse
import csv
import sys
import html
import json
import os
from collections import Counter

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common.bench import VIDEO, read_frames  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--out-root", default=os.path.join(HERE, "outputs"))
_ap.add_argument("--source", default=VIDEO, help="the video / image folder the pipelines were run on")
_args = _ap.parse_args()
OUT = os.path.abspath(_args.out_root)
VIDEO = _args.source
ORDER = ["p1_dtd_every_frame", "p4_dtd_confirm_then_track", "p4b_dtd_confirm_strict", "p2_ram_yoloe_botsort",
         "p3a_ram_gdino_sam2_botsort", "p3b_ram_gdino_sam2_videotrack"]


def main():
    names = [n for n in ORDER if os.path.exists(os.path.join(OUT, n, "predictions.jsonl"))]
    tracks = {}      # pipeline -> id -> info
    want = {}        # frame -> [(pipeline, id, box)]
    for p in names:
        t = {}
        with open(os.path.join(OUT, p, "predictions.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                for o in r["objects"]:
                    d = t.setdefault(o["id"], {"frames": [], "boxes": [], "labels": Counter(), "last": ""})
                    d["frames"].append(r["frame"])
                    d["boxes"].append(o["box"])
                    if o["label"]:
                        d["labels"][o["label"]] += 1
                        d["last"] = o["label"]
        for tid, d in t.items():
            k = len(d["frames"]) // 2
            want.setdefault(d["frames"][k], []).append((p, tid, d["boxes"][k]))
            d["crop"] = f"object_crops/track_{tid:04d}.jpg"
        tracks[p] = t
        os.makedirs(os.path.join(OUT, p, "object_crops"), exist_ok=True)

    last = max(want) if want else -1
    for i, frame in read_frames(VIDEO):
        if i > last:
            break
        H, W = frame.shape[:2]
        for p, tid, (x1, y1, x2, y2) in want.get(i, []):
            px, py = (x2 - x1) * 0.1, (y2 - y1) * 0.1
            a, b = max(0, int(x1 - px)), max(0, int(y1 - py))
            c, d = min(W, int(x2 + px)), min(H, int(y2 + py))
            crop = frame[b:d, a:c]
            s = 320 / max(crop.shape[:2])
            if s < 1:
                crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(OUT, p, tracks[p][tid]["crop"]), crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

    sections = []
    for p in names:
        rows = sorted(tracks[p].items(), key=lambda kv: -len(kv[1]["frames"]))
        with open(os.path.join(OUT, p, "objects.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["track_id", "final_name", "all_names_seen (frames)", "frames_seen", "first_frame", "last_frame",
                        "crop", "correct (y/n)", "true name"])
            for tid, d in rows:
                seen = "; ".join(f"{k} ({v})" for k, v in d["labels"].most_common())
                w.writerow([tid, d["last"], seen, len(d["frames"]), d["frames"][0], d["frames"][-1], d["crop"], "", ""])
        cards = "".join(
            f'<figure><img src="{p}/{d["crop"]}" loading="lazy"><figcaption><b>#{tid} {html.escape(d["last"] or "(no name)")}</b>'
            f'<br>{len(d["frames"])} frames ({d["frames"][0]}–{d["frames"][-1]})'
            + (f'<br><small>{html.escape(", ".join(k for k, _ in d["labels"].most_common(4)))}</small>' if len(d["labels"]) > 1 else "")
            + "</figcaption></figure>"
            for tid, d in rows)
        sections.append(f"<h2>{p} <small>({len(rows)} tracks)</small></h2><div class=g>{cards}</div>")
        print(f"{p}: {len(rows)} tracks -> {os.path.join(OUT, p, 'objects.csv')}")

    with open(os.path.join(OUT, "objects_review.html"), "w") as f:
        f.write("<!doctype html><meta charset=utf-8><title>Tracked objects</title><style>"
                "body{font:14px system-ui;margin:16px;background:#fafafa}h2{margin-top:32px}"
                ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}"
                "figure{margin:0;background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px}"
                "img{width:100%;height:150px;object-fit:contain;background:#eee}figcaption{margin-top:4px}"
                "small{color:#666}</style><h1>Tracked objects per pipeline (" + html.escape(os.path.basename(VIDEO.rstrip("/")) if not os.path.isdir(VIDEO) else VIDEO) + ")</h1>"
                "<p>Sorted by how many frames each track was seen. Crop is from the middle frame of the track.</p>"
                + "".join(sections))
    print(os.path.join(OUT, "objects_review.html"))


if __name__ == "__main__":
    main()
