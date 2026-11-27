"""Split prompt-free YOLOE's built-in vocabulary (4,585 tags) into physical objects and non-objects, for the
detection.validation.label_vocab filter (perception/detection/detection_validation.py).

A tag is an OBJECT when any of its 3 most common WordNet noun senses (for a multi-word tag WordNet doesn't know, its
last word's) is a physical entity outside these sub-trees: place, room, building, surface, material (food excepted),
landform, body of water, body part, natural process. It is a NON-object otherwise (abstract tags such as "action
film", places such as "office"), and always when it is a surface / place word of detection_validation.STUFF_LABELS. KEEP overrides tags whose
first WordNet sense is not the one YOLOE means indoors.

    python scripts/build_yoloe_object_vocab.py            # writes configs/vocab/yoloe_pf_nonobject.txt + a report
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# WordNet sub-trees that are not objects a robot would map, checked on the tag's first noun sense
NON_OBJECT_ROOTS = {
    "location.n.01": "place", "room.n.01": "room", "area.n.05": "room / area", "building.n.01": "building",
    "housing.n.01": "building", "establishment.n.04": "building",
    "surface.n.01": "surface", "wall.n.01": "wall", "floor.n.01": "floor", "ceiling.n.01": "ceiling",
    "roof.n.01": "roof", "substance.n.01": "material", "matter.n.03": "material",
    "geological_formation.n.01": "landform", "body_of_water.n.01": "water", "body_part.n.01": "body part",
    "process.n.06": "process", "natural_phenomenon.n.01": "phenomenon",
}
# first WordNet sense is not the indoor-object meaning YOLOE uses
KEEP = {"light", "lighting", "lamp light", "plant", "houseplant", "potted plant", "glass", "paper", "tissue",
        "towel", "cloth", "fabric", "food", "fruit", "bread", "cake", "water bottle", "tile", "cushion", "pillow",
        "screen", "monitor", "board", "panel", "vent", "switch", "plug", "outlet", "socket", "pipe", "door",
        "window", "pillar", "column", "stair", "stairs", "shelf", "rug", "carpet", "mat", "curtain", "blind",
        "table", "toilet paper", "card", "cardboard", "CD", "detergent", "chessboard", "clipboard", "dartboard",
        "pegboard", "projection screen", "fireplace", "drawer", "locker", "vacuum", "table top"}
# Reasons that mark a tag as a scene word (place, room, building, surface, landscape). The narrow filter
# (configs/vocab/yoloe_pf_scene_tags.txt) drops only these: YOLOE often gives real objects odd abstract, shape or
# colour tags ("rectangle" for a switch, "dot" for a screen, "straddle" for a chair), so dropping every non-object
# tag also loses 15% of real-object detections on Replica (rashmika/option3_yoloe_labels/label_filter_check.json).
SCENE_REASONS = {"surface / place word", "place", "room", "room / area", "building", "surface", "landform", "water",
                 "roof"}
# places whose other WordNet senses are artifacts (a corridor is a "passage", a library a "room")
DROP = {"corridor", "library", "hallway", "aisle", "lobby", "passageway"}


def senses(word: str, k: int = 3):
    """The tag's k most common WordNet noun senses (multi-word tags WordNet doesn't know: their last word's)."""
    from nltk.corpus import wordnet as wn
    s = wn.synsets(word.replace(" ", "_"), pos=wn.NOUN)
    if not s and " " in word:
        s = wn.synsets(word.split()[-1], pos=wn.NOUN)
    return s[:k]


def sense_kind(s) -> tuple[bool, str]:
    hyper = {h.name() for path in s.hypernym_paths() for h in path}
    if "physical_entity.n.01" not in hyper:
        return False, "abstract"
    for root, why in NON_OBJECT_ROOTS.items():
        if root in hyper:
            if root in ("substance.n.01", "matter.n.03") and ("food.n.01" in hyper or "food.n.02" in hyper):
                continue                                  # a food item on a table is an object, not a material
            return False, why
    return True, "physical object"


def classify(tag: str) -> tuple[bool, str]:
    """-> (is_object, reason). Object if ANY of the tag's 3 most common noun senses is a physical object outside
    the excluded sub-trees (WordNet's first sense is often not the indoor one: mug = face, microwave = radiation);
    surface / place words are non-objects whatever their senses."""
    from perception.detection.detection_validation import is_stuff_label
    t = tag.lower().strip()
    if t in KEEP:
        return True, "keep list"
    if t in DROP:
        return False, "place"
    if is_stuff_label(t):
        return False, "surface / place word"
    ss = senses(t)
    if not ss:
        return True, "not in WordNet (kept)"
    kinds = [sense_kind(x) for x in ss]
    for ok, why in kinds:
        if ok:
            return True, "physical object"
    return False, kinds[0][1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(REPO_ROOT / "external/Scene-Graph-Generation/yoloe-26x-seg-pf.pt"))
    ap.add_argument("--out", default=str(REPO_ROOT / "configs/vocab/yoloe_pf_nonobject.txt"))
    args = ap.parse_args()
    from ultralytics import YOLOE
    names = [str(n) for n in YOLOE(args.weights).names.values()]
    res = {n: classify(n) for n in names}
    non = sorted(n for n, (ok, _) in res.items() if not ok)
    Path(args.out).write_text("# prompt-free YOLOE tags that are not physical objects "
                              "(scripts/build_yoloe_object_vocab.py)\n" + "\n".join(non) + "\n")
    scene = sorted(n for n, (ok, r) in res.items() if not ok and r in SCENE_REASONS)
    (Path(args.out).parent / "yoloe_pf_scene_tags.txt").write_text(
        "# prompt-free YOLOE tags that name a place, room, building or surface (narrow label_vocab filter; "
        "scripts/build_yoloe_object_vocab.py)\n" + "\n".join(scene) + "\n")
    print(f"narrow (scene) list: {len(scene)} tags")
    why = Counter(r for ok, r in res.values() if not ok)
    print(f"{len(names)} tags: {len(names) - len(non)} objects, {len(non)} non-objects -> {args.out}")
    print("non-object reasons:", dict(why.most_common()))
    by = {}
    for n, (ok, r) in res.items():
        by.setdefault(("OBJ" if ok else "NON") + " " + r, []).append(n)
    for k, v in sorted(by.items()):
        print(f"  {k} ({len(v)}): {', '.join(v[:25])}")
    # sanity: Replica's countable classes that are also YOLOE tags must be objects
    rep = REPO_ROOT / "results/option2_panoptic/run1"
    import csv
    classes = {r["class"] for s in rep.iterdir() if (s / "gt_objects.csv").exists()
               for r in csv.DictReader(open(s / "gt_objects.csv")) if r["kind"] == "countable"}
    bad = sorted(c for c in classes if c in res and not res[c][0])
    print("Replica countable classes classified as non-objects:", bad or "none")
    (Path(args.out).with_suffix(".report.json")).write_text(json.dumps({n: r for n, (_, r) in res.items()}, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
