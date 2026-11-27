"""Where does object fusion spend its time? Reads py-spy raw profiles
(results/fusion_profile/<scene>.txt, from results/fusion_profile/run_profiles.sh)
and splits the samples of the fusion thread, and of the final fusion stage on the
main thread, into the steps of the fusion algorithm.

py-spy's raw format: one line per distinct stack, "frame;frame;...;frame count",
outermost first, each frame "function (file:line)". At 10 samples/s a count of
10 is about one second.

    python scripts/analyze_fusion_profile.py results/fusion_profile/office_1.txt [...]
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

RATE = 10.0          # samples per second (run_profiles.sh --rate 10)

# (label, test on the stack's list of frames); the first match wins, so specific before general
STEPS = [
    ("commit: voxel overlap, reverse (each candidate vs the track)",
     lambda f: has(f, "compute_overlap_matrix_2set") and has(f, "_similarity") and on_line(f, "_similarity", REVERSE)),
    ("commit: voxel overlap, forward (track vs each candidate)",
     lambda f: has(f, "compute_overlap_matrix_2set") and has(f, "_similarity")),
    ("commit: other spatial similarity (boxes, IoU)", lambda f: has(f, "compute_spatial_similarities") and has(f, "_commit")),
    ("commit: visual (cosine) similarity", lambda f: has(f, "compute_visual_similarities") and has(f, "_commit")),
    ("commit: DBSCAN clean-up of the track", lambda f: has(f, "_commit") and has(f, "process_indexes")),
    ("commit: merge into the matched object", lambda f: has(f, "_commit") and has(f, "merge_obj_to_obj")),
    ("commit: other", lambda f: has(f, "_commit")),
    ("every 20 frames: DBSCAN over all objects", lambda f: has(f, "denoise_objects") and has(f, "add_frame")),
    ("per frame: merge detection into its pending track", lambda f: has(f, "add_frame") and has(f, "merge_obj_to_obj")),
    ("per frame: build detections (mask -> voxels)", lambda f: has(f, "build_detection_list_from_frame")),
    ("per frame: other", lambda f: has(f, "add_frame")),
    ("waiting (for voxels / detections)", lambda f: leaf_waits(f)),
]
FINAL_STEPS = [
    ("final: re-match pass, voxel overlap", lambda f: has(f, "_rematch") and has(f, "compute_overlap_matrix_2set")),
    ("final: re-match pass, other", lambda f: has(f, "_rematch")),
    ("final: DBSCAN over all objects", lambda f: has(f, "denoise_objects")),
    ("final: overlap merge (merge_objects)", lambda f: has(f, "merge_objects")),
    ("final: voxel ownership (indexes_align_objects)", lambda f: has(f, "indexes_align_objects")),
    ("final: other", lambda f: True),
]
REVERSE = set()       # line numbers of the reverse overlap call in _similarity, read from the code


def has(frames, name):
    return any(fr[0] == name for fr in frames)


def on_line(frames, func, lines):
    return any(fr[0] == func and fr[2] in lines for fr in frames)


def leaf_waits(frames):
    return frames[-1][0] in ("wait", "_wait_for_tstate_lock", "acquire", "get", "result")


def parse(line):
    stack, _, count = line.rpartition(" ")
    frames = []
    for fr in stack.split(";"):
        m = re.match(r"(.*?) \((.*?):(\d+)\)$", fr.strip())
        frames.append((m.group(1), m.group(2), int(m.group(3))) if m else (fr.strip(), "", 0))
    return frames, int(count)


def reverse_lines():
    """Line of the second compute_spatial_similarities call in _similarity (the reverse direction)."""
    src = (Path(__file__).resolve().parent.parent / "object_fusion" / "instance_fusion.py").read_text().splitlines()
    start = next(i for i, l in enumerate(src) if "def _similarity" in l)
    calls = [i + 1 for i in range(start, start + 40) if "compute_spatial_similarities" in src[i]]
    return {calls[1]} if len(calls) > 1 else set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("profiles", nargs="+")
    ap.add_argument("--top", type=int, default=12, help="hottest leaf lines inside fusion to list")
    args = ap.parse_args()
    REVERSE.update(reverse_lines())

    for path in map(Path, args.profiles):
        fusion, final, leaves = Counter(), Counter(), Counter()
        n_fusion = n_final = 0
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            frames, count = parse(line)
            if has(frames, "_fusion_loop"):
                n_fusion += count
                fusion[next(lbl for lbl, test in STEPS + [("other", lambda f: True)] if test(frames))] += count
                if not leaf_waits(frames):
                    leaf = next((fr for fr in reversed(frames) if fr[1].endswith(("instance_fusion.py", "instance_process.py"))), frames[-1])
                    leaves[f"{leaf[0]} ({Path(leaf[1]).name}:{leaf[2]}) -> {frames[-1][0]}"] += count
            elif has(frames, "finalize") or has(frames, "_finalize"):
                n_final += count
                final[next(lbl for lbl, test in FINAL_STEPS if test(frames))] += count

        print(f"\n=== {path.stem}: fusion thread {n_fusion / RATE / 60:.1f} min sampled, "
              f"final fusion stage {n_final / RATE / 60:.1f} min")
        for title, counter, total in (("Fusion thread (during streaming)", fusion, n_fusion),
                                      ("Final fusion stage", final, n_final)):
            if not total:
                continue
            print(f"\n{title}:")
            for label, c in counter.most_common():
                print(f"  {c / total:6.1%}  {c / RATE / 60:6.1f} min  {label}")
        print(f"\nHottest lines inside fusion code (non-waiting fusion-thread samples):")
        busy = sum(leaves.values())
        for k, c in leaves.most_common(args.top):
            print(f"  {c / busy:6.1%}  {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
