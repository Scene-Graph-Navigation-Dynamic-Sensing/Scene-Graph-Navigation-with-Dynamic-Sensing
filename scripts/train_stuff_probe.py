"""Train the thing/stuff probe used at track commit (object_fusion.track_validation.stuff_probe).

Logistic regression on node embeddings (NOVIC's CLIP, the same averaged embedding a track has at commit):
stuff = nodes lying mostly on wall / floor / ceiling (status "structural"), thing = nodes on one real object
(correct / wrong_name). Mixed, unlabelled and floating nodes are left out. The threshold is the lowest probability
that rejects at most --max_thing_reject of the training things.

--extra adds non-Replica samples (scripts/build_probe_data_3rscan.py: 3RScan captures run through the same front
end), so the probe is not fitted to Replica's look alone.

Leave-scenes-out (--folds): one probe per held-out pair of Replica scenes, trained on the other six (+ --extra) and
tested on the pair; each fold also writes configs/replica_variants/probe_heldout_<a>_<b>.yaml, so an evaluation of
scene a or b uses a probe that never saw it.

    python scripts/train_stuff_probe.py --out configs/stuff_probe_replica.npz                 # one probe
    python scripts/train_stuff_probe.py --folds --extra data/cache/probe_3rscan.npz          # 4 held-out probes
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCENES = ["office_0", "office_1", "office_2", "office_3", "office_4", "room_0", "room_1", "room_2"]
FOLDS = [("office_0", "room_0"), ("office_1", "room_1"), ("office_2", "room_2"), ("office_3", "office_4")]


def load(results_root: Path, scenes: list[str]):
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "pipeline.yaml"))
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from scene_graph.storage.serializer import load_scene_graph
    X, y, meta = [], [], []
    for s in scenes:
        rd = results_root / s
        emb = load_scene_graph(next((rd / "scene_graph").glob("*.pkl")))["node_embeddings"]
        for r in csv.DictReader(open(rd / "nodes.csv")):
            lab = {"structural": 1, "correct": 0, "wrong_name": 0}.get(r["status"])
            if lab is None or r["node_id"] not in emb:
                continue
            e = np.asarray(emb[r["node_id"]], np.float32).ravel()
            X.append(e / np.linalg.norm(e))
            y.append(lab)
            meta.append((s, r["node_id"], r["gt_class"], r["novic_name"]))
    return np.stack(X), np.array(y), meta


def fit(X, y, max_thing_reject):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(X, y)
    w, b = clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
    things = np.sort(1 / (1 + np.exp(-(X[y == 0] @ w + b))))
    thr = float(things[int(np.ceil((1 - max_thing_reject) * len(things))) - 1]) if len(things) else 0.9
    return w, b, thr


def prob(X, w, b):
    return 1 / (1 + np.exp(-(X @ w + b)))


def report(name, X, y, meta, w, b, thr) -> dict:
    rej = prob(X, w, b) > thr
    r = {"nodes": int(len(y)), "stuff": int(y.sum()), "stuff_rejected": int(rej[y == 1].sum()),
         "things": int((y == 0).sum()), "things_rejected": int(rej[y == 0].sum()),
         "things_rejected_list": [f"{m[0]}:{m[2]}({m[3]})" for m, k, t in zip(meta, rej, y) if k and t == 0]}
    print(f"  {name}: rejects {r['stuff_rejected']}/{r['stuff']} stuff and {r['things_rejected']}/{r['things']} things"
          + (f"  {r['things_rejected_list']}" if r["things_rejected_list"] else ""))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_root", default="results/replica_final")
    ap.add_argument("--train", nargs="+", default=["office_1", "office_2", "office_3", "office_4", "room_1", "room_2"])
    ap.add_argument("--test", nargs="+", default=["office_0", "room_0"])
    ap.add_argument("--extra", nargs="*", default=[], help="npz files with X, y (e.g. data/cache/probe_3rscan.npz)")
    ap.add_argument("--max_thing_reject", type=float, default=0.02)
    ap.add_argument("--out", default="configs/stuff_probe_replica.npz")
    ap.add_argument("--folds", action="store_true", help="train one held-out probe per pair in FOLDS")
    ap.add_argument("--fold_dir", default="configs/stuff_probe")
    args = ap.parse_args()

    data = {s: load(REPO_ROOT / args.results_root, [s]) for s in SCENES}
    ex = [np.load(REPO_ROOT / p) for p in args.extra]
    Xe = np.concatenate([e["X"] for e in ex]) if ex else np.zeros((0, data[SCENES[0]][0].shape[1]), np.float32)
    ye = np.concatenate([e["y"] for e in ex]) if ex else np.zeros(0, int)
    if len(ye):
        print(f"extra (non-Replica): {len(ye)} samples, {int(ye.sum())} stuff")

    def cat(scenes):
        X = np.concatenate([data[s][0] for s in scenes])
        y = np.concatenate([data[s][1] for s in scenes])
        return X, y, [m for s in scenes for m in data[s][2]]

    folds = FOLDS if args.folds else [tuple(args.test)]
    summary = {}
    for test in folds:
        train = [s for s in (SCENES if args.folds else args.train) if s not in test]
        Xr, yr, _ = cat(train)
        Xt, yt, mt = cat(list(test))
        print(f"held out {', '.join(test)}; trained on {', '.join(train)}" + (" + extra" if len(ye) else ""))
        res = {}
        for name, (X, y) in {"replica_only": (Xr, yr), "replica+extra": (np.concatenate([Xr, Xe]),
                                                                           np.concatenate([yr, ye]))}.items():
            if name == "replica+extra" and not len(ye):
                continue
            w, b, thr = fit(X, y, args.max_thing_reject)
            print(f" {name}: threshold {thr:.3f}")
            res[name] = {"threshold": thr, "test": report("held-out", Xt, yt, mt, w, b, thr)}
            if len(ye) and name == "replica_only":
                res[name]["extra"] = report("3RScan (unseen)", Xe, ye, [("3rscan", "", "", "")] * len(ye), w, b, thr)
            final = (w, b, thr)                                          # the last one written is replica+extra
        w, b, thr = final
        if args.folds:
            out = REPO_ROOT / args.fold_dir / f"heldout_{test[0]}_{test[1]}.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out, w=w, b=b, threshold=thr, train=np.array(train))
            (REPO_ROOT / "configs" / "replica_variants" / f"probe_heldout_{test[0]}_{test[1]}.yaml").write_text(
                f"# Thing/stuff probe trained WITHOUT {test[0]} and {test[1]} (scripts/train_stuff_probe.py --folds)"
                + (" + non-Replica data" if len(ye) else "") + "\nobject_fusion:\n  track_validation:\n"
                f"    stuff_probe: {{enabled: true, path: {out.relative_to(REPO_ROOT)}, threshold: {thr:.4f}}}\n")
        else:
            out = REPO_ROOT / args.out
            np.savez(out, w=w, b=b, threshold=thr, train=np.array(train))
        print(f" wrote {out.relative_to(REPO_ROOT)}")
        summary["+".join(test)] = res
    if args.folds:
        (REPO_ROOT / args.fold_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
