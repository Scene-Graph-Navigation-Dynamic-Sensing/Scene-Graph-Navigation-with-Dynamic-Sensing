"""Evaluate open-vocabulary object quality on one Replica scene.

Runs the full chain for one scene and collects everything in one folder,
results/replica/<scene>/:

  1. prepare   NICE-SLAM frames -> data/raw/replica_<scene> (scripts/prepare_replica_dataset.py)
  2. run       the static pipeline with ground-truth depth and poses
               (its log, config and visualisations land in the results folder)
  3. evaluate  group A of docs/evaluation_metrics.md: object metrics against
               the Replica semantic mesh (evaluation/open_vocab.py), and
               mIoU / mAcc / F-mIoU with ConceptGraphs' own protocol and
               ground truth (evaluation/conceptgraphs_semseg.py)
  4. report    results.md, metrics.json, tables, video, point clouds, gallery

Steps 1-2 are skipped when their output already exists (or with
--skip_prepare / --skip_run), so the evaluation can be re-run on a finished
pipeline run in a minute. Combine scenes later with
scripts/combine_replica_results.py.

    HF_HUB_OFFLINE=1 python scripts/eval_replica_scene.py --scene office_0 --gpu 2

Tuning: --override merges YAML files onto --config (later files win), and
--tag keeps the variant's run and results apart from the baseline:

    python scripts/eval_replica_scene.py --scene office_0 --tag bright \
        --override configs/replica_variants/brightness.yaml
    -> results/replica_tuning/bright/office_0/
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import artifacts, conceptgraphs_semseg as cg            # noqa: E402
import eval_open_vocab_naming as naming_eval  # noqa: E402  (scripts/, on sys.path when run as a script)
from evaluation.open_vocab import (Node, classify_nodes, match_nodes, node_metrics,  # noqa: E402
                                   retrieval_metrics, text_embeddings)
from evaluation.replica_gt import compute_visibility, load_replica_gt  # noqa: E402

logger = logging.getLogger("eval_replica")
GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")


def run(cmd: list[str], log_path: Path, env: dict) -> None:
    logger.info("$ %s", " ".join(cmd))
    with open(log_path, "a") as log:
        rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    if rc != 0:
        raise SystemExit(f"command failed ({rc}); see {log_path}")


def deep_merge(base: dict, over: dict) -> dict:
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(args) -> dict:
    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    for path in args.override:
        deep_merge(cfg, yaml.safe_load(open(REPO_ROOT / path)) or {})
    return cfg


def pipeline_step(args, out: Path, scene_name: str, raw_dir: Path) -> None:
    cfg = load_config(args)
    cfg["source"]["stride"] = 1                 # stride is applied when preparing the dataset
    viz_rel = (out / "pipeline_visualizations").relative_to(REPO_ROOT)
    cfg["outputs"]["visualizations_dir"] = str(viz_rel)
    cfg_path = out / "config_used.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = dict(os.environ, HF_HUB_OFFLINE="1", CUDA_VISIBLE_DEVICES=str(args.gpu))
    log_path = out / "pipeline.log"
    log_path.unlink(missing_ok=True)
    t0 = time.time()
    run([sys.executable, "scripts/run_static_scene_graph.py", "--config", str(cfg_path), "--source", str(raw_dir),
         "--scene_name", scene_name, "--depth_backend", "reference", "--pose_backend", "reference"], log_path, env)
    log = log_path.read_text()
    if "NOVIC decoder loaded" not in log:
        raise SystemExit("NOVIC did not load (stub decoder used); results would be meaningless. See pipeline.log")
    (out / "pipeline_wall_time_s.txt").write_text(f"{time.time() - t0:.1f}\n")
    # the per-frame debug folders are ~4 MB a frame; keep every 20th
    for sub in ("detection", "semantic_decoding"):
        d = out / "pipeline_visualizations" / sub
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if f.is_dir() and f.name.isdigit() and int(f.name) % 20:
                    shutil.rmtree(f)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return pid > 0
    except OSError:
        return False


def load_frames(processed: Path, capture: dict) -> list[dict]:
    meta = json.loads((processed / "metadata.json").read_text())
    K = np.array(meta["K"], dtype=np.float64).reshape(3, 3)
    frames = []
    for name in capture["frame_names"]:
        pt = np.load(processed / "point" / f"{name}.npy")
        valid = np.load(processed / "mask" / f"{name}.npy")
        depth = np.where(valid, pt[..., 2], 0.0)
        frames.append({"name": name, "K": K, "c2w": np.loadtxt(processed / "poses" / f"{name}.txt"),
                       "depth": depth, "rgb_path": processed / "rgb" / f"{name}.jpg"})
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--stride", type=int, default=10, help="every Nth NICE-SLAM frame (2000 per scene)")
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--results_root", default=None,
                    help="default results/replica, or results/replica_tuning/<tag> with --tag")
    ap.add_argument("--override", nargs="*", default=[], help="YAML files merged onto --config")
    ap.add_argument("--tag", default=None, help="name of this variant (keeps its run and results separate)")
    ap.add_argument("--trim_processed", action="store_true",
                    help="after evaluating, delete the run's per-pixel point/ and mask/ arrays (~560 MB); "
                         "re-evaluating it with --skip_run then needs a fresh pipeline run")
    ap.add_argument("--skip_prepare", action="store_true")
    ap.add_argument("--skip_run", action="store_true")
    ap.add_argument("--prompt", default=cg.PROMPT, help="ConceptGraphs' class prompt")
    ap.add_argument("--match_dist_m", type=float, default=0.05)
    ap.add_argument("--min_purity", type=float, default=0.5)
    ap.add_argument("--min_visible_vertices", type=int, default=50)
    ap.add_argument("--video_fps", type=int, default=5)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    # a variant file may also set the frame stride (replica_eval: {stride: N}); the pipeline ignores the key
    for path in args.override:
        args.stride = int(((yaml.safe_load(open(REPO_ROOT / path)) or {}).get("replica_eval") or {}).get(
            "stride", args.stride))
    results_root = args.results_root or ("results/replica_tuning/" + args.tag if args.tag else "results/replica")
    scene_name = f"replica_{args.scene}" + (f"_{args.tag}" if args.tag else "")
    out = REPO_ROOT / results_root / args.scene
    out.mkdir(parents=True, exist_ok=True)
    # the prepared frames depend only on the stride, so variants share them
    raw_dir = REPO_ROOT / "data" / "raw" / (f"replica_{args.scene}" + ("" if args.stride == 10 else f"_s{args.stride}"))
    processed = REPO_ROOT / "data" / "processed" / scene_name
    pkl = REPO_ROOT / "outputs" / "scene_graphs" / f"{scene_name}.pkl"

    # 1. prepare
    if not (args.skip_prepare or (raw_dir / "source.json").exists()):
        run([sys.executable, "scripts/prepare_replica_dataset.py", "--scene", args.scene,
             "--stride", str(args.stride), "--out", str(raw_dir)], out / "prepare.log", dict(os.environ))
    # 2. run (a lock file stops two processes running the same scene and variant at once;
    # the second one waits and then reuses the first one's scene graph)
    lock = out / ".pipeline_running"
    while lock.exists() and pid_alive(int(lock.read_text() or 0)):
        logger.info("another process (pid %s) is running this pipeline; waiting", lock.read_text())
        time.sleep(60)
    if not (args.skip_run or pkl.exists()):
        lock.write_text(str(os.getpid()))
        try:
            pipeline_step(args, out, scene_name, raw_dir)
        finally:
            lock.unlink(missing_ok=True)

    # 3. evaluate
    cfg = load_config(args)
    from object_fusion._dovsg_import import ensure_dovsg_importable
    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import load_scene_graph
    from semantic_decoding.novic.novic_decoder import NOVICDecoder

    payload = load_scene_graph(pkl)
    vmap = PersistedVoxelMap.from_dict(payload["voxel_map"])
    capture = json.loads((processed / "capture.json").read_text())
    T_align = np.array(json.loads((processed / "gravity_alignment.json").read_text())["transform"])
    frames = load_frames(processed, capture)

    nodes = []
    for obj in payload["instance_objects"]:
        nid = obj["class_id"]
        if nid not in payload["node_embeddings"] or len(obj.get("indexes", [])) == 0:
            continue
        nodes.append(Node(
            node_id=nid, name=obj.get("class_name", nid),
            novic_top_k=[str(t[0]) for t in (obj.get("novic_top_k") or [])][:5] or [obj.get("class_name", nid)],
            embedding=np.asarray(payload["node_embeddings"][nid], dtype=np.float32),
            points=vmap.index_to_point(np.asarray(obj["indexes"])) + vmap.resolution / 2,
            detections=[(str(f), np.asarray(b)) for f, b in zip(obj.get("image_name", []), obj.get("xyxy", []))]))
    logger.info("%d nodes with embeddings", len(nodes))

    gt = load_replica_gt(GT_ROOT / args.scene)
    gt_xyz_map = gt.xyz @ T_align[:3, :3].T + T_align[:3, 3]        # mesh frame == trajectory frame
    compute_visibility(gt, [(f["K"], f["c2w"], f["depth"]) for f in frames], T_align,
                       min_visible_vertices=args.min_visible_vertices)
    logger.info("GT: %d objects, %d visible countable", len(gt.objects), len(gt.countable_visible()))

    match_nodes(nodes, gt_xyz_map, gt.object_id, gt.kind, gt.label, args.match_dist_m, args.min_purity)
    novic_cfg = cfg["semantic_decoding"]
    decoder = NOVICDecoder(novic_dir=novic_cfg["novic_dir"], checkpoint_path=novic_cfg["novic_checkpoint"],
                           gencfg=novic_cfg["gencfg"], device="cuda")
    if decoder.is_stub:
        raise SystemExit("NOVIC failed to load; run with HF_HUB_OFFLINE=1")
    vocab = gt.vocabulary
    vocab_emb = text_embeddings(decoder, vocab, args.prompt)
    classify_nodes(nodes, vocab, vocab_emb, decoder, args.prompt)
    cg_feats = cg.class_text_features(decoder)
    naming_vocab = naming_eval.load_vocabulary(decoder)       # NOVIC's target nouns + their text embeddings
    decoder.close()

    ret = retrieval_metrics(nodes, len(vocab))
    node_m = node_metrics(nodes, gt.countable_visible(), lambda o: gt.label(o))

    # ConceptGraphs segmentation: their GT cloud, moved into our map frame
    all_idx = np.array(list(vmap.indexes_colors_mapping_dict.keys()))
    recon = vmap.index_to_point(all_idx) + vmap.resolution / 2
    try:
        cg_xyz, cg_cls = cg.load_gt(args.scene)
    except (FileNotFoundError, OSError) as e:      # object metrics only need Replica's own mesh
        logger.warning("ConceptGraphs GT cloud missing for %s (%s): segmentation skipped", args.scene, e)
        seg_m = {}
    else:
        cg_xyz = cg_xyz @ T_align[:3, :3].T + T_align[:3, 3]
        seg_m = {n: cg.evaluate(cg_xyz, cg_cls, [n_.points for n_ in nodes],
                                np.stack([n_.embedding for n_ in nodes]), cg_feats, recon, n)
                 for n in (1, 4, 6)}

    # 4. report
    timings = json.loads((processed / "stage_timings.json").read_text()) if (processed / "stage_timings.json").exists() else {}
    source = json.loads((raw_dir / "source.json").read_text())
    info = {"scene": args.scene, "evaluated": time.strftime("%Y-%m-%d %H:%M"), "frames": len(frames),
            "stride": source.get("stride"), "pose_depth": "ground truth", "prompt": args.prompt,
            "match_dist_m": args.match_dist_m, "min_purity": args.min_purity,
            "min_visible_vertices": args.min_visible_vertices, "alignment_check": source.get("alignment_check"),
            "variant": args.tag or "baseline", "overrides": args.override,
            "pipeline_stage_seconds": timings}
    artifacts.dump_json(out / "metrics.json", {"info": info, "retrieval": ret, "nodes": node_m,
                                               "segmentation": seg_m})

    node_colors = {n.node_id: c for n, c in zip(nodes, artifacts.palette(len(nodes)))}
    artifacts.write_tables(out, nodes, gt, node_m, seg_m)
    all_rgb = np.array(list(vmap.indexes_colors_mapping_dict.values()), dtype=np.float64)
    if all_rgb.max() <= 1.0:
        all_rgb *= 255
    artifacts.write_point_clouds(out, nodes, recon, all_rgb,
                                 gt_xyz_map, gt, node_colors)
    artifacts.write_topdown(out, nodes, gt_xyz_map, gt, node_colors)
    artifacts.write_gallery(out, nodes, lambda name: processed / "rgb" / f"{name}.jpg",
                            {f["name"]: f for f in frames})
    for f in frames:
        f["rgb"] = cv2.cvtColor(cv2.imread(str(f["rgb_path"])), cv2.COLOR_BGR2RGB)
    artifacts.write_video(out / "scene_with_object_names.mp4", frames, nodes, node_colors, gt_xyz_map, gt,
                          fps=args.video_fps)

    sg_dir = out / "scene_graph"
    sg_dir.mkdir(exist_ok=True)
    for src, dst in ((pkl, sg_dir / pkl.name), (pkl.with_suffix(".json"), sg_dir / "scene_graph.json"),
                     (out / "pipeline_visualizations" / "scene_graph" / "scene_graph.pdf", sg_dir / "scene_graph.pdf"),
                     (out / "pipeline_visualizations" / "scene_graph" / "scene_graph", sg_dir / "scene_graph.dot")):
        artifacts.copy_if_exists(src, dst)

    files = [
        ("results.md", "this report"),
        ("metrics.json", "every number in this report, plus raw ranks and the confusion matrix for pooling scenes"),
        ("scene_with_object_names.mp4", "the walkthrough: left, our nodes re-projected with each detection boxed "
                                        "and named (box colour = status); right, the GT objects with their class"),
        ("nodes_gallery.jpg", "one crop per node (the detection that best matches the node's 3D points) with "
                              "name, GT class and status"),
        ("node_crops/", "the same crops, one file per node"),
        ("topdown_prediction_vs_gt.png", "top-down map: our nodes vs the GT objects"),
        ("voxel_map_rgb.ply", "the reconstructed voxel map in true colour (open in MeshLab / CloudCompare)"),
        ("voxel_map_objects.ply", "object voxels, one colour per node"),
        ("voxel_map_status.ply", "object voxels coloured by status (green correct, orange wrong name, red invalid, "
                                 "grey not counted)"),
        ("gt_objects.ply", "the GT surface samples in the same frame, one colour per countable object"),
        ("scene_graph/", "the scene graph: .pkl (full), .json, .pdf render, .dot"),
        ("nodes.csv", "per node: name, top-5, GT match, purity, embedding ranking, status"),
        ("gt_objects.csv", "per GT object: class, visibility, which nodes found it"),
        ("per_class_segmentation.csv", "IoU and accuracy per class, for each ConceptGraphs exclusion setting"),
        ("pipeline.log", "the pipeline run's log"),
        ("config_used.yaml", "the exact pipeline configuration"),
        ("pipeline_visualizations/", "the pipeline's own stage images (every 20th frame of detection / naming)"),
    ]
    artifacts.write_results_md(out, args.scene, info, nodes, ret, node_m, seg_m, files)
    naming_eval.evaluate_dir(out, payload["node_embeddings"], *naming_vocab)   # open-vocabulary naming section
    if args.trim_processed:
        for sub in ("point", "mask"):
            shutil.rmtree(processed / sub, ignore_errors=True)
    logger.info("node precision %s, recall %s, N_dup %d, Acc@1 %s, mIoU %s -> %s",
                node_m["node_precision"], node_m["node_recall"], node_m["n_dup"],
                ret.get("acc_at_k", {}).get(1), seg_m.get(1, {}).get("miou"), out / "results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
