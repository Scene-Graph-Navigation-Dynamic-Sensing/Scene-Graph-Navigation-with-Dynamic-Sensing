#!/usr/bin/env python3
"""Run the co-visibility gate as a live stream: a recorded video is replayed keyframe by keyframe at its
camera rate and the engine (engine.py) keeps the belief of every memory object up to date.

The replay stands in for the live front end: each keyframe arrives with the depth and pose that
`run_static_scene_graph.py --pose_only` already stored (DA3 depth, DROID-SLAM pose); the keyframe is
pushed at its scheduled time (--rate keyframes per second; 0 = as fast as possible).

    CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python dynamic_updates/covis_gate/online/run_covis_live.py \\
        --config configs/pipeline_lap_videos.yaml --memory lap_in_1 --dataset_dir data/processed/lap_remove_2 \\
        --scene_name lap_remove_2_live --rate 10

Writes outputs/covis_live/<scene_name>/{summary.json, events.json, timeline.csv}.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GATE = HERE.parent
REPO_ROOT = GATE.parents[1]
for p in (str(REPO_ROOT), str(GATE.parent / "dovsg_method"), str(GATE.parent / "projection_method"), str(GATE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

logger = logging.getLogger("covis_gate")


def load_keyframes(video_dir: Path, workers: int = 8) -> tuple[list, np.ndarray]:
    """The recorded video as the live front end would hand it over: rgb, depth, valid-depth mask, pose."""
    from PIL import Image

    from ace_utils import read_K
    from engine import Keyframe

    names = sorted(p.stem for p in (video_dir / "poses").glob("*.txt"))
    K = read_K(video_dir)

    def one(i_name):
        i, name = i_name
        rgb = np.asarray(Image.open(video_dir / "rgb" / f"{name}.jpg"), dtype=np.uint8)
        point = np.load(video_dir / "point" / f"{name}.npy")
        mask = np.load(video_dir / "mask" / f"{name}.npy").astype(bool)
        return Keyframe(i, name, rgb, np.ascontiguousarray(point[..., 2], dtype=np.float32), mask,
                        np.loadtxt(video_dir / "poses" / f"{name}.txt"))

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, enumerate(names))), K


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline_lap_videos.yaml")
    ap.add_argument("--memory", required=True)
    ap.add_argument("--dataset_dir", required=True, help="the video to replay (a --pose_only run of run_static_scene_graph.py)")
    ap.add_argument("--scene_name", required=True)
    ap.add_argument("--rate", type=float, default=10.0, help="keyframes per second to replay at (0: as fast as possible)")
    ap.add_argument("--delta_t_days", type=float, help="days between the memory's visit and this one (sets the survival prior S(dt))")
    ap.add_argument("--memory_data", help="folder of data/processed with the memory's frames when it is not named like the memory: "
                                          "a memory written by an earlier visit has the frames of the video that built it")
    ap.add_argument("--no_observe", action="store_true",
                    help="evidence only: absent objects are removed, nothing is added or moved (no detector, no graph update)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--switch_interval_ms", type=float, default=0.2,
                    help="Python's thread switch interval; the default 5 ms makes the main loop wait for the localizer thread")
    args = ap.parse_args()
    sys.setswitchinterval(args.switch_interval_ms / 1000)

    from scripts.run_static_scene_graph import load_config

    cfg = load_config(REPO_ROOT / args.config)
    import run_covis_update as RC
    from engine import ONLINE_DEFAULTS, MemoryModel, OnlineCovis
    from observe import OBSERVE_DEFAULTS

    ccfg = {**RC.DEFAULT_CFG, **ONLINE_DEFAULTS, **OBSERVE_DEFAULTS, **(cfg.get("covis_update") or {})}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in ccfg:
            sys.exit(f"unknown setting {k}")
        ccfg[k] = (v.lower() in ("1", "true", "yes", "on")) if isinstance(ccfg[k], bool) else type(ccfg[k])(v)
    if args.delta_t_days is not None:
        ccfg["delta_t_days"] = args.delta_t_days
    out = REPO_ROOT / "outputs" / "covis_live" / args.scene_name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    from object_fusion._dovsg_import import ensure_dovsg_importable

    ensure_dovsg_importable(cfg["external"]["scene_graph_generation_root"])
    import torch

    video_dir = Path(args.dataset_dir)
    video_dir = video_dir if video_dir.is_absolute() else REPO_ROOT / video_dir
    frames, K = load_keyframes(video_dir)
    observe = ccfg["observe"] and not args.no_observe

    t = time.time()
    worker = None
    if observe:                                    # the detector's models load in their own process while the rest starts
        import selective
        import warm_detect

        job_root = Path(ccfg["job_dir"])
        job_root.mkdir(parents=True, exist_ok=True)
        worker = selective.WorkerPool(ccfg["observe_workers"], warm_detect.run_config(cfg, job_root / "viz"), job_root,
                                      {"every": ccfg["embed_every"], "growth": ccfg["embed_growth"], "fast_voxel_map": True})
    model = MemoryModel(REPO_ROOT, cfg, ccfg, args.memory, "cuda", args.memory_data)
    engine = OnlineCovis(model, ccfg, K)
    manager = None
    if observe:
        from observe import ObservationManager

        manager = ObservationManager(model, engine, worker, ccfg, K, json.loads((video_dir / "metadata.json").read_text()),
                                     source_dir=video_dir)
        worker.wait_loaded()
    torch.cuda.synchronize()
    startup_s = time.time() - t
    logger.info(f"[startup] memory {model.load_s}s, engine {engine.setup_s}s, detector {'ready' if observe else 'off'} "
                f"({startup_s:.1f}s in all); {len(frames)} keyframes to replay at {args.rate} Hz")

    import update_graph as UG

    publisher = UG.GraphWorker(REPO_ROOT, args.config, args.memory)
    t0 = time.perf_counter()
    for i, kf in enumerate(frames):
        if args.rate > 0:
            due = t0 + i / args.rate
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            kf.t_arrival = due
        else:
            kf.t_arrival = time.perf_counter()
        rec = engine.push(kf)
        for e in rec["events"]:
            logger.info(f"[t={e['wall_s']:5.1f}s frame {e['frame']:3d}] {e['object']}: {e['previous']} -> {e['label']} "
                        f"(P {e['P']}, coverage {e['coverage']}, {e['evidence_frames']} evidence frames)")
        publisher.poll()
        if manager is not None:
            manager.on_events(rec["events"], kf.idx)
            for job in manager.step(kf.idx):      # a window finished: decide with what is known now
                assoc = manager.associate(list(engine.labels), ccfg)
                final = UG.decisions(model, engine.labels, assoc)
                publisher.request(final, assoc)
                logger.info(f"[t={time.perf_counter() - t0:5.1f}s] after job {job.id}: "
                            f"added {[r['object'] for r in assoc['rows'] if r['decision'] == 'added']}, "
                            f"moved {list(assoc['moved'])}, removed {[c for c, l in final.items() if l == 'removed']}")
    stream_s = time.perf_counter() - t0
    t_fin = time.perf_counter()
    for e in engine.finish():
        logger.info(f"[flush frame {e['frame']:3d}] {e['object']}: {e['previous']} -> {e['label']} (P {e['P']}, coverage {e['coverage']})")
        if manager is not None:
            manager.on_events([e], frames[-1].idx)
    finish_s = time.perf_counter() - t_fin
    engine.close()

    # ── the end of the stream: let the open windows finish, associate, change the graph, learn ──
    update = {}
    if manager is not None:
        for i in range(model.n_obj):               # objects still undecided with evidence: look at them too, as the batch method does
            if engine.labels[i] == "deferred" and engine.best_frame[i] >= 0:
                manager._schedule(int(engine.best_frame[i]), f"deferred {model.names[i]}", frames[-1].idx)
        manager.step(frames[-1].idx, flush=True)
        t_wait = time.perf_counter()
        while not manager.settled() or manager.triggers:
            manager.step(frames[-1].idx, flush=True)
            time.sleep(0.05)
            if time.perf_counter() - t_wait > 300:
                logger.error("[observe] jobs did not finish in 300 s")
                break
        update["jobs_wait_s"] = round(time.perf_counter() - t_wait, 2)
    t_up = time.perf_counter()
    assoc = manager.associate(list(engine.labels), ccfg) if manager is not None and manager.observed else None
    final = UG.decisions(model, engine.labels, assoc)
    publisher.request(final, assoc)
    publisher.wait()
    saved = publisher.save(args.scene_name, model.sg_dir)
    prior_table = {}                               # §3: each object's move-rate distribution before this visit
    for n, c in enumerate(model.names):
        r = model.dyn.rec(c)
        prior_table[c] = {"a": round(r["a"], 3), "b": round(r["b"], 3), "rate_per_day": round(r["a"] / r["b"], 4),
                          "dt_days": round(max(model.t_now - r["t_last"], 0.0), 3), "S": round(float(model.S[n]), 4),
                          "L0": round(float(model.prior_L[n]), 3), "anchor": bool(model.is_anchor[n])}
    learned = UG.learn_visit(model, ccfg, args.scene_name, final, saved, model.sg_dir)["learned"]   # §9
    update.update(graph_wait_s=round(time.perf_counter() - t_up, 2), graph_versions=publisher.history,
                  after_last_keyframe_s=round(time.perf_counter() - t_fin, 2))
    publisher.close()
    for c, l in final.items():
        if l != "present":
            logger.info(f"[final] {c}: {l}")
    for o in saved["added"]:
        logger.info(f"[final] added {o['class_id']} ({o.get('novic_label')}) {o['voxels']} voxels")
    if worker is not None:
        update["crops"] = worker.stats()
        worker.close()

    P = engine.sol["P"]
    walls = np.array([r["wall_s"] for r in engine.timings]) * 1000
    lags = np.array([r["lag_s"] for r in engine.timings]) * 1000
    pmin = np.min(np.stack(engine.p_history), axis=0) if engine.p_history else P
    video_s = len(frames) / args.rate if args.rate > 0 else None
    summary = {
        "memory": args.memory, "video": str(video_dir), "keyframes": len(frames), "rate_hz": args.rate,
        "stream_wall_s": round(stream_s, 2), "finish_s": round(finish_s, 2), "video_s": video_s,
        "per_keyframe_ms": {"mean": round(float(walls.mean()), 2), "median": round(float(np.median(walls)), 2),
                            "p95": round(float(np.percentile(walls, 95)), 2), "max": round(float(walls.max()), 2)},
        "lag_ms": {"mean": round(float(lags.mean()), 2), "max": round(float(lags.max()), 2)},
        "evidence_frames": engine.belief.F, "frames_skipped_poor_fit": engine.frames_skipped,
        "localizer": {"keyframes": engine.loc.n_keyframes, "timing_s": {k: round(v, 2) for k, v in engine.loc.timing.items()},
                      "alignment_version": engine.alignment().version if engine.alignment() else None,
                      "demoted": [model.names[i] for i in engine.loc.demoted if 0 <= i < model.n_obj]},
        "first_evidence_at_keyframe": engine.frames_used[0] if engine.frames_used else None,
        "evidence_frame_split_ms": {
            which: {k: {"median": round(float(np.median([s[k] for s in sel])), 2),
                        "p95": round(float(np.percentile([s[k] for s in sel], 95)), 2)}
                    for k in ("fit_ms", "evidence_ms", "belief_ms", "coverage_ms", "commit_ms")}
            for which, sel in (("localizer_busy", [s for s in engine.split[10:] if s["localizer_busy"]] or engine.split[-1:]),
                               ("localizer_idle", [s for s in engine.split[10:] if not s["localizer_busy"]] or engine.split[-1:]))},
        "evidence_frames_localizer_busy_share": round(float(np.mean([s["localizer_busy"] for s in engine.split[10:]])), 3)
        if len(engine.split) > 10 else None,
        "objects": {c: {"label": engine.labels[n], "P": round(float(P[n]), 4), "P_min_over_time": round(float(pmin[n]), 4),
                        "coverage": round(float(engine.coverage[n]), 3),
                        "committed": engine.committed.get(c, {}).get("label"),
                        "committed_at_frame": engine.committed.get(c, {}).get("frame")}
                    for n, c in enumerate(model.names)},
        "final": {"labels": final, "removed": [c for c, l in final.items() if l == "removed"],
                  "moved": {c: assoc["rows"][u]["centroid"] for c, u in (assoc["moved"] if assoc else {}).items()},
                  "added": [o["class_id"] for o in saved["added"]], "deferred": [c for c, l in final.items() if l == "deferred"],
                  "voxels_before": int(len(model.idx)), "voxels_after": saved["voxels_after"], "saved": saved["saved"]},
        "update": update, "startup_s": round(startup_s, 1),
        "observation": (manager.log if manager is not None else None),
        "observed_objects": ([{"uid": o.uid, "name": o.raw.get("class_name"), "detections": o.n_det,
                               "centroid": np.round(o.pts.mean(0), 3).tolist()} for o in manager.observed] if manager else None),
        "association": ({"rows": assoc["rows"], "pairs": assoc["pair_scores"], "redetected": assoc["redetected_in_place"]}
                        if assoc else None),
        "observed_objects_final": ([{"uid": o.uid, "name": o.raw.get("class_name"), "detections": o.n_det, "voxels": len(o.grid_idx)}
                                    for o in assoc["observed"]] if assoc else None),
        "dynamism": {"first_visit": not model.dyn_existed, "t_memory_days": model.t_mem, "t_now_days": model.t_now,
                     "prior": prior_table, "learned": learned},
        "time": {"startup_s": round(startup_s, 1), "stream_s": round(stream_s, 2), "after_last_keyframe_s": update["after_last_keyframe_s"],
                 "update_s": round(stream_s + update["after_last_keyframe_s"], 2),
                 "engine_busy_s": round(float(sum(r["wall_s"] for r in engine.timings)), 2)},
        "settings": ccfg,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "events.json").write_text(json.dumps(engine.events, indent=2))
    with open(out / "timeline.csv", "w") as f:
        f.write("idx,wall_ms,lag_ms,pending,aligned\n")
        for r in engine.timings:
            f.write(f"{r['idx']},{r['wall_s'] * 1000:.2f},{r['lag_s'] * 1000:.2f},{r['pending']},{int(r['aligned'])}\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("settings", "objects", "dynamism", "association", "observation", "observed_objects", "observed_objects_final")}, indent=1, default=str))
    dy = summary["dynamism"]
    print(f"\nDynamism (move rate lambda ~ Gamma(a, b); survival S(dt) = (b / (b + dt))^a), memory at t = {dy['t_memory_days']} d, now t = {dy['t_now_days']} d"
          f"{' - first visit, rates from the §3.4 table' if dy['first_visit'] else ' - rates learned from earlier visits'}")
    print(f"  {'object':20s} {'a':>5s} {'b':>7s} {'rate/d':>7s} {'dt':>4s} {'S(dt)':>6s} {'L0':>6s}  {'P after':>7s}  {'label':9s} {'rate after':>10s}")
    for c, pr in dy["prior"].items():
        o = summary["objects"][c]
        print(f"  {c:20s} {pr['a']:5.2f} {pr['b']:7.2f} {pr['rate_per_day']:7.3f} {pr['dt_days']:4.1f} {pr['S']:6.3f} {pr['L0']:6.2f}  {o['P']:7.4f}  "
              f"{summary['final']['labels'][c]:9s} {dy['learned'][c]['rate_after']:10.4f}{'  (anchor)' if pr['anchor'] else ''}")
    for c, v in summary["objects"].items():
        print(f"  {c:20s} {v['label']:9s} P {v['P']:.4f} (min over time {v['P_min_over_time']:.4f}) coverage {v['coverage']:.2f} "
              f"committed {v['committed']}@{v['committed_at_frame']}")


if __name__ == "__main__":
    main()
