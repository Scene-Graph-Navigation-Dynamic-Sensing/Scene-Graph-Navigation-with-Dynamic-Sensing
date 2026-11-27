#!/usr/bin/env python3
"""Co-visibility gated change detection (covis-change-detection-theory.pdf).

Updates a scene graph made by this project's pipeline with a new video of the
same scene. A third method next to dynamic_updates/dovsg_method and
dynamic_updates/projection_method; it reuses their loaders, LightGlue wrapper
and graph rebuild, and changes none of their code.

  0. prior          §3  each object's move rate lambda ~ Gamma(a, b) -> S(dt) (Lomax) -> prior log-odds
  1. relocalise     §5  no ACE: LightGlue matches kept only on anchors (low move rate, or background),
                        RANSAC + Umeyama; anchors that disagree with the pose (phi < tau) are demoted
  2. refine         optional colored ICP against the anchors' voxels only
  3. evidence       §6  memory voxels projected into every new frame; signed residual: hit / through / occluded
                    §4  per-object log-odds, voxel correlation discount alpha (Kish)
                    §7  absence evidence gated by rho = noisy-OR over the object's graph neighbours
  4. decide         §8  Chow's rule with a defer option and the coverage condition
  5. new objects        the new video through the memory's own pipeline (as projection_method step 5)
  6. associate      §8  absent objects vs unexplained new objects (Hungarian): moved / removed; added
  7. learn          §9  stayed -> conjugate; moved / removed -> moment-matched; deferred -> nothing

Run in the scene_graph env (GPU for LightGlue and step 5):

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python dynamic_updates/covis_gate/run_covis_update.py \\
        --config configs/pipeline_lap_videos.yaml --memory lap_in_1 \\
        --dataset_dir data/processed/lap_remove_2 --scene_name lap_remove_2_covis

Outputs:
  outputs/covis_gate/<scene_name>/                 every step: pictures, numbers, report.md, evidence.mp4, raw.npz
  outputs/scene_graphs/<scene_name>.pkl/.json      the updated scene graph (same format as the memory)
  outputs/scene_graphs/<scene_name>_dynamism.json  the learned move rates, read by the next visit
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
for p in (str(REPO_ROOT), str(HERE.parent / "dovsg_method"), str(HERE.parent / "projection_method"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import covis_viz  # noqa: E402
import decide as D  # noqa: E402
import evidence as E  # noqa: E402
import theory as T  # noqa: E402
from dynamism import Dynamism  # noqa: E402
import targeted  # noqa: E402
from relocalize import BACKGROUND, NO_OWNER, relocalize  # noqa: E402

logger = logging.getLogger("covis_gate")

DEFAULT_CFG = {
    "dovsg_root": str(REPO_ROOT / "external" / "DovSG"),
    "load_workers": 8,
    "fast_load": True,              # no per-frame statistical outlier removal (edge-free mask only); False: as projection_method
    # §3 dynamism
    "rate_prior": "table",          # "table": §3.4 warm start (dynamism.RATE_TABLE); "uniform": rate_default for all
    "rate_default": 1.0,            # moves per day
    "kappa": 2.0,                   # pseudo-moves: strength of the initial guess
    "delta_t_days": 1.0,            # time since the memory, when not given on the command line
    "prior": "on",                  # "off": S = 0.5 for every object (ablation 3)
    "forgetting": 1.0,              # §9.3 gamma
    "rate_update": "moment",        # "naive": (a + 1, b + dt) (ablation 4)
    # §5 relocalisation
    "anchor_rate": 0.05,            # lambda_A: an object is an anchor when a / b is below this
    "background_anchor": True,      # voxels in no object (walls, floor) are anchors too
    "lightglue_max_keypoints": 1024,
    "retrieval_new_frames": 8,      # one T* for the whole video: a few well-spread frames find it
    "retrieval_memory_frames": 30,
    "retrieval_top_k": 3,
    "refine_candidates": 3,
    "candidate_angle_weight_m": 0.5,
    "pose_frame_stride": 4,         # refinement matches every 4th new frame
    "min_pair_matches": 30,
    "owner_distance_m": 0.03,       # a memory keypoint belongs to the object whose voxel is this close
    "ransac_thresh_m": 0.05,        # epsilon of eq. 10
    "rigid": False,                 # True: SE(3) as in the PDF; False: Sim3 (monocular depth has its own scale)
    "tau_phi": 0.3,                 # §5.5: anchors with fewer inliers than this moved
    "phi_min_matches": 30,          # ... judged only with at least this many matches
    "min_inliers": 200,             # fewer: the whole visit is deferred (§5.5 case i)
    "min_inlier_ratio": 0.15,
    "anchor_icp": True,             # colored ICP against the anchors' voxels after RANSAC
    "icp_frame_stride": 4,          # ... with every 4th frame's points
    "frame_fit_distance_m": 0.03,
    "min_frame_fit": 0.3,           # frames that fit the map worse give no evidence
    # §6 residual
    "residual": "signed",           # "unsigned": DoVSG's |dz| test (ablation 6)
    "sigma0_m": 0.005,              # sigma(z) = sigma0 + k1 z + k2 z^2; 2 sigma ~ 15% of the distance at 1 m
    "sigma_k1": 0.06,
    "sigma_k2": 0.01,
    "max_depth_m": 6.0,
    "colour_refine": True,          # a hit with a different colour is a through (§6.3 guards)
    "colour_thresh": 0.45,          # RGB distance after the exposure gain; lap_in_1 -> lap_remove_2: removed laptop median 0.66, present objects <= 0.35
    "evidence_frame_stride": 2,     # every 2nd frame gives evidence (neighbouring frames are near duplicates)
    "reliability_mask": True,       # judge only voxels the memory's own frames confirm (projection_method's self-test)
    # §4 belief
    "p_hit_present": 0.85,
    "p_hit_absent": 0.08,
    "rho_c": 0.2,                   # intra-object voxel correlation (eq. 6)
    "alpha_fixed": 0.0,             # > 0: a constant alpha instead of Kish's
    "rho_f": 0.1,                   # correlation between frames of one video (0 = independent frames)
    "l_max": 8.0,
    # §7 gate
    "gate": True,                   # False: rho = 1 (ablation 1)
    "update": "tempered",           # "mixture": eq. 18 (ablation 2)
    "next_to_weight": 0.6,
    "next_to_max_gap_m": 0.3,
    "background_weight": 0.7,       # 0: no local-background certifier
    "background_radius_m": 0.15,
    "background_min_voxels": 50,
    "max_sweeps": 10,
    # §8 decision
    "cost_false_removal": 1.0,
    "cost_missed_change": 1.0,
    "cost_defer": 0.2,
    "v_min": 0.3,
    "match_threshold": 0.75,
    "min_geometric": 0.2,
    "match_distance_m": 0.05,
    "novelty_distance_m": 0.05,
    "min_novelty": 0.4,
    "shape_distance_m": 0.05,
    "delta": 0.75,
    "beta": 1.0,
    "redetection_overrides_absent": True,
    # step 5: where the detector runs (targeted.py)
    "detect": "needed",             # "needed": windows around absent / deferred objects and new geometry;
                                    # "all": every frame (as before); "off": no detection
    "warm_models": True,            # load the detector / NOVIC in a worker process from the start (warm_detect.py)
    "detect_window": 6,             # consecutive frames per window (the tracker needs consecutive frames;
                                    # fusion keeps objects detected >= 3 times)
    "detect_max_windows": 4,
    "novel_distance_m": 0.05,       # new geometry: farther than this from every memory voxel ...
    "novel_voxel_m": 0.02,
    "novel_min_frames": 3,          # ... seen in at least this many frames ...
    "novel_min_voxels": 60,         # ... in a group at least this big
    "novel_points_per_frame": 4000,
    "novel_frame_stride": 4,
    "map_update": "changes",        # "changes": only removed / moved / added objects' voxels change;
                                    # "all": every new frame voxelised into the map (as before)
    # pictures
    "viz": "light",                 # "light": plots and a few frames; "full": also the evidence video of every frame; "off"
    "viz_frames": 3,
    "video_fps": 8,
}


def owner_names_of(objects: list) -> dict:
    return {**{n: o["class_id"] for n, o in enumerate(objects)}, BACKGROUND: "background", NO_OWNER: "not on the map"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/pipeline_lap_videos.yaml")
    ap.add_argument("--memory", required=True, help="outputs/scene_graphs/<memory>.pkl + data/processed/<memory>")
    ap.add_argument("--dataset_dir", required=True, help="the new video, run_static_scene_graph.py --pose_only")
    ap.add_argument("--scene_name", required=True)
    ap.add_argument("--delta_t_days", type=float, help="days between the memory and this visit")
    ap.add_argument("--reuse_observation", action="store_true",
                    help="skip step 5 when outputs/scene_graphs/<scene_name>_observation.pkl exists")
    ap.add_argument("--skip_observation", action="store_true",
                    help="no step 5/6: absent objects are removed, nothing is added or moved")
    ap.add_argument("--force_update", action="store_true", help="continue when relocalisation fails")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override a setting, e.g. --set gate=false update=mixture prior=off rate_update=naive")
    args = ap.parse_args()

    from scripts.run_static_scene_graph import load_config

    cfg = load_config(REPO_ROOT / args.config)
    ccfg = {**DEFAULT_CFG, **(cfg.get("covis_update") or {})}
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k not in ccfg:
            sys.exit(f"unknown setting {k}")
        ccfg[k] = (v.lower() in ("1", "true", "yes", "on")) if isinstance(ccfg[k], bool) else type(ccfg[k])(v)
    if args.delta_t_days is not None:
        ccfg["delta_t_days"] = args.delta_t_days
    sgg_root = cfg["external"]["scene_graph_generation_root"]
    out = REPO_ROOT / "outputs" / "covis_gate" / args.scene_name
    keep_obs = args.reuse_observation and (out / "5_observation").exists()
    if out.exists():
        for child in out.iterdir():
            if not (keep_obs and child.name == "5_observation"):
                shutil.rmtree(child) if child.is_dir() else child.unlink()
    dirs = {k: out / v for k, v in (("prior", "0_prior"), ("reloc", "1_relocalization"), ("ev", "3_evidence"),
                                     ("dec", "4_decision"), ("obs", "5_observation"), ("upd", "6_update"))}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])
    from object_fusion._dovsg_import import ensure_dovsg_importable

    ensure_dovsg_importable(sgg_root)
    from scipy.spatial import cKDTree

    import run_projection_update as PR
    import viz as dovsg_viz
    import warm_detect
    from object_fusion.streaming import prepare_native_libraries

    prepare_native_libraries()  # before any thread starts (see its docstring)
    warm = None
    if (not args.skip_observation and ccfg["detect"] != "off" and ccfg["warm_models"] and not keep_obs
            and warm_detect.supported(cfg)):
        warm = warm_detect.WarmWorker(warm_detect.run_config(cfg, dirs["obs"] / "visualizations"),
                                      dirs["obs"] / "pipeline.log")
    from navigation.voxel_map import PersistedVoxelMap
    from scene_graph.storage.serializer import (load_scene_graph, save_scene_graph, scene_graph_to_next_to,
                                                scene_graph_to_relations)
    from scene_graph.storage.visualizer import save_scene_graph_visualization

    t_start, timings = time.time(), {}
    summary = {"memory": args.memory, "new_video": args.dataset_dir, "scene_name": args.scene_name,
               "settings": ccfg, "steps": {}}

    def write_outputs():
        timings["total_s"] = round(time.time() - t_start, 1)
        summary["timings_s"] = timings
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        (out / "report.md").write_text(report(summary))

    # ── memory ──
    t = time.time()
    sg_dir = REPO_ROOT / cfg["outputs"]["scene_graphs_dir"]
    memory_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / args.memory
    memory = load_scene_graph(sg_dir / f"{args.memory}.pkl")
    memory_map = PersistedVoxelMap.from_dict(memory["voxel_map"])
    objects = memory["instance_objects"]
    names = [o["class_id"] for o in objects]
    n_obj = len(objects)
    res = memory_map.resolution
    mem_idx = np.fromiter(memory_map.indexes_colors_mapping_dict.keys(), dtype=np.int64)
    mem_pts = memory_map.index_to_point(mem_idx) + res / 2
    mem_cols = np.array([memory_map.indexes_colors_mapping_dict[i] for i in mem_idx.tolist()], dtype=np.float64)
    row_of = dict(zip(mem_idx.tolist(), range(len(mem_idx))))
    owner = np.full(len(mem_idx), BACKGROUND, np.int64)
    for n, o in enumerate(objects):
        owner[[row_of[i] for i in o["indexes"] if i in row_of]] = n
    relations = scene_graph_to_relations(memory["scene_graph"])
    next_to = scene_graph_to_next_to(memory["scene_graph"])
    root_id = memory["scene_graph"].root.node_id
    parent_of = {r["child"]: r["parent"] for r in relations}
    n_receptacles = len({r["parent"] for r in relations if r["relation"] in ("on", "inside")} | {root_id})
    centroids = {o["class_id"]: mem_pts[owner == n].mean(0) for n, o in enumerate(objects)}
    memory_frames = PR.MemoryFrames(memory_dir)
    timings["load_memory_s"] = round(time.time() - t, 1)

    # ── 0. prior (§3) ──
    dyn_path = sg_dir / f"{args.memory}_dynamism.json"
    dyn, existed = Dynamism.load_or_init(dyn_path, objects, 0.0, ccfg, parent_of)
    t_mem = dyn.state["visit_time_days"]
    t_now = t_mem + ccfg["delta_t_days"]
    dt = {c: max(t_now - dyn.rec(c)["t_last"], 0.0) for c in names}
    S = np.array([dyn.survival(c, t_now) for c in names]) if ccfg["prior"] == "on" else np.full(n_obj, 0.5)
    prior_L = np.array([T.logit(s) for s in S])
    anchor_ids = set(dyn.anchors(names, ccfg["anchor_rate"]))
    is_anchor = np.array([c in anchor_ids for c in names])
    anchor_codes = {n for n in range(n_obj) if is_anchor[n]} | ({BACKGROUND} if ccfg["background_anchor"] else set())
    prior_rows = {c: {"class": dyn.rec(c)["class_name"], "a": round(dyn.rec(c)["a"], 3), "b": round(dyn.rec(c)["b"], 3),
                      "rate_per_day": round(dyn.mean_rate(c), 4), "dt_days": round(dt[c], 3), "S": round(float(S[n]), 4),
                      "L0": round(float(prior_L[n]), 3), "anchor": bool(is_anchor[n])} for n, c in enumerate(names)}
    summary["steps"]["0_prior"] = {"dynamism_file": str(dyn_path) if existed else "none: first visit, §3.4 initialisation",
                                   "t_memory_days": t_mem, "t_now_days": t_now, "objects": prior_rows}
    covis_viz.survival_curves(dirs["prior"] / "survival.png", {c: dyn.rec(c) for c in names}, dt)
    logger.info(f"[0 prior] dt {ccfg['delta_t_days']} d; anchors: {sorted(anchor_ids)}"
                f"{' + background' if ccfg['background_anchor'] else ''}")

    # ── 1. relocalise (§5) ──
    t = time.time()
    new_dir = Path(args.dataset_dir)
    new_dir = new_dir if new_dir.is_absolute() else REPO_ROOT / new_dir
    observations = (targeted.load_frames(new_dir, ccfg["load_workers"]) if ccfg["fast_load"]
                    else PR.load_new_frames(new_dir, sgg_root, ccfg["load_workers"]))
    keys = list(observations)
    viz_keys = sorted({keys[i] for i in np.linspace(0, len(keys) - 1, min(ccfg["viz_frames"], len(keys))).round().astype(int)})
    timings["load_new_s"] = round(time.time() - t, 1)
    mem_tree = cKDTree(mem_pts)
    reloc = relocalize(observations, memory_frames, mem_tree, owner, anchor_codes, ccfg, dirs["reloc"],
                       viz_keys if ccfg["viz"] != "off" else [], owner_names_of(objects))
    timings["relocalization_s"] = reloc.pop("time_s", None)
    import torch

    torch.cuda.empty_cache()
    Tm = reloc.pop("T")
    summary["steps"]["1_relocalization"] = reloc
    if "phi" in reloc:
        covis_viz.phi_bars(dirs["reloc"] / "1_phi_per_anchor.png", reloc["phi"], ccfg["tau_phi"])
    if Tm is None or ("failure" in reloc and not args.force_update):
        summary["visit"] = f"DEFERRED: relocalisation failed ({reloc.get('failure')}); nothing changed"
        logger.error(summary["visit"] + "; --force_update continues anyway")
        write_outputs()
        return
    demoted = set(reloc["demoted"])
    reloc["demoted"] = [names[n] for n in sorted(demoted)]
    for n in demoted:
        is_anchor[n] = False
    logger.info(f"[1 reloc] {reloc['inliers']}/{reloc['anchor_matches']} anchor matches inliers; scale {reloc['scale']}, "
                f"{reloc['rotation_deg']} deg, {reloc['translation_m']} m; demoted {[names[n] for n in demoted] or 'none'}")

    # ── 2. anchor ICP + frame fit ──
    t = time.time()
    anchor_rows = np.isin(owner, list(anchor_codes - demoted))
    icp = {"used": False}
    if ccfg["anchor_icp"]:
        T_icp, icp, source = PR.colored_icp({k: observations[k] for k in keys[::ccfg["icp_frame_stride"]]},
                                            mem_pts[anchor_rows], mem_cols[anchor_rows])
        for o in observations.values():
            o["pose"] = T_icp @ o["pose"]
        icp["correction_deg"] = round(np.degrees(np.arccos(np.clip((np.trace(T_icp[:3, :3]) - 1) / 2, -1, 1))), 2)
        icp["correction_m"] = round(float(np.linalg.norm(T_icp[:3, 3])), 3)
    ev_keys = keys[::ccfg["evidence_frame_stride"]]
    fit = PR.frame_fit({k: observations[k] for k in ev_keys}, mem_tree, ccfg["frame_fit_distance_m"])
    good = [k for k in ev_keys if fit[k] >= ccfg["min_frame_fit"]]
    icp.update({"frames_fit_median": round(float(np.median(list(fit.values()))), 3), "frames_used": len(good),
                "frames_judged": len(ev_keys), "frames_left_out": [observations[k]["frame"] for k in ev_keys if k not in good]})
    summary["steps"]["2_refine"] = icp
    timings["refine_s"] = round(time.time() - t, 1)
    for k in (viz_keys if ccfg["viz"] != "off" else []):
        o = observations[k]
        h, w = o["valid"].shape
        renders = [dovsg_viz.render_voxels(mem_pts, mem_cols, o[p], o["intrinsic"], h, w) for p in ("sim3_pose", "pose")]
        dovsg_viz.pose_check(dirs["reloc"] / f"2_pose_check_{o['frame']}.png", o["rgb"], renders,
                             ["memory from the anchor RANSAC pose", "after anchor ICP"], o["frame"])
    np.save(dirs["reloc"] / "final_poses.npy", np.stack([observations[k]["pose"] for k in keys]))
    logger.info(f"[2 refine] ICP used={icp['used']}; {len(good)}/{len(ev_keys)} evidence frames fit >= {ccfg['min_frame_fit']}")

    # ── 3. evidence + gate (§4, §6, §7) ──
    t = time.time()
    reliable = np.ones(len(mem_idx), bool)
    if ccfg["reliability_mask"]:
        pcfg = {**PR.DEFAULT_CFG, **(cfg.get("projection_update") or {})}
        reliable = PR.memory_stable_voxels(memory_dir, mem_pts, mem_cols, pcfg, sgg_root)
    owner_eff = np.where(reliable, owner, -3)  # -3: not judged
    frames = [observations[k] for k in good]
    nbrs = E.neighbours(objects, relations, next_to, ccfg)
    bg_groups = E.background_groups(mem_pts, owner, reliable, [n for n in range(n_obj) if not is_anchor[n]], ccfg)
    counts = E.gather(frames, mem_pts, mem_cols, owner_eff, n_obj, bg_groups, ccfg)
    bel = E.beliefs(counts, prior_L, is_anchor, nbrs, ccfg)
    coverage = E.gated_coverage(counts, bel["rho"], owner_eff, reliable, n_obj)
    timings["evidence_s"] = round(time.time() - t, 1)

    # ── 4. decide (§8.1) ──
    labels, thresholds = D.chow_labels(bel["P"], coverage, ccfg)
    judged = np.bincount(owner[(owner >= 0) & reliable], minlength=n_obj)
    per_obj = {}
    for n, c in enumerate(names):
        tot = counts["voxel_totals"][owner_eff == n].sum(0) if (owner_eff == n).any() else np.zeros(3, int)
        per_obj[c] = {
            "anchor": bool(is_anchor[n]), "demoted": n in demoted, "voxels": int((owner == n).sum()),
            "judged_voxels": int(judged[n]), "frames_with_evidence": int(bel["frames_informative"][n]),
            "hits": int(counts["n_hit"][:, n].sum()), "throughs": int(counts["n_thr"][:, n].sum()),
            "occluded": int(tot[2]), "S": round(float(S[n]), 4), "L0": round(float(prior_L[n]), 3),
            "evidence_for": round(float(bel["evidence_pos"][n]), 2),
            "evidence_against_ungated": round(float(bel["evidence_neg_raw"][n]), 2),
            "rho_mean_when_against": round(float(bel["rho"][counts["n_thr"][:, n] > 0, n].mean()), 3)
            if (counts["n_thr"][:, n] > 0).any() else None,
            "neighbours": [f"{why} (w {w:.2f})" for _, w, why in nbrs[n]]
            + ([f"background within {ccfg['background_radius_m']} m (w {ccfg['background_weight']})"] if n in bg_groups else []),
            "L": round(float(bel["L"][n]), 3), "P": round(float(bel["P"][n]), 4),
            "coverage": round(float(coverage[n]), 3), "label": labels[n]}
    summary["steps"]["3_evidence"] = {"frames": len(frames), "voxel_llr": round(bel["ell"], 3), "sweeps": bel["sweeps"],
                                      "voxels_judged": int(reliable.sum()), "voxels": int(len(reliable))}
    summary["steps"]["4_decision"] = {"thresholds": {"absent_below": thresholds[0], "present_above": thresholds[1],
                                                     "v_min": ccfg["v_min"]}, "objects": per_obj}
    logger.info("[4 decide] " + ", ".join(f"{c} {labels[n]} {bel['P'][n]:.2f}" for n, c in enumerate(names)))
    t = time.time()
    if ccfg["viz"] != "off":
        covis_viz.belief_trajectories(dirs["ev"] / "beliefs.png", names, bel["trajectory"], prior_L, labels, thresholds,
                                      [o["frame"] for o in frames])
        covis_viz.gate_over_frames(dirs["ev"] / "gate_rho.png", names, bel["rho"], is_anchor)
        covis_viz.decision_bars(dirs["dec"] / "decision.png", names, bel["P"], coverage, labels, thresholds, ccfg["v_min"])
        covis_viz.outcome_top_down(dirs["ev"] / "outcomes_top_down.png", mem_pts, mem_cols, counts["voxel_totals"],
                                   centroids, dict(zip(names, labels)), dict(zip(names, bel["P"])),
                                   "memory from above: each voxel's most frequent outcome this visit")
    np.savez_compressed(out / "raw.npz", names=np.array(names), frames=np.array([o["frame"] for o in frames]),
                        n_valid=counts["n_valid"], n_hit=counts["n_hit"], n_thr=counts["n_thr"], rho=bel["rho"],
                        trajectory=bel["trajectory"], prior_L=prior_L, P=bel["P"], coverage=coverage,
                        is_anchor=is_anchor, voxel_index=mem_idx, voxel_owner=owner, voxel_reliable=reliable,
                        voxel_totals_hit_through_occluded=counts["voxel_totals"],
                        through_events=counts["through_events"])
    if ccfg["viz"] == "full":
        evidence_video(dirs["ev"] / "evidence.mp4", observations, keys, good, mem_pts, mem_cols, owner_eff, names,
                       bel, labels, ccfg, viz_keys, dirs["ev"], dovsg_viz)
    elif ccfg["viz"] == "light":
        evidence_video(None, observations, viz_keys, good, mem_pts, mem_cols, owner_eff, names,
                       bel, labels, ccfg, viz_keys, dirs["ev"], dovsg_viz)
    timings["pictures_s"] = round(time.time() - t, 1)

    # ── 5. the new video's objects ──
    t = time.time()
    obs_objects, obs_pts, obs_parent_obs, obs_map = [], [], [], None
    detect = "off" if args.skip_observation else ccfg["detect"]
    det_keys = keys
    if detect == "needed":
        pos = {k: f for f, k in enumerate(good)}
        centres, why = [], []
        for n in range(n_obj):  # absent first, then deferred: the frame that projects most of the object
            if labels[n] == "absent" and counts["n_valid"][:, n].any():
                centres.append(good[int(np.argmax(counts["n_valid"][:, n]))]); why.append(f"absent {names[n]}")
        for n in range(n_obj):
            if labels[n] == "deferred" and counts["n_valid"][:, n].any():
                centres.append(good[int(np.argmax(counts["n_valid"][:, n]))]); why.append(f"deferred {names[n]}")
        nov_keys = good[::max(1, ccfg["novel_frame_stride"] // ccfg["evidence_frame_stride"])]
        novel = targeted.novel_regions([observations[k] for k in nov_keys], mem_tree, memory_map.bounds, ccfg)
        for r in novel["regions"]:
            r["best_frame"] = observations[nov_keys[r["best_frame"]]]["frame"]
            centres.append(int(next(k for k in nov_keys if observations[k]["frame"] == r["best_frame"])))
            why.append(f"new geometry {r['voxels']} voxels at {r['centroid']}")
        det_keys = targeted.detection_windows(len(keys), centres, ccfg)
        summary["steps"]["5_where_detected"] = {
            "reasons": [{"centre_frame": observations[c]["frame"], "why": w} for c, w in zip(centres, why)],
            "windows_used": min(len(centres), ccfg["detect_max_windows"]), "new_geometry": novel["regions"],
            "frames_detected": [observations[k]["frame"] for k in det_keys], "of_frames": len(keys)}
        logger.info(f"[5 where] detect {len(det_keys)}/{len(keys)} frames: " + "; ".join(why[:ccfg['detect_max_windows']])
                    if det_keys else "[5 where] nothing absent, deferred or new: no detection")
        if not det_keys:
            detect = "off"
    if detect != "off":
        obs_name = f"{args.scene_name}_observation"
        obs_dir = REPO_ROOT / cfg["outputs"]["data_processed_dir"] / obs_name
        obs_pkl = sg_dir / f"{obs_name}.pkl"
        if not (keep_obs and obs_pkl.exists()):
            if detect == "needed":
                targeted.prepare_subset_dataset(new_dir, obs_dir, observations, det_keys, reloc["scale"])
            else:
                PR.prepare_observation_dataset(new_dir, obs_dir, observations, reloc["scale"])
            if warm is not None:
                summary["steps"]["5_pipeline_warm_worker"] = warm.run(obs_dir, obs_name, sg_dir)
            else:
                PR.run_memory_pipeline(cfg, dirs["obs"] / "config.yaml", obs_dir, obs_name, dirs["obs"] / "visualizations")
                log = REPO_ROOT / cfg["outputs"]["logs_dir"] / f"{obs_name}.log"
                if log.exists():
                    shutil.copy2(log, dirs["obs"] / "pipeline.log")
        observed = load_scene_graph(obs_pkl)
        obs_map = PersistedVoxelMap.from_dict(observed["voxel_map"])
        obs_objects = observed["instance_objects"]
        obs_pts = [obs_map.index_to_point(np.asarray(o["indexes"])) + obs_map.resolution / 2 for o in obs_objects]
        obs_par = {r["child"]: r["parent"] for r in scene_graph_to_relations(observed["scene_graph"])
                   if r["relation"] in ("on", "inside")}
        obs_root = observed["scene_graph"].root.node_id
        obs_parent_obs = [root_id if obs_par.get(o["class_id"]) == obs_root else obs_par.get(o["class_id"])
                          for o in obs_objects]
        summary["steps"]["5_observation"] = {"scene_graph": str(obs_pkl), "objects": [o["class_id"] for o in obs_objects]}
    timings["observation_s"] = round(time.time() - t, 1)
    if warm is not None:
        warm.close()

    # ── 6. associate (§8.2) ──
    t = time.time()
    absent_rows = np.isin(owner, [n for n in range(n_obj) if labels[n] == "absent"])
    remaining_tree = cKDTree(mem_pts[~absent_rows])
    mem_obj_pts = [mem_pts[owner == n] for n in range(n_obj)]
    if obs_objects:
        assoc = D.associate(objects, mem_obj_pts, list(labels), obs_objects, obs_pts, obs_parent_obs, root_id,
                            [dyn.rec(c)["receptacles"] for c in names], n_receptacles, remaining_tree, ccfg)
    else:
        assoc = {"rows": [], "moved": {}, "redetected_in_place": [], "pair_scores": [],
                 "labels": {c: ("removed" if labels[n] == "absent" else labels[n]) for n, c in enumerate(names)}}
    final = assoc["labels"]
    moved = {c: assoc["rows"][u] for c, u in assoc["moved"].items()}
    removed = [c for c, l in final.items() if l == "removed"]

    from run_dovsg_update import UpdateViewDataset

    from dovsg_core import update_voxel_colors
    from scripts.update_scene_graph import renumber, update_scene_graph

    new_objects, gone_idx, new_colours = [], [], {}
    for mo in objects:
        cid = mo["class_id"]
        if cid in removed:
            gone_idx += list(mo["indexes"])
            continue
        m = dict(mo)
        if cid in moved:
            gone_idx += list(mo["indexes"])
            m["indexes"], cols = targeted.object_in_memory_grid(memory_map, obs_map, obs_objects[moved[cid]["index"]])
            new_colours.update(cols)
            m["moved_from"] = np.round(centroids[cid], 3).tolist()
            m["moved_to"] = moved[cid]["centroid"]
        new_objects.append(m)
    added, added_rows = [], []
    for r in assoc["rows"]:
        if r["decision"] == "added":
            o = dict(obs_objects[r["index"]])
            o["indexes"], cols = targeted.object_in_memory_grid(memory_map, obs_map, obs_objects[r["index"]])
            new_colours.update(cols)
            if o["indexes"]:
                added.append(o)
                added_rows.append(r)
    renumber(added, set(names))
    for o, r in zip(added, added_rows):
        r["added_as"] = o["class_id"]
    new_objects += added
    if ccfg["map_update"] == "changes":
        vd = targeted.ChangedMap(memory_map, gone_idx, new_colours)
    else:
        vd = UpdateViewDataset(memory_map, {k: {**observations[k], "mask": observations[k]["valid"]} for k in keys})
        vd.indexes_colors_mapping_dict = update_voxel_colors(memory_map.indexes_colors_mapping_dict,
                                                             np.asarray(gone_idx, np.int64), vd.observed)
    if removed or moved or added:
        graph, orphans = update_scene_graph(memory["scene_graph"], removed + list(moved), new_objects, vd, cfg)
    else:  # nothing changed: the memory's graph as it is
        graph, orphans = copy.deepcopy(memory["scene_graph"]), []
    timings["update_s"] = round(time.time() - t, 1)

    # ── 7. learn (§9) ──
    new_parent = {r["child"]: r["parent"] for r in scene_graph_to_relations(graph) if r["relation"] in ("on", "inside")}
    learned = {}
    for c in names:
        learned[c] = dyn.learn(c, final[c], t_now, ccfg, parent=new_parent.get(c, parent_of.get(c)))
        if final[c] == "removed":
            dyn.archive(c)
    for o in added:
        dyn.add(o, t_now, ccfg, parent=new_parent.get(o["class_id"]))
    dyn.state["visits"].append({"t_days": t_now, "scene_name": args.scene_name, "memory": args.memory,
                                "labels": final, "added": [o["class_id"] for o in added]})
    dyn.state["visit_time_days"] = t_now
    dyn.save(sg_dir / f"{args.scene_name}_dynamism.json")
    shutil.copy2(sg_dir / f"{args.scene_name}_dynamism.json", dirs["upd"] / "dynamism.json")

    # ── save ──
    t = time.time()
    for o in new_objects:
        o.pop("mask", None)
    emb = {o["class_id"]: np.asarray(o["semantic_embedding"], np.float32) for o in new_objects if "semantic_embedding" in o}
    pkl_path, json_path = save_scene_graph(sg_dir, args.scene_name, graph, new_objects, emb, view_dataset=vd)
    timings["save_s"] = round(time.time() - t, 1)
    t = time.time()
    if ccfg["viz"] != "off":
        save_scene_graph_visualization(memory["scene_graph"], dirs["upd"] / "scene_graph_before")
        save_scene_graph_visualization(graph, dirs["upd"] / "scene_graph_after")
        now_idx = np.fromiter(vd.indexes_colors_mapping_dict.keys(), dtype=np.int64)
        dovsg_viz.changes_map(dirs["upd"] / "changes_map.png", memory_map.index_to_point(now_idx)[::2],
                              np.array([vd.indexes_colors_mapping_dict[i] for i in now_idx.tolist()])[::2],
                              {**{c: centroids[c] for c in removed}, **{f"{c} (moved from)": centroids[c] for c in moved}},
                              {**{o["class_id"]: memory_map.index_to_point(np.asarray(o["indexes"])).mean(0) for o in added},
                               **{f"{c} (moved to)": np.asarray(moved[c]["centroid"]) for c in moved}},
                              f"updated map: {len(removed)} removed, {len(moved)} moved, {len(added)} added, "
                              f"{sum(l == 'deferred' for l in final.values())} deferred")
    timings["pictures_s"] = round(timings.get("pictures_s", 0) + time.time() - t, 1)
    summary["visit"] = "updated"
    summary["steps"]["6_update"] = {
        "labels": final, "removed": removed,
        "moved": {c: {"from": np.round(centroids[c], 3).tolist(), "to": r["centroid"], "new_video_object": r["object"]}
                  for c, r in moved.items()},
        "added": [o["class_id"] for o in added], "deferred": [c for c, l in final.items() if l == "deferred"],
        "redetected_in_place": assoc["redetected_in_place"], "new_video_objects": assoc["rows"],
        "association_scores": assoc["pair_scores"], "re_placed_after_removal": orphans,
        "voxels_before": int(len(mem_idx)), "voxels_after": int(len(vd.indexes_colors_mapping_dict)),
        "saved": [str(pkl_path), str(json_path), str(sg_dir / f"{args.scene_name}_dynamism.json")]}
    summary["steps"]["7_learning"] = learned
    logger.info(f"[6 update] removed {removed or 'nothing'}; moved {list(moved) or 'nothing'}; added "
                f"{[o['class_id'] for o in added] or 'nothing'}; deferred {summary['steps']['6_update']['deferred'] or 'nothing'}")
    write_outputs()
    logger.info(f"Outputs: {out}")


def evidence_video(path, observations, keys, good, mem_pts, mem_cols, owner_eff, names, bel, labels, ccfg, viz_keys,
                   frame_dir, dovsg_viz):
    """Every new frame: object voxels' outcomes (hit green, through red, occluded
    blue) with each object's belief after that frame and its gate | the memory
    rendered from the same pose."""
    colour = {"present": (60, 230, 60), "deferred": (255, 180, 0), "absent": (255, 60, 60)}
    good_pos = {k: f for f, k in enumerate(good)}
    writer = None
    for k in keys:  # path None: only the frames in viz_keys, as PNGs
        o = observations[k]
        h, w = o["valid"].shape
        ii, r, c, outc = E.classify_frame(o, mem_pts, mem_cols, ccfg)
        own = owner_eff[ii]
        labels_pos = {}
        f = good_pos.get(k)
        for n in np.unique(own[own >= 0]):
            m = own == n
            if m.sum() < 30:
                continue
            P = T.sigmoid(bel["trajectory"][f, n]) if f is not None else T.sigmoid(bel["L"][n])
            rho = bel["rho"][f, n] if f is not None else float("nan")
            labels_pos[f"{names[n]} P{P:.2f} rho{rho:.2f}"] = (int(np.median(c[m])), int(np.median(r[m])),
                                                               colour.get(labels[n], (255, 255, 255)))
        left = covis_viz.frame_overlay(o["rgb"], r, c, outc, own >= 0, labels_pos)
        right = (np.clip(dovsg_viz.render_voxels(mem_pts, mem_cols, o["pose"], o["intrinsic"], h, w), 0, 1) * 255).astype(np.uint8)
        for img, text in ((left, f"new {o['frame']}{'' if f is not None else ' (not used: poor fit)'}  "
                                 "green hit / red through / blue occluded"), (right, "memory from this pose")):
            cv2.rectangle(img, (0, 0), (w, 20), (0, 0, 0), -1)
            cv2.putText(img, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.concatenate([left, right], axis=1)
        if path is not None and writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), ccfg["video_fps"],
                                     (panel.shape[1], panel.shape[0]))
        if writer is not None:
            writer.write(cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        if k in viz_keys:
            Image.fromarray(panel).save(frame_dir / f"frame_{o['frame']}.png")
    if writer is not None:
        writer.release()


def report(s: dict) -> str:
    st, tm = s["steps"], s.get("timings_s", {})
    L = [f"# Scene update (co-visibility gate): {s['scene_name']}", "",
         f"Memory `{s['memory']}`, new video `{s['new_video']}`. Method: `dynamic_updates/covis_gate` "
         "(covis-change-detection-theory.pdf).", ""]
    if s.get("visit", "").startswith("DEFERRED"):
        L += [f"**{s['visit']}**", ""]
    if "0_prior" in st:
        p = st["0_prior"]
        L += ["## 0. Dynamism prior (§3)", "", f"Rates from: {p['dynamism_file']}. dt = {p['t_now_days'] - p['t_memory_days']:g} days.", "",
              "| object | a | b | rate /day | dt (d) | S(dt) | L0 | anchor |", "|---|---:|---:|---:|---:|---:|---:|---|"]
        for c, v in p["objects"].items():
            L.append(f"| {c} | {v['a']} | {v['b']} | {v['rate_per_day']} | {v['dt_days']} | {v['S']} | {v['L0']} | "
                     f"{'yes' if v['anchor'] else ''} |")
        L += ["", "Picture: `0_prior/survival.png`.", ""]
    if "1_relocalization" in st:
        g = st["1_relocalization"]
        L += ["## 1-2. Relocalisation from anchors (§5, no ACE)", ""]
        if "inliers" in g:
            L += [f"Retrieval: {g['retrieval']['new_frames']} new x {g['retrieval']['memory_frames']} memory frames. "
                  f"Refinement: {g['pairs_matched']} frame pairs, {g['matches_3d']} 3D matches, {g['anchor_matches']} on anchors, "
                  f"{g['inliers']} inliers ({g['inlier_ratio']:.0%}); RANSAC ran {g['ransac_iterations']} iterations "
                  f"(eq. 12 asks {g['ransac_iterations_eq12']}).",
                  f"T*: scale {g['scale']}, {g['rotation_deg']}°, {g['translation_m']} m; median inlier error "
                  f"{g['median_error_inliers_mm']} mm.", "",
                  "| anchor | matches | phi | median error (m) |", "|---|---:|---:|---:|"]
            for c, v in g["phi"].items():
                L.append(f"| {c} | {v['matches']} | {v['phi']} | {v['median_error_m']} |")
            L += ["", f"Demoted (moved anchors): {', '.join(g['demoted']) or 'none'}", "",
                  "How every owner's matches agree with the anchors' pose (non-anchors were not used; a moved object shows a low share):", "",
                  "| owner | matches | inlier share | median error (m) |", "|---|---:|---:|---:|"]
            for c, v in g["agreement_all_owners"].items():
                L.append(f"| {c} | {v['matches']} | {v['phi']} | {v['median_error_m']} |")
        if g.get("failure"):
            L += ["", f"Failure: {g['failure']}"]
        L += ["", "Pictures: `1_relocalization/1_retrieval_matches.png`, `1_matches_*.png` (green: anchor matches, used), "
              "`1_phi_per_anchor.png`, `2_pose_check_*.png`.", ""]
    if "2_refine" in st:
        c = st["2_refine"]
        L += [f"Anchor ICP: {'used' if c.get('used') else 'not used'}"
              + (f" (correction {c.get('correction_deg')}° / {c.get('correction_m')} m)" if "correction_m" in c else "")
              + f". {c['frames_used']} frames fit the map (median fit {c['frames_fit_median']}).", ""]
    if "4_decision" in st:
        d = st["4_decision"]
        e = st["3_evidence"]
        th = d["thresholds"]
        L += ["## 3-4. Evidence, gate and decision (§4, §6, §7, §8.1)", "",
              f"{e['frames']} frames; {e['voxels_judged']} of {e['voxels']} memory voxels judged; per-voxel LLR "
              f"{e['voxel_llr']}; {e['sweeps']} gate sweep(s). Chow: absent below {th['absent_below']:.2f} (and coverage "
              f">= {th['v_min']}), present above {th['present_above']:.2f}.", "",
              "| object | anchor | judged vox | frames | hits | throughs | occluded | L0 | for | against (ungated) | mean rho | P | coverage | label |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for c, v in d["objects"].items():
            anc = "demoted" if v["demoted"] else ("yes" if v["anchor"] else "")
            L.append(f"| {c} | {anc} | {v['judged_voxels']} | {v['frames_with_evidence']} | {v['hits']} | {v['throughs']} | "
                     f"{v['occluded']} | {v['L0']} | {v['evidence_for']} | {v['evidence_against_ungated']} | "
                     f"{v['rho_mean_when_against'] if v['rho_mean_when_against'] is not None else '-'} | {v['P']} | "
                     f"{v['coverage']} | {v['label']} |")
        L += ["", "Gate neighbours:", ""] + [f"- {c}: {'; '.join(v['neighbours']) or 'none (rho = 1 for anchors, 0 otherwise)'}"
                                              for c, v in d["objects"].items() if not v["anchor"]]
        L += ["", "Pictures: `3_evidence/evidence.mp4` (every frame), `frame_*.png`, `beliefs.png`, `gate_rho.png`, "
              "`outcomes_top_down.png`, `4_decision/decision.png`. Raw counts: `raw.npz`.", ""]
    if "5_where_detected" in st:
        w = st["5_where_detected"]
        L += ["## 5. Where the detector ran", "",
              f"{len(w['frames_detected'])} of {w['of_frames']} frames, in windows around (first {w['windows_used']} used):", ""]
        L += [f"- frame {r['centre_frame']}: {r['why']}" for r in w["reasons"]] or ["- nothing: no detection"]
        L += [""]
    if "6_update" in st:
        u = st["6_update"]
        L += ["## 5-6. Association and update (§8.2, §8.3)", "",
              "| new video object | detections | decision | memory object | cosine | geometric | novelty | parent |",
              "|---|---:|---|---|---:|---:|---:|---|"]
        for r in u["new_video_objects"]:
            dec = r["decision"] + (f" → {r['added_as']}" if r.get("added_as") else "") + \
                (f" (inside {r['inside']})" if r.get("inside") else "")
            L.append(f"| {r['object']} | {r['detections']} | {dec} | {r.get('memory_id', r['best_memory'])} | "
                     f"{r['cosine']} | {r['geometric']} | {r['novelty']} | {r.get('parent')} |")
        if u["association_scores"]:
            L += ["", "Absent x unexplained pairs (moved when s >= delta):", "",
                  "| memory | new | cosine | shape | s | receptacle prior | new parent |", "|---|---|---:|---:|---:|---:|---|"]
            for p in u["association_scores"]:
                L.append(f"| {p['memory']} | {p['new']} | {p['cosine']} | {p['shape']} | {p['s']} | "
                         f"{p['receptacle_prior']} | {p['new_parent']} |")
        moved = ", ".join(f"{c} {v['from']} -> {v['to']}" for c, v in u["moved"].items())
        L += ["", f"- Removed: {', '.join(u['removed']) or 'nothing'}", f"- Moved: {moved or 'nothing'}",
              f"- Added: {', '.join(u['added']) or 'nothing'}", f"- Deferred: {', '.join(u['deferred']) or 'nothing'}",
              f"- Absent by belief but re-detected in place (kept): {', '.join(u['redetected_in_place']) or 'none'}",
              f"- Voxels: {u['voxels_before']} -> {u['voxels_after']}", "",
              "Pictures: `6_update/changes_map.png`, `scene_graph_before/`, `scene_graph_after/`.", ""]
    if "7_learning" in st:
        L += ["## 7. Rate learning (§9)", "", "| object | outcome | dt (d) | rate before | rate after | (a, b) after |",
              "|---|---|---:|---:|---:|---|"]
        for c, v in st["7_learning"].items():
            L.append(f"| {c} | {v['outcome']} | {v['dt_days']} | {v['rate_before']} | {v['rate_after']} | {v['a_b_after']} |")
        L += ["", "Saved for the next visit: `6_update/dynamism.json` (also next to the scene graph).", ""]
    L += ["## Settings", "", "```json", json.dumps(s["settings"], indent=1), "```", "",
          "## Time", "", "| step | s |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in tm.items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
