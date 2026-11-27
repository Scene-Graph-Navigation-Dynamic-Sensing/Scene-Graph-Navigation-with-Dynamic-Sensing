# DoVSG dynamic update method

Updates a scene graph made by this project's pipeline with a new video of the
same scene, using **DoVSG's update logic** (relocalization, change detection,
removal) and **this project's perception** (YOLOE, NOVIC, fusion, relations).
Every step saves pictures and numbers.

The flowchart is in `docs/pipeline_flow.html`.

## What comes from where

| Step | From | Code |
|---|---|---|
| 1. ACE: rough pose of each new frame in the memory map | DoVSG `ace/` | `ace_utils.py` |
| 2. LightGlue anchor: the frame with the most matches keeps its ACE pose, the others follow with `b2w @ c2b` | DoVSG `correct_pose_observations` | `dovsg_core.py` |
| (scale, only with `--estimate_scale`) | not DoVSG | `run_dovsg_update.py` |
| 3. Colored ICP: one rigid correction for all frames (4, 2, 1 cm). If it fails, the update stops | DoVSG `calculate_alignment_colored_icp` | `dovsg_core.py` |
| 4. See-through test: a memory voxel is deleted when the new depth is > 2 cm behind it, or > 1 cm behind with a colour change, in any one frame | DoVSG `find_need_to_delete_indexes` | `dovsg_core.py` |
| 5a. An object with > 50% of its voxels deleted is removed. Other objects lose their deleted voxels | DoVSG `update_instance_objects` | `dovsg_core.py` |
| 5b. YOLOE + NOVIC on the new frames, then fused into the surviving objects (score ≥ 0.75 joins, else a new object, named by NOVIC) | this project | `object_fusion/`, `semantic_decoding/` |
| 5c. Removed nodes go and what they carried is placed again. New objects are placed with `on` / `inside` / `belongs` | this project (DoVSG rules) | `scene_graph/builder/dovsg_relations.py` |

`dovsg_core.py` copies DoVSG's functions, citing the source lines in
`external/DovSG/dovsg/controller.py`. DoVSG's `dovsg` package cannot be
imported next to Scene-Graph-Generation's, which has the same name.

Differences from DoVSG, and why:
- **Portrait frames are rotated for ACE.** ACE scales the intrinsics by image height but resizes the shorter side, so a portrait image (the fyp videos are 478×850) gets the wrong camera model. The frames go into ACE rotated 90° and the pose comes back un-rotated.
- **Colored ICP target.** DoVSG adds every matched memory frame at full resolution, repeats included. With all new frames that would be tens of millions of points, so each matched memory frame is added once at 1 cm, which is ICP's finest radius.
- **All new frames are used.** DoVSG uses a few robot views. Use `--observation_stride N` to use every Nth frame.

## Run it

All commands are run from the repo root.

**1. ACE's pose solver.** This is already built on this machine; rerun only if `DovSG/third_party/dsacstar_ext/` is missing.
```bash
dynamic_updates/dovsg_method/build_dsacstar.sh
```

**2. Initial scene graph.** You need `outputs/scene_graphs/<memory>.pkl` (with its voxel map) and `data/processed/<memory>/`.
```bash
conda activate scene_graph
python scripts/run_static_scene_graph.py --config configs/pipeline_fyp_videos.yaml \
    --source data/initial.mp4 --scene_name fyp_initial_v2
```

**3. Train ACE on the memory** (once per scene, about 9 minutes, whatever the number of frames). Runs in the `scene_graph` env.
```bash
python dynamic_updates/dovsg_method/train_ace.py --memory fyp_initial_v2
```
This writes `data/processed/fyp_initial_v2/ace/ace.pt`.

**4. Depth and poses for the new video.**
```bash
python scripts/run_static_scene_graph.py --config configs/pipeline_fyp_videos.yaml \
    --source data/updated_same_angle.mp4 --scene_name fyp_updated_same_angle --pose_only
```

**5. The update.**
```bash
python dynamic_updates/dovsg_method/run_dovsg_update.py --config configs/pipeline_fyp_videos.yaml \
    --memory fyp_initial_v2 --dataset_dir data/processed/fyp_updated_same_angle \
    --scene_name fyp_update_same_angle_dovsg
```

| Flag | Effect |
|---|---|
| `--estimate_scale` | Corrects the scale between the two runs before ICP. This is not DoVSG; compare runs with and without it |
| `--safeguards` | Adds this project's re-detection guard, novelty check, and final merge among new objects only. Not DoVSG |
| `--observation_stride N` | Uses every Nth new frame (default 1: all) |
| `--force_update` | Continues when colored ICP fails |

Settings can go in the config under `dovsg_update:`. Defaults are in
`DEFAULT_CFG` in `run_dovsg_update.py`.

## Outputs

`outputs/dovsg_method/<scene_name>/`:

| Folder | Contents |
|---|---|
| `1_ace/` | `ace_inliers.png` (inliers per frame), `ace_poses.npy` |
| `2_lightglue/` | `matches_per_frame.png` (the anchor in orange); `frame_*.png`: a new frame next to its best memory frame, with the matches |
| `3_colored_icp/` | `top_down_before_after.png`; `pose_check_*.png`: the memory rendered from the ACE pose, after the anchor, and after ICP, over the new frame; `final_poses.npy` |
| `4_changes/` | `seen_through_top_down.png`: deleted voxels in red, each object's deleted share; `seen_through_*.png`: the same on frames |
| `5_update/` | `changes_top_down.png` (removed in red, added in green), `scene_graph_before/`, `scene_graph_after/`, `detections/` |
| `summary.json`, `report.md`, `run.log` | every step's numbers |

The updated scene graph is saved to `outputs/scene_graphs/<scene_name>.pkl` /
`.json`, in the initial run's format, so it can be the memory of the next
update. That update needs its own capture and ACE model; for now, run steps
2–3 for a new memory.

## Reading the results

- **ACE** is only a rough pose. Look at its error against the final pose in `report.md`.
- **Colored ICP** fitness@2cm should go up. If ICP fails, DoVSG stops.
- **Changes.** For each object, `report.md` lists how many frames saw through its deleted voxels. A median of 1 means single frames did it, which with phone depth often means noise, not a real removal.
