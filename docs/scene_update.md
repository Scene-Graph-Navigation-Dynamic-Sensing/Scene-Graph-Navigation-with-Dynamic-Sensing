# Scene update: new video → changed scene graph

Updates a saved scene graph with a new video of the same scene. It finds the
objects that were added or removed and changes only those nodes; every other
node stays exactly as it was. The method follows DoVSG's scene update
(`external/DovSG/dovsg/controller.py`, `update_scene()`), with the same step
names.

```bash
conda activate scene_graph
python scripts/update_scene_graph.py --config configs/pipeline_fyp_videos.yaml \
    --memory fyp_initial_v2 --source data/updated_same_angle.mp4 --scene_name fyp_update_same_angle
```

- `--memory` is the scene name of the initial run. The script reads
  `outputs/scene_graphs/<memory>.pkl` and the capture in
  `data/processed/<memory>/`.
- `--source` is the new video. Depth (DA3) and camera poses (DROID-SLAM) are
  computed for it first, which takes about 1 minute and ~850 MB of disk.
- `--dataset_dir data/processed/<run>` can replace `--source` when the new
  video has already been processed; it is read, not copied.

## Steps

| # | DoVSG step | What runs here | Code |
|---|---|---|---|
| 1 | `get_align_observations` | Places the new video in the initial map's coordinates. SuperPoint + LightGlue matches between keyframes of both videos are lifted to 3D, a RANSAC Sim3 is fitted (two runs' monocular depth need not share a scale), then ICP refines it on the downsampled maps | `get_align_observations()`, `scripts/compare_video_poses.py` |
| 2 | `find_need_to_delete_indexes` | Projects every voxel of the initial map into each new frame and marks the ones the new video sees through (observed depth behind the stored surface) | `find_need_to_delete_indexes()` |
| 3 | `update_view_dataset` | The new frames are voxelized in the initial map's own voxel grid (same bounds and resolution), so a voxel index means the same place in both videos. See-through voxels leave the map; newly observed ones join it | `ObservationMap`, `update_view_dataset()` |
| 4 | `update_instance_objects` | An object is removed when more than half its voxels were seen through. Fusion then starts from the initial objects: each new detection joins the object it matches (`0.3 × 3D overlap + 0.7 × cosine similarity` of NOVIC embeddings ≥ 0.75), or starts a new object. New objects are kept only if they mostly occupy space the initial map had empty, and NOVIC names them confidently | `update_instance_objects()`, `object_fusion/instance_fusion.py` (`initial_objects=`) |
| 5 | `update_scene_graph` | Nodes of removed objects are deleted and anything they supported is placed again. New objects are placed onto the existing graph with the same `on` / `inside` / `belongs` rules as the initial run | `update_scene_graph()`, `scene_graph/builder/dovsg_relations.py` (`remove_nodes()`, `history_graph=`) |

Like DoVSG, which updates from a few robot views, the script uses only some
of the new video's frames: every 5th (`observation_stride`), 26 frames for
`updated_same_angle.mp4`.

## Where it differs from DoVSG, and why

DoVSG runs on a robot with an RGB-D camera. Here both captures are phone
videos with monocular depth, which changes four things:

- **Relocalization.** DoVSG relocalizes each frame with a pre-trained ACE
  model plus LightGlue and ICP. There is no ACE model for these scenes, so
  the whole new video is registered once from image matches and the 3D maps.
- **Removal margin.** A voxel counts as "seen through" when the observed
  depth is more than 10 cm (or 8% of the distance) behind it, not DoVSG's 2
  voxels, and at least 2 frames with a 60% majority must agree. Monocular
  depth is noisy: the fused tabletop alone is about 15 cm thick.
- **Final merge step.** DoVSG ends fusion by merging overlapping objects.
  After the table top is filled in (`complete_support_surfaces`), everything
  standing on it counts as "overlapping" the table, and that merge glued the
  table, the watch, the mouse and the new bottle into one object. It
  therefore runs only among the new objects, at the initial run's threshold
  (0.55). That is how the bottle's cap joins the bottle.
- **Novelty check (added).** A new object must have at least 50% of its
  voxels farther than 5 cm from anything in the initial map. Without it,
  monocular depth turns a new view of a bare wall or table edge into a "new
  object".

The match threshold is DoVSG's own 0.75, stricter than the 0.55 the initial
run fuses at. At 0.55 the clear bottle merged into the laptop: its depth
shows the laptop behind it, so the 3D overlap is ~1.0, and a cosine of 0.57
brought the score to 0.70.

## Result on the fyp videos

Initial run: `fyp_initial_v2` (`initial.mp4`). New video:
`updated_same_angle.mp4`, the same table with a water bottle added.

| Step | Result |
|---|---|
| Relocalization | scale 1.02, 7.8°, 0.13 m between the two runs; maps agree to a median 10.5 mm |
| Removal | nothing removed; the largest see-through share on any object was 11% (cutoff 50%) |
| Matching | all 8 existing objects matched again |
| Addition | one new object: 1,934 voxels at (−0.20, 1.10, 0.83), 57% in previously empty space, named `water bottle` (confidence 0.46) |
| Graph | `water bottle_0` added `on` the table; no other node changed (checked node by node against the initial graph) |
| Time | ~5 minutes: relocalization 1.3 min, detection + naming 2.5 min (mostly model loading) |

```
floor
├── table_0
├── table_1
│   ├── wristwatch, laptop, computer mouse, backpack, rubik's cube   (unchanged)
│   └── water bottle_0   (on)   <-- added
└── chair_0
```

## Outputs

- `outputs/scene_graphs/<scene_name>.pkl` / `.json`: the updated scene graph,
  in the same format as the initial run's. The per-frame detection masks are
  left out: nothing reads them after fusion, and they made each `.pkl`
  ~300 MB.
- `outputs/scene_graphs/<scene_name>_changes.json`: relocalization quality,
  each object's see-through share, how many new detections matched each
  existing object, and every new object with its closest existing object
  (cosine and 3D-overlap scores), novelty and the decision taken.
- `outputs/visualizations/scene_update/<scene_name>/`: `changes.png`
  (top-down view; added objects in green, removed in red) and
  `scene_graph.pdf`.

Scene maps of both graphs, in the same coordinates, so they can be compared
directly:

```bash
python scripts/visualize_scene_map.py --scene_graph outputs/scene_graphs/fyp_initial_v2.pkl --config configs/pipeline_fyp_videos.yaml
python scripts/visualize_scene_map.py --scene_graph outputs/scene_graphs/fyp_update_same_angle.pkl --config configs/pipeline_fyp_videos.yaml
```

## Settings

`scene_update:` in `configs/pipeline_fyp_videos.yaml`. Keys left unset use
the defaults in `DEFAULT_UPDATE_CFG` in the script.

| Key | Default | Meaning |
|---|---|---|
| `observation_stride` | 5 | every Nth frame of the new video is used |
| `memory_keyframe_stride`, `observation_keyframe_stride` | 8, 2 | frames matched for relocalization |
| `min_registration_fscore` | 0.3 | below this map overlap the run stops: the video probably shows a different scene |
| `delete_depth_m`, `delete_depth_rel` | 0.10, 0.08 | see-through margin: the larger of these metres, or this share of the depth |
| `delete_min_votes`, `delete_vote_share` | 2, 0.6 | frames that must agree before a voxel is deleted |
| `delete_rate` | 0.5 | an object is removed when this share of its voxels is deleted (DoVSG) |
| `redetect_min_detections` | 3 | a "removed" object is kept if the new video detects it again this many times |
| `sim_threshold` | 0.75 | a detection joins an existing object at `0.3 × overlap + 0.7 × cosine` ≥ this (DoVSG) |
| `novelty_distance_m`, `min_novelty` | 0.05, 0.5 | a new object needs this share of voxels farther than this from the initial map |

## Known limitations

- **Transparent objects.** Depth on clear plastic is unreliable. The added
  bottle has a thin tail of points reaching about 50 cm toward the back of
  the table. Its position and its `on table` relation are right, but its
  outline is too long.
- **Naming.** NOVIC decoded the bottle as "orange juice concentrate" (its
  cap is orange); the everyday-name mapping turned that into `water bottle`
  at confidence 0.46. Another video may come out differently. Use
  `label_mapper.manual_labels` in the config to force a name.
- **Removal is only tested on synthetic data.** The fyp videos only add an
  object. `tests/unit/test_object_removal.py` covers the see-through test,
  the 50% rule, node deletion and the re-detection guard, but no real video
  yet takes an object away.
