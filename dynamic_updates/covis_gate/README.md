# Co-visibility gated change detection

A third way to update a scene graph with a new video of the same scene, next to
`dovsg_method/` and `projection_method/`. It implements
`covis-change-detection-theory.pdf` (in this folder). None of the other methods'
code is changed: their loaders, LightGlue wrapper and graph rebuild are imported.

The initial (memory) scene graph is made with the existing pipeline,
`scripts/run_static_scene_graph.py`. This method only does the change detection.

## What the PDF asks for, and where it is

| PDF | What | Code |
|---|---|---|
| §3 | Move rate per object, λ ~ Gamma(a, b); Lomax survival S(Δt) → prior log-odds | `theory.survival`, `dynamism.py` |
| §3.4 | Initial (a, b): uniform weak prior, or a fixed warm-start table (`rate_prior: table`, default) | `dynamism.RATE_TABLE` |
| §4 | Log-odds of presence; ℓ = ln 0.85/0.08; Kish discount α; clamping ±8 | `evidence.beliefs` |
| §5 | No ACE. LightGlue matches kept only on anchors (rate < λ_A, plus the background); RANSAC + Umeyama with the eq. 12 iteration count; φ_j per anchor; moved anchors demoted | `relocalize.py` |
| §6 | Signed residual: hit / through / occluded with σ(z); invalid depth and depth edges discarded; colour turns a hit into a through | `evidence.classify_frame` |
| §7 | Noisy-OR gate ρ over graph neighbours (0.9 belongs/inside, 0.8 on, next-to edges); tempered update (eq. 17) or mixture (eq. 18); anchors first, then target sweeps | `evidence.neighbours`, `evidence.beliefs` |
| §8.1 | Chow's rule (thresholds from costs) + coverage condition | `decide.chow_labels`, `evidence.gated_coverage` |
| §8.2 | Absent objects vs unexplained detections: Hungarian assignment weighted by the receptacle prior (eq. 20) → moved / removed / added | `decide.associate` |
| §8.3 | Only the changed nodes' edges are rebuilt | `scripts/update_scene_graph.update_scene_graph` (reused) |
| §9 | Stayed → (a, b+Δt); moved/removed → moment-matched (eq. 22-23); deferred → no update; forgetting γ | `theory.update_*`, `Dynamism.learn` |

`test_theory.py` checks `theory.py` against the numbers printed in the PDF:
Figure 2, the RANSAC table, the §7.5 worked example and the §9.2 table.
The Figure 2 caption prints 0.980 for the monitor, but eq. 3 gives 0.971; the test uses 0.971.

```bash
python -m pytest dynamic_updates/covis_gate/test_theory.py -q
```

## Where it differs from the PDF, and why

- **One T\* for the whole video.** The new video's frames are already tied together by its own DROID-SLAM trajectory, so the anchor matches of every frame go into one RANSAC.
- **Sim3, not SE(3).** The depth is monocular, so the two videos' scales differ by a few percent. Use `--set rigid=true` for the PDF's SE(3) when the depth is metric.
- **σ(z) = σ0 + k1 z + k2 z².** The PDF's σ0 + k z² models stereo depth; a monocular network's error is a share of the distance, which is the linear term. Set `sigma_k1=0` to get the PDF's form.
- **Colour threshold 0.45.** A thin object that is removed (the laptop) leaves the desk 1-2 cm lower, which is inside the depth tolerance; only colour shows it. On lap_in_1 → lap_remove_2, hits on the removed laptop had a median RGB distance of 0.66, and hits on present objects 0.10-0.35. At 0.2, about half of every present object's voxels became throughs.
- **Frame correlation.** About 200 nearly identical video frames are not independent observations. The same Kish discount as eq. 6 is applied across the frames that saw an object (`rho_f`, 0.1; 0 turns it off).
- **Local background as a certifier.** Background voxels (walls, floor, unsegmented surface) within 15 cm of a target are one more noisy-OR term (`background_weight`, 0 turns it off).
- **Coverage.** Each judged voxel counts once, with the largest ρ of a frame that saw through it, divided by the object's judged voxels. A sum over ~200 frames would not stay in [0, 1].
- **Judged voxels.** Only voxels the memory's own frames confirm are judged (projection_method's self-test, shared cache). Monocular surfaces are layers several cm thick (`reliability_mask: false` turns this off).
- **Association score.** s = 0.3 × shape + 0.7 × cosine, where shape is the overlap once the centroids coincide (a moved object does not overlap its old place). The receptacle prior weights the Hungarian assignment; the δ = 0.75 test is on s itself.
- **Re-detected in place.** An object the belief calls absent, but which the new video's detector finds at the same place with the same look, is kept present (`redetection_overrides_absent`).
- **Time.** Without a dynamism file the memory is visit 0 at t = 0, and the new video is `--delta_t_days` later (default 1).

## Speed

The defaults are set for an update right after recording. On lap_in_1 → lap_remove_2 an update takes
76 s (667 s before these settings), and 56 s when nothing changed; see
`results/dynamic_update_timing/lap_in_1_to_lap_remove_2/report.md`. That excludes the new video's
depth and poses (`run_static_scene_graph.py --pose_only`, ~80 s), which every method needs first.

| What | How | Setting |
|---|---|---|
| Detection only where needed | The detector serves only §8.2 (moved / added / re-detected). It runs on 6-frame windows around absent or deferred objects and around new geometry (new-frame points > 5 cm from the map, seen in >= 3 frames); not at all when nothing is absent, deferred or new | `detect: needed` (`all`, `off`) |
| Models loaded early | YOLOE, CLIP ViT-H and NOVIC load in a worker process started with the update, while relocalisation runs, so detection does not wait ~30 s for them (`warm_detect.py`: the pipeline's own functions in its own order) | `warm_models` |
| Lighter relocalisation | One T* for the whole video needs only a spread of frames: retrieval 8 x 30, refinement every 4th frame against 3 memory frames, ICP on every 4th frame. Poses within 0.7° / 3 cm of the full version, same fit | `retrieval_*`, `pose_frame_stride`, `refine_candidates`, `icp_frame_stride` |
| Fewer evidence frames | every 2nd frame (neighbouring frames are near duplicates; the frame discount `rho_f` adapts) | `evidence_frame_stride` |
| Cheap loading | edge-free depth mask only, no per-frame statistical outlier removal | `fast_load` |
| Map update | only the voxels of removed / moved / added objects change; graph rebuilt only if something changed | `map_update: changes` (`all`) |
| Pictures | plots and 3 frames; the evidence video of every frame only with `viz=full` | `viz: light` (`full`, `off`) |

The earlier, slower behaviour, for comparison:
`--set detect=all warm_models=false map_update=all viz=full fast_load=false retrieval_new_frames=24
retrieval_memory_frames=60 pose_frame_stride=1 refine_candidates=4 icp_frame_stride=1 evidence_frame_stride=1 detect_window=8`.

## Live

The batch settings above are as fast as a run over a finished video gets (76 s for a 21 s video). `online/` runs the same
method on a stream, keyframe by keyframe, and changes the scene graph while the video plays: the evidence of a keyframe
takes 4.6 ms (batch: 64 ms) with identical counts, the laptop in `lap_remove_2` is decided removed 1.5 s into the video,
and the whole 20.7 s video is processed in 20.7 s. See `online/README.md` for the design, what differs from this method
and why, and the measurements.

## Run it

In the `scene_graph` env, from the repo root. The memory needs `outputs/scene_graphs/<memory>.pkl` and
`data/processed/<memory>/`; the new video needs `run_static_scene_graph.py --pose_only` (same as the other methods).
No ACE training is needed.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python dynamic_updates/covis_gate/run_covis_update.py \
    --config configs/pipeline_lap_videos.yaml --memory lap_in_1 \
    --dataset_dir data/processed/lap_remove_2 --scene_name lap_remove_2_covis
```

| Flag | Effect |
|---|---|
| `--delta_t_days D` | days since the memory (sets the prior S(Δt)) |
| `--skip_observation` | no detection on the new video: absent → removed, nothing moved or added (fast, for checking the evidence) |
| `--reuse_observation` | reuse the new video's scene graph from an earlier run with the same name |
| `--force_update` | continue when relocalisation fails (otherwise the whole visit is deferred, §5.5) |
| `--set KEY=VALUE ...` | any setting in `DEFAULT_CFG`; also config section `covis_update:` |

Ablations (§12): `--set gate=false`, `update=mixture`, `prior=off`, `rate_prior=uniform`,
`rate_update=naive`, `residual=unsigned`, `rigid=true`.

The next visit reads `outputs/scene_graphs/<scene_name>_dynamism.json` when `--memory <scene_name>`
is used, so the learned rates carry over.

## Outputs

`outputs/covis_gate/<scene_name>/`:

| Folder | Contents |
|---|---|
| `0_prior/` | `survival.png`: S(Δt) per object |
| `1_relocalization/` | `1_retrieval_matches.png`, `1_matches_*.png` (anchor matches green, others red), `1_phi_per_anchor.png`, `2_pose_check_*.png`, `final_poses.npy` |
| `3_evidence/` | `evidence.mp4` (with `viz=full`): every frame with each object voxel's outcome (green hit, red through, blue occluded) and each object's P and ρ after that frame, next to the memory rendered from the same pose; `frame_*.png`; `beliefs.png`; `gate_rho.png`; `outcomes_top_down.png` |
| `4_decision/` | `decision.png`: P with the Chow thresholds, and the coverage |
| `5_observation/` | the detector's run on the chosen frames (log, pictures); which frames and why: `report.md` section 5 |
| `6_update/` | `changes_map.png`, `scene_graph_before/`, `scene_graph_after/`, `dynamism.json` |
| `report.md`, `summary.json`, `raw.npz`, `run.log` | every number; `raw.npz` has the per-frame per-object counts, ρ, belief trajectories and per-voxel outcomes |

Updated scene graph: `outputs/scene_graphs/<scene_name>.pkl/.json`, in the memory's format.
