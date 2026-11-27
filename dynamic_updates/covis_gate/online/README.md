# Co-visibility gate, live

`../run_covis_update.py` takes a whole recorded video and returns an updated scene graph (76 s on `lap_in_1` →
`lap_remove_2`, a 21 s video). This folder runs the same method on a stream: keyframes go in one at a time, the belief
of every memory object is up to date after each, and the scene graph is changed while the video plays. None of the
batch code is edited; the stream code imports its `theory`, `evidence`, `decide`, `targeted`, `dynamism` and
`relocalize` and reuses the pipeline's own detector, tracker, fusion, naming and graph builder.

```bash
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 python dynamic_updates/covis_gate/online/run_covis_live.py \
    --config configs/pipeline_lap_videos.yaml --memory lap_in_1 \
    --dataset_dir data/processed/lap_remove_2 --scene_name lap_remove_2_live --rate 10
```

`--rate 10` replays the recorded keyframes at 10 per second (the rate the lap videos were sampled at), each pushed at
its scheduled time; `--rate 0` pushes as fast as it can. `--no_observe` runs the evidence and decisions only (absent
objects are removed; nothing is added or moved). The replay stands in for the live front end: each keyframe arrives with
the depth and pose that `run_static_scene_graph.py --pose_only` already stored.

Outputs in `outputs/covis_live/<scene_name>/`: `summary.json` (timings, decisions, what the detector saw and when, graph
versions), `events.json` (every committed decision with its time), `timeline.csv` (per keyframe cost and lag),
`run.log`; the updated graph is `outputs/scene_graphs/<scene_name>.pkl/.json`, in the memory's format, as in the batch run.

## What changed, and why it is faster

The batch run's 76 s, and where it went (GPU 1, `results/dynamic_update_timing/.../report.md` and `prof_batch`):

| batch stage | s | what it was spending on | here |
|---|---:|---|---|
| load frames | 2 | numpy, 207 frames at once | frames arrive one by one |
| relocalisation | 16 | 420 LightGlue pairs; 80 SuperPoint extractions of the *same memory frames* again and again | memory features cached; 1 SuperPoint + 3 pairs per 4th keyframe, on its own thread |
| ICP + frame fit | 10 | colored ICP on 20 M points | not used (see below); frame fit is a lookup in a dilated occupancy volume |
| evidence | 8 | 64 ms/frame in numpy: project 338 k voxels (18 ms), colour median (16 ms), `np.add.at` (15 ms) | 4.7 ms/frame on the GPU, **counts identical** |
| belief + gate | 0.04 | one pass over all frames | recomputed from the counts after every keyframe, 2-4 ms, equal to the batch belief of the keyframes so far |
| detection | 31 | CLIP ViT-H on every crop of every detection (17.6 a frame, 83 ms each) and a voxel map rebuilt by 8 fresh processes per call | embedded on first sight + every 5th frame (about two thirds of the crops skipped); the voxel map built in-process; run on the 6-frame windows only, in a worker, off the keyframe loop |
| graph update + map | 5 | alpha shapes etc. | same code, in its own process, started as soon as a window settles |

### Files

| file | role |
|---|---|
| `gpu_evidence.py` | §6 on the GPU: `GpuMemory.process()` = `evidence.classify_frame` + `gather` for one keyframe; edge-free depth mask; dilated occupancy volume |
| `belief.py` | §4/§7 as a running belief: `OnlineBelief.add()` / `.solve()`; after t keyframes equals `evidence.beliefs` on those t keyframes |
| `reloc_online.py` | `FeatureBank` (memory SuperPoint features + each keypoint lifted to 3D with its owner, cached in `data/processed/<memory>/covis_reloc_bank.pt`) and `OnlineLocalizer` (anchor matches accumulated, Sim3 refitted over a sliding window) |
| `engine.py` | `OnlineCovis.push(keyframe)`: alignment → evidence → belief → Chow labels + coverage → committed decisions |
| `novelty.py` | new geometry while the video plays (`targeted.novel_regions`, per keyframe, on the GPU) |
| `observe.py` | when and where the detector runs; window datasets in tmpfs; merging windows; §8.2 association |
| `selective.py` | the pipeline's detection worker with the embedding policy and the quick voxel map |
| `update_graph.py` | §8.3 / §9: build the changed graph (in a process), learn the move rates, save |
| `run_covis_live.py` | the replay driver |
| `test_online.py` | belief = batch (numpy, runs anywhere); GPU evidence = numpy `classify_frame`; edge-free mask = cv2 |

## Decisions that differ from the batch method

- **Sliding-window alignment instead of one T\* for the video.** The batch fits one Sim3 to all matches and then refines
  it with colored ICP. A stream has no end, and the DROID-SLAM trajectory drifts: the accumulated fit moved by 2-3 cm
  over the 21 s video. T\* is refitted from the anchor matches of the last 24 relocalisation keyframes (`reloc_window`;
  0 = all, as the batch). Measured on lap_remove_2 with the through share on objects that are really present as the
  proxy for misalignment (lower is better): batch T\* 5.68 %, batch T\* + ICP 5.79 %, accumulated 6.00 %, window 24
  **5.55 %**, window 12 6.61 %, window 6 6.82 %. So no ICP: it did not improve the evidence, and a fixed ICP correction
  applied to a windowed T\* made it worse (one present object fell to P = 0.13).
- **Decisions are committed after `commit_frames` = 3 evidence frames in a row** with the same Chow label. The belief
  itself is the batch's.
- **Additions need evidence the batch gets for free from fusing all windows**: at least `min_added_detections` = 8
  detections (windows merged by the pipeline's own identity rule), `min_added_voxels` = 200, and the batch's novelty test.
  A glimpse of 5 detections is "tentative", not an addition.
- **New geometry waits** until the localizer has seen 8 keyframes (`novel_warmup_keyframes`): the first alignments are
  2-3 cm off and their errors look like new objects. A region must be seen by 6 keyframes and have 120 cells
  (`novel_trigger_voxels`; the batch keeps regions of 60 cells, computed once at the end).

## Measured

elgin, one Quadro RTX 5000 shared by the engine and the detector, keyframes replayed at 10 per second. Numbers are from
`outputs/covis_live/*/summary.json`. "Batch" is `../run_covis_update.py` with its fast defaults.

| | batch | live |
|---|---:|---:|
| `lap_in_1` → `lap_remove_2` (207 keyframes, 20.7 s of video): time to process it | 76 s, after the video ended | **20.6 s**, while it plays (real time) |
| ... laptop decided removed | at the end | 1.5 s into the video (first alignment after 4 localizer keyframes) |
| ... first graph with laptop removed and iron added | at the end (76 s) | ~17 s into the video |
| ... final graph saved, after the last keyframe | 76 s | 6-9 s (open detection windows 3-4 s, graph 2-6 s) |
| ... per keyframe: mean / median / p95 / max | | 36 / 14 / 108 / 233 ms |
| ... lateness of a keyframe: mean / max | | 8 / 220 ms |
| evidence of one keyframe (GPU busy) | 64 ms | **4.7 ms**, counts identical |
| belief + gate after a keyframe | 35 ms (whole video) | 2-4 ms |
| relocalisation | 15.7 s + ICP 8 s | 5.6 s of background work for 52 keyframes (about 90 ms each); no ICP |
| crops embedded in the detection windows | all (17.6 a frame) | a third: 84 embedded, 174 reused (258 detections) |
| detection window of 6 keyframes | | 6.4-11.7 s each, in a worker process, three windows |
| `lap_in_1` replayed against itself (233 keyframes) | all present | all present, no window ever started, graph unchanged, 23.25 s for 23.3 s of video |
| `lap_in` → `lap_remove` (223 keyframes, a different, larger memory) | nothing changed (117 s, 83 s of it the one-off reliability mask) | nothing changed, the same moved anchor (`bunk bed`) demoted, 22.2 s for 22.3 s of video |

Start-up, before the camera runs: memory 2.4 s; the memory's SuperPoint bank 0.2 s from its cache (10 s the first time for a
memory of 233 frames); the detector's models about 28 s, loaded in their own process while the rest starts.

Results against the batch run on `lap_remove_2`: the laptop is removed and the iron is added, with the same name
("iron", score 0.56-0.59) and the same `on` relation; the iron's centroid differs by up to 6 cm along one axis because the
batch fused 29 detections from three windows and the live run fused its own windows. Everything else stays present; across all the
replays the lowest P a present object ever had was 0.44 (it only ever drifted into "deferred", never to "absent"). The
one repeatable difference: **a plastic bag is added in some runs** (3 of 5 full runs; in the canonical run it stayed
"tentative" with 5 detections). It is the loose transparent bag at the back of the desk, which both videos show in a
different shape; the batch rejects it at novelty 0.381 (threshold 0.4), the live runs measure 0.55-0.75 depending on which
windows saw it. `--set min_novelty=0.6` did not remove it (0.62 in that run), so no stricter default was set on one
scene; `min_added_detections` and `min_support_share` are the other knobs.

## What this replay does not cover

- **Depth and poses are the recorded ones.** Live, DA3 costs 177 ms a frame on this GPU (fixed, independent of resolution
  and precision) and DROID-SLAM has to run online, so the keyframe rate is set by the front end (about 5 a second on one
  GPU), not by this engine, which needs 4.7 ms per evidence frame. The poses used here are DROID-SLAM's final ones; a
  live trajectory is revised by bundle adjustment, and the engine has no hook yet to evaluate a revised keyframe again
  (the keyframes are kept, and the evidence is cheap to repeat).
- **Who is added or moved is known 8-10 s after the evidence says something changed.** The detector is the memory
  pipeline's own (YOLOE + tracker + CLIP ViT-H + fusion + naming), which takes 6-10 s for a window of six keyframes even
  with two thirds of the embeddings skipped; it shares the GPU with the engine. A second GPU for it, or replacing the pipeline
  for these short windows by detections on the new-geometry region alone, are the next steps.
- **Tested on two pairs of videos** (one with a removal and an addition, one with nothing) and the video of the memory
  against itself. A moved object has not been exercised live; `decide.associate` is the batch's.
- **A long session** keeps the evidence of every keyframe in the belief (`thr_mat` grows) and the last 400 keyframes in
  RAM; a rolling window of evidence is needed beyond a few minutes.
- The GPU clocks drop in the idle gaps of a replay with nothing else running, so the evidence step looks 4x slower
  (21-27 ms) there than the 4.7 ms it takes when the GPU is busy, as it is in a live system.

## Optimised further (2026-10-23): same results, less waiting

`lap_in_1` then `lap_remove_2` (chained visits, see `outputs/covis_live/chain_lap_in_1_then_lap_remove_2.md`) rerun after four changes, each checked to give identical numbers. Same loaded server (all GPUs 78-100 % busy with other users' jobs), back to back.

| | camera → graph, before (s) | after (s) | after last keyframe, before | after | worst keyframe before (ms) | after | removed / added (after) |
|---|---:|---:|---:|---:|---:|---:|---|
| Visit 1: `lap_in_1` | 25.7 | **25.6** | 2.4 | 2.3 | 945 | 363 | - / - |
| Visit 2: `lap_remove_2` | 47.3 | **40.2** | 26.4 | 19.6 | 1018 | 376 | laptop computer_0 / iron_0 |
| both | 73.0 | **65.8** | | | | | |

| change | where | effect | why it loses nothing |
|---|---|---|---|
| footprint cache | graph process (`update_graph._install_footprint_cache`) | graph rebuild 5.1 s → **0.09 s**; cache warmed (6.6 s) at start-up, during the video | the alpha-shape footprint of an unchanged object depends only on its points; same 14 edges as uncached (and the original jitters with unseeded noise anyway) |
| CLIP batch 32 | detection worker (`selective`) | 145 → 69 ms per crop | embeddings identical (max difference 6e-8) |
| windows coalesced | `observe` | windows that come due while a job runs go out as one job (commits, naming, graph once) | the batch method fuses all its windows in one run too |
| catch-up spread | `engine.max_drain_per_push` | worst keyframe ~1 s → ~0.4 s | same evidence, evaluated over the next few keyframes |

Tried and not kept as default: separate CUDA streams for YOLOE and CLIP inside the worker (kept, but the window job only got 7 % faster: it is limited by the CPU/GIL of its three threads and by the other user's process on the GPU); two detection workers (`--set observe_workers=2`): on the loaded GPU the first window took 31.5 s instead of 15 s, so the default is 1.

What is left after the last keyframe (~20 s in visit 2) is the last detection window: 15-19 s on this loaded GPU, 6-10 s on the idle machine a week earlier. That is the YOLOE + CLIP + fusion pipeline itself, sharing a GPU with another job.
