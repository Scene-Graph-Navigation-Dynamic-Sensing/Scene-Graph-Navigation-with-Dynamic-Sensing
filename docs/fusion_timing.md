# Mapping time: where it goes, step by step

Where the time goes when the pipeline builds a scene graph, and which step makes
object fusion slow. **Goal: build a Replica scene (200 frames) in under 10 minutes.**
Today it takes 10–64 minutes.

Status (2026-10-06): stage and thread timings are measured for all 8 Replica scenes;
the per-function profile covers two scenes, office_1 (the fastest) and room_2 (a
mid-sized one). Profiling office_0 and room_0 was stopped as unnecessary: the two
profiles agree, and the overlap checks' share grows with map size, so the larger
scenes would only show the same cause more strongly. No code has been changed for speed.

## Setup

- Replica, final evaluation configuration: YOLOE-26x prompt-free at 1024 px and
  confidence 0.10, merge fixes, brightness correction, fusion settings
  (`configs/replica_variants/{merge_fixes,brightness,fusion_naming,discovery}.yaml`).
- 200 frames per scene (every 10th of 2,000), ground-truth depth and poses, so no
  depth or SLAM model runs.
- Machine: 3× Quadro RTX 5000 (16 GB), shared. The 8 final runs ran three at a time,
  and GPUs 1–2 also carried other users' training jobs, so absolute times are
  somewhat higher than on a quiet machine. The profiled runs ran one at a time on GPU 0.
- Sources: `results/replica_final/<scene>/pipeline.log` and `metrics.json` (stage and
  thread timings), `results/fusion_profile/` (py-spy profiles).

## 1. Every pipeline step, per scene

Minutes. The streaming stage runs detection, embedding and fusion in parallel
threads, so its length equals that of the slowest thread, which is always fusion.

| Scene | Total | Frames + depth | Poses | Gravity | **Streaming** | Final fusion | Clean-up | Naming | Relations |
|---|---|---|---|---|---|---|---|---|---|
| office_1 | 9.8 | 0.1 | 0.0 | 0.0 | **7.8** | 0.6 | 0.0 | 0.1 | 0.2 |
| room_2 | 26.2 | 0.1 | 0.0 | 0.0 | **22.5** | 1.9 | 0.0 | 0.1 | 0.5 |
| room_1 | 27.6 | 0.1 | 0.0 | 0.0 | **24.2** | 1.3 | 0.0 | 0.0 | 0.9 |
| office_3 | 34.9 | 0.1 | 0.0 | 0.0 | **29.2** | 3.1 | 0.0 | 0.1 | 1.2 |
| office_2 | 39.2 | 0.1 | 0.0 | 0.0 | **35.3** | 2.0 | 0.0 | 0.0 | 0.6 |
| office_4 | 46.8 | 0.1 | 0.0 | 0.0 | **41.7** | 2.7 | 0.0 | 0.1 | 1.1 |
| office_0 | 63.6 | 0.2 | 0.0 | 0.0 | **46.9** | 13.7 | 0.0 | 0.1 | 1.4 |
| room_0 | 64.4 | 0.1 | 0.0 | 0.0 | **55.1** | 6.2 | 0.0 | 0.0 | 1.5 |

(The total also includes model loading, about 0.4 min.) Streaming is 80–92% of the
total; the final fusion stage is the only other step that matters (up to 13.7 min,
on office_0).

## 2. Inside streaming: the three threads

| Scene | Detection (YOLOE + BoT-SORT) | Embedding (CLIP ViT-H, every crop) | **Fusion** | of which waiting for voxels | Track commits | Detections |
|---|---|---|---|---|---|---|
| office_1 | 4.0 | 4.0 | **7.8** | 1.5 | 135 | 2,357 |
| room_2 | 4.2 | 4.3 | **22.5** | 1.6 | 171 | 2,663 |
| room_1 | 4.2 | 4.3 | **24.2** | 1.5 | 188 | 2,637 |
| office_3 | 5.2 | 5.2 | **29.2** | 1.7 | 178 | 3,183 |
| office_2 | 4.4 | 4.4 | **35.3** | 1.5 | 208 | 2,754 |
| office_4 | 4.0 | 4.0 | **41.7** | 1.9 | 206 | 2,373 |
| office_0 | 4.4 | 4.4 | **46.9** | 1.8 | 236 | 2,600 |
| room_0 | 5.3 | 5.7 | **55.1** | 1.5 | 240 | 3,747 |

Detection and embedding finish in 4–6 minutes on every scene; after that the
pipeline waits for fusion. **Fusion is 85–95% of mapping time.**

## 3. Why fusion time varies so much: cost per commit

A commit happens when a track ends (unseen for 10 frames); it is compared with
every object already in the map and merged or added.

| Scene | Commits | Fusion | **Per commit** |
|---|---|---|---|
| office_1 | 135 | 7.8 min | **3.5 s** |
| room_1 | 188 | 24.2 min | 7.7 s |
| room_2 | 171 | 22.5 min | 7.9 s |
| office_3 | 178 | 29.2 min | 9.9 s |
| office_2 | 208 | 35.3 min | 10.2 s |
| office_0 | 236 | 46.9 min | 11.9 s |
| office_4 | 206 | 41.7 min | 12.1 s |
| room_0 | 240 | 55.1 min | **13.8 s** |
| room_0, baseline config | 203 | 27.1 min | 8.0 s |

The number of detections barely matters (office_1 and office_4 have about the
same). The time per commit grows with the size of the map, so total cost grows
roughly with commits × map size. The final configuration roughly doubles fusion
time (room_0: 27 → 55 min), because its detector produces more tracks and more
objects to compare against.

## 4. Inside fusion: the profile

py-spy sampled every thread 10×/s without pausing the process
(`results/fusion_profile/run_profiles.sh`, analysed with
`scripts/analyze_fusion_profile.py`). The profiled office_1 run took 9.6 min against
9.8 min unprofiled, so the profiler does not distort the timings.

### office_1 (7.7 min of fusion; 6.2 min busy + 1.5 min waiting for voxels)

| Step | Share of busy fusion time | Minutes |
|---|---|---|
| **Commit: voxel overlap, forward** (the track vs each mapped object) | **32.9%** | 2.0 |
| **Commit: voxel overlap, reverse** (each mapped object vs the track) | **30.9%** | 1.9 |
| Per frame: detection masks → voxels | 19.2% | 1.2 |
| Every 20 frames: DBSCAN clean-up of all objects | 7.6% | 0.5 |
| Per frame: merge a detection into its pending track | 4.7% | 0.3 |
| Commit: DBSCAN clean-up of the track | 2.4% | 0.1 |
| Commit: box overlap, cosine, merge, other | ~2% | 0.1 |

Final fusion stage (0.4 min): overlap merge 68%, DBSCAN over all objects 25%.

### Hottest lines (office_1)

| Share | Code | What it does |
|---|---|---|
| 17.5% | `instance_fusion.py:169` | converts each mapped object's whole voxel list to an int64 array — for every object, on every commit, to build the cache key |
| 4.3% | `instance_fusion.py:170` | fingerprints (blake2b) that array: the cache key |
| ~25% | `instance_fusion.py:189–222` (`_grid_neighbourhood`, `searchsorted`) | reverse check: neighbourhood of every mapped object's voxels, each looked up in the track |
| 19.6% | `instance_process.py:791, 795` | DBSCAN (k-nearest-neighbour fit + clustering) |
| 5.1% | `instance_process.py:886` | bounding box via an Open3D point cloud |

### room_2 (19.5 min of fusion; 17.9 min busy + 1.5 min waiting for voxels)

Run alone: 23.3 min total (26.2 min in the final run, which shared the machine).

| Step | Share of busy fusion time | Minutes |
|---|---|---|
| **Commit: voxel overlap, reverse** | **44.1%** | 7.9 |
| **Commit: voxel overlap, forward** | **31.7%** | 5.7 |
| Per frame: detection masks → voxels | 12.5% | 2.2 |
| Every 20 frames: DBSCAN clean-up of all objects | 5.4% | 1.0 |
| Per frame: merge a detection into its pending track | 3.2% | 0.6 |
| Commit: DBSCAN clean-up of the track | 1.9% | 0.3 |
| Commit: box overlap, cosine, merge, other | ~1% | 0.2 |

Final fusion stage (2.0 min): voxel ownership between overlapping objects
(`indexes_align_objects`) 58%, overlap merge 33%, DBSCAN 9%.

The two overlap checks rise from 64% of busy fusion time on office_1 to **76%** on
room_2: the larger the map, the more they dominate.


## 5. What this means

Both overlap directions process every object in the map on every commit:

- **Forward:** before checking overlap, it converts and fingerprints every mapped
  object's full voxel list just to see whether its cached sorted copy is still valid.
  The work done on objects the track can't overlap is pure overhead.
- **Reverse:** it builds the 7-voxel neighbourhood of every mapped object and looks
  it up in the track, again including objects that are nowhere near it.

So each commit costs about as much as the total number of voxels in the map, which
explains the 3.5 → 13.8 s per commit growth. Candidate fixes (none applied yet):

1. Restrict both directions to objects whose bounding boxes overlap the track's (the
   forward direction already filters this way before its inner loop, but only after
   preparing every object).
2. Key the cache by object id plus a version number bumped on every merge, instead of
   converting and hashing the whole voxel list.
3. Run the 20-frame DBSCAN clean-up only on objects that changed since the last one.

None of these changes the fusion result, only how it is computed; a fix would be
checked by confirming that a re-run produces identical objects.
