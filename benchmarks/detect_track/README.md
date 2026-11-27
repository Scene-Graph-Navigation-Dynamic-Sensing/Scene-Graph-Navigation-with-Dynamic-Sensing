# Open-vocabulary detect + track pipeline benchmark

All pipelines were run on the same input, `IMG_6100.mp4`: 3840x2160, 30 fps, 212 frames (about 7 s). The clip pans across an office desk.
- **GPU:** one Quadro RTX 5000 (16 GB), one pipeline at a time.
- **Timing:** starts after the frame is decoded and ends when all boxes, labels and masks for that frame are ready. Model loading and visualisation are not counted, and frame 0 is dropped as warm-up.

## Results

| pipeline | ms/frame | FPS | p95 ms | GPU GB | det recall | name acc | correct recall | precision | F1 (correct) | ID switches | label flicker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 DTD: YOLOE-pf + BoT-SORT + CLIP/NOVIC every frame | 974 | 1.03 | 1413 | 4.0 | **0.58** | 0.84 | 0.49 | 0.74 | 0.54 | 2/47 | 0.017 |
| P4 DTD confirm-then-track (lock after 3 votes, 60 %) | **164** | **6.09** | 306 | 4.0 | **0.58** | 0.67 | 0.39 | 0.74 | 0.44 | 2/47 | 0.003 |
| P4b DTD confirm-then-track, strict (5 votes, 70 %) | 215 | 4.65 | 399 | 4.0 | **0.58** | 0.87 | **0.50** | 0.74 | **0.56** | 2/47 | 0.008 |
| P2 RAM++ + YOLOE-26s (text prompts) + BoT-SORT | 314 | 3.18 | 896 | 2.1 | 0.36 | **0.93** | 0.33 | **0.93** | 0.48 | 1/28 | 0.026 |
| P3a RAM++ + GroundingDINO + SAM2 (image) + BoT-SORT | 735 | 1.36 | 802 | 4.4 | 0.39 | 0.89 | 0.35 | 0.88 | 0.48 | 1/32 | 0.002 |
| P3b RAM++ + GroundingDINO + SAM2 video tracking (re-detect every 10 frames) | 728 | 1.37 | 1481 | 14.8 | 0.41 | 0.89 | 0.36 | 0.89 | 0.50 | 1/32 | 0.003 |

Mean time per stage in ms (from `outputs/results.json`):

| pipeline | stages |
|---|---|
| P1 | CLIP ViT-H crops 765 · NOVIC 107 · YOLOE+BoT-SORT 101 |
| P4 | YOLOE+BoT-SORT 100 · CLIP 50 · NOVIC 14 |
| P4b | YOLOE+BoT-SORT 96 · CLIP 94 · NOVIC 25 |
| P2 | RAM++ 144 · YOLOE track 124 · set_classes (text re-encode when new tags appear) 46 |
| P3a | GroundingDINO 368 · SAM2 194 · RAM++ 119 · BoT-SORT 29 |
| P3b | SAM2 video propagation 632 · GroundingDINO 36 · SAM2 image 17 · RAM++ 11 (the detection stages run only 1 frame in 10) |

### Main findings
1. **None of the pipelines is real-time at 30 fps on this GPU.** The fastest is P4 at 6 FPS, about 20 % of real-time.
2. **Confirm-then-track is the best trade-off, but only with a strict lock.**
   - Locking a name after 3 votes (P4) runs 6x faster than P1 but locks early mistakes, for example "silverware" for the backpack and "brown bent" for the wallet. Name accuracy drops from 0.84 to 0.67.
   - Locking after 5 votes with 70 % agreement (P4b) matches or beats P1's accuracy (F1 0.56 vs 0.54) at 4.5x P1's speed.
   - Once CLIP work is cut, the bottleneck is YOLOE + BoT-SORT at about 100 ms. Much of that is likely BoT-SORT's camera-motion compensation running on the 4K frame (sub-stage timing wasn't measured).
3. **The DTD pipelines find far more objects** (recall 0.58 vs 0.36–0.41).
   - YOLOE prompt-free proposes boxes without needing class names, while the RAM-based pipelines can only find what RAM++ tags.
   - RAM++ never produced wallet, box, earbuds case or mouse reliably, and the existing tag filter removes "person".
   - The RAM pipelines are more precise (0.88–0.93) and their names are cleaner.
4. **The SAM2 pipelines give masks but cost the most.**
   - With SAM2's own tracker (P3b), SAM2.1-L video propagation costs about 630 ms per frame. GPU memory peaks at 14.8 GB of 16, and p95 latency is 1.5 s.
   - With BoT-SORT (P3a), average speed is about the same, but latency is steadier (p95 0.8 s) and memory is lower (4.4 GB).
   - GroundingDINO is slow here (368 ms) because it runs without its compiled CUDA op, using the pure-PyTorch fallback.
5. **ID switches are rare in every pipeline** (at most 2 across keyframes 1 s apart). Label flicker is lowest where names are locked or taken from the best detection (P3a/b, P4).
6. **No pipeline found cables or the mouse pad.** Only P1/P4 found the cardboard box, and only on 2 of 8 keyframes.

## How accuracy is measured
- **Ground truth:** `gt/gt.json` covers 8 keyframes (0, 30, …, 210): 23 objects with persistent IDs, 115 boxes in total.
  - Drafted with GroundingDINO **Swin-B**, which no benchmarked pipeline uses (`gt/propose_boxes.py`).
  - Then corrected by eye on each keyframe: duplicates removed, missed objects added, wrong names fixed. `gt/gt_vis/` shows the boxes.
  - Desks, truncated or tiny objects, and cable runs are `ignore` regions, so they are neither required nor counted as false positives.
  - **The annotation was done by Claude, not a human annotator.** Please spot-check `gt/gt_vis/` and edit `gt.json` if needed, then re-run `evaluate.py`.
- **Matching:** one-to-one (Hungarian) at IoU ≥ 0.5.
- **Name scoring:** a name counts as correct if it is in the object's list of accepted names in `gt.json`, or ends with one of them ("office chair" counts as "chair"). Generic but true names such as "computer" for a laptop are accepted. "earplug" and "charger" for the AirPods case are not.
- **Metrics:**
  - *correct recall*: found **and** correctly named, divided by GT objects.
  - *F1 (correct)*: combines correct recall with correct precision.
  - *ID switches*: changes of predicted track ID for the same GT object between consecutive keyframes.
  - *label flicker*: fraction of frame-to-frame steps where a track's displayed name changes.

## Setup details and caveats
- **P1/P4:** reuse `yolo_clip_novic/` (copied unchanged from `dtd_pipeline`), with `yoloe-26s-seg-pf.pt`, CLIP ViT-H-14-378 DFN5B in **fp16** (shown equal to fp32 in that folder's README), and NOVIC `ovod_chunk0433`.
- **P2/P3:** use **RAM++** (`ram_plus_swin_large_14m`) and the tag filter from the existing comparison scripts.
- **P3:** GroundingDINO SwinT-OGC (box/text threshold 0.3), SAM2.1 hiera-L under fp16 autocast. GDINO, SAM2 and tracking run on a 1280x720 copy of the frame.
- **P3a change from `run_dovsg_pipeline.py`:** BoT-SORT gets the frame for camera-motion compensation, and lost tracks are not drawn.
- **CPU contention:** another user's CPU-heavy job (about 19 cores) was running during the benchmark, which may inflate CPU-bound stages (tracking, preprocessing) a little.
- **Scope:** one short clip, 8 keyframes. Treat differences of a few points as noise.

## Layout and re-running
```
pipelines/dtd_yolo_clip_novic.py   P1 (--policy every), P4/P4b (--policy confirm)   [scene_graph env]
pipelines/ram_yoloe_botsort.py     P2                                               [locate-env]
pipelines/ram_gdino_sam2.py        P3a (--tracker botsort), P3b (--tracker sam2)     [locate-env]
common/bench.py                    shared timing / prediction log / 720p annotated video
gt/                                keyframes, GT, drafting + rendering scripts
evaluate.py                        accuracy + speed -> outputs/results.json, table on stdout
make_grid.py                       outputs/comparison_grid.mp4 (P1, P4, P2, P3a, P3b side by side)
run_all.sh                         runs everything sequentially on GPU $GPU (default 0), then evaluates
outputs/<pipeline>/                predictions.jsonl (per-frame boxes/ids/names/stage ms), summary.json, annotated.mp4
```
Use `./run_all.sh` for a full re-run (about 25 min). Use `--frames N` on any pipeline script for a quick test.
