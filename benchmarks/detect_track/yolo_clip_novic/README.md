# yolo_clip_novic — old approach (baseline)

```
frame -> YOLO + BoT-SORT -> batch pad-crop -> CLIP ViT-H (batched) -> NOVIC (batched) -> majority-vote noun per track
```

Self-contained: its own `detectors/`, `tagging/`, `trajectory/`, `config.yaml`, `run.py`.

```bash
export NOVIC_CHECKPOINT=/path/to/checkpoint.train
python yolo_clip_novic/run.py --source data/kitchen.mp4 --no-display
python yolo_clip_novic/run.py --source 0                     # webcam, with display
```

On exit it prints the **most common noun per track** (the majority vote accumulated in
`Track.label_frequencies` by `TrajectoryManager.register_label`) with its vote fraction.
A low fraction (e.g. `gas stove 59%`) marks a track whose label is unstable.

## Measured cost (RTX 5000, original fp32 / tag-every-frame configuration)

CLIP cost is ~250 ms **per crop** and scales linearly with object count.
Batching helps NOVIC (2-2.7x for N>=4) but barely helps CLIP (1.04-1.13x), because
ViT-H at 378px is already compute-bound at batch size 1. See the docstring in `pipeline.py`.


## Defaults and options

Defaults are the original behavior: fp32 CLIP, every track re-tagged every frame, and a noun must be
seen at least `min_label_freq: 2` times before it becomes a track's tag (single-observation tracks show
no tag and are left out of the report).

`--fp16` is now wired into `run.py`/`config.yaml`/`pipeline.py` (2026-09-21): 60 frames of living_room.mp4, same YOLOE weights, top nouns **32/32 identical** to fp32, 1.19 FPS vs 0.33 (GPU shared with training).

Re-tag budget (2026-09-21, `--retag-min-votes N --retag-interval K`, implemented in `pipeline.py`): tag every frame until a
track has N votes, then every K frames. Living room, first 60 frames, fp16, vs fp32 tag-every-frame: 
`--retag-min-votes 8 --retag-interval 10` -> **28/32 tracks same top noun, 3.09 FPS**; `5 / 20` -> 25/32, 4.40 FPS.
Baseline fp16 every frame: 32/32, ~1.2 FPS. Pick N/K for your accuracy/speed trade-off.

Opt-in speedups (`--fp16`, `--retag-min-votes N`, `--retag-interval N`) were tried and **not** adopted
as defaults: on a 60-frame check of living_room.mp4 the re-tag policy changed the top noun on 2 of 23
tracks, and fp16 alone matched on 23/23 but has not been checked on the full video.

| config (60 frames, RTX 5000, shared GPU, wall incl. ~35 s load) | wall | FPS at frame 50 | crops | nouns vs fp32 |
|---|---|---|---|---|
| fp32, tag every frame (default) | 346 s | 0.19 | 879 | - |
| `--fp16` | 126 s | 0.66 | 879 | 23/23 same |
| `--fp16 --retag-min-votes 8` | 93 s | 1.15 | 527 | 21/23 same |
