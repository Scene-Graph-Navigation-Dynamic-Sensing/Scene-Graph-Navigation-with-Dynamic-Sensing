# Dynamic_Attention

Attention for dynamic scene-graph updates, built on the co-visibility gate (CoViS). The prior, the log-odds
belief, Chow's rule, the graph update and the rate learning stay those of `covis_gate`. Two learned modules
replace the hand-tuned rules at the two places where those rules fail:

| Module | Replaces | Where |
|---|---|---|
| **Attention matcher**: tokens = CLIP appearance + Fourier position + PointNet shape + side; 3 blocks of self- and cross-attention; Sinkhorn with a dustbin | `decide.associate`: `0.3 × overlap + 0.7 × cosine ≥ 0.75`, the Hungarian with `≥ 0.75`, the 8-detection minimum for an addition | §8.2 association: same object / moved / new |
| **Presence head**: the memory object's token (CLIP embedding + its own YOLOE patch descriptor from a memory frame) attends to the current keyframe's YOLOE patches inside its projected footprint; outputs present / absent / occluded | adds learned appearance evidence where depth is weak (thin objects; the colour threshold 0.45 was hand-set) | one more term in the CoViS log-odds per frame, with the same frame discount and neighbour gate |

None of the other methods is changed. `online_attention.py` subclasses `OnlineBelief`, `OnlineCovis` and
`ObservationManager`. `run_attention_live.py` is `run_covis_live.py`'s `main()` with those classes swapped in.
With the presence term off, `AttentionBelief` gives exactly `OnlineBelief`'s numbers (tested).

## Files

| File | Role |
|---|---|
| `common.py` | paths; `YoloeFeatures` (YOLOE-26s backbone and neck up to layer 19, stride-16 patches, 384-d); footprint projection |
| `models.py` | `AttentionMatcher`, `PresenceHead`, log-domain Sinkhorn |
| `build_replica_data.py` | training and test data from the 8 Replica scenes already evaluated (`results/replica_final`) |
| `train_eval.py` | trains both modules; tests them on held-out Replica scenes against the rule-based logic |
| `online_attention.py` | the live engine pieces (belief, engine, observation manager) |
| `run_attention_live.py` | the live replay driver |
| `test_dynamic_attention.py` | unit tests |

## Data and training (no own videos)

Each Replica scene's pipeline run gives fused objects with every detection's frame and box, ground-truth object
ids and ground-truth depth. Each walk-through is split into two visits: memory = frames 0–109, new visit =
frames 90–199. For each object and visit, the builder keeps the voxels that visit's detections really saw and a
CLIP embedding of that visit's crops. Changes are simulated per training window:

- **removed:** a track is dropped
- **added:** a memory object is dropped
- **moved:** a track is shifted 0.5–1.8 m

Training adds monocular-style noise. Presence samples are labelled from ground-truth depth: present, occluded,
and absent (the footprint shifted onto bare surface, or another object in the footprint).

Split: train office_0–3, room_0, room_1. office_4 is validation (matcher threshold, presence temperature).
room_2 is the test scene. The lap and fyp videos are test only.

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python dynamic_updates/Dynamic_Attention/build_replica_data.py
CUDA_VISIBLE_DEVICES=0 python dynamic_updates/Dynamic_Attention/train_eval.py
python -m pytest dynamic_updates/Dynamic_Attention/test_dynamic_attention.py -q
```

Outputs: `outputs/dynamic_attention/data/*.pt`, `checkpoints/{matcher,presence}.pt`, `replica_eval.json`.

## Live

```bash
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python dynamic_updates/Dynamic_Attention/run_attention_live.py \
    --config configs/pipeline_lap_landscape_videos.yaml --memory lap_in_1 \
    --dataset_dir data/processed/lap_remove_2 --scene_name lap_remove_2_attention --rate 10
```

Ablations:
- `--set presence=false`: matcher only.
- `--set matcher=false`: presence head only, with `decide.associate`.

Settings: `ATT_DEFAULTS` in `online_attention.py`.

Outputs: `outputs/dynamic_attention/live/<scene>/` holds `summary.json` (with an `attention` block),
`events.json`, `timeline.csv` and `presence.json` (per keyframe, per object: P(present, absent, occluded)).
The graph is saved as `outputs/scene_graphs/<scene>.pkl/.json`.

## Results (2026-11-25)

Full numbers: `outputs/dynamic_attention/REPORT.md`.

- **Presence head, held-out Replica:** present vs absent 0.92–0.93 accuracy and AUROC 0.98, against 0.65–0.72 and
  0.66–0.80 for a cosine baseline. ECE 0.02–0.04. The occluded class is weak.
- **Matcher v2, held-out Replica:** no better than the rule overall. On the room_2 test scene it keeps 93 % of
  unchanged objects against 100 %, with added F1 0.93 against 0.97 and moved F1 0.64 against 0.74. It has fewer false
  moves.
- **Live lap_in_1 → lap_remove_2:** the laptop is removed and the iron added, as with the rule. The iron reaches the
  graph about 10 s earlier (3 detections needed instead of 8), and the borderline plastic bag is added too.
- **Live no-change replay:** nothing changed. The head voted "absent" for cup_0 on about half the frames, but depth
  kept it present.
- **Cost:** the presence head adds about 30–40 ms per evidence frame; one association takes about 260 ms.

Settings added after the first live test (`ATT_DEFAULTS`). They were found on lap_remove_2 itself:
- `presence_on_anchors=false`
- `matcher_redetect_min_p=0.8`
- `matcher_redetect_min_cos=0.75`
- a weak match to an absent object counts as unmatched
