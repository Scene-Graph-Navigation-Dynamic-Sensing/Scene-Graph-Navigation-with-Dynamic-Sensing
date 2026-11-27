#!/usr/bin/env python3
"""
main.py — Pipeline Orchestrator  (with per-step result saving)
==============================================================
Pipeline stages and what gets saved to disk for each:

  Step 1 — YOLOE-26s-seg-pf detection
             outputs/<run>/step1_detection/
               <img>_boxes.jpg        ← bboxes drawn, objectness conf labels
               <img>_detections.json  ← bbox coords + conf per object (text)

  Step 2 — Instance masks (binary, one per object)
             outputs/<run>/step2_masks/
               <img>/obj_00_mask.png  ← white-on-black binary mask
               <img>/obj_00_mask_overlay.jpg ← coloured mask on original

  Step 3 — Segmented crops (background zeroed by mask)
             outputs/<run>/step3_crops/
               <img>/obj_00_crop.jpg  ← isolated object crop fed to tagger

  Step 4 — Embeddings  (float32 vectors, NOVIC only; zeros when using vLLM)
             outputs/<run>/step4_embeddings/
               <img>_embeddings.npy   ← (N, F) float32  array
               <img>_embeddings.json  ← human-readable shape / norm info

  Step 5 — Tags  (open-vocabulary nouns from NOVIC or vLLM VLM)
             outputs/<run>/step5_tags/
               <img>_tags.json        ← tag + score + top-K per object

  Step 6 — Final annotated image (masks + bboxes + noun labels)
             outputs/<run>/step6_final/
               <img>_annotated.jpg

  Summary — one JSON covering all images in the run
             outputs/<run>/run_summary.json

Taggers
-------
  --tagger novic  (default)
      Uses the trained NOVIC decoder checkpoint for open-vocabulary tagging.
      NOVIC weights (confirmed present):
        /home/ravindu/SceneGraphNav/external/novic/outputs/ovod_20240628_142131/
            ovod_chunk0433_20240630_235415.train   (42,919-noun vocab)
            Encoder: openclip:apple/DFN5B-CLIP-ViT-H-14-378 — top-ranked
            embedder in NOVIC's own benchmark (70.90%/84.37% top1/top5,
            novic/config/train.yaml). Confirmed via A/B test to beat the
            older ViT-B-16-SigLIP checkpoint (ovod_20240610_105233) on
            real crops — e.g. correctly tags a mouse as "computer mouse"
            (0.92) instead of "mouse button" (0.71) or "crease" (0.05).

  --tagger vllm
      Uses a vLLM-served Vision-Language Model (VLM) for fast batched tagging.
      vLLM brings PagedAttention + continuous batching over all crops in one
      GPU pass — much faster for large batches than serial NOVIC beam-search.
      Supported VLMs (via --vllm_model):
        llava-hf/llava-1.5-7b-hf          (default, 7B, fast)
        llava-hf/llava-v1.6-mistral-7b-hf  (LLaVA-NeXT, higher accuracy)
        Qwen/Qwen2-VL-7B-Instruct          (strong multimodal reasoning)
        microsoft/Phi-3.5-vision-instruct   (efficient, 4B)
      Install: pip install vllm

YOLOE-26-seg-pf weights (confirmed present):
  /home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26s-seg-pf.pt   (small, default)
  /home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26m-seg-pf.pt   (medium)
  /home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26l-seg-pf.pt   (large)

Usage:
    # NOVIC tagger (default)
    python main.py path/to/images/

    # vLLM tagger (fast batched VLM inference)
    python main.py images/ --tagger vllm
    python main.py images/ --tagger vllm --vllm_model Qwen/Qwen2-VL-7B-Instruct
    python main.py images/ --tagger vllm --vllm_tp 2   # 2-GPU tensor parallel

    # Common options
    python main.py images/ --run_name my_experiment
    python main.py images/ --yoloe_weights .../yoloe-26l-seg-pf.pt
    python main.py images/ --no_save    # dry-run, nothing written
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ── project root on sys.path ────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.detection.yoloe_seg_detector import YOLOESegDetector, Detection
from semantic_decoding.novic.crop_utils import extract_crop
from semantic_decoding.novic.novic_decoder import NOVICDecoder, TagResult
from semantic_decoding.vllm_tagger import VLLMTagger

# Generic type alias accepted anywhere a tagger is expected
Tagger = NOVICDecoder | VLLMTagger

logger = logging.getLogger("main")


# ════════════════════════════════════════════════════════════════════════════
# Default weight paths  (confirmed present on this machine)
# ════════════════════════════════════════════════════════════════════════════

NOVIC_DIR_DEFAULT = "/home/ravindu/SceneGraphNav/external/novic"
NOVIC_CHECKPOINT_DEFAULT = (
    "/home/ravindu/SceneGraphNav/external/novic/outputs/"
    "ovod_20240628_142131/ovod_chunk0433_20240630_235415.train"
)
YOLOE_WEIGHTS_DEFAULT = "/home/ravindu/SceneGraphNav/models/checkpoints/yoloe-26s-seg-pf.pt"


# ════════════════════════════════════════════════════════════════════════════
# Per-object result dataclass
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ObjectResult:
    """One detected + tagged object in a single image."""
    object_id:  int
    bbox:       np.ndarray   # [x1, y1, x2, y2]
    conf:       float        # YOLOE objectness score
    tag:        str          # NOVIC top-1 noun
    score:      float        # NOVIC top-1 probability
    top_k:      list[tuple[str, float]]
    embedding:  np.ndarray   # (F,) float32


# ════════════════════════════════════════════════════════════════════════════
# Colour palette (BGR)
# ════════════════════════════════════════════════════════════════════════════

_PALETTE_BGR = [
    (56, 214, 255), (80, 255, 130), (255, 100, 60),  (100,  60, 255),
    (255, 200,  50), (200,  50, 255), (50, 255, 200), (255,  50, 200),
    (130,  80, 255), (255, 130,  80), (60, 255, 100), (100, 255,  60),
    (60,  100, 255), (255,  60, 100), (200, 255,  50), (50,  200, 255),
    (255, 255,  80), (80,  255, 255), (255,  80, 255), (128, 128, 255),
]

def _colour(i: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[i % len(_PALETTE_BGR)]


# ════════════════════════════════════════════════════════════════════════════
# Drawing helper
# ════════════════════════════════════════════════════════════════════════════

def _draw_boxes(
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    labels: list[str] | None = None,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Return a new annotated image (RGB uint8)."""
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]

    # Scale box/text thickness to the frame size so labels stay readable on
    # high-res (e.g. 4K video) frames instead of shrinking to near-invisible.
    scale     = max(w, h) / 1000.0
    box_lw    = max(2, round(2 * scale))
    fs        = max(0.6, 0.7 * scale)
    th_lw     = max(1, round(2 * scale))

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = box.astype(int)
        colour = _colour(i)

        if masks is not None:
            mask = masks[i].astype(np.uint8)
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = canvas.copy()
            overlay[mask > 0] = (
                overlay[mask > 0] * 0.55 + np.array(colour) * 0.45
            ).astype(np.uint8)
            canvas = cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, box_lw)

        text = labels[i] if labels is not None else f"obj {i}  {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, fs, th_lw)
        pad = max(4, round(4 * scale))
        ly = max(y1 - pad - 2, th + pad)
        cv2.rectangle(canvas, (x1, ly - th - pad), (x1 + tw + 2 * pad, ly + pad // 2), colour, -1)
        cv2.putText(canvas, text, (x1 + pad, ly), font, fs, (0, 0, 0), th_lw, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


# ════════════════════════════════════════════════════════════════════════════
# Per-step savers
# ════════════════════════════════════════════════════════════════════════════

def _save_img(rgb: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def save_step1_detection(
    stem: str,
    image_rgb: np.ndarray,
    boxes: np.ndarray,
    confs: np.ndarray,
    out_dir: Path,
) -> None:
    """Step 1 — bounding-box visualisation + raw detection JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # image with boxes
    viz = _draw_boxes(image_rgb, boxes, confs, labels=None)
    _save_img(viz, out_dir / f"{stem}_boxes.jpg")

    # machine-readable detections
    detections_data = [
        {"object_id": i, "bbox": boxes[i].tolist(), "conf": float(confs[i])}
        for i in range(len(boxes))
    ]
    with open(out_dir / f"{stem}_detections.json", "w") as f:
        json.dump({"image": stem, "num_objects": len(boxes), "detections": detections_data}, f, indent=2)

    logger.info(f"    [step1] saved {len(boxes)} detections → {out_dir}")


def save_step2_masks(
    stem: str,
    image_rgb: np.ndarray,
    masks: np.ndarray,
    confs: np.ndarray,
    out_dir: Path,
) -> None:
    """Step 2 — binary mask PNGs + coloured overlay JPEGs, one per object."""
    obj_dir = out_dir / stem
    obj_dir.mkdir(parents=True, exist_ok=True)

    h, w = image_rgb.shape[:2]
    for i, mask in enumerate(masks):
        colour = _colour(i)

        # binary mask (white on black)
        mask_u8 = (mask.astype(np.uint8) * 255)
        if mask_u8.shape != (h, w):
            mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(obj_dir / f"obj_{i:02d}_mask.png"), mask_u8)

        # coloured overlay on original
        canvas_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        overlay = canvas_bgr.copy()
        m = mask_u8 > 0
        overlay[m] = (overlay[m] * 0.45 + np.array(colour) * 0.55).astype(np.uint8)
        canvas_bgr = cv2.addWeighted(overlay, 0.8, canvas_bgr, 0.2, 0)
        x1, y1, x2, y2 = _bbox_from_mask(m)
        cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), colour, 2)
        label = f"obj {i:02d}  conf={confs[i]:.2f}"
        cv2.putText(canvas_bgr, label, (x1 + 3, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
        cv2.imwrite(str(obj_dir / f"obj_{i:02d}_mask_overlay.jpg"), canvas_bgr)

    logger.info(f"    [step2] saved {len(masks)} mask(s) → {obj_dir}")


def save_step3_crops(
    stem: str,
    crops: list[np.ndarray],
    out_dir: Path,
) -> None:
    """Step 3 — segmented crop image per object (fed directly to NOVIC)."""
    obj_dir = out_dir / stem
    obj_dir.mkdir(parents=True, exist_ok=True)

    for i, crop in enumerate(crops):
        _save_img(crop, obj_dir / f"obj_{i:02d}_crop.jpg")

    logger.info(f"    [step3] saved {len(crops)} crop(s) → {obj_dir}")


def save_step4_embeddings(
    stem: str,
    embeddings: np.ndarray,
    out_dir: Path,
) -> None:
    """Step 4 — NOVIC embeddings as .npy (binary, exact) + .json (human-readable)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(out_dir / f"{stem}_embeddings.npy"), embeddings)

    meta = {
        "image": stem,
        "num_objects": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim > 1 else 0,
        "dtype": str(embeddings.dtype),
        "per_object": [
            {
                "object_id": i,
                "norm": float(np.linalg.norm(embeddings[i])),
                "mean": float(embeddings[i].mean()),
                "std":  float(embeddings[i].std()),
            }
            for i in range(embeddings.shape[0])
        ],
    }
    with open(out_dir / f"{stem}_embeddings.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"    [step4] saved embeddings {embeddings.shape} → {out_dir}")


def save_step5_tags(
    stem: str,
    results: list[ObjectResult],
    out_dir: Path,
) -> None:
    """Step 5 — NOVIC noun tags as JSON (tag + score + top-K alternatives)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "image": stem,
        "num_objects": len(results),
        "objects": [
            {
                "object_id": r.object_id,
                "bbox": r.bbox.astype(int).tolist(),
                "yoloe_conf": round(r.conf, 4),
                "novic_tag": r.tag,
                "novic_score": round(r.score, 4),
                "top_k": [
                    {"label": label, "score": round(float(prob), 4)}
                    for label, prob in r.top_k
                ],
            }
            for r in results
        ],
    }
    with open(out_dir / f"{stem}_tags.json", "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"    [step5] saved {len(results)} tag(s) → {out_dir}")


def save_step6_final(
    stem: str,
    image_rgb: np.ndarray,
    results: list[ObjectResult],
    masks: np.ndarray,
    out_dir: Path,
) -> np.ndarray:
    """Step 6 — Final annotated image with masks + bboxes + NOVIC labels."""
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes  = np.stack([r.bbox for r in results])
    confs  = np.array([r.conf for r in results])
    labels = [f"{r.tag}  {r.score:.2f}" for r in results]

    viz = _draw_boxes(image_rgb, boxes, confs, labels=labels, masks=masks)
    _save_img(viz, out_dir / f"{stem}_annotated.jpg")

    logger.info(f"    [step6] saved final annotated image → {out_dir}")
    return viz


def save_run_summary(
    run_dir: Path,
    all_results: dict[str, list[ObjectResult]],
    run_name: str,
    elapsed_sec: float,
) -> None:
    """Write a single JSON summarising the entire run across all images."""
    summary = {
        "run_name": run_name,
        "elapsed_seconds": round(elapsed_sec, 2),
        "total_images": len(all_results),
        "total_objects": sum(len(v) for v in all_results.values()),
        "images": {
            stem: [
                {
                    "object_id": r.object_id,
                    "bbox": r.bbox.astype(int).tolist(),
                    "yoloe_conf": round(r.conf, 4),
                    "novic_tag": r.tag,
                    "novic_score": round(r.score, 4),
                }
                for r in results
            ]
            for stem, results in all_results.items()
        },
    }
    path = run_dir / "run_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Run summary saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# Core pipeline  (returns intermediate artefacts for saving)
# ════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    image_rgb: np.ndarray,
    detector: YOLOESegDetector,
    tagger: "Tagger",           # NOVICDecoder  OR  VLLMTagger
    crop_mode: str = "bbox",
    crop_padding_px: int = 15,
) -> tuple[
    list[Detection],        # raw detections (bbox+mask+conf)
    list[np.ndarray],       # segmented crops
    np.ndarray,             # embeddings (N, F)
    list[ObjectResult],     # tagged results
]:
    """
    Run the full 5-stage pipeline on one RGB image.

    Returns all intermediate artefacts so the caller can save each stage
    independently (or skip saving for a dry-run).

    Stages
    ------
    1  YOLOE-26s-seg-pf  → detections (bbox + mask + conf)
    2  Instance masks     → already inside detections[i].mask
    3  extract_crop()     → segmented crops (background = 0)
    4  Embeddings         → (N, F) float32 (NOVIC) or zeros (vLLM)
    5  Tag decode         → noun tags via tagger (NOVICDecoder or VLLMTagger)
    """
    h, w = image_rgb.shape[:2]

    # ── 1. Detect & segment ──────────────────────────────────────────────
    detections: list[Detection] = detector.detect(image_rgb)
    logger.info(f"  [1] YOLOE: {len(detections)} object(s) detected.")

    if not detections:
        empty = np.empty((0, 1), dtype=np.float32)
        return detections, [], empty, []

    boxes = np.stack([d.bbox  for d in detections])
    masks = np.stack([d.mask  for d in detections])
    confs = np.array([d.conf  for d in detections])

    # ── 2. Masks (already in detections; nothing extra to compute) ───────
    logger.info(f"  [2] Masks: {len(masks)} instance mask(s) ready.")

    # ── 3. Segmented crops ───────────────────────────────────────────────
    crops = [
        extract_crop(image_rgb, boxes[i], masks[i],
                     mode=crop_mode, padding_px=crop_padding_px)
        for i in range(len(detections))
    ]
    logger.info(f"  [3] Crops: {len(crops)} segmented crop(s) extracted.")

    # ── 4. Embed (NOVIC only; vLLM does embed+decode together in step 5) ──
    if isinstance(tagger, NOVICDecoder):
        embeddings: np.ndarray = tagger.embed_crops(crops)   # (N, F)
        logger.info(f"  [4] NOVIC embeddings: shape={embeddings.shape}.")
    else:
        # vLLM VLM: no separate CLIP embedding step
        n_feat = 512
        embeddings = np.zeros((len(crops), n_feat), dtype=np.float32)
        logger.info(f"  [4] vLLM mode: embedding step skipped (zeros placeholder).")

    # ── 5. Tag decode ────────────────────────────────────────────────────
    if isinstance(tagger, NOVICDecoder):
        # NOVIC: cheap beam-search over pre-computed embeddings
        tag_results: list[TagResult] = tagger.decode_embeddings(embeddings)
    else:
        # vLLM: one batched VLM generate() call covers all crops at once
        tag_results = tagger.tag(crops)

    results = [
        ObjectResult(
            object_id = i,
            bbox      = detections[i].bbox,
            conf      = detections[i].conf,
            tag       = tag.label,
            score     = tag.score,
            top_k     = tag.top_k,
            embedding = tag.embedding,
        )
        for i, tag in enumerate(tag_results)
    ]
    tagger_name = "NOVIC" if isinstance(tagger, NOVICDecoder) else "vLLM"
    logger.info(f"  [5] {tagger_name} tags: {[r.tag for r in results]}")

    return detections, crops, embeddings, results


# ════════════════════════════════════════════════════════════════════════════
# Console printer
# ════════════════════════════════════════════════════════════════════════════

def print_results(stem: str, results: list[ObjectResult], top_k: int = 5) -> None:
    print(f"\n{'─'*72}")
    print(f"  Image : {stem}")
    print(f"  Found : {len(results)} object(s)")
    print(f"{'─'*72}")
    for r in results:
        print(f"\n  [obj {r.object_id:02d}]  yoloe_conf={r.conf:.3f}   "
              f"bbox={r.bbox.astype(int).tolist()}")
        print(f"            NOVIC tag  →  \"{r.tag}\"   (score={r.score:.4f})")
        if len(r.top_k) > 1:
            print(f"            Top-{min(top_k, len(r.top_k))} alternatives:")
            for label, prob in r.top_k[1:top_k]:
                print(f"                {prob:.4f}  {label}")
        print(f"            embedding  : shape={r.embedding.shape}  "
              f"norm={np.linalg.norm(r.embedding):.3f}")
    print()


# ════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ════════════════════════════════════════════════════════════════════════════

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def collect_image_paths(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            for child in sorted(p.iterdir()):
                if child.suffix.lower() in _IMG_EXTS:
                    paths.append(child)
        elif p.is_file() and p.suffix.lower() in _IMG_EXTS:
            paths.append(p)
        else:
            logger.warning(f"Skipping '{inp}' — not an image file or directory.")
    return paths


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _bbox_from_mask(mask_bool: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box [x1,y1,x2,y2] of the True region in a bool mask."""
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return 0, 0, mask_bool.shape[1], mask_bool.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Pipeline orchestrator: images → YOLOE-26s-seg-pf → segmented crops "
            "→ NOVIC → meaningful noun tags.  Saves every intermediate step."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("inputs", nargs="+",
                   help="Image file(s) or directory of images.")

    # YOLOE
    p.add_argument("--yoloe_weights", default=YOLOE_WEIGHTS_DEFAULT)
    p.add_argument("--conf_thresh",   type=float, default=0.25)
    p.add_argument("--iou_thresh",    type=float, default=0.50)
    p.add_argument("--img_size",      type=int,   default=640)

    # ── Tagger selection ────────────────────────────────────────────────────
    p.add_argument(
        "--tagger", default="novic", choices=["novic", "vllm"],
        help=(
            "Semantic tagger to use.  "
            "'novic' (default): custom NOVIC beam-search decoder checkpoint.  "
            "'vllm': Vision-Language Model served by vLLM inference engine — "
            "faster for large batches via PagedAttention + continuous batching."
        ),
    )

    # NOVIC (used when --tagger novic)
    p.add_argument("--novic_dir",        default=NOVIC_DIR_DEFAULT)
    p.add_argument("--novic_checkpoint", default=NOVIC_CHECKPOINT_DEFAULT)
    p.add_argument("--gencfg",           default="beam_k10_vnone_gp_t1_a0")
    p.add_argument("--top_k",    type=int, default=5,
                   help="Top-K NOVIC alternatives to display / save.")

    # vLLM (used when --tagger vllm)
    p.add_argument(
        "--vllm_model", default="llava-hf/llava-1.5-7b-hf",
        help=(
            "HuggingFace model ID (or local path) of the VLM to load via vLLM. "
            "Examples: llava-hf/llava-1.5-7b-hf (default),  "
            "llava-hf/llava-v1.6-mistral-7b-hf,  Qwen/Qwen2-VL-7B-Instruct,  "
            "microsoft/Phi-3.5-vision-instruct"
        ),
    )
    p.add_argument(
        "--vllm_dtype", default="float16",
        choices=["float16", "bfloat16", "float32", "auto"],
        help="Weight dtype for the vLLM engine.  Default: float16",
    )
    p.add_argument(
        "--vllm_tp", type=int, default=1, dest="vllm_tensor_parallel",
        help="Tensor-parallel size (number of GPUs to shard across).  Default: 1",
    )
    p.add_argument(
        "--vllm_max_model_len", type=int, default=2048,
        help="Max sequence length (tokens) for the vLLM engine.  Default: 2048",
    )
    p.add_argument(
        "--vllm_gpu_util", type=float, default=0.90,
        help="GPU memory fraction vLLM may use (0–1).  Default: 0.90",
    )
    p.add_argument(
        "--vllm_noun_hint", default=None,
        help=(
            "Optional comma-separated list of candidate nouns to inject into the "
            "vLLM prompt, e.g. 'chair,table,bottle'.  Ignored for NOVIC."
        ),
    )

    # Crop
    p.add_argument("--crop_mode",       default="bbox",
                   choices=["segmented", "bbox"])
    p.add_argument("--crop_padding_px", type=int, default=15)

    # Output
    p.add_argument("--output_dir", default="outputs",
                   help="Root output directory.  A timestamped sub-folder is "
                        "created unless --run_name is specified.")
    p.add_argument("--run_name", default=None,
                   help="Name for this run's output sub-folder (default: timestamp).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_save", action="store_true",
                   help="Dry-run: run the pipeline but skip all file writes.")
    return p


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args()

    # ── Collect images ─────────────────────────────────────────────────────
    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        logger.error("No valid image files found.  Exiting.")
        sys.exit(1)
    logger.info(f"Found {len(image_paths)} image(s) to process.")

    # ── Build run output directory ─────────────────────────────────────────
    run_name = args.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    run_dir  = REPO_ROOT / args.output_dir / run_name

    if not args.no_save:
        # Create all 6 step directories upfront so they always exist
        for step in [
            "step1_detection",
            "step2_masks",
            "step3_crops",
            "step4_embeddings",
            "step5_tags",
            "step6_final",
        ]:
            (run_dir / step).mkdir(parents=True, exist_ok=True)

        logger.info(f"Output root: {run_dir}")
        logger.info(
            "Sub-folders:\n"
            f"  step1_detection/   — bbox images + detections.json\n"
            f"  step2_masks/       — binary masks + coloured overlays\n"
            f"  step3_crops/       — segmented crop images (fed to NOVIC)\n"
            f"  step4_embeddings/  — .npy arrays + embedding meta JSON\n"
            f"  step5_tags/        — NOVIC tag JSON (tag + score + top-K)\n"
            f"  step6_final/       — final annotated images\n"
            f"  run_summary.json   — full run overview"
        )

    # ── Load models ────────────────────────────────────────────────────────
    logger.info(f"Loading YOLOE-26-seg-pf  : {args.yoloe_weights}")
    detector = YOLOESegDetector(
        weights_path = args.yoloe_weights,
        conf_thresh  = args.conf_thresh,
        iou_thresh   = args.iou_thresh,
        img_size     = args.img_size,
        device       = args.device,
    )

    # ── Build the tagger (NOVIC or vLLM) ──────────────────────────────────
    if args.tagger == "vllm":
        logger.info(f"Building vLLM tagger     : {args.vllm_model}")
        tagger: Tagger = VLLMTagger(
            model                  = args.vllm_model,
            dtype                  = args.vllm_dtype,
            max_model_len          = args.vllm_max_model_len,
            tensor_parallel_size   = args.vllm_tensor_parallel,
            gpu_memory_utilization = args.vllm_gpu_util,
            noun_hint              = args.vllm_noun_hint,
            device                 = args.device,
        )
        if tagger.is_stub:
            logger.warning(
                "vLLM VLM could not be loaded → stub-tagger mode.  "
                "Install vLLM: pip install vllm"
            )
    else:
        logger.info(f"Building NOVIC decoder   : {args.novic_checkpoint}")
        tagger = NOVICDecoder(
            novic_dir       = args.novic_dir,
            checkpoint_path = args.novic_checkpoint,
            gencfg          = args.gencfg,
            device          = args.device,
        )
        if tagger.is_stub:
            logger.warning("NOVIC checkpoint not loaded → stub-decoder mode (placeholder tags).")

    # ═══════════════════════════════════════════════════════════════════════
    # Per-image loop
    # ═══════════════════════════════════════════════════════════════════════
    all_results: dict[str, list[ObjectResult]] = {}
    t_start = time.time()

    for img_path in image_paths:
        stem = img_path.stem
        logger.info(f"\n{'═'*60}")
        logger.info(f"Processing: {img_path.name}")

        try:
            image_rgb = load_rgb(img_path)
        except FileNotFoundError as exc:
            logger.error(str(exc))
            continue

        # ── Run pipeline  (get all intermediates back) ─────────────────
        detections, crops, embeddings, results = run_pipeline(
            image_rgb       = image_rgb,
            detector        = detector,
            tagger          = tagger,
            crop_mode       = args.crop_mode,
            crop_padding_px = args.crop_padding_px,
        )

        # ── Console output ─────────────────────────────────────────────
        print_results(img_path.name, results, top_k=args.top_k)

        if not detections:
            logger.info("  No objects detected — skipping save steps.")
            all_results[stem] = []
            continue

        boxes = np.stack([d.bbox for d in detections])
        masks = np.stack([d.mask for d in detections])
        confs = np.array([d.conf for d in detections])

        # ── Save each step ─────────────────────────────────────────────
        if not args.no_save:
            save_step1_detection(stem, image_rgb, boxes, confs,
                                 run_dir / "step1_detection")

            save_step2_masks(stem, image_rgb, masks, confs,
                             run_dir / "step2_masks")

            save_step3_crops(stem, crops,
                             run_dir / "step3_crops")

            save_step4_embeddings(stem, embeddings,
                                  run_dir / "step4_embeddings")

            save_step5_tags(stem, results,
                            run_dir / "step5_tags")

            save_step6_final(stem, image_rgb, results, masks,
                             run_dir / "step6_final")

        all_results[stem] = results

    # ── Run summary JSON ───────────────────────────────────────────────────
    elapsed = time.time() - t_start
    if not args.no_save:
        save_run_summary(run_dir, all_results, run_name, elapsed)

    # ── Final console summary ──────────────────────────────────────────────
    total_objects = sum(len(v) for v in all_results.values())
    print(f"\n{'═'*72}")
    print(f"  Pipeline complete.")
    print(f"  Run name         : {run_name}")
    print(f"  Images processed : {len(all_results)}")
    print(f"  Objects tagged   : {total_objects}")
    print(f"  Time elapsed     : {elapsed:.1f}s")
    if not args.no_save:
        print(f"\n  Saved outputs under: {run_dir}/")
        print(f"    step1_detection/   ← bbox images + detections.json")
        print(f"    step2_masks/       ← binary masks + coloured overlays")
        print(f"    step3_crops/       ← segmented crops (fed to NOVIC)")
        print(f"    step4_embeddings/  ← .npy arrays + embedding meta JSON")
        print(f"    step5_tags/        ← NOVIC tag JSON (tag + score + top-K)")
        print(f"    step6_final/       ← final annotated images")
        print(f"    run_summary.json   ← full run overview")
    print(f"{'═'*72}\n")

    tagger.close()


if __name__ == "__main__":
    main()
