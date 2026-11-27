"""Assigns the final NOVIC name to each fused 3D instance.

Names are decided during fusion (object_fusion/instance_fusion.py): when a
BoT-SORT track ends, the running-average embedding of the object it fed is
decoded by NOVIC, and the result is appended to that object's 'track_names'.
An instance takes the latest record that still describes it. Instances that
changed after that decode (merged by DoVSG's merge_objects() or by
postprocess.merge_fragmented_instances()), and instances built only from
untracked detections, are decoded here from their final running-average
embedding, in one batch. label_mapper then canonicalizes the labels, and a
name NOVIC scored below min_name_score becomes "unknown" (or the instance is
dropped). This is the only place class_name / class_id / semantic_embedding
are set.
"""
from __future__ import annotations

import logging

import numpy as np

from semantic_decoding.label_mapper.label_mapper import canonicalize, merge_near_duplicates
from semantic_decoding.novic.novic_decoder import NOVICDecoder

logger = logging.getLogger(__name__)


def current_track_name(obj) -> dict | None:
    """The latest track-end decode that still matches this object. A merge
    adds detections and moves the averaged embedding, so a record only counts
    if the object's detection count is unchanged since it was decoded."""
    for record in reversed(obj.get("track_names") or []):
        if record["num_detections"] == obj["num_detections"]:
            return record
    return None


UNKNOWN_LABEL = "unknown"
DEFAULT_SURFACE_LABEL = "table"
# head nouns (after canonicalize) that name a surface things are placed on
SURFACE_NOUNS = frozenset({
    "table", "desk", "workbench", "bench", "counter", "countertop", "worktop",
    "bureau", "escritoire", "shelf", "cabinet", "dresser", "sideboard", "nightstand",
})


def surface_label(top_k) -> tuple[str, float | None]:
    """The first surface noun among NOVIC's beam alternatives, with its score,
    or ("table", None) when none of them names a surface."""
    for label, score in top_k or []:
        head = canonicalize(label)
        if head in SURFACE_NOUNS:
            return head, float(score)
    return DEFAULT_SURFACE_LABEL, None


def _unit(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8, None)


def to_everyday_names(novic_labels, embeddings, everyday_names, embed_text,
                      label_weight: float = 0.2) -> list[str]:
    """Replaces each NOVIC label that isn't an everyday name with the everyday
    name that best fits the object: CLIP similarity between the object's
    image embedding and "a photo of a <name>", plus `label_weight` times the
    text similarity between <name> and NOVIC's own label.

    NOVIC's vocabulary is 42,919 nouns and it often picks a rare or overly
    general one: IMG_6100's wallet decoded as "billfold" and its office
    chairs as "office furniture" (0.85, with "chair" at 0.01 in the beam).
    The image term picks the name that looks right ("wallet", "chair"); the
    label term keeps NOVIC's meaning when the image alone is ambiguous
    ("earplug" becomes "earphone", where the image alone says "charger").
    Labels already in the list are kept as they are.
    """
    names = list(everyday_names)
    name_set = set(names)
    todo = [i for i, label in enumerate(novic_labels) if label not in name_set]
    out = list(novic_labels)
    if not todo or not names:
        return out
    prompts = _unit(embed_text([f"a photo of a {name}" for name in names]))
    words = _unit(embed_text(names))
    own = _unit(embed_text([novic_labels[i] for i in todo]))
    for row, i in enumerate(todo):
        score = prompts @ _unit(np.ravel(embeddings[i])) + label_weight * (words @ own[row])
        out[i] = names[int(np.argmax(score))]
    return out


def name_confidences(top_ks, embed_text, synonym_similarity: float = 0.8) -> list[float]:
    """Per instance: the beam score of its top label plus every alternative
    whose text embedding is at least `synonym_similarity` alike (desk +
    bureau + escritoire). NOVIC spreads its probability over synonyms, so the
    top score alone understates a real object (the second desk on IMG_6100:
    0.10, pooled 0.25), while a guess's alternatives are unrelated words
    (hashish, flatiron, inkpad: 0.09 either way)."""
    labels = sorted({label for top_k in top_ks for label, _ in (top_k or [])})
    if not labels:
        return [0.0] * len(top_ks)
    vectors = np.asarray(embed_text(labels), dtype=np.float32)
    vectors /= np.clip(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8, None)
    vector_of = dict(zip(labels, vectors))

    confidences = []
    for top_k in top_ks:
        if not top_k:
            confidences.append(0.0)
            continue
        reference = vector_of[top_k[0][0]]
        confidences.append(float(sum(score for label, score in top_k
                                     if vector_of[label] @ reference >= synonym_similarity)))
    return confidences


def tag_instances(objects, novic_decoder: NOVICDecoder, merge_similarity_threshold: float = 0.9,
                  min_name_score: float = 0.15, low_score_action: str = "drop",
                  support_surfaces=(), synonym_similarity: float = 0.8,
                  everyday_names=None, everyday_name_weight: float = 0.2, manual_labels=None,
                  fallback_min_share: float = 0.25):
    """Sets on each instance of `objects` (a DoVSG MapObjectList):
        - class_name: label_mapper-canonicalized NOVIC label, or "unknown"
        - class_id: f"{class_name}_{n}" per-class running id
        - novic_score / novic_top_k: top-label score and beam alternatives
        - name_confidence: novic_score pooled over synonyms (name_confidences())
        - novic_guess: the decoded label, kept when class_name is "unknown"
        - semantic_embedding: the instance's running-average NOVIC embedding
        - named_by: "track_end", "final_decode" or "support_surface"
        - novic_label: NOVIC's own label, when it was replaced by an
          everyday name (to_everyday_names())

    A name whose name_confidence is below `min_name_score` is NOVIC guessing. With
    low_score_action "drop" the instance is removed, along with its tracks,
    track-end names and running-average embedding; with "unknown" it stays
    under the label "unknown".

    `support_surfaces` holds indexes of instances shaped like a table top
    (postprocess.find_support_surfaces()). Each is named with the first
    surface noun in its beam alternatives, or "table", and is kept whatever
    its score: the geometry is stronger evidence than a crop mostly showing
    the clutter on top of it.

    A name decoded from the final averaged embedding can be weak where each
    track's was not: a laptop's screen, lid and sticker tracks average to
    "five" (0.09) although its main track ended as "laptop". Such an instance
    falls back to its most confident track-end name, among the tracks that
    backed at least `fallback_min_share` of its detections at the time.

    `manual_labels` maps a NOVIC label to the name a person says the object
    has ({"brand": "laptop"}: NOVIC reads a laptop lid's logo). Matching
    instances take that name and are kept whatever their score. Returns the
    instances that remain.
    """
    if low_score_action not in (UNKNOWN_LABEL, "drop"):
        raise ValueError(f"low_score_action must be 'unknown' or 'drop', got {low_score_action!r}")
    if len(objects) == 0:
        return objects

    embeddings = np.stack([np.asarray(obj["clip_ft"], dtype=np.float32).ravel() for obj in objects])
    current = [current_track_name(obj) for obj in objects]
    need_decode = [i for i, record in enumerate(current) if record is None]
    decoded = (dict(zip(need_decode, novic_decoder.decode_embeddings(embeddings[need_decode])))
               if need_decode else {})
    logger.info(f"Named {len(objects) - len(need_decode)}/{len(objects)} instances from their track-end "
                f"decode, {len(need_decode)} decoded from their final averaged embedding.")

    novic_labels, scores, top_ks = [], [], []
    for i in range(len(objects)):
        if i in decoded:
            r = decoded[i]
            label, score, top_k = r.label, r.score, r.top_k
        else:
            label, score, top_k = current[i]["label"], current[i]["score"], current[i]["top_k"]
        novic_labels.append(label)
        scores.append(float(score))
        top_ks.append(top_k)

    names = novic_labels
    if everyday_names:
        names = to_everyday_names(novic_labels, embeddings, everyday_names, novic_decoder.embed_text,
                                  everyday_name_weight)
        renamed = [f"{a} -> {b}" for a, b in zip(novic_labels, names) if a != b]
        if renamed:
            logger.info(f"Everyday names for {len(renamed)} instance(s): {renamed}")
    raw_labels = [canonicalize(name) for name in names]

    confidences = name_confidences(top_ks, novic_decoder.embed_text, synonym_similarity)
    for i, top_k in enumerate(top_ks):
        if not top_k:  # no beam kept: fall back to the top score itself
            confidences[i] = scores[i]

    fallback = set()
    for i, obj in enumerate(objects):
        if confidences[i] >= min_name_score or i in support_surfaces:
            continue
        records = [r for r in obj.get("track_names") or []
                   if r.get("top_k") and r["num_detections"] >= fallback_min_share * obj.get("num_detections", 0)]
        if not records:
            continue
        record_conf = name_confidences([r["top_k"] for r in records], novic_decoder.embed_text, synonym_similarity)
        b = int(np.argmax(record_conf))
        if record_conf[b] < min_name_score:
            continue
        r = records[b]
        name = (to_everyday_names([r["label"]], embeddings[i:i + 1], everyday_names, novic_decoder.embed_text,
                                  everyday_name_weight)[0] if everyday_names else r["label"])
        logger.info(f"Weak name '{raw_labels[i]}' ({confidences[i]:.2f}) replaced by the track-end name "
                    f"'{name}' ({record_conf[b]:.2f}, a track backing {r['num_detections']} of "
                    f"{obj.get('num_detections')} detections).")
        novic_labels[i], scores[i], top_ks[i] = r["label"], float(r["score"]), r["top_k"]
        confidences[i] = float(record_conf[b])
        names[i] = name
        raw_labels[i] = canonicalize(name)
        fallback.add(i)

    surfaces = set(support_surfaces)
    for i in sorted(surfaces):
        label, score = surface_label(top_ks[i])
        logger.info(f"Support surface: '{raw_labels[i]}' ({scores[i]:.2f}) renamed '{label}' "
                    f"({'beam score ' + format(score, '.2f') if score is not None else 'no surface noun in beam'}).")
        raw_labels[i] = label
        if score is not None:
            scores[i] = score

    manual = set()
    for i, novic_label in enumerate(novic_labels):
        if manual_labels and novic_label in manual_labels and i not in surfaces:
            logger.info(f"Manual label: '{novic_label}' ({raw_labels[i]}, {confidences[i]:.2f}) "
                        f"named '{manual_labels[novic_label]}'.")
            raw_labels[i] = canonicalize(manual_labels[novic_label])
            manual.add(i)

    confident = [i for i in range(len(objects))
                 if confidences[i] >= min_name_score or i in surfaces or i in manual]
    low = [i for i in range(len(objects)) if i not in confident]
    if low:
        logger.info(f"{len(low)} instance(s) with name confidence below {min_name_score} "
                    f"({'dropped' if low_score_action == 'drop' else 'labelled unknown'}): "
                    f"{[f'{raw_labels[i]} {confidences[i]:.2f}' for i in low]}")
    kept = confident if low_score_action == "drop" else list(range(len(objects)))

    # An instance split off a support surface (postprocess._split_raised_part) shares its source's
    # embedding, so the near-duplicate merge would fold its label into the table's ("laptop" -> "table",
    # the shortest of the two): it keeps the name it decoded, like a manual label.
    split_off = {i for i, obj in enumerate(objects) if obj.get("split_from_surface") and i not in surfaces}
    # unknowns stay out of the near-duplicate merge, which would otherwise be
    # free to rename a real label to its shortest group member, "unknown"
    mergeable = [i for i in confident if i not in split_off]
    merge_map = merge_near_duplicates([raw_labels[i] for i in mergeable],
                                      [embeddings[i] for i in mergeable], threshold=merge_similarity_threshold)

    class_id_counts: dict[str, int] = {}
    for i in kept:
        obj, raw_label = objects[i], raw_labels[i]
        if i in surfaces or i in manual or i in split_off:
            final_label = raw_label
        elif confidences[i] >= min_name_score:
            final_label = merge_map.get(raw_label, raw_label)
        else:
            final_label = UNKNOWN_LABEL
            obj["novic_guess"] = raw_label
        count = class_id_counts.get(final_label, 0)
        class_id_counts[final_label] = count + 1

        obj["class_name"] = final_label
        obj["class_id"] = f"{final_label}_{count}"
        obj["novic_score"] = scores[i]
        obj["name_confidence"] = confidences[i]
        if names[i] != novic_labels[i]:
            obj["novic_label"] = novic_labels[i]
        obj["novic_top_k"] = top_ks[i]
        obj["semantic_embedding"] = embeddings[i]
        obj["named_by"] = ("support_surface" if i in surfaces else "manual" if i in manual
                           else "track_end_fallback" if i in fallback
                           else "final_decode" if i in decoded else "track_end")

        logger.debug(f"Instance {obj['class_id']}: '{raw_label}' -> '{final_label}' "
                     f"(score={scores[i]:.3f}, {obj['named_by']})")

    return type(objects)([objects[i] for i in kept])
