"""Canonicalizes NOVIC's free-form noun predictions into stable scene-graph
node labels.

NOVIC decodes open-vocabulary noun phrases (e.g. "brown wooden office
chair"), which can vary across views of the same physical object ("office
chair" vs "chair" vs "swivel chair"). This keeps scene graph labels
readable and consistent without requiring an external taxonomy: strip
leading modifiers to the head noun (English noun phrases put the head
noun last: "brown wooden office chair" -> "chair"), then merge
near-duplicate labels within one scene by embedding similarity.
"""
from __future__ import annotations

import numpy as np

_STOPWORDS = {"a", "an", "the", "of", "and", "or"}


def canonicalize(label: str) -> str:
    """Strip leading modifiers, keep the head noun. Multi-word compound
    nouns that are themselves common (e.g. "coffee table", "light switch")
    would ideally be preserved by a real POS tagger/taxonomy — this keeps
    the last two tokens when there are 3+, since compound heads are
    usually the final one or two words, and just the last token otherwise.
    """
    tokens = [t for t in label.strip().lower().split() if t not in _STOPWORDS]
    if not tokens:
        return label.strip().lower()
    if len(tokens) == 1:
        return tokens[0]
    if len(tokens) == 2:
        return " ".join(tokens)
    return " ".join(tokens[-2:])


def merge_near_duplicates(
    labels: list[str],
    embeddings: list[np.ndarray],
    threshold: float = 0.9,
) -> dict[str, str]:
    """Given canonicalized labels + their NOVIC embeddings for all instances
    in one scene, merges labels whose embeddings are highly similar
    (same object type observed under slightly different decoded names).
    Returns {original_label: merged_label}.
    """
    n = len(labels)
    if n == 0:
        return {}

    mapping = {label: label for label in labels}
    embeds = np.stack(embeddings)
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    normed = embeds / np.clip(norms, 1e-8, None)
    sims = normed @ normed.T

    canonical_for_group: dict[int, str] = {}
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] != labels[j] and sims[i, j] >= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    for members in groups.values():
        group_labels = [labels[i] for i in members]
        canonical = min(group_labels, key=len)  # shortest label as the merged name
        for i in members:
            mapping[labels[i]] = canonical

    return mapping
