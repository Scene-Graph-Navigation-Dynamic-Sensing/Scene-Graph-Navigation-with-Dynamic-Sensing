import numpy as np

from semantic_decoding.label_mapper.label_mapper import canonicalize, merge_near_duplicates


def test_canonicalize_strips_leading_modifiers():
    assert canonicalize("brown wooden office chair") == "office chair"
    assert canonicalize("chair") == "chair"
    assert canonicalize("red cup") == "red cup"
    assert canonicalize("the ceramic coffee mug") == "coffee mug"


def test_merge_near_duplicates_merges_similar_embeddings():
    rng = np.random.default_rng(0)
    base = rng.normal(size=64)
    embeddings = [base, base + rng.normal(size=64) * 1e-4, rng.normal(size=64) * 10]
    labels = ["chair", "office chair", "lamp"]

    mapping = merge_near_duplicates(labels, embeddings, threshold=0.99)

    assert mapping["chair"] == mapping["office chair"]
    assert mapping["lamp"] != mapping["chair"]


def test_merge_near_duplicates_empty_input():
    assert merge_near_duplicates([], []) == {}
