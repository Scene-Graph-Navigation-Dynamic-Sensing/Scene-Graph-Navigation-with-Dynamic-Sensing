import numpy as np

from task_planning.clip_tool import make_clip_similarity_tool


class FakeDecoder:
    def __init__(self):
        self.embed_text_calls = 0

    def embed_text(self, texts):
        self.embed_text_calls += 1
        # deterministic: "red" queries point along +x, everything else along +y
        vec = np.array([1.0, 0.0, 0.0]) if "red" in texts[0].lower() else np.array([0.0, 1.0, 0.0])
        return np.stack([vec])


def _fixture():
    node_embeddings = {
        "mug_2": np.array([1.0, 0.0, 0.0]),   # matches "red" queries well
        "mug_5": np.array([0.0, 1.0, 0.0]),   # matches "red" queries poorly
    }
    decoder = FakeDecoder()
    tool = make_clip_similarity_tool(node_embeddings, lambda: decoder)
    return tool, decoder


def test_cosine_similarity_ranks_the_better_match_first():
    tool, _ = _fixture()
    result = tool(["mug_2", "mug_5"], "the red mug")
    scores = result["scores"]
    assert list(scores.keys())[0] == "mug_2"
    assert scores["mug_2"] > scores["mug_5"]
    assert abs(scores["mug_2"] - 1.0) < 1e-6
    assert abs(scores["mug_5"] - 0.0) < 1e-6


def test_unknown_candidate_id_gets_null_score_not_a_crash():
    tool, _ = _fixture()
    result = tool(["mug_2", "nonexistent_99"], "the red mug")
    assert result["scores"]["nonexistent_99"] is None
    assert result["scores"]["mug_2"] is not None


def test_decoder_factory_called_at_most_once():
    node_embeddings = {"mug_2": np.array([1.0, 0.0, 0.0])}
    decoder = FakeDecoder()
    calls = {"n": 0}

    def factory():
        calls["n"] += 1
        return decoder

    tool = make_clip_similarity_tool(node_embeddings, factory)
    tool(["mug_2"], "red")
    tool(["mug_2"], "blue")
    assert calls["n"] == 1  # lazy-loaded once, reused after
    assert decoder.embed_text_calls == 2  # but embed_text runs each call
