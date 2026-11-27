"""A callable tool for the task-planning agent: given candidate object/
receptacle ids and a text query, embeds the query with NOVIC's text encoder
and returns cosine similarity against each candidate's stored visual
embedding, computed locally (never sending raw embeddings to Gemini).

google-genai's automatic function calling introspects a plain Python
callable's signature and docstring to build the tool schema the model sees
(GenerateContentConfig.tools accepts callables directly) -- both matter here,
not just for humans reading this file.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


def make_clip_similarity_tool(
    node_embeddings: dict[str, np.ndarray],
    novic_decoder_factory: Callable[[], object],
) -> Callable[[list[str], str], dict]:
    """node_embeddings: {node_id: embedding} from the loaded scene graph's
    pickle (scene_graph/storage/serializer.py). novic_decoder_factory: a
    zero-arg callable returning a NOVICDecoder -- called at most once, the
    first time the returned tool is actually invoked, so a query that never
    needs disambiguation never pays NOVIC's checkpoint-load cost.
    """
    decoder_holder: dict[str, object] = {}

    def get_clip_similarity(candidate_ids: list[str], query: str) -> dict:
        """Returns cosine similarity between `query` and each candidate object
        or receptacle's stored visual embedding. Use this to pick between
        multiple objects of the same class when the request has a
        distinguishing visual detail (colour, material, size, etc.) that the
        class name alone can't resolve. candidate_ids must be ids already
        present in the room JSON. An id with no stored embedding gets a null
        score instead of an error.
        """
        if "decoder" not in decoder_holder:
            decoder_holder["decoder"] = novic_decoder_factory()
        decoder = decoder_holder["decoder"]

        query_emb = decoder.embed_text([query])[0]
        q_norm = np.linalg.norm(query_emb)

        scores: dict[str, float | None] = {}
        for cid in candidate_ids:
            emb = node_embeddings.get(cid)
            if emb is None:
                scores[cid] = None
                continue
            scores[cid] = float(np.dot(query_emb, emb) / (q_norm * np.linalg.norm(emb) + 1e-8))

        ranked = dict(sorted(scores.items(), key=lambda kv: (kv[1] is None, -(kv[1] or 0.0))))
        logger.info(f"[tool] get_clip_similarity({candidate_ids!r}, {query!r}) -> {ranked}")
        return {"scores": ranked}

    return get_clip_similarity
