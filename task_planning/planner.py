"""Orchestrates a single task-planning query: load a saved scene graph, ask
Gemini which receptacle a human command targets, and resolve that into a
concrete navigation point using the centroids persisted by
scene_graph/storage/serializer.py.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from scene_graph.storage.serializer import load_scene_graph
from task_planning.clip_tool import make_clip_similarity_tool
from task_planning.gemini_client import GeminiClient
from task_planning.prompts import SYSTEM_PROMPT, USER_TEMPLATE
from task_planning.scene_memory import build_room_json

logger = logging.getLogger(__name__)

# Gemini's exact echo format varies in practice (seen both, not hypothetical):
#   default_api.tool_navigate_to_receptacle("forklift_0")   -- namespaced, positional
#   tool_navigate_to_receptacle(receptacle_id='forklift_0')  -- bare, keyword arg
# so both an optional dotted namespace prefix and an optional `receptacle_id=`
# keyword before the quoted value are tolerated.
_CALL_RE = re.compile(
    r'CALL:\s*(?:[\w.]+\.)?tool_navigate_to_receptacle\(\s*(?:receptacle_id\s*=\s*)?["\']([^"\']+)["\']\s*\)'
)
_THOUGHT_RE = re.compile(r'THOUGHT:\s*(.*?)(?:\nCALL:|\Z)', re.DOTALL)


@dataclass
class TaskPlan:
    command: str
    room: str
    receptacle_id: str
    receptacle_class: str
    navigation_target: list[float] | None
    thought: str
    raw_response: str

    def to_dict(self) -> dict:
        return asdict(self)


def parse_response(response: str, valid_receptacle_ids) -> tuple[str, str]:
    """Returns (thought, receptacle_id). Raises ValueError if the response
    doesn't match the expected format or names a receptacle_id that isn't
    actually in the graph (a hallucinated id) -- never silently proceeds."""
    call_match = _CALL_RE.search(response)
    if not call_match:
        raise ValueError(f"Could not find a CALL: tool_navigate_to_receptacle(...) line in response: {response!r}")
    receptacle_id = call_match.group(1)

    if receptacle_id not in valid_receptacle_ids:
        raise ValueError(
            f"Gemini returned an unknown receptacle_id {receptacle_id!r} -- "
            f"not one of the graph's actual node ids."
        )

    thought_match = _THOUGHT_RE.search(response)
    thought = thought_match.group(1).strip() if thought_match else ""

    return thought, receptacle_id


def plan(
    scene_graph_path: Path,
    command: str,
    gemini_client: GeminiClient,
    semantic_decoding_cfg: dict | None = None,
) -> TaskPlan:
    """semantic_decoding_cfg: the pipeline config's `semantic_decoding:`
    section (same NOVIC checkpoint used elsewhere) -- only used to lazily
    build a NOVICDecoder if Gemini actually calls the get_clip_similarity
    tool (see task_planning/clip_tool.py); a query that never needs visual
    disambiguation never loads it. Pass None to disable the tool entirely.
    """
    data = load_scene_graph(scene_graph_path)
    scene_graph = data["scene_graph"]
    node_centroids = data.get("node_centroids", {})
    node_embeddings = data.get("node_embeddings", {})

    room_json = build_room_json(scene_graph)
    user_prompt = USER_TEMPLATE.format(user_request=command, room_json=json.dumps(room_json, indent=2))

    tools = None
    if semantic_decoding_cfg is not None:
        def _build_novic_decoder():
            from semantic_decoding.novic.novic_decoder import NOVICDecoder
            return NOVICDecoder(
                novic_dir=semantic_decoding_cfg["novic_dir"],
                checkpoint_path=semantic_decoding_cfg["novic_checkpoint"],
                gencfg=semantic_decoding_cfg["gencfg"],
                device=semantic_decoding_cfg.get("device", "cuda"),
            )
        tools = [make_clip_similarity_tool(node_embeddings, _build_novic_decoder)]

    response = gemini_client.generate(SYSTEM_PROMPT, user_prompt, tools=tools)
    thought, receptacle_id = parse_response(response, scene_graph.object_nodes.keys())

    centroid = node_centroids.get(receptacle_id)
    if centroid is None:
        logger.warning(
            f"No centroid stored for {receptacle_id!r} (scene graph saved before centroid "
            "persistence, or this node has no matching fused instance) -- navigation_target will be null."
        )

    return TaskPlan(
        command=command,
        room=Path(scene_graph_path).stem,
        receptacle_id=receptacle_id,
        receptacle_class=scene_graph.object_nodes[receptacle_id].node_class,
        navigation_target=centroid,
        thought=thought,
        raw_response=response,
    )
