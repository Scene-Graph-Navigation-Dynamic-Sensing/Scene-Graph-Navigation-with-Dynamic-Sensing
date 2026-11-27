"""Builds the "Room JSON" (task_planning/prompts.py's expected input shape)
from a loaded scene graph: every node with children becomes a receptacle
entry keyed by node_id, listing its DIRECT contents only. A receptacle
nested inside another (e.g. a shelf on a desk) still gets its own top-level
entry, so the LLM can look further into it if needed.
"""
from __future__ import annotations


def build_room_json(scene_graph) -> dict:
    room: dict[str, dict] = {}

    def visit(node) -> None:
        if node.children:
            room[node.node_id] = {
                "class": node.node_class,
                "contains": [
                    {"id": child.node_id, "class": child.node_class}
                    for child in node.children.values()
                ],
            }
        for child in node.children.values():
            visit(child)

    visit(scene_graph.root)
    return room
