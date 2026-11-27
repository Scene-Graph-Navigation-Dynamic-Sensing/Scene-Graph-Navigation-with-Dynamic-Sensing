"""A relation must be measured, never assumed.

The support-resolution loop deadlocks when two objects each resolve to the
other. Breaking that tie must not invent contact: only the object physically
able to hold the other up (the lower one) may become a parent, and anything
still unresolved is attached to the floor with NO relation, so a guess can
never be mistaken for a measured "on".
"""
import yaml
from pathlib import Path

from scene_graph.builder.graph_builder import _CycleBreakingNodes, _new_scene_graph
from scene_graph.storage.serializer import scene_graph_to_relations

REPO_ROOT = Path(__file__).resolve().parents[2]
SGG_ROOT = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())["external"][
    "scene_graph_generation_root"]


def _nodes(bottom_z, patience=3):
    graph = _new_scene_graph(SGG_ROOT, cycle_patience=patience, bottom_z=bottom_z)
    return graph, graph.object_nodes


def test_present_nodes_are_returned_normally():
    graph, nodes = _nodes({"a_0": 0.0})
    graph.add_node(parent=graph.root, node_class="a", node_id="a_0", parent_relation="on")
    assert "a_0" in nodes and nodes["a_0"].node_id == "a_0"


def test_missing_node_is_deferred_until_patience_runs_out():
    _, nodes = _nodes({"table_0": 0.0, "cup_0": 0.8}, patience=3)
    assert "table_0" not in nodes          # miss 1 -> defer, as upstream expects
    assert "table_0" not in nodes          # miss 2 -> defer
    assert "table_0" in nodes              # miss 3 -> patience exhausted, break the cycle


def test_cycle_is_broken_in_favour_of_the_lower_object():
    # table (base 0.0) and cup (base 0.8) each resolve to the other
    graph, nodes = _nodes({"table_0": 0.0, "cup_0": 0.8}, patience=2)
    for _ in range(2):
        "table_0" in nodes
        "cup_0" in nodes
    parent = nodes["table_0"]              # the lower one may support
    assert parent.node_id == "table_0"
    assert parent.parent is graph.root
    assert parent.parent_relation is None  # its own support was never measured


def test_unresolvable_support_claims_no_relation():
    graph, nodes = _nodes({"mouse_0": 0.75, "keyboard_0": 0.75}, patience=1)
    # ask for the higher/equal object: it cannot be a supporter, so the
    # dependent is parked under the marker node instead of claiming "on"
    holder = nodes["keyboard_0"]
    graph.add_node(parent=holder, node_class="mouse", node_id="mouse_0", parent_relation="on")

    lifted = nodes.resolve_unverified_parents()
    assert lifted == ["mouse_0"]
    relations = scene_graph_to_relations(graph)
    assert [(r["child"], r["relation"]) for r in relations] == [("mouse_0", None)]
    assert all(r["parent"] == "floor_0" for r in relations)


def test_no_relation_is_ever_invented_as_on():
    graph, nodes = _nodes({"x_0": 1.0}, patience=1)
    holder = nodes["x_0"]
    graph.add_node(parent=holder, node_class="y", node_id="y_0", parent_relation="on")
    nodes.resolve_unverified_parents()
    assert all(r["relation"] in (None, "on", "inside", "belongs")
               for r in scene_graph_to_relations(graph))
    assert scene_graph_to_relations(graph)[0]["relation"] is None
