from task_planning.scene_memory import build_room_json


class FakeNode:
    def __init__(self, node_id, node_class, children=None):
        self.node_id = node_id
        self.node_class = node_class
        self.children = children or {}


class FakeSceneGraph:
    def __init__(self, root):
        self.root = root


def test_leaf_nodes_are_not_receptacles():
    mouse = FakeNode("computer_mouse_0", "computer mouse")
    floor = FakeNode("floor_0", "floor", {"computer_mouse_0": mouse})
    room = build_room_json(FakeSceneGraph(floor))

    assert "floor_0" in room
    assert "computer_mouse_0" not in room  # no children -> not a receptacle
    assert room["floor_0"]["contains"] == [{"id": "computer_mouse_0", "class": "computer mouse"}]


def test_only_direct_children_go_in_contains():
    book = FakeNode("book_0", "book")
    shelf = FakeNode("shelf_0", "shelf", {"book_0": book})
    desk = FakeNode("desk_0", "desk", {"shelf_0": shelf})
    floor = FakeNode("floor_0", "floor", {"desk_0": desk})
    room = build_room_json(FakeSceneGraph(floor))

    # desk's contains lists shelf_0 directly, not book_0 (a grandchild)
    assert room["desk_0"]["contains"] == [{"id": "shelf_0", "class": "shelf"}]
    # shelf gets its own top-level entry since it has children too
    assert room["shelf_0"]["contains"] == [{"id": "book_0", "class": "book"}]


def test_empty_scene_has_no_receptacles_besides_floor_if_it_has_children():
    floor = FakeNode("floor_0", "floor")  # no children at all
    room = build_room_json(FakeSceneGraph(floor))
    assert room == {}
