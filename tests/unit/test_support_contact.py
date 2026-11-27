"""An object is only "on" something whose top meets its base."""
from shapely.geometry import box

from scene_graph.builder.dovsg_relations import _find_support


def _info(class_id, lower_z, upper_z, footprint=box(0, 0, 1, 1)):
    return {"ins_obj": {"class_id": class_id}, "lower_z": lower_z, "base_z": lower_z,
            "upper_z": upper_z, "alpha_shape": footprint}


def _support(child, *others):
    mapping = {i["ins_obj"]["class_id"]: i for i in (child, *others)}
    found, _ = _find_support(child["ins_obj"]["class_id"], child, mapping, 0.1, 0.10, 0.15)
    return None if found is None else found["class_id"]


def test_object_resting_on_a_desk_top_is_on_it():
    assert _support(_info("laptop", 0.81, 1.0), _info("desk", 0.0, 0.90)) == "desk"


def test_object_floating_above_a_low_object_is_not_on_it():
    # a chair's reconstructed part 22 cm above a flip flop on the floor
    assert _support(_info("chair", 0.38, 1.0), _info("flip flop", 0.0, 0.16)) is None


def test_object_sunk_into_a_taller_one_is_not_on_it():
    # a laptop whose base is 22 cm below the other object's top
    assert _support(_info("laptop", 0.83, 1.0), _info("person", 0.38, 1.05)) is None


def test_the_support_whose_top_meets_the_base_wins():
    assert _support(_info("mug", 0.91, 1.0), _info("rug", 0.0, 0.02), _info("table", 0.0, 0.90)) == "table"


def test_reports_whether_anything_lies_beneath():
    chair = _info("chair", 0.38, 1.0)
    mapping = {"chair": chair, "flip flop": _info("flip flop", 0.0, 0.16)}
    assert _find_support("chair", chair, mapping, 0.1, 0.10, 0.15) == (None, True)
    desk = _info("desk", 0.45, 0.9)
    assert _find_support("desk", desk, {"desk": desk}, 0.1, 0.10, 0.15) == (None, False)
