import pickle

from navigation.planner import plan_navigation


def _base_cfg(**overrides):
    cfg = {
        "resolution": 0.1,
        "min_height": 0.2,
        "max_height": 1.4,
        "occ_threshold": 100,
        "conservative": False,
        "occ_avoid_radius": 0.2,
        "heuristic": "euclidean",
        "pointcloud_visualization": False,
    }
    cfg.update(overrides)
    return cfg


def _write_scene_graph_pkl(path, voxel_map_dict):
    with open(path, "wb") as f:
        pickle.dump({"voxel_map": voxel_map_dict}, f)


def _empty_room_voxel_map():
    # No observed points anywhere -- an "empty room" with conservative=False,
    # or (in the failure test below) a fully-unmapped/blocked grid with
    # conservative=True, since every cell then has zero observations.
    return {
        "resolution": 0.5,
        "bounds": (0.0, 5.0, 0.0, 5.0, 0.0, 2.0),
        "indexes_colors_mapping_dict": {},
    }


def test_clear_path_succeeds_with_waypoints_near_start_and_goal(tmp_path):
    pkl_path = tmp_path / "scene.pkl"
    _write_scene_graph_pkl(pkl_path, _empty_room_voxel_map())

    plan = plan_navigation(
        scene_graph_pkl_path=pkl_path,
        goal_xyz=[4.5, 4.5, 1.0],
        start_xyz=[0.5, 0.5, 1.0],
        navigation_cfg=_base_cfg(),  # conservative=False: unobserved space stays free
        room="test_room",
        visualize=False,
        output_dir=tmp_path / "out",
    )

    assert plan.success
    assert plan.waypoints
    assert abs(plan.waypoints[0][0] - 0.5) < 1.0
    assert abs(plan.waypoints[0][1] - 0.5) < 1.0
    # The goal is snapped to a standoff point near it (~0.4m, an intentional
    # "don't stand on top of the target" behavior -- see AStarPlanner.get_end_xy),
    # not the literal goal coordinate, so allow a bit of slack.
    last_x, last_y, _ = plan.waypoints[-1]
    assert ((last_x - 4.5) ** 2 + (last_y - 4.5) ** 2) ** 0.5 < 1.0


def test_start_with_no_free_neighbor_fails_gracefully(tmp_path):
    pkl_path = tmp_path / "scene.pkl"
    _write_scene_graph_pkl(pkl_path, _empty_room_voxel_map())

    plan = plan_navigation(
        scene_graph_pkl_path=pkl_path,
        goal_xyz=[4.5, 4.5, 1.0],
        start_xyz=[0.5, 0.5, 1.0],
        # conservative=True + zero observed points anywhere -> the entire
        # grid is "unmapped" and therefore treated as occupied, so the start
        # point has no free neighbor at all.
        navigation_cfg=_base_cfg(conservative=True),
        room="test_room",
        visualize=False,
        output_dir=tmp_path / "out",
    )

    assert not plan.success
    assert plan.waypoints is None
    assert plan.message
