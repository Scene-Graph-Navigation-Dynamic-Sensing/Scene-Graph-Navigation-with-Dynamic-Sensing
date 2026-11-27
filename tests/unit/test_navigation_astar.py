import numpy as np

from navigation.core.astar import AStar, AStarPlanner
from navigation.core.grid_map import Map


def _open_grid(size=10):
    return np.zeros((size, size), dtype=bool)


def test_run_astar_routes_around_a_wall():
    grid = _open_grid(10)
    grid[5, 0:8] = True  # a wall across most of the grid, gap at x=8,9
    astar = AStar(is_occ=grid, origin=(0.0, 0.0), resolution=1.0, heuristic="euclidean")
    path = astar.run_astar(start_xy=(1.0, 1.0), end_xy=(1.0, 8.0), remove_line_of_sight_points=False)
    # every waypoint must land on a free cell
    for xy in path:
        pt = astar.to_pt(xy)
        assert not astar.point_is_occupied(*pt)
    # the path must actually detour through the gap (some waypoint at x>=8)
    assert any(astar.to_pt(xy)[0] >= 8 for xy in path)


def test_clean_path_simplifies_when_line_of_sight_is_clear():
    grid = _open_grid(10)
    astar = AStar(is_occ=grid, origin=(0.0, 0.0), resolution=1.0)
    staircase = [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2), (3, 3)]
    cleaned = astar.clean_path(staircase)
    assert len(cleaned) < len(staircase)
    assert cleaned[0] == staircase[0]
    assert cleaned[-1] == staircase[-1]


def test_get_reachable_points_excludes_isolated_island():
    grid = _open_grid(6)
    grid[:, 3] = True  # a wall splitting the grid into two halves, no gap
    astar = AStar(is_occ=grid, origin=(0.0, 0.0), resolution=1.0)
    reachable = astar.get_reachable_points((0, 0))
    assert (0, 0) in reachable
    assert (5, 5) not in reachable  # other side of the wall, unreachable


def test_compute_theta_axis_aligned_cases():
    planner = AStarPlanner(
        occ_map=Map(grid=_open_grid(5), resolution=1.0, origin=(0.0, 0.0)),
        resolution=1.0,
    )
    assert planner.compute_theta(0, 0, 1, 0) == 0
    assert np.isclose(planner.compute_theta(0, 0, 0, 1), np.pi / 2)
    assert np.isclose(planner.compute_theta(0, 0, 0, -1), -np.pi / 2)
    assert np.isclose(abs(planner.compute_theta(0, 0, -1, 0)), np.pi)


def test_compute_theta_does_not_double_wrap():
    # Regression test for the upstream wraparound bug: the second guard used
    # to be `theta < np.pi`, which is true for almost any already-valid
    # theta (e.g. a plain first-quadrant direction), so it unconditionally
    # re-added 2*pi and pushed the result outside [-pi, pi].
    planner = AStarPlanner(
        occ_map=Map(grid=_open_grid(5), resolution=1.0, origin=(0.0, 0.0)),
        resolution=1.0,
    )
    theta = planner.compute_theta(0, 0, 1, 1)  # simple 45-degree direction
    assert -np.pi <= theta <= np.pi
    assert np.isclose(theta, np.pi / 4)
