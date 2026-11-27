"""
 This file implements GridPlanner as a wrapper for AStar planner
    in USA-Net (https://github.com/codekansas/usa) project
 Most codes are adapted from:
    1. https://github.com/codekansas/usa/blob/master/usa/planners/clip_sdf.py

 License:
 MIT License

 Copyright (c) 2023 Ben Bolte

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

---
Ported from Scene-Graph-Generation/dovsg/navigation/astar_planner.py (DoVSG),
itself adapted from usa-net as above. Deltas applied during this port:
  - dropped an unused `from torch.utils.data import Dataset` import (never
    referenced anywhere in the original file) -- navigation/ has no torch
    dependency as a result.
  - dropped AStarPlanner's `memory_dir` constructor param: verified unused
    inside this file itself (only the outer facade used its own memory_dir,
    for saving navigation_vis.jpg -- that responsibility now lives in
    navigation/core/path_facade.py's `output_dir`).
  - fixed compute_theta's angle-wraparound: the original's second guard was
    `if theta < np.pi:` (line ~399), which is true for nearly *any* theta
    after the first guard already applied `-= 2*pi`, so it re-added 2*pi and
    silently undid the wrap in most cases. Correct guard is `< -np.pi`.
"""
from __future__ import annotations

import heapq
import math
from typing import Literal

import numpy as np

from navigation.core.grid_map import Map

Heuristic = Literal["manhattan", "euclidean", "octile", "chebyshev"]


def neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    return [(pt[0] + dx, pt[1] + dy) for dx in range(-1, 2) for dy in range(-1, 2) if (dx, dy) != (0, 0)]


class AStar:
    def __init__(
        self,
        is_occ: np.ndarray,
        origin: tuple[float, float],
        resolution: float,
        heuristic: Heuristic = "euclidean",
    ) -> None:
        self.heuristic = heuristic
        self.is_occ = is_occ
        self.origin = origin
        self.resolution = resolution

    def point_is_occupied(self, x: int, y: int) -> bool:
        occ_map = self.is_occ
        if x < 0 or y < 0 or x >= occ_map.shape[1] or y >= occ_map.shape[0]:
            return True
        return bool(occ_map[y][x])

    def xy_is_occupied(self, x: float, y: float) -> bool:
        return self.point_is_occupied(*self.to_pt((x, y)))

    def to_pt(self, xy: tuple[float, float]) -> tuple[int, int]:
        return self.get_map().to_pt(xy)

    def to_xy(self, pt: tuple[int, int]) -> tuple[float, float]:
        return self.get_map().to_xy(pt)

    def get_map(self) -> Map:
        return Map(grid=self.is_occ, resolution=self.resolution, origin=self.origin)

    def compute_dis(self, a: tuple[int, int], b: tuple[int, int]):
        if self.heuristic == "manhattan":
            dis = abs(a[0] - b[0]) + abs(a[1] - b[1])
        if self.heuristic == "euclidean":
            dis = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
        if self.heuristic == "octile":
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            dis = (dx + dy) + (1 - 2) * min(dx, dy)
        if self.heuristic == "chebyshev":
            dis = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
        return dis

    def compute_obstacle_punishment(self, a: tuple[int, int], weight: int, avoid: int) -> float:
        obstacle_punishment = 0
        # Checking a (2*avoid) x (2*avoid) square centered at a keeps this O(1)
        # instead of computing distance to every obstacle in the map.
        for i in range(-avoid, avoid + 1):
            for j in range(-avoid, avoid + 1):
                if self.point_is_occupied(a[0] + i, a[1] + j):
                    b = [a[0] + i, a[1] + j]
                    if self.heuristic == "manhattan":
                        obs_dis = abs(a[0] - b[0]) + abs(a[1] - b[1])
                    if self.heuristic == "euclidean":
                        obs_dis = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                    if self.heuristic == "octile":
                        dx = abs(a[0] - b[0])
                        dy = abs(a[1] - b[1])
                        obs_dis = (dx + dy) + (1 - 2) * min(dx, dy)
                    if self.heuristic == "chebyshev":
                        obs_dis = max(abs(a[0] - b[0]), abs(a[1] - b[1]))
                    obstacle_punishment = max((weight / max(obs_dis, 1)), obstacle_punishment)
        return obstacle_punishment

    def compute_s1(self, a: tuple[int, int], obj: tuple[int, int]) -> float:
        return self.compute_dis(a, obj)

    def compute_s2(self, a: tuple[int, int], obj: tuple[int, int], weight=8, ideal_dis=4) -> float:
        return weight * (4 - min(self.compute_dis(a, obj), 4))

    def compute_s3(self, a: tuple[int, int], weight=8, avoid=1) -> float:
        return self.compute_obstacle_punishment(a, weight, avoid)

    # A* heuristic: distance-to-target + a penalty for hugging obstacles,
    # applied both to the candidate cell and (weighted down) to the target.
    def compute_heuristic(self, a: tuple[int, int], b: tuple[int, int], weight=12, avoid=3) -> float:
        return (
            self.compute_dis(a, b)
            + weight * self.compute_obstacle_punishment(a, weight, avoid)
            + self.compute_obstacle_punishment(b, weight, avoid)
        )

    def is_in_line_of_sight(self, start_pt: tuple[int, int], end_pt: tuple[int, int]) -> bool:
        """Bresenham line-of-sight check between two grid points."""
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]

        if abs(dx) > abs(dy):
            if dx < 0:
                start_pt, end_pt = end_pt, start_pt
            for x in range(start_pt[0], end_pt[0] + 1):
                yf = start_pt[1] + (x - start_pt[0]) / dx * dy
                for y in list({math.floor(yf), math.ceil(yf)}):
                    if self.point_is_occupied(x, y):
                        return False
        else:
            if dy < 0:
                start_pt, end_pt = end_pt, start_pt
            for y in range(start_pt[1], end_pt[1] + 1):
                xf = start_pt[0] + (y - start_pt[1]) / dy * dx
                for x in list({math.floor(xf), math.ceil(xf)}):
                    if self.point_is_occupied(x, y):
                        return False
        return True

    def clean_path(self, path: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Simplifies a raw grid path: repeatedly jump to the farthest
        still-in-line-of-sight point, instead of keeping every grid step."""
        cleaned_path = [path[0]]
        i = 0
        while i < len(path) - 1:
            for j in range(len(path) - 1, i, -1):
                if self.is_in_line_of_sight(path[i], path[j]):
                    break
            else:
                j = i + 1
            if j - i >= 2:
                cleaned_path.append(path[(i + j) // 2])
            cleaned_path.append(path[j])
            i = j
        return cleaned_path

    def get_unoccupied_neighbor(self, pt: tuple[int, int], goal_pt=None) -> tuple[int, int]:
        if not self.point_is_occupied(*pt):
            return pt

        # A point exactly on an obstacle's edge (e.g. the start point) is
        # snapped to the nearest free neighbor instead of failing outright.
        neighbor_pts = neighbors(pt)
        if goal_pt is not None:
            goal_pt_non_null = goal_pt
            neighbor_pts.sort(key=lambda n: self.compute_heuristic(n, goal_pt_non_null))
        for neighbor_pt in neighbor_pts:
            if not self.point_is_occupied(*neighbor_pt):
                return neighbor_pt
        raise ValueError("Start/goal point sits on a non-navigable cell with no free neighbor -- check the occupancy map / min_height / max_height settings.")

    def get_reachable_points(self, start_pt: tuple[int, int]) -> set[tuple[int, int]]:
        """Flood-fill of every free cell reachable from start_pt."""
        start_pt = self.get_unoccupied_neighbor(start_pt)

        reachable_points: set[tuple[int, int]] = set()
        to_visit = [start_pt]
        while to_visit:
            pt = to_visit.pop()
            if pt in reachable_points:
                continue
            reachable_points.add(pt)
            for new_pt in neighbors(pt):
                if new_pt in reachable_points:
                    continue
                if self.point_is_occupied(new_pt[0], new_pt[1]):
                    continue
                to_visit.append(new_pt)
        return reachable_points

    def run_astar(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        remove_line_of_sight_points: bool = True,
    ) -> list[tuple[float, float]]:
        start_pt, end_pt = self.to_pt(start_xy), self.to_pt(end_xy)

        start_pt = self.get_unoccupied_neighbor(start_pt)
        end_pt = self.get_unoccupied_neighbor(end_pt, start_pt)

        q = [(0, start_pt)]
        came_from: dict = {start_pt: None}
        cost_so_far: dict[tuple[int, int], float] = {start_pt: 0.0}
        while q:
            _, current = heapq.heappop(q)

            if current == end_pt:
                break

            for nxt in neighbors(current):
                if self.point_is_occupied(nxt[0], nxt[1]):
                    continue
                new_cost = cost_so_far[current] + self.compute_heuristic(current, nxt)
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + self.compute_heuristic(end_pt, nxt)
                    heapq.heappush(q, (priority, nxt))
                    came_from[nxt] = current
        else:
            raise ValueError("No path found")

        path = []
        current = end_pt
        while current != start_pt:
            path.append(current)
            prev = came_from[current]
            if prev is None:
                break
            current = prev
        path.append(start_pt)
        path.reverse()

        if remove_line_of_sight_points:
            path = self.clean_path(path)

        return [start_xy] + [self.to_xy(pt) for pt in path[1:]]

    def is_valid_starting_point(self, xy: tuple[float, float]) -> bool:
        return not self.point_is_occupied(*self.to_pt(xy))


class AStarPlanner:
    def __init__(
        self,
        occ_map: Map,
        resolution: float = 0.1,
        occ_avoid_radius: float = 0.2,
        conservative: bool = True,
        heuristic: Heuristic = "euclidean",
    ):
        self.occ_avoid_radius = occ_avoid_radius
        self.resolution = resolution
        self.occ_avoid = int(np.ceil(self.occ_avoid_radius / self.resolution))
        self.conservative = conservative
        self.heuristic = heuristic

        self.occ_map = occ_map

        is_occ = np.expand_dims(self.occ_map.grid, axis=-1)

        self.astar = AStar(
            is_occ=is_occ,
            origin=self.occ_map.origin,
            resolution=self.resolution,
            heuristic=self.heuristic,
        )

    def plan(
        self,
        start_xy: tuple[float, float],
        end_xy: tuple[float, float],
        remove_line_of_sight_points: bool = True,
    ) -> list[tuple[float, float, float]]:
        end_xy, end_theta = self.get_end_xy(start_xy, end_xy)
        waypoints = self.astar.run_astar(
            start_xy=start_xy,
            end_xy=end_xy,
            remove_line_of_sight_points=remove_line_of_sight_points,
        )
        xyt_points = []
        for i in range(len(waypoints) - 1):
            theta = self.compute_theta(waypoints[i][0], waypoints[i][1], waypoints[i + 1][0], waypoints[i + 1][1])
            xyt_points.append((waypoints[i][0], waypoints[i][1], float(theta)))
        xyt_points.append((waypoints[-1][0], waypoints[-1][1], end_theta))
        return xyt_points

    def get_end_xy(self, start_xy: tuple[float, float], end_xy: tuple[float, float]):
        """The queried goal (e.g. an object's centroid) usually sits inside
        an obstacle, so this snaps it to the best reachable free cell near
        it: close to the goal, at a ~0.4m standoff, away from other
        obstacles. 0.3/0.4 are the standoff constants from the OK-Robot
        paper, same as upstream."""
        start_pt = self.astar.to_pt(start_xy)
        reachable_pts = self.astar.get_reachable_points(start_pt)
        reachable_pts = list(reachable_pts)
        end_pt = self.astar.to_pt(end_xy)
        avoid = math.ceil((0.3 - self.occ_avoid_radius) / self.resolution)
        ideal_dis = math.floor(0.4 / self.resolution)
        inds = np.array([
            self.astar.compute_s1(end_pt, reachable_pt)
            + self.astar.compute_s2(end_pt, reachable_pt, weight=8, ideal_dis=ideal_dis)
            + self.astar.compute_s3(reachable_pt, weight=8, avoid=avoid)
            for reachable_pt in reachable_pts
        ])
        ind = np.argmin(inds)
        end_pt = reachable_pts[ind]
        x, y = self.astar.to_xy(end_pt)

        theta = self.compute_theta(x, y, end_xy[0], end_xy[1])

        return (float(x), float(y)), float(theta)

    def is_valid_starting_point(self, xy: tuple[float, float]) -> bool:
        return self.astar.is_valid_starting_point(xy)

    def compute_theta(self, cur_x, cur_y, end_x, end_y):
        theta = 0
        if end_x == cur_x and end_y >= cur_y:
            theta = np.pi / 2
        elif end_x == cur_x and end_y < cur_y:
            theta = -np.pi / 2
        else:
            theta = np.arctan((end_y - cur_y) / (end_x - cur_x))
            if end_x < cur_x:
                theta = theta + np.pi
            # Wrap into [-pi, pi]. Bugfixed from upstream: the second guard
            # was `< np.pi`, which is true for almost any theta once the
            # first guard has already subtracted 2*pi, silently re-adding it
            # right back in most cases.
            if theta > np.pi:
                theta = theta - 2 * np.pi
            if theta < -np.pi:
                theta = theta + 2 * np.pi
        return theta
