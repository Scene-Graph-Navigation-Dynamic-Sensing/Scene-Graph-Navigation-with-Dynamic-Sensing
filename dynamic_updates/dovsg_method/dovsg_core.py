"""DoVSG's scene-update functions, ported from external/DovSG/dovsg/controller.py.

DoVSG's `dovsg` package cannot be imported next to Scene-Graph-Generation's
(both are called `dovsg`), and its Controller needs a robot socket. So the
functions are copied here with the same names, thresholds and order. Each
one cites the lines it comes from; where something differs, the comment
says what and why.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ─── LightGlue: dovsg/scripts/rgb_feature_match.py (RGBFeatureMatch) ───

class RGBFeatureMatch:
    """SuperPoint (1024 keypoints) + LightGlue, as DoVSG uses them to find the
    memory image most similar to a new frame (score = sum of match scores).
    LightGlue is loaded from DoVSG's third_party checkout."""

    def __init__(self, lightglue_dir: str, max_num_keypoints: int = 1024, device: str = "cuda"):
        import torch
        from scripts.compare_video_poses import _load_lightglue

        SuperPoint, LightGlue = _load_lightglue(lightglue_dir)
        from lightglue.utils import numpy_image_to_torch, rbd

        self.torch, self.rbd, self.to_torch = torch, rbd, numpy_image_to_torch
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def extract_feature(self, image: np.ndarray):
        with self.torch.no_grad():
            return self.extractor.extract(self.to_torch(image).to(self.device))

    def match_features(self, f0, f1):
        with self.torch.no_grad():
            m = self.matcher({"image0": f0, "image1": f1})
        return [self.rbd(x) for x in (f0, f1, m)]

    def find_most_similar_image(self, image: np.ndarray, features: list) -> dict:
        """rgb_feature_match.py find_most_similar_image(): the memory frame with
        the highest summed match score. Also returns that pair's matched
        keypoints, which DoVSG does not keep (used for plots and --estimate_scale)."""
        f = self.extract_feature(image)
        best = None
        for index, fm in enumerate(features):
            r0, r1, m = self.match_features(f, fm)
            score = float(m["scores"].sum().cpu())
            if best is None or score > best["score"]:
                pairs = m["matches"].cpu().numpy()
                best = {"index": index, "score": score, "matches": len(pairs),
                        "kp_new": r0["keypoints"].cpu().numpy()[pairs[:, 0]],
                        "kp_memory": r1["keypoints"].cpu().numpy()[pairs[:, 1]]}
        return best


# ─── Anchor: controller.py correct_pose_observations() L668-696 ───

def anchor_poses(observations: dict, best_matches_len_list: list) -> tuple[int, np.ndarray]:
    """The frame with the most LightGlue matches keeps its ACE pose; every
    other frame is placed from it through its own c2b: pose = b2w @ c2b.
    (DoVSG first scores 0.1 x matches + 0.9 x ACE inliers, then overwrites
    that with argmax(matches) on the next line; the second line is what runs.)"""
    keys = list(observations.keys())
    best = int(np.argmax(best_matches_len_list))
    b2w = observations[keys[best]]["pose"] @ np.linalg.inv(observations[keys[best]]["c2b"])
    for k in keys:
        if k != keys[best]:
            observations[k]["pose"] = b2w @ observations[k]["c2b"]
    return keys[best], b2w


# ─── Colored ICP: controller.py calculate_alignment_colored_icp() L613-666 ───

def calculate_alignment_colored_icp(source, target):
    """Multi-scale colored ICP (4, 2, 1 cm). A scale's result is kept only when
    at least 1/5 of the smaller cloud has a correspondence; success means the
    1 cm scale passed. Returns (is_success, transformation, per-scale log)."""
    voxel_radius = [0.04, 0.02, 0.01]
    max_iter = [50, 30, 14]
    is_success = False
    transformation = np.identity(4)
    log = []
    for idx in range(len(voxel_radius)):
        radius = voxel_radius[idx]
        source_down = source.voxel_down_sample(radius)
        target_down = target.voxel_down_sample(radius)
        try:
            source_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 3, max_nn=30))
            target_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 3, max_nn=30))
            result_icp = o3d.pipelines.registration.registration_colored_icp(
                source_down, target_down, radius, transformation,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(lambda_geometric=0.9999),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=max_iter[idx]))
            n_corr = len(result_icp.correspondence_set)
            needed = (1 / 5) * min(len(source_down.points), len(target_down.points))
            accepted = n_corr >= needed
            if accepted:
                transformation = result_icp.transformation
                if idx == len(voxel_radius) - 1:
                    is_success = True
            log.append({"radius_m": radius, "correspondences": n_corr, "needed": int(needed),
                        "accepted": bool(accepted), "fitness": round(result_icp.fitness, 4),
                        "inlier_rmse_mm": round(result_icp.inlier_rmse * 1000, 2)})
        except Exception as e:  # noqa: BLE001  (DoVSG catches and prints, then tries the next scale)
            log.append({"radius_m": radius, "error": str(e), "accepted": False})
    return is_success, np.asarray(transformation), log


# ─── Change detection: controller.py find_need_to_delete_indexes() L1219-1276 ───

def find_need_to_delete_indexes(voxel_indexes: np.ndarray, voxel_points: np.ndarray, voxel_colors: np.ndarray,
                                observations: dict, resolution: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Every memory voxel in front of a camera and inside its image is compared
    with that frame's depth at its pixel (against the nearest memory voxel on
    that pixel). It is deleted when the observed depth is more than
    2 x resolution (2 cm) behind it, or more than 1 x resolution (1 cm) behind
    with an RGB distance > 0.1. One frame is enough (the union over frames).
    Pixels without valid depth never delete. Returns (deleted indexes, per
    voxel the number of frames that saw through it; DoVSG keeps only the union)."""
    depth_thres = resolution * 2
    color_depth_thres = resolution
    color_thres = 0.1
    seen_through = np.zeros(len(voxel_indexes), np.int32)
    for obs in observations.values():
        pose, color, intrinsic = obs["pose"], obs["rgb"], obs["intrinsic"]
        height, width = color.shape[:2]
        pose_inv = np.linalg.inv(pose)
        points_camera = voxel_points @ pose_inv[:3, :3].T + pose_inv[:3, 3]
        mask = points_camera[:, 2] > 0
        pc_xy = np.zeros((points_camera.shape[0], 2), dtype=np.int64)  # float pixels truncate on assignment
        pc_xy[mask] = project_point_to_pixel(points_camera[mask], intrinsic)
        mask *= (pc_xy[:, 0] >= 0) * (pc_xy[:, 0] <= height - 1) * (pc_xy[:, 1] >= 0) * (pc_xy[:, 1] <= width - 1)
        depth = obs["depth"]
        depth_mask = obs["mask"]
        min_depth_map = np.full((height, width), np.inf)
        np.minimum.at(min_depth_map, (pc_xy[mask, 0], pc_xy[mask, 1]), points_camera[mask, 2])
        depth_differ = (depth - min_depth_map)[pc_xy[mask, 0], pc_xy[mask, 1]]
        color_differ = ((voxel_colors[mask] - color[pc_xy[mask, 0], pc_xy[mask, 1]]) ** 2).sum(1) ** 0.5
        delete_valid_mask = np.logical_or(depth_differ > depth_thres,
                                          (depth_differ > color_depth_thres) * (color_differ > color_thres))
        valid_depth_mask = depth_mask[pc_xy[mask, 0], pc_xy[mask, 1]]
        delete_valid_mask[~valid_depth_mask] = False  # invalid depth can't give reference
        seen_through[np.where(mask)[0][delete_valid_mask]] += 1
    return voxel_indexes[seen_through > 0], seen_through


def find_seen_through_voted(voxel_indexes: np.ndarray, voxel_points: np.ndarray, observations: dict,
                            rel_tol: float = 0.15, abs_tol: float = 0.02, vote_share: float = 0.7,
                            min_votes: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Not DoVSG: find_need_to_delete_indexes() for estimated (monocular) depth.
    Three changes:
      1. tolerance grows with distance: tau = max(abs_tol, rel_tol x the voxel's depth), not a fixed 2 cm;
      2. each voxel is judged by its own depth, not the nearest voxel on its pixel's;
      3. a voxel is deleted only when at least min_votes frames see through it and they are more than
         vote_share of the frames that see it (through or confirmed), not when any one frame does.
    A frame confirms a voxel when its depth is within tau; when the depth is in front of it by more
    than tau, the voxel is hidden and the frame does not vote. No colour test.
    Returns (deleted indexes, per voxel the through votes, per voxel the confirm votes)."""
    through = np.zeros(len(voxel_indexes), np.int32)
    confirm = np.zeros(len(voxel_indexes), np.int32)
    for obs in observations.values():
        pose, intrinsic = obs["pose"], obs["intrinsic"]
        height, width = obs["mask"].shape
        pose_inv = np.linalg.inv(pose)
        points_camera = voxel_points @ pose_inv[:3, :3].T + pose_inv[:3, 3]
        mask = points_camera[:, 2] > 0
        pc_xy = np.zeros((points_camera.shape[0], 2), dtype=np.int64)
        pc_xy[mask] = project_point_to_pixel(points_camera[mask], intrinsic)
        mask &= (pc_xy[:, 0] >= 0) & (pc_xy[:, 0] <= height - 1) & (pc_xy[:, 1] >= 0) & (pc_xy[:, 1] <= width - 1)
        ii = np.where(mask)[0]
        r, c = pc_xy[ii, 0], pc_xy[ii, 1]
        valid = obs["mask"][r, c]  # invalid depth can't give reference
        ii, r, c = ii[valid], r[valid], c[valid]
        depth = points_camera[ii, 2]
        differ = obs["depth"][r, c] - depth
        tau = np.maximum(abs_tol, rel_tol * depth)
        through[ii[differ > tau]] += 1
        confirm[ii[np.abs(differ) <= tau]] += 1
    deleted = (through >= min_votes) & (through > vote_share * (through + confirm))
    return voxel_indexes[deleted], through, confirm


def project_point_to_pixel(points, intrinsic):
    """controller.py L802-815: camera-frame points to (row, col) pixels."""
    pixels = cv2.projectPoints(np.asarray(points, dtype=np.float64), np.zeros(3), np.zeros(3),
                               np.asarray(intrinsic, dtype=np.float64), np.zeros(5))[0][:, 0, :]
    return pixels[:, ::-1]


# ─── Removal rule: controller.py update_instance_objects() L1318-1339 ───

def split_removed(objects: list, need_delete_indexes: np.ndarray, delete_rate: float = 0.5):
    """An object goes when more than delete_rate of its voxels were deleted;
    any other object loses its deleted voxels. Returns (survivors with
    trimmed indexes, removed class_ids, per object the deleted share)."""
    survivors, removed, share = [], [], {}
    for obj in objects:
        indexes = np.asarray(obj["indexes"])
        inter = np.intersect1d(indexes, need_delete_indexes)
        rate = len(inter) / max(len(indexes), 1)
        share[obj["class_id"]] = round(float(rate), 4)
        if rate > delete_rate:
            removed.append(obj["class_id"])
        else:
            kept = dict(obj)
            kept["indexes"] = list(np.setdiff1d(indexes, inter))
            survivors.append(kept)
    return survivors, removed, share


# ─── View dataset: controller.py update_view_dataset() L1278-1314 ───

def update_voxel_colors(memory_colors: dict, need_delete_indexes: np.ndarray, new_colors: dict) -> dict:
    """Deleted voxels leave the map, then the new frames' voxels are added
    (overwriting the colour of voxels both runs saw)."""
    deleted = set(need_delete_indexes.tolist())
    out = {ix: c for ix, c in memory_colors.items() if ix not in deleted}
    out.update(new_colors)
    return out
