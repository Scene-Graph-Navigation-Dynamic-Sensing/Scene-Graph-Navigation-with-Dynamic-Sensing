#!/usr/bin/env python3
"""Compare the camera poses of two captures of the same scene by registering
their voxel maps, and measure how well they agree.

Each run of run_static_scene_graph.py (--pose_only is enough) leaves a dataset
whose DROID-SLAM poses live in that capture's own world: the first camera's
frame. Two videos of one scene therefore have no common frame, and their poses
cannot be compared until one map is registered onto the other. This script
does that registration and reports how the camera moved between the videos.

Registration methods, all B (query) -> A (reference):

  baseline   FPFH + RANSAC on the voxel maps, then point-to-plane ICP. Rigid.
             Geometry only: on a scene of flat walls, a tabletop and a floor,
             planes can overlap well in a wrong pose, and it cannot absorb a
             scale difference between the two runs.
  sim3       1. SuperPoint + LightGlue matches between keyframes of A and B
                (SIFT fallback), verified per image pair by a fundamental-
                matrix RANSAC. SIFT on its own fails here: the viewpoints
                differ by tens of degrees and the wood grain repeats, so its
                "matches" collapse onto one point;
             2. each match lifted to 3D with that frame's depth and pose,
                dropping pixels on depth edges (flying pixels);
             3. RANSAC Sim3 (Umeyama with scale) over all 3D-3D matches --
                monocular metric depth need not have the same scale in two
                runs, so scale is solved, not assumed;
             4. rigid point-to-plane ICP, then colored ICP, on the scaled map.
                Scale stays fixed from step 3: ICP that also solves scale
                shrinks the query map into a corner of the reference. Each
                refinement is kept only if the symmetric F-score improves.
             Every step takes its rotation from monocular depth, which bends
             differently from each viewpoint: on the fyp videos it lands 8 deg
             off the epipolar rotation, worse than the baseline.
  epipolar   (the improved method) rotation and translation/scale from
             separate sources:
             1. same verified matches as sim3;
             2. rotation without depth: each image pair's essential matrix
                gives R_Ai^T R R_Bj, so every pair votes for the alignment
                rotation R; they are combined by robust (L1) rotation
                averaging;
             3. with R fixed, scale and translation from the 3D-3D matches by
                2-point RANSAC -- depth is only trusted for where things are,
                not for which way the map is turned;
             4. translation-only point-to-plane ICP on the voxel maps, kept
                only if the symmetric F-score improves.

Estimation uses the matches of every other query keyframe; the rest are held
out, so the image-based checks below never see pairs a method was fitted to.

Accuracy is measured three ways:

  map overlap        symmetric F-score at 1/2/5 cm (precision = share of
                     query voxels with a reference voxel within tau, recall =
                     the reverse), inlier RMSE, median closest-point distance
                     in the region both videos saw, voxel IoU;
  relative rotation  independent of depth and of the maps: for held-out image
                     pairs (A_i, B_j), the rotation from the essential matrix
                     of the 2D matches vs. the one implied by the aligned
                     trajectories. The same check inside one video (A_i, A_j)
                     measures DROID-SLAM's own rotation accuracy, the floor for
                     the cross-video number;
  relocalization     each held-out query keyframe localized in the reference
                     map by PnP (the query's 2D keypoints against the
                     reference's 3D points -- no query depth, no query SLAM),
                     compared with its aligned DROID-SLAM pose: camera-centre
                     error (cm) and rotation error (deg) per frame.

Usage:
    python scripts/compare_video_poses.py \\
        --ref   data/processed/fyp_updated_same_angle \\
        --query data/processed/fyp_updated_diff_angle \\
        --out   outputs/pose_comparison/same_vs_diff

Outputs (under --out):
  report.json                 all metrics, transforms, camera-motion summary
  overlay_topdown.png         both maps + trajectories, per method, from above
  overlay_side.png            same, from the side
  rotation_check.png          per-pair rotation error histogram
  merged_<method>.ply         A in red, aligned B in blue
  query_poses_in_ref/*.txt    B's camera-to-world poses in A's world (epipolar)

Voxel maps are cached under --cache_dir, keyed by dataset and resolution;
delete the cache after re-running pose estimation on a dataset.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml  # noqa: E402

from object_fusion.view_dataset import load_view_dataset  # noqa: E402
from scene_graph.builder.gravity_alignment import estimate_gravity_alignment  # noqa: E402

logger = logging.getLogger("compare_video_poses")
reg = o3d.pipelines.registration

DEFAULT_LIGHTGLUE_DIR = "/home/ravindu/SceneGraphNav/external/DovSG/third_party/LightGlue"


# ─── Loading ───

class Sequence:
    """A prepared dataset: poses (camera-to-world), intrinsics, rgb, camera-frame points."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.name = self.root.name
        meta = json.loads((self.root / "metadata.json").read_text())
        k = meta["K"]
        self.K = np.array([[k[0], 0, k[2]], [0, k[4], k[5]], [0, 0, 1]], dtype=np.float64)
        self.length = meta["length"]
        self.poses = np.stack([np.loadtxt(self.root / "poses" / f"{i:06d}.txt") for i in range(self.length)])

    def rgb(self, i: int) -> np.ndarray:
        return cv2.cvtColor(cv2.imread(str(self.root / "rgb" / f"{i:06d}.jpg")), cv2.COLOR_BGR2RGB)

    def points(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        return (np.load(self.root / "point" / f"{i:06d}.npy"),
                np.load(self.root / "mask" / f"{i:06d}.npy"))


def voxel_map_cloud(seq: Sequence, resolution: float, scene_graph_generation_root: str,
                    cache_dir: Path | None = None) -> o3d.geometry.PointCloud:
    """The same voxel map the scene-graph pipeline builds (DoVSG ViewDataset),
    one point per occupied voxel, coloured by its mean colour. Building it takes
    ~4 min per video, so it is cached as a .ply."""
    cache = cache_dir / f"{seq.name}_r{resolution:g}.ply" if cache_dir else None
    if cache is not None and cache.exists():
        logger.info(f"Voxel map from cache: {cache}")
        return o3d.io.read_point_cloud(str(cache))
    vd = load_view_dataset(recorder_dir=str(seq.root), scene_graph_generation_root=scene_graph_generation_root,
                           resolution=resolution)
    indexes = np.array(list(vd.indexes_colors_mapping_dict.keys()))
    colors = np.array(list(vd.indexes_colors_mapping_dict.values()))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(vd.index_to_point(indexes), dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(str(cache), pcd)
    return pcd


# ─── Sim3 helpers ───

def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> np.ndarray:
    """4x4 similarity T with dst ~= T @ src (points as rows)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / xs.var(0).sum()) if with_scale else 1.0
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = mu_d - s * R @ mu_s
    return T


def decompose_sim3(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    s = float(np.cbrt(np.linalg.det(T[:3, :3])))
    return s, T[:3, :3] / s, T[:3, 3]


def apply(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return pts @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def transform_poses(T_sim3: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Camera-to-world poses of B expressed in A's world. The rotation part of
    a Sim3 is s*R, so only R rotates the camera; the scale moves its centre."""
    s, R, t = decompose_sim3(T_sim3)
    out = poses.copy()
    out[:, :3, :3] = R @ poses[:, :3, :3]
    out[:, :3, 3] = poses[:, :3, 3] @ (s * R).T + t
    return out


# ─── Baseline: FPFH + RANSAC + ICP (rigid) ───

def preprocess(pcd: o3d.geometry.PointCloud, voxel: float):
    down = pcd.voxel_down_sample(voxel)
    down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fpfh = reg.compute_fpfh_feature(down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return down, fpfh


def register_baseline(src: o3d.geometry.PointCloud, dst: o3d.geometry.PointCloud, voxel: float = 0.05) -> np.ndarray:
    src_d, src_f = preprocess(src, voxel)
    dst_d, dst_f = preprocess(dst, voxel)
    ransac = reg.registration_ransac_based_on_feature_matching(
        src_d, dst_d, src_f, dst_f, True, voxel * 1.5,
        reg.TransformationEstimationPointToPoint(False), 3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9), reg.CorrespondenceCheckerBasedOnDistance(voxel * 1.5)],
        reg.RANSACConvergenceCriteria(1_000_000, 0.999))
    icp = reg.registration_icp(src_d, dst_d, voxel * 0.8, ransac.transformation,
                               reg.TransformationEstimationPointToPlane(),
                               reg.ICPConvergenceCriteria(max_iteration=100))
    return np.asarray(icp.transformation)


# ─── Image matching ───

def _load_lightglue(lightglue_dir: str):
    """Imports SuperPoint + LightGlue from a source checkout without installing
    it. The package __init__ pulls in ALIKED, which needs kornia; SuperPoint
    only uses kornia for an RGB->gray conversion and a resize, so those two are
    provided here when kornia is absent."""
    import torch
    import torch.nn.functional as F

    try:
        import kornia  # noqa: F401
    except ImportError:
        k = types.ModuleType("kornia")
        color = types.ModuleType("kornia.color")
        geometry = types.ModuleType("kornia.geometry")
        transform = types.ModuleType("kornia.geometry.transform")
        color.rgb_to_grayscale = lambda x: (x * torch.tensor([0.299, 0.587, 0.114], device=x.device)
                                            .view(1, 3, 1, 1)).sum(1, keepdim=True)
        transform.resize = lambda x, size, interpolation="bilinear", align_corners=None, side=None, antialias=False: \
            F.interpolate(x, size=size, mode=interpolation, align_corners=align_corners, antialias=antialias)
        k.color, k.geometry, geometry.transform = color, geometry, transform
        sys.modules.update({"kornia": k, "kornia.color": color, "kornia.geometry": geometry,
                            "kornia.geometry.transform": transform})
    if "lightglue" not in sys.modules:
        pkg = types.ModuleType("lightglue")
        pkg.__path__ = [str(Path(lightglue_dir) / "lightglue")]
        sys.modules["lightglue"] = pkg
    from lightglue.lightglue import LightGlue
    from lightglue.superpoint import SuperPoint
    return SuperPoint, LightGlue


class Matcher:
    """SuperPoint + LightGlue when available, SIFT + ratio test otherwise."""

    def __init__(self, backend: str = "lightglue", lightglue_dir: str = DEFAULT_LIGHTGLUE_DIR, device: str = "cuda"):
        self.backend = backend
        if backend == "lightglue":
            try:
                import torch
                SuperPoint, LightGlue = _load_lightglue(lightglue_dir)
                self.device = device if torch.cuda.is_available() else "cpu"
                self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
                self.lightglue = LightGlue(features="superpoint").eval().to(self.device)
                self._torch = torch
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LightGlue unavailable ({e}); falling back to SIFT, which is much weaker "
                               "across large viewpoint changes.")
                self.backend = "sift"
        if self.backend == "sift":
            self.sift = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def features(self, rgb: np.ndarray):
        if self.backend == "lightglue":
            t = self._torch.from_numpy(rgb).permute(2, 0, 1).float().to(self.device)[None] / 255.0
            with self._torch.no_grad():
                f = self.extractor.extract(t)
            return f["keypoints"][0].cpu().numpy(), f
        kps, desc = self.sift.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        return np.array([k.pt for k in kps], dtype=np.float32).reshape(-1, 2), desc

    def match(self, fa, fb) -> tuple[np.ndarray, np.ndarray]:
        if self.backend == "lightglue":
            with self._torch.no_grad():
                m = self.lightglue({"image0": fa, "image1": fb})["matches"][0].cpu().numpy()
            return m[:, 0], m[:, 1]
        if fa is None or fb is None:
            return np.array([], int), np.array([], int)
        pairs = [(m.queryIdx, m.trainIdx) for p in self.bf.knnMatch(fa, fb, k=2)
                 if len(p) == 2 for m, n in [p] if m.distance < 0.8 * n.distance]
        if not pairs:
            return np.array([], int), np.array([], int)
        ia, ib = map(np.array, zip(*pairs))
        return ia, ib


def edge_free_mask(points: np.ndarray, valid: np.ndarray, rel: float = 0.03, k: int = 5) -> np.ndarray:
    """Valid pixels whose depth is locally smooth: depth edges produce flying
    pixels half-way between foreground and background."""
    z = np.where(valid, points[..., 2], 0).astype(np.float32)
    kernel = np.ones((k, k), np.uint8)
    spread = cv2.dilate(z, kernel) - cv2.erode(np.where(valid, z, 1e6).astype(np.float32), kernel)
    return valid & (spread < rel * np.maximum(z, 1e-6))


class KeyframeFeatures:
    """Keypoints of every stride-th frame, with their world position from the
    frame's depth and pose, and whether that depth is trustworthy."""

    def __init__(self, seq: Sequence, stride: int, matcher: Matcher):
        self.seq = seq
        self.frames = list(range(0, seq.length, stride))
        self.kp, self.feat, self.xyz_world, self.ok = {}, {}, {}, {}
        for i in self.frames:
            uv, feat = matcher.features(seq.rgb(i))
            pts, valid = seq.points(i)
            good = edge_free_mask(pts, valid)
            ui = np.clip(np.round(uv[:, 0]).astype(int), 0, pts.shape[1] - 1)
            vi = np.clip(np.round(uv[:, 1]).astype(int), 0, pts.shape[0] - 1)
            T = seq.poses[i]
            self.kp[i] = uv
            self.feat[i] = feat
            self.xyz_world[i] = pts[vi, ui].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
            self.ok[i] = good[vi, ui]


def match_pair(fa: KeyframeFeatures, i: int, fb: KeyframeFeatures, j: int, matcher: Matcher,
               min_inliers: int = 30):
    """Matches between A_i and B_j that a fundamental-matrix RANSAC agrees
    with. Returns (idx_a, idx_b) or None."""
    ia, ib = matcher.match(fa.feat[i], fb.feat[j])
    if len(ia) < min_inliers:
        return None
    _, inl = cv2.findFundamentalMat(fa.kp[i][ia], fb.kp[j][ib], cv2.USAC_MAGSAC, 1.5, 0.999, 10000)
    if inl is None:
        return None
    inl = inl.ravel().astype(bool)
    if inl.sum() < min_inliers:
        return None
    return ia[inl], ib[inl]


def ransac_sim3(src: np.ndarray, dst: np.ndarray, thresh: float, iters: int = 20000,
                rng=np.random.default_rng(0)) -> tuple[np.ndarray, np.ndarray]:
    best_inl, n = None, len(src)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if np.linalg.norm(np.cross(src[idx[1]] - src[idx[0]], src[idx[2]] - src[idx[0]])) < 1e-4:
            continue  # near-colinear sample
        T = umeyama(src[idx], dst[idx])
        if not 0.66 < np.cbrt(np.linalg.det(T[:3, :3])) < 1.5:
            continue
        inl = np.linalg.norm(apply(T, src) - dst, axis=1) < thresh
        if best_inl is None or inl.sum() > best_inl.sum():
            best_inl = inl
    T = umeyama(src[best_inl], dst[best_inl])
    for _ in range(3):  # re-fit on the refined inlier set
        inl = np.linalg.norm(apply(T, src) - dst, axis=1) < thresh
        T = umeyama(src[inl], dst[inl])
    return T, inl


def collect_matches(fa: KeyframeFeatures, fb: KeyframeFeatures, matcher: Matcher, max_gap: int | None = None):
    """Verified matches for every keyframe pair (A_i, B_j); with max_gap, only
    pairs 0 < j - i <= max_gap (for matching a sequence against itself)."""
    out = []
    for i in fa.frames:
        for j in fb.frames:
            if max_gap is not None and not 0 < j - i <= max_gap:
                continue
            m = match_pair(fa, i, fb, j, matcher)
            if m is not None:
                out.append((i, j, m[0], m[1]))
    return out


def register_improved(fa, fb, pair_matches, cloud_a, cloud_b, ransac_thresh: float) -> dict:
    if not pair_matches:
        raise RuntimeError("No keyframe pair of the two videos shares enough verified matches.")
    src, dst = [], []
    for i, j, ia, ib in pair_matches:
        ok = fa.ok[i][ia] & fb.ok[j][ib]
        src.append(fb.xyz_world[j][ib[ok]])
        dst.append(fa.xyz_world[i][ia[ok]])
    src, dst = np.concatenate(src), np.concatenate(dst)
    T0, inl = ransac_sim3(src, dst, ransac_thresh)
    s = decompose_sim3(T0)[0]
    logger.info(f"RANSAC Sim3: {inl.sum()}/{len(src)} 3D matches within {ransac_thresh * 100:.0f} cm, scale={s:.4f}")

    # Refine the rigid part on the scaled map; keep a step only if it helps.
    S = np.diag([s, s, s, 1.0])
    scaled_b = copy.deepcopy(cloud_b).transform(S)
    rigid = T0 @ np.linalg.inv(S)
    for c in (cloud_a, scaled_b):
        c.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=30))
    best_f = symmetric_fscore(cloud_a, scaled_b, rigid, 0.02)
    steps = [("ransac", round(best_f, 4))]
    for name, dist in (("icp_p2plane_5cm", 0.05), ("icp_p2plane_2cm", 0.02), ("colored_icp_2cm", 0.02),
                       ("colored_icp_1cm", 0.01)):
        try:
            if name.startswith("icp"):
                r = reg.registration_icp(scaled_b, cloud_a, dist, rigid, reg.TransformationEstimationPointToPlane(),
                                         reg.ICPConvergenceCriteria(max_iteration=50))
            else:
                r = reg.registration_colored_icp(scaled_b, cloud_a, dist, rigid,
                                                 reg.TransformationEstimationForColoredICP(),
                                                 reg.ICPConvergenceCriteria(max_iteration=50))
        except RuntimeError as e:
            logger.warning(f"{name} skipped: {e}")
            continue
        f = symmetric_fscore(cloud_a, scaled_b, np.asarray(r.transformation), 0.02)
        if f > best_f:
            rigid, best_f = np.asarray(r.transformation), f
            steps.append((name, round(f, 4)))
        else:
            steps.append((name + " (rejected)", round(f, 4)))
    logger.info(f"Refinement (F-score@2cm): {steps}")
    return {"T": rigid @ S, "T_ransac": T0, "match_inliers": int(inl.sum()), "matches": int(len(src)),
            "refinement_steps": steps}


# ─── Accuracy ───

def symmetric_fscore(cloud_a, cloud_b, T, tau: float) -> float:
    b = copy.deepcopy(cloud_b).transform(T)
    p = reg.evaluate_registration(b, cloud_a, tau).fitness
    r = reg.evaluate_registration(cloud_a, b, tau).fitness
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def overlap_metrics(cloud_a, cloud_b, T, voxel_iou_sizes=(0.02, 0.05)) -> dict:
    b = copy.deepcopy(cloud_b).transform(T)
    out = {}
    for th in (0.01, 0.02, 0.05):
        e_ba = reg.evaluate_registration(b, cloud_a, th)
        e_ab = reg.evaluate_registration(cloud_a, b, th)
        p, r = e_ba.fitness, e_ab.fitness
        tag = f"{int(th * 100)}cm"
        out[f"precision@{tag}"] = round(p, 4)
        out[f"recall@{tag}"] = round(r, 4)
        out[f"fscore@{tag}"] = round(2 * p * r / (p + r) if p + r else 0.0, 4)
        out[f"inlier_rmse@{tag}_mm"] = round(e_ba.inlier_rmse * 1000, 2)
    d_ba = np.asarray(b.compute_point_cloud_distance(cloud_a))
    d_ab = np.asarray(cloud_a.compute_point_cloud_distance(b))
    # the region both videos saw = points with a partner within 10 cm;
    # outside it the videos simply saw different surfaces
    ov = np.concatenate([d_ba[d_ba < 0.10], d_ab[d_ab < 0.10]])
    out["overlap_median_dist_mm"] = round(float(np.median(ov)) * 1000, 2) if len(ov) else None
    out["overlap_p90_dist_mm"] = round(float(np.percentile(ov, 90)) * 1000, 2) if len(ov) else None
    pa, pb = np.asarray(cloud_a.points), np.asarray(b.points)
    for v in voxel_iou_sizes:
        va = set(map(tuple, np.floor(pa / v).astype(np.int64)))
        vb = set(map(tuple, np.floor(pb / v).astype(np.int64)))
        out[f"voxel_iou@{int(v * 100)}cm"] = round(len(va & vb) / len(va | vb), 4)
    return out


def relative_rotation_check(K, poses_a, poses_b, pair_matches, fa, fb) -> dict:
    """Per matched image pair: rotation (and translation direction) from the
    essential matrix of the 2D matches vs. the relative pose of the two
    cameras in the (aligned) trajectories. Uses no depth."""
    rot_err, dir_err = [], []
    for i, j, ia, ib in pair_matches:
        pa, pb = fa.kp[i][ia].astype(np.float64), fb.kp[j][ib].astype(np.float64)
        E, m = cv2.findEssentialMat(pb, pa, K, cv2.RANSAC, 0.999, 1.0)
        if E is None or E.shape != (3, 3):
            continue
        n, R_e, t_e, _ = cv2.recoverPose(E, pb, pa, K, mask=m)
        if n < 20:
            continue
        # recoverPose: x_A = R x_B + t   <=>   inv(T_A_i) @ T_B_j (camera-to-world)
        T_rel = np.linalg.inv(poses_a[i]) @ poses_b[j]
        rot_err.append(rotation_angle_deg(T_rel[:3, :3].T @ R_e))
        t_pred = T_rel[:3, 3]
        if np.linalg.norm(t_pred) > 0.10:  # direction is meaningless at a tiny baseline
            c = float(np.dot(t_pred / np.linalg.norm(t_pred), t_e.ravel()))
            dir_err.append(float(np.degrees(np.arccos(np.clip(c, -1, 1)))))
    rot_err = np.array(rot_err)
    return {
        "pairs_checked": int(len(rot_err)),
        "rotation_err_deg_median": round(float(np.median(rot_err)), 3) if len(rot_err) else None,
        "rotation_err_deg_p90": round(float(np.percentile(rot_err, 90)), 3) if len(rot_err) else None,
        "share_under_5deg": round(float((rot_err < 5).mean()), 3) if len(rot_err) else None,
        "translation_dir_err_deg_median": round(float(np.median(dir_err)), 2) if dir_err else None,
        "_rot_err": rot_err.tolist(),
    }


# ─── Camera motion between the videos ───

def camera_motion(seq_a, seq_b, T, T_grav: np.ndarray, scene_center: np.ndarray) -> dict:
    """How the capture differs, in a gravity-aligned frame of A (z up).
    Heading is the compass direction the camera looks in (about z); azimuth is
    where the camera stands, seen from the middle of the shared scene."""
    pa = T_grav @ seq_a.poses
    pb = T_grav @ transform_poses(T, seq_b.poses)
    c = apply(T_grav, scene_center[None])[0]

    def summary(poses):
        fwd = poses[:, :3, 2]  # camera looks along +z
        heading = np.degrees(np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0])))
        pitch = np.degrees(np.arcsin(np.clip(fwd[:, 2], -1, 1)))
        centers = poses[:, :3, 3]
        rel = centers[:, :2] - c[:2]
        return {
            "centroid_m": np.round(centers.mean(0), 3).tolist(),
            "camera_height_m": round(float(np.median(centers[:, 2])), 3),
            "path_length_m": round(float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()), 3),
            "heading_deg_start_end": [round(float(heading[0]), 1), round(float(heading[-1]), 1)],
            "heading_sweep_deg": round(float(heading.max() - heading.min()), 1),
            "pitch_deg_median": round(float(np.median(pitch)), 1),
            "azimuth_around_scene_deg": round(float(np.degrees(np.arctan2(rel[:, 1].mean(), rel[:, 0].mean()))), 1),
            "distance_to_scene_m": round(float(np.linalg.norm(rel, axis=1).mean()), 3),
            "_heading_mean": float(np.degrees(np.arctan2(fwd[:, 1].mean(), fwd[:, 0].mean()))),
        }

    sa, sb = summary(pa), summary(pb)
    first = np.linalg.inv(pa[0]) @ pb[0]
    wrap = lambda d: (d + 180) % 360 - 180  # noqa: E731
    return {
        "reference": {k: v for k, v in sa.items() if not k.startswith("_")},
        "query": {k: v for k, v in sb.items() if not k.startswith("_")},
        "first_frame_relative": {
            "rotation_deg": round(rotation_angle_deg(first[:3, :3]), 2),
            "translation_m": round(float(np.linalg.norm(first[:3, 3])), 3),
        },
        "centroid_offset_m": round(float(np.linalg.norm(np.array(sa["centroid_m"]) - np.array(sb["centroid_m"]))), 3),
        "orbit_angle_around_scene_deg": round(float(wrap(sb["azimuth_around_scene_deg"] - sa["azimuth_around_scene_deg"])), 1),
        "mean_heading_change_deg": round(float(wrap(sb["_heading_mean"] - sa["_heading_mean"])), 1),
        "pitch_change_deg": round(sb["pitch_deg_median"] - sa["pitch_deg_median"], 1),
        "camera_height_change_m": round(sb["camera_height_m"] - sa["camera_height_m"], 3),
    }


# ─── Plots ───

def plot_overlays(out_dir, cloud_a, cloud_b, seq_a, seq_b, transforms: dict, T_grav):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    def sample(p, n=40000):
        return p[rng.choice(len(p), min(n, len(p)), replace=False)]

    pa = sample(apply(T_grav, np.asarray(cloud_a.points)))
    ca = apply(T_grav, seq_a.poses[:, :3, 3])
    for view, (ix, iy, lx, ly) in {"topdown": (0, 1, "x (m)", "y (m)"), "side": (0, 2, "x (m)", "z up (m)")}.items():
        fig, axes = plt.subplots(1, len(transforms), figsize=(6 * len(transforms), 6), squeeze=False)
        for ax, (name, T) in zip(axes[0], transforms.items()):
            pb = sample(apply(T_grav @ T, np.asarray(cloud_b.points)))
            cb = apply(T_grav, transform_poses(T, seq_b.poses)[:, :3, 3])
            ax.scatter(pa[:, ix], pa[:, iy], s=0.2, c="tab:red", alpha=0.35, label=f"ref: {seq_a.name}")
            ax.scatter(pb[:, ix], pb[:, iy], s=0.2, c="tab:blue", alpha=0.35, label=f"query: {seq_b.name}")
            ax.plot(ca[:, ix], ca[:, iy], "-", c="darkred", lw=2)
            ax.plot(cb[:, ix], cb[:, iy], "-", c="navy", lw=2)
            ax.plot(*ca[0, [ix, iy]], "o", c="darkred")
            ax.plot(*cb[0, [ix, iy]], "o", c="navy")
            ax.set_title(name)
            ax.set_xlabel(lx)
            ax.set_ylabel(ly)
            ax.set_aspect("equal")
        axes[0][0].legend(markerscale=20, loc="upper left", fontsize=8)
        fig.suptitle("Voxel maps overlaid in the reference's gravity-aligned world (dots = start camera)")
        fig.tight_layout()
        fig.savefig(out_dir / f"overlay_{view}.png", dpi=110)
        plt.close(fig)


def plot_rotation_check(out_dir, checks: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 30, 61)
    for name, c in checks.items():
        if c["_rot_err"]:
            ax.hist(np.clip(c["_rot_err"], 0, 30), bins=bins, alpha=0.5, density=True,
                    label=f"{name} (median {c['rotation_err_deg_median']:.2f} deg, n={c['pairs_checked']})")
    ax.set_xlabel("rotation error vs. essential matrix (deg, clipped at 30)")
    ax.set_ylabel("share of image pairs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "rotation_check.png", dpi=110)
    plt.close(fig)


# ─── Main ───

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference dataset dir (its world is the common frame)")
    ap.add_argument("--query", required=True, help="dataset dir registered onto the reference")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/pipeline.yaml", help="for external.scene_graph_generation_root")
    ap.add_argument("--resolution", type=float, default=0.01, help="voxel map resolution (m)")
    ap.add_argument("--keyframe_stride", type=int, default=4)
    ap.add_argument("--ransac_thresh", type=float, default=0.05, help="Sim3 RANSAC inlier distance (m)")
    ap.add_argument("--matcher", choices=["lightglue", "sift"], default="lightglue")
    ap.add_argument("--lightglue_dir", default=DEFAULT_LIGHTGLUE_DIR)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text())
    sgg_root = cfg["external"]["scene_graph_generation_root"]

    seq_a, seq_b = Sequence(REPO_ROOT / args.ref), Sequence(REPO_ROOT / args.query)
    timings = {}
    t = time.time()
    cloud_a = voxel_map_cloud(seq_a, args.resolution, sgg_root)
    cloud_b = voxel_map_cloud(seq_b, args.resolution, sgg_root)
    timings["voxel_maps_s"] = round(time.time() - t, 1)
    logger.info(f"Voxel maps: {len(cloud_a.points)} (ref) and {len(cloud_b.points)} (query) occupied voxels")

    t = time.time()
    T_base = register_baseline(cloud_b, cloud_a)
    timings["baseline_s"] = round(time.time() - t, 1)

    t = time.time()
    matcher = Matcher(args.matcher, args.lightglue_dir)
    fa = KeyframeFeatures(seq_a, args.keyframe_stride, matcher)
    fb = KeyframeFeatures(seq_b, args.keyframe_stride, matcher)
    cross = collect_matches(fa, fb, matcher)
    logger.info(f"{matcher.backend}: {len(fa.frames)} x {len(fb.frames)} keyframe pairs, {len(cross)} verified "
                f"({sum(len(m[2]) for m in cross)} matches)")
    imp = register_improved(fa, fb, cross, cloud_a, cloud_b, args.ransac_thresh)
    timings["improved_s"] = round(time.time() - t, 1)

    transforms = {"identity (no registration)": np.eye(4), "baseline FPFH+ICP": T_base,
                  f"improved {matcher.backend}-Sim3+ICP": imp["T"]}

    # DROID-SLAM's own rotation accuracy inside each video: the floor for the
    # cross-video check
    within = {}
    for label, seq, f in (("reference", seq_a, fa), ("query", seq_b, fb)):
        pairs = collect_matches(f, f, matcher, max_gap=10 * args.keyframe_stride)
        within[label] = relative_rotation_check(seq.K, seq.poses, seq.poses, pairs, f, f)

    T_grav, grav_info = estimate_gravity_alignment(np.asarray(cloud_a.points), seq_a.poses[:, :3, :3])

    results, checks = {}, {}
    for name, T in transforms.items():
        s, R, tr = decompose_sim3(T)
        checks[name] = relative_rotation_check(seq_a.K, seq_a.poses, transform_poses(T, seq_b.poses), cross, fa, fb)
        results[name] = {
            "transform_query_to_ref": np.round(T, 6).tolist(),
            "scale": round(s, 4),
            "rotation_deg": round(rotation_angle_deg(R), 2),
            "translation_m": round(float(np.linalg.norm(tr)), 3),
            "map_overlap": overlap_metrics(cloud_a, cloud_b, T),
            "relative_rotation_check": {k: v for k, v in checks[name].items() if not k.startswith("_")},
        }
        logger.info(f"{name}: {json.dumps(results[name]['map_overlap'])} | "
                    f"{json.dumps(results[name]['relative_rotation_check'])}")

    # middle of the region both videos saw, under the improved alignment
    b_aligned = copy.deepcopy(cloud_b).transform(imp["T"])
    d = np.asarray(cloud_a.compute_point_cloud_distance(b_aligned))
    shared = np.asarray(cloud_a.points)[d < 0.05]
    scene_center = np.median(shared if len(shared) > 100 else np.asarray(cloud_a.points), axis=0)

    report = {
        "reference": str(seq_a.root), "query": str(seq_b.root),
        "frames": {"reference": seq_a.length, "query": seq_b.length},
        "intrinsics": {"reference": seq_a.K.tolist(), "query": seq_b.K.tolist()},
        "voxel_map_points": {"reference": len(cloud_a.points), "query": len(cloud_b.points)},
        "gravity_alignment": grav_info,
        "improved_registration": {
            "matcher": matcher.backend,
            "image_pairs_verified": len(cross),
            "matches_3d": imp["matches"],
            "sim3_ransac_inliers": imp["match_inliers"],
            "transform_after_ransac": np.round(imp["T_ransac"], 6).tolist(),
            "refinement_steps_fscore@2cm": imp["refinement_steps"],
        },
        "within_video_rotation_check": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                                        for k, v in within.items()},
        "methods": results,
        "camera_motion_between_videos": camera_motion(seq_a, seq_b, imp["T"], T_grav, scene_center),
        "timings": timings,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))

    pose_dir = out / "query_poses_in_ref"
    pose_dir.mkdir(exist_ok=True)
    for k, P in enumerate(transform_poses(imp["T"], seq_b.poses)):
        np.savetxt(pose_dir / f"{k:06d}.txt", P)
    for name, T in transforms.items():
        a = copy.deepcopy(cloud_a).paint_uniform_color([0.85, 0.2, 0.2])
        b = copy.deepcopy(cloud_b).transform(T).paint_uniform_color([0.2, 0.35, 0.85])
        o3d.io.write_point_cloud(str(out / f"merged_{name.split()[0]}.ply"), a + b)

    plot_overlays(out, cloud_a, cloud_b, seq_a, seq_b, transforms, T_grav)
    plot_rotation_check(out, {**{f"within {k}": v for k, v in within.items()},
                              **{k: v for k, v in checks.items() if not k.startswith("identity")}})
    logger.info(f"Camera motion between videos: {json.dumps(report['camera_motion_between_videos'], indent=2)}")
    logger.info(f"Report: {out / 'report.json'}")


if __name__ == "__main__":
    main()
