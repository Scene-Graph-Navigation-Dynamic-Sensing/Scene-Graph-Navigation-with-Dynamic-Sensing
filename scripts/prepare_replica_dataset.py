"""Convert a NICE-SLAM Replica scene into a data/raw dataset folder.

NICE-SLAM's Replica release (the RGB-D trajectories ConceptGraphs evaluates on)
holds, per scene, results/frame{i:06d}.jpg, results/depth{i:06d}.png and
traj.txt (one row-major 4x4 camera-to-world matrix per line), plus a shared
cam_params.json. This writes the layout data/raw/README.md describes:

    data/raw/replica_<scene>/
        rgb/{i:06d}.jpg       every --stride'th frame, renumbered from 0
        depth/{i:06d}.png     uint16 millimetres (camera.yaml depth_scale: 1000)
        poses/{i:06d}.txt     4x4 camera-to-world, OpenCV axes
        camera.yaml
        source.json           which source frames were kept, and the check below

NICE-SLAM's own loader flips the y and z columns of each pose to reach its
OpenGL-style renderer convention, so traj.txt itself is in OpenCV axes (x
right, y down, z forward), which is what the pipeline expects. The script
does not take that on trust: it back-projects a few frames' depth with their
poses and measures the distance to the scene's ground-truth semantic mesh
(--gt_root). A median of a centimetre or two means the poses, the depth
scale and the mesh share one frame; the script stops if it is worse.

Usage:
    python scripts/prepare_replica_dataset.py --scene office_0
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("prepare_replica")

NICE_SLAM_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/nice_slam/Replica")
GT_ROOT = Path("/home/ravindu/SceneGraphNav/datasets/replica/extracted")


def nice_slam_name(scene: str) -> str:
    """office_0 (Replica) -> office0 (NICE-SLAM)."""
    return scene.replace("_", "")


def load_camera(nice_root: Path) -> dict:
    cam = json.loads((nice_root / "cam_params.json").read_text())["camera"]
    return {k: float(cam[k]) for k in ("w", "h", "fx", "fy", "cx", "cy", "scale")}


def load_trajectory(scene_dir: Path) -> np.ndarray:
    rows = np.loadtxt(scene_dir / "traj.txt", dtype=np.float64)
    return rows.reshape(-1, 4, 4)


def backproject(depth_m: np.ndarray, cam: dict, step: int = 8) -> np.ndarray:
    """Camera-frame XYZ of every step'th pixel with valid depth."""
    h, w = depth_m.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth_m[v, u]
    ok = z > 0
    x = (u[ok] - cam["cx"]) * z[ok] / cam["fx"]
    y = (v[ok] - cam["cy"]) * z[ok] / cam["fy"]
    return np.stack([x, y, z[ok]], axis=1)


def check_alignment(scene_dir: Path, gt_scene: Path, cam: dict, poses: np.ndarray,
                    frames: list[int]) -> dict:
    from plyfile import PlyData
    from scipy.spatial import cKDTree

    v = PlyData.read(str(gt_scene / "habitat" / "mesh_semantic.ply"))["vertex"]
    tree = cKDTree(np.stack([v["x"], v["y"], v["z"]], axis=1))
    dists = []
    for i in frames:
        depth = cv2.imread(str(scene_dir / "results" / f"depth{i:06d}.png"), cv2.IMREAD_UNCHANGED)
        pts = backproject(depth.astype(np.float64) / cam["scale"], cam)
        T = poses[i]
        world = pts @ T[:3, :3].T + T[:3, 3]
        dists.append(tree.query(world)[0])
    d = np.concatenate(dists)
    return {"frames_checked": frames, "median_m": float(np.median(d)),
            "p90_m": float(np.percentile(d, 90)), "within_5cm": float((d < 0.05).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", required=True, help="Replica scene, e.g. office_0, room_2")
    ap.add_argument("--nice_slam_root", default=str(NICE_SLAM_ROOT))
    ap.add_argument("--gt_root", default=str(GT_ROOT))
    ap.add_argument("--stride", type=int, default=10, help="keep every Nth of the 2000 frames")
    ap.add_argument("--out", default=None, help="default: data/raw/replica_<scene>")
    ap.add_argument("--max_median_m", type=float, default=0.03,
                    help="stop if back-projected depth sits further than this from the GT mesh")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    nice_root = Path(args.nice_slam_root)
    scene_dir = nice_root / nice_slam_name(args.scene)
    out = Path(args.out) if args.out else REPO_ROOT / "data" / "raw" / f"replica_{args.scene}"
    if not scene_dir.is_dir():
        logger.error("no NICE-SLAM scene at %s", scene_dir)
        return 1

    cam = load_camera(nice_root)
    poses = load_trajectory(scene_dir)
    n_src = len(list((scene_dir / "results").glob("frame*.jpg")))
    if n_src != len(poses):
        logger.error("%d frames but %d poses in %s", n_src, len(poses), scene_dir)
        return 1

    check = check_alignment(scene_dir, Path(args.gt_root) / args.scene, cam, poses,
                            frames=list(range(0, n_src, max(n_src // 5, 1)))[:5])
    logger.info("depth vs GT mesh: median %.1f mm, p90 %.1f mm, %.1f%% within 5 cm",
                check["median_m"] * 1e3, check["p90_m"] * 1e3, 100 * check["within_5cm"])
    if check["median_m"] > args.max_median_m:
        logger.error("poses/depth do not line up with the GT mesh; not writing a dataset")
        return 1

    if out.exists():
        shutil.rmtree(out)
    for sub in ("rgb", "depth", "poses"):
        (out / sub).mkdir(parents=True)

    kept = list(range(0, n_src, args.stride))
    for idx, src in enumerate(kept):
        shutil.copyfile(scene_dir / "results" / f"frame{src:06d}.jpg", out / "rgb" / f"{idx:06d}.jpg")
        raw = cv2.imread(str(scene_dir / "results" / f"depth{src:06d}.png"), cv2.IMREAD_UNCHANGED)
        mm = np.clip(raw.astype(np.float64) / cam["scale"] * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(str(out / "depth" / f"{idx:06d}.png"), mm)
        np.savetxt(out / "poses" / f"{idx:06d}.txt", poses[src])

    (out / "camera.yaml").write_text(
        f"# NICE-SLAM Replica {nice_slam_name(args.scene)} (cam_params.json)\n"
        f"fx: {cam['fx']}\nfy: {cam['fy']}\ncx: {cam['cx']}\ncy: {cam['cy']}\n"
        f"width: {int(cam['w'])}\nheight: {int(cam['h'])}\ndepth_scale: 1000\n")
    (out / "source.json").write_text(json.dumps({
        "scene": args.scene,
        "nice_slam_dir": str(scene_dir),
        "stride": args.stride,
        "source_frames": kept,
        "frames": len(kept),
        "alignment_check": check,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    logger.info("wrote %d frames (every %d of %d) to %s", len(kept), args.stride, n_src, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
