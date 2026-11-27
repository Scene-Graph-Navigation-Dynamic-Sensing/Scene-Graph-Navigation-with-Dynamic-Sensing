"""Compare estimated camera poses against ground truth.

Both inputs are directories of {frame:06d}.txt 4x4 camera-to-world matrices,
matched by frame number — e.g. a prepared dataset's poses/ (ground truth, as
written by prepare_3rscan_dataset.py) against poses_droidslam/ produced by
dovsg/scripts/pose_estimation.py. run_static_scene_graph.py calls evaluate()
directly when its input dataset ships ground-truth poses.

Reports ATE (Umeyama-aligned, with and without solving for scale — a
depth-estimated pipeline is only scale-correct up to its depth's own scale)
and frame-to-frame RPE.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def load_poses(poses_dir: Path) -> dict[int, np.ndarray]:
    poses: dict[int, np.ndarray] = {}
    for pose_file in sorted(poses_dir.glob("*.txt")):
        T = np.loadtxt(pose_file)
        if T.shape == (4, 4) and np.isfinite(T).all():
            poses[int(pose_file.stem)] = T
    return poses


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool) -> tuple[np.ndarray, np.ndarray, float]:
    """Least-squares similarity transform mapping src (N,3) onto dst (N,3)."""
    mu_src, mu_dst = src.mean(0), dst.mean(0)
    sc, dc = src - mu_src, dst - mu_dst
    U, D, Vt = np.linalg.svd(dc.T @ sc / len(src))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = float(D @ np.diag(S).T / (sc**2).sum(axis=1).mean()) if with_scale else 1.0
    t = mu_dst - scale * R @ mu_src
    return R, t, scale


def rotation_angles_deg(R_err: np.ndarray) -> np.ndarray:
    """Geodesic angle of a stack of (N,3,3) rotation errors."""
    trace = np.trace(R_err, axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))


def stats(errs: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt((errs**2).mean())),
        "mean": float(errs.mean()),
        "median": float(np.median(errs)),
        "std": float(errs.std()),
        "min": float(errs.min()),
        "max": float(errs.max()),
    }


def align_trajectories(gt: dict[int, np.ndarray], est: dict[int, np.ndarray], with_scale: bool = True) -> dict:
    """Per-frame ground truth vs estimate, with the estimate brought into the
    ground-truth frame (Umeyama). Returns positions and per-frame errors so a
    trajectory can be plotted and inspected frame by frame."""
    frames = sorted(set(gt) & set(est))
    gt_T = np.stack([gt[i] for i in frames])
    est_T = np.stack([est[i] for i in frames])

    R, t, s = umeyama(est_T[:, :3, 3], gt_T[:, :3, 3], with_scale)
    est_aligned_t = s * (est_T[:, :3, 3] @ R.T) + t
    est_aligned_R = R @ est_T[:, :3, :3]

    return {
        "frames": np.array(frames),
        "gt_positions": gt_T[:, :3, 3],
        "est_positions": est_aligned_t,
        "translation_error_m": np.linalg.norm(est_aligned_t - gt_T[:, :3, 3], axis=1),
        "rotation_error_deg": rotation_angles_deg(np.transpose(gt_T[:, :3, :3], (0, 2, 1)) @ est_aligned_R),
        "scale": s,
    }


def write_per_frame_csv(path: Path, aligned: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["frame,gt_x,gt_y,gt_z,est_x,est_y,est_z,translation_error_m,rotation_error_deg"]
    for i, f in enumerate(aligned["frames"]):
        g, e = aligned["gt_positions"][i], aligned["est_positions"][i]
        rows.append(f"{f},{g[0]:.6f},{g[1]:.6f},{g[2]:.6f},{e[0]:.6f},{e[1]:.6f},{e[2]:.6f},"
                    f"{aligned['translation_error_m'][i]:.6f},{aligned['rotation_error_deg'][i]:.4f}")
    path.write_text("\n".join(rows) + "\n")


def evaluate(gt: dict[int, np.ndarray], est: dict[int, np.ndarray]) -> dict:
    frames = sorted(set(gt) & set(est))
    if len(frames) < 3:
        raise SystemExit(f"only {len(frames)} overlapping frames between GT and estimate — nothing to evaluate")

    gt_T = np.stack([gt[i] for i in frames])
    est_T = np.stack([est[i] for i in frames])
    gt_t, est_t = gt_T[:, :3, 3], est_T[:, :3, 3]

    result: dict = {
        "frames_evaluated": len(frames),
        "frames_gt": len(gt),
        "frames_est": len(est),
        "gt_trajectory_length_m": float(np.linalg.norm(np.diff(gt_t, axis=0), axis=1).sum()),
    }

    for label, with_scale in (("ate_sim3", True), ("ate_se3", False)):
        R, t, s = umeyama(est_t, gt_t, with_scale)
        aligned_t = s * (est_t @ R.T) + t
        aligned_R = R @ est_T[:, :3, :3]
        result[label] = {
            "scale": s,
            "translation": stats(np.linalg.norm(aligned_t - gt_t, axis=1)),
            "rotation_deg": stats(rotation_angles_deg(np.transpose(gt_T[:, :3, :3], (0, 2, 1)) @ aligned_R)),
        }

    # RPE between consecutive evaluated frames — alignment-free, measures drift rate
    gt_rel = np.linalg.inv(gt_T[:-1]) @ gt_T[1:]
    est_rel = np.linalg.inv(est_T[:-1]) @ est_T[1:]
    rel_err = np.linalg.inv(gt_rel) @ est_rel
    result["rpe"] = {
        "translation_m": stats(np.linalg.norm(rel_err[:, :3, 3], axis=1)),
        "rotation_deg": stats(rotation_angles_deg(rel_err[:, :3, :3])),
    }
    return result


def format_report(result: dict) -> str:
    lines = [
        f"frames evaluated : {result['frames_evaluated']}  (gt={result['frames_gt']}, est={result['frames_est']})",
        f"GT path length   : {result['gt_trajectory_length_m']:.3f} m",
    ]
    for label, title in (("ate_sim3", "ATE (Sim3-aligned, scale solved)"), ("ate_se3", "ATE (SE3-aligned, scale fixed at 1)")):
        block = result[label]
        lines += [
            "",
            f"{title}   scale={block['scale']:.4f}",
            f"  translation [m]   rmse={block['translation']['rmse']:.4f}  mean={block['translation']['mean']:.4f}  "
            f"median={block['translation']['median']:.4f}  max={block['translation']['max']:.4f}",
            f"  rotation   [deg]  rmse={block['rotation_deg']['rmse']:.3f}  mean={block['rotation_deg']['mean']:.3f}  "
            f"median={block['rotation_deg']['median']:.3f}  max={block['rotation_deg']['max']:.3f}",
        ]
    rpe = result["rpe"]
    lines += [
        "",
        "RPE (consecutive evaluated frames)",
        f"  translation [m]   rmse={rpe['translation_m']['rmse']:.4f}  mean={rpe['translation_m']['mean']:.4f}  "
        f"median={rpe['translation_m']['median']:.4f}  max={rpe['translation_m']['max']:.4f}",
        f"  rotation   [deg]  rmse={rpe['rotation_deg']['rmse']:.3f}  mean={rpe['rotation_deg']['mean']:.3f}  "
        f"median={rpe['rotation_deg']['median']:.3f}  max={rpe['rotation_deg']['max']:.3f}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gt", type=Path, required=True, help="directory of ground-truth {frame:06d}.txt poses")
    parser.add_argument("--est", type=Path, required=True, help="directory of estimated {frame:06d}.txt poses")
    parser.add_argument("--out", type=Path, help="write the full metrics as JSON here")
    parser.add_argument("--csv", type=Path, help="write per-frame ground truth, estimate and error here")
    parser.add_argument("--plot", type=Path, help="directory for the trajectory-vs-ground-truth plot")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    gt, est = load_poses(args.gt), load_poses(args.est)
    result = evaluate(gt, est)
    result["gt_dir"] = str(args.gt)
    result["est_dir"] = str(args.est)

    print(format_report(result))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        logger.info("wrote %s", args.out)

    if args.csv or args.plot:
        aligned = align_trajectories(gt, est)
        if args.csv:
            write_per_frame_csv(args.csv, aligned)
            logger.info("wrote %s", args.csv)
        if args.plot:
            from perception.pose_estimation.visualize import save_trajectory_comparison

            logger.info("wrote %s", save_trajectory_comparison(
                args.plot, aligned,
                title=f"estimate vs ground truth — {len(aligned['frames'])} frames, "
                      f"ATE {result['ate_sim3']['translation']['rmse']*100:.1f} cm"))


if __name__ == "__main__":
    main()
