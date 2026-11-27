#!/usr/bin/env python3
"""Train DoVSG's ACE relocalizer on the memory capture (once per scene).

ACE learns, for the memory's images, where each pixel is in the memory map
(the gravity-aligned poses/ the voxel map is built in). The update uses it to
give each new frame a rough pose in that map.

Runs in the scene_graph env:

    python dynamic_updates/dovsg_method/train_ace.py --memory fyp_initial_v2 [--wandb]

Reads  data/processed/<memory>/{rgb,poses,metadata.json}
Writes data/processed/<memory>/ace/dataset/   (ACE's layout; portrait frames rotated, see ace_utils.py)
       data/processed/<memory>/ace/ace.pt     (~9 min on a Quadro RTX 5000)

Training time hardly depends on the number of frames: ACE fills a fixed
buffer of 8M pixel features (training_buffer_size) and trains 16 epochs over
it, 25,000 iterations of 5,120 pixels either way. DovSG measured 552 s for
125 frames and 552 s for 212.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ace_utils import make_ace_importable, working_dir, write_ace_dataset  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def allow_missing_pyrender() -> None:
    """ACE's trainer imports pyrender (ace_vis_util.py) at the top but only
    uses it for render_visualization, which DoVSG's train.yml turns off. The
    scene_graph env has no pyrender, so a placeholder stands in; using it fails."""
    try:
        import pyrender  # noqa: F401
    except ImportError:
        import types

        def missing(name):
            raise AttributeError(f"pyrender is not installed (needed only for ACE's render_visualization; "
                                 f"tried pyrender.{name})")

        stub = types.ModuleType("pyrender")
        stub.__getattr__ = missing
        sys.modules["pyrender"] = stub


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", required=True, help="scene name of the initial run (data/processed/<memory>)")
    ap.add_argument("--dovsg_root", type=Path, default=Path("/home/ravindu/SceneGraphNav/external/DovSG"))
    ap.add_argument("--wandb", action="store_true", help="log the training to Weights & Biases")
    ap.add_argument("--wandb_project", default="ace-relocalizer")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline"],
                    help="offline writes the run to disk; upload later with `wandb sync`")
    args = ap.parse_args()

    capture = REPO_ROOT / "data" / "processed" / args.memory
    if not (capture / "poses").is_dir():
        sys.exit(f"{capture}/poses not found: run the initial pipeline for '{args.memory}' first.")
    ace_dir = capture / "ace"
    dataset = ace_dir / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    n = write_ace_dataset(capture, dataset)
    print(f"ACE dataset: {n} frames in {dataset}")

    make_ace_importable(args.dovsg_root)
    allow_missing_pyrender()
    from ace.train_ace import train_ace

    t = time.time()
    with working_dir(args.dovsg_root):  # ace/configs/train.yml and the encoder are relative to DovSG's root
        wandb_cfg = ({"project": args.wandb_project, "name": f"ace-{args.memory}", "mode": args.wandb_mode,
                      "dir": str(ace_dir.resolve())} if args.wandb else None)
        train_ace(dataset.resolve(), (ace_dir / "ace.pt").resolve(), wandb_cfg)
    print(f"ACE model: {ace_dir / 'ace.pt'} ({time.time() - t:.0f} s)")


if __name__ == "__main__":
    main()
