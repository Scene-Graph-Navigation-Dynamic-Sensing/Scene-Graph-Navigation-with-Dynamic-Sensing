"""The streaming engine's pieces against the batch code they replace.

    python -m pytest dynamic_updates/covis_gate/online/test_online.py -q

The belief test is pure numpy and runs anywhere; the GPU tests skip without CUDA (run them on the server).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
GATE = HERE.parent
REPO = GATE.parents[1]
for p in (str(REPO), str(GATE.parent / "projection_method"), str(GATE), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import evidence as E  # noqa: E402
import theory as T  # noqa: E402
from belief import OnlineBelief  # noqa: E402

CFG = {"p_hit_present": 0.85, "p_hit_absent": 0.08, "rho_c": 0.2, "alpha_fixed": 0.0, "rho_f": 0.1, "l_max": 8.0,
       "gate": True, "update": "tempered", "background_weight": 0.7, "max_sweeps": 10, "next_to_weight": 0.6,
       "sigma0_m": 0.005, "sigma_k1": 0.06, "sigma_k2": 0.01, "max_depth_m": 6.0, "residual": "signed",
       "colour_refine": True, "colour_thresh": 0.45}


def _random_counts(rng, F, n, groups):
    n_hit = rng.integers(0, 300, (F, n))
    n_thr = rng.integers(0, 60, (F, n)) * (rng.random((F, n)) < 0.5)
    n_thr[:, 2] = rng.integers(0, 200, F)            # object 2 is being removed: lots of through
    n_hit[:, 2] = rng.integers(0, 20, F)
    n_valid = n_hit + n_thr + rng.integers(0, 80, (F, n))
    bg_valid = {i: rng.integers(50, 400, F) for i in groups}
    bg_hit = {i: (bg_valid[i] * rng.uniform(0.5, 1.0)).astype(int) for i in groups}
    return {"n_valid": n_valid, "n_hit": n_hit, "n_thr": n_thr, "bg_valid": bg_valid, "bg_hit": bg_hit}


def _setup(seed=0, n=8, F=70):
    rng = np.random.default_rng(seed)
    groups = [3, 4, 5, 6, 7]
    counts = _random_counts(rng, F, n, groups)
    prior_L = rng.normal(1.0, 1.0, n)
    is_anchor = np.array([True, True, False, False, False, False, False, False])
    nbrs = {i: [] for i in range(n)}
    for a, b, w in [(2, 3, 0.9), (3, 2, 0.9), (3, 4, 0.8), (4, 3, 0.8), (2, 0, 0.8), (5, 6, 0.6), (6, 5, 0.6), (7, 1, 0.9)]:
        nbrs[a].append((b, w, "test"))
    return counts, prior_L, is_anchor, nbrs, groups


def _feed(ob, counts, groups, upto):
    for f in range(upto):
        ob.add({"n_valid": counts["n_valid"][f], "n_hit": counts["n_hit"][f], "n_thr": counts["n_thr"][f],
                "bg_valid": np.array([counts["bg_valid"][i][f] for i in groups]),
                "bg_hit": np.array([counts["bg_hit"][i][f] for i in groups])})


def test_online_belief_equals_batch_on_the_whole_video():
    counts, prior_L, is_anchor, nbrs, groups = _setup()
    batch = E.beliefs(counts, prior_L, is_anchor, nbrs, CFG)
    ob = OnlineBelief(prior_L, is_anchor, nbrs, groups, CFG, capacity=8)     # capacity 8: the buffers must grow
    _feed(ob, counts, groups, 70)
    sol = ob.solve()
    assert np.allclose(sol["L"], batch["L"], atol=1e-9)
    assert np.allclose(sol["rho"], batch["rho"], atol=1e-12)
    assert np.allclose(sol["trajectory"], batch["trajectory"], atol=1e-9)
    assert sol["sweeps"] == batch["sweeps"]


@pytest.mark.parametrize("t", [1, 5, 23, 51])
def test_online_belief_after_t_keyframes_is_the_batch_belief_of_those_keyframes(t):
    counts, prior_L, is_anchor, nbrs, groups = _setup(seed=3)
    prefix = {k: (v[:t] if not isinstance(v, dict) else {i: a[:t] for i, a in v.items()}) for k, v in counts.items()}
    batch = E.beliefs(prefix, prior_L, is_anchor, nbrs, CFG)
    ob = OnlineBelief(prior_L, is_anchor, nbrs, groups, CFG)
    _feed(ob, counts, groups, t)
    assert np.allclose(ob.solve()["L"], batch["L"], atol=1e-9)


def test_online_belief_ablations_follow_the_batch():
    counts, prior_L, is_anchor, nbrs, groups = _setup(seed=5)
    for change in ({"gate": False}, {"update": "mixture"}, {"alpha_fixed": 0.3}):
        cfg = {**CFG, **change}
        batch = E.beliefs(counts, prior_L, is_anchor, nbrs, cfg)
        ob = OnlineBelief(prior_L, is_anchor, nbrs, groups, cfg)
        _feed(ob, counts, groups, 70)
        assert np.allclose(ob.solve()["L"], batch["L"], atol=1e-9), change


def test_no_frames_gives_the_prior():
    counts, prior_L, is_anchor, nbrs, groups = _setup()
    sol = OnlineBelief(prior_L, is_anchor, nbrs, groups, CFG).solve()
    assert np.allclose(sol["L"], prior_L)


# ── GPU ──

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:          # the laptop has no torch: the numpy tests above still run
    torch, HAS_CUDA = None, False
cuda = pytest.mark.skipif(not HAS_CUDA, reason="needs CUDA")


def _scene(rng):
    """A wall of memory voxels 1.5 m in front of the camera, two 'objects' on it, and a frame whose depth sees the wall,
    sees through one object (a removed one) and has an occluder in front of a corner."""
    xs, ys = np.meshgrid(np.arange(-0.4, 0.4, 0.01), np.arange(-0.3, 0.3, 0.01))
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 1.5)], 1)
    cols = rng.uniform(0.2, 0.8, (len(pts), 3))
    owner = np.full(len(pts), -1)
    owner[(pts[:, 0] < -0.1) & (pts[:, 0] > -0.3) & (np.abs(pts[:, 1]) < 0.1)] = 0
    owner[(pts[:, 0] > 0.1) & (pts[:, 0] < 0.3) & (np.abs(pts[:, 1]) < 0.1)] = 1
    return pts, cols, owner


@cuda
def test_gpu_evidence_counts_equal_the_numpy_classify_frame():
    import gpu_evidence as G

    rng = np.random.default_rng(1)
    pts, cols, owner = _scene(rng)
    h, w = 120, 160
    K = np.array([[200.0, 0, 80], [0, 200.0, 60], [0, 0, 1]])
    v, u = np.mgrid[0:h, 0:w]
    depth = np.full((h, w), 1.5)
    depth[:, 100:130] = 1.8                                  # behind the memory's wall: through for an object there
    depth[10:30, 20:40] = 1.0                                # something in front: occluded
    depth += rng.normal(0, 0.004, depth.shape)
    rgb = np.broadcast_to(np.array([0.5, 0.4, 0.3]), (h, w, 3)).copy()
    pose = np.eye(4)
    valid = np.ones((h, w), bool)
    cfg = dict(CFG)
    ii, r, c, out = E.classify_frame({"valid": valid, "depth": depth, "rgb": rgb, "pose": pose, "intrinsic": K}, pts, cols, cfg)
    own = owner[ii]
    want = {"n_valid": np.bincount(own[own >= 0], minlength=2), "n_hit": np.bincount(own[(own >= 0) & (out == T.HIT)], minlength=2),
            "n_thr": np.bincount(own[(own >= 0) & (out == T.THROUGH)], minlength=2)}
    gm = G.GpuMemory(pts, cols, owner, 2, {}, cfg, "cuda")
    got = gm.process(torch.as_tensor(rgb, dtype=torch.float32).cuda(), torch.as_tensor(depth, dtype=torch.float32).cuda(),
                     torch.as_tensor(valid).cuda(), pose, K, 1.0)
    for k in want:
        assert np.array_equal(got[k], want[k]), (k, got[k], want[k])
    assert got["n_thr"].sum() > 0 and got["n_hit"].sum() > 0


@cuda
def test_gpu_edge_free_mask_equals_cv2():
    import gpu_evidence as G

    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(2)
    z = rng.uniform(0.5, 3.0, (60, 80)).astype(np.float32)
    z[:, 40:] += 1.0                                         # a depth edge
    valid = rng.random(z.shape) > 0.05
    zz = np.where(valid, z, 0).astype(np.float32)
    k = np.ones((5, 5), np.uint8)
    spread = cv2.dilate(zz, k) - cv2.erode(np.where(valid, zz, 1e6).astype(np.float32), k)
    want = valid & (spread < 0.03 * np.maximum(zz, 1e-6))
    got = G.edge_free_mask_gpu(torch.as_tensor(z).cuda(), torch.as_tensor(valid).cuda()).cpu().numpy()
    assert np.array_equal(got, want)


@cuda
def test_occupancy_is_a_sphere_dilation():
    import gpu_evidence as G

    pts = np.array([[0.5, 0.5, 0.5]])
    occ = G.Occupancy(pts, np.zeros(3), 0.01, np.array([100, 100, 100]), 0.03, torch.device("cuda"))
    near = occ.near(torch.tensor([[0.5, 0.5, 0.5], [0.52, 0.5, 0.5], [0.56, 0.5, 0.5], [2.0, 0.5, 0.5]],
                                 dtype=torch.float64, device="cuda")).cpu().numpy()
    assert near.tolist() == [True, True, False, False]
