"""Checks of the Dynamic_Attention pieces.

    python -m pytest dynamic_updates/Dynamic_Attention/test_dynamic_attention.py -q

  - with no presence term, AttentionBelief is exactly covis_gate's OnlineBelief (so the attention path changes nothing
    unless the head speaks), and a negative presence term lowers P only through the same gate
  - Sinkhorn with dustbins: every real row and column of the assignment sums to 1
  - the matcher is permutation-equivariant (tracks in a different order -> rows in that order)
  - the presence head ignores padded patches
  - footprint(): visible / occluded / behind shares on a synthetic plane
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

C.setup_paths()
sys.path.insert(0, str(C.GATE / "online"))
from belief import OnlineBelief  # noqa: E402
from test_online import CFG, _feed, _setup  # noqa: E402

from models import AttentionMatcher, PresenceHead, log_optimal_transport  # noqa: E402


def _att_belief():
    from online_attention import AttentionBelief

    return AttentionBelief


def test_belief_without_presence_equals_online_belief():
    AB = _att_belief()
    counts, prior_L, is_anchor, nbrs, groups = _setup()
    a = OnlineBelief(prior_L, is_anchor, nbrs, groups, CFG, capacity=8)
    b = AB(prior_L, is_anchor, nbrs, groups, CFG, capacity=8)
    _feed(a, counts, groups, 70)
    _feed(b, counts, groups, 70)
    sa, sb = a.solve(), b.solve()
    assert np.allclose(sa["L"], sb["L"], atol=1e-12)
    assert np.allclose(sa["trajectory"], sb["trajectory"], atol=1e-12)
    assert sa["sweeps"] == sb["sweeps"]


def test_negative_presence_lowers_target_belief():
    AB = _att_belief()
    counts, prior_L, is_anchor, nbrs, groups = _setup()
    counts["n_hit"][:, 5], counts["n_thr"][:, 5] = 0, 0                 # object 5: no depth evidence, L stays at its prior
    cfg = dict(CFG, gate=False)
    base = AB(prior_L, is_anchor, nbrs, groups, cfg)
    att = AB(prior_L, is_anchor, nbrs, groups, cfg)
    _feed(base, counts, groups, 70)
    for f in range(70):
        att.add({"n_valid": counts["n_valid"][f], "n_hit": counts["n_hit"][f], "n_thr": counts["n_thr"][f],
                 "bg_valid": np.array([counts["bg_valid"][i][f] for i in groups]),
                 "bg_hit": np.array([counts["bg_hit"][i][f] for i in groups])},
                np.where(np.arange(8) == 5, -2.0, 0.0), np.arange(8) == 5)
    s0, s1 = base.solve(), att.solve()
    assert s1["L"][5] < s0["L"][5]
    assert abs(s0["L"][5]) < 7.9 and s1["L"][5] < s0["L"][5] - 1.0
    assert np.allclose(s1["L"][[0, 1]], s0["L"][[0, 1]])           # the anchors are unchanged


def test_sinkhorn_marginals():
    torch.manual_seed(0)
    lp = log_optimal_transport(torch.randn(5, 7), torch.tensor(1.0), 100)
    P = lp.exp()
    assert torch.allclose(P[:5].sum(1), torch.ones(5), atol=1e-3)
    assert torch.allclose(P[:, :7].sum(0), torch.ones(7), atol=1e-3)


def _side(n, g):
    return {"emb": torch.nn.functional.normalize(torch.randn(n, 1024, generator=g), dim=1),
            "cen": torch.randn(n, 3, generator=g), "ext": torch.rand(n, 3, generator=g) + 0.05,
            "pts": torch.randn(n, 256, 3, generator=g) * 0.1}


def test_matcher_permutation_equivariant():
    g = torch.Generator().manual_seed(0)
    net = AttentionMatcher().eval()
    T, M = _side(4, g), _side(6, g)
    perm = torch.tensor([2, 0, 3, 1])
    pair = torch.randn(4, 6, AttentionMatcher.PAIR_DIM, generator=g)
    with torch.no_grad():
        a = net(T, M, pair)
        b = net({k: v[perm] for k, v in T.items()}, M, pair[perm])
    assert torch.allclose(a[:4][perm], b[:4], atol=1e-4)


def test_pair_features_overlap():
    from models import pair_features

    rng = np.random.default_rng(0)
    table = np.c_[rng.uniform(0, 1, (2000, 2)), np.zeros(2000)]
    on_table = np.c_[rng.uniform(0.4, 0.5, (200, 2)), rng.uniform(0.0, 0.1, 200)]   # an object lying on it
    e = np.eye(3, 1024)
    f = pair_features([on_table, table], [table], e[:2], e[:1])
    assert f[0, 0, 1] > 0.3 and f[0, 0, 2] < 0.05          # most of it touches the table; almost none of the table touches it
    assert f[1, 0, 1] > 0.99 and f[1, 0, 2] > 0.99


def test_presence_ignores_padding():
    g = torch.Generator().manual_seed(0)
    net = PresenceHead().eval()
    emb = torch.nn.functional.normalize(torch.randn(1, 1024, generator=g), dim=1)
    md = torch.randn(1, C.FEAT_DIM, generator=g)
    pf = torch.randn(1, 10, C.FEAT_DIM, generator=g)
    occ = torch.rand(1, 10, generator=g)
    pad = torch.zeros(1, 10, dtype=torch.bool)
    pf2 = torch.cat([pf, torch.randn(1, 6, C.FEAT_DIM, generator=g)], 1)
    occ2 = torch.cat([occ, torch.rand(1, 6, generator=g)], 1)
    pad2 = torch.cat([pad, torch.ones(1, 6, dtype=torch.bool)], 1)
    with torch.no_grad():
        assert torch.allclose(net(emb, md, pf, occ, pad), net(emb, md, pf2, occ2, pad2), atol=1e-5)


def test_footprint_shares():
    K = np.array([[100.0, 0, 64], [0, 100.0, 48], [0, 0, 1]])
    xs, ys = np.meshgrid(np.linspace(-0.3, 0.3, 30), np.linspace(-0.2, 0.2, 20))
    pts = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 2.0)], 1)
    c2w = np.eye(4)
    for d, key in ((2.0, "vis"), (1.0, "occ"), (3.0, "behind")):
        fp = C.footprint(pts, c2w, K, np.full((96, 128), d, np.float32), None)
        assert fp is not None and np.allclose(fp[key], 1.0)
