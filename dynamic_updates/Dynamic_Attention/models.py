"""The two attention modules.

AttentionMatcher (association: same object / new / moved)
    tokens  = appearance MLP(CLIP 1024) + position MLP(Fourier(centroid, log extent)) + shape PointNet(256 voxels)
              + a side embedding (new track or memory object)
    3 blocks of self-attention inside each set and cross-attention both ways (LightGlue style)
    score matrix + learned dustbin -> log-domain Sinkhorn (SuperGlue's optimal transport) -> (T+1, M+1) assignment

PresenceHead (removal evidence)
    query   = MLP(memory CLIP embedding) + MLP(memory patch descriptor: the object's own YOLOE features in a memory frame)
    keys    = the current keyframe's YOLOE patches inside the object's projected footprint, each with its occluded share
              (from depth) and its cosine to the memory descriptor
    2 masked cross-attention blocks -> 3 logits: present / absent / occluded
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(i: int, h: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def fourier(x: torch.Tensor, n_freq: int = 6) -> torch.Tensor:
    freqs = (2.0 ** torch.arange(n_freq, device=x.device, dtype=x.dtype)) * math.pi / 4
    ang = x[..., None] * freqs
    return torch.cat([x, torch.sin(ang).flatten(-2), torch.cos(ang).flatten(-2)], -1)


class TinyPointNet(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU(), nn.Linear(128, d))

    def forward(self, pts: torch.Tensor) -> torch.Tensor:      # (n, P, 3), centred, metres
        return self.f(pts * 4.0).max(1).values


class MatchBlock(nn.Module):
    def __init__(self, d: int, h: int, dropout: float = 0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, batch_first=True, dropout=dropout)
        self.ca = nn.MultiheadAttention(d, h, batch_first=True, dropout=dropout)
        self.ff = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def self_attn(self, x):
        y = self.n1(x)[None]
        return x + self.sa(y, y, y, need_weights=False)[0][0]

    def cross(self, x, ctx):
        q, kv = self.n2(x)[None], self.n2(ctx)[None]
        y = self.ca(q, kv, kv, need_weights=False)[0][0]
        return x + self.ff(torch.cat([x, y], -1))

    def forward(self, a, b):
        a, b = self.self_attn(a), self.self_attn(b)
        return self.cross(a, b), self.cross(b, a)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """SuperGlue's Sinkhorn with a dustbin row and column, in log space. scores (m, n) -> (m+1, n+1) log P."""
    m, n = scores.shape
    one = scores.new_tensor(1.0)
    ms, ns = m * one, n * one
    couplings = torch.cat([torch.cat([scores, alpha.expand(m, 1)], 1), torch.cat([alpha.expand(1, n), alpha.view(1, 1)], 1)], 0)
    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(couplings + v[None, :], 1)
        v = log_nu - torch.logsumexp(couplings + u[:, None], 0)
    return couplings + u[:, None] + v[None, :] - norm


class AttentionMatcher(nn.Module):
    PAIR_DIM = 5

    def __init__(self, d: int = 128, heads: int = 4, layers: int = 3, emb_dim: int = 1024, n_freq: int = 6,
                 sinkhorn_iters: int = 20):
        super().__init__()
        # pairwise evidence the rule uses, as an input to the score (v2): cosine, track->memory overlap,
        # memory->track overlap, centroid distance, log size ratio. The attention score corrects it.
        self.pair = nn.Sequential(nn.Linear(self.PAIR_DIM, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        self.d, self.iters, self.n_freq = d, sinkhorn_iters, n_freq
        self.app = mlp(emb_dim, 2 * d, d)
        self.pos = mlp(6 * (1 + 2 * n_freq), 2 * d, d)
        self.shape = TinyPointNet(d)
        self.side = nn.Embedding(2, d)
        self.blocks = nn.ModuleList([MatchBlock(d, heads) for _ in range(layers)])
        self.final = nn.Linear(d, d)
        self.bin_score = nn.Parameter(torch.tensor(1.0))

    def tokens(self, s: dict, side: int) -> torch.Tensor:
        pos = fourier(torch.cat([s["cen"], torch.log(s["ext"].clamp_min(0.01))], -1), self.n_freq)
        return self.app(s["emb"]) + self.pos(pos) + self.shape(s["pts"]) + self.side.weight[side]

    def forward(self, tracks: dict, memory: dict, pair: torch.Tensor | None = None) -> torch.Tensor:
        """Each dict: emb (n, 1024) unit, cen (n, 3) metres relative to the window centre, ext (n, 3) box size in metres,
        pts (n, P, 3) sampled voxels minus their centroid; pair (T, M, 5) from pair_features(). Returns log P (T+1, M+1); the last row / column is the dustbin."""
        a, b = self.tokens(tracks, 0), self.tokens(memory, 1)
        for blk in self.blocks:
            a, b = blk(a, b)
        sc = self.final(a) @ self.final(b).T / self.d ** 0.5
        if pair is not None:
            sc = sc + self.pair(pair).squeeze(-1)
        return log_optimal_transport(sc, self.bin_score, self.iters)


class PresenceBlock(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.nq, self.nk = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, h, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q, k, pad):
        y = self.ca(self.nq(q), self.nk(k), self.nk(k), key_padding_mask=pad, need_weights=False)[0]
        q = q + y
        return q + self.ff(q)


class PresenceHead(nn.Module):
    CLASSES = ("present", "absent", "occluded")

    def __init__(self, feat_dim: int = 384, d: int = 128, heads: int = 4, emb_dim: int = 1024, layers: int = 2):
        super().__init__()
        self.fn = nn.LayerNorm(feat_dim)
        self.q_app = mlp(emb_dim, 2 * d, d)
        self.q_mem = mlp(feat_dim, 2 * d, d)
        self.k_feat = nn.Linear(feat_dim, d)
        self.k_geo = mlp(2, 32, d)                 # occluded share (from depth) + cosine to the memory descriptor
        self.blocks = nn.ModuleList([PresenceBlock(d, heads) for _ in range(layers)])
        self.out = mlp(d + 4, d, 3)
        self.register_buffer("temperature", torch.tensor(1.0))

    def forward(self, emb, mem_desc, patches, occ, pad):
        """emb (B, 1024), mem_desc (B, F), patches (B, P, F), occ (B, P) share of the patch's voxels with something in
        front (from depth; the head gets no hit / through counts, those are CoViS's own evidence), pad (B, P) True = empty.
        Returns logits (B, 3), divided by the calibrated temperature."""
        pf, md = self.fn(patches.float()), self.fn(mem_desc.float())
        cos = F.cosine_similarity(pf, md[:, None, :], dim=-1)                       # (B, P)
        g = torch.stack([occ.float(), cos], -1)
        k = self.k_feat(pf) + self.k_geo(g)
        q = (self.q_app(emb.float()) + self.q_mem(md))[:, None, :]
        for blk in self.blocks:
            q = blk(q, k, pad)
        w = (~pad).float()
        n = w.sum(1).clamp_min(1)
        stats = torch.stack([(cos * w).sum(1) / n, cos.masked_fill(pad, -1).max(1).values,
                             (occ.float() * w).sum(1) / n, torch.log(n) / 4], 1)
        return self.out(torch.cat([q[:, 0], stats], 1)) / self.temperature


def pair_features(t_pts: list, m_pts: list, t_emb, m_emb, radius: float = 0.05, n: int = 256, seed: int = 0):
    """(T, M, 5) float32: cosine, share of track voxels within `radius` of the memory object, the reverse share,
    centroid distance (m, clipped at 3), log of the size ratio. One KD-tree per side over every object's points
    (k nearest within the radius tell which objects are near)."""
    import numpy as np
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    sub = lambda p, k: p if len(p) <= k else p[rng.choice(len(p), k, replace=False)]
    tq, mq = [sub(np.asarray(p, np.float64), n) for p in t_pts], [sub(np.asarray(p, np.float64), n) for p in m_pts]
    tf, mf = [sub(np.asarray(p, np.float64), 16 * n) for p in t_pts], [sub(np.asarray(p, np.float64), 16 * n) for p in m_pts]
    tp, mp = tf, mf

    def share(a_list, b_list):
        """a_list: query points (n per object); b_list: the searched side, kept dense (up to 16 n per object)."""
        allb = np.concatenate(b_list)
        own = np.concatenate([np.full(len(p), k) for k, p in enumerate(b_list)])
        tree = cKDTree(allb)
        out = np.zeros((len(a_list), len(b_list)))
        for i, p in enumerate(a_list):
            d, idx = tree.query(p, k=8, distance_upper_bound=radius)
            hit = np.isfinite(d)
            near = np.zeros((len(p), len(b_list)), bool)
            rows = np.repeat(np.arange(len(p))[:, None], 8, 1)[hit]
            near[rows, own[idx[hit]]] = True
            out[i] = near.mean(0)
        return out

    g_tm = share(tq, mf)
    g_mt = share(mq, tf).T
    tc, mc = np.array([p.mean(0) for p in tp]), np.array([p.mean(0) for p in mp])
    dist = np.minimum(np.linalg.norm(tc[:, None] - mc[None], axis=-1), 3.0)
    size = lambda p: np.log(np.prod(np.maximum(np.percentile(p, 95, 0) - np.percentile(p, 5, 0), 0.01)))
    ts, ms = np.array([size(p) for p in tp]), np.array([size(p) for p in mp])
    cos = np.asarray(t_emb, np.float64) @ np.asarray(m_emb, np.float64).T
    return np.stack([cos, g_tm, g_mt, dist, np.clip(ts[:, None] - ms[None], -5, 5) / 5], -1).astype(np.float32)
