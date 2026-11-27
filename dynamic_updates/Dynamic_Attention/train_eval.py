"""Train the attention matcher and the presence head on Replica, and test both against the current rules.

Split: train office_0-3, room_0, room_1; office_4 = validation (presence-head temperature, matcher thresholds);
room_2 = test, never seen. The lap / fyp videos are never used here.

Matcher samples (a detection window): memory objects within 2 m of a centre (memory visit), tracks within 2 m (new
visit, a random 50-100 % of them). Changes are simulated per sample:
  removed  a track is dropped              -> its memory object has no partner (dustbin row)
  added    a memory object is dropped      -> its track has no partner (dustbin column), and its voxels leave the map
  moved    a track is moved 0.5-1.8 m      -> same identity, new place
Training adds monocular-style noise on the track side (scale +-8 %, 2 cm / 2 deg pose jitter, voxel dropout).

The rule baseline is decide.associate's logic on the same samples: s = 0.3 overlap + 0.7 cosine >= 0.75 (overlap >= 0.2)
explains a memory object; otherwise novelty >= 0.4 -> unexplained; Hungarian over absent x unexplained with
0.3 shape + 0.7 cosine >= 0.75 -> moved; the rest of the unexplained -> added. Both methods get the same oracle "absent"
set (the memory objects whose track was removed or moved: what the CoViS belief provides) and the same novelty veto.

  python dynamic_updates/Dynamic_Attention/train_eval.py [--matcher_steps 6000 --presence_steps 3000]
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np

import common as C

C.setup_paths()
logger = logging.getLogger("dynamic_attention")
TRAIN = ["office_0", "office_1", "office_2", "office_3", "room_0", "room_1"]
VAL, TEST = ["office_4"], ["room_2"]
R_WINDOW, N_PTS = 2.0, 256


def load(scenes):
    import torch

    return {s: torch.load(C.DATA_DIR / f"{s}.pt", weights_only=False) for s in scenes}


# ─────────────────────────── matcher samples ───────────────────────────

def make_sample(d, rng, train: bool, max_mem=64, max_trk=16):
    A, B = d["obs"]["A"], d["obs"]["B"]
    gts = np.array([o["gt"] for o in d["objects"]])
    ia = [i for i, o in enumerate(A) if o is not None]
    ib = [i for i, o in enumerate(B) if o is not None]
    if not ia or not ib:
        return None
    c = B[ib[rng.integers(len(ib))]]["cen"].astype(np.float64)
    mem = sorted([i for i in ia if np.linalg.norm(A[i]["cen"][:2] - c[:2]) < R_WINDOW],
                 key=lambda i: np.linalg.norm(A[i]["cen"] - c))[:max_mem]
    trk = [i for i in ib if np.linalg.norm(B[i]["cen"][:2] - c[:2]) < R_WINDOW]
    trk = [i for i in trk if rng.random() < rng.uniform(0.5, 1.0)][:max_trk]
    if len(trk) < 1 or len(mem) < 2:
        return None
    both = [i for i in trk if i in mem]
    rng.shuffle(both)
    n_rem, n_add = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    removed = both[:n_rem]
    added = both[n_rem:n_rem + n_add]
    moved = []
    rest = both[n_rem + n_add:]
    if rest and rng.random() < 0.4:
        moved = [rest[0]]
    trk = [i for i in trk if i not in removed]
    mem = [i for i in mem if i not in added]
    if not trk or len(mem) < 2:
        return None
    th = rng.uniform(0, 2 * np.pi)
    Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    scale = rng.uniform(0.92, 1.08) if train else 1.0
    jit = rng.normal(0, 2 / 180 * np.pi) if train else 0.0
    Rj = np.array([[np.cos(jit), -np.sin(jit), 0], [np.sin(jit), np.cos(jit), 0], [0, 0, 1]])
    tj = rng.normal(0, 0.02, 3) if train else rng.normal(0, 0.01, 3)
    shift = {}
    for i in moved:
        a = rng.uniform(0, 2 * np.pi)
        r = rng.uniform(0.5, 1.8)
        shift[i] = np.array([r * np.cos(a), r * np.sin(a), rng.uniform(-0.3, 0.3)])

    def side(ids, obs, is_trk):
        cens, exts, pts, embs, world = [], [], [], [], []
        for i in ids:
            o = obs[i]
            p = o["pts"].astype(np.float64)
            if is_trk and train and rng.random() < 0.7:
                p = p[rng.random(len(p)) < rng.uniform(0.3, 1.0)] if len(p) > 40 else p
            if is_trk:
                p = (p - c) @ Rj.T * scale + c + tj + shift.get(i, 0)
            cen = p.mean(0)
            pw = p.copy()
            lo, hi = np.percentile(p, 5, 0), np.percentile(p, 95, 0)
            q = p[rng.choice(len(p), N_PTS, replace=len(p) < N_PTS)] - cen
            e = o["emb"].astype(np.float64)
            if is_trk and train:
                e = e + rng.normal(0, 0.01, e.shape)
                e /= np.linalg.norm(e)
            cens.append(Rz @ (cen - c)); exts.append(np.maximum(hi - lo, 0.01)); pts.append(q @ Rz.T); embs.append(e)
            world.append(pw)
        return {"cen": np.array(cens), "ext": np.array(exts), "pts": np.array(pts), "emb": np.array(embs)}, world

    T_, Tw = side(trk, B, True)
    M_, Mw = side(mem, A, False)
    m_of = {i: k for k, i in enumerate(mem)}
    lab_t = np.array([m_of.get(i, -1) for i in trk])
    lab_m = np.array([(trk.index(i) if i in trk else -1) for i in mem])
    ign_t = np.array([lab_t[k] < 0 and gts[i] >= 0 and any(gts[j] == gts[i] for j in mem) for k, i in enumerate(trk)])
    ign_m = np.array([lab_m[k] < 0 and gts[i] >= 0 and any(gts[j] == gts[i] for j in trk) for k, i in enumerate(mem)])
    absent = [m_of[i] for i in moved] + [k for k, i in enumerate(mem) if i in removed]
    truth = []
    for k, i in enumerate(trk):
        if ign_t[k]:
            truth.append(("ignore", None))
        elif lab_t[k] < 0:
            truth.append(("added", None))
        elif i in moved:
            truth.append(("moved", int(lab_t[k])))
        else:
            truth.append(("same", int(lab_t[k])))
    from models import pair_features

    pair = pair_features(Tw, Mw, T_["emb"], M_["emb"], seed=int(rng.integers(1 << 30)))
    return {"T": T_, "M": M_, "Tw": Tw, "Mw": Mw, "pair": pair, "lab_t": lab_t, "lab_m": lab_m, "ign_t": ign_t, "ign_m": ign_m,
            "absent": absent, "truth": truth, "removed_mem": [m_of[i] for i in removed if i in m_of]}


def to_torch(s, dev):
    import torch

    def t(x):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=dev) for k, v in x.items()}

    return t(s["T"]), t(s["M"]), torch.as_tensor(s["pair"], device=dev)


def match_loss(logP, s):
    import torch

    T, M = len(s["lab_t"]), len(s["lab_m"])
    terms = []
    for k in range(T):
        if s["ign_t"][k]:
            continue
        terms.append(-logP[k, s["lab_t"][k] if s["lab_t"][k] >= 0 else M])
    for k in range(M):
        if s["lab_m"][k] < 0 and not s["ign_m"][k]:
            terms.append(-logP[T, k])
    return torch.stack(terms).mean() if terms else logP.sum() * 0


# ─────────────────────────── decisions ───────────────────────────

def novelty(track_world, mem_world, dist=0.05):
    from scipy.spatial import cKDTree

    if not mem_world:
        return 1.0
    tree = cKDTree(np.concatenate(mem_world))
    d, _ = tree.query(track_world, k=1, distance_upper_bound=dist)
    return float(np.isinf(d).mean())


def rule_decide(s, match_threshold=0.75, min_geo=0.2, min_novelty=0.4, delta=0.75, shape_dist=0.05):
    """decide.associate on a sample (receptacle prior = 1: no learned rates in Replica)."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    import decide as D

    trees = [cKDTree(p) for p in s["Mw"]]
    me = s["M"]["emb"]
    out = []
    unexplained = []
    keep_map = [p for k, p in enumerate(s["Mw"]) if k not in s["absent"]]
    for u, p in enumerate(s["Tw"]):
        cos = me @ s["T"]["emb"][u]
        geo = np.array([np.isfinite(t.query(p, k=1, distance_upper_bound=0.05)[0]).mean() for t in trees])
        sc = np.where(geo >= min_geo, 0.3 * geo + 0.7 * cos, -1)
        j = int(np.argmax(sc))
        if sc[j] >= match_threshold:
            out.append(("same", j))
        elif novelty(p, keep_map) >= min_novelty:
            out.append(("unexplained", None)); unexplained.append(u)
        else:
            out.append(("in_map", None))
    absent = list(s["absent"])
    for k, (lab, j) in enumerate(out):                 # re-detected in place
        if lab == "same" and j in absent:
            out[k] = ("same", j)
    if absent and unexplained:
        S = np.zeros((len(absent), len(unexplained)))
        for a, i in enumerate(absent):
            for b, u in enumerate(unexplained):
                S[a, b] = 0.3 * D.shape_overlap(s["Mw"][i], s["Tw"][u], shape_dist) + 0.7 * float(me[i] @ s["T"]["emb"][u])
        ra, cb = linear_sum_assignment(-S)
        for a, b in zip(ra, cb):
            if S[a, b] >= delta:
                out[unexplained[b]] = ("moved", absent[a])
    return [("added", None) if lab == "unexplained" else (lab, j) for lab, j in out]


def attention_decide(logP, s, tau_match=0.2, move_m=0.3, min_novelty=0.4):
    P = np.exp(logP.detach().cpu().numpy())
    T, M = len(s["Tw"]), len(s["Mw"])
    keep_map = [p for k, p in enumerate(s["Mw"]) if k not in s["absent"]]
    out = []
    for u in range(T):
        j = int(np.argmax(P[u, :M + 1]))
        if j < M and P[u, j] >= tau_match:
            if j in s["absent"] and np.linalg.norm(s["Tw"][u].mean(0) - s["Mw"][j].mean(0)) > move_m:
                out.append(("moved", j))
            else:
                out.append(("same", j))
        elif novelty(s["Tw"][u], keep_map) >= min_novelty:
            out.append(("added", None))
        else:
            out.append(("in_map", None))
    return out


def score(samples_preds):
    c = {k: 0 for k in ("same_ok", "same_n", "add_tp", "add_fp", "add_fn", "mov_tp", "mov_fp", "mov_fn",
                        "added_as_same", "removed_as_moved")}
    for s, pred in samples_preds:
        for (tl, tj), (pl, pj) in zip(s["truth"], pred):
            if tl == "ignore":
                continue
            if tl == "same":
                c["same_n"] += 1
                c["same_ok"] += int(pl == "same" and pj == tj)
            if tl == "added":
                c["add_tp" if pl == "added" else "add_fn"] += 1
                c["added_as_same"] += int(pl in ("same", "moved"))
            elif pl == "added":
                c["add_fp"] += 1
            if tl == "moved":
                if pl == "moved" and pj == tj:
                    c["mov_tp"] += 1
                else:
                    c["mov_fn"] += 1
            if pl == "moved" and not (tl == "moved" and pj == tj):
                c["mov_fp"] += 1
                c["removed_as_moved"] += int(pj in s["removed_mem"])

    def prf(tp, fp, fn):
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(p + r, 1e-9), 3), "tp": tp, "fp": fp, "fn": fn}

    return {"same_object_accuracy": round(c["same_ok"] / max(c["same_n"], 1), 3), "same_n": c["same_n"],
            "added": prf(c["add_tp"], c["add_fp"], c["add_fn"]), "moved": prf(c["mov_tp"], c["mov_fp"], c["mov_fn"]),
            "added_taken_as_existing": c["added_as_same"], "removed_paired_as_moved": c["removed_as_moved"]}


def fixed_samples(d, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = make_sample(d, rng, train=False)
        if s is not None:
            out.append(s)
    return out


def train_matcher(data, steps, dev, log):
    import torch

    from models import AttentionMatcher

    torch.manual_seed(0)
    net = AttentionMatcher().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(1)
    scenes = [s for s in TRAIN if s in data]
    t = time.time()
    run = []
    for step in range(steps):
        net.train()
        loss = 0
        n = 0
        while n < 8:
            s = make_sample(data[scenes[rng.integers(len(scenes))]], rng, train=True)
            if s is None:
                continue
            T_, M_, pr = to_torch(s, dev)
            loss = loss + match_loss(net(T_, M_, pr), s)
            n += 1
        loss = loss / n
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        run.append(float(loss))
        if (step + 1) % 500 == 0:
            log(f"[matcher] step {step + 1}/{steps} loss {np.mean(run[-500:]):.3f} ({time.time() - t:.0f}s)")
    return net, {"steps": steps, "seconds": round(time.time() - t, 1), "final_loss": round(float(np.mean(run[-500:])), 4)}


def eval_matcher(net, data, scenes, dev, n=400, tau=0.2):
    import torch

    res = {}
    net.eval()
    for sc in scenes:
        samples = fixed_samples(data[sc], n, seed=123)
        rule = [(s, rule_decide(s)) for s in samples]
        att = []
        t = time.time()
        with torch.no_grad():
            for s in samples:
                att.append((s, attention_decide(net(*to_torch(s, dev)), s, tau_match=tau)))
        ms = (time.time() - t) / len(samples) * 1000
        res[sc] = {"samples": n, "tracks": sum(len(s["truth"]) for s in samples),
                   "true_added": sum(t == "added" for s in samples for t, _ in s["truth"]),
                   "true_moved": sum(t == "moved" for s in samples for t, _ in s["truth"]),
                   "rule": score(rule), "attention": score(att), "attention_ms_per_window": round(ms, 2)}
    return res


# ─────────────────────────── presence head ───────────────────────────

def presence_batch(data, items, dev, train, rng, P_max=64):
    import torch

    B = len(items)
    emb = np.zeros((B, 1024), np.float32); md = np.zeros((B, C.FEAT_DIM), np.float32)
    pf = torch.zeros((B, P_max, C.FEAT_DIM), dtype=torch.float16, device=dev)
    occ = np.zeros((B, P_max), np.float32); pad = np.ones((B, P_max), bool); y = np.zeros(B, np.int64)
    for b, (sc, k) in enumerate(items):
        d = data[sc]; s = d["samples"][k]
        i = s["obj"]
        emb[b] = d["obs"]["A"][i]["emb"]; md[b] = d["mem_desc"][i]; y[b] = s["label"]
        p, o = s["patches"], s["occ"]
        if train and len(p) > 3:
            keep = rng.random(len(p)) < rng.uniform(0.5, 1.0)
            if keep.sum() >= 2:
                p, o = p[keep], o[keep]
        p, o = p[:P_max], o[:P_max]
        flat = d["feats_B"][s["slot"]].reshape(-1, C.FEAT_DIM)
        pf[b, :len(p)] = flat[torch.as_tensor(p, dtype=torch.long)].to(dev)
        occ[b, :len(p)] = np.clip(o + (rng.normal(0, 0.05, len(o)) if train else 0), 0, 1)
        pad[b, :len(p)] = False
    t = lambda x, dt=torch.float32: torch.as_tensor(x, dtype=dt, device=dev)
    return t(emb), t(md), pf, t(occ), t(pad, torch.bool), t(y, torch.long)


def presence_items(data, scenes):
    return [(sc, k) for sc in scenes for k, s in enumerate(data[sc]["samples"])
            if data[sc]["obs"]["A"][s["obj"]] is not None and data[sc]["mem_ok"][s["obj"]]]


def train_presence(data, steps, dev, log):
    import torch

    from models import PresenceHead

    torch.manual_seed(0)
    net = PresenceHead().to(dev)
    items = presence_items(data, [s for s in TRAIN if s in data])
    ys = np.array([data[sc]["samples"][k]["label"] for sc, k in items])
    freq = np.bincount(ys, minlength=3) / len(ys)
    wcls = torch.tensor(1 / np.maximum(freq, 1e-3), dtype=torch.float32, device=dev)
    wcls = wcls / wcls.sum() * 3
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, total_steps=steps, pct_start=0.05)
    rng = np.random.default_rng(2)
    t = time.time(); run = []
    for step in range(steps):
        net.train()
        batch = [items[k] for k in rng.integers(0, len(items), 64)]
        e, m, pf, o, pad, y = presence_batch(data, batch, dev, True, rng)
        loss = torch.nn.functional.cross_entropy(net(e, m, pf, o, pad), y, weight=wcls)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        run.append(float(loss))
        if (step + 1) % 500 == 0:
            log(f"[presence] step {step + 1}/{steps} loss {np.mean(run[-500:]):.3f} ({time.time() - t:.0f}s)")
    return net, {"steps": steps, "seconds": round(time.time() - t, 1), "train_items": len(items),
                 "train_class_share": {c: round(float(f), 3) for c, f in zip(("present", "absent", "occluded"), freq)}}


def presence_logits(net, data, scenes, dev):
    import torch

    items = presence_items(data, scenes)
    rng = np.random.default_rng(0)
    L, Y, kinds, cosm = [], [], [], []
    net.eval()
    with torch.no_grad():
        for k in range(0, len(items), 256):
            e, m, pf, o, pad, y = presence_batch(data, items[k:k + 256], dev, False, rng)
            L.append(net(e, m, pf, o, pad).float().cpu().numpy()); Y.append(y.cpu().numpy())
            pfn = torch.nn.functional.normalize(net.fn(pf.float()), dim=-1)
            mdn = torch.nn.functional.normalize(net.fn(m), dim=-1)
            cs = (pfn * mdn[:, None]).sum(-1).masked_fill(pad, 0).sum(1) / (~pad).sum(1)
            cosm.append(cs.cpu().numpy())
    kinds = [data[sc]["samples"][k]["kind"] for sc, k in items]
    return np.concatenate(L), np.concatenate(Y), kinds, np.concatenate(cosm)


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, y):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.3), np.log(5), 60)):
        p = softmax(logits / T)
        nll = -np.log(p[np.arange(len(y)), y] + 1e-9).mean()
        best = min(best, (nll, float(T)))
    return best[1]


def presence_metrics(logits, y, kinds, cosm, cos_thr):
    from sklearn.metrics import roc_auc_score

    p = softmax(logits)
    pred = p.argmax(1)
    conf = np.zeros((3, 3), int)
    for a, b in zip(y, pred):
        conf[a, b] += 1
    ece, bins = 0.0, np.linspace(0, 1, 11)
    cmax = p.max(1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = (cmax > lo) & (cmax <= hi)
        if s.any():
            ece += s.mean() * abs((pred[s] == y[s]).mean() - cmax[s].mean())
    pa = y != 2
    llr = np.log(p[:, 0] + 1e-6) - np.log(p[:, 1] + 1e-6)
    by_kind = {k: round(float((pred[np.array(kinds) == k] == y[np.array(kinds) == k]).mean()), 3)
               for k in sorted(set(kinds))}
    base_pred = np.where(cosm >= cos_thr, 0, 1)
    return {"n": int(len(y)), "accuracy": round(float((pred == y).mean()), 3),
            "recall": {c: round(float(conf[i, i] / max(conf[i].sum(), 1)), 3) for i, c in enumerate(("present", "absent", "occluded"))},
            "confusion_rows_true_cols_pred": conf.tolist(), "ece": round(float(ece), 3),
            "auroc_present_vs_absent": round(float(roc_auc_score((y[pa] == 0).astype(int), llr[pa])), 3),
            "accuracy_by_kind": by_kind,
            "cosine_baseline": {"threshold": round(float(cos_thr), 3),
                                "accuracy_present_vs_absent": round(float((base_pred[pa] == y[pa]).mean()), 3),
                                "auroc": round(float(roc_auc_score((y[pa] == 0).astype(int), cosm[pa])), 3)},
            "attention_accuracy_present_vs_absent": round(float((np.where(llr >= 0, 0, 1)[pa] == y[pa]).mean()), 3)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matcher_steps", type=int, default=6000)
    ap.add_argument("--presence_steps", type=int, default=3000)
    ap.add_argument("--eval_samples", type=int, default=400)
    ap.add_argument("--skip_presence", action="store_true", help="keep presence.pt and its results; retrain the matcher only")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    import torch

    dev = torch.device("cuda")
    C.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    data = load(TRAIN + VAL + TEST)
    log = logger.info
    report = {"split": {"train": TRAIN, "val": VAL, "test": TEST},
              "data": {s: {k: v for k, v in d["stats"].items()} for s, d in data.items()}}

    net_m, report["matcher_training"] = train_matcher(data, args.matcher_steps, dev, log)
    # threshold on the validation scene only
    best = None
    for tau in (0.1, 0.2, 0.3, 0.4, 0.5):
        r = eval_matcher(net_m, data, VAL, dev, n=200, tau=tau)[VAL[0]]["attention"]
        f = (r["added"]["f1"] + r["moved"]["f1"] + r["same_object_accuracy"]) / 3
        log(f"[matcher] val tau {tau}: {json.dumps(r)}")
        if best is None or f > best[0]:
            best = (f, tau)
    report["matcher_tau_match"] = best[1]
    report["matcher_eval"] = eval_matcher(net_m, data, VAL + TEST, dev, n=args.eval_samples, tau=best[1])
    torch.save({"state_dict": net_m.state_dict(), "tau_match": best[1], "config": {"d": 128, "heads": 4, "layers": 3}},
               C.CKPT_DIR / "matcher.pt")

    if args.skip_presence:
        old = json.loads((C.OUT_ROOT / "replica_eval.json").read_text())
        for k in ("presence_training", "presence_temperature", "presence_eval"):
            report[k] = old[k]
        (C.OUT_ROOT / "replica_eval.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=1))
        return
    net_p, report["presence_training"] = train_presence(data, args.presence_steps, dev, log)
    Lv, Yv, Kv, Cv = presence_logits(net_p, data, VAL, dev)
    T = fit_temperature(Lv, Yv)
    net_p.temperature.fill_(T)
    Ltr, Ytr, _, Ctr = presence_logits(net_p, data, TRAIN, dev)
    pa = Ytr != 2
    thr = max(np.unique(Ctr[pa]), key=lambda t: ((Ctr[pa] >= t) == (Ytr[pa] == 0)).mean()) if pa.any() else 0.5
    report["presence_temperature"] = round(T, 3)
    report["presence_eval"] = {}
    for sc in VAL + TEST:
        L, Y, K, Cs = presence_logits(net_p, data, [sc], dev)
        report["presence_eval"][sc] = presence_metrics(L, Y, K, Cs, thr)
    torch.save({"state_dict": net_p.state_dict(), "temperature": T, "config": {"feat_dim": C.FEAT_DIM, "d": 128}},
               C.CKPT_DIR / "presence.pt")
    (C.OUT_ROOT / "replica_eval.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
