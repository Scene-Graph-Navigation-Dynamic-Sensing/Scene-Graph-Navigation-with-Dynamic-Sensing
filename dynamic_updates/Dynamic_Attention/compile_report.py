"""Collect replica_eval.json and the live runs into outputs/dynamic_attention/REPORT.md.

  python dynamic_updates/Dynamic_Attention/compile_report.py --runs NAME=attention_scene:rule_scene ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import common as C


def live(path: Path) -> dict | None:
    p = path / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def presence_stats(path: Path, names_absent: set) -> dict:
    p = path / "presence.json"
    if not p.exists():
        return {}
    rows = json.loads(p.read_text())
    out = {}
    for r in rows:
        for obj, (pp, pa, po) in r["objects"].items():
            o = out.setdefault(obj, {"frames": 0, "said_absent": 0, "said_present": 0, "said_occluded": 0, "p_absent": []})
            o["frames"] += 1
            k = int(np.argmax([pp, pa, po]))
            o[("said_present", "said_absent", "said_occluded")[k]] += 1
            o["p_absent"].append(pa)
    for o in out.values():
        o["median_p_absent"] = round(float(np.median(o.pop("p_absent"))), 3)
    return out


def fmt_prf(d):
    return f"{d['precision']:.2f} / {d['recall']:.2f} / **{d['f1']:.2f}** ({d['tp']} TP, {d['fp']} FP, {d['fn']} FN)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="label=attention_scene:rule_scene (rule under outputs/covis_live)")
    ap.add_argument("--ablations", nargs="*", default=[], help="label=attention_scene")
    args = ap.parse_args()
    rep = json.loads((C.OUT_ROOT / "replica_eval.json").read_text())
    L = ["# Dynamic_Attention: test report", ""]
    L += ["## 1. Replica (held-out scenes, simulated changes)", "",
          f"Train {', '.join(rep['split']['train'])}; validation {', '.join(rep['split']['val'])} (thresholds and temperature only); "
          f"test {', '.join(rep['split']['test'])} (never seen). Matcher: {rep['matcher_training']['steps']} steps in "
          f"{rep['matcher_training']['seconds'] / 60:.0f} min, final loss {rep['matcher_training']['final_loss']}; match threshold "
          f"tau = {rep['matcher_tau_match']} (chosen on validation). Presence head: {rep['presence_training']['steps']} steps in "
          f"{rep['presence_training']['seconds'] / 60:.0f} min on {rep['presence_training']['train_items']} samples; "
          f"temperature {rep['presence_temperature']}.", ""]
    L += ["### Association: attention matcher vs the rule (decide.associate)", "",
          "Same windows for both, same oracle absent set, same novelty veto. Precision / recall / F1.", "",
          "| scene | tracks (added / moved) | method | same object kept | added P / R / F1 | moved P / R / F1 | added taken as an existing object | removed paired as moved | ms per window |",
          "|---|---|---|---:|---|---|---:|---:|---:|"]
    for sc, r in rep["matcher_eval"].items():
        tag = "test" if sc in rep["split"]["test"] else "val"
        for meth in ("rule", "attention"):
            m = r[meth]
            L.append(f"| {sc} ({tag}) | {r['tracks']} ({r['true_added']} / {r['true_moved']}) | {meth} | {m['same_object_accuracy']:.3f} | "
                     f"{fmt_prf(m['added'])} | {fmt_prf(m['moved'])} | {m['added_taken_as_existing']} | {m['removed_paired_as_moved']} | "
                     f"{r['attention_ms_per_window'] if meth == 'attention' else ''} |")
    L += ["", "### Presence head", "",
          "| scene | samples | accuracy (3 classes) | recall present / absent / occluded | ECE | present-vs-absent accuracy: head / cosine baseline | AUROC: head / cosine |",
          "|---|---:|---:|---|---:|---|---|"]
    for sc, r in rep["presence_eval"].items():
        tag = "test" if sc in rep["split"]["test"] else "val"
        rc = r["recall"]
        L.append(f"| {sc} ({tag}) | {r['n']} | {r['accuracy']:.3f} | {rc['present']:.2f} / {rc['absent']:.2f} / {rc['occluded']:.2f} | "
                 f"{r['ece']:.3f} | {r['attention_accuracy_present_vs_absent']:.3f} / {r['cosine_baseline']['accuracy_present_vs_absent']:.3f} | "
                 f"{r['auroc_present_vs_absent']:.3f} / {r['cosine_baseline']['auroc']:.3f} |")
    L += ["", "Accuracy by sample kind: " + "; ".join(f"{sc}: " + ", ".join(f"{k} {v:.2f}" for k, v in r["accuracy_by_kind"].items())
                                                    for sc, r in rep["presence_eval"].items()), ""]

    L += ["## 2. Live replays (own videos, never trained on)", ""]
    for spec in args.runs:
        label, pair = spec.split("=", 1)
        att_name, rule_name = pair.split(":")
        a = live(C.OUT_ROOT / "live" / att_name)
        r = live(C.REPO_ROOT / "outputs" / "covis_live" / rule_name)
        L += [f"### {label}", "", "| | rule (CoViS live) | Dynamic_Attention |", "|---|---|---|"]
        rows = [("removed", lambda s: ", ".join(s["final"]["removed"]) or "none"),
                ("added", lambda s: ", ".join(s["final"]["added"]) or "none"),
                ("moved", lambda s: ", ".join(s["final"]["moved"]) or "none"),
                ("graph versions (time since start: change)", lambda s: "; ".join(
                    f"{g['t_since_start']} s: -{len(g['removed'])} +{len(g['added'])}" for g in s["update"].get("graph_versions", [])) or "none"),
                ("startup", lambda s: f"{s['startup_s']} s"),
                ("stream / video", lambda s: f"{s['stream_wall_s']} s / {s['video_s']} s"),
                ("per keyframe median / p95 / max (ms)", lambda s: f"{s['per_keyframe_ms']['median']} / {s['per_keyframe_ms']['p95']} / {s['per_keyframe_ms']['max']}"),
                ("keyframe lag mean / max (ms)", lambda s: f"{s['lag_ms']['mean']} / {s['lag_ms']['max']}"),
                ("after last keyframe", lambda s: f"{s['update']['after_last_keyframe_s']} s"),
                ("detector jobs (s each)", lambda s: ", ".join(str(o["s"]) for o in (s["observation"] or []) if o.get("event") == "done") or "none")]
        for name, f in rows:
            L.append(f"| {name} | {f(r) if r else 'n/a'} | {f(a) if a else 'n/a'} |")
        if a:
            at = a["attention"]
            L += ["", f"Presence head per evidence frame: median {at['presence_ms']['median']} ms, p95 {at['presence_ms']['p95']} ms "
                      f"over {at['presence_ms']['frames']} frames. Evidence-frame split (median ms): {json.dumps(at['evidence_split_ms'])}. "
                      f"Matcher, last association: {at['matcher_ms_last']} ms.", ""]
            L += ["| memory object | final (rule) | final (attention) | P end (rule) | P end (attention) | presence LLR sum | frames seen by head | head said absent / present / occluded | median P(absent) |",
                  "|---|---|---|---:|---:|---:|---:|---|---:|"]
            ps = presence_stats(C.OUT_ROOT / "live" / att_name, set())
            for c, o in a["objects"].items():
                q = ps.get(c, {})
                L.append(f"| {c} | {r['final']['labels'].get(c) if r else ''} | {a['final']['labels'][c]} | "
                         f"{r['objects'][c]['P'] if r else ''} | {o['P']} | {at['presence_llr_sum'][c]} | {at['presence_frames'][c]} | "
                         f"{q.get('said_absent', 0)} / {q.get('said_present', 0)} / {q.get('said_occluded', 0)} | {q.get('median_p_absent', '')} |")
            if a.get("association"):
                L += ["", "Matcher rows (each observed object):", "", "| observed | decision | best memory | P(best) | P(new) | novelty | detections |", "|---|---|---|---:|---:|---:|---:|"]
                for row in a["association"]["rows"]:
                    L.append(f"| {row['object']} | {row['decision']} | {row.get('best_memory')} | {row.get('p_best')} | {row.get('p_new')} | {row['novelty']} | {row['detections']} |")
            if r and r.get("association"):
                L += ["", "Rule rows (same video):", "", "| observed | decision | best memory | score | novelty | detections |", "|---|---|---|---:|---:|---:|"]
                for row in r["association"]["rows"]:
                    L.append(f"| {row['object']} | {row['decision']} | {row.get('best_memory')} | {row.get('score')} | {row['novelty']} | {row['detections']} |")
        L.append("")
    if args.ablations:
        L += ["### Ablations (lap_in_1 → lap_remove_2)", "", "| run | removed | added | moved | per keyframe median / p95 (ms) |", "|---|---|---|---|---|"]
        for spec in args.ablations:
            label, name = spec.split("=", 1)
            a = live(C.OUT_ROOT / "live" / name)
            if a:
                L.append(f"| {label} | {', '.join(a['final']['removed']) or 'none'} | {', '.join(a['final']['added']) or 'none'} | "
                         f"{', '.join(a['final']['moved']) or 'none'} | {a['per_keyframe_ms']['median']} / {a['per_keyframe_ms']['p95']} |")
        L.append("")
    (C.OUT_ROOT / "REPORT.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
