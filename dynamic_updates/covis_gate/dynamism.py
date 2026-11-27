"""Per-object move rates across visits (§3, §9): lambda_i ~ Gamma(a_i, b_i).

The scene graph pickle is left in the format every other script reads; the
rates live next to it in outputs/scene_graphs/<scene>_dynamism.json:

    {"visit_time_days": t, "visits": [...],
     "objects": {class_id: {"a", "b", "t_last", "receptacles": {parent: count}, "history": [...]}},
     "archived": {class_id: {...}}}     # removed objects, kept so their history is not lost

A memory without this file (the first visit) gets the §3.4 initialisation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import theory as T

# §3.4 option 2, "language-model warm start", as a fixed cached table: the
# guess a model would give for how often each kind of thing is moved, per day.
# kappa stays small, so a few visits override a wrong guess.
RATE_TABLE = [
    # structure and furniture: anchors
    (r"\b(wall|floor|ceiling|door|window|radiator|pillar|column|stair)", 0.001),
    (r"\b(desk|table|bed|bunk|shelf|shelves|bookcase|bookshelf|cabinet|cupboard|wardrobe|dresser|drawer|counter"
     r"|sofa|couch|bench|wardrobe|whiteboard|blackboard|board|refrigerator|fridge|oven|stove|sink|toilet|bathtub"
     r"|washing machine|closet|nightstand)", 0.005),
    (r"\b(light switch|switch|socket|outlet|curtain|blind|mirror|painting|picture|poster|clock|air condition"
     r"|fan|heater|lamp|light)", 0.01),
    (r"\b(laptop|notebook computer)", 0.3),  # before "computer"
    (r"\b(monitor|screen|television|tv|printer|router|speaker|computer|pc)\b", 0.02),
    (r"\b(chair|stool|trash|bin|basket|box|plant|pot)\b", 0.3),
    # small things people carry
    (r"\b(pen|pencil|eraser|marker|paper|book|notebook|charger|cable|cord|earphone|headphone|mouse|keyboard)", 0.5),
    (r"\b(cup|mug|glass|bottle|thermos|can|plate|bowl|knife|fork|spoon|food|fruit|bag|cap|hat|ball|toy|towel"
     r"|clothes|shirt|shoe|remote|iron)", 1.0),
    (r"\b(key|phone|wallet|smartphone|cellphone)", 1.5),
]


def table_rate(class_name: str, default: float) -> float:
    name = class_name.lower()  # first matching row wins
    for pattern, rate in RATE_TABLE:
        if re.search(pattern, name):
            return rate
    return default


class Dynamism:
    def __init__(self, state: dict):
        self.state = state

    # ── creation / storage ──

    @classmethod
    def load_or_init(cls, path: Path, objects: list, memory_time_days: float, dcfg: dict,
                     parents: dict) -> tuple["Dynamism", bool]:
        """(store, True) when path existed; otherwise the §3.4 initialisation (False).
        parents: class_id -> its receptacle in the memory graph, the first entry of its history."""
        if path.exists():
            d = cls(json.loads(path.read_text()))
            for o in objects:  # objects added to the graph by another method: initialise them
                if o["class_id"] not in d.state["objects"]:
                    d.add(o, d.state["visit_time_days"], dcfg, parent=parents.get(o["class_id"]))
            return d, True
        d = cls({"visit_time_days": float(memory_time_days), "visits": [], "objects": {}, "archived": {}})
        for o in objects:
            d.add(o, memory_time_days, dcfg, parent=parents.get(o["class_id"]))
        return d, False

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.state, indent=2))

    def add(self, obj: dict, t: float, dcfg: dict, parent: str | None) -> None:
        guess = (table_rate(obj["class_name"], dcfg["rate_default"]) if dcfg["rate_prior"] == "table"
                 else dcfg["rate_default"])
        a, b = T.init_rate(guess, dcfg["kappa"])
        self.state["objects"][obj["class_id"]] = {
            "class_name": obj["class_name"], "a": a, "b": b, "t_last": float(t), "rate_guess": guess,
            "receptacles": {parent: 1} if parent else {}, "history": []}

    # ── queries ──

    def rec(self, cid: str) -> dict:
        return self.state["objects"][cid]

    def mean_rate(self, cid: str) -> float:
        r = self.rec(cid)
        return r["a"] / r["b"]

    def survival(self, cid: str, t_now: float) -> float:
        r = self.rec(cid)
        return T.survival(r["a"], r["b"], max(t_now - r["t_last"], 0.0))

    def anchors(self, cids: list, rate_threshold: float) -> list:
        """eq. 8: A = {i : a_i / b_i < lambda_A}."""
        return [c for c in cids if self.mean_rate(c) < rate_threshold]

    # ── §9 learning ──

    def learn(self, cid: str, outcome: str, t_now: float, dcfg: dict, parent: str | None = None) -> dict:
        """Updates (a, b) from this visit's outcome; returns the before/after record."""
        r = self.rec(cid)
        dt = max(t_now - r["t_last"], 0.0)
        before = (r["a"], r["b"])
        a, b = T.forget(r["a"], r["b"], dcfg["forgetting"])
        if outcome == "present":
            a, b = T.update_stayed(a, b, dt)
        elif outcome in ("moved", "removed"):
            if dt <= 0:  # no elapsed time: the moved likelihood 1 - e^0 is 0, nothing to learn from
                pass
            elif dcfg["rate_update"] == "naive":
                a, b = T.update_moved_naive(a, b, dt)
            else:
                a, b = T.update_moved(a, b, dt)
        else:  # deferred: not observed, no update (§9.1)
            a, b = before
        r["a"], r["b"] = float(a), float(b)
        if outcome in ("present", "moved"):
            r["t_last"] = float(t_now)
            if parent:
                r["receptacles"][parent] = r["receptacles"].get(parent, 0) + 1
        entry = {"t": float(t_now), "dt_days": round(dt, 4), "outcome": outcome,
                 "rate_before": round(before[0] / before[1], 5), "rate_after": round(r["a"] / r["b"], 5),
                 "a_b_before": [round(x, 4) for x in before], "a_b_after": [round(r["a"], 4), round(r["b"], 4)]}
        r["history"].append(entry)
        return entry

    def archive(self, cid: str) -> None:
        self.state["archived"][cid] = self.state["objects"].pop(cid)

    def rename(self, old: str, new: str) -> None:
        if old != new:
            self.state["objects"][new] = self.state["objects"].pop(old)
