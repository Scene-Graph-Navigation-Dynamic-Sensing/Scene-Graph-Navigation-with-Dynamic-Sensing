"""Open-vocabulary naming on an equal footing (docs/evaluation_metrics.md, group A).

Acc@k ranks the node embedding against the dataset's own class list, which a
closed-set method gets for free and NOVIC never sees, so it can't show whether
open-vocabulary naming works. Here both methods name each node from the same
open vocabulary, NOVIC's target nouns (42,919), and one judge grades both names:

  NOVIC        its decoded top-1 noun for the node's embedding
  embedding    the target noun whose text embedding (same prompt) is nearest
               to the node's image embedding (nearest-noun lookup, as a
               closed-set method would name it if given NOVIC's vocabulary)
  pipeline     the node's final label, for reference only: it is mapped onto a
               245-word everyday list (configs/vocab/indoor_objects.txt) that
               contains 53 of Replica's 101 classes, so it is not open-set

The judge is WordNet. Each ground-truth class is pinned to the sense it means
(CLASS_SENSES, else WordNet's most frequent physical-object sense); the
predicted noun may be any of its physical-object senses (under artifact or
plant), and gets the best grade any of them allows:

  exact      same synset (synonyms included: "sofa" for couch)
  specific   a kind of the GT class ("armchair" for chair, "pillow" for cushion)
  generic    a GT class's ancestor ("seat", "furniture" for chair): right, but
             too general
  wrong      none of the above
  no_name    not in WordNet as a physical object, even by its head noun

"Acceptable" = exact + specific. Multi-word names not in WordNet fall back to
their head noun ("display panel" -> panel). Taking the best predicted sense
is lenient, equally so for both methods. WordNet is strict about near-misses
that people would accept ("panel light" for a ceiling lamp, "wastebasket"
vs "trash can" are siblings), so absolute numbers are a lower bound; a human
check on a sample of nodes is the way to calibrate it.
"""
from __future__ import annotations

import functools
import logging
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

GRADES = ("exact", "specific", "generic", "wrong", "no_name")
ACCEPTABLE = ("exact", "specific")
METHODS = ("novic", "embedding", "pipeline")

# The sense each Replica class means. Default: WordNet's most frequent physical-object sense
# of the class name; listed here where WordNet lacks the class or its first sense is another
# object ("plate" = baseball base, "cushion" = shock absorber). "tablet" (computer) is not in
# WordNet 3.0, so only the word itself counts.
CLASS_SENSES = {
    "bike": ["bicycle.n.01"],
    "bin": ["bin.n.01", "wastebasket.n.01"],
    "blinds": ["blind.n.03"],
    "book": ["book.n.02"],
    "cushion": ["cushion.n.03"],
    "monitor": ["computer_monitor.n.01"],
    "pipe": ["pipe.n.02"],
    "plate": ["plate.n.04"],
    "pot": ["pot.n.01", "pot.n.04"],
    "tablet": [],
    "tissue-paper": ["tissue.n.02"],
    "toilet": ["toilet.n.02"],
    "wall-plug": ["wall_socket.n.01", "socket.n.03"],
    "base-cabinet": ["cabinet.n.01"],
    "wall-cabinet": ["cabinet.n.01"],
    "cooktop": ["stove.n.01"],
    "desk-organizer": ["holder.n.01"],
    "utensil-holder": ["holder.n.01"],
    "knife-block": ["holder.n.01", "rack.n.01"],
    "exercise-ball": ["ball.n.01", "ball.n.06"],
    "indoor-plant": ["houseplant.n.01", "pot_plant.n.01"],
    "major-appliance": ["white_goods.n.01", "home_appliance.n.01"],
    "small-appliance": ["home_appliance.n.01"],
    "nightstand": ["stand.n.04", "table.n.02"],
    "plant-stand": ["stand.n.04", "rack.n.05"],
    "tv-stand": ["stand.n.04"],
    "set-of-clothing": ["clothing.n.01"],
    "table-runner": ["table_linen.n.01"],
    "logo": ["logo.n.01"],
    "tv-screen": ["television_receiver.n.01", "computer_screen.n.01", "flat_panel_display.n.01",
                  "monitor.n.04"],
}


# common names WordNet only knows under another spelling
ALIASES = {"potted plant": "pot plant", "nightstand": "night table", "night stand": "night table",
           "tv": "television receiver", "tv screen": "television receiver"}


@functools.lru_cache(maxsize=None)
def _wn():
    from nltk.corpus import wordnet as wn
    try:
        wn.synsets("chair")
    except LookupError:
        import nltk
        nltk.download("wordnet", quiet=True)
    return wn


@functools.lru_cache(maxsize=None)
def _roots():
    wn = _wn()
    return {wn.synset("artifact.n.01"), wn.synset("plant.n.02")}


@functools.lru_cache(maxsize=None)
def _ancestors(synset) -> frozenset:
    return frozenset(synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))


def _physical(synsets) -> list:
    return [s for s in synsets if (_ancestors(s) | {s}) & _roots()]


@functools.lru_cache(maxsize=None)
def noun_senses(word: str) -> tuple:
    """Physical-object noun senses of `word`, falling back to its head noun."""
    wn = _wn()
    tokens = ALIASES.get(word.lower(), word.lower()).replace("-", " ").split()
    for k in range(len(tokens)):                 # full phrase, then drop leading modifiers
        found = _physical(wn.synsets("_".join(tokens[k:]), pos="n"))
        if found:
            return tuple(found)
    return ()


@functools.lru_cache(maxsize=None)
def class_senses(gt_class: str) -> tuple:
    key = gt_class.lower().replace(" ", "-")          # reports write "indoor plant", Replica "indoor-plant"
    if key in CLASS_SENSES:
        return tuple(_wn().synset(s) for s in CLASS_SENSES[key])
    return noun_senses(gt_class)[:1]


@functools.lru_cache(maxsize=None)
def grade(pred: str, gt_class: str) -> str:
    if not pred or pred.lower() == "unknown":
        return "no_name"
    if pred.lower().replace(" ", "-") == gt_class.lower().replace(" ", "-"):
        return "exact"
    p, g = set(noun_senses(pred)), set(class_senses(gt_class))
    if not p:
        return "no_name"
    if p & g:
        return "exact"
    if any(g & _ancestors(s) for s in p):
        return "specific"
    if any(p & _ancestors(s) for s in g):
        return "generic"
    return "wrong"


def vocabulary_text_embeddings(decoder, nouns: list[str], prompt: str, cache: Path | None = None,
                               batch: int = 512) -> np.ndarray:
    """Unit text embeddings of NOVIC's target nouns, cached (they don't depend on the scene)."""
    if cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=False)
        if list(data["nouns"]) == nouns and str(data["prompt"]) == prompt:
            return data["emb"]
    emb = np.concatenate([decoder.embed_text([prompt.format(n) for n in nouns[i:i + batch]])
                          for i in range(0, len(nouns), batch)]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, nouns=np.array(nouns), prompt=np.array(prompt), emb=emb)
    return emb


def nearest_nouns(embeddings: np.ndarray, nouns: list[str], noun_emb: np.ndarray) -> list[str]:
    e = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    return [nouns[i] for i in np.argmax(e @ noun_emb.T, axis=1)]


def naming_metrics(rows: list[dict]) -> dict:
    """rows: one per scored node, with gt_class and a name per method in METHODS.
    Adds grade_<method> to each row; returns grade counts and shares per method."""
    out = {"nodes_scored": len(rows), "judge": "WordNet (exact/specific/generic/wrong)", "methods": {}}
    for m in METHODS:
        for r in rows:
            r[f"grade_{m}"] = grade(r[m], r["gt_class"])
        counts = Counter(r[f"grade_{m}"] for r in rows)
        n = max(len(rows), 1)
        out["methods"][m] = {"counts": {g: counts.get(g, 0) for g in GRADES},
                             "share": {g: counts.get(g, 0) / n for g in GRADES},
                             "acceptable": sum(counts.get(g, 0) for g in ACCEPTABLE) / n}
    return out
