"""Objects are named from their running-average embedding when a track ends,
and re-decoded at naming time only if they changed after that decode."""
from pathlib import Path

import numpy as np
import yaml

from object_fusion.instance_fusion import build_instance_fusion_class
from object_fusion.instance_tagger import current_track_name, tag_instances, to_everyday_names
from semantic_decoding.novic.novic_decoder import TagResult

REPO_ROOT = Path(__file__).resolve().parents[2]
SGG_ROOT = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text())["external"][
    "scene_graph_generation_root"]


class _FakeDecoder:
    """Text embeddings are one-hot per synonym group, so only listed synonyms pool."""

    SYNONYMS = {"bureau": "desk", "escritoire": "desk"}

    def __init__(self):
        self.decoded_rows = 0
        self._groups: dict[str, int] = {}

    def embed_text(self, texts):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for row, text in enumerate(texts):
            group = self.SYNONYMS.get(text, text)
            out[row, self._groups.setdefault(group, len(self._groups))] = 1.0
        return out

    def decode_embeddings(self, embeddings):
        self.decoded_rows += len(embeddings)
        return [TagResult(label=f"thing{int(np.argmax(e))}", score=0.5, embedding=e, top_k=[])
                for e in embeddings]


def _det(track_id):
    return {"track_ids": [track_id]}


def _unit(i, dim=4):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


def test_last_seen_frames_marks_each_tracks_final_frame():
    InstanceFusion = build_instance_fusion_class(SGG_ROOT)
    frames = [[_det(1), _det(2)], [_det(1), _det(-1)], [], [_det(2)]]
    assert InstanceFusion.last_seen_frames(frames) == {1: 1, 2: 3}


def test_ended_tracks_decode_their_objects_average_once_per_object():
    fusion = build_instance_fusion_class(SGG_ROOT)()
    objects = [
        {"clip_ft": _unit(0), "num_detections": 7, "track_names": []},
        {"clip_ft": _unit(2), "num_detections": 3, "track_names": []},
    ]
    fusion._track_to_object_index = {5: 0, 6: 0, 9: 1}
    decoder = _FakeDecoder()

    fusion.name_ended_tracks(objects, [5, 6, 9], frame_idx=12, decode_fn=decoder.decode_embeddings)

    assert decoder.decoded_rows == 2                      # two objects, not three tracks
    assert [r["track_id"] for r in objects[0]["track_names"]] == [5, 6]
    assert {r["label"] for r in objects[0]["track_names"]} == {"thing0"}
    assert objects[0]["track_names"][0]["num_detections"] == 7
    assert objects[1]["track_names"][0]["label"] == "thing2"
    assert len(fusion.track_name_log) == 3


def test_record_goes_stale_once_the_object_gains_detections():
    obj = {"num_detections": 4, "track_names": [
        {"label": "cup", "score": 0.9, "top_k": [], "num_detections": 2},
        {"label": "mug", "score": 0.8, "top_k": [], "num_detections": 4},
    ]}
    assert current_track_name(obj)["label"] == "mug"
    obj["num_detections"] = 6
    assert current_track_name(obj) is None


def test_tagging_reuses_valid_track_names_and_decodes_the_rest():
    objects = [
        {"clip_ft": _unit(0), "num_detections": 5,
         "track_names": [{"label": "chair", "score": 0.8, "top_k": [], "num_detections": 5}]},
        {"clip_ft": _unit(1), "num_detections": 9,     # merged after it was named
         "track_names": [{"label": "table", "score": 0.7, "top_k": [], "num_detections": 4}]},
        {"clip_ft": _unit(3), "num_detections": 3, "track_names": []},   # untracked only
    ]
    decoder = _FakeDecoder()

    objects = tag_instances(objects, decoder, min_name_score=0.15)

    assert decoder.decoded_rows == 2
    assert [o["named_by"] for o in objects] == ["track_end", "final_decode", "final_decode"]
    assert [o["class_name"] for o in objects] == ["chair", "thing1", "thing3"]
    assert np.allclose(objects[1]["semantic_embedding"], _unit(1))


def _named_obj(label, score, axis):
    return {"clip_ft": _unit(axis), "num_detections": 2,
            "track_names": [{"label": label, "score": score,
                             "top_k": [(label, score), ("other", 0.01)], "num_detections": 2}]}


def test_low_confidence_names_become_unknown_and_keep_their_guesses():
    objects = [_named_obj("laptop", 0.6, 0), _named_obj("eyebrow", 0.02, 1), _named_obj("hashish", 0.09, 2)]

    objects = tag_instances(objects, _FakeDecoder(), min_name_score=0.15, low_score_action="unknown")

    assert [o["class_id"] for o in objects] == ["laptop_0", "unknown_0", "unknown_1"]
    assert objects[1]["novic_guess"] == "eyebrow"
    assert objects[1]["novic_top_k"][0] == ("eyebrow", 0.02)
    assert "novic_guess" not in objects[0]


def test_drop_removes_low_confidence_instances():
    objects = [_named_obj("laptop", 0.6, 0), _named_obj("eyebrow", 0.02, 1)]

    objects = tag_instances(objects, _FakeDecoder(), min_name_score=0.15, low_score_action="drop")

    assert [o["class_id"] for o in objects] == ["laptop_0"]


def test_support_surface_takes_the_surface_noun_from_its_beam_and_is_never_dropped():
    desk = _named_obj("laptop", 0.17, 0)
    desk["track_names"][0]["top_k"] = [("laptop", 0.17), ("workstation", 0.11), ("desk", 0.11)]
    faint_table = _named_obj("hashish", 0.05, 1)                    # no surface noun in its beam
    objects = [desk, faint_table, _named_obj("backpack", 0.56, 2)]

    objects = tag_instances(objects, _FakeDecoder(), min_name_score=0.1, support_surfaces={0, 1})

    assert [o["class_id"] for o in objects] == ["desk_0", "table_0", "backpack_0"]
    assert objects[0]["named_by"] == "support_surface"
    assert objects[0]["novic_score"] == 0.11


def test_synonym_scores_pool_into_the_name_confidence():
    desk = _named_obj("desk", 0.10, 0)
    desk["track_names"][0]["top_k"] = [("desk", 0.10), ("bureau", 0.08), ("escritoire", 0.07), ("lectern", 0.02)]
    guess = _named_obj("hashish", 0.106, 1)
    guess["track_names"][0]["top_k"] = [("hashish", 0.106), ("flatiron", 0.06), ("inkpad", 0.04)]

    objects = tag_instances([desk, guess], _FakeDecoder(), min_name_score=0.15)

    assert [o["class_id"] for o in objects] == ["desk_0"]
    assert abs(objects[0]["name_confidence"] - 0.25) < 1e-9
    assert objects[0]["novic_score"] == 0.10


def test_uncommon_labels_become_the_everyday_name_that_fits():
    axes = {"chair": 0, "wallet": 1, "earphone": 2, "charger": 3}
    words = {**{w: np.eye(4)[i] for w, i in axes.items()},
             "billfold": np.eye(4)[1], "earplug": np.eye(4)[2], "office furniture": np.array([0.6, 0, 0, 0.8])}

    def embed_text(texts):
        return np.stack([words[t.removeprefix("a photo of a ")] for t in texts])

    images = [np.eye(4)[0],                        # an office chair
              np.array([0, 0.9, 0, 0.1]),           # a wallet
              np.array([0, 0, 0.62, 0.78]),         # earbuds: the image alone leans to "charger"
              np.eye(4)[0]]                         # already an everyday name
    labels = ["office furniture", "billfold", "earplug", "chair"]

    assert to_everyday_names(labels, images, list(axes), embed_text, 0.2) == ["chair", "wallet", "earphone", "chair"]
    assert to_everyday_names(labels, images, list(axes), embed_text, 0.0)[2] == "charger"


def test_deferred_track_end_names_use_the_embedding_at_track_end():
    """Streaming defers the NOVIC decode (name_ended_tracks(defer=True)) and
    decodes in one batch later: the name must still come from the object's
    embedding as it was when the track ended, not as it is at the flush."""
    fusion = build_instance_fusion_class(SGG_ROOT)()
    decoder = _FakeDecoder()
    fusion._decode_fn = decoder.decode_embeddings
    obj = {"clip_ft": _unit(1), "num_detections": 4, "track_names": []}
    fusion._track_to_object_index = {7: 0, 8: 0}
    event = {}
    fusion.name_ended_tracks([obj], [7], 10, decoder.decode_embeddings, defer=True, events=[event])
    assert obj["track_names"][0]["label"] is None and decoder.decoded_rows == 0

    obj["clip_ft"] = _unit(3)  # the object changes before the batch decode
    fusion.name_ended_tracks([obj], [8], 12, decoder.decode_embeddings, defer=True)
    fusion.flush_track_names()

    assert [r["label"] for r in obj["track_names"]] == ["thing1", "thing3"]
    assert event["name"] == "thing1"
    assert [r["label"] for r in fusion.track_name_log] == ["thing1", "thing3"]
    assert decoder.decoded_rows == 2
    fusion.flush_track_names()  # nothing left: no second decode
    assert decoder.decoded_rows == 2
