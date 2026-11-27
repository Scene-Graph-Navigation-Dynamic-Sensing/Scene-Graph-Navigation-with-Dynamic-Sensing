import tempfile
from pathlib import Path

import numpy as np

from object_fusion.track_label_voter import TrackLabelVoter


def test_most_frequent_noun_wins_not_the_latest():
    v = TrackLabelVoter()
    for _ in range(4):
        v.register(1, "mug", 0.8)
    v.register(1, "bowl", 0.99)          # one confident but rare disagreement
    label, stats = v.label_for_tracks([1])
    assert label == "mug"
    assert stats["vote_count"] == 4 and stats["total_votes"] == 5
    assert abs(stats["vote_fraction"] - 0.8) < 1e-9


def test_votes_pool_across_every_track_of_one_instance():
    # the tracker drops an id through an occlusion and restarts as a new one;
    # both feed the same fused 3D instance
    v = TrackLabelVoter()
    for _ in range(2):
        v.register(1, "chair", 0.7)
    for _ in range(3):
        v.register(2, "chair", 0.6)
    v.register(2, "stool", 0.9)
    label, stats = v.label_for_tracks([1, 2])
    assert label == "chair"
    assert stats["vote_count"] == 5
    assert stats["track_ids"] == [1, 2]


def test_no_votes_for_unknown_tracks():
    assert TrackLabelVoter().label_for_tracks([7]) == (None, {})


def test_decoding_is_throttled_once_a_track_has_enough_votes():
    v = TrackLabelVoter(min_votes=2, revote_interval=10)
    assert v.needs_vote(5, frame_idx=0)          # nothing yet
    v.register(5, "lamp", 0.5)
    assert v.needs_vote(5, frame_idx=1)          # still under min_votes
    v.register(5, "lamp", 0.5)
    assert not v.needs_vote(5, frame_idx=1)      # satisfied, and not a revote frame
    assert v.needs_vote(5, frame_idx=5)          # (5 + 5) % 10 == 0 -> revote


def test_untracked_detections_always_decode():
    v = TrackLabelVoter(min_votes=1)
    for frame in range(3):
        assert v.needs_vote(-1, frame_idx=frame)


def test_best_crop_per_noun_is_saved_with_dtd_naming():
    v = TrackLabelVoter()
    low = np.zeros((8, 8, 3), dtype=np.uint8)
    high = np.full((8, 8, 3), 255, dtype=np.uint8)
    v.register(3, "kettle", 0.40, low)
    v.register(3, "kettle", 0.90, high)          # better score: replaces the crop
    with tempfile.TemporaryDirectory() as d:
        assert v.save_crops(Path(d)) == 1
        name = next(Path(d).iterdir()).name
    assert name == "track_3_kettle_freq_2_avgconf_0.650.jpg"
