import pickle
from types import SimpleNamespace

from scene_graph.storage.serializer import build_voxel_map_snapshot, load_scene_graph


class _FakeBounds:
    xmin, xmax, ymin, ymax, zmin, zmax = 0.0, 5.0, 0.0, 5.0, 0.0, 2.0


def test_build_voxel_map_snapshot_extracts_expected_fields():
    view_dataset = SimpleNamespace(
        resolution=0.01,
        bounds=_FakeBounds(),
        indexes_colors_mapping_dict={1: (0.1, 0.2, 0.3)},
    )
    snapshot = build_voxel_map_snapshot(view_dataset)
    assert snapshot["resolution"] == 0.01
    assert snapshot["bounds"] == (0.0, 5.0, 0.0, 5.0, 0.0, 2.0)
    assert snapshot["indexes_colors_mapping_dict"] == {1: (0.1, 0.2, 0.3)}


def test_load_scene_graph_tolerates_missing_voxel_map(tmp_path):
    # Simulates a pickle saved before navigation/ existed -- no "voxel_map" key at all.
    pkl_path = tmp_path / "old_scene.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"scene_graph": None, "instance_objects": [], "node_embeddings": {}, "node_centroids": {}}, f)

    data = load_scene_graph(pkl_path)
    assert data["voxel_map"] is None


def test_load_scene_graph_roundtrips_voxel_map(tmp_path):
    pkl_path = tmp_path / "new_scene.pkl"
    voxel_map = {"resolution": 0.05, "bounds": (0, 1, 0, 1, 0, 1), "indexes_colors_mapping_dict": {7: (1, 1, 1)}}
    with open(pkl_path, "wb") as f:
        pickle.dump({"scene_graph": None, "instance_objects": [], "node_embeddings": {},
                     "node_centroids": {}, "voxel_map": voxel_map}, f)

    data = load_scene_graph(pkl_path)
    assert data["voxel_map"] == voxel_map
