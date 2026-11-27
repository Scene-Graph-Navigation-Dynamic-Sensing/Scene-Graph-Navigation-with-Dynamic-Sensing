import numpy as np

from scripts.prepare_synthetic_dataset import BODY_TO_OPTICAL_T, odom_pose_to_camera_to_world, quat_to_matrix


def test_identity_quaternion_gives_body_to_optical_rotation():
    T = odom_pose_to_camera_to_world(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert np.allclose(T[:3, :3], BODY_TO_OPTICAL_T)
    assert np.allclose(T[:3, 3], [0.0, 0.0, 0.0])
    assert np.allclose(T[3], [0.0, 0.0, 0.0, 1.0])


def test_translation_passes_through_unchanged():
    T = odom_pose_to_camera_to_world(1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0)
    assert np.allclose(T[:3, 3], [1.0, 2.0, 3.0])


def test_result_is_a_valid_rigid_transform_for_arbitrary_quaternions():
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        qx, qy, qz, qw = q
        R = quat_to_matrix(qx, qy, qz, qw)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-8)

        T = odom_pose_to_camera_to_world(0.0, 0.0, 0.0, qx, qy, qz, qw)
        R_wc = T[:3, :3]
        assert np.allclose(R_wc @ R_wc.T, np.eye(3), atol=1e-8)
        assert np.isclose(np.linalg.det(R_wc), 1.0, atol=1e-8)
