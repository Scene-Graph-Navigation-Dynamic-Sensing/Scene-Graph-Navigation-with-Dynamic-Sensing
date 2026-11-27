import numpy as np

from scene_graph.builder.gravity_alignment import estimate_gravity_alignment


def _random_rotation(rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _synthetic_scene(rng):
    """Z-up room: 4x4 m floor at z=0, a 1x2 m tabletop at z=0.75, and cameras
    looking horizontally (OpenCV axes: camera y points down = world -z)."""
    floor = np.c_[rng.uniform(-2, 2, (20000, 2)), np.zeros(20000)]
    table = np.c_[rng.uniform(0, 1, 4000), rng.uniform(0, 2, 4000), np.full(4000, 0.75)]
    points = np.vstack([floor, table]) + rng.normal(0, 0.005, (24000, 3))

    cams = []
    for yaw in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        down = np.array([0.0, 0.0, -1.0])
        right = np.cross(down, forward)
        R = np.c_[right, down, forward]            # camera-to-world, columns = camera axes
        tilt = 0.15 * rng.normal(size=3)           # hand-held wobble
        cams.append(_rotation_about(tilt) @ R)
    return points, np.stack(cams)


def _rotation_about(rotvec) -> np.ndarray:
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


def test_recovers_z_up_and_floor_height_from_an_arbitrarily_rotated_world():
    rng = np.random.default_rng(0)
    points, cams = _synthetic_scene(rng)

    # Scramble the world frame the way DROID-SLAM's first-camera frame would.
    W = np.eye(4)
    W[:3, :3] = _random_rotation(rng)
    W[:3, 3] = [3.0, -1.0, 2.0]
    points_w = points @ W[:3, :3].T + W[:3, 3]
    cams_w = W[:3, :3] @ cams

    T, info = estimate_gravity_alignment(points_w, cams_w)
    aligned = points_w @ T[:3, :3].T + T[:3, 3]

    floor_z = aligned[: 20000, 2]
    table_z = aligned[20000:, 2]
    assert info["up_source"] == "plane"
    assert abs(np.median(floor_z)) < 0.03             # floor sits at z = 0
    assert abs(np.median(table_z) - 0.75) < 0.03      # table 0.75 m above it, not below
    assert np.std(floor_z) < 0.02                     # floor is level, not tilted
