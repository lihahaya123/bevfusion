import numpy as np

from data_generation.robot_bev.projection import (
    depth_to_base_points,
    dilate_binary,
    rasterize_semantics,
    semantic_boundary_mask,
)
from data_generation.robot_bev.schema import MAP_CLASSES


def test_opencv_depth_projects_into_canonical_base_axes():
    depth = np.array([[2.0]], dtype=np.float32)
    intrinsic = np.eye(3, dtype=np.float32)
    camera2base = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    points, semantic_ids = depth_to_base_points(
        depth, intrinsic, camera2base, semantic=np.array([[4]])
    )

    np.testing.assert_allclose(points[0], [2.0, 0.0, 0.7, 0.0, 0.0])
    np.testing.assert_array_equal(semantic_ids, [4])


def test_semantic_boundary_marks_both_sides_and_dilates():
    labels = np.array(
        [
            [4, 4, 15, 15],
            [4, 4, 15, 15],
        ],
        dtype=np.uint8,
    )

    boundary = semantic_boundary_mask(labels, radius=0)

    np.testing.assert_array_equal(
        boundary,
        np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        dilate_binary(np.array([[1, 0], [0, 0]], dtype=np.uint8), 1),
        np.ones((2, 2), dtype=np.uint8),
    )


def test_semantic_rasterization_remains_multihot():
    points = np.array(
        [
            [0.25, 0.0, 0.0, 0.0, 0.0],
            [0.25, 0.0, 0.5, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = rasterize_semantics(
        points,
        np.array([4, 1], dtype=np.int64),
        {4: "floor", 1: "furniture"},
        (0.0, 1.0, 0.5),
        (-0.5, 0.5, 0.5),
        (-0.5, 2.0),
    )

    assert labels[MAP_CLASSES.index("floor"), 0, 1] == 1
    assert labels[MAP_CLASSES.index("furniture"), 0, 1] == 1

