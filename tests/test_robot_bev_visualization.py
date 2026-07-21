import numpy as np

from data_generation.robot_bev.schema import MAP_CLASSES
from mmdet3d.core.utils.visualize import scores_to_single_label_masks


def test_robotbev_prediction_visualization_uses_thresholded_priority():
    scores = np.zeros((len(MAP_CLASSES), 1, 4), dtype=np.float32)
    scores[MAP_CLASSES.index("floor"), 0, 0] = 0.95
    scores[MAP_CLASSES.index("door"), 0, 0] = 0.51
    scores[MAP_CLASSES.index("wall"), 0, 1] = 0.95
    scores[MAP_CLASSES.index("furniture"), 0, 1] = 0.51
    scores[MAP_CLASSES.index("clutter"), 0, 2] = 0.95
    scores[MAP_CLASSES.index("wall"), 0, 2] = 0.51
    scores[MAP_CLASSES.index("door"), 0, 3] = 0.5
    scores[MAP_CLASSES.index("floor"), 0, 3] = 0.49

    masks = scores_to_single_label_masks(scores, 0.5, classes=list(MAP_CLASSES))

    assert masks[MAP_CLASSES.index("door"), 0, 0]
    assert masks[MAP_CLASSES.index("furniture"), 0, 1]
    assert masks[MAP_CLASSES.index("wall"), 0, 2]
    assert not masks[:, 0, 3].any()
