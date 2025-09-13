import numpy as np
from typing import Optional


def merge_xy_conf(xy: np.ndarray, conf: Optional[np.ndarray]) -> np.ndarray:
    """
    Merge xy coordinates and confidence scores
    
    Args:
        xy: Array of shape (K, 2) containing x, y coordinates
        conf: Array of shape (K,) containing confidence scores, or None
        
    Returns:
        Array of shape (K, 3) containing (x, y, confidence)
    """
    if conf is None:
        conf = np.ones((xy.shape[0],), dtype=float)
    return np.concatenate([xy, conf[:, None]], axis=1)

def filter_keypoints_by_confidence(keypoints: np.ndarray, min_conf: float = 0.25) -> np.ndarray:
    """Filter keypoints by confidence threshold"""
    if keypoints.shape[1] < 3:
        return keypoints
    return keypoints[keypoints[:, 2] >= min_conf]

def get_visible_keypoints_count(keypoints: np.ndarray, min_conf: float = 0.25) -> int:
    """Count number of visible keypoints above confidence threshold"""
    if keypoints.shape[1] < 3:
        return len(keypoints)
    return np.sum(keypoints[:, 2] >= min_conf)