from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch
import numpy as np

class BasePoseEstimator(ABC):
    """Abstract base class for pose estimators"""
    
    def __init__(self, weights: str, device: str = "cpu"):
        self.weights = weights
        self.device = device
        self.model = None
        self.load_model()
        
    @abstractmethod
    def load_model(self):
        """Load the pose estimation model"""
        pass
        
    @abstractmethod
    def infer(self, image_tensor: torch.Tensor, resized_rgb: np.ndarray, person_dets: List[Dict]) -> List[Dict]:
        """
        Run pose estimation
        
        Args:
            image_tensor: Input image tensor
            resized_rgb: Resized RGB image array
            person_dets: List of person detection dictionaries
            
        Returns:
            List of pose result dictionaries with keys:
            - bbox: [x1, y1, x2, y2] (from original detection)
            - cls: class id (from original detection)
            - kpts: numpy array of shape (K, 3) containing (x, y, confidence)
            - det_score: detection confidence score
        """
        pass
        
    def validate_keypoints(self, keypoints: np.ndarray, min_confidence: float = 0.3) -> bool:
        """Validate if keypoints are reasonable"""
        if keypoints is None or len(keypoints) == 0:
            return False
            
        # Check if at least some keypoints have good confidence
        valid_kpts = keypoints[:, 2] > min_confidence
        return np.sum(valid_kpts) > len(keypoints) * 0.3  # At least 30% visible