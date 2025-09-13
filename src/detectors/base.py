from abc import ABC, abstractmethod
from typing import List, Dict
import torch

class BaseDetector(ABC):
    """Abstract base class for object detectors"""
    
    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.5):
        self.weights = weights
        self.device = device
        self.conf = conf
        self.model = None
        
    @abstractmethod
    def load_model(self):
        """Load the detection model"""
        pass
        
    @abstractmethod
    def infer(self, image_tensor: torch.Tensor) -> List[Dict]:
        """
        Run inference on image tensor
        
        Args:
            image_tensor: Input image tensor
            
        Returns:
            List of detection dictionaries with keys:
            - bbox: [x1, y1, x2, y2]
            - score: confidence score
            - cls: class id
        """
        pass
        
    def filter_by_confidence(self, detections: List[Dict]) -> List[Dict]:
        """Filter detections by confidence threshold"""
        return [det for det in detections if det["score"] >= self.conf]
        
    def filter_by_class(self, detections: List[Dict], class_names: List[str], target_class: str) -> List[Dict]:
        """Filter detections by specific class name"""
        filtered = []
        for det in detections:
            cls_id = det["cls"]
            if 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id].lower()
                if class_name == target_class.lower():
                    filtered.append(det)
        return filtered