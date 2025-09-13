import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from .base import BasePoseEstimator
from utils.pose_helper import merge_xy_conf

class CropBasedYOLOPoseEstimator(BasePoseEstimator):
    """Crop-based YOLO pose estimator with padding"""
    
    def __init__(self, weights: str, device: str = "cpu", crop_size: int = 320):
        super().__init__(weights, device)
        self.crop_size = crop_size
        self.load_model()
        
    def load_model(self):
        """Load YOLO pose model"""
        self.model = YOLO(self.weights)
        
    def infer(self, image_tensor: torch.Tensor, resized_rgb: np.ndarray, person_dets: List[Dict]) -> List[Dict]:
        """Run pose estimation on padded person crops"""
        poses = []
        if not person_dets:
            return poses
            
        for person_det in person_dets:
            pose_result = self._process_padded_crop(resized_rgb, person_det)
            if pose_result:
                poses.append(pose_result)
                
        return poses
        
    def _process_padded_crop(self, resized_rgb: np.ndarray, person_det: Dict) -> Dict:
        """Process person crop with padding"""
        x1, y1, x2, y2 = map(int, person_det["bbox"])
        
        # Add padding around person box
        pad = 20
        h, w = resized_rgb.shape[:2]
        x1_pad = max(0, x1 - pad)
        y1_pad = max(0, y1 - pad)
        x2_pad = min(w, x2 + pad)
        y2_pad = min(h, y2 + pad)
        
        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            return None
            
        # Extract and process crop
        crop = resized_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
        crop_pil = Image.fromarray(crop)
        
        # Prepare crop tensor
        transform = transforms.Compose([
            transforms.Resize((self.crop_size, self.crop_size)),
            transforms.ToTensor()
        ])
        crop_tensor = transform(crop_pil).unsqueeze(0).to(self.device)
        
        # Pose inference on crop
        with torch.no_grad():
            results = self.model(crop_tensor)
        
        if not results or not hasattr(results[0], 'keypoints') or results[0].keypoints is None:
            return None
            
        # Process keypoints
        kpts_xy = results[0].keypoints.xy.cpu().numpy()
        kpts_conf = None
        if hasattr(results[0].keypoints, 'conf') and results[0].keypoints.conf is not None:
            kpts_conf = results[0].keypoints.conf.cpu().numpy()
        
        # Take best pose detection from crop
        if len(kpts_xy) > 0:
            # Transform keypoints back to original image coordinates
            crop_h, crop_w = crop.shape[:2]
            kpts_orig = kpts_xy[0].copy()  # Take first/best detection
            
            # Scale from crop_size back to original coordinates
            kpts_orig[:, 0] = (kpts_orig[:, 0] / self.crop_size) * crop_w + x1_pad
            kpts_orig[:, 1] = (kpts_orig[:, 1] / self.crop_size) * crop_h + y1_pad
            
            keypoints = merge_xy_conf(kpts_orig, kpts_conf[0] if kpts_conf is not None else None)
            
            # Validate keypoints
            if self.validate_keypoints(keypoints):
                return {
                    "bbox": person_det["bbox"],
                    "cls": person_det["cls"],
                    "kpts": keypoints,
                    "det_score": person_det["score"]
                }
                
        return None