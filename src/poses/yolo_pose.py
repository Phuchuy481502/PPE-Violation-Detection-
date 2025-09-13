import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO
from poses.base import BasePoseEstimator
from utils.pose_helper import merge_xy_conf

class YOLOPoseEstimator(BasePoseEstimator):
    """YOLO-based pose estimator using person crop approach"""
    
    def __init__(self, weights: str, device: str = "cpu"):
        super().__init__(weights, device)
        self.load_model()
        
    def load_model(self):
        """Load YOLO pose model"""
        self.model = YOLO(self.weights)
        
    def infer(self, image_tensor: torch.Tensor, resized_rgb: np.ndarray, person_dets: List[Dict]) -> List[Dict]:
        """Run pose estimation on person crops"""
        poses = []
        if not person_dets:
            return poses
            
        # Get original image dimensions
        _, h, w = image_tensor.shape[1:]
        
        for person_det in person_dets:
            x1, y1, x2, y2 = person_det["bbox"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Ensure bounds are within image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Crop person region
            person_crop = resized_rgb[y1:y2, x1:x2]
            
            if person_crop.size == 0:
                continue
                
            # Process crop
            pose_result = self._process_person_crop(person_crop, person_det, x1, y1)
            if pose_result:
                poses.append(pose_result)
                
        return poses
        
    def _process_person_crop(self, person_crop: np.ndarray, person_det: Dict, offset_x: int, offset_y: int) -> Dict:
        """Process individual person crop for pose estimation"""
        # Convert crop to tensor
        pil_crop = Image.fromarray(person_crop)
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor()
        ])
        crop_tensor = transform(pil_crop).unsqueeze(0).to(self.device)
        
        # Run pose estimation on crop
        with torch.no_grad():
            crop_results = self.model(crop_tensor)
            
        if len(crop_results) == 0:
            return None
            
        result = crop_results[0]
        if not hasattr(result, "keypoints") or result.keypoints is None:
            return None
            
        # Extract keypoints
        kpts_xy = result.keypoints.xy.cpu().numpy()      # (N,K,2)
        kpts_conf = None
        if hasattr(result.keypoints, "conf") and result.keypoints.conf is not None:
            kpts_conf = result.keypoints.conf.cpu().numpy()  # (N,K)
        elif hasattr(result.keypoints, "data"):
            data = result.keypoints.data
            if data.shape[-1] == 3:
                kpts_conf = data[..., 2].cpu().numpy()
        
        # Process first detection (best one)
        if len(kpts_xy) > 0:
            crop_h, crop_w = person_crop.shape[:2]
            kpts_scaled = kpts_xy[0].copy()
            
            # Scale from 640x640 back to crop size, then to original coordinates
            kpts_scaled[:, 0] = (kpts_scaled[:, 0] / 640.0) * crop_w + offset_x
            kpts_scaled[:, 1] = (kpts_scaled[:, 1] / 640.0) * crop_h + offset_y
            
            keypoints = merge_xy_conf(kpts_scaled, None if kpts_conf is None else kpts_conf[0])
            
            # Validate keypoints
            if self.validate_keypoints(keypoints):
                return {
                    "bbox": person_det["bbox"],
                    "cls": person_det["cls"],
                    "kpts": keypoints,
                    "det_score": person_det["score"]
                }
                
        return None