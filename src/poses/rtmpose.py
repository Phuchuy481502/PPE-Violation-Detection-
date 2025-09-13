import numpy as np
import cv2
import time
import os
from typing import List, Dict, Optional
from .base import BasePoseEstimator

class RTMPoseEstimator(BasePoseEstimator):
    """RTMPose estimator using rtmlib library with optimized direct calls"""
    
    def __init__(self, onnx_model_path: str, model_input_size: tuple = (192, 256), 
                 backend: str = 'onnxruntime', device: str = "cpu"):
        self.onnx_model_path = onnx_model_path
        self.model_input_size = model_input_size
        self.backend = backend
        super().__init__(onnx_model_path, device)
        
    def load_model(self):
        """Load RTMPose model with proper warmup"""
        try:
            from rtmlib import RTMPose
            
            print(f"🚀 Loading RTMPose on {self.device}")
            self.model = RTMPose(
                onnx_model=self.onnx_model_path,
                model_input_size=self.model_input_size,
                backend=self.backend,
                device=self.device
            )
            # warm up for fast inference
            self._warmup_model()
            
            
        except ImportError:
            raise ImportError("rtmlib not installed. Install with: pip install rtmlib")
        except Exception as e:
            print(f"❌ RTMPose loading error: {e}")
            raise

    def _warmup_model(self):
        """Warmup RTMPose model to avoid cold start penalty"""
        try:
            # Create dummy input matching your typical pipeline
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            dummy_bbox = [[100, 100, 200, 300]]  # Single dummy bbox
            
            warmup_start = time.time()
            
            # Run several warmup iterations
            for i in range(3):
                _, _ = self.model(dummy_img, dummy_bbox)
                
            warmup_time = time.time() - warmup_start
            print(f"  ✅ Warmup completed in {warmup_time*1000:.1f}ms (3 iterations)")
            
        except Exception as e:
            print(f"  ⚠️  Warmup failed: {e} (model will still work, but first inference will be slow)")

    def infer(self, image_tensor, resized_rgb: np.ndarray, person_dets: List[Dict]) -> List[Dict]:
        """Run RTMPose inference (now should be fast after warmup!)"""
        if not person_dets:
            return []
        
        total_start = time.time()
        
        # Convert to BGR (slight performance improvement)
        resized_bgr = cv2.cvtColor(resized_rgb, cv2.COLOR_RGB2BGR)
        
        # Prepare bboxes
        bboxes = []
        for det in person_dets:
            x1, y1, x2, y2 = det["bbox"]
            bboxes.append([int(x1), int(y1), int(x2), int(y2)])
        
        if not bboxes:
            return []
        
        try:
            # This should now be consistently fast (~30-40ms)
            inference_start = time.time()
            keypoints, scores = self.model(resized_bgr, bboxes)
            inference_time = time.time() - inference_start
            
            # Process results (same as before)
            pose_results = []
            if isinstance(keypoints, np.ndarray) and len(keypoints.shape) == 3:
                for i, (person_kpts, person_scores) in enumerate(zip(keypoints, scores)):
                    if i >= len(person_dets):
                        break
                        
                    processed_keypoints = self._process_rtmpose_output(person_kpts, person_scores)
                    
                    pose_results.append({
                        "bbox": person_dets[i]["bbox"],
                        "cls": person_dets[i]["cls"],
                        "kpts": processed_keypoints,
                        "det_score": person_dets[i]["score"]
                    })
            
            total_time = time.time() - total_start
            # print(f"RTMPose: {inference_time*1000:.1f}ms inference, {total_time*1000:.1f}ms total")
            
            return pose_results
            
        except Exception as e:
            print(f"❌ RTMPose inference error: {e}")
            return []
            
    def _process_rtmpose_output(self, keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Process RTMPose output format efficiently"""
        if scores is None:
            scores = np.ones(keypoints.shape[0], dtype=np.float32) * 0.5
        elif hasattr(scores, 'shape') and len(scores.shape) > 1:
            scores = scores.flatten()[:keypoints.shape[0]]
        
        if keypoints.shape[0] != scores.shape[0]:
            min_len = min(keypoints.shape[0], len(scores))
            keypoints = keypoints[:min_len]
            scores = scores[:min_len] if len(scores) >= min_len else np.pad(scores, (0, keypoints.shape[0] - len(scores)), constant_values=0.5)
            
        processed_kpts = np.zeros((keypoints.shape[0], 3), dtype=np.float32)
        processed_kpts[:, :2] = keypoints
        processed_kpts[:, 2] = scores
        
        return processed_kpts
    
    def validate_keypoints(self, keypoints: np.ndarray, min_confidence: float = 0.1) -> bool:
        """Fast keypoint validation"""
        if keypoints is None or keypoints.size == 0:
            return False
        if keypoints.shape[1] < 3:
            return False
        valid_count = np.sum(keypoints[:, 2] > min_confidence)
        return valid_count > 2