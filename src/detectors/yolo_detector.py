import torch
import cv2
import numpy as np
from typing import List, Dict
from ultralytics import YOLO
from detectors.base import BaseDetector
import os

class YOLODetector(BaseDetector):
    """YOLO-based object detector supporting both PyTorch (.pt) and ONNX (.onnx) formats"""
    
    def __init__(self, weights: str, device: str = "cpu", conf: float = 0.5):
        self.weights = weights
        self.device = device
        self.conf = conf
        self.is_exported = not weights.lower().endswith('.pt')
        super().__init__(weights, device, conf)
        self.load_model()
        
    def load_model(self):
        print(f"🚀 Loading YOLO model: {self.weights}")
        print(f"  📊 Format: {'Exported (ONNX/TensorRT/etc.)' if self.is_exported else 'PyTorch'}")
        print(f"  🖥️  Device: {self.device}")
        
        try:
            self.model = YOLO(self.weights, task='detect')
            if not self.is_exported:
                print("⚙️  Applying PyTorch optimizations...")
                
                # Move to device
                if self.device != "cpu":
                    self.model.to(self.device)
                
                # Apply half precision for CUDA
                if self.device != "cpu" and torch.cuda.is_available():
                    try:
                        self.model.half()
                        self.model.eval()
                        print("  ✅ Half precision enabled")
                    except Exception as e:
                        print(f"  ⚠️  Half precision failed: {e}")
            
            print("✅ YOLO model loaded successfully")
                        
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            raise
        
    def infer(self, image_tensor: torch.Tensor) -> List[Dict]:
        """Single image inference"""
        try:
            if self.is_exported:
                results = self.model(image_tensor, device=self.device, verbose=False, stream=True)
            else:
                with torch.no_grad():
                    results = self.model(image_tensor, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        score = float(box.conf[0])
                        if score < self.conf:
                            continue
                            
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": score,
                            "cls": cls_id
                        })
                        
            return detections
            
        except Exception as e:
            print(f"❌ YOLO inference failed: {e}")
            return []
    
    def infer_batch(self, image_crops: List[np.ndarray]) -> List[List[Dict]]:
        """
        Batch inference on multiple image crops
        
        Args:
            image_crops: List of numpy arrays (H, W, C) in RGB format
            
        Returns:
            List of detection lists, one for each crop
        """
        if not image_crops:
            return []
        
        try:
            print(f"🔄 Running batch inference on {len(image_crops)} crops...")
            
            # Ultralytics handles batch processing automatically
            if self.is_exported:
                results = self.model(image_crops, device=self.device, verbose=False)
            else:
                with torch.no_grad():
                    results = self.model(image_crops, verbose=False)
            
            # Process results for each crop
            all_detections = []
            
            for i, result in enumerate(results):
                detections = []
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        score = float(box.conf[0])
                        if score < self.conf:
                            continue
                            
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                        
                        detections.append({
                            "bbox": [x1, y1, x2, y2],
                            "score": score,
                            "cls": cls_id
                        })
                
                all_detections.append(detections)
                
            print(f"✅ Batch inference completed: {sum(len(dets) for dets in all_detections)} total detections")
            return all_detections
            
        except Exception as e:
            print(f"❌ Batch inference failed: {e}")
            return [[] for _ in image_crops]  # Return empty lists for each crop
        
    def get_person_detections(self, detections: List[Dict], class_names: List[str], person_name: str = "Person") -> List[Dict]:
        """Extract person detections specifically"""
        return self.filter_by_class(detections, class_names, person_name)
        
    def get_object_detections(self, detections: List[Dict], class_names: List[str], person_name: str = "Person") -> List[Dict]:
        """Extract non-person object detections"""
        objects = []
        for det in detections:
            cls_id = det["cls"]
            if 0 <= cls_id < len(class_names):
                class_name = class_names[cls_id].lower()
                if class_name != person_name.lower():
                    objects.append(det)
        return objects