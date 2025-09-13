import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Optional

from detectors.yolo_detector import YOLODetector
from poses.rtmpose import RTMPoseEstimator
from utils.yaml_helper import read_yaml

DEFAULT_IMG_SIZE = 640
PERSON_NAME = "Person"

def load_and_preprocess_image_exact(img_path: str, size: int = DEFAULT_IMG_SIZE, device="cpu"):
    """Load and preprocess image EXACTLY like pipeline"""
    print("📸 Loading image exactly like pipeline...")
    start_time = time.time()
    
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    pil = Image.fromarray(rgb)
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    tensor = tfm(pil).unsqueeze(0).to(device)
    resized_rgb = cv2.resize(rgb, (size, size))
    
    load_time = time.time() - start_time
    print(f"  ⏱️  Image loading/preprocessing: {load_time*1000:.1f}ms")
    
    return bgr, rgb, resized_rgb, tensor

def test_pipeline_exact():
    """Test RTMPose using EXACT same pipeline approach"""
    
    print("🔬 TESTING PIPELINE-EXACT RTMPose Processing")
    print("="*60)
    
    img_path = "sample/images/17.jpg"
    weights_path = "weights/best_yolo_10_70epochs.pt"
    pose_weights = "weights/rtmpose-t-17.onnx"
    yaml_path = "data/data-ppe_v4-kaggle.yaml"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    # Step 1: Load YAML 
    print("\n1️⃣  Loading YAML config...")
    yaml_data = read_yaml(yaml_path)
    class_names = yaml_data["names"]
    
    # Step 2: Initialize detector 
    print("2️⃣  Initializing detector...")
    detector_start = time.time()
    detector = YOLODetector(weights_path, device=device, conf=0.4)
    detector_time = time.time() - detector_start
    print(f"  ⏱️  Detector init: {detector_time*1000:.1f}ms")
    
    # Step 3: Initialize RTMPose
    print("3️⃣  Initializing RTMPose...")
    pose_start = time.time()
    pose_estimator = RTMPoseEstimator(
        onnx_model_path=pose_weights,
        model_input_size=(192, 256),
        backend='onnxruntime',
        device=device
    )
    pose_init_time = time.time() - pose_start
    print(f"  ⏱️  RTMPose init: {pose_init_time*1000:.1f}ms")
    
    # Step 4: Load and preprocess image 
    print("4️⃣  Loading and preprocessing image...")
    bgr, rgb, resized_rgb, image_tensor = load_and_preprocess_image_exact(img_path, DEFAULT_IMG_SIZE, device)
    
    # Step 5: Detection 
    print("5️⃣  Running object detection...")
    detection_start = time.time()
    all_detections = detector.infer(image_tensor)
    detection_time = time.time() - detection_start
    print(f"  ⏱️  Detection: {detection_time*1000:.1f}ms")
    print(f"  📊 Found {len(all_detections)} total detections")
    
    # Step 6: Filter persons 
    print("6️⃣  Filtering person detections...")
    filter_start = time.time()
    person_dets = detector.get_person_detections(all_detections, class_names, PERSON_NAME)
    filter_time = time.time() - filter_start
    print(f"  ⏱️  Filtering: {filter_time*1000:.1f}ms")
    print(f"  👥 Found {len(person_dets)} persons")
    
    if not person_dets:
        print("❌ No persons detected, cannot test pose estimation")
        return
    
    for i, det in enumerate(person_dets):
        x1, y1, x2, y2 = det["bbox"]
        print(f"    Person {i+1}: bbox=[{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}], score={det['score']:.3f}")
    
    # Step 7: RTMPose inference 
    print("\n7️⃣  Running RTMPose inference EXACTLY like pipeline...")
        # Warm up
    for i in range(3):
        _ = pose_estimator.infer(image_tensor, resized_rgb, person_dets)

    pose_inference_start = time.time()
    pose_results = pose_estimator.infer(image_tensor, resized_rgb, person_dets)
    
    pose_inference_time = time.time() - pose_inference_start
    print(f"  ⏱️  RTMPose pipeline call: {pose_inference_time*1000:.1f}ms")
    print(f"  🎯 Pose results: {len(pose_results)} poses found")
    
    # Step 8: Compare with direct RTMPose call (like your fast test)
    print("\n8️⃣  Comparing with DIRECT RTMPose call...")
    
    bboxes = []
    for det in person_dets:
        x1, y1, x2, y2 = det["bbox"]
        bboxes.append([int(x1), int(y1), int(x2), int(y2)])
    
    direct_start = time.time()
    try:
        keypoints, scores = pose_estimator.model(resized_rgb, bboxes)
        direct_time = time.time() - direct_start
        print(f"  🚀 Direct RTMPose call: {direct_time*1000:.1f}ms")
        print(f"  📊 Direct result shape: {keypoints.shape if isinstance(keypoints, np.ndarray) else 'N/A'}")
    except Exception as e:
        print(f"  ❌ Direct call failed: {e}")
        direct_time = 0
    
    # Step 9: Analysis
    print("\n" + "="*60)
    print("📊 PERFORMANCE ANALYSIS")
    print("="*60)
    print(f"Detector initialization:    {detector_time*1000:>8.1f}ms")
    print(f"RTMPose initialization:     {pose_init_time*1000:>8.1f}ms")
    print(f"Image loading/preprocessing: {0:>8.1f}ms (included above)")
    print(f"Object detection:           {detection_time*1000:>8.1f}ms")
    print(f"Person filtering:           {filter_time*1000:>8.1f}ms")
    print(f"RTMPose (pipeline wrapper): {pose_inference_time*1000:>8.1f}ms ⚠️")
    print(f"RTMPose (direct call):      {direct_time*1000:>8.1f}ms ✅")
    print("-"*60)
    
    if direct_time > 0:
        slowdown_factor = pose_inference_time / direct_time
        overhead = pose_inference_time - direct_time
        print(f"Pipeline wrapper overhead:  {overhead*1000:>8.1f}ms")
        print(f"Slowdown factor:            {slowdown_factor:>8.1f}x")
        
        if slowdown_factor > 2:
            print("\n🚨 ISSUE FOUND: Pipeline wrapper is significantly slower!")
            print("   The bottleneck is in the RTMPoseEstimator.infer() method")
        else:
            print("\n✅ Pipeline wrapper performance is reasonable")
    
    print("\n🔍 RECOMMENDATIONS:")
    if pose_inference_time > 100:  # > 100ms
        print("  • RTMPose inference is slow - check GPU utilization")
        print("  • Consider reducing model input size")
        print("  • Check if CUDA providers are properly loaded")
    if direct_time < 50 and pose_inference_time > direct_time * 2:
        print("  • Pipeline wrapper has significant overhead")
        print("  • Review RTMPoseEstimator.infer() implementation")
        print("  • Consider calling RTMPose model directly")

if __name__ == "__main__":
    test_pipeline_exact()