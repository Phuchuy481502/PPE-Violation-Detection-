import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
from typing import List, Dict, Optional, Tuple
import time
from collections import defaultdict
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Import components
from detectors.yolo_detector import YOLODetector
from poses.crop_pose import CropBasedYOLOPoseEstimator
from poses.rtmpose import RTMPoseEstimator

from utils.yaml_helper import read_yaml
from utils.function import draw_bbox
from utils.detect_css_violations import detect_css_violations, STrack
from utils.profiler import TimingProfiler
from config.color import ColorPalette

# Constants
DEFAULT_IMG_SIZE = 640
PERSON_NAME = "Person"

YOLO_COCO_SKELETON = [
    (0,1),(1,3),(0,2),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

def load_and_preprocess_image(img_path: str, size: int = DEFAULT_IMG_SIZE, device="cpu", profiler=None):
    """Load and preprocess image for inference"""
    if profiler:
        profiler.start("image_loading")
        
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    if profiler:
        profiler.end("image_loading")
        profiler.start("image_preprocessing")
        
    pil = Image.fromarray(rgb)
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    tensor = tfm(pil).unsqueeze(0).to(device)
    resized_rgb = cv2.resize(rgb, (size, size))
    
    if profiler:
        profiler.end("image_preprocessing")
        
    return rgb, resized_rgb, tensor


# Also update the detect_humans_only function:
def detect_humans_only(detector: YOLODetector, image_tensor, class_names: List[str], profiler=None) -> List[Dict]:
    """Step 1: Detect humans only"""
    if profiler:
        profiler.start("human_detection")
        
    all_detections = detector.infer(image_tensor)
    person_dets = detector.get_person_detections(all_detections, class_names, PERSON_NAME)
    
    if profiler:
        profiler.end("human_detection")
        
    print(f"Step 1: Found {len(person_dets)} humans")
    return person_dets


def run_pose_estimation_task(pose_estimator, image_tensor, resized_rgb, person_dets, profiler=None):
    """Task for parallel execution: Pose estimation on full image"""
    if profiler:
        profiler.start("parallel_pose_estimation")
        
    print(f"🏃 Starting pose estimation for {len(person_dets)} persons...")
    pose_results = pose_estimator.infer(image_tensor, resized_rgb, person_dets)
    
    if profiler:
        profiler.end("parallel_pose_estimation")
        
    print(f"🏃 Pose estimation completed: {len(pose_results)} poses")
    return pose_results

def crop_person_regions(image_rgb, person_dets: List[Dict], expand_factor=1.3):
    """Crop person regions from image with expansion for PPE detection"""
    crops = []
    crop_info = []  # Store mapping info
    
    h, w = image_rgb.shape[:2]
    
    for i, person in enumerate(person_dets):
        x1, y1, x2, y2 = person["bbox"]
        
        # Expand bounding box to capture PPE items around person
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = (x2 - x1) * expand_factor, (y2 - y1) * expand_factor
        
        # Calculate expanded coordinates
        exp_x1 = max(0, int(cx - width / 2))
        exp_y1 = max(0, int(cy - height / 2))
        exp_x2 = min(w, int(cx + width / 2))
        exp_y2 = min(h, int(cy + height / 2))
        
        # Crop the region (RGB format)
        crop = image_rgb[exp_y1:exp_y2, exp_x1:exp_x2]
        
        if crop.size > 0:
            crops.append(crop)  # Keep as numpy array in RGB format
            crop_info.append({
                'person_id': i,
                'original_bbox': [x1, y1, x2, y2],
                'crop_bbox': [exp_x1, exp_y1, exp_x2, exp_y2],
                'person_det': person
            })
    
    return crops, crop_info

def run_crop_ppe_detection_task(detector: YOLODetector, resized_rgb, person_dets: List[Dict], class_names: List[str], profiler=None):
    """Simplified task for parallel execution: PPE detection on cropped person regions using batch inference"""
    if profiler:
        profiler.start("parallel_crop_detection")
    
    print(f"🎯 Starting crop-based PPE detection for {len(person_dets)} persons...")
    
    # Step 1: Crop person regions
    if profiler:
        profiler.start("cropping_persons")
    crops, crop_info = crop_person_regions(resized_rgb, person_dets, expand_factor=1.3)
    if profiler:
        profiler.end("cropping_persons")
    
    print(f"  📦 Created {len(crops)} person crops")
    
    if not crops:
        if profiler:
            profiler.end("parallel_crop_detection")
        return []
    
    # Step 2: Run batch inference on all crops at once
    if profiler:
        profiler.start("batch_inference")
    
    # 🚀 Use native Ultralytics batch processing
    all_crop_detections = detector.infer_batch(crops)
    
    if profiler:
        profiler.end("batch_inference")
    
    # Step 3: Filter for PPE objects only and transform coordinates back
    if profiler:
        profiler.start("coordinate_transformation")
    
    final_ppe_detections = []
    
    for crop_idx, (crop_detections, crop_meta) in enumerate(zip(all_crop_detections, crop_info)):
        crop_bbox = crop_meta['crop_bbox']
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox
        crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
        
        # Filter for PPE objects only (non-person)
        ppe_detections = detector.get_object_detections(crop_detections, class_names, PERSON_NAME)
        
        for detection in ppe_detections:
            # Transform coordinates from crop space to original image space
            det_x1, det_y1, det_x2, det_y2 = detection["bbox"]
            
            # Scale from crop coordinates back to original image coordinates
            scale_x = crop_w / crops[crop_idx].shape[1]  # crop.shape[1] = width
            scale_y = crop_h / crops[crop_idx].shape[0]  # crop.shape[0] = height
            
            # Convert to original image coordinates
            orig_x1 = crop_x1 + (det_x1 * scale_x)
            orig_y1 = crop_y1 + (det_y1 * scale_y)
            orig_x2 = crop_x1 + (det_x2 * scale_x)
            orig_y2 = crop_y1 + (det_y2 * scale_y)
            
            # Create new detection with transformed coordinates
            transformed_detection = {
                "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                "score": detection["score"],
                "cls": detection["cls"],
                "person_id": crop_meta['person_id']  # Link to which person this PPE belongs
            }
            
            final_ppe_detections.append(transformed_detection)
    
    if profiler:
        profiler.end("coordinate_transformation")
        profiler.end("parallel_crop_detection")
    
    print(f"🎯 Crop-based PPE detection completed: {len(final_ppe_detections)} PPE items")
    return final_ppe_detections

def run_parallel_inference(
    detector: YOLODetector,
    pose_estimator: Optional,
    image_tensor,
    resized_rgb,
    person_dets: List[Dict],
    class_names: List[str],
    profiler: TimingProfiler = None
) -> Tuple[List[Dict], List[Dict]]:
    """Step 2: Run pose estimation and crop-based PPE detection in parallel"""
    
    if profiler:
        profiler.start("parallel_processing")
    
    # Prepare tasks
    tasks = []
    pose_results = []
    obj_dets = []
    
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference") as executor:
        
        # Submit pose estimation task (if enabled) - uses full image
        if pose_estimator is not None:
            pose_future = executor.submit(
                run_pose_estimation_task, 
                pose_estimator, image_tensor, resized_rgb, person_dets, profiler
            )
            tasks.append(("pose", pose_future))
        
        # Submit crop-based PPE detection task - uses person crops
        obj_future = executor.submit(
            run_crop_ppe_detection_task,
            detector, resized_rgb, person_dets, class_names, profiler
        )
        tasks.append(("objects", obj_future))
        
        # Collect results as they complete
        print(f"🚀 Running {len(tasks)} tasks in parallel...")
        
        for task_name, future in tasks:
            try:
                result = future.result(timeout=30)  # 30s timeout
                if task_name == "pose":
                    pose_results = result
                    print(f"✅ Pose estimation completed")
                elif task_name == "objects":
                    obj_dets = result
                    print(f"✅ Crop-based PPE detection completed")
            except Exception as e:
                print(f"❌ Task {task_name} failed: {e}")
                if task_name == "pose":
                    pose_results = []
                elif task_name == "objects":
                    obj_dets = []
    
    if profiler:
        profiler.end("parallel_processing")
    
    print(f"🎉 Parallel processing completed: {len(pose_results)} poses, {len(obj_dets)} objects")
    return pose_results, obj_dets

def draw_detections(image, detections: List[Dict], class_names, tag_type='detect', profiler=None):
    """Draw detection bounding boxes"""
    if profiler:
        profiler.start("drawing_detections")
        
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det["cls"]
        score = det["score"]
        draw_bbox(image, cls_id, x1, y1, x2, y2, score, type=tag_type, class_names=class_names)
    
    if profiler:
        profiler.end("drawing_detections")

def draw_poses(image, pose_results: List[Dict], kp_thresh: float = 0.25, skeleton=YOLO_COCO_SKELETON, profiler=None):
    """Draw pose keypoints and skeleton"""
    if profiler:
        profiler.start("drawing_poses")
        
    for pr in pose_results:
        kpts = pr["kpts"]  # (K,3)
        # skeleton lines
        for (a,b) in skeleton:
            if a < len(kpts) and b < len(kpts):
                xa, ya, ca = kpts[a]
                xb, yb, cb = kpts[b]
                if ca < kp_thresh or cb < kp_thresh:
                    continue
                if (xa == 0 and ya == 0) or (xb == 0 and yb == 0):
                    continue
                cv2.line(image, (int(xa), int(ya)), (int(xb), int(yb)), ColorPalette.SAFETY_GREEN, 2)
        # keypoints
        for (x,y,c) in kpts:
            if c < kp_thresh or (x == 0 and y == 0):
                continue
            cv2.circle(image, (int(x), int(y)), 3, ColorPalette.SAFETY_RED, -1)
            
    if profiler:
        profiler.end("drawing_poses")

def draw_violations(image, track, x1, y1, x2, y2, class_names):
    """Draw simple bounding box with PPE status"""
    from config.color import ColorPalette
    
    # Simple color logic: RED if missing PPE, GREEN if compliant
    has_violations = len(track.missing) > 0
    color = ColorPalette.VIOLATION if has_violations else ColorPalette.COMPLIANT
    
    # Draw person bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Only show text if there are violations
    if has_violations:
        # Create missing PPE text
        missing_text = f"Missing: {', '.join(track.missing)}"
        
        # Draw text background
        text_size = cv2.getTextSize(missing_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, 
                     (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0] + 10, y1), 
                     ColorPalette.VIOLATION, -1)
        
        # Draw text
        cv2.putText(image, missing_text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ColorPalette.TEXT_WHITE, 2)

def build_pose_estimator(args, device, profiler=None):
    """Build pose estimator based on arguments"""
    if not args.enable_pose:
        return None
        
    if profiler:
        profiler.start("pose_model_initialization")
        
    backend = args.pose_backend.lower()
    estimator = None
    
    try:
        if backend == "yolo":
            if not args.pose_weights:
                raise ValueError("Provide --pose_weights for YOLO pose backend.")
            estimator = CropBasedYOLOPoseEstimator(args.pose_weights, device=device)
        elif backend == "rtmpose":
            if not args.pose_weights:
                raise ValueError("Provide --pose_weights (ONNX model path) for RTMPose backend.")
            input_size = (192, 256)  # default
            if args.pose_input_size:
                w, h = map(int, args.pose_input_size.split(','))
                input_size = (w, h)
            rtm_backend = args.rtmpose_backend if hasattr(args, 'rtmpose_backend') else 'onnxruntime'
            
            estimator = RTMPoseEstimator(
                onnx_model_path=args.pose_weights,
                model_input_size=input_size,
                backend=rtm_backend,
                device=device
            )
        elif backend == "none":
            estimator = None
        else:
            raise ValueError(f"Unknown pose backend: {backend}")
    finally:
        if profiler:
            profiler.end("pose_model_initialization")
            
    return estimator

def build_detector(weights, device, conf):
    """Build detector with optimized settings"""
    detector = YOLODetector(weights, device=device, conf=conf)
    
    # Warmup
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        _ = detector.model(dummy_input)
    
    return detector

def run_parallel_pipeline(
    img_path: str,
    detector: YOLODetector,
    pose_estimator: Optional,
    class_names: List[str],
    kp_thresh: float = 0.25,
    profiler: TimingProfiler = None
):
    """Optimized parallel pipeline with simplified batch processing"""
    device = detector.device
    
    print(f"\n🚀 Starting optimized parallel pipeline...")
    print(f"  📊 Pipeline: Human Detection → Parallel(Pose + Batch-Crop-PPE) → Violations")
    
    # Image loading and preprocessing
    _, resized_rgb, image_tensor = load_and_preprocess_image(
        img_path, size=DEFAULT_IMG_SIZE, device=device, profiler=profiler
    )

    # Step 1: Detect humans only (fast, focused detection)
    person_dets = detect_humans_only(detector, image_tensor, class_names, profiler)
    
    if len(person_dets) == 0:
        print("⚠️  No humans detected, skipping pose and violation detection")
        # Create empty results
        image_detection = resized_rgb.copy()
        image_violate = resized_rgb.copy()
        pose_image = resized_rgb.copy()
        return image_detection, image_violate, pose_image
    
    # Step 2: Parallel processing (Pose estimation + Batch crop-based PPE detection)
    pose_results, obj_dets = run_parallel_inference(
        detector, pose_estimator, image_tensor, resized_rgb, 
        person_dets, class_names, profiler
    )
    
    # Step 3: Violation detection
    if profiler:
        profiler.start("violation_detection")
    
    print(f"🔍 Step 3: Running violation detection...")
    
    online_targets = []
    for i, pd in enumerate(person_dets):
        x1, y1, x2, y2 = pd["bbox"]
        tlwh = [x1, y1, x2 - x1, y2 - y1]
        online_targets.append(STrack(tlwh, pd["score"], i))

    obj_det_list = []
    for od in obj_dets:
        x1, y1, x2, y2 = od["bbox"]
        obj_det_list.append([x1, y1, x2, y2, od["score"], od["cls"]])

    # Import and use the simplified detection function
    online_targets = detect_css_violations(
        online_targets, 
        obj_det_list, 
        pose_results=pose_results,
        use_pose=(pose_estimator is not None),
        z_factor=1.2,
        class_names=class_names
    )
    
    if profiler:
        profiler.end("violation_detection")
    
    print(f"🔍 Violation detection completed for {len(online_targets)} persons")
    
    # Step 4: Generate visualization images (same as before)
    if profiler:
        profiler.start("image_generation")
    
    # Detection image (all detections)
    image_detection = resized_rgb.copy()
    all_detections = person_dets + obj_dets
    draw_detections(image_detection, all_detections, class_names, tag_type='detect', profiler=profiler)
    
    # Pose image
    pose_image = None
    if pose_results:
        pose_image = resized_rgb.copy()
        # Draw person bounding boxes
        for pr in pose_results:
            x1, y1, x2, y2 = map(int, pr["bbox"])
            cv2.rectangle(pose_image, (x1, y1), (x2, y2), ColorPalette.PERSON_COLOR, 2)
        draw_poses(pose_image, pose_results, kp_thresh=kp_thresh, profiler=profiler)
    else:
        pose_image = resized_rgb.copy()
    
    # Violation image
    image_violate = resized_rgb.copy()
    for t in online_targets:
        x, y, w, h = t.tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        draw_violations(image_violate, t, x1, y1, x2, y2, class_names)
    
    if profiler:
        profiler.end("image_generation")
    
    print(f"✅ Optimized parallel pipeline completed successfully!")

    return image_detection, image_violate, pose_image

def save_outputs(det_img, vio_img, pose_img, out_dir="output", base_name="parallel_inference", profiler=None):
    """Save output images"""
    if profiler:
        profiler.start("saving_outputs")
        
    os.makedirs(out_dir, exist_ok=True)
    num = 1
    while os.path.exists(os.path.join(out_dir, f"{base_name}-{num}-yolo.jpg")):
        num += 1

    det_path = os.path.join(out_dir, f"{base_name}-{num}-yolo.jpg")
    vio_path = os.path.join(out_dir, f"{base_name}-{num}-yolo-violate.jpg")
    cv2.imwrite(det_path, cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(vio_path, cv2.cvtColor(vio_img, cv2.COLOR_RGB2BGR))
    print(f"💾 Saved detection: {det_path}")
    print(f"💾 Saved violation: {vio_path}")

    if pose_img is not None:
        pose_path = os.path.join(out_dir, f"{base_name}-{num}-yolo-pose.jpg")
        cv2.imwrite(pose_path, cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
        print(f"💾 Saved pose: {pose_path}")
        
    if profiler:
        profiler.end("saving_outputs")

def print_system_info():
    """Print system information for debugging"""
    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"OpenCV version: {cv2.__version__}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Parallel Pipeline: Human Detection → Parallel(Pose + Objects) → Violations")
    parser.add_argument("--det_weights", type=str, required=True, help="Detection weights (YOLO)")
    parser.add_argument("--pose_backend", type=str, default="none", 
                        choices=["none", "yolo", "yolo_crop", "mmpose", "rtmpose"],
                        help="Pose backend")
    parser.add_argument("--pose_weights", type=str, default="", help="Pose weights (ONNX for RTMPose)")
    parser.add_argument("--pose_config", type=str, default="", help="MMPose config")
    parser.add_argument("--pose_input_size", type=str, default="192,256", 
                        help="RTMPose input size as 'width,height'")
    parser.add_argument("--rtmpose_backend", type=str, default="onnxruntime",
                        choices=["opencv", "onnxruntime", "openvino"],
                        help="RTMPose backend")
    parser.add_argument("--enable_pose", action="store_true", help="Enable pose estimation")
    parser.add_argument("--kp_thresh", type=float, default=0.25, help="Keypoint confidence threshold")
    parser.add_argument("--img_path", type=str, required=True, help="Image path")
    parser.add_argument("--yaml_class", type=str, required=True, help="Class YAML path")
    parser.add_argument("--det_conf", type=float, default=0.45, help="Detection confidence threshold")
    parser.add_argument("--profile", action="store_true", help="Enable detailed timing profiling")
    parser.add_argument("--system_info", action="store_true", help="Show system information")
    args = parser.parse_args()

    # Initialize profiler
    profiler = TimingProfiler() if args.profile else None
    
    # Start total timing
    total_start_time = time.time()
    
    if args.system_info:
        print_system_info()

    # Load configuration
    if profiler:
        profiler.start("config_loading")
    yaml_data = read_yaml(args.yaml_class)
    class_names = yaml_data["names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if profiler:
        profiler.end("config_loading")

    # Initialize detector
    if profiler:
        profiler.start("detector_initialization")
    detector = build_detector(args.det_weights, device=device, conf=args.det_conf)
    if profiler:
        profiler.end("detector_initialization")
        
    # Initialize pose estimator
    pose_estimator = build_pose_estimator(args, device, profiler)
    
    # 🚀 START INFERENCE TIMING HERE (after initialization)
    inference_start_time = time.time()

    det_img, vio_img, pose_img = run_parallel_pipeline(
        img_path=args.img_path,
        detector=detector,
        pose_estimator=pose_estimator,
        class_names=class_names,
        kp_thresh=args.kp_thresh,
        profiler=profiler
    )

    # Save outputs
    save_outputs(det_img, vio_img, pose_img, profiler=profiler)
    
    # 🚀 END INFERENCE TIMING HERE
    inference_time = time.time() - inference_start_time
    total_time = time.time() - total_start_time
    
    print(f"\n🎉 Parallel pipeline completed successfully!")
    print(f"📊 Total execution time: {total_time:.3f} seconds")
    
    if profiler:
        profiler.print_summary()
        
        # 🎯 FPS CALCULATIONS - Only inference time
        inference_fps = 1.0 / inference_time
        
        # Get pure inference operations (no init, no saving)
        summary = profiler.get_summary()
        core_inference_ops = ['human_detection', 'parallel_processing', 'violation_detection']
        core_inference_time = sum(summary[op]['total'] for op in core_inference_ops if op in summary)
        core_fps = 1.0 / core_inference_time if core_inference_time > 0 else 0
        
        print(f"\n" + "="*50)
        print("📊 PARALLEL PIPELINE PERFORMANCE")
        print("="*50)
        print(f"⏱️  Total inference time:     {inference_time:.3f}s")
        print(f"⏱️  Core inference time:      {core_inference_time:.3f}s")
        print(f"🚀 Inference FPS:             {inference_fps:.2f} fps")
        print(f"🚀 Core inference FPS:        {core_fps:.2f} fps")
        print(f"📊 Total pipeline time:       {total_time:.3f}s")
        print(f"📊 Initialization overhead:   {(total_time - inference_time):.3f}s")
        
        # Parallel processing breakdown
        parallel_time = summary.get('parallel_processing', {}).get('total', 0)
        pose_time = summary.get('parallel_pose_estimation', {}).get('total', 0)
        obj_time = summary.get('parallel_object_detection', {}).get('total', 0)
        
        print(f"\n🔀 PARALLEL PROCESSING BREAKDOWN:")
        print(f"  📊 Total parallel time:      {parallel_time:.3f}s")
        print(f"  🏃 Pose estimation:          {pose_time:.3f}s")
        print(f"  🎯 Object detection:         {obj_time:.3f}s")
        if parallel_time > 0:
            efficiency = max(pose_time, obj_time) / parallel_time
            print(f"  ⚡ Parallel efficiency:       {efficiency:.1%}")
        print("="*50)

if __name__ == "__main__":
    main()