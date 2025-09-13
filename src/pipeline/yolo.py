import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import List, Dict, Optional

from utils.function import draw_bbox
from utils.detect_css_violations import detect_css_violations, STrack
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

# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def load_and_preprocess_image(img_path: str, size: int = DEFAULT_IMG_SIZE, device="cpu", dtype=torch.float32):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    # Resize in BGR (fast), then convert once to RGB
    resized_bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    # HWC RGB -> CHW, normalize in one shot
    tensor = torch.from_numpy(resized_rgb).to(device)
    tensor = tensor.permute(2, 0, 1).contiguous().to(dtype) / 255.0
    tensor = tensor.unsqueeze(0)  # (1,3,H,W)

    # Return original rgb only if you need it; keep None to avoid extra copy
    return None, resized_rgb, tensor

def preprocess_frame(frame: np.ndarray, size: int = DEFAULT_IMG_SIZE, device="cpu", dtype=torch.float32):
    resized_bgr = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(resized_rgb).to(device)
    tensor = tensor.permute(2, 0, 1).contiguous().to(dtype) / 255.0
    tensor = tensor.unsqueeze(0)
    return None, resized_rgb, tensor

# =============================================================================
# DETECTION FUNCTIONS (MODULAR)
# =============================================================================

def run_object_detection(detector, image_tensor):
    """Run object detection on image tensor"""
    all_detections = detector.infer(image_tensor)
    return all_detections

def split_detections(all_detections: List[Dict], detector, class_names: List[str], person_name: str = PERSON_NAME):
    """Split detections into persons and objects"""
    person_dets = detector.get_person_detections(all_detections, class_names, person_name)
    obj_dets = detector.get_object_detections(all_detections, class_names, person_name)
    return person_dets, obj_dets

def run_pose_estimation(pose_estimator, image_tensor, resized_rgb, person_dets: List[Dict]):
    """Run pose estimation on person detections"""
    pose_results = []
    
    if pose_estimator is not None and person_dets:
        pose_results = pose_estimator.infer(image_tensor, resized_rgb, person_dets)
    
    return pose_results



# =============================================================================
# VIOLATION DETECTION FUNCTIONS
# =============================================================================

def prepare_violations_detection(person_dets: List[Dict], obj_dets: List[Dict]):
    """Prepare data for violation detection"""
    # Convert person detections to STrack format
    online_targets = []
    for i, pd in enumerate(person_dets):
        x1, y1, x2, y2 = pd["bbox"]
        tlwh = [x1, y1, x2 - x1, y2 - y1]
        # Preserve track_id if available (from tracking)
        track_id = pd.get("track_id", i)
        strack = STrack(tlwh, pd["score"], track_id)
        online_targets.append(strack)

    # Convert objects to expected format
    obj_det_list = []
    for od in obj_dets:
        x1, y1, x2, y2 = od["bbox"]
        obj_det_list.append([x1, y1, x2, y2, od["score"], od["cls"]])
    
    return online_targets, obj_det_list

def run_violation_detection(
    person_dets: List[Dict],
    obj_dets: List[Dict], 
    pose_results: List[Dict],
    pose_estimator: Optional,
    class_names: List[str]
):
    
    # Prepare data
    online_targets, obj_det_list = prepare_violations_detection(person_dets, obj_dets)
    
    # Run violation detection
    online_targets = detect_css_violations(
        online_targets, 
        obj_det_list, 
        pose_results=pose_results,
        use_pose=(pose_estimator is not None),
        z_factor=1.2,
        class_names=class_names
    )
    
    return online_targets

# =============================================================================
# MODEL BUILDING FUNCTIONS
# =============================================================================

def build_pose_estimator(pose_backend, pose_weights, pose_input_size=None, rtmpose_backend="onnxruntime", device="cpu"):
    """Build pose estimator based on arguments"""
    if pose_backend == "none":
        return None
        
    from poses.crop_pose import CropBasedYOLOPoseEstimator
    from poses.rtmpose import RTMPoseEstimator
    
    backend = pose_backend.lower()
    estimator = None
    
    if backend == "yolo":
        if not pose_weights:
            raise ValueError("Provide pose_weights for YOLO pose backend.")
        estimator = CropBasedYOLOPoseEstimator(pose_weights, device=device)
    elif backend == "rtmpose":
        if not pose_weights:
            raise ValueError("Provide pose_weights (ONNX model path) for RTMPose backend.")
        input_size = (192, 256)  # default
        if pose_input_size:
            w, h = map(int, pose_input_size.split(','))
            input_size = (w, h)
        estimator = RTMPoseEstimator(
            onnx_model_path=pose_weights,
            model_input_size=input_size,
            backend=rtmpose_backend,
            device=device
        )
    else:
        raise ValueError(f"Unknown pose backend: {backend}")
            
    return estimator

def build_detector(weights, device, conf):
    """Build detector with optimized settings"""
    from detectors.yolo_detector import YOLODetector
    
    detector = YOLODetector(weights, device=device, conf=conf)
    
    # Warmup
    dummy_input = torch.randn(1, 3, 640, 640).to(device)
    with torch.no_grad():
        _ = detector.model(dummy_input)
    
    return detector

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def draw_detections(image, detections: List[Dict], class_names, tag_type='detect'):
    """Draw detection bounding boxes"""
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det["cls"]
        score = det["score"]
        draw_bbox(image, cls_id, x1, y1, x2, y2, score, type=tag_type, class_names=class_names)

def draw_poses(image, pose_results: List[Dict], kp_thresh: float = 0.25, skeleton=YOLO_COCO_SKELETON):
    """Draw pose keypoints and skeleton"""
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

def draw_violations(image, track, x1, y1, x2, y2, class_names):
    """Draw simple bounding box with PPE status"""
    # Simple color logic: RED if missing PPE, GREEN if compliant
    has_violations = len(track.missing) > 0
    color = ColorPalette.VIOLATION if has_violations else ColorPalette.COMPLIANT
    # Draw person bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    
    # Only show text if there are violations
    if has_violations:
        # Create missing PPE text
        missing_text = f"{track.track_id} Missing: {', '.join(track.missing)}"

        # Draw text background
        text_size = cv2.getTextSize(missing_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(image, 
                     (x1, y1 - text_size[1] - 10), 
                     (x1 + text_size[0] + 10, y1), 
                     ColorPalette.VIOLATION, -1)
        
        # Draw text
        cv2.putText(image, missing_text, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ColorPalette.TEXT_WHITE, 2)

def generate_output_images(
    resized_rgb,
    all_detections: List[Dict],
    pose_results: List[Dict],
    online_targets: List,
    class_names: List[str],
    kp_thresh: float = 0.25,
    debug: bool = False  # 🎯 ADD: Control what images to generate
):
    """Generate output images - only violation by default, all in debug mode"""
    
    # Always generate violation image (main output)
    image_violate = resized_rgb.copy()
    for t in online_targets:
        x, y, w, h = t.tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        draw_violations(image_violate, t, x1, y1, x2, y2, class_names)

    # Only generate detection and pose images in debug mode
    image_detection = None
    pose_image = None
    
    if debug:
        # Detection image (all detections)
        image_detection = resized_rgb.copy()
        draw_detections(image_detection, all_detections, class_names, tag_type='detect')
        
        # Pose image
        if pose_results:
            pose_image = resized_rgb.copy()
            # Draw person bounding boxes
            for pr in pose_results:
                x1, y1, x2, y2 = map(int, pr["bbox"])
                cv2.rectangle(pose_image, (x1, y1), (x2, y2), ColorPalette.PERSON_COLOR, 2)
            draw_poses(pose_image, pose_results, kp_thresh=kp_thresh)
        else:
            pose_image = resized_rgb.copy()

    return image_detection, image_violate, pose_image