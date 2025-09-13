import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))
import time
import cv2
import torch

from utils.yaml_helper import read_yaml
from utils.profiler import TimingProfiler
from parsers.detect_yolo import parse_args
from pipeline.yolo import (
    load_and_preprocess_image,
    build_detector, 
    build_pose_estimator,
    run_object_detection,
    split_detections,
    run_pose_estimation,
    run_violation_detection,
    generate_output_images
)

def run_pipeline(
    img_path: str,
    detector,
    pose_estimator,
    class_names,
    kp_thresh: float = 0.25,
    debug: bool = False,  # 🎯 ADD: Debug flag
    profiler=None
):
    """Static image pipeline: Detection → Pose → Violation with profiling"""
    device = detector.device
    
    # Image loading and preprocessing
    if profiler:
        profiler.start("image_loading")
    _, resized_rgb, image_tensor = load_and_preprocess_image(img_path, device=device)
    if profiler:
        profiler.end("image_loading")

    # Step 1: Object detection
    if profiler:
        profiler.start("object_detection")
    all_detections = run_object_detection(detector, image_tensor)
    if profiler:
        profiler.end("object_detection")
    
    # Step 2: Split detections
    if profiler:
        profiler.start("detection_filtering")
    person_dets, obj_dets = split_detections(all_detections, detector, class_names, "Person")
    if profiler:
        profiler.end("detection_filtering")
    
    # Step 3: Pose estimation
    if profiler:
        profiler.start("pose_estimation")
    pose_results = run_pose_estimation(pose_estimator, image_tensor, resized_rgb, person_dets)
    if profiler:
        profiler.end("pose_estimation")

    if debug:
        print(f"Found {len(all_detections)} total detections")
        print(f"Found {len(person_dets)} persons and {len(obj_dets)} objects")
        if pose_results:
            print(f"Found {len(pose_results)} poses")

    # Step 4: Violation detection
    if profiler:
        profiler.start("violation_detection")
    online_targets = run_violation_detection(
        person_dets, obj_dets, pose_results, pose_estimator, class_names
    )
    if profiler:
        profiler.end("violation_detection")

    # Step 5: Generate output images - 🎯 Pass debug flag
    if profiler:
        profiler.start("image_generation")
    image_detection, image_violate, pose_image = generate_output_images(
        resized_rgb, all_detections, pose_results, online_targets, class_names, kp_thresh, debug=debug
    )
    if profiler:
        profiler.end("image_generation")

    return image_detection, image_violate, pose_image

def save_outputs(det_img, vio_img, pose_img, debug: bool = False, out_dir="output", base_name="inference", profiler=None):
    """Save output images - only violation by default, all in debug mode"""
    if profiler:
        profiler.start("saving_outputs")
        
    os.makedirs(out_dir, exist_ok=True)
    num = 1
    while os.path.exists(os.path.join(out_dir, f"{base_name}-{num}-yolo-violate.jpg")):
        num += 1

    # Always save violation image (main output)
    vio_path = os.path.join(out_dir, f"{base_name}-{num}-yolo-violate.jpg")
    cv2.imwrite(vio_path, cv2.cvtColor(vio_img, cv2.COLOR_RGB2BGR))
    print(f"💾 Saved violation: {vio_path}")

    # Only save detection and pose images in debug mode
    if debug:
        if det_img is not None:
            det_path = os.path.join(out_dir, f"{base_name}-{num}-yolo.jpg")
            cv2.imwrite(det_path, cv2.cvtColor(det_img, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved detection: {det_path}")

        if pose_img is not None:
            pose_path = os.path.join(out_dir, f"{base_name}-{num}-yolo-pose.jpg")
            cv2.imwrite(pose_path, cv2.cvtColor(pose_img, cv2.COLOR_RGB2BGR))
            print(f"💾 Saved pose: {pose_path}")
        
    if profiler:
        profiler.end("saving_outputs")

def main():
    args = parse_args()

    # Initialize profiler
    profiler = TimingProfiler() if args.profile else None
    total_start_time = time.time()

    # Load configuration
    if profiler:
        profiler.start("config_loading")
    yaml_data = read_yaml(args.yaml_class)
    class_names = yaml_data["names"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    if args.debug:
        print(f"🔍 Debug mode: ON - Will save detection and pose images")
    else:
        print(f"🚀 Production mode: ON - Will only save violation image")
    if profiler:
        profiler.end("config_loading")

    # Initialize detector
    if profiler:
        profiler.start("detector_initialization")
    detector = build_detector(args.det_weights, device, args.det_conf)
    if profiler:
        profiler.end("detector_initialization")
        
    # Initialize pose estimator
    pose_estimator = None
    if args.enable_pose:
        if profiler:
            profiler.start("pose_model_initialization")
        pose_estimator = build_pose_estimator(
            args.pose_backend, args.pose_weights, args.pose_input_size, 
            args.rtmpose_backend, device
        )
        if profiler:
            profiler.end("pose_model_initialization")
    
    inference_start_time = time.time()

    det_img, vio_img, pose_img = run_pipeline(
        img_path=args.img_path,
        detector=detector,
        pose_estimator=pose_estimator,
        class_names=class_names,
        kp_thresh=args.kp_thresh,
        debug=args.debug,  # 🎯 Pass debug flag
        profiler=profiler
    )

    # Save outputs
    save_outputs(det_img, vio_img, pose_img, debug=args.debug, profiler=profiler)
    
    inference_time = time.time() - inference_start_time
    total_time = time.time() - total_start_time
    print(f"📊 Total inference time: {inference_time:.3f} seconds")
    print(f"📊 Total execution time: {total_time:.3f} seconds")
    
    if profiler:
        profiler.print_summary()

if __name__ == "__main__":
    main()