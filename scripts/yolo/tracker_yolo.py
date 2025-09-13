import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))

import cv2
import time
import numpy as np
from typing import List, Dict, Optional
import queue
from collections import deque

from parsers.tracker_yolo import parse_args
from utils.yaml_helper import read_yaml
from utils.function import draw_bbox, draw_fps
from utils.profiler import TimingProfiler
from trackers.byte_tracker import BYTETracker
from pipeline.yolo import (
    # Preprocessing
    preprocess_frame,
    build_detector,
    build_pose_estimator,
    
    # Granular pipeline functions
    run_object_detection,
    split_detections,
    run_pose_estimation,
    run_violation_detection,
    
    # Visualization 
    draw_detections,
    draw_poses,
    draw_violations  
)

# Constants
DEFAULT_IMG_SIZE = 640
PERSON_NAME = "Person"

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

# =============================================================================
# TRACKING FUNCTIONS
# =============================================================================

def prepare_detections_for_tracking(person_dets: List[Dict]):
    """Convert person detections to tracking format"""
    per_detections = []
    for det in person_dets:
        x1, y1, x2, y2 = det["bbox"]
        score = det["score"]
        per_detections.append([x1, y1, x2, y2, score])
    
    if per_detections:
        per_detections = np.array(per_detections)
    else:
        per_detections = np.empty((0, 5))
    
    return per_detections

def run_tracking(tracker, per_detections, frame_height, frame_width):
    """Run tracking on person detections"""
    online_targets = tracker.update(per_detections, [frame_height, frame_width], [frame_height, frame_width])
    return online_targets

def convert_tracks_to_person_dets(online_targets: List) -> List[Dict]:
    """Convert tracking results back to person detection format for pose estimation"""
    person_dets = []
    
    for track in online_targets:
        x, y, w, h = track.tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        person_det = {
            "bbox": [x1, y1, x2, y2],
            "score": track.score,
            "cls": 0,  # Person class ID
            "track_id": track.track_id
        }
        person_dets.append(person_det)
    
    return person_dets

def process_frame_pipeline(
    frame: np.ndarray,
    detector,
    tracker: BYTETracker,
    pose_estimator: Optional,
    class_names: List[str],
    kp_thresh: float = 0.25,
    run_violation: bool = True, 
    cached_violations: Dict = None,  
    debug: bool = False, 
    profiler=None
):
    """Video pipeline: Detection → Tracking → Pose (from tracker bbox) → Violation"""
    device = detector.device
    frame_height, frame_width = frame.shape[:2]
    
    # Step 1: Preprocess frame
    if profiler:
        profiler.start("frame_preprocessing")
    _, resized_rgb, image_tensor = preprocess_frame(frame, device=device)
    if profiler:
        profiler.end("frame_preprocessing")

    # Step 2: Object detection
    if profiler:
        profiler.start("object_detection")
    all_detections = run_object_detection(detector, image_tensor)
    if profiler:
        profiler.end("object_detection")

    # Step 3: Split detections
    if profiler:
        profiler.start("detection_filtering")
    person_dets, obj_dets = split_detections(all_detections, detector, class_names, PERSON_NAME)
    if profiler:
        profiler.end("detection_filtering")

    # Step 4: Prepare for tracking
    if profiler:
        profiler.start("tracking_preparation")
    per_detections = prepare_detections_for_tracking(person_dets)
    if profiler:
        profiler.end("tracking_preparation")

    # Step 5: Run tracking
    if profiler:
        profiler.start("tracking")
    online_targets = run_tracking(tracker, per_detections, frame_height, frame_width)
    if profiler:
        profiler.end("tracking")

    # Step 6: Convert tracker results back to person detections
    if profiler:
        profiler.start("track_to_detection_conversion")
    tracked_person_dets = convert_tracks_to_person_dets(online_targets)
    if profiler:
        profiler.end("track_to_detection_conversion")

    # Step 7: Pose estimation using TRACKED bounding boxes
    pose_results = []
    if run_violation:  # Only run pose when doing violation detection
        if profiler:
            profiler.start("pose_estimation")
        pose_results = run_pose_estimation(pose_estimator, image_tensor, resized_rgb, tracked_person_dets)
        
        # Link pose results back to track IDs
        for i, pose_result in enumerate(pose_results):
            if i < len(tracked_person_dets):
                pose_result["track_id"] = tracked_person_dets[i]["track_id"]
        if profiler:
            profiler.end("pose_estimation")

    # Step 8: Violation detection using tracking targets
    if run_violation:
        if profiler:
            profiler.start("violation_detection")
        
        # Use tracked person detections (with track IDs) for violations
        violations_targets = run_violation_detection(
            tracked_person_dets, obj_dets, pose_results, pose_estimator, class_names
        )
        
        # Copy violation results back to tracking targets and cache them
        for track in online_targets:
            track.missing = []
            for viol_target in violations_targets:
                if hasattr(viol_target, 'track_id') and viol_target.track_id == track.track_id:
                    track.missing = getattr(viol_target, 'missing', [])
                    # 🎯 Cache violation results by track ID
                    if cached_violations is not None:
                        cached_violations[track.track_id] = track.missing
                    break
        
        if profiler:
            profiler.end("violation_detection")
    else:
        # 🎯 Reuse cached violation results
        for track in online_targets:
            track.missing = cached_violations.get(track.track_id, []) if cached_violations else []

    # Step 9: Generate output frames based on debug mode
    if profiler:
        profiler.start("image_generation")

    frame_detected = None
    frame_tracked = None
    pose_frame = None
    
    # Always generate violation frame
    frame_violated = resized_rgb.copy()
    if profiler:
        profiler.start("drawing_violations")
    for t in online_targets:
        x, y, w, h = t.tlwh
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        draw_violations(frame_violated, t, x1, y1, x2, y2, class_names)
    if profiler:
        profiler.end("drawing_violations")

    # Generate debug frames only if debug mode is enabled
    if debug:
        # Detection frame (all original detections)
        frame_detected = resized_rgb.copy()
        draw_detections(frame_detected, all_detections, class_names, tag_type='detect')

        # Tracking frame (using tracker bboxes)
        frame_tracked = resized_rgb.copy()
        if profiler:
            profiler.start("drawing_tracking")
        for t in online_targets:
            x, y, w, h = t.tlwh
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            
            # Draw tracking with track ID
            draw_bbox(frame_tracked, t.track_id, x1, y1, x2, y2, t.score, 
                     missing=getattr(t, 'missing', []), type='track', class_names=class_names)
        if profiler:
            profiler.end("drawing_tracking")

        # Pose frame (using tracker bboxes for consistency)
        if pose_results:
            pose_frame = resized_rgb.copy()
            if profiler:
                profiler.start("drawing_pose_boxes")
            
            # Draw person bounding boxes using TRACKER coordinates
            for track in online_targets:
                x, y, w, h = track.tlwh
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                cv2.rectangle(pose_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add track ID text
                cv2.putText(pose_frame, f"ID:{track.track_id}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            if profiler:
                profiler.end("drawing_pose_boxes")
            
            # Draw pose keypoints and skeleton
            if profiler:
                profiler.start("drawing_poses")
            draw_poses(pose_frame, pose_results, kp_thresh=kp_thresh)
            if profiler:
                profiler.end("drawing_poses")

    if profiler:
        profiler.end("image_generation")

    return frame_detected, frame_tracked, frame_violated, pose_frame

def tracking(
    weights="weights/best_yolo.pt", 
    video_path=None, 
    class_names=None, 
    detect_thresh=0.3, 
    device="cpu",
    pose_backend="none",
    pose_weights="",
    pose_input_size="192,256",
    rtmpose_backend="onnxruntime",
    enable_pose=False,
    kp_thresh=0.25,
    violate_stride=2,  
    debug=False,
    show_realtime=False,  # 🎯 ADD: Real-time display flag
    display_scale=0.8,    # 🎯 ADD: Display scaling
    profile=False,  
):
    """Main tracking function with real-time streaming display"""
    
    # Initialize profiler
    profiler = TimingProfiler() if profile else None
    total_start_time = time.time()

    # Setup tracker params
    TRACK_THRESH = 0.5
    TRACK_BUFFER = 30
    MATCH_THRESH = 0.85
    FUSE_SCORE = False

    print(f"🚀 Starting video tracking pipeline...")
    print(f"📊 Pipeline: Detection → Tracking → Pose (tracker bbox) → Violation")
    print(f"Using device: {device}")
    print(f"Processing video: {video_path}")
    print(f"Detection threshold: {detect_thresh}")
    print(f"🚀 Violation stride: {violate_stride} (process every {violate_stride} frames)")
    if show_realtime:
        print(f"📺 Real-time display: ON (Scale: {display_scale:.1f}x)")
    if debug:
        print(f"🔍 Debug mode: ON - Will save all output videos")
    else:
        print(f"🚀 Production mode: ON - Will only save violation video")
    if enable_pose:
        print(f"Pose backend: {pose_backend}")
        print(f"🎯 Pose will use TRACKER bounding boxes (more stable)")

    # Initialize detector
    if profiler:
        profiler.start("detector_initialization")
    detector = build_detector(weights, device=device, conf=detect_thresh)
    if profiler:
        profiler.end("detector_initialization")

    # Initialize tracker
    if profiler:
        profiler.start("tracker_initialization")
    tracker_args = TrackerArgs(TRACK_THRESH, TRACK_BUFFER, MATCH_THRESH, FUSE_SCORE)
    tracker = BYTETracker(tracker_args)
    if profiler:
        profiler.end("tracker_initialization")

    # Initialize pose estimator
    pose_estimator = None
    if enable_pose:
        if profiler:
            profiler.start("pose_model_initialization")
        pose_estimator = build_pose_estimator(
            pose_backend, pose_weights, pose_input_size, rtmpose_backend, device
        )
        if profiler:
            profiler.end("pose_model_initialization")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video info: {width}x{height}, {fps} FPS, {total_frames} frames")

    # Create output directory
    OUTPUT_PATH = "output/videos/"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Create unique output paths
    num = 1
    while os.path.exists(os.path.join(OUTPUT_PATH, f"out_violate_{num}_yolo.mp4")):
        num += 1

    # Always create violation output
    violate_path = os.path.join(OUTPUT_PATH, f"out_violate_{num}_yolo.mp4")
    
    # Create debug outputs only if debug mode is enabled
    detect_path = None
    track_path = None
    pose_path = None
    
    if debug:
        detect_path = os.path.join(OUTPUT_PATH, f"out_detect_{num}_yolo.mp4")
        track_path = os.path.join(OUTPUT_PATH, f"out_track_{num}_yolo.mp4")
        if enable_pose and pose_estimator is not None:
            pose_path = os.path.join(OUTPUT_PATH, f"out_pose_{num}_yolo.mp4")

    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_violate = cv2.VideoWriter(violate_path, fourcc, fps, (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE))
    
    out_detect = None
    out_track = None
    out_pose = None
    
    if debug:
        if detect_path:
            out_detect = cv2.VideoWriter(detect_path, fourcc, fps, (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE))
        if track_path:
            out_track = cv2.VideoWriter(track_path, fourcc, fps, (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE))
        if pose_path:
            out_pose = cv2.VideoWriter(pose_path, fourcc, fps, (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE))

    display_size = None
    if show_realtime:
        # Calculate display size
        display_width = int(DEFAULT_IMG_SIZE * display_scale)
        display_height = int(DEFAULT_IMG_SIZE * display_scale)
        display_size = (display_width, display_height)
        
        # Initialize OpenCV window
        window_name = "🚨 PPE Violation Detection - Real-time Stream"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        
        print(f"📺 Real-time display initialized: {display_width}x{display_height}")
        print(f"💡 Controls:")
        print(f"   - Press 'q' to quit")
        print(f"   - Press 's' to save current frame")
        print(f"   - Press SPACE to pause/resume")

    # Initialize processing state
    cached_violations = {}  
    frame_count = 0
    processing_times = []
    paused = False
    
    # FPS smoothing for display
    fps_history = deque(maxlen=10)    

    try:
        print(f"🎬 Starting video processing...")
        start_inference_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_start_time = time.time()
            
            # 🎯 Handle real-time display controls
            if show_realtime:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n⏹️  User requested quit")
                    break
                elif key == ord('s'):
                    # Save current frame
                    save_path = f"output/saved_frame_{frame_count:06d}.jpg"
                    os.makedirs("output", exist_ok=True)
                    print(f"💾 Saved frame: {save_path}")
                elif key == ord(' '):
                    paused = not paused
                    if paused:
                        print("⏸️  Paused - Press SPACE to resume")
                    else:
                        print("▶️  Resumed")
                
                # Handle pause
                while paused and show_realtime:
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord(' '):
                        paused = False
                        print("▶️  Resumed")
                    elif key == ord('q'):
                        print("\n⏹️  User requested quit")
                        break
                
                if key == ord('q'):
                    break
            
            # Determine if we should run violation detection this frame
            run_violation_now = (frame_count % violate_stride == 0)
            
            # Process frame through modular pipeline with stride optimization
            frame_detected, frame_tracked, frame_violated, pose_frame = process_frame_pipeline(
                frame, detector, tracker, pose_estimator, class_names, kp_thresh, 
                run_violation=run_violation_now,
                cached_violations=cached_violations,
                debug=debug,
                profiler=profiler
            )

            # Calculate FPS
            frame_process_time = time.time() - frame_start_time
            processing_times.append(frame_process_time)
            current_fps = 1 / frame_process_time if frame_process_time > 0 else 0
            fps_history.append(current_fps)
            smoothed_fps = sum(fps_history) / len(fps_history)
            
            # 🎯 Real-time display
            if show_realtime:
                display_frame = cv2.resize(frame_violated, display_size)
                display_frame_bgr = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                
                overlay_y = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                
                # Background for text
                overlay_bg = display_frame_bgr.copy()
                cv2.rectangle(overlay_bg, (10, 10), (400, 120), (0, 0, 0), -1)
                cv2.addWeighted(display_frame_bgr, 0.7, overlay_bg, 0.3, 0, display_frame_bgr)
                
                # FPS info
                cv2.putText(display_frame_bgr, f"FPS: {smoothed_fps:.1f}", 
                           (15, overlay_y), font, font_scale, (0, 255, 0), thickness)
            
                # Show the frame
                cv2.imshow(window_name, display_frame_bgr)
            
            # Draw FPS on saved frames
            draw_fps(cap, frame_violated, smoothed_fps)
            
            if debug:
                if frame_detected is not None:
                    draw_fps(cap, frame_detected, smoothed_fps)
                if frame_tracked is not None:
                    draw_fps(cap, frame_tracked, smoothed_fps)
                if pose_frame is not None:
                    draw_fps(cap, pose_frame, smoothed_fps)

            # Convert to BGR and write frames to files
            frame_violated_bgr = cv2.cvtColor(frame_violated, cv2.COLOR_RGB2BGR)
            out_violate.write(frame_violated_bgr)
            
            if debug:
                if out_detect is not None and frame_detected is not None:
                    frame_detected_bgr = cv2.cvtColor(frame_detected, cv2.COLOR_RGB2BGR)
                    out_detect.write(frame_detected_bgr)
                
                if out_track is not None and frame_tracked is not None:
                    frame_tracked_bgr = cv2.cvtColor(frame_tracked, cv2.COLOR_RGB2BGR)
                    out_track.write(frame_tracked_bgr)
                
                if out_pose is not None and pose_frame is not None:
                    pose_frame_bgr = cv2.cvtColor(pose_frame, cv2.COLOR_RGB2BGR)
                    out_pose.write(pose_frame_bgr)

            frame_count += 1
            
            # Print progress every 60 frames (less frequent when showing real-time)
            progress_interval = 60 if show_realtime else 30
            if frame_count % progress_interval == 0:
                progress = (frame_count / total_frames) * 100
                avg_fps_so_far = frame_count / sum(processing_times) if sum(processing_times) > 0 else 0
                violation_processed = len([i for i in range(frame_count) if i % violate_stride == 0])
                print(f"📊 Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
                print(f"   Current FPS: {smoothed_fps:.2f}, Avg FPS: {avg_fps_so_far:.2f}")
                print(f"   Violations processed: {violation_processed}/{frame_count} frames")

    except KeyboardInterrupt:
        print("\n⚠️  Processing interrupted by user")
        
    finally:
        # Clean up
        cap.release()
        out_violate.release()
        if out_detect is not None:
            out_detect.release()
        if out_track is not None:
            out_track.release()
        if out_pose is not None:
            out_pose.release()
            
        # Close display window
        if show_realtime:
            cv2.destroyAllWindows()

    # Calculate final statistics
    total_time = time.time() - total_start_time
    total_inference_time = time.time() - start_inference_time
    initialization_time = total_time - total_inference_time  

    pure_processing_time = sum(processing_times)
    avg_frame_processing_time = pure_processing_time / frame_count if frame_count > 0 else 0
    
    pure_processing_fps = frame_count / pure_processing_time if pure_processing_time > 0 else 0
    total_inference_fps = frame_count / total_inference_time if total_inference_time > 0 else 0
    
    violation_frames_processed = len([i for i in range(frame_count) if i % violate_stride == 0])
    violation_skip_ratio = (frame_count - violation_frames_processed) / frame_count if frame_count > 0 else 0

    print(f"\n🎉 Video processing completed successfully!")
    print(f"📊 Performance Summary:")
    print(f"  🔧 Initialization time: {initialization_time:.3f}s")
    print(f"  🚀 Pure processing time: {pure_processing_time:.3f}s")
    print(f"  📊 Total inference time: {total_inference_time:.3f}s (including I/O)")
    print(f"  ⏱️  Total pipeline time: {total_time:.3f}s")
    print(f"")
    print(f"  Processed frames: {frame_count}")
    print(f"  Average frame time: {avg_frame_processing_time*1000:.1f}ms")
    print(f"")
    print(f"🚀 Violation Optimization:")
    print(f"  Violation stride: {violate_stride}")
    print(f"  Violation frames processed: {violation_frames_processed}/{frame_count} ({(1-violation_skip_ratio)*100:.1f}%)")
    print(f"  Violation frames skipped: {violation_skip_ratio*100:.1f}%")
    print(f"")
    print(f"💾 Output videos saved:")
    print(f"  ⚠️  Violations: {violate_path}")
    if debug:
        if detect_path:
            print(f"  🎯 Detection: {detect_path}")
        if track_path:
            print(f"  🔄 Tracking: {track_path}")
        if pose_path:
            print(f"  🏃 Poses: {pose_path}")

    # 🎯 FIXED: Consistent FPS Analysis
    print(f"\n==================================================")
    print(f"📊 FPS ANALYSIS")
    print(f"==================================================")
    print(f"🎯 Pure Processing FPS:   {pure_processing_fps:.2f} fps")
    print(f"   (Core pipeline only)   ({pure_processing_time:.3f}s)")
    print(f"🚀 Total Inference FPS:   {total_inference_fps:.2f} fps") 
    print(f"   (Including I/O & viz)   ({total_inference_time:.3f}s)")
    print(f"📊 Initialization overhead: {initialization_time:.3f}s")
    print(f"==================================================")

    if profiler:
        print(f"\n📈 Detailed Performance Analysis:")
        profiler.print_summary()

    return detect_path, track_path, violate_path, pose_path

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Load class names
    yaml_class = read_yaml(args.yaml_class)
    CLASS_NAMES = yaml_class["names"]

    # Call tracking function
    tracking(
        weights=args.weights,
        video_path=args.vid_dir,
        class_names=CLASS_NAMES,
        detect_thresh=args.detect_thresh,
        device=args.device,
        pose_backend=args.pose_backend,
        pose_weights=args.pose_weights,
        pose_input_size=args.pose_input_size,
        rtmpose_backend=args.rtmpose_backend,
        enable_pose=args.enable_pose,
        kp_thresh=args.kp_thresh,
        violate_stride=args.violate_stride,  
        debug=args.debug,
        show_realtime=args.show_realtime,  
        display_scale=args.display_scale, 
        profile=args.profile
    )