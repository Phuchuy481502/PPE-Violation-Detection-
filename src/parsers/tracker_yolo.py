import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Parser for Tracker with Pose Estimation")
    
    # Basic tracking arguments
    parser.add_argument("--vid_dir", type=str, default="sample/videos/1.mp4", help="Path to video directory")
    parser.add_argument("--yaml_class", type=str, default="data/data-ppe_v4-kaggle.yaml", help="Path to yaml file")
    parser.add_argument("--weights", type=str, default="weights/best_yolo.pt", help="Path to weights file")
    parser.add_argument("--detect_thresh", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    
    # Pose estimation arguments
    parser.add_argument("--pose_backend", type=str, default="none", 
                        choices=["none", "yolo", "yolo_crop", "mmpose", "rtmpose"],
                        help="Pose backend")
    parser.add_argument("--pose_weights", type=str, default="", 
                        help="Pose weights (ONNX for RTMPose)")
    parser.add_argument("--pose_config", type=str, default="", 
                        help="MMPose config")
    parser.add_argument("--pose_input_size", type=str, default="192,256", 
                        help="RTMPose input size as 'width,height'")
    parser.add_argument("--rtmpose_backend", type=str, default="onnxruntime",
                        choices=["opencv", "onnxruntime", "openvino"],
                        help="RTMPose backend")
    parser.add_argument("--enable_pose", action="store_true", 
                        help="Enable pose estimation")
    parser.add_argument("--kp_thresh", type=float, default=0.25, 
                        help="Keypoint confidence threshold")
    
    # Performance optimization arguments
    parser.add_argument("--violate_stride", type=int, default=2,
                        help="Process violation detection every N frames (default: 2)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode - save all output videos")
    
    # Performance and debugging arguments
    parser.add_argument("--profile", action="store_true", 
                        help="Enable detailed timing profiling")
    parser.add_argument("--system_info", action="store_true", 
                        help="Show system information")
    
    # Real-time display arguments
    parser.add_argument("--show_realtime", action="store_true", 
                        help="Show real-time video streaming (WSL compatible)")
    parser.add_argument("--display_scale", type=float, default=0.8,
                        help="Display scale factor for streaming window (default: 0.8)")
   
    return parser

def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args