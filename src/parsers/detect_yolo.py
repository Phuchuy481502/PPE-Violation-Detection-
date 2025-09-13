import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Modular detection + pose + violation detection")
    parser.add_argument("--det_weights", type=str, required=True, help="Detection weights (YOLO)")
    parser.add_argument("--pose_backend", type=str, default="none", 
                        choices=["none", "yolo", "yolo_crop", "mmpose", "rtmpose"],
                        help="Pose backend")
    parser.add_argument("--pose_weights", type=str, default="", help="Pose weights")
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
    parser.add_argument("--debug", action="store_true", help="Enable save detect + pose output")
    return parser

def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args