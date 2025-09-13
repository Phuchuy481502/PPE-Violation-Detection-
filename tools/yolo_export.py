import torch
from ultralytics import YOLO
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert YOLO model to ONNX format")
    parser.add_argument("--format", type=str, default="torchscript", help="Export format (torchscript/onnx)")

    parser.add_argument("--weights", type=str, default="yolo11n.pt", help="Path to the YOLO weights file")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for export (cpu/cuda)")
    parser.add_argument("--half", action="store_true", help="Export in half precision (FP16) - requires GPU")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument("--simplify", action="store_true", help="Simplify model")
    
    args = parser.parse_args()
    
    # Check if weights file exists
    if not os.path.exists(args.weights):
        print(f"❌ Weights file not found: {args.weights}")
        return
    
    print(f"🚀 Converting YOLO model to {args.format}:")
    print(f"  📁 Input:  {args.weights}")
    print(f"  🖥️  Device: {args.device}")
    print(f"  📊 Image size: {args.imgsz}")
    print(f"  📊 Batch size: {args.batch}")
    
    try:
        # Load the model
        print("📦 Loading YOLO model...")
        model = YOLO(args.weights)
        
        # Determine if we can use half precision
        use_half = args.half and (args.device != "cpu" and torch.cuda.is_available())
        if args.half and not use_half:
            print("⚠️  Half precision requested but not available (requires GPU). Using FP32.")
        
        # Export the model
        print(f"🔄 Exporting to {args.format}...")
        success = model.export(
            format=args.format,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            half=use_half,
            simplify=args.simplify,
            verbose=True
        )
        
        if success:
            # Generate output filename
            output_path = args.weights.replace('.pt', f'.{args.format}')
            print(f"✅ Export successful!")
            print(f"  📁 Exported model saved to: {output_path}")
            
            # Check file sizes
            if os.path.exists(output_path):
                pt_size = os.path.getsize(args.weights) / (1024 * 1024)
                onnx_size = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  📊 PyTorch model: {pt_size:.1f} MB")
                print(f"  📊 {args.format} model:    {onnx_size:.1f} MB")
        else:
            print("❌ Export failed!")
            
    except Exception as e:
        print(f"❌ Error during export: {e}")

if __name__ == "__main__":
    main()