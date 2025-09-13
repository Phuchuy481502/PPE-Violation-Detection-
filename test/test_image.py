import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import time
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

try:
    from rtmlib import RTMPose
    
    print("🧪 TESTING IMAGE FORMAT HYPOTHESIS")
    print("="*50)
    
    # Load image EXACTLY like your pipeline
    img_path = "sample/images/17.jpg"
    bgr = cv2.imread(img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    # Pipeline processing
    pil = Image.fromarray(rgb)
    tfm = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor()
    ])
    tensor = tfm(pil).unsqueeze(0)
    resized_rgb_pipeline = cv2.resize(rgb, (640, 640))  # This is what your pipeline uses
    
    # Direct test processing
    test_img_direct = cv2.imread(img_path)  # This is what your direct test uses
    
    print(f"📊 Pipeline image - shape: {resized_rgb_pipeline.shape}, dtype: {resized_rgb_pipeline.dtype}")
    print(f"📊 Direct test image - shape: {test_img_direct.shape}, dtype: {test_img_direct.dtype}")
    
    # Same bboxes
    test_bboxes = [
        [171, 82, 259, 382],
        [307, 152, 390, 480],
        [64, 216, 188, 468],
        [429, 348, 559, 528],
        [392, 188, 469, 413],
        [228, 241, 297, 407]
    ]
    
    print(f"👥 Testing with {len(test_bboxes)} persons")
    
    rtmpose = RTMPose(
        onnx_model="weights/rtmpose-t-17.onnx",
        model_input_size=(192, 256),
        backend='onnxruntime',
        device='cuda'
    )
    # Warm up
    for i in range(3):
        _, _ = rtmpose(rgb, test_bboxes)
        
    # Test 1: Pipeline format (RGB, resized)
    print("\n1️⃣  Testing PIPELINE format (RGB, resized)...")
    start_time = time.time()
    keypoints1, scores1 = rtmpose(resized_rgb_pipeline, test_bboxes)
    time1 = time.time() - start_time
    print(f"  ⏱️  Time: {time1*1000:.1f}ms")
    
    # Test 2: Direct format (BGR, original size, scaled bboxes)
    print("\n2️⃣  Testing DIRECT format (BGR, original)...")
    
    # Scale bboxes to original image size
    orig_h, orig_w = test_img_direct.shape[:2]
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    
    scaled_bboxes = []
    for bbox in test_bboxes:
        x1, y1, x2, y2 = bbox
        scaled_bboxes.append([
            int(x1 * scale_x), int(y1 * scale_y),
            int(x2 * scale_x), int(y2 * scale_y)
        ])
    
    start_time = time.time()
    keypoints2, scores2 = rtmpose(test_img_direct, scaled_bboxes)
    time2 = time.time() - start_time
    print(f"  ⏱️  Time: {time2*1000:.1f}ms")
    
    # Test 3: Convert pipeline image to BGR
    print("\n3️⃣  Testing PIPELINE image converted to BGR...")
    resized_bgr_pipeline = cv2.cvtColor(resized_rgb_pipeline, cv2.COLOR_RGB2BGR)
    
    start_time = time.time()
    keypoints3, scores3 = rtmpose(resized_bgr_pipeline, test_bboxes)
    time3 = time.time() - start_time
    print(f"  ⏱️  Time: {time3*1000:.1f}ms")
    
    print("\n" + "="*50)
    print("📊 RESULTS:")
    print(f"Pipeline RGB resized:     {time1*1000:>8.1f}ms ⚠️")
    print(f"Direct BGR original:      {time2*1000:>8.1f}ms ✅")
    print(f"Pipeline BGR resized:     {time3*1000:>8.1f}ms")
    print("="*50)
    
    if time1 > time2 * 2:
        print("🚨 FOUND IT! Pipeline RGB format is the issue!")
        print("   RTMPose prefers BGR format or original image size")
    elif time3 < time1 * 0.8:
        print("🚨 FOUND IT! RGB->BGR conversion fixes the issue!")
        print("   RTMPose works better with BGR format")
    else:
        print("🤔 Image format is not the issue...")
        
except Exception as e:
    print(f"Test failed: {e}")
    import traceback
    traceback.print_exc()