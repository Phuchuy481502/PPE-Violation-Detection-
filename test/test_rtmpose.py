import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
try:
    from rtmlib import RTMPose, draw_skeleton
    
    # Load a real image (same as your pipeline)
    test_img = cv2.imread("sample/images/17.jpg")
    
    test_bboxes = [
        [170, 427, 473, 900],
        [433, 167, 650, 707],
        [580, 470, 820, 890],
        [673, 402, 830, 700],
        [805, 290, 965, 962], 
        [935, 355, 1175, 805],
        [1095, 675, 1398, 1032],
    ]
    
    print(f"Testing RTMPose with {len(test_bboxes)} persons...")
    
    rtmpose = RTMPose(
        onnx_model="weights/rtmpose-t-17.onnx",
        model_input_size=(192, 256),
        backend='onnxruntime',
        device='cuda'
    )
    
    # Warm up
    for i in range(3):
        _, _ = rtmpose(test_img, test_bboxes)
    
    # Test batch processing
    start_time = time.time()
    keypoints, scores = rtmpose(test_img, test_bboxes)
    end_time = time.time()
    
    img_pose = draw_skeleton(test_img, keypoints, scores, kpt_thr=0.5)
    plt.imshow(img_pose)
    plt.savefig("test/output.jpg")
    print(f"Batch processing time: {(end_time - start_time) * 1000:.2f}ms")
    
    # Test individual processing
    total_individual_time = 0
    for i, bbox in enumerate(test_bboxes):
        start_time = time.time()
        kp, sc = rtmpose(test_img, [bbox])
        end_time = time.time()
        individual_time = end_time - start_time
        total_individual_time += individual_time
        print(f"Person {i+1} individual time: {individual_time * 1000:.2f}ms")
    
    print(f"Total individual processing time: {total_individual_time * 1000:.2f}ms")

    
except Exception as e:
    print(f"Test failed: {e}")