import cv2
from config.color import ColorPalette

def get_contrast_color(bg_color):
    """
    Returns:
        tuple: RGB color for text (either black or white)
    """
    # Calculate luminance
    r, g, b = bg_color
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    return ColorPalette.TEXT_WHITE if luminance < 0.5 else ColorPalette.TEXT_BLACK


# def non_max_suppression(boxes, scores, iou_threshold):
#     # Input validation
#     if len(boxes) == 0 or len(scores) == 0:
#         return []
    
#     # Convert to tensor if not already
#     if not isinstance(boxes, torch.Tensor):
#         boxes = torch.tensor(boxes)
#     if not isinstance(scores, torch.Tensor):
#         scores = torch.tensor(scores)
    
#     # Ensure boxes and scores have same first dimension
#     if boxes.shape[0] != scores.shape[0]:
#         raise ValueError(f"boxes and scores must have same length, got {boxes.shape[0]} and {scores.shape[0]}")

#     # Handle single box case
#     if boxes.shape[0] == 1:
#         return [0]

#     # Get coordinates
#     x1 = boxes[:, 0] # x_min of all boxes -> (N,)
#     y1 = boxes[:, 1]
#     x2 = boxes[:, 2]
#     y2 = boxes[:, 3]

#     # Compute areas
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)

#     # Sort by score
#     _, order = scores.sort(0, descending=True)
#     order = order.reshape(-1)  # Ensure order is 1D

#     keep = []
#     while order.numel() > 0:
#         if order.numel() == 1:
#             keep.append(order[0].item())
#             break
        
#         #! box with highest score
#         i = order[0].item()
#         keep.append(i)

#         # Compute IoU
#         xx1 = torch.max(x1[i], x1[order[1:]])
#         yy1 = torch.max(y1[i], y1[order[1:]])
#         xx2 = torch.min(x2[i], x2[order[1:]])
#         yy2 = torch.min(y2[i], y2[order[1:]])

#         w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
#         h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)

#         inter = w * h
#         iou = inter / (areas[i] + areas[order[1:]] - inter)

#         # Keep boxes with IoU less than threshold
#         mask = iou <= iou_threshold
#         if not mask.any():
#             break
            
#         inds = mask.nonzero().reshape(-1)
#         order = order[inds + 1]

#     return keep

def remove_invalid_boxes(targets):
    valid_targets = []
    for target in targets:
        boxes = target['boxes']
        labels = target['labels']
        valid_indices = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])  # x1 < x2 and y1 < y2
        valid_boxes = boxes[valid_indices]
        valid_labels = labels[valid_indices]
        valid_targets.append({'boxes': valid_boxes, 'labels': valid_labels})
    return valid_targets

def draw_fps(cap, frame, fps):
    fps_text = f' FPS: {fps:.3f}' + ' Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
    cv2.putText(frame, fps_text, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ColorPalette.SAFETY_GREEN, 2)
    return frame

def draw_bbox(frame, id, x1, y1, x2, y2, conf, missing=None, type='detect', class_names=None, color_scheme='default'):
    """
    
    Args:
        frame: Image frame
        id: Class ID or track ID
        x1, y1, x2, y2: Bounding box coordinates
        conf: Confidence score
        missing: Missing PPE items (for violation type)
        type: Type of bbox ('track', 'violate', 'detect')
        class_names: List of class names
        color_scheme: Color scheme to use
    """
    if type == "track":
        color = ColorPalette.TRACKING_ACTIVE
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f'ID: {id}, Score: {conf:.2f}'
        
        # Add semi-transparent background for text
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1-25), (x1 + text_size[0], y1), color, -1)
        cv2.putText(frame, text, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ColorPalette.TEXT_BLACK, 2)
        
    elif type == "violate":
        # Use semantic colors for violations vs compliance
        color = ColorPalette.get_violation_color(bool(missing))

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        # cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        if missing:
            # Violation text with semantic colors
            text1 = f'ID {id}'
            print(missing)
            text2 = ", ".join([str(class_names[m]) if class_names and m < len(class_names) else str(m) for m in missing])
            
            # Background for text using violation background color
            text1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            text2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            cv2.rectangle(frame, (x1, y1-35), (x1 + max(text1_size[0], text2_size[0]) + 5, y1), ColorPalette.BG_VIOLATION, -1)
            cv2.putText(frame, text1, (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ColorPalette.TEXT_YELLOW, 2)
            cv2.putText(frame, text2, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ColorPalette.TEXT_WHITE, 2)
        
    elif type == "detect":
        # Enhanced detection boxes with semantic colors
        bbox_color = ColorPalette.get_detection_color(id, color_scheme)
        text_color = get_contrast_color(bbox_color)

        cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)

        class_name = class_names[id] if class_names and 0 <= id < len(class_names) else f'Class_{id}'
        text = f"{class_name}: {conf:.2f}"
        
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        
        # Draw text background with padding
        padding = 4
        bg_x1 = max(0, x1)
        bg_y1 = max(text_h + padding, y1 - text_h - padding)
        bg_x2 = min(frame.shape[1], x1 + text_w + 2*padding)
        bg_y2 = bg_y1 + text_h + padding
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bbox_color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Draw text
        cv2.putText(frame, text, (x1 + padding, y1 - padding), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        