import numpy as np
from .metrics import associate_score

class STrack(object):
    def __init__(self, tlwh, score, track_id):
        self.tlwh = tlwh
        self.score = score
        self.track_id = track_id
        self.missing = []  # List to store missing equipment names

def scale_bbox(bbox, scale_factor=1.2):
    """Scale bounding box by factor Z"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = (x2 - x1) * scale_factor
    h = (y2 - y1) * scale_factor
    
    new_x1 = cx - w / 2
    new_y1 = cy - h / 2
    new_x2 = cx + w / 2
    new_y2 = cy + h / 2
    
    return [new_x1, new_y1, new_x2, new_y2]

def keypoint_in_bbox(keypoint, bbox):
    """Check if keypoint is inside bounding box"""
    x, y, conf = keypoint
    if conf <= 0 or x <= 0 or y <= 0:  # Invalid keypoint
        return False
    
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def check_hardhat_pose(person_keypoints, hardhat_bbox, z_factor=1.2, min_keypoints=2):
    """Check if head keypoints are in scaled hardhat bbox"""
    # Head keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
    head_indices = [0, 1, 2, 3, 4]
    
    scaled_bbox = scale_bbox(hardhat_bbox, z_factor)
    keypoints_in_bbox = 0
    valid_keypoints = 0
    
    for idx in head_indices:
        if idx < len(person_keypoints):
            keypoint = person_keypoints[idx]
            if keypoint[2] > 0.3:  # Valid keypoint
                valid_keypoints += 1
                if keypoint_in_bbox(keypoint, scaled_bbox):
                    keypoints_in_bbox += 1
    
    # Need at least min_keypoints valid keypoints in bbox
    return keypoints_in_bbox >= min_keypoints and valid_keypoints >= min_keypoints

def check_vest_pose(person_keypoints, vest_bbox, z_factor=1.2, min_keypoints=2):
    """Check if torso keypoints are in scaled vest bbox"""
    # Torso keypoints: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
    torso_indices = [5, 6, 11, 12]
    
    scaled_bbox = scale_bbox(vest_bbox, z_factor)
    keypoints_in_bbox = 0
    valid_keypoints = 0
    
    for idx in torso_indices:
        if idx < len(person_keypoints):
            keypoint = person_keypoints[idx]
            if keypoint[2] > 0.3:  # Valid keypoint
                valid_keypoints += 1
                if keypoint_in_bbox(keypoint, scaled_bbox):
                    keypoints_in_bbox += 1
    
    return keypoints_in_bbox >= min_keypoints and valid_keypoints >= min_keypoints

def check_gloves_pose(person_keypoints, gloves_bbox, z_factor=1.2, min_keypoints=1):
    """Check if wrist keypoints are in scaled gloves bbox"""
    # Wrist keypoints: 9=left_wrist, 10=right_wrist
    wrist_indices = [9, 10]
    
    scaled_bbox = scale_bbox(gloves_bbox, z_factor)
    keypoints_in_bbox = 0
    valid_keypoints = 0
    
    for idx in wrist_indices:
        if idx < len(person_keypoints):
            keypoint = person_keypoints[idx]
            if keypoint[2] > 0.3:  # Valid keypoint
                valid_keypoints += 1
                if keypoint_in_bbox(keypoint, scaled_bbox):
                    keypoints_in_bbox += 1
    
    # For gloves, just need 1 wrist keypoint in bbox
    return keypoints_in_bbox >= min_keypoints and valid_keypoints >= min_keypoints

def detect_css_violations(online_targets, obj_detections, pose_results=None, 
                                use_pose=True, z_factor=1.2, class_names=None):
    """
    Simple PPE violation detection using scaled bounding box + keypoint check
    
    Args:
        online_targets: List of person tracks
        obj_detections: List of detected objects [x1,y1,x2,y2,score,class_id]
        pose_results: List of pose estimation results with keypoints
        use_pose: Whether to use pose-based detection
        z_factor: Scaling factor for PPE bounding boxes (default 1.2)
        class_names: List of class names for display
    """
    # Map class IDs to equipment names and check functions
    EQUIPMENT_MAP = {
        0: ("Hardhat", check_hardhat_pose),
        1: ("Safety Vest", check_vest_pose), 
        2: ("Gloves", check_gloves_pose)
    }
    
    if not class_names:
        class_names = ["Hardhat", "Safety Vest", "Gloves", "Person"]
    
    # Create pose keypoints mapping
    pose_keypoints_map = {}
    if pose_results and use_pose:
        for i, pose_result in enumerate(pose_results):
            if i < len(online_targets):
                track_id = online_targets[i].track_id
                pose_keypoints_map[track_id] = pose_result.get('kpts', [])
    
    results = []
    
    for t in online_targets:
        new_track = STrack(t.tlwh, t.score, t.track_id)
        
        # Get pose keypoints for this person
        person_keypoints = pose_keypoints_map.get(t.track_id, [])
        has_pose = len(person_keypoints) > 0 and use_pose
        
        # Track which equipment is detected/worn
        equipment_worn = set()  # Set of equipment class IDs that are worn
        
        # Check each detected PPE object
        for obj in obj_detections:
            obj_bbox = obj[:4]  # x1,y1,x2,y2
            obj_class = obj[5]  # class_id
            
            # Only check known PPE equipment
            if obj_class not in EQUIPMENT_MAP:
                continue
            
            equipment_name, check_function = EQUIPMENT_MAP[obj_class]
            
            is_worn = False
            
            if has_pose:
                # Use pose-based check with scaled bbox
                is_worn = check_function(person_keypoints, obj_bbox, z_factor)
            else:
                # Fallback to simple bbox overlap
                x1, y1 = t.tlwh[0], t.tlwh[1]
                x2, y2 = x1 + t.tlwh[2], y1 + t.tlwh[3]
                person_bbox = [x1, y1, x2, y2]
                overlap = associate_score(person_bbox, obj_bbox)
                is_worn = overlap > 0.5
            
            if is_worn:
                equipment_worn.add(obj_class)
        
        # Determine missing equipment
        new_track.missing = []
        for eq_class in EQUIPMENT_MAP.keys():
            if eq_class not in equipment_worn:
                equipment_name = EQUIPMENT_MAP[eq_class][0]
                new_track.missing.append(equipment_name)
        
        results.append(new_track)
    
    return results
