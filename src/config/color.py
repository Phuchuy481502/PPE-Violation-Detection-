class ColorPalette:
    """Comprehensive color palette with semantic naming for object detection and PPE monitoring"""
    
    # ================= BASIC COLORS =================
    # Primary colors
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    
    # Secondary colors
    YELLOW = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    CYAN = (0, 255, 255)
    
    # ================= SAFETY & PPE COLORS =================
    # Safety colors following industry standards
    SAFETY_RED = (220, 20, 60)           # Danger, violations, critical alerts
    SAFETY_GREEN = (50, 205, 50)         # Safe, compliant, good status
    SAFETY_YELLOW = (255, 215, 0)        # Warning, caution, high visibility
    SAFETY_ORANGE = (255, 140, 0)        # High visibility equipment, warning
    
    # PPE-specific semantic colors
    PERSON_COLOR = (255, 100, 100)       # Light red for person detection
    HELMET_COLOR = (100, 255, 100)       # Light green - protection/safety
    VEST_COLOR = (255, 255, 100)         # Light yellow - high visibility
    GLOVES_COLOR = (100, 255, 255)       # Light cyan - hand protection
    BOOTS_COLOR = (255, 100, 255)        # Light magenta - foot protection
    GLASSES_COLOR = (255, 165, 0)        # Orange - eye protection
    MASK_COLOR = (128, 0, 128)           # Purple - respiratory protection
    
    # ================= STATUS COLORS =================
    # Compliance status
    COMPLIANT = (0, 255, 0)              # Bright green - fully compliant
    VIOLATION = (255, 0, 0)              # Bright red - safety violation
    WARNING = (255, 255, 0)              # Yellow - potential issue
    UNKNOWN = (128, 128, 128)            # Gray - unknown/uncertain status
    
    # Detection confidence levels
    HIGH_CONFIDENCE = (0, 255, 0)        # Green - high confidence detection
    MEDIUM_CONFIDENCE = (255, 255, 0)    # Yellow - medium confidence
    LOW_CONFIDENCE = (255, 165, 0)       # Orange - low confidence
    
    # ================= TRACKING COLORS =================
    # Object tracking
    TRACKING_ACTIVE = (0, 254, 255)      # Bright cyan - active tracking
    TRACKING_LOST = (255, 69, 0)         # Red-orange - lost track
    NEW_DETECTION = (50, 205, 50)        # Lime green - new object
    
    # ================= UI & DISPLAY COLORS =================
    # Text and backgrounds
    TEXT_BLACK = (0, 0, 0)               # Black text for light backgrounds
    TEXT_WHITE = (255, 255, 255)         # White text for dark backgrounds
    TEXT_YELLOW = (255, 255, 0)          # Yellow text for emphasis
    
    # Background colors
    BG_TRANSPARENT = (0, 0, 0, 0)        # Transparent background
    BG_SEMI_DARK = (0, 0, 0, 128)        # Semi-transparent dark
    BG_VIOLATION = (220, 20, 60)         # Dark red for violation backgrounds
    BG_SUCCESS = (0, 100, 0)             # Dark green for success backgrounds
    
    # ================= PROFESSIONAL PALETTE =================
    # Professional/corporate colors (more subtle)
    PROF_RED = (220, 50, 50)
    PROF_GREEN = (50, 180, 50)
    PROF_BLUE = (50, 50, 220)
    PROF_YELLOW = (200, 200, 50)
    PROF_ORANGE = (255, 140, 50)
    PROF_PURPLE = (180, 80, 180)
    PROF_CYAN = (50, 200, 200)
    PROF_GRAY = (128, 128, 128)
    
    # ================= BRIGHT/VIVID PALETTE =================
    # High contrast colors for dark backgrounds
    BRIGHT_RED = (255, 0, 0)
    BRIGHT_GREEN = (0, 255, 0)
    BRIGHT_BLUE = (0, 0, 255)
    BRIGHT_YELLOW = (255, 255, 0)
    BRIGHT_MAGENTA = (255, 0, 255)
    BRIGHT_CYAN = (0, 255, 255)
    BRIGHT_ORANGE = (255, 128, 0)
    BRIGHT_LIME = (128, 255, 0)
    BRIGHT_PINK = (255, 0, 128)
    BRIGHT_SKY = (0, 128, 255)
    
    # ================= EXTENDED DETECTION PALETTE =================
    # Large palette for multi-class detection
    DETECTION_COLORS = [
        RED, GREEN, BLUE, YELLOW, MAGENTA, CYAN,
        (255, 165, 0),    # Orange
        (128, 0, 128),    # Purple
        (255, 20, 147),   # Deep Pink
        (0, 191, 255),    # Deep Sky Blue
        (50, 205, 50),    # Lime Green
        (255, 69, 0),     # Red Orange
        (138, 43, 226),   # Blue Violet
        (255, 215, 0),    # Gold
        (32, 178, 170),   # Light Sea Green
        (220, 20, 60),    # Crimson
        (0, 206, 209),    # Dark Turquoise
        (148, 0, 211),    # Dark Violet
        (0, 250, 154),    # Medium Spring Green
        (255, 105, 180),  # Hot Pink
        (65, 105, 225),   # Royal Blue
        (255, 127, 80),   # Coral
        (154, 205, 50),   # Yellow Green
        (255, 182, 193),  # Light Pink
        (135, 206, 235),  # Sky Blue
        (240, 128, 128),  # Light Coral
        (175, 238, 238),  # Pale Turquoise
        (255, 218, 185),  # Peach Puff
    ]
    
    # ================= PPE COLOR MAPPING =================
    PPE_SEMANTIC_COLORS = {
        'person': PERSON_COLOR,
        'helmet': HELMET_COLOR,
        'safety_helmet': SAFETY_GREEN,
        'vest': VEST_COLOR,
        'safety_vest': SAFETY_YELLOW,
        'gloves': GLOVES_COLOR,
        'safety_gloves': CYAN,
        'boots': BOOTS_COLOR,
        'safety_boots': MAGENTA,
        'glasses': GLASSES_COLOR,
        'safety_glasses': SAFETY_ORANGE,
        'mask': MASK_COLOR,
        'face_mask': (128, 0, 128),
        'hard_hat': SAFETY_GREEN,
        'high_vis': SAFETY_YELLOW,
    }
    
    # ================= SCHEME COLLECTIONS =================
    PROFESSIONAL_SCHEME = [
        PROF_RED, PROF_GREEN, PROF_BLUE, PROF_YELLOW,
        PROF_ORANGE, PROF_PURPLE, PROF_CYAN, PROF_GRAY
    ]
    
    BRIGHT_SCHEME = [
        BRIGHT_RED, BRIGHT_GREEN, BRIGHT_BLUE, BRIGHT_YELLOW,
        BRIGHT_MAGENTA, BRIGHT_CYAN, BRIGHT_ORANGE, BRIGHT_LIME,
        BRIGHT_PINK, BRIGHT_SKY
    ]
    
    SAFETY_SCHEME = [
        SAFETY_RED, SAFETY_GREEN, SAFETY_YELLOW, SAFETY_ORANGE,
        VIOLATION, COMPLIANT, WARNING, HIGH_CONFIDENCE
    ]

    @classmethod
    def get_detection_color(cls, class_id: int, scheme: str = 'default') -> tuple:
        """Get color for detection based on class ID and scheme"""
        if scheme == 'professional':
            return cls.PROFESSIONAL_SCHEME[class_id % len(cls.PROFESSIONAL_SCHEME)]
        elif scheme == 'bright':
            return cls.BRIGHT_SCHEME[class_id % len(cls.BRIGHT_SCHEME)]
        elif scheme == 'safety':
            return cls.SAFETY_SCHEME[class_id % len(cls.SAFETY_SCHEME)]
        else:  # default
            return cls.DETECTION_COLORS[class_id % len(cls.DETECTION_COLORS)]
    
    @classmethod
    def get_ppe_color(cls, class_name: str) -> tuple:
        """Get semantic color for PPE item"""
        class_lower = class_name.lower() if isinstance(class_name, str) else str(class_name).lower()
        return cls.PPE_SEMANTIC_COLORS.get(class_lower, cls.UNKNOWN)
    
    @classmethod
    def get_confidence_color(cls, confidence: float) -> tuple:
        """Get color based on confidence level"""
        if confidence >= 0.8:
            return cls.HIGH_CONFIDENCE
        elif confidence >= 0.5:
            return cls.MEDIUM_CONFIDENCE
        else:
            return cls.LOW_CONFIDENCE
    
    @classmethod
    def get_violation_color(cls, has_violation: bool) -> tuple:
        """Get color for violation status"""
        return cls.VIOLATION if has_violation else cls.COMPLIANT

# Legacy support - keep existing variable names
COLORS = ColorPalette.DETECTION_COLORS
BRIGHT_COLORS = ColorPalette.BRIGHT_SCHEME
PROFESSIONAL_COLORS = ColorPalette.PROFESSIONAL_SCHEME
PPE_COLORS = ColorPalette.PPE_SEMANTIC_COLORS