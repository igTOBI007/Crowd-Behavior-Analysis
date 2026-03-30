import datetime

# ================= VIDEO SETTINGS =================
VIDEO_CONFIG = {
    # Use 0 for webcam OR put video file name like "test_crowd.mp4"
    "VIDEO_CAP": "test_crowd.mp4.mp4.mp4",
    
    # True for webcam, False for video file
    "IS_CAM": True,
    
    "CAM_APPROX_FPS": 3,
    "HIGH_CAM": False,
    
    "START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}

# ================= YOLO SETTINGS =================
YOLO_CONFIG = {
    "WEIGHTS_PATH": "YOLOv4-tiny/yolov4-tiny.weights",
    "CONFIG_PATH": "YOLOv4-tiny/yolov4-tiny.cfg"
}

# ================= DISPLAY OPTIONS =================
SHOW_PROCESSING_OUTPUT = True
SHOW_DETECT = True

# ================= DATA RECORDING =================
DATA_RECORD = True
DATA_RECORD_RATE = 10

# ================= RESTRICTED ENTRY =================
RE_CHECK = False
RE_START_TIME = datetime.time(22, 0, 0)
RE_END_TIME = datetime.time(6, 0, 0)

# ================= SOCIAL DISTANCING =================
SD_CHECK = True
SHOW_VIOLATION_COUNT = False
SHOW_TRACKING_ID = False
SOCIAL_DISTANCE = 50

# ================= ABNORMAL BEHAVIOR =================
ABNORMAL_CHECK = True
ABNORMAL_MIN_PEOPLE = 5
ABNORMAL_ENERGY = 100
ABNORMAL_THRESH = 0.66
DATA_RECORD_ABNORMAL_CHECK = True

# ================= MODEL PARAMETERS =================
MIN_CONF = 0.3
NMS_THRESH = 0.2
FRAME_SIZE = 720
TRACK_MAX_AGE = 3