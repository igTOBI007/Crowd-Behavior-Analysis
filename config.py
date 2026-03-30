import datetime

# Video Path
VIDEO_CONFIG = {
    "VIDEO_CAP": 0,
    "IS_CAM": False,
    "CAM_APPROX_FPS": 3,
    "HIGH_CAM": False,
    "START_TIME": datetime.datetime(2020, 11, 5, 0, 0, 0, 0)
}

# Load YOLOv3-tiny weights and config
YOLO_CONFIG = {
    "WEIGHTS_PATH": "YOLOv4-tiny/yolov4-tiny.weights",
    "CONFIG_PATH": "YOLOv4-tiny/yolov4-tiny.cfg"
}

# Show individuals detected
SHOW_PROCESSING_OUTPUT = True
# Show individuals detected
SHOW_DETECT = True
# Data record
DATA_RECORD = True
DATA_RECORD_RATE = 10
# Restricted entry
RE_CHECK = False
RE_START_TIME = datetime.time(22, 0, 0)
RE_END_TIME = datetime.time(6, 0, 0)
# Social distance
SD_CHECK = True
SHOW_VIOLATION_COUNT = False
SHOW_TRACKING_ID = False
SOCIAL_DISTANCE = 50
# Abnormal
ABNORMAL_CHECK = True
ABNORMAL_MIN_PEOPLE = 5
ABNORMAL_ENERGY = 100
ABNORMAL_THRESH = 0.66
DATA_RECORD_ABNORMAL_CHECK = True

MIN_CONF = 0.3
NMS_THRESH = 0.2
FRAME_SIZE = 720
TRACK_MAX_AGE = 3