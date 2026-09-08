import os

CAMERA_INDEX = 0

LOG_DIR    = "logs"
OUTPUT_DIR = "outputs"
MODEL_DIR  = "models"

YOLO_MODEL_PATH       = os.path.join(MODEL_DIR, "yolov8s.pt")
MIDAS_MODEL_PATH      = os.path.join(MODEL_DIR, "midas_small.pt")
MIDAS_TRANSFORMS_PATH = os.path.join(MODEL_DIR, "midas_transforms.pt")

ATTENTION_THRESHOLD   = 0.25
DANGER_THRESHOLD      = 0.45
MOTION_WEIGHT         = 0.25
DEPTH_WEIGHT          = 0.40
APPROACH_WEIGHT       = 0.35
NEAR_REGION_THRESHOLD = 0.82
DEPTH_TAU             = 0.008

DIKKAT_THRESHOLD  = 0.30
TEHLIKE_THRESHOLD = 0.45

DANGER_SMOOTH_WINDOW = 10
TREND_THRESHOLD      = 0.04
TREND_BOOST          = 0.12

RISK_W_CLASS    = 0.28
RISK_W_DEPTH    = 0.22
RISK_W_MOTION   = 0.20
RISK_W_APPROACH = 0.18
RISK_W_BBOX     = 0.07
RISK_W_CONF     = 0.05

YOLO_CONF_THRESHOLD = 0.4

OUTPUT_VIDEO_PATH         = "outputs/detection_output.mp4"
OUTPUT_TRACKED_VIDEO_PATH = "outputs/tracked_output.mp4"

ENABLE_YOLO         = True
ENABLE_BYTETRACK    = True
OCCLUSION_TEST_MODE = False

DEPTH_EVERY_N_FRAMES = 3
FLOW_EVERY_N_FRAMES  = 2
FRAME_SCALE          = 0.5

PRIORITY_CLASSES = {
    2:  {"name": "car",           "risk": 3, "color": (0,   0, 255)},
    3:  {"name": "motorcycle",    "risk": 3, "color": (0,  40, 255)},
    5:  {"name": "bus",           "risk": 3, "color": (0,  80, 255)},
    6:  {"name": "train",         "risk": 3, "color": (0, 120, 255)},
    7:  {"name": "truck",         "risk": 3, "color": (0, 160, 255)},
    0:  {"name": "person",        "risk": 2, "color": (0, 140, 255)},
    1:  {"name": "bicycle",       "risk": 2, "color": (0, 165, 255)},
    16: {"name": "dog",           "risk": 2, "color": (0, 190, 255)},
    9:  {"name": "traffic light", "risk": 1, "color": (0, 255, 255)},
    11: {"name": "stop sign",     "risk": 1, "color": (0, 255, 220)},
    12: {"name": "parking meter", "risk": 1, "color": (0, 255, 200)},
    13: {"name": "bench",         "risk": 1, "color": (0, 255, 180)},
    41: {"name": "skateboard",    "risk": 1, "color": (0, 255, 160)}
}