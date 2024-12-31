"""layers init"""
from .yolov3_head import *
from .yolov4_head import *
from .yolov5_head import *
from .yolov7_head import *
from .yolov8_head import *
from .yolov9_head import *
from .yolox_head import *
from .yolov10_head import *
from .yolov11_head import *

__all__ = [
    "YOLOv3Head",
    "YOLOv4Head",
    "YOLOv5Head",
    "YOLOv7Head", "YOLOv7AuxHead",
    "YOLOv8Head", "YOLOv8SegHead",
    "YOLOXHead",
    "YOLOv9Head",
    "YOLOv10Head",
    "YOLOv11Head"
]
