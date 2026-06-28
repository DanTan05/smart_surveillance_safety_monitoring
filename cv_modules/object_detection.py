"""
Object Detection & Tracking Module
====================================
Uses YOLOv8 (Ultralytics) for detecting and tracking objects.

Model choice — yolov8s (small):
  YOLOv8 comes in several sizes: n (nano) < s (small) < m (medium) < l < x
  We use 'small' which offers a strong accuracy / speed trade-off.
  On a modern laptop CPU it still runs comfortably above 10 FPS for video.

Dual-model design:
  This class is purely parametrized by model_path — it does NOT automatically
  swap to a custom model. The app instantiates TWO separate ObjectDetector
  objects: one loading the general-purpose yolov8s.pt (COCO, 80 classes) for
  general surveillance, and a second loading models/custom_model.pt (our
  fine-tuned head/helmet/person detector) for the dedicated "PPE / Safety
  Detection" toggle. Running them side-by-side — rather than having the
  fine-tuned model silently replace the general detector — means the system
  keeps full COCO-class detection available even when the safety-specific
  model is active. This also avoids relying on the fine-tuned model's weak
  "person" class (see Section 3.7 of the report) for general person detection,
  since the base COCO model already handles that reliably.

Accuracy tuning:
  conf_threshold (default 0.45) — minimum confidence to keep a detection.
  iou_threshold  (default 0.45) — IoU used for Non-Max Suppression (NMS).

Two modes:
  detect()  → single-frame detection (no memory between frames)
  track()   → ByteTrack multi-object tracking (persistent unique IDs)
"""

import os
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


PALETTE = [
    (34,  197,  94),   # green
    (239,  68,  68),   # red
    ( 59, 130, 246),   # blue
    (234, 179,   8),   # yellow
    (168,  85, 247),   # purple
    ( 20, 184, 166),   # teal
    (249, 115,  22),   # orange
    (236,  72, 153),   # pink
]


class ObjectDetector:
    """
    Wraps a single YOLOv8 model for detection and tracking.

    Parameters
    ----------
    model_path : str
        Path or name of the YOLO weights to load. Pass "yolov8s.pt" for the
        general-purpose COCO model, or "models/custom_model.pt" for the
        fine-tuned safety/PPE model. The caller decides which to instantiate
        — this class never swaps models on its own.
    """

    def __init__(self, model_path: str = "yolov8s.pt",
                 conf_threshold: float = 0.45,
                 iou_threshold: float = 0.45):
        self.model       = YOLO(model_path)
        self._model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold
        self.class_names    = self.model.names

    def detect(self, frame: np.ndarray, conf: Optional[float] = None,
               iou: Optional[float] = None, fixed_color: Optional[tuple] = None,
               label_prefix: str = ""):
        conf    = conf if conf is not None else self.conf_threshold
        iou     = iou if iou is not None else self.iou_threshold
        results = self.model(frame, conf=conf, iou=iou, verbose=False)
        output  = self._draw_boxes(frame.copy(), results[0], fixed_color, label_prefix)
        return output, len(results[0].boxes)

    def track(self, frame: np.ndarray, conf: Optional[float] = None,
              iou: Optional[float] = None):
        conf    = conf if conf is not None else self.conf_threshold
        iou     = iou if iou is not None else self.iou_threshold
        results = self.model.track(
            frame, conf=conf, iou=iou, persist=True,
            tracker="bytetrack.yaml", verbose=False,
        )
        output = self._draw_tracks(frame.copy(), results[0])
        active = (len(results[0].boxes.id)
                  if results[0].boxes.id is not None else 0)
        return output, active

    def reset_tracker(self):
        self.model = YOLO(self._model_path)

    def _color(self, idx: int):
        return PALETTE[int(idx) % len(PALETTE)]

    def _put_label(self, frame, text, x, y, color):
        (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x, y - th - bl - 6), (x + tw + 6, y), color, -1)
        cv2.putText(frame, text, (x + 3, y - bl - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    def _draw_boxes(self, frame, result, fixed_color=None, label_prefix=""):
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls   = int(box.cls[0])
            conf  = float(box.conf[0])
            color = fixed_color if fixed_color is not None else self._color(cls)
            label = f"{label_prefix}{self.class_names[cls]}  {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._put_label(frame, label, x1, y1, color)
        return frame

    def _draw_tracks(self, frame, result):
        if result.boxes.id is None:
            return frame
        for box, track_id in zip(result.boxes, result.boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls   = int(box.cls[0])
            tid   = int(track_id)
            color = self._color(tid)
            label = f"ID {tid}  {self.class_names[cls]}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            self._put_label(frame, label, x1, y1, color)
        return frame
