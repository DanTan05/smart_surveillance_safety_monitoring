"""
Motion Detection Module  (Change Detection / Background Modelling)
===================================================================
Uses OpenCV's BackgroundSubtractorMOG2 to detect moving objects.
"""

import cv2
import numpy as np


class MotionDetector:

    def __init__(self, history: int = 500,
                 var_threshold: float = 16.0,
                 min_area: int = 1500):
        self.history       = history
        self.var_threshold = var_threshold
        self.min_area      = min_area
        self._init_subtractor()
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def detect(self, frame: np.ndarray):
        fg_mask = self.backSub.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  self._kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        annotated       = frame.copy()
        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(annotated, "MOTION",
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

        if motion_detected:
            cv2.putText(annotated, "! MOTION DETECTED !",
                        (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

        return annotated, fg_mask, motion_detected

    def reset(self):
        self._init_subtractor()

    def _init_subtractor(self):
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=True,
        )
