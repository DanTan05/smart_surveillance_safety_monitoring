"""
Edge Detection Module
======================
Applies Canny edge detection to highlight structural boundaries in a frame.
"""

import cv2
import numpy as np


class EdgeDetector:

    EDGE_COLOR = (0, 255, 255)

    def detect(self, frame: np.ndarray,
               threshold1: int   = 50,
               threshold2: int   = 150,
               blend_alpha: float = 0.65) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1, threshold2)
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges_bgr[edges > 0] = self.EDGE_COLOR
        blended = cv2.addWeighted(frame, blend_alpha,
                                  edges_bgr, 1.0 - blend_alpha, 0)
        return blended
