"""
Face Detection Module
======================
Uses OpenCV's DNN face detector — a Single-Shot Detector (SSD) with a
ResNet-10 backbone, trained specifically for face localisation.

Why DNN instead of Haar Cascade?
  Haar Cascade (Viola-Jones) is fast but prone to false positives — it
  frequently misclassifies textured backgrounds, fabric patterns, or
  random edge clusters as faces. This is a well-documented weakness of
  the algorithm: it relies on simple intensity-difference features that
  can coincidentally match non-face regions.

  The DNN detector instead uses a small convolutional neural network
  (only ~10 MB) trained on a large face dataset. It learns much richer,
  more discriminative features, which gives substantially fewer false
  positives while still running comfortably in real time on CPU.

Model files (auto-downloaded on first run, cached locally):
  deploy.prototxt                            — network architecture definition
  res10_300x300_ssd_iter_140000.caffemodel   — pre-trained weights

Pipeline per frame:
  1. Resize frame to the network's expected 300×300 input
  2. Build a 4D "blob" with per-channel mean subtraction (104, 177, 123)
     — these are the dataset mean values the network was trained with
  3. Forward pass through the SSD network
  4. Each of the (up to 200) candidate detections has a confidence score
     and a relative bounding box
  5. Keep detections above the confidence threshold; scale boxes back
     to the original frame's pixel dimensions
"""

import os
import urllib.request
from typing import Tuple

import cv2
import numpy as np


PROTOTXT_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/"
    "samples/dnn/face_detector/deploy.prototxt"
)
WEIGHTS_URL = (
    "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
    "dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)

MODELS_DIR    = "models"
PROTOTXT_PATH = os.path.join(MODELS_DIR, "deploy.prototxt")
WEIGHTS_PATH  = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


def _ensure_model_files():
    """Download the DNN model files on first run if not already cached."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(PROTOTXT_PATH):
        urllib.request.urlretrieve(PROTOTXT_URL, PROTOTXT_PATH)
    if not os.path.exists(WEIGHTS_PATH):
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)


class FaceDetector:
    """
    Frontal/near-frontal face detector using OpenCV's DNN SSD face model.

    Parameters
    ----------
    confidence_threshold : float
        Minimum confidence (0-1) required to accept a detection.
        Raised to 0.6 by default — significantly stricter than typical
        Haar Cascade settings, which eliminates almost all false positives
        while still reliably catching real faces.
    """

    BOX_COLOR = (200, 0, 255)   # Magenta / pink

    def __init__(self, confidence_threshold: float = 0.6):
        _ensure_model_files()
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, WEIGHTS_PATH)
        self.confidence_threshold = confidence_threshold

    # ── Public API ────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect faces in a BGR frame.

        Parameters
        ----------
        frame : np.ndarray  BGR image (H × W × 3)

        Returns
        -------
        annotated_frame : np.ndarray  Frame with face bounding boxes drawn
        face_count      : int         Number of faces detected
        """
        h, w = frame.shape[:2]

        # Build the input blob: resize to 300x300, subtract per-channel means
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()   # shape: (1, 1, N, 7)

        annotated  = frame.copy()
        face_count = 0

        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue

            face_count += 1

            # Detection box is relative [0,1] — scale to actual frame size
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            label = f"Face  {confidence:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.BOX_COLOR, 2)

            (tw, th), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(annotated,
                          (x1, y1 - th - bl - 6), (x1 + tw + 6, y1),
                          self.BOX_COLOR, -1)
            cv2.putText(annotated, label,
                        (x1 + 3, y1 - bl - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        return annotated, face_count
