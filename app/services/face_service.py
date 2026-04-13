"""
Face detection service — lazy-loads YOLOv8 / OpenCV.
Falls back to OpenCV Haar cascade if ultralytics is not installed.
"""
import cv2
import numpy as np
from typing import List, Dict
import os


class FaceDetectionService:
    def __init__(self):
        self._yolo = None
        self._haar = None
        self._mode = None  # "yolo" | "haar" | None

    def _load(self):
        if self._mode is not None:
            return
        # 1. Try YOLOv8
        try:
            from ultralytics import YOLO
            model_path = "./yolov8n.pt"
            if not os.path.exists(model_path):
                model_path = os.path.join(os.path.dirname(__file__), "../../yolov8n.pt")
            self._yolo = YOLO(model_path)
            self._mode = "yolo"
            print("[FaceDetectionService] YOLOv8 loaded")
            return
        except Exception:
            pass

        # 2. Fallback: Haar cascade (always available with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(cascade_path)
        self._mode = "haar"
        print("[FaceDetectionService] Using OpenCV Haar cascade (fallback)")

    def detect_faces(self, img: np.ndarray) -> List[Dict]:
        self._load()
        results = []

        if self._mode == "yolo" and self._yolo:
            try:
                preds = self._yolo.predict(img, conf=0.5, classes=[0], verbose=False)
                for pred in preds:
                    for box in pred.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])
                        results.append({"box": [x1, y1, x2, y2], "confidence": conf})
                return results
            except Exception as e:
                print(f"[FaceDetectionService] YOLO error, falling back: {e}")
                self._mode = "haar"

        if self._mode == "haar" and self._haar is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                results.append({"box": [x, y, x + w, y + h], "confidence": 0.8})

        return results


face_detector = FaceDetectionService()
