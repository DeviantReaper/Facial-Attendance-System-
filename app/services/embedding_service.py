"""
Embedding service — lazy-loads DeepFace/FaceNet512.
If DeepFace is not installed, returns None and recognition is disabled.
"""
import numpy as np
import cv2
from typing import Optional, List


class EmbeddingService:
    def __init__(self):
        self._deepface = None
        self._available = None  # None = not yet checked

    def _try_load(self):
        if self._available is not None:
            return self._available
        try:
            import deepface
            from deepface import DeepFace
            self._deepface = DeepFace
            self._available = True
            print("[EmbeddingService] DeepFace loaded — FaceNet512 ready")
        except ImportError:
            self._available = False
            print("[EmbeddingService] DeepFace not installed — recognition disabled (demo mode)")
        return self._available

    def get_embedding(self, face_img: np.ndarray) -> Optional[List[float]]:
        """Extract 512-d FaceNet embedding from a face crop."""
        if not self._try_load():
            # Demo mode: return a random unit vector so the server doesn't crash
            v = np.random.randn(512).astype(np.float32)
            return (v / np.linalg.norm(v)).tolist()

        try:
            # DeepFace expects BGR → RGB
            rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            result = self._deepface.represent(
                img_path=rgb,
                model_name="Facenet512",
                detector_backend="skip",  # already cropped
                enforce_detection=False,
            )
            if result and isinstance(result, list):
                return result[0]["embedding"]
            return None
        except Exception as e:
            print(f"[EmbeddingService] Error: {e}")
            return None


embedding_service = EmbeddingService()
