import numpy as np
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class RecognitionService:
    """
    Matches a query embedding against stored face descriptors
    using cosine similarity (1 - cosine distance).
    """

    def _cosine_similarity(self, a: list, b: list) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        n_a = np.linalg.norm(va)
        n_b = np.linalg.norm(vb)
        if n_a == 0 or n_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (n_a * n_b))

    def match_face(self, query_emb: list, emb_map: dict[int, list[list]]) -> tuple[int | None, float]:
        """
        emb_map: {student_id: [embedding_1, embedding_2, ...]}
        Returns (best_student_id, best_cosine_similarity) or (None, 0.0)
        """
        best_id = None
        best_sim = -1.0

        for student_id, embeddings in emb_map.items():
            for stored_emb in embeddings:
                sim = self._cosine_similarity(query_emb, stored_emb)
                if sim > best_sim:
                    best_sim = sim
                    best_id = student_id

        # Threshold: cosine similarity must be >= (1 - RECOGNITION_THRESHOLD)
        # e.g. threshold=0.4 means similarity must be >= 0.6
        min_similarity = 1.0 - settings.RECOGNITION_THRESHOLD
        if best_sim >= min_similarity:
            return best_id, best_sim

        return None, best_sim


recognition_service = RecognitionService()
