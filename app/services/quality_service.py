import cv2
import numpy as np
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)


class QualityService:
    """
    Checks image quality before face enrollment.
    Returns scores for blur, brightness, and an overall quality rating.
    """

    def check_quality(self, img: np.ndarray) -> dict:
        if img is None or img.size == 0:
            return {"blur_score": 0, "brightness_score": 0, "overall_score": 0, "passed": False, "issues": ["Empty image"]}

        issues = []

        # ── Blur detection via Laplacian variance ────────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ── Brightness analysis ──────────────────────────────────────────────
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        brightness_score = float(hsv[:, :, 2].mean())

        # ── Validate thresholds ──────────────────────────────────────────────
        if blur_score < settings.QUALITY_BLUR_THRESHOLD:
            issues.append(f"Image too blurry (score={blur_score:.1f}, minimum={settings.QUALITY_BLUR_THRESHOLD})")

        if brightness_score < settings.QUALITY_MIN_BRIGHTNESS:
            issues.append(f"Image too dark (brightness={brightness_score:.1f})")
        elif brightness_score > settings.QUALITY_MAX_BRIGHTNESS:
            issues.append(f"Image too bright / overexposed (brightness={brightness_score:.1f})")

        # ── Resolution check ──────────────────────────────────────────────────
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            issues.append(f"Face crop too small ({w}x{h}px)")

        # ── Overall quality score (0-1) ────────────────────────────────────
        blur_norm = min(blur_score / 300.0, 1.0)
        brightness_norm = 1.0 - abs(brightness_score - 135) / 135.0
        brightness_norm = max(0, brightness_norm)
        overall_score = round((blur_norm * 0.6 + brightness_norm * 0.4), 3)

        return {
            "blur_score": round(blur_score, 2),
            "brightness_score": round(brightness_score, 2),
            "overall_score": overall_score,
            "passed": len(issues) == 0,
            "issues": issues,
            "resolution": f"{w}x{h}",
        }


quality_service = QualityService()
