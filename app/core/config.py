import os
from pydantic_settings import BaseSettings
from typing import List, Optional


class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "FaceAttend — Attendance Portal"
    API_V1_STR: str = "/api/v1"
    INSTITUTION_NAME: str = "Manipal University Jaipur"
    ACADEMIC_YEAR: str = "2025-26"

    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./facial_attendance.db")

    # ML Model Settings
    RECOGNITION_MODEL: str = "Facenet512"         # DeepFace model — best accuracy
    DETECTION_MODEL: str = "yolov8n.pt"           # fallback detection
    FACE_MODEL_DIR: str = "./frontend/public/models"  # face-api.js model weights

    # Detection / Recognition thresholds
    DETECTION_THRESHOLD: float = 0.5
    RECOGNITION_THRESHOLD: float = 0.40           # cosine distance (lower = stricter)
    ATTENDANCE_CONFIDENCE_THRESHOLD: float = 0.75 # min confidence to auto-mark present
    QUALITY_BLUR_THRESHOLD: float = 80.0          # Laplacian variance min
    QUALITY_MIN_BRIGHTNESS: float = 50.0
    QUALITY_MAX_BRIGHTNESS: float = 220.0

    # Attendance Settings
    ATTENDANCE_COOLDOWN_MINUTES: int = 5

    # JWT Security
    JWT_SECRET: str = os.getenv("JWT_SECRET", "CHANGE_THIS_TO_A_RANDOM_SECRET_IN_PRODUCTION")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440   # 24 hours

    # Email / SMTP
    SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER: str = os.getenv("SMTP_USER", "")
    SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
    EMAILS_ENABLED: bool = os.getenv("EMAILS_ENABLED", "false").lower() == "true"
    FROM_EMAIL: str = os.getenv("FROM_EMAIL", "noreply@faceattend.edu")

    # File Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    MAX_UPLOAD_SIZE_MB: int = 10

    # CORS
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"


settings = Settings()
