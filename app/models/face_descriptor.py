from sqlalchemy import Column, Integer, ForeignKey, DateTime, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class FaceDescriptor(Base):
    __tablename__ = "face_descriptors"

    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), nullable=False)
    embedding = Column(JSON, nullable=False)   # 512-d FaceNet vector stored as JSON array
    quality_score = Column(Float)              # 0.0 - 1.0 enrollment quality
    blur_score = Column(Float)
    brightness_score = Column(Float)
    source = Column(Integer, default=0)        # 0=upload, 1=webcam
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    student = relationship("Student", back_populates="face_descriptors")
