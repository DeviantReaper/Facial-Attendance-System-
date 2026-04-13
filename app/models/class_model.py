from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, JSON, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Class(Base):
    __tablename__ = "classes"

    id = Column(Integer, primary_key=True, index=True)
    subject_name = Column(String(200), nullable=False)
    subject_code = Column(String(50), nullable=False)
    teacher_id = Column(Integer, ForeignKey("teachers.id"), nullable=False)
    room_number = Column(String(50))
    schedule = Column(JSON)          # e.g. {"day": "Monday", "time": "09:00", "duration": 60}
    color = Column(String(20), default="#1e3a5f")
    min_attendance_pct = Column(Float, default=75.0)
    late_threshold_minutes = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    teacher = relationship("Teacher", back_populates="classes")
    enrollments = relationship("Enrollment", back_populates="class_", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="class_")
