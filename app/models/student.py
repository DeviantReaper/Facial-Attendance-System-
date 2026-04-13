from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Student(Base):
    __tablename__ = "students"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    roll_no = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(200), unique=True, nullable=False)
    phone = Column(String(20))
    password_hash = Column(String(255))
    college_id = Column(Integer, ForeignKey("colleges.id"), nullable=True)
    department = Column(String(100))
    semester = Column(Integer)
    profile_photo_path = Column(String(500))
    is_enrolled = Column(Boolean, default=False)
    enrollment_status = Column(String(20), default="pending")  # pending, approved, rejected
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    college = relationship("College", back_populates="students")
    face_descriptors = relationship("FaceDescriptor", back_populates="student", cascade="all, delete-orphan")
    enrollments = relationship("Enrollment", back_populates="student", cascade="all, delete-orphan")
    attendance_records = relationship("AttendanceRecord", back_populates="student")
    notifications = relationship("Notification", back_populates="student")
