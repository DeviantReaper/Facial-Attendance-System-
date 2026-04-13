from sqlalchemy import Column, Integer, String, Boolean, DateTime, JSON, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class College(Base):
    __tablename__ = "colleges"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(300), nullable=False)
    code = Column(String(50), unique=True, nullable=False)
    api_key_hash = Column(String(255))         # bcrypt hash of the raw API key
    permissions = Column(JSON, default={"enroll": True, "read": True, "write": False})
    webhook_url = Column(String(500))
    ip_whitelist = Column(JSON, default=[])    # list of allowed IP ranges
    contact_email = Column(String(200))
    contact_name = Column(String(200))
    logo_url = Column(String(500))
    is_active = Column(Boolean, default=True)
    auto_approve_enrollment = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    students = relationship("Student", back_populates="college")
