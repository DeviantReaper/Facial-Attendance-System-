from sqlalchemy import Column, Integer, String, JSON
from app.core.database import Base

# Legacy model kept for backward compatibility
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    embeddings = Column(JSON, default=[])
