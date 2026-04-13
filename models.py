from sqlalchemy import Column, Integer, String, LargeBinary
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    # Stores a pickle-serialized list of numpy arrays (multiple encodings per person)
    encoding = Column(LargeBinary, nullable=False)
