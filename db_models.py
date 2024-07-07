import os
from sqlalchemy import create_engine, Column, Integer, String, Float, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Use the new declarative_base import for SQLAlchemy 2.0
from sqlalchemy.orm import declarative_base

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://username:password@localhost:5432/test")

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class FaceModel(Base):
    __tablename__ = "faces"
    id = Column(Integer, primary_key=True, index=True)
    embedding = Column(ARRAY(Float))
    image_path = Column(String)

# Create tables
Base.metadata.create_all(bind=engine)
