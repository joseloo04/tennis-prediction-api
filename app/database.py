from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get database URL from environment
DATABASE_URL = os.getenv("DATABASE_URL")

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a configured "Session" class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

class Prediction(Base):
    """Database model for storing predictions"""
    __tablename__ = "predictions"
    
    # Primary key - unique ID for each prediction
    id = Column(Integer, primary_key=True, index=True)
    
    # Timestamp when prediction was made
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Input features
    rank_1 = Column(Float, nullable=False)
    rank_2 = Column(Float, nullable=False)
    pts_1 = Column(Float, nullable=False)
    pts_2 = Column(Float, nullable=False)
    odd_1 = Column(Float, nullable=False)
    odd_2 = Column(Float, nullable=False)
    
    # Prediction results
    prediction = Column(Integer, nullable=False)  # 0 or 1
    winner = Column(String, nullable=False)  # "Player_1" or "Player_2"
    confidence = Column(Float, nullable=False)
    probability_player1 = Column(Float, nullable=False)
    probability_player2 = Column(Float, nullable=False)
    
    # Performance tracking
    processing_time = Column(Float, nullable=False)  # in seconds

def get_db():
    """
    Dependency that provides a database session.
    Automatically closes the session after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")