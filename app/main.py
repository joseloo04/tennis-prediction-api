from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time
from sqlalchemy.orm import Session

from app.models import MatchFeatures, PredictionResponse, HealthResponse
from app.ml.model import predictor
from app.database import create_tables, get_db, Prediction

# Define lifespan events for startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This runs when the API STARTS
    print("🚀 Starting up API...")
    
    # Load ML model
    success = predictor.load_model()
    if not success:
        print("⚠️  WARNING: Model failed to load!")
    
    # Create database tables
    try:
        create_tables()
        print("✓ Database initialized successfully")
    except Exception as e:
        print(f"⚠️  WARNING: Database initialization failed: {e}")
    
    yield
    
    # This runs when the API STOPS
    print("👋 Shutting down API...")

# # Initialize FastAPI
app = FastAPI(
    title="Tennis Match Prediction API",
    description="Predict ATP tennis match outcomes using Random Forest ML model",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    # TODO: In production, change to specific origins like ["https://yourdomain.com"]
    allow_origins=["*"],  # Allow all origins (development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root():
    """Welcome endpoint"""
    return {
        "message": "Tennis Match Prediction API",
        "author": "Jose Calderon",
        "docs": "/docs",
        "model_info": "/model/info"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if API and model are ready"""
    return {
        "status": "healthy" if predictor.model_loaded else "unhealthy",
        "model_loaded": predictor.model_loaded,
        "message": "Model loaded successfully" if predictor.model_loaded else "Model not loaded"
    }

# Model info endpoint
@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "model_type": predictor.metadata['model_type'],
        "n_estimators": predictor.metadata['n_estimators'],
        "min_samples_leaf": predictor.metadata['min_samples_leaf'],
        "test_accuracy": predictor.metadata['test_accuracy'],
        "validation_accuracy": predictor.metadata['validation_accuracy'],
        "required_features": predictor.feature_names
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_match(
    features: MatchFeatures,
    db: Session = Depends(get_db)  # Inject database session
):
    """
    Predict the winner of a tennis match and save to database
    
    Returns:
    - prediction: 0 = Player_2 wins, 1 = Player_1 wins
    - winner: "Player_1" or "Player_2"
    - confidence: Model confidence (0-1)
    - probabilities for each player
    """
    # Step 1: Check if model is loaded
    if not predictor.model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Step 2: Convert Pydantic model to dictionary
        features_dict = features.model_dump()
        
        # Step 3: Make prediction and time it
        start_time = time.time()
        result = predictor.predict(features_dict)
        processing_time = time.time() - start_time
        
        # Step 4: Save prediction to database
        db_prediction = Prediction(
            rank_1=features.Rank_1,
            rank_2=features.Rank_2,
            pts_1=features.Pts_1,
            pts_2=features.Pts_2,
            odd_1=features.Odd_1,
            odd_2=features.Odd_2,
            prediction=result['prediction'],
            winner=result['winner'],
            confidence=result['confidence'],
            probability_player1=result['probability_player1'],
            probability_player2=result['probability_player2'],
            processing_time=processing_time
        )
        
        # Add to database and commit
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)  # Get the ID back
        
        # Step 5: Log it
        print(f"✓ Prediction #{db_prediction.id}: {result['winner']} "
              f"(confidence: {result['confidence']:.2%}, time: {processing_time:.4f}s)")
        
        return result
        
    except ValueError as e:
        # Validation errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Roll back database changes if error
        db.rollback()
        # Unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )
    
# Added analytics endpoint
@app.get("/analytics")
async def get_analytics(db: Session = Depends(get_db)):
    """
    Get analytics about all predictions made
    
    Returns statistics about API usage and predictions
    """
    try:
        # Get all predictions from database
        all_predictions = db.query(Prediction).all()
        
        # Calculate statistics
        total_predictions = len(all_predictions)
        
        if total_predictions == 0:
            return {
                "message": "No predictions made yet",
                "total_predictions": 0
            }
        
        # Count winners
        player1_wins = sum(1 for p in all_predictions if p.winner == "Player_1")
        player2_wins = sum(1 for p in all_predictions if p.winner == "Player_2")
        
        # Average confidence
        avg_confidence = sum(p.confidence for p in all_predictions) / total_predictions
        
        # Average processing time
        avg_processing_time = sum(p.processing_time for p in all_predictions) / total_predictions
        
        # Most recent prediction
        most_recent = max(all_predictions, key=lambda p: p.timestamp)
        
        return {
            "total_predictions": total_predictions,
            "predictions_by_winner": {
                "Player_1": player1_wins,
                "Player_2": player2_wins
            },
            "average_confidence": round(avg_confidence, 4),
            "average_processing_time_seconds": round(avg_processing_time, 6),
            "most_recent_prediction": {
                "id": most_recent.id,
                "timestamp": most_recent.timestamp.isoformat(),
                "winner": most_recent.winner,
                "confidence": most_recent.confidence
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)