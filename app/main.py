from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import time

from app.models import MatchFeatures, PredictionResponse, HealthResponse
from app.ml.model import predictor

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Actions
    print("🚀 Starting up API...")
    success = predictor.load_model()
    # Log if model failed to load
    if not success:
        print("⚠️  WARNING: Model failed to load!")
    yield
    # This runs when the API STOPS
    print("👋 Shutting down API...")

# Create FastAPI app instance
app = FastAPI(
    title="Tennis Match Prediction API",
    description="Predict ATP tennis match outcomes using Random Forest ML model",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    # TODO: In production, change to specific origins like ["https://yourdomain.com"]
    allow_origins=["*"],  # Allow all origins (development only)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/predict", response_model=PredictionResponse)
async def predict_match(features: MatchFeatures):
    """
    Predict the winner of a tennis match
    
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
        
        # Step 4: Log it (helpful for debugging)
        print(f"✓ Prediction: {result['winner']} (confidence: {result['confidence']:.2%}, time: {processing_time:.4f}s)")
        
        return result
        
    except ValueError as e:
        # Validation errors (missing features, etc.)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        # Unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )           

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)