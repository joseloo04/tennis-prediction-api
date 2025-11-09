from pydantic import BaseModel, Field
from typing import Optional

class MatchFeatures(BaseModel):
    """Input features for tennis match prediction"""
    # Greater than or equal to zero = ge
    # Greater than = gt 
    Rank_1: float = Field(..., description="ATP ranking of Player 1", ge=1)
    Rank_2: float = Field(..., description="ATP ranking of Player 2", ge=1)
    Pts_1: float = Field(..., description="ATP points of Player 1", ge=0)
    Pts_2: float = Field(..., description="ATP points of Player 2", ge=0)
    Odd_1: float = Field(..., description="Betting odds for Player 1", gt=0)
    Odd_2: float = Field(..., description="Betting odds for Player 2", gt=0)

    class Config:
        json_schema_extra = {
            "example": {
                "Rank_1": 5.0,
                "Rank_2": 12.0,
                "Pts_1": 5500.0,
                "Pts_2": 3200.0,
                "Odd_1": 1.45,
                "Odd_2": 2.75
            }
        }

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: int = Field(..., description="0 for Player_2 wins, 1 for Player_1 wins")
    winner: str = Field(..., description="Predicted winner (Player_1 or Player_2)")
    confidence: float = Field(..., description="Confidence of prediction (0-1)")
    probability_player1: float = Field(..., description="Probability Player 1 wins")
    probability_player2: float = Field(..., description="Probability Player 2 wins")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    message: str
    model_config = {"protected_namespaces": ()}