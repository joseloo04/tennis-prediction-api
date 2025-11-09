import joblib
import json
from pathlib import Path
import pandas as pd
from typing import Dict

class TennisPredictor:
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_names = None
        self.model_loaded = False
        
    def load_model(self):
        """Load the trained Random Forest model and metadata"""
        try:
            # Get the path to where our model is saved
            current_dir = Path(__file__).parent.parent.parent
            model_path = current_dir / "saved_models" / "tennis_rf_small_model.joblib"
            metadata_path = current_dir / "saved_models" / "model_metadata.json"
            
            # Load the model using joblib
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from: {model_path}")
            
            # Load the metadata (info about the model)
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            # Save the feature names we need
            self.feature_names = self.metadata['features']
            self.model_loaded = True
            
            print(f"✓ Expected features: {self.feature_names}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """Make a prediction given match features"""
        
        # Step 1: Check if model is loaded
        if not self.model_loaded:
            raise Exception("Model not loaded!")
        
        # Step 2: Validate we have all required features
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Step 3: Convert dictionary to DataFrame (what sklearn expects)
        input_data = pd.DataFrame([features])[self.feature_names]
        
        # Step 4: Make prediction
        prediction = self.model.predict(input_data)[0]
        probabilities = self.model.predict_proba(input_data)[0]
        
        # Step 5: Format the result nicely
        result = {
            'prediction': int(prediction),
            'winner': 'Player_1' if prediction == 1 else 'Player_2',
            'confidence': float(max(probabilities)),
            'probability_player1': float(probabilities[1]),
            'probability_player2': float(probabilities[0])
        }
        
        return result

# Initialize the predictor
predictor = TennisPredictor()