# Tennis Match Prediction API

A REST API that predicts ATP tennis match outcomes using a Random Forest machine learning model.

## 🎯 Features

- **Machine Learning**: Random Forest classifier with 68.9% test accuracy
- **REST API**: FastAPI-based API with automatic documentation
- **Data Validation**: Pydantic models ensure data quality
- **Interactive Docs**: Swagger UI at `/docs` endpoint
- **Real-time Predictions**: Sub-100ms prediction latency

## 🛠️ Tech Stack

- **Python 3.11**
- **FastAPI** - Modern web framework
- **scikit-learn** - Machine learning
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

## 📋 Prerequisites

- Python 3.11+
- Conda (recommended) or pip

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/joseloo04/tennis-prediction-api.git
cd tennis-prediction-api
```

2. Create and activate conda environment:
```bash
conda create -n tennis-api python=3.11 -y
conda activate tennis-api
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Train the model (or use pre-trained):
```bash
## ⚠️ Note About Model Files

The trained model file (`tennis_rf_small_model.joblib`) is not included in this repository due to its size. To use this API, you'll need to:

1. Train the model using your own dataset, or
2. Contact me for access to the pre-trained model

The model metadata is included in `saved_models/model_metadata.json`.
python train_model.py
```

## 🎮 Usage

1. Start the API server:
```bash
uvicorn app.main:app --reload
```

2. Open your browser and visit:
   - API: http://127.0.0.1:8000
   - Interactive Docs: http://127.0.0.1:8000/docs

3. Make a prediction:
```python
import requests

url = "http://127.0.0.1:8000/predict"
match = {
    "Rank_1": 5.0,
    "Rank_2": 12.0,
    "Pts_1": 5500.0,
    "Pts_2": 3200.0,
    "Odd_1": 1.45,
    "Odd_2": 2.75
}

response = requests.post(url, json=match)
print(response.json())
```

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Player rankings, ATP points, betting odds
- **Accuracy**: 68.9% on test set
- **Training Data**: ATP tennis matches dataset

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | API health check |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Predict match outcome |

## 📝 Example Response
```json
{
  "prediction": 1,
  "winner": "Player_1",
  "confidence": 0.72,
  "probability_player1": 0.72,
  "probability_player2": 0.28
}
```

## 🚧 Project Status

Currently in development. Upcoming features:
- [ ] PostgreSQL database integration
- [ ] API authentication
- [ ] Docker containerization
- [ ] AWS deployment
- [ ] Performance monitoring

## 👨‍💻 Author

**Jose Calderon**
- GitHub: [@joseloo04](https://github.com/joseloo04)
- Email: josefco.calderon387@gmail.com

## 📄 License

This project is open source and available under the MIT License.