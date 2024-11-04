from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl')
model = joblib.load(MODEL_PATH)


# Define input schemas for validation
class SinglePredictionRequest(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    Year: int
    Month: int
    Day: int
    CompetitionDistance: float = 0.0


class BatchPredictionRequest(BaseModel):
    data: list[SinglePredictionRequest]


# Define a helper function to preprocess input data
def preprocess_input(data):
    """
    Preprocesses input data to match the model's feature requirements.
    Assumes the input is either a single dictionary or a list of dictionaries.
    """
    df = pd.DataFrame([data.dict()]) if isinstance(data, SinglePredictionRequest) else pd.DataFrame(
        [item.dict() for item in data])

    # Ensure all required columns are present and fill missing values
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(df['CompetitionDistance'].mean())
    return df[['Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'CompetitionDistance']]


# Endpoint 1: Health check
@app.get("/health")
def health_check():
    return {"status": "OK"}


# Endpoint 2: Single prediction
@app.post("/predict")
def predict_single(request: SinglePredictionRequest):
    try:
        # Preprocess input and make prediction
        processed_data = preprocess_input(request)
        prediction = model.predict(processed_data)[0]

        # Convert the prediction to a standard Python float
        return {"prediction": float(prediction)}
    except Exception as e:
        print(f"Error during prediction: {e}")  # Log error details to the console
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint 3: Batch predictions
@app.post("/predict_batch")
def predict_batch(request: BatchPredictionRequest):
    try:
        # Preprocess input and make batch predictions
        processed_data = preprocess_input(request.data)
        predictions = model.predict(processed_data)

        # Convert each prediction to a float
        predictions = [float(pred) for pred in predictions]

        return {"predictions": predictions}
    except Exception as e:
        print(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the app with: uvicorn server:app --reload --host 0.0.0.0 --port 8000
