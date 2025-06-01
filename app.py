from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load model and columns
model, model_columns = joblib.load('model.pkl')

# FastAPI app
app = FastAPI()

# Request body
class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    contract: str
    internet_service: str

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Create input DataFrame
    input_data = pd.DataFrame([{
        'tenure': data.tenure,
        'MonthlyCharges': data.monthly_charges,
        'TotalCharges': data.total_charges,
        'gender': data.gender,
        'SeniorCitizen': data.senior_citizen,
        'Partner': data.partner,
        'Dependents': data.dependents,
        'Contract': data.contract,
        'InternetService': data.internet_service
    }])

    # One-hot encode
    input_data_encoded = pd.get_dummies(input_data)
    input_data_encoded = input_data_encoded.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(input_data_encoded)[0]
    result = "Churn" if prediction == 1 else "Not Churn"

    # Save input + prediction to CSV
    log_data = input_data.copy()
    log_data['prediction'] = result

    # If file doesn't exist, create it with header
    if not os.path.isfile('predictions_log.csv'):
        log_data.to_csv('predictions_log.csv', index=False, mode='w')
    else:
        log_data.to_csv('predictions_log.csv', index=False, mode='a', header=False)

    return {"prediction": result}
