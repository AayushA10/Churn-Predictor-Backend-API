📚 Customer Churn Predictor — AI/ML Backend API
This project is a backend-only AI/ML API that predicts whether a telecom customer is likely to churn or stay based on real customer data. No frontend — just pure FastAPI + Machine Learning magic.

🚀 Tech Stack
FastAPI — for building high-performance APIs

Scikit-learn — for machine learning

Pandas — for data manipulation

Joblib — for model serialization

Real-world Dataset — Telco Customer Churn Dataset (7K+ samples)

📦 Project Structure
bash
Copy
Edit
customer-churn-predictor/
├── app.py                 # FastAPI application (API logic)
├── train_model.py          # Model training script
├── churn.csv               # Customer churn dataset (real-world)
├── model.pkl               # Trained ML model (Random Forest + Features)
├── predictions_log.csv     # Auto-generated logs of all API predictions
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
🔥 Features
Train an ML model (Random Forest Classifier) on real customer churn data

POST API to predict customer churn

Auto-log predictions to a CSV file (predictions_log.csv)

Deployed locally via FastAPI Swagger UI for easy testing

Uses important features like:

Tenure

Monthly Charges

Total Charges

Gender

Senior Citizen

Partner Status

Dependents

Contract Type

Internet Service Type

🛠️ How to Run Locally
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/your-username/customer-churn-predictor.git
cd customer-churn-predictor
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Train the model
bash
Copy
Edit
python train_model.py
4. Start FastAPI Server
bash
Copy
Edit
uvicorn app:app --reload --port 8000
5. Access API
Open your browser and go to:

arduino
Copy
Edit
http://127.0.0.1:8000/docs
You’ll see Swagger UI to test your API.

📮 API Example (POST /predict)
Request Body:
json
Copy
Edit
{
  "tenure": 12,
  "monthly_charges": 75.5,
  "total_charges": 700.5,
  "gender": "Female",
  "senior_citizen": 0,
  "partner": "Yes",
  "dependents": "No",
  "contract": "Month-to-month",
  "internet_service": "Fiber optic"
}
Example Response:
json
Copy
Edit
{
  "prediction": "Not Churn"
}
📊 Prediction Logs
Every time you make a prediction, a new row will be appended in:

Copy
Edit
predictions_log.csv
It contains:

Customer features

Model prediction

🎯 Future Improvements
Deploy the API publicly on Render/Heroku

Add Streamlit Dashboard to visualize churn predictions

Connect to PostgreSQL for better logging

Model Explainability using SHAP / LIME