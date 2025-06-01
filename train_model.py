import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv('churn.csv')

# Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Select features
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 
            'gender', 'SeniorCitizen', 'Partner', 
            'Dependents', 'Contract', 'InternetService']

X = df[features]
y = df['Churn']

# One-hot encode categorical features
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save both model and columns (needed for prediction later)
joblib.dump((model, X.columns.tolist()), 'model.pkl')

print("✅ Model trained with more features and saved as model.pkl")
