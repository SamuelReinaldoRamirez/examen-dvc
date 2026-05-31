import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

best_params = joblib.load("models/best_params.pkl")

print("Best parameters:", best_params)

model = RandomForestRegressor(
    random_state=42,
    **best_params
)

model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)

model_path = "models/trained_model.joblib"
joblib.dump(model, model_path)

print(f"Model saved at {model_path}")