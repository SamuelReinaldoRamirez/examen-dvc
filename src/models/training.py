import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

best_params = joblib.load("models/best_params.pkl")

print("Best parameters:", best_params)

mlflow.set_experiment("random_forest_experiment")

with mlflow.start_run():

    model = RandomForestRegressor(
        random_state=42,
        **best_params
    )

    model.fit(X_train, y_train.values.ravel())

    mlflow.log_params(best_params)
    mlflow.log_param("model_type", "RandomForestRegressor")

    os.makedirs("models", exist_ok=True)

    model_path = "models/trained_model.pkl"
    joblib.dump(model, model_path)

    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RandomForestRegressor"
    )

    print(f"Model saved at {model_path}")