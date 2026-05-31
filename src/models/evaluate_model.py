import pandas as pd
import joblib
import os
import json
from sklearn.metrics import mean_squared_error, r2_score

# 📥 Load test data
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

# 📥 Load trained model
model = joblib.load("models/trained_model.joblib")

print("Model loaded ✔")

# 🔮 Predictions
y_pred = model.predict(X_test)

# 📊 Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R2: {r2}")

os.makedirs("metrics", exist_ok=True)

metrics = {
    "mse": mse,
    "r2": r2
}

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Metrics saved ✔")

df_pred = X_test.copy()
df_pred["y_true"] = y_test
df_pred["y_pred"] = y_pred

df_pred.to_csv("data/processed/predictions.csv", index=False)

print("Predictions saved ✔")