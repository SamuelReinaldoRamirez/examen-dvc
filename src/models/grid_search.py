import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
import os

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv")

rf = RandomForestRegressor(random_state=42)

param_grid = {
    "n_estimators": [100, 1000],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Best params:")
print(grid_search.best_params_)

os.makedirs("models", exist_ok=True)

joblib.dump(
    grid_search.best_params_,
    "models/best_params.pkl"
)

print("Best params saved ✔")