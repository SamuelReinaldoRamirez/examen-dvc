import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

scaler = StandardScaler()

# on ne garde pas la colonne date qui est une string
X_train = X_train.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)

os.makedirs("data/processed", exist_ok=True)

X_train_scaled.to_csv(
    "data/processed/X_train_scaled.csv",
    index=False
)

X_test_scaled.to_csv(
    "data/processed/X_test_scaled.csv",
    index=False
)

print("Scaling completed")