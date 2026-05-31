# Ce script sert à voir si nos données splitées ont des distributions qui se ressemblent.
import pandas as pd
import matplotlib.pyplot as plt
import os


os.makedirs("metrics", exist_ok=True)

df = pd.read_csv("data/raw/raw.csv")

y = df["silica_concentrate"]

y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]
y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

print("GLOBAL")
print(y.describe())

print("\nTRAIN")
print(y_train.describe())

print("\nTEST")
print(y_test.describe())

plt.hist(y, bins=30, alpha=0.5, label="all")
plt.hist(y_train, bins=30, alpha=0.5, label="train")
plt.hist(y_test, bins=30, alpha=0.5, label="test")

plt.legend()
plt.title("Distribution silica_concentrate")

plt.savefig("metrics/y_distribution.png")
print("Saved: metrics/y_distribution.png")