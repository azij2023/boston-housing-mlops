from utils import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os, csv

# Ensure folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Load and split the data
df = load_data()
X = df.drop(columns=["MEDV"])
y = df["MEDV"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define regression models
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Initialize CSV for metric logging
csv_path = os.path.join("results", "model_metrics.csv")
with open(csv_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "MSE", "R2"])

# Train, evaluate, and save each model
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"\n{name} - MSE: {mse:.2f} | R2: {r2:.2f}")


    joblib.dump(model, f"models/{name}.pkl")

    with open(csv_path, mode="a", newline="") as f:
        csv.writer(f).writerow([name, mse, r2])
import pandas as pd
import matplotlib.pyplot as plt

df_metrics = pd.read_csv("results/model_metrics.csv")

# Bar plot for MSE and R²
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(df_metrics["Model"], df_metrics["MSE"], color="skyblue")
axes[0].set_title("MSE Comparison")
axes[0].set_ylabel("Mean Squared Error")

axes[1].bar(df_metrics["Model"], df_metrics["R2"], color="lightgreen")
axes[1].set_title("R² Comparison")
axes[1].set_ylabel("R-squared")

plt.tight_layout()
os.makedirs("plots", exist_ok=True)
plt.savefig("plots/model_comparison.png")
plt.close()