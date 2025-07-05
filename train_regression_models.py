import os

# Make sure model output directory exists
os.makedirs("models", exist_ok=True)



import pandas as pd
df = pd.read_csv("data/BostonHousing.csv")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load dataset
data_path = os.path.join("data", "BostonHousing.csv")
df = pd.read_csv(data_path)

# Features and target
X = df.drop(columns=["medv"])  # 'medv' is the target: Median value of owner-occupied homes
y = df["medv"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models to train
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

# Train, evaluate, and save models
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"\n{name}")
    print(f"  MSE: {mse:.2f}")
    print(f"  R²: {r2:.2f}")

    # Save model
    model_path = os.path.join("models", f"{name}.pkl")
    joblib.dump(model, model_path)
from utils import load_data, save_correlation_heatmap
save_correlation_heatmap(df)

