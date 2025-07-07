# boston-housing-mlops
# 🏠 Boston Housing Price Prediction — MLOps Workflow

A complete machine learning pipeline that predicts housing prices using regression models — powered by Python, scikit-learn, and automated with GitHub Actions.

---

## 📦 Project Structure

```text
├── data/                   # BostonHousing.csv dataset  
├── models/                # Saved regression models (.pkl)  
├── plots/                 # Auto-generated visualizations  
├── results/               # Logged metrics and hyperparameters  
├── tests/                 # Unit tests for utility functions  
├── .github/workflows/     # GitHub Actions workflows  
├── utils.py               # Data loading and plotting functions  
├── regression.py          # Baseline regression models  
├── train_regression_models.py # GridSearchCV hyperparameter tuning  
├── requirements.txt       # Python dependencies  
└── README.md              # This file 📄

🚀 Features
✅ Trains Linear Regression, Random Forest, and Gradient Boosting

🔍 Hyperparameter tuning using GridSearchCV

📊 Logs performance metrics (MSE, R²)

🖼️ Auto-generates heatmaps & model comparison plots

🧪 Unit testing with pytest

⚙️ CI/CD automation via GitHub Actions
🛠️ How to Run Locally
Clone this repository:

git clone https://github.com/azij2023/boston-housing-mlops.git
cd boston-housing-mlops
Create a virtual environment:


python -m venv mlops-env
mlops-env\Scripts\activate  # Windows
Install dependencies:


pip install -r requirements.txt
Run baseline model training:


python regression.py
Run hyperparameter tuning (optional):


python train_regression_models.py

📸 Sample Outputs
📁 plots/model_comparison.png — Bar chart of MSE & R² scores

📁 plots/correlation_heatmap.png — Heatmap of feature correlations

📄 results/model_metrics.csv — Model evaluation results

📄 results/hyperparams_log.csv — Best tuned parameters per model
🤖 GitHub Actions CI
This project uses CI workflows to:

Install dependencies

Run regression pipelines

Upload models, plots, and results as artifacts

Run tests with pytest
📚 Dataset
Based on: Boston Housing Dataset (from sklearn)

Features: 13 numeric attributes about housing areas in Boston

Target: Median value of owner-occupied homes (MEDV)

👨‍💻 Author
Azijur Rahaman Data Science & MLOps Enthusiast 📍 India GitHub: @azij2023
