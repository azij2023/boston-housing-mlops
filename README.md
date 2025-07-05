🏠 Boston Housing Price Prediction — MLOps Workflow
A complete machine learning pipeline that predicts housing prices using regression models — powered by Python, scikit-learn, and automated with GitHub Actions.

📦 Project Structure
text
├── data/                   # BostonHousing.csv dataset
├── models/                # Saved regression models (.pkl)
├── plots/                 # Auto-generated visualizations (heatmaps, comparisons)
├── results/               # Logged metrics and hyperparameters
├── tests/                 # Unit tests for utility functions
├── .github/workflows/     # GitHub Actions workflows (CI/CD)
├── utils.py               # Modular data loading and plotting functions
├── regression.py          # Baseline regression model training
├── train_regression_models.py # GridSearchCV-based hyperparameter tuning
├── requirements.txt       # Python dependencies
└── README.md              # This file 📄
🚀 Features
✅ Trains Linear Regression, Random Forest, and Gradient Boosting

🔍 Implements GridSearchCV for hyperparameter optimization

📊 Logs metrics (MSE, R²) to CSV

🖼️ Auto-generates model comparison plots and heatmaps

🧪 Includes pytest-based unit tests

⚙️ Continuous integration with GitHub Actions

📁 Clean, modular folder structure for reproducibility

🛠️ How to Run Locally
Clone the repo:

bash
git clone https://github.com/azij2023/boston-housing-mlops.git
cd boston-housing-mlops
Create virtual environment and install dependencies:

bash
python -m venv mlops-env
mlops-env\Scripts\activate  # Windows
pip install -r requirements.txt
Run the baseline model (regression):

bash
python regression.py
Run hyperparameter tuning (optional):

bash
python train_regression_models.py
📸 Sample Outputs
Plots and results are auto-saved here:

📊 plots/model_comparison.png

🔥 plots/correlation_heatmap.png

📁 results/model_metrics.csv

🧠 results/hyperparams_log.csv

🤖 CI/CD with GitHub Actions
Automatically installs dependencies and runs training/testing

View latest build → Actions tab

📚 Dataset
Source: sklearn.datasets.load_boston() (deprecated, local CSV used)

Features: 13 numeric attributes about housing areas in Boston

👨‍💻 Author
Azijur Rahaman Data Science & MLOps Enthusiast 📍 India