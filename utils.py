import pandas as pd
import numpy as np
import os
def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    # The dataset is structured such that every two rows represent one data point
    # The first row contains the features and the second row contains the target value
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target
    return df

import seaborn as sns
import matplotlib.pyplot as plt

def save_correlation_heatmap(df, path="plots/correlation_heatmap.png"):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
