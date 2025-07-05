import pandas as pd
from utils import load_data

def test_load_data_shape():
    df = load_data()
    assert df.shape[0] == 506
    assert df.shape[1] == 14
    assert "MEDV" in df.columns
