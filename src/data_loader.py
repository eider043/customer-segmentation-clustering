import pandas as pd
import numpy as np

def load_data(path="../data/mall_customers.csv"):
    df = pd.read_csv(path)
    df.columns = ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore"]
    df["Gender_encoded"] = (df["Gender"] == "Male").astype(int)
    print(f"Dataset cargado: {df.shape}")
    print(df.dtypes)
    return df

def feature_matrix(df, features=["Age", "AnnualIncome", "SpendingScore"]):
    return df[features].copy()