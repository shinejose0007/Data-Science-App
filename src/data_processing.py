import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, fit_scaler=True, scaler_path=None):
    X = df[["age","years_experience","education","salary"]].copy()
    y = df["target_highvalue"] if "target_highvalue" in df.columns else None
    scaler = StandardScaler()
    if fit_scaler:
        X_scaled = scaler.fit_transform(X)
        if scaler_path:
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        X_scaled = scaler.transform(X)
    return X_scaled, y, scaler