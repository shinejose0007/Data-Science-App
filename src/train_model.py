#!/usr/bin/env python3
import argparse, os, joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from src.data_processing import load_data, preprocess

def main():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_customers.csv")
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")

    df = load_data(data_path)
    X, y, scaler = preprocess(df, fit_scaler=True, scaler_path=scaler_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, preds))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"Saved model to {model_path}")
    print(f"Saved scaler to {scaler_path}")

if __name__ == "__main__":
    main()