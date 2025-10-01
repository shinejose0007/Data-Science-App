#!/usr/bin/env python3
import argparse, os, joblib
import numpy as np

def load_model_and_scaler():
    base = os.path.join(os.path.dirname(__file__), "..")
    model_path = os.path.join(base, "models", "model.joblib")
    scaler_path = os.path.join(base, "models", "scaler.joblib")
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return clf, scaler

def main():
    parser = argparse.ArgumentParser(description="Predict with trained model (synthetic demo)")
    parser.add_argument("--age", type=int, required=True)
    parser.add_argument("--years_experience", type=int, required=True)
    parser.add_argument("--education", type=int, required=True, help="1=HS,2=BSc,3=Master+")
    parser.add_argument("--salary", type=float, required=True)
    args = parser.parse_args()
    clf, scaler = load_model_and_scaler()
    X = [[args.age, args.years_experience, args.education, args.salary]]
    Xs = scaler.transform(X)
    pred = clf.predict(Xs)[0]
    proba = clf.predict_proba(Xs)[0,1] if hasattr(clf, "predict_proba") else None
    print("Prediction (1 means high-value / positive class):", int(pred))
    if proba is not None:
        print(f"Predicted probability: {proba:.3f}")

if __name__ == "__main__":
    main()