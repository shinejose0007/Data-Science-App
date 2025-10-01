#!/usr/bin/env python3
"""
Mock example: "ingesting" data from SAP (simulated) into a local processing pipeline.
This demonstrates how you'd structure code to fetch OData/REST payloads from SAP BTP,
map fields, and persist to parquet/csv for downstream ML processing.
Note: This is a mock; no real SAP credentials or network calls are made.
"""
import os, json, pandas as pd

def simulate_sap_response(n=50):
    import random, numpy as np
    ids = list(range(1000, 1000+n))
    ages = np.random.randint(22, 60, size=n)
    years = np.clip((ages-20)//2 + np.random.randint(-1,4,size=n), 0, 40)
    education = np.random.choice([1,2,3], size=n, p=[0.4,0.4,0.2])
    salary = (20000 + years*2400 + education*3800 + np.random.normal(0,5000,size=n)).astype(int)
    rows = []
    for i,a,y,e,s in zip(ids, ages, years, education, salary):
        rows.append({
            "EMP_ID": i,
            "AGE": int(a),
            "YEARS_EXP": int(y),
            "EDU_LEVEL": int(e),
            "SALARY_EUR": int(s)
        })
    return rows

def map_sap_to_model(df_sap):
    # Map SAP-style fields to model input
    df = pd.DataFrame(df_sap)
    df_mapped = pd.DataFrame({
        "age": df["AGE"],
        "years_experience": df["YEARS_EXP"],
        "education": df["EDU_LEVEL"],
        "salary": df["SALARY_EUR"]
    })
    return df_mapped

def main():
    # Simulate pull from SAP
    print("Simulating SAP BTP data pull...")
    sap_rows = simulate_sap_response(100)
    df_ingested = map_sap_to_model(sap_rows)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "sap_ingested_mock.csv")
    df_ingested.to_csv(csv_path, index=False)
    print(f"Saved mocked SAP ingestion to {csv_path} (rows={len(df_ingested)})")

if __name__ == "__main__":
    main()