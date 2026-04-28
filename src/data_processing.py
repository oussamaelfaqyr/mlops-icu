import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path


def load_data(base_path):
    """
    Loads necessary CSV files from the MIMIC-IV demo directory.
    """
    base_path = Path(base_path)
    hosp_path = base_path / "hosp"
    icu_path = base_path / "icu"

    print("Loading data...")
    patients = pd.read_csv(hosp_path / "patients.csv.gz", compression="gzip")
    admissions = pd.read_csv(hosp_path / "admissions.csv.gz", compression="gzip")
    icustays = pd.read_csv(icu_path / "icustays.csv.gz", compression="gzip")

    # Selecting core vitals from chartevents
    # In MIMIC-IV, typical itemids for vitals:
    # Heart Rate: 220045, SpO2: 220277, Resp Rate: 220210, Temp: 223762, MAP: 220052
    vital_itemids = [220045, 220277, 220210, 223762, 220052]

    # We load chartevents in chunks if it's large, but demo is small.
    chartevents = pd.read_csv(icu_path / "chartevents.csv.gz", compression="gzip")
    chartevents = chartevents[chartevents["itemid"].isin(vital_itemids)]

    return {
        "patients": patients,
        "admissions": admissions,
        "icustays": icustays,
        "chartevents": chartevents,
    }


def clean_data(data):
    """
    Basic cleaning and merging.
    """
    patients = data["patients"]
    admissions = data["admissions"]
    icustays = data["icustays"]
    chartevents = data["chartevents"]

    # Merge icustays with patients for age/gender
    df_stays = icustays.merge(
        patients[["subject_id", "gender", "anchor_age"]], on="subject_id", how="left"
    )

    # Convert timestamps
    df_stays["intime"] = pd.to_datetime(df_stays["intime"])
    df_stays["outtime"] = pd.to_datetime(df_stays["outtime"])
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])

    # Map itemids to names
    item_map = {
        220045: "heart_rate",
        220277: "spo2",
        220210: "resp_rate",
        223762: "temp",
        220052: "mean_bp",
    }
    chartevents["vital_name"] = chartevents["itemid"].map(item_map)

    # Filter only vitals that fall within ICU stay window
    chartevents = chartevents.merge(
        df_stays[["stay_id", "intime", "outtime"]], on="stay_id", how="left"
    )
    chartevents = chartevents[
        (chartevents["charttime"] >= chartevents["intime"])
        & (chartevents["charttime"] <= chartevents["outtime"])
    ]

    return df_stays, chartevents


if __name__ == "__main__":
    # Priority: 1. Env Var, 2. Mock Folder (CI), 3. Hardcoded Local (User)
    env_path = os.environ.get("MIMIC_DATA_PATH")
    local_hardcoded = (
        r"C:\Users\21270\Desktop\mlops\physionet.org\files\mimic-iv-demo\2.2"
    )
    mock_path = "mock_data"

    if env_path and os.path.exists(env_path):
        BASE_DATA_PATH = env_path
    elif os.path.exists(mock_path):
        BASE_DATA_PATH = mock_path
    elif os.path.exists(local_hardcoded):
        BASE_DATA_PATH = local_hardcoded
    else:
        print(f"[ERROR] No data found at {local_hardcoded} or {mock_path}")
        sys.exit(1)

    print(f"Using data from: {BASE_DATA_PATH}")
    data = load_data(BASE_DATA_PATH)
    df_stays, chartevents = clean_data(data)
    print(f"Loaded {len(df_stays)} stays and {len(chartevents)} vital events.")

    # Save processed for feature engineering
    os.makedirs("data", exist_ok=True)
    df_stays.to_csv("data/stays_cleaned.csv", index=False)
    chartevents.to_csv("data/vitals_cleaned.csv", index=False)
