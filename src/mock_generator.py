import pandas as pd
import numpy as np
import os
import gzip
from datetime import datetime, timedelta


def create_mock_mimic():
    """Creates a tiny synthetic MIMIC-IV structure for CI/CD testing."""
    os.makedirs("mock_data/hosp", exist_ok=True)
    os.makedirs("mock_data/icu", exist_ok=True)

    # Patients
    df_pts = pd.DataFrame(
        {"subject_id": [1001, 1002], "gender": ["M", "F"], "anchor_age": [65, 72]}
    )
    df_pts.to_csv("mock_data/hosp/patients.csv.gz", compression="gzip", index=False)

    # Admissions
    df_adm = pd.DataFrame(
        {
            "subject_id": [1001, 1002],
            "hadm_id": [2001, 2002],
            "admittime": ["2170-01-01 00:00:00", "2170-02-01 00:00:00"],
        }
    )
    df_adm.to_csv("mock_data/hosp/admissions.csv.gz", compression="gzip", index=False)

    # ICU Stays
    df_stays = pd.DataFrame(
        {
            "subject_id": [1001, 1002],
            "hadm_id": [2001, 2002],
            "stay_id": [3001, 3002],
            "intime": ["2170-01-01 10:00:00", "2170-02-01 10:00:00"],
            "outtime": ["2170-01-05 10:00:00", "2170-02-05 10:00:00"],
        }
    )
    df_stays.to_csv("mock_data/icu/icustays.csv.gz", compression="gzip", index=False)

    # Chartevents (Vital Signs)
    vitals = []
    # Generate 5 days of hourly vitals for 2 patients
    for sid, stay_id, start in [
        (1001, 3001, "2170-01-01 10:00:00"),
        (1002, 3002, "2170-02-01 10:00:00"),
    ]:
        base_time = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        for h in range(120):
            t = base_time + timedelta(hours=h)
            # Add the 5 core vitals
            vitals.append([sid, stay_id, t, 220045, 80 + np.random.normal(0, 5)])  # HR
            vitals.append(
                [sid, stay_id, t, 220277, 98 - np.random.uniform(0, 2)]
            )  # SpO2
            vitals.append([sid, stay_id, t, 220210, 16 + np.random.normal(0, 2)])  # RR
            vitals.append(
                [sid, stay_id, t, 223762, 37 + np.random.normal(0, 0.5)]
            )  # Temp
            vitals.append(
                [sid, stay_id, t, 220052, 90 + np.random.normal(0, 10)]
            )  # MAP

    df_vitals = pd.DataFrame(
        vitals, columns=["subject_id", "stay_id", "charttime", "itemid", "valuenum"]
    )
    df_vitals.to_csv(
        "mock_data/icu/chartevents.csv.gz", compression="gzip", index=False
    )
    print("Mock MIMIC-IV data generated in /mock_data")


if __name__ == "__main__":
    create_mock_mimic()
