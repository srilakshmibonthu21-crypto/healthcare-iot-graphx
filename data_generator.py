"""
Synthetic Heterogeneous Healthcare IoT Data Generator
Simulates real-time physiological telemetry, patient EHR profiles, and clinical risk scores (NEWS-2).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


def calculate_news2(hr: float, spo2: float, sbp: float, rr: float, temp: float) -> int:
    """
    Computes standard National Early Warning Score 2 (NEWS-2).
    Score determines clinical deterioration risk (0-4: Low, 5-6: Medium, 7+: High/Critical).
    """
    score = 0
    # Respiration Rate
    if rr <= 8 or rr >= 25:
        score += 3
    elif 21 <= rr <= 24:
        score += 2
    elif 9 <= rr <= 11:
        score += 1

    # SpO2
    if spo2 <= 91:
        score += 3
    elif 92 <= spo2 <= 93:
        score += 2
    elif 94 <= spo2 <= 95:
        score += 1

    # Systolic BP
    if sbp <= 90 or sbp >= 220:
        score += 3
    elif 91 <= sbp <= 100:
        score += 2
    elif 101 <= sbp <= 110:
        score += 1

    # Heart Rate
    if hr <= 40 or hr >= 131:
        score += 3
    elif 111 <= hr <= 130:
        score += 2
    elif 41 <= hr <= 50 or 91 <= hr <= 110:
        score += 1

    # Temperature
    if temp <= 35.0:
        score += 3
    elif temp >= 39.1:
        score += 2
    elif 35.1 <= temp <= 36.0 or 38.1 <= temp <= 39.0:
        score += 1

    return score


def generate_healthcare_iot_dataset(num_patients: int = 200, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates heterogeneous patient monitoring dataset with 5 distinct clinical cohorts:
    1. Stable (Routine monitoring)
    2. Sepsis / Critical Shock (Hypotension, Tachycardia, Fever/Hypothermia, High NEWS-2)
    3. Respiratory Distress (Tachypnea, Hypoxia)
    4. Hypertensive Crisis (Extreme SBP)
    5. Sensor Artifact / Fault (Disconnected leads, invalid signals)
    """
    np.random.seed(random_seed)

    cohort_distribution = [0.55, 0.15, 0.15, 0.10, 0.05]
    cohorts = np.random.choice(
        ["Stable", "Sepsis_Risk", "Respiratory_Distress", "Hypertensive_Crisis", "Sensor_Fault"],
        size=num_patients,
        p=cohort_distribution
    )

    records = []
    conditions_pool = ["Diabetes_Type2", "COPD", "Congestive_Heart_Failure", "Chronic_Kidney_Disease", "None"]

    for i in range(num_patients):
        pid = 1000 + i
        cohort = cohorts[i]
        age = int(np.clip(np.random.normal(58, 16), 18, 92))
        gender = np.random.choice(["M", "F"])
        comorbidity = np.random.choice(conditions_pool, p=[0.25, 0.20, 0.20, 0.15, 0.20])
        ward = np.random.choice(["ICU_A", "ICU_B", "StepDown_Ward", "Cardio_Care"])

        if cohort == "Stable":
            hr = float(np.clip(np.random.normal(72, 8), 58, 88))
            spo2 = float(np.clip(np.random.normal(98, 1.2), 95, 100))
            sbp = float(np.clip(np.random.normal(120, 10), 105, 138))
            rr = float(np.clip(np.random.normal(15, 2), 12, 19))
            temp = float(np.clip(np.random.normal(36.8, 0.3), 36.4, 37.3))

        elif cohort == "Sepsis_Risk":
            # Tachycardia, severe hypotension, high temp, low SpO2
            hr = float(np.clip(np.random.normal(124, 12), 100, 160))
            spo2 = float(np.clip(np.random.normal(91, 3), 82, 95))
            sbp = float(np.clip(np.random.normal(82, 8), 65, 95))
            rr = float(np.clip(np.random.normal(26, 3), 22, 34))
            temp = float(np.clip(np.random.normal(39.2, 0.6), 38.2, 40.5))

        elif cohort == "Respiratory_Distress":
            # Tachypneic, severe hypoxia, mild tachycardia
            hr = float(np.clip(np.random.normal(108, 10), 92, 130))
            spo2 = float(np.clip(np.random.normal(86, 4), 75, 92))
            sbp = float(np.clip(np.random.normal(132, 12), 110, 155))
            rr = float(np.clip(np.random.normal(30, 4), 24, 40))
            temp = float(np.clip(np.random.normal(37.4, 0.5), 36.7, 38.2))

        elif cohort == "Hypertensive_Crisis":
            # Extremely elevated blood pressure
            hr = float(np.clip(np.random.normal(95, 12), 75, 120))
            spo2 = float(np.clip(np.random.normal(96, 2), 92, 99))
            sbp = float(np.clip(np.random.normal(195, 15), 170, 235))
            rr = float(np.clip(np.random.normal(18, 3), 14, 24))
            temp = float(np.clip(np.random.normal(37.0, 0.4), 36.5, 37.6))

        else:  # Sensor_Fault
            # Impossible or erratic values (e.g. detached sensor)
            hr = float(np.random.choice([0.0, 245.0, 15.0]))
            spo2 = float(np.random.choice([0.0, 35.0, 100.0]))
            sbp = float(np.random.choice([0.0, 310.0, 40.0]))
            rr = float(np.random.choice([0.0, 65.0]))
            temp = float(np.random.choice([25.0, 44.0]))

        news2 = calculate_news2(hr, spo2, sbp, rr, temp)

        records.append({
            "patient_id": pid,
            "age": age,
            "gender": gender,
            "ward": ward,
            "comorbidity": comorbidity,
            "heart_rate": round(hr, 1),
            "spo2": round(spo2, 1),
            "systolic_bp": round(sbp, 1),
            "resp_rate": round(rr, 1),
            "temperature": round(temp, 1),
            "news2_score": news2,
            "true_cohort": cohort
        })

    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_healthcare_iot_dataset(num_patients=200)
    print("Dataset generated successfully:")
    print(df.head())
    print("\nCohort distribution:")
    print(df["true_cohort"].value_counts())
