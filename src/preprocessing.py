"""Data loading and feature preparation utilities."""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURES = ["study_hours", "attendance_rate", "assignments_submitted", "exam_score"]
REQUIRED_COLUMNS = ["student_id"] + FEATURES


def load_student_data(path="data/student_performance.csv"):
    data = pd.read_csv(Path(path))
    missing = [col for col in REQUIRED_COLUMNS if col not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data = data[REQUIRED_COLUMNS].drop_duplicates(subset="student_id").copy()
    for col in FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=FEATURES)
    data = data[
        data["study_hours"].between(0, 24)
        & data["attendance_rate"].between(0, 100)
        & data["assignments_submitted"].between(0, 100)
        & data["exam_score"].between(0, 100)
    ].reset_index(drop=True)
    if len(data) < 10:
        raise ValueError("At least 10 valid student records are required.")
    return data


def scale_features(data):
    """Fit scaling on the supplied analysis dataset and return scaled values + scaler."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data[FEATURES])
    return scaled, scaler
