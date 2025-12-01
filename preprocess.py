# preprocess.py
"""
Preprocessing helpers for the TB default model.

- Update/confirm the mapping dictionaries below if your training-time encodings differ.
- Functions:
    - preprocess_single(input_dict) -> numpy array (1, n_features)
    - preprocess_batch(df) -> numpy array (n_samples, n_features)
    - decode_output(raw_output) -> human-readable label ('default' / 'not-default')
"""

import pandas as pd
import numpy as np

# ----------------------------
# FEATURE ORDER (must match model input order used during training)
# Update this list if your model expects a different order.
FEATURE_ORDER = [
    "Age",
    "DiabetesStatus",
    "Microbiologically_Confirmed",
    "TypeOfCase",
    "SiteOfDisease",
    "Gender",
    "Inter-state/Inter-district enrollment",
    "Weight",
    "HIV_Status"
]

# ----------------------------
# MAPPINGS

# Gender
GENDER_MAP = {
    "Female": 0,
    "Male": 1,
    "Transgender": 2
}

# SiteOfDisease
SITE_MAP = {
    "Pulmonary": 1,
    "Extra Pulmonary": 0
}

# Inter-state/Inter-district enrollment
INTERSTATE_MAP = {
    "Inter-District": 0,
    "Inter-State": 1
}

# HIV_Status
HIV_MAP = {
    "Non-Reactive": 0,
    "Unknown": 3,
    "Positive": 1,
    "Reactive": 2
}

# TypeOfCase
TYPEOCASE_MAP = {
    "Retreatment: Recurrent": 3,
    "New": 0,
    "Retreatment: Others": 2,
    "PMDT": 1,
    "Retreatment: Treatment after failure": 4,
    "Retreatment: Treatment after lost to follow up": 5
}

# Microbiologically_Confirmed
MICROBIO_MAP = {
    "No": 0,
    "Yes": 1
}

# DiabetesStatus
DIABETES_MAP = {
    "Non-diabetic": 1,
    # "Unknown": 0,
    "Diabetic": 0    # <-- If your training used only 0/1, change this to 0 or 1 as required.
}

# OUTPUT
OUTPUT_MAP = {
    1: "not-default",
    0: "default"
}

# Probability threshold for scalar model outputs
THRESHOLD = 0.61

# ----------------------------
# Helper functions
# ----------------------------
def _map_series(series: pd.Series, mapping: dict, colname: str):
    """Map values in a series, raise if unmapped values found."""
    unmapped = set(series.unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"Unmapped values in column '{colname}': {unmapped}. Update mapping dict.")
    return series.map(mapping)

def preprocess_single(input_dict: dict):
    """
    input_dict example:
    {
      "Gender": "Female",
      "Age": 32,
      "Weight": 55.0,
      "HIV_Status": "Non-Reactive",
      "DiabetesStatus": "Non-diabetic",
      "Microbiologically_Confirmed": "Yes",
      "TypeOfCase": "New",
      "SiteOfDisease": "Pulmonary",
      "Inter-state/Inter-district enrollment": "Inter-District"
    }
    Returns: numpy array shaped (1, n_features)
    """
    df = pd.DataFrame([input_dict])
    X = _preprocess_dataframe(df)
    return X.values.astype(np.float32)

def preprocess_batch(df: pd.DataFrame):
    """
    df: DataFrame with columns named as FEATURE_ORDER (or a superset).
    Returns: numpy array shaped (n_samples, n_features)
    """
    X = _preprocess_dataframe(df.copy())
    return X.values.astype(np.float32)

def _preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply mappings and ensure column order. Raises errors for missing columns."""
    # Check required columns
    missing = [c for c in FEATURE_ORDER if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input data: {missing}")

    # Map categorical columns
    df["Gender"] = _map_series(df["Gender"], GENDER_MAP, "Gender")
    df["SiteOfDisease"] = _map_series(df["SiteOfDisease"], SITE_MAP, "SiteOfDisease")
    df["Inter-state/Inter-district enrollment"] = _map_series(
        df["Inter-state/Inter-district enrollment"], INTERSTATE_MAP, "Inter-state/Inter-district enrollment"
    )
    df["HIV_Status"] = _map_series(df["HIV_Status"], HIV_MAP, "HIV_Status")
    df["TypeOfCase"] = _map_series(df["TypeOfCase"], TYPEOCASE_MAP, "TypeOfCase")
    df["Microbiologically_Confirmed"] = _map_series(df["Microbiologically_Confirmed"], MICROBIO_MAP, "Microbiologically_Confirmed")
    df["DiabetesStatus"] = _map_series(df["DiabetesStatus"], DIABETES_MAP, "DiabetesStatus")

    # Ensure numeric columns for Age and Weight
    df["Age"] = pd.to_numeric(df["Age"], errors="raise")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="raise")

    # Reorder
    X = df[FEATURE_ORDER]
    return X

def decode_output(raw_output):
    """
    Accepts:
      - a scalar probability (e.g., 0.78)
      - a 1D array-like probability vector (e.g., [0.2, 0.8])
      - a 1D numeric label (e.g., 1 or 0)
    Returns human-readable label (string).
    """
    # If numpy array or list
    if hasattr(raw_output, "shape") and len(np.array(raw_output).shape) > 0:
        arr = np.array(raw_output)
        # If shape is (n_classes,) or (n,):
        if arr.size > 1:
            # assume model returned class-probabilities or one-hot -> use argmax
            idx = int(np.argmax(arr))
            # if training used multi-output with mapping different from index, you'll need to adjust
            # We try to map idx directly to OUTPUT_MAP; if not found, return idx as string
            return OUTPUT_MAP.get(idx, str(idx))
        else:
            # single value
            val = float(arr.ravel()[0])
            label_numeric = 1 if val >= THRESHOLD else 0
            return OUTPUT_MAP.get(label_numeric, str(label_numeric))
    else:
        # scalar
        try:
            val = float(raw_output)
            label_numeric = 1 if val >= THRESHOLD else 0
            return OUTPUT_MAP.get(label_numeric, str(label_numeric))
        except Exception:
            # fallback: try interpret as int label
            try:
                label_numeric = int(raw_output)
                return OUTPUT_MAP.get(label_numeric, str(label_numeric))
            except Exception:
                return str(raw_output)