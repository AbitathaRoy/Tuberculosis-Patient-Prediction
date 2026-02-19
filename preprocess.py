import pandas as pd
import numpy as np
import joblib

# Load scaler trained during model training
scaler = joblib.load("saved_models/tb_default_model_scaler_2.pkl")

FEATURE_ORDER = ['SiteOfDisease', 
                 'Age', 
                 'Microbiologically_Confirmed', 
                 'DiabetesStatus', 
                 'Weight', 
                 'Inter-state/Inter-district enrollment', 
                 'Gender', 
                 'HIV_Status', 
                 'TypeOfCase',
                 'Bank_details',
                 'urban_rural_background'
                 
                 ]

# --- MAPPINGS ---

GENDER_MAP = {"Female": 0, "Male": 1, "Transgender": 2, "Unknown": 3}

SITE_MAP = {"Pulmonary": 1, "Extra Pulmonary": 0, "Unknown": 2}

INTERSTATE_MAP = {"Inter-District": 0, "Inter-State": 1, "Unknown": 2}

HIV_MAP = {"Non-Reactive": 0, "Positive": 1, "Reactive": 2, "Unknown": 3}

TYPEOCASE_MAP = {
    "New": 0,
    "PMDT": 1,
    "Retreatment: Others": 2,
    "Retreatment: Recurrent": 3,
    "Retreatment: Treatment after failure": 4,
    "Retreatment: Treatment after lost to follow up": 5,
    "Unknown": 6
}

MICROBIO_MAP = {
    "No": 0, "Yes": 1,
    0: 0, 1: 1,
    "0": 0, "1": 1,
    "Unknown": 2
}

DIABETES_MAP = {"Diabetic": 0, "Non-diabetic": 1, "Unknown": 2}

URBAN_RURAL_MAP = {"urban": 0, "rural": 1, "Unknown": 2}

BANK_DETAILS_MAP = {"Eligible": 0, "Not Eligible": 1, "Received": 2, "Unknown": 3}


OUTPUT_MAP = {1: "not-default", 0: "default"}
THRESHOLD = 0.60


def _map_series(series, mapping, colname):
    unmapped = set(series.unique()) - set(mapping.keys())
    if unmapped:
        raise ValueError(f"Unmapped values in '{colname}': {unmapped}")
    return series.map(mapping)


def _preprocess_dataframe(df):
    """
    Apply mappings, reorder columns to match scaler.feature_names_in_,
    scale values, return DataFrame.
    """
    df = df.copy()

    df["Gender"] = _map_series(df["Gender"], GENDER_MAP, "Gender")
    df["SiteOfDisease"] = _map_series(df["SiteOfDisease"], SITE_MAP, "SiteOfDisease")
    df["Inter-state/Inter-district enrollment"] = _map_series(
        df["Inter-state/Inter-district enrollment"], INTERSTATE_MAP,
        "Inter-state/Inter-district enrollment")
    df["HIV_Status"] = _map_series(df["HIV_Status"], HIV_MAP, "HIV_Status")
    df["TypeOfCase"] = _map_series(df["TypeOfCase"], TYPEOCASE_MAP, "TypeOfCase")
    df["Microbiologically_Confirmed"] = _map_series(
        df["Microbiologically_Confirmed"], MICROBIO_MAP, "Microbiologically_Confirmed")
    df["DiabetesStatus"] = _map_series(df["DiabetesStatus"], DIABETES_MAP, "DiabetesStatus")

    df["Age"] = pd.to_numeric(df["Age"])
    df["Weight"] = pd.to_numeric(df["Weight"])

    df["urban_rural_background"] = _map_series(
    df["urban_rural_background"], URBAN_RURAL_MAP,
    "urban_rural_background")

    df["Bank_details"] = _map_series(
    df["Bank_details"], BANK_DETAILS_MAP,
    "Bank_details")


    # --- IMPORTANT FIX: Correct scaling order ---
    ordered = scaler.feature_names_in_
    print("Scaler expects:", scaler.feature_names_in_)

    X = df[ordered].astype(float)

    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=ordered)


def preprocess_single(input_dict: dict):
    df = pd.DataFrame([input_dict])
    X = _preprocess_dataframe(df)
    return X.values.astype(np.float32)


def preprocess_batch(df: pd.DataFrame):
    X = _preprocess_dataframe(df)
    return X.values.astype(np.float32)


def decode_output(raw_output):
    arr = np.array(raw_output).ravel()
    if arr.size > 1:
        idx = int(np.argmax(arr))
        return OUTPUT_MAP.get(idx, str(idx))
    else:
        v = float(arr[0])
        label = 1 if v >= THRESHOLD else 0
        return OUTPUT_MAP[label]
