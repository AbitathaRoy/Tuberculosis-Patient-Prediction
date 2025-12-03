import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

def evaluate_predictions(df, actual_col="Actual", pred_col="Prediction"):
    if actual_col not in df.columns:
        raise ValueError(f"Missing actual column '{actual_col}'")
    if pred_col not in df.columns:
        raise ValueError(f"Missing prediction column '{pred_col}'")

    # --- ROBUSTNESS FIX ---
    # Force 'Actual' to match the string format of 'Prediction'
    # If the user uploaded 0/1, map them to "default"/"not-default"
    # This aligns with your preprocess.py logic: 0=default, 1=not-default
    
    # Check if data is numeric (0/1) and map it
    if pd.to_numeric(df[actual_col], errors='coerce').notna().all():
        # Assuming 0 is default based on your preprocess.py
        mapping = {0: "default", 1: "not-default", "0": "default", "1": "not-default"}
        y_true = df[actual_col].map(mapping).fillna(df[actual_col].astype(str))
    else:
        y_true = df[actual_col].astype(str)

    y_pred = df[pred_col].astype(str)
    
    # ----------------------

    # Now calculate metrics using "default" as the positive class
    # (Since "default" is the condition we are trying to detect)
    pos_label = "default"
    
    cm = confusion_matrix(y_true, y_pred, labels=["default", "not-default"])
    cr = classification_report(y_true, y_pred, labels=["default", "not-default"])
    acc = accuracy_score(y_true, y_pred)
    
    try:
        precision = precision_score(y_true, y_pred, pos_label=pos_label)
        recall = recall_score(y_true, y_pred, pos_label=pos_label)
        f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    except ValueError:
        # Fallback if labels don't match exactly (e.g. dataset only has "not-default")
        precision, recall, f1 = 0.0, 0.0, 0.0

    return {
        "confusion_matrix": cm,
        "classification_report": cr,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }