# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model

# --- IMPORTANT ---
if 'df_out' not in st.session_state:
    st.session_state.df_out = None

st.set_page_config(page_title="Tuberculosis Default Prediction", layout="centered")

debug_mode = False

# The custom tokenizer MUST be defined before loading the preprocessor
def split_comma_tags(text):
    if pd.isna(text):
        return []
    return [item.strip() for item in text.split(',') if item.strip()]

@st.cache_resource
def load_my_pipeline():
    # Load exactly based on your exported file names
    prep = joblib.load('saved_models/tuberculosis_data_preprocessor.joblib')
    xgb_m = xgb.XGBClassifier()
    xgb_m.load_model('saved_models/tuberculosis_xgb_model.json') 
    nn_m = load_model('saved_models/tuberculosis_nn_model.keras')
    meta_m = joblib.load('saved_models/tuberculosis_meta_model.joblib')
    return prep, xgb_m, nn_m, meta_m

preprocessor, xgb_model, nn_model, meta_model = load_my_pipeline()

# The mathematically optimal threshold we found
OPTIMAL_THRESHOLD = 0.4313 

st.title("TB Default Prediction")
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------- SINGLE ----------------
with tab1:
    st.subheader("Single Patient Prediction")

    name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")

    gender = st.selectbox("Gender", ["Female", "Male", "Transgender"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0)
    hiv = st.selectbox("HIV Status", ["Negative", "Positive", "Unknown"])
    diabetes = st.selectbox("Diabetes Status", ["Non-diabetic", "Diabetic", "Unknown"])
    micro = st.selectbox("Microbiologically Confirmed?", ["0", "1"])
    typeofcase = st.selectbox("Type of TB Case", [
        "New", "PMDT", "Retreatment: Others", "Retreatment: Recurrent",
        "Retreatment: Treatment after failure",
        "Retreatment: Treatment after lost to follow up",
        "Unknown"
    ])
    site = st.selectbox("Site of Disease", ["Pulmonary", "Extra Pulmonary", "Unknown"])
    interstate = st.selectbox("Inter-state / Inter-district enrollment", ["Inter-District", "Inter-State"])
    urban_rural = st.selectbox("Urban / Rural Background", ["rural", "urban"])
    bank_details = st.selectbox("Bank Details Added", ["Yes", "No", "Unknown"])
    
    key_pop_options = [
        'Anti-TNF treatment', 'Bronchial Asthma', 'COPD', 'COVID recovered patients', 
        'Cancer', 'Contact of Known TB Patients', 'Diabetes', 'Dialysis', 
        'Elderly (age >60 years)', 'H/o Adult BCG Vaccination', 'Health Care Worker', 
        'Hypertensive', 'Lactating mother', 'Liver Impairment', 'Migrant', 'Miner', 
        'Not Applicable', 'Other', 'Person exposed to indoor air pollution', 
        'Substance abuse (alcoholic/ intravenous drug users)', 'Tobacco', 
        'Undernourished / Malnourished (BMI <18.5 kg/m2)', 'Urban Slum'
    ]
    keypopulation = st.multiselect("Key Population (Select all that apply)", key_pop_options, default=["Not Applicable"])


    if st.button("Predict for this patient"):
        input_df = pd.DataFrame([{
            "Age": age,
            "Weight": weight,
            "Gender": gender,
            "HIV_Status": hiv,
            "DiabetesStatus": diabetes,
            "Microbiologically_Confirmed": micro,
            "TypeOfCase": typeofcase,
            "SiteOfDisease": site,
            "Inter-state/Inter-district enrollment": interstate,
            "urban_rural_background": urban_rural,
            "BankDetailsAdded": bank_details,
            "KeyPopulation": ", ".join(keypopulation) # Combine for custom tokenizer
        }])

        with st.spinner('Running Hybrid AI diagnosis...'):
            X_processed = preprocessor.transform(input_df)
            X_dense = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
            
            xgb_prob = xgb_model.predict_proba(X_processed)[0][1] 
            nn_prob = nn_model.predict(X_dense, verbose=0)[0][0]
            
            meta_features = np.array([[xgb_prob, nn_prob]])
            final_risk_prob = meta_model.predict_proba(meta_features)[0][1]
            
            if final_risk_prob >= OPTIMAL_THRESHOLD:
                label = "default"
            else:
                label = "not-default"

        st.success(f"Prediction for {name} (ID: {patient_id}) → {label}")
        st.write(f"**Overall Risk Score:** {final_risk_prob:.1%}")
        st.caption(f"(XGBoost Prob: {xgb_prob:.1%} | Neural Network Prob: {nn_prob:.1%})")

# ------------Manual Batch--------------

with tab2:
    st.subheader("Batch Patient Prediction")

    if "batch_df" not in st.session_state:
        st.session_state.batch_df = pd.DataFrame({
            "Select": [False],  
            "Name": [""],
            "Patient_ID": [""],
            "Gender": ["Female"],
            "Age": [30],
            "Weight": [50.0],
            "HIV_Status": ["Unknown"],
            "DiabetesStatus": ["Unknown"],
            "Microbiologically_Confirmed": ["0"],
            "TypeOfCase": ["New"],
            "SiteOfDisease": ["Pulmonary"],
            "Inter-state/Inter-district enrollment": ["Unknown"],
            "urban_rural_background": ["urban"],
            "BankDetailsAdded": ["Yes"],
            "KeyPopulation": ["Not Applicable"]
        })

    edited_df = st.data_editor(
        st.session_state.batch_df,
        num_rows="fixed",   
        use_container_width=True,
        hide_index=False,   
        key="batch_editor",
        column_config={
            "Select": st.column_config.CheckboxColumn("Select", help="Select rows to delete", default=False),
            "Name": st.column_config.TextColumn(default=""),
            "Patient_ID": st.column_config.TextColumn(default=""),
            "Gender": st.column_config.SelectboxColumn(options=["Female", "Male", "Transgender"], default="Female"),
            "Age": st.column_config.NumberColumn(min_value=0, max_value=120, default=30),
            "Weight": st.column_config.NumberColumn(min_value=1.0, max_value=250.0, default=50.0),
            "HIV_Status": st.column_config.SelectboxColumn(options=["Negative", "Positive", "Unknown"], default="Unknown"),
            "DiabetesStatus": st.column_config.SelectboxColumn(options=["Non-diabetic", "Diabetic", "Unknown"], default="Unknown"),
            "Microbiologically_Confirmed": st.column_config.SelectboxColumn(options=["0", "1"], default="0"),
            "TypeOfCase": st.column_config.SelectboxColumn(options=["New", "PMDT", "Retreatment: Others", "Retreatment: Recurrent", "Retreatment: Treatment after failure", "Retreatment: Treatment after lost to follow up", "Unknown"], default="New"),
            "SiteOfDisease": st.column_config.SelectboxColumn(options=["Pulmonary", "Extra Pulmonary", "Unknown"], default="Pulmonary"),
            "Inter-state/Inter-district enrollment": st.column_config.SelectboxColumn(options=["Inter-District", "Inter-State"], default="Inter-State"),
            "urban_rural_background": st.column_config.SelectboxColumn(options=["rural", "urban"], default="urban"),
            "BankDetailsAdded": st.column_config.SelectboxColumn(options=["Yes", "No", "Unknown"], default="Yes"),
            "KeyPopulation": st.column_config.TextColumn(help="Enter comma-separated values (e.g. Migrant, Tobacco)"),
        }
    )

    # Place custom buttons BELOW the table
    col_add, col_del, col_space = st.columns([2, 2, 4]) 
    
    with col_add:
        if st.button("➕ Add Patient Row", use_container_width=True):
            # Save current edits before appending!
            st.session_state.batch_df = edited_df 
            
            # UPDATED: Matches the new Hybrid Preprocessor Schema exactly
            new_row = pd.DataFrame([{
                "Select": False, "Name": "", "Patient_ID": "", "Gender": "Female", 
                "Age": 30, "Weight": 50.0, "HIV_Status": "Unknown", 
                "DiabetesStatus": "Unknown", "Microbiologically_Confirmed": "0",
                "TypeOfCase": "New", "SiteOfDisease": "Pulmonary", 
                "Inter-state/Inter-district enrollment": "Unknown",
                "urban_rural_background": "urban", "BankDetailsAdded": "Yes",
                "KeyPopulation": "Not Applicable"
            }])
            
            st.session_state.batch_df = pd.concat([st.session_state.batch_df, new_row], ignore_index=True)
            
            if "batch_editor" in st.session_state:
                del st.session_state["batch_editor"]
            st.rerun()

    with col_del:
        if st.button("🗑️ Delete Selected", use_container_width=True):
            # Keep only rows where 'Select' is False
            st.session_state.batch_df = edited_df[~edited_df["Select"]].reset_index(drop=True)
            
            # If they delete everything, leave at least one empty row
            if st.session_state.batch_df.empty:
                # UPDATED: Matches the new Hybrid Preprocessor Schema exactly
                st.session_state.batch_df = pd.DataFrame([{
                    "Select": False, "Name": "", "Patient_ID": "", "Gender": "Female", 
                    "Age": 30, "Weight": 50.0, "HIV_Status": "Unknown", 
                    "DiabetesStatus": "Unknown", "Microbiologically_Confirmed": "0",
                    "TypeOfCase": "New", "SiteOfDisease": "Pulmonary", 
                    "Inter-state/Inter-district enrollment": "Unknown",
                    "urban_rural_background": "urban", "BankDetailsAdded": "Yes",
                    "KeyPopulation": "Not Applicable"
                }])

            if "batch_editor" in st.session_state:
                del st.session_state["batch_editor"]
            st.rerun()

    st.divider()

    # --- Run and Reset Buttons ---
    col1, col2 = st.columns(2)

    with col1:
        run_clicked = st.button("▶ Run Batch Prediction", use_container_width=True)

    with col2:
        if st.button("🔄 Reset Table", use_container_width=True):
            st.session_state.batch_df = st.session_state.batch_df.iloc[0:1]
            # Make sure the reset row is deselected
            st.session_state.batch_df.at[0, "Select"] = False 
            if "batch_editor" in st.session_state:
                del st.session_state["batch_editor"]
            st.rerun()

    # --- Prediction Execution ---
    if run_clicked and not edited_df.empty:
        # Drop non-ML columns
        X_df = edited_df.drop(columns=["Select", "Name", "Patient_ID"])

        with st.spinner("Running batch hybrid predictions..."):
            X_processed = preprocessor.transform(X_df)
            X_dense = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
            
            xgb_probs = xgb_model.predict_proba(X_processed)[:, 1] 
            nn_probs = nn_model.predict(X_dense, verbose=0).ravel()
            
            meta_features = np.column_stack((xgb_probs, nn_probs))
            final_risk_probs = meta_model.predict_proba(meta_features)[:, 1]

        decoded = ["default" if p >= OPTIMAL_THRESHOLD else "not-default" for p in final_risk_probs]

        result_df = edited_df.drop(columns=["Select"]).copy()
        result_df["Risk Score"] = [f"{p:.1%}" for p in final_risk_probs]
        result_df["Prediction"] = decoded

        result_df = result_df.reset_index(drop=True)
        result_df.insert(0, "S.No", range(1, len(result_df) + 1))

        st.session_state.df_out = result_df  # Saved for evaluation mode
        st.dataframe(result_df, use_container_width=True, hide_index=True)

# ---------------- EVALUATION ----------------
if debug_mode:
    st.subheader("Evaluation Mode")

    eval_file = st.file_uploader("Upload file with Actual outcomes", type=["xlsx"], key="eval")

    if eval_file:
        actual_df = pd.read_excel(eval_file)

        if st.session_state.df_out is None:
            st.error("Run batch prediction first.")
            st.stop()

        if "Actual" not in actual_df.columns:
            st.error("Evaluation file must contain 'Actual' column.")
            st.stop()

        if len(actual_df) != len(st.session_state.df_out):
            st.error("Row count mismatch.")
            st.stop()

        combined = st.session_state.df_out.copy()
        combined["Actual"] = actual_df["Actual"]

        from evaluation import evaluate_predictions
        results = evaluate_predictions(combined)

        st.write("Confusion Matrix:")
        st.write(results["confusion_matrix"])

        st.text("Classification Report:\n" + results["classification_report"])
        st.write(f"Accuracy: {results['accuracy']:.4f}")
        st.write(f"Precision: {results['precision']:.4f}")
        st.write(f"Recall: {results['recall']:.4f}")
        st.write(f"F1 Score: {results['f1']:.4f}")
