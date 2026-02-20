import streamlit as st
import pandas as pd
import numpy as np
import io
# import keras
from keras.models import load_model
# from saved_models.register_activation import ModifiedReLU

from preprocess import preprocess_single, preprocess_batch, decode_output

# --- IMPORTANT ---
if 'df_out' not in st.session_state:
    st.session_state.df_out = None

st.set_page_config(page_title="TB Default Prediction", layout="centered")

# debug_mode = st.sidebar.checkbox("Enable Evaluation Mode", value=False)
debug_mode = False

@st.cache_resource
def load_my_model():
    return load_model(
        "saved_models/tb_default_model_new.keras"
        # custom_objects={"ModifiedReLU": ModifiedReLU}
    )

model = load_my_model()

st.title("TB Default Prediction")
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------- SINGLE ----------------
with tab1:
    st.subheader("Single Patient Prediction")

    name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")
    gender = st.selectbox("Gender", ["Female", "Male", "Transgender", "Unknown"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0)
    hiv = st.selectbox("HIV Status", ["Non-Reactive", "Positive", "Reactive", "Unknown"])
    diabetes = st.selectbox("Diabetes Status", ["Non-diabetic", "Diabetic", "Unknown"])
    micro = st.selectbox("Microbiologically Confirmed?", ["Yes", "No", "Unknown"])
    typeofcase = st.selectbox("Type of TB Case", [
        "New", "PMDT", "Retreatment: Others", "Retreatment: Recurrent",
        "Retreatment: Treatment after failure",
        "Retreatment: Treatment after lost to follow up",
        "Unknown"
    ])
    site = st.selectbox("Site of Disease", ["Pulmonary", "Extra Pulmonary", "Unknown"])
    interstate = st.selectbox("Inter-state / Inter-district", ["Inter-District", "Inter-State", "Unknown"])
    urban_rural = st.selectbox(
    "Urban / Rural Background",
    ["urban", "rural", "Unknown"]
    )

    bank_details = st.selectbox(
    "Bank Details Status",
    ["Eligible", "Not Eligible", "Received", "Unknown"]
    )


    if st.button("Predict for this patient"):
        input_dict = {
            "Gender": gender,
            "Age": age,
            "Weight": weight,
            "HIV_Status": hiv,
            "DiabetesStatus": diabetes,
            "Microbiologically_Confirmed": micro,
            "TypeOfCase": typeofcase,
            "SiteOfDisease": site,
            "Inter-state/Inter-district enrollment": interstate,
            "urban_rural_background": urban_rural,
            "Bank details": bank_details
        }

        X = preprocess_single(input_dict)
        with st.spinner('Running AI diagnosis...'):
            raw_preds = model.predict(X)
        raw_out = raw_preds[0]
        label = decode_output(raw_out)

        st.success(f"Prediction for {name} (ID: {patient_id}) → {label}")
        st.write("Raw model output:", raw_out)

# ------------Manual Batch--------------

with tab2:
    st.subheader("Batch Patient Prediction")

    # Initialize once
    if "batch_df" not in st.session_state:
        st.session_state.batch_df = pd.DataFrame({
            "Name": [""],
            "Patient_ID": [""],
            "Gender": ["Female"],
            "Age": [30],
            "Weight": [50.0],
            "HIV_Status": ["Unknown"],
            "DiabetesStatus": ["Unknown"],
            "Microbiologically_Confirmed": ["Unknown"],
            "TypeOfCase": ["New"],
            "SiteOfDisease": ["Pulmonary"],
            "Inter-state/Inter-district enrollment": ["Unknown"],
            "urban_rural_background": ["urban"],
            "Bank_details": ["Eligible"]
        })

    edited_df = st.data_editor(
        st.session_state.batch_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=False,   # <-- critical
        key="batch_editor",
        column_config={
            "Name": st.column_config.TextColumn(default=""),
            "Patient_ID": st.column_config.TextColumn(default=""),
            "Gender": st.column_config.SelectboxColumn(
                options=["Female", "Male", "Transgender", "Unknown"],
                default="Female"
            ),
            "Age": st.column_config.NumberColumn(
                min_value=0,
                max_value=120,
                default=30
            ),
            "Weight": st.column_config.NumberColumn(
                min_value=1.0,
                max_value=250.0,
                default=50.0
            ),
            "HIV_Status": st.column_config.SelectboxColumn(
                options=["Non-Reactive", "Positive", "Reactive", "Unknown"],
                default="Unknown"
            ),
            "DiabetesStatus": st.column_config.SelectboxColumn(
                options=["Non-diabetic", "Diabetic", "Unknown"],
                default="Unknown"
            ),
            "Microbiologically_Confirmed": st.column_config.SelectboxColumn(
                options=["Yes", "No", "Unknown"],
                default="Unknown"
            ),
            "TypeOfCase": st.column_config.SelectboxColumn(
                options=[
                    "New",
                    "PMDT",
                    "Retreatment: Others",
                    "Retreatment: Recurrent",
                    "Retreatment: Treatment after failure",
                    "Retreatment: Treatment after lost to follow up",
                    "Unknown"
                ],
                default="New"
            ),
            "SiteOfDisease": st.column_config.SelectboxColumn(
                options=["Pulmonary", "Extra Pulmonary", "Unknown"],
                default="Pulmonary"
            ),
            "Inter-state/Inter-district enrollment": st.column_config.SelectboxColumn(
                options=["Inter-District", "Inter-State", "Unknown"],
                default="Unknown"
            ),
            "urban_rural_background": st.column_config.SelectboxColumn(
                options=["urban", "rural", "Unknown"],
                default="urban"
            ),
            "Bank_details": st.column_config.SelectboxColumn(
                options=["Eligible", "Not Eligible", "Received", "Unknown"],
                default="Eligible"
            ),
        }
    )

    edited_df = edited_df.reset_index(drop=True)
    st.session_state.batch_df = edited_df

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        run_clicked = st.button("▶ Run Batch Prediction", use_container_width=True)

    with col2:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.batch_df = st.session_state.batch_df.iloc[0:1]
            st.rerun()

    if run_clicked and not edited_df.empty:
        X = preprocess_batch(edited_df)

        with st.spinner("Running batch predictions..."):
            raw_preds = model.predict(X)

        decoded = [decode_output(r) for r in raw_preds]

        result_df = edited_df.copy()
        result_df["Prediction"] = decoded

        result_df = result_df.reset_index(drop=True)
        result_df.insert(0, "S.No", range(1, len(result_df) + 1))

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
