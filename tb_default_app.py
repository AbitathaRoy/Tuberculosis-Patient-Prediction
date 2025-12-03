import streamlit as st
import pandas as pd
import numpy as np
import io
import keras
from keras.models import load_model
from saved_models.register_activation import ModifiedReLU

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
        "saved_models/tb_default_model.keras",
        custom_objects={"ModifiedReLU": ModifiedReLU}
    )

model = load_my_model()

st.title("TB Default Prediction")
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ---------------- SINGLE ----------------
with tab1:
    st.subheader("Single patient prediction")

    name = st.text_input("Patient Name")
    patient_id = st.text_input("Patient ID")
    gender = st.selectbox("Gender", ["Female", "Male", "Transgender"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0)
    hiv = st.selectbox("HIV Status", ["Non-Reactive", "Positive", "Reactive", "Unknown"])
    diabetes = st.selectbox("Diabetes Status", ["Non-diabetic", "Diabetic"])
    micro = st.selectbox("Microbiologically Confirmed?", ["Yes", "No"])
    typeofcase = st.selectbox("Type of TB Case", [
        "New", "PMDT", "Retreatment: Others", "Retreatment: Recurrent",
        "Retreatment: Treatment after failure",
        "Retreatment: Treatment after lost to follow up"
    ])
    site = st.selectbox("Site of Disease", ["Pulmonary", "Extra Pulmonary"])
    interstate = st.selectbox("Inter-state / Inter-district", ["Inter-District", "Inter-State"])

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
            "Inter-state/Inter-district enrollment": interstate
        }

        X = preprocess_single(input_dict)
        with st.spinner('Running AI diagnosis...'):
            raw_preds = model.predict(X)
        raw_out = raw_preds[0]
        label = decode_output(raw_out)

        st.success(f"Prediction for {name} (ID: {patient_id}) → {label}")
        st.write("Raw model output:", raw_out)

# ---------------- BATCH ----------------
with tab2:
    st.subheader("Batch predictions via XLSX")

    if st.button("Download XLSX template"):
        from excel_template import create_template
        import io 

        # Create an in-memory buffer (a virtual file)
        buffer = io.BytesIO()
        
        # Pass the buffer instead of "template.xlsx"
        # Pandas will write the excel data into this variable in RAM
        create_template(buffer, max_rows=5000)
        
        # Rewind the buffer to the beginning so it can be read
        buffer.seek(0)
        
        st.download_button(
            label="Download template.xlsx",
            data=buffer,
            file_name="template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    uploaded = st.file_uploader("Upload filled template", type=["xlsx"])

    if uploaded:
        df = pd.read_excel(uploaded)
        st.write("Preview:")
        st.dataframe(df.head())

        if st.button("Run batch prediction"):

            initial_count = len(df)
            df_clean = df.dropna(how="any")
            dropped_count = initial_count - len(df_clean)
            if dropped_count > 0:
                st.warning(f"{dropped_count} rows were dropped due to missing values.")

            features_df = df_clean.drop(columns=["PatientName", "PatientID"])

            X = preprocess_batch(features_df)
            with st.spinner('Running AI diagnosis...'):
                raw_preds = model.predict(X)

            decoded = [decode_output(r) for r in raw_preds]

            # store predictions globally
            st.session_state.df_out = df_clean.copy()
            st.session_state.df_out["Prediction"] = decoded

            st.success("Batch predictions complete.")
            st.dataframe(st.session_state.df_out)

            buf = io.BytesIO()
            st.session_state.df_out.to_excel(buf, index=False)
            buf.seek(0)
            st.download_button("Download predictions.xlsx", data=buf,
                               file_name="tb_predictions.xlsx")



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
