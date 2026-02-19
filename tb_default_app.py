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

    if "batch_data" not in st.session_state:
        st.session_state.batch_data = []

    # Add custom CSS for scrollable rows
    st.markdown("""
    <style>
    .scrollable-container {
        overflow-x: auto;
        overflow-y: visible;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Display patient records as horizontal rows with scrolling
    if len(st.session_state.batch_data) > 0:
        st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
        
        for i in range(len(st.session_state.batch_data)):
            # Create wider columns to force horizontal scrolling
            cols = st.columns([2.0, 2.0, 2.0, 1.2, 1.5, 2.0, 2.0, 2.5, 2.5, 2.0, 2.2, 2.0, 2.2, 0.8])
            
            with cols[0]:
                st.text_input("Patient Name", key=f"name_{i}", label_visibility="visible")
            with cols[1]:
                st.text_input("Patient ID", key=f"id_{i}", label_visibility="visible")
            with cols[2]:
                st.selectbox("Gender", 
                    ["Female", "Male", "Transgender", "Unknown"], 
                    key=f"gender_{i}", label_visibility="visible")
            with cols[3]:
                st.number_input("Age", 0, 120, 30, key=f"age_{i}", label_visibility="visible")
            with cols[4]:
                st.number_input("Weight (kg)", 1.0, 250.0, key=f"weight_{i}", label_visibility="visible")
            with cols[5]:
                st.selectbox("HIV Status", 
                    ["Non-Reactive", "Positive", "Reactive", "Unknown"],
                    key=f"hiv_{i}", label_visibility="visible")
            with cols[6]:
                st.selectbox("Diabetes Status", 
                    ["Non-diabetic", "Diabetic", "Unknown"],
                    key=f"diabetes_{i}", label_visibility="visible")
            with cols[7]:
                st.selectbox("Microbiologically Confirmed",
                    ["Yes", "No", "Unknown"],
                    key=f"micro_{i}", label_visibility="visible")
            with cols[8]:
                st.selectbox("Type of TB Case", [
                    "New", "PMDT", "Retreatment: Others",
                    "Retreatment: Recurrent",
                    "Retreatment: Treatment after failure",
                    "Retreatment: Treatment after lost to follow up",
                    "Unknown"
                ], key=f"type_{i}", label_visibility="visible")
            with cols[9]:
                st.selectbox("Site of Disease",
                    ["Pulmonary", "Extra Pulmonary", "Unknown"],
                    key=f"site_{i}", label_visibility="visible")
            with cols[10]:
                st.selectbox("Inter-state/Inter-district",
                    ["Inter-District", "Inter-State", "Unknown"],
                    key=f"interstate_{i}", label_visibility="visible")
            with cols[11]:
                st.selectbox("Urban/Rural Background",
                    ["urban", "rural", "Unknown"],
                    key=f"urban_{i}", label_visibility="visible")
            with cols[12]:
                st.selectbox("Bank Details Status",
                    ["Eligible", "Not Eligible", "Received", "Unknown"],
                    key=f"bank_{i}", label_visibility="visible")
            with cols[13]:
                if st.button("✕", key=f"delete_{i}"):
                    st.session_state.batch_data.pop(i)
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Bottom control buttons
    st.divider()
    col1, col2, col3, col_spacer = st.columns([1, 1, 1, 2])
    
    with col1:
        if st.button("➕ Add New Patient", use_container_width=True):
            st.session_state.batch_data.append({})
            st.rerun()
    
    with col2:
        if st.button("▶ Run Batch Prediction", use_container_width=True):
            if len(st.session_state.batch_data) > 0:
                records = []

                for i in range(len(st.session_state.batch_data)):
                    records.append({
                        "Name": st.session_state.get(f"name_{i}", ""),
                        "Patient_ID": st.session_state.get(f"id_{i}", ""),
                        "Gender": st.session_state[f"gender_{i}"],
                        "Age": st.session_state[f"age_{i}"],
                        "Weight": st.session_state[f"weight_{i}"],
                        "HIV_Status": st.session_state[f"hiv_{i}"],
                        "DiabetesStatus": st.session_state[f"diabetes_{i}"],
                        "Microbiologically_Confirmed": st.session_state[f"micro_{i}"],
                        "TypeOfCase": st.session_state[f"type_{i}"],
                        "SiteOfDisease": st.session_state[f"site_{i}"],
                        "Inter-state/Inter-district enrollment": st.session_state[f"interstate_{i}"],
                        "urban_rural_background": st.session_state[f"urban_{i}"],
                        "Bank_details": st.session_state[f"bank_{i}"]
                    })

                df_batch = pd.DataFrame(records)

                X = preprocess_batch(df_batch)
                with st.spinner('Running batch predictions...'):
                    raw_preds = model.predict(X)

                decoded = [decode_output(r) for r in raw_preds]

                df_batch["Prediction"] = decoded

                st.success("Batch predictions complete.")
                st.dataframe(df_batch)
            else:
                st.warning("Please add at least one patient.")
    
    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.batch_data = []
            st.rerun()




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
