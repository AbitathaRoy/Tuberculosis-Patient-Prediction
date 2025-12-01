# tb_default_app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
from tensorflow.keras.models import load_model

from preprocess import preprocess_single, preprocess_batch, decode_output, FEATURE_ORDER

st.set_page_config(page_title="TB Default Prediction", layout="centered")

@st.cache_resource
def load_my_model():
    return load_model("tb_default_model.keras")   # your saved model

model = load_my_model()

st.title("TB Default Prediction")
st.caption("Provide patient information (single or batch) to predict treatment default risk.")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

# ----------------------
# SINGLE PREDICTION TAB
# ----------------------
with tab1:
    st.subheader("Single patient prediction")

    # Short, clear prompts (as requested)
    gender = st.selectbox("What is the patient's gender?", ["Female", "Male", "Transgender"])
    age = st.number_input("What is the patient's age (years)?", min_value=0, max_value=120, value=30)
    weight = st.number_input("What is the patient's weight (kg)?", min_value=1.0, max_value=250.0, value=55.0, format="%.1f")
    hiv = st.selectbox("HIV status?", ["Non-Reactive", "Unknown", "Positive", "Reactive"])
    diabetes = st.selectbox("Diabetes status?", ["Non-diabetic", "Unknown", "Diabetic"])
    micro = st.selectbox("Microbiologically confirmed?", ["Yes", "No"])
    typeofcase = st.selectbox(
        "Type of TB case?",
        [
            "Retreatment: Recurrent",
            "New",
            "Retreatment: Others",
            "PMDT",
            "Retreatment: Treatment after failure",
            "Retreatment: Treatment after lost to follow up"
        ]
    )
    site = st.selectbox("Site of disease?", ["Pulmonary", "Extra Pulmonary"])
    interstate = st.selectbox("Is this inter-district / inter-state enrollment?", ["Inter-District", "Inter-State"])

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

        try:
            X = preprocess_single(input_dict)   # shape (1, n_features)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
        else:
            raw = model.predict(X)
            # Support model.predict returning (1,) or (1, n_classes)
            raw_out = raw[0] if hasattr(raw, "__len__") else raw
            label = decode_output(raw_out)

            st.success(f"Prediction: **{label}**")
            st.write("Raw model output:", raw_out)


# ----------------------
# BATCH PREDICTION TAB
# ----------------------
with tab2:
    st.subheader("Batch predictions via XLSX")

    # Provide the template to download (create a template on the fly)
    if st.button("Download XLSX template"):
        # Create empty template with correct columns
        template_df = pd.DataFrame(columns=FEATURE_ORDER)
        towrite = io.BytesIO()
        template_df.to_excel(towrite, index=False)
        towrite.seek(0)
        st.download_button(
            "Download template.xlsx",
            data=towrite,
            file_name="template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    uploaded = st.file_uploader("Upload filled template (XLSX)", type=["xlsx"])

    if uploaded is not None:
        try:
            df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
        else:
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(50))

            if st.button("Run batch prediction"):
                try:
                    X = preprocess_batch(df)
                except Exception as e:
                    st.error(f"Preprocessing error: {e}")
                else:
                    raw_preds = model.predict(X)
                    # raw_preds shape (n, ) or (n, n_classes)
                    decoded = []
                    for r in raw_preds:
                        decoded.append(decode_output(r))

                    df_out = df.copy()
                    df_out["Prediction"] = decoded

                    st.success("Batch predictions completed.")
                    st.dataframe(df_out)

                    # Offer download
                    buffer = io.BytesIO()
                    df_out.to_excel(buffer, index=False)
                    buffer.seek(0)
                    st.download_button(
                        "Download predictions.xlsx",
                        data=buffer,
                        file_name="tb_predictions.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
