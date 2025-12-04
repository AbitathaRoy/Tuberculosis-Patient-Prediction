# Tuberculosis Default Prediction Dashboard 🏥

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning-powered dashboard designed to predict the likelihood of a Tuberculosis (TB) patient "defaulting" (dropping out) of their treatment regimen. Early identification allows healthcare providers to intervene with counseling and support, ultimately improving treatment adherence and cure rates.

🔗 **[Live Demo: Click Here to Launch App](https://tb-default-patient-prediction.streamlit.app/)**

## 🎯 Problem Statement
TB treatment requires a strict 6-9 month regimen. "Defaulting" (stopping medication) leads to drug-resistant TB (MDR-TB), which is harder to cure and more contagious. This tool uses patient demographic and clinical data to flag high-risk patients *before* they default.

## 🧠 Model & Tech Stack
* **Algorithm:** Neural Network (Deep Learning) built with TensorFlow/Keras.
* **Architecture:**
    * Input Layer: 9 Clinical Features
    * Hidden Layers: Dense layers with Custom `ModifiedReLU` activation & Dropout for regularization.
    * Output: Binary Classification (Default vs. Not-Default).
* **Web Framework:** [Streamlit](https://streamlit.io/) for the interactive dashboard.
* **Deployment:** Streamlit Community Cloud (connected to GitHub).

## 🚀 Key Features
1.  **Single Patient Mode:** Input specific vitals (Age, Weight, HIV status, etc.) for a quick risk assessment.
2.  **Batch Prediction:** Upload an `.xlsx` file with hundreds of patient records to generate bulk predictions instantly.
3.  **Smart Template System:** Generates a pre-validated Excel template in-memory to prevent data entry errors.
4.  **Privacy-First:** No patient data is saved to the server's disk; all processing happens in RAM.

## 🛠️ How to Run Locally
If you want to run this dashboard on your own machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/AbitathaRoy/Tuberculosis-Patient-Prediction.git](https://github.com/AbitathaRoy/Tuberculosis-Patient-Prediction.git)
    cd Tuberculosis-Patient-Prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app:**
    ```bash
    streamlit run tb_default_app.py
    ```

## 📂 Project Structure
```text
├── tb_default_app.py       # Main Application Logic (Frontend)
├── preprocess.py           # Data Cleaning & Scaling Pipeline
├── evaluation.py           # Metrics & Scoring Logic
├── excel_template.py       # Dynamic Excel Generator
├── saved_models/           # Trained Keras Models & Scalers
└── requirements.txt        # Dependency List