import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------
# Load model & scaler (relative path for deployment)
# -------------------
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "cervical_cancer_rf_model_10features.pkl")
scaler_path = os.path.join(base_dir, "scaler_10features.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# -------------------
# Page setup
# -------------------
st.set_page_config(page_title="Cervical Cancer Risk Analysis", layout="centered")
st.title("🩺 Cervical Cancer Risk Prediction")
st.markdown("Fill in the details below to estimate your **risk** based on health and lifestyle factors.")

# -------------------
# Input form
# -------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        Age = st.number_input("Age", min_value=15, max_value=90, step=1)
        Number_of_sexual_partners = st.number_input("Number of sexual partners", min_value=0, max_value=30, step=1)
        First_sexual_intercourse = st.number_input("Age at first sexual intercourse", min_value=10, max_value=50, step=1)
        Num_of_pregnancies = st.number_input("Number of pregnancies", min_value=0, max_value=20, step=1)
        Smokes = st.number_input("Smokes (packs/year)", min_value=0, max_value=40, step=1)

    with col2:
        Smokes_years = st.number_input("Years smoked", min_value=0, max_value=50, step=1)
        Hormonal_Contraceptives = st.number_input("Hormonal contraceptives (years)", min_value=0, max_value=30, step=1)
        IUD = st.number_input("IUD use (years)", min_value=0, max_value=30, step=1)
        STDs = st.number_input("Number of STDs", min_value=0, max_value=10, step=1)
        STD_HPV = st.selectbox("STD: HPV", ["No", "Yes"])

    submit = st.form_submit_button("Predict Risk")

# -------------------
# Prediction
# -------------------
if submit:
    STD_HPV_val = 1 if STD_HPV == "Yes" else 0

    input_data = np.array([[
        Age,
        Number_of_sexual_partners,
        First_sexual_intercourse,
        Num_of_pregnancies,
        Smokes,
        Smokes_years,
        Hormonal_Contraceptives,
        IUD,
        STDs,
        STD_HPV_val
    ]])

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"High Risk of Cervical Cancer — Probability: {probability:.2%}")
    else:
        st.success(f"Low Risk of Cervical Cancer — Probability: {probability:.2%}")

    st.markdown("**Note:** This prediction is based on the trained model and should not replace professional medical advice.")
