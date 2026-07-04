import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease App", layout="centered")

st.title("❤️ Heart Disease Prediction System")
st.write("Enter patient details below to predict heart disease risk")

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Female", "Male"])
    trestbps = st.number_input("Resting Blood Pressure")
    fbs = st.selectbox("Fasting Blood Sugar", ["No", "Yes"])
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])

with col2:
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    chol = st.number_input("Cholesterol")
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalch = st.number_input("Max Heart Rate")
    oldpeak = st.number_input("Oldpeak")
    slope = st.selectbox("Slope", [0, 1, 2])

# Predict
if st.button("Predict"):

    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    features = np.array([[
        age, sex, 0, cp,
        trestbps, chol, fbs,
        restecg, thalch, exang,
        oldpeak, slope
    ]])

    # scaling
    features = scaler.transform(features)

    # prediction
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    # output
    st.subheader("Result:")

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.write(f"Risk Probability: {prob[1]*100:.2f}%")
