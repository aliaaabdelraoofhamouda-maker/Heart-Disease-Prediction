import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient details below:")

# Inputs
age = st.number_input("Age", 1, 120)

sex = st.selectbox("Sex", [0, 1])  # 0=Female, 1=Male

dataset = st.number_input("Dataset (encoded value)", 0, 10)

cp = st.number_input("Chest Pain Type (encoded)", 0, 10)

trestbps = st.number_input("Resting Blood Pressure")

chol = st.number_input("Cholesterol")

fbs = st.selectbox("Fasting Blood Sugar", [0, 1])

restecg = st.number_input("Resting ECG (encoded)")

thalch = st.number_input("Max Heart Rate")

exang = st.selectbox("Exercise Induced Angina", [0, 1])

oldpeak = st.number_input("Oldpeak")

slope = st.number_input("Slope (encoded)")

# Predict button
if st.button("Predict"):

    features = np.array([[
        age, sex, dataset, cp,
        trestbps, chol, fbs,
        restecg, thalch, exang,
        oldpeak, slope
    ]])

    # Scaling
    features = scaler.transform(features)

    # Prediction
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features)[0]

    st.subheader("Result:")

    if prediction == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")

    st.write(f"Probability No Disease: {prob[0]:.2f}")
    st.write(f"Probability Disease: {prob[1]:.2f}")
