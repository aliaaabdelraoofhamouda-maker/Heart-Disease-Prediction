import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease App", layout="centered")

# 🎨 Light UI styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }

    h1 {
        color: #d90429;
        text-align: center;
    }

    .stButton>button {
        background-color: #2b6cb0;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }

    .stButton>button:hover {
        background-color: #1e4e8c;
        color: white;
    }

    </style>
""", unsafe_allow_html=True)

st.title("❤️ Heart Disease Prediction System")
st.write("AI-powered medical diagnosis tool")

# Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 120)
    sex = st.selectbox("Sex", ["Female", "Male"])
    trestbps = st.number_input("Blood Pressure")
    fbs = st.selectbox("Fasting Blood Sugar", ["No", "Yes"])
    exang = st.selectbox("Exercise Angina", ["No", "Yes"])

with col2:
    cp = st.selectbox("Chest Pain Type", [0,1,2,3])
    chol = st.number_input("Cholesterol")
    restecg = st.selectbox("ECG", [0,1,2])
    thalch = st.number_input("Max Heart Rate")
    oldpeak = st.number_input("Oldpeak")
    slope = st.selectbox("Slope", [0,1,2])

# Predict
if st.button("Predict"):

    with st.spinner("Processing data..."):

        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        features = np.array([[
            age, sex, 0, cp,
            trestbps, chol, fbs,
            restecg, thalch, exang,
            oldpeak, slope
        ]])

        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        prob = model.predict_proba(features)[0]

    st.subheader("Result:")

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease ({prob[1]*100:.2f}%)")
    else:
        st.success(f"✅ Low Risk of Heart Disease ({prob[0]*100:.2f}%)")
