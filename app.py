import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time
import plotly.graph_objects as go

# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Load Model
# =========================

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

# =========================
# Custom CSS
# =========================

st.markdown("""
<style>

#MainMenu{
visibility:hidden;
}

footer{
visibility:hidden;
}

header{
visibility:hidden;
}

.main{
background:#f5f7fb;
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
}

.title{
text-align:center;
font-size:45px;
font-weight:bold;
color:#d62828;
}

.subtitle{
text-align:center;
font-size:18px;
color:gray;
margin-bottom:25px;
}

.card{

background:white;

padding:25px;

border-radius:18px;

box-shadow:0px 0px 15px rgba(0,0,0,.08);

margin-bottom:25px;

}

.result-good{

background:#d8f3dc;

padding:20px;

border-radius:15px;

border-left:10px solid green;

}

.result-bad{

background:#ffe5e5;

padding:20px;

border-radius:15px;

border-left:10px solid red;

}

.footer{

text-align:center;

color:gray;

font-size:14px;

margin-top:50px;

}

</style>
""",unsafe_allow_html=True)

# =========================
# Header
# =========================

st.markdown("""

<div class='title'>

🩺 Heart Disease Prediction System

</div>

<div class='subtitle'>

Machine Learning Based Clinical Decision Support Tool

</div>

""",unsafe_allow_html=True)

st.divider()

# =========================
# Sidebar
# =========================

with st.sidebar:

    st.title("🏥 About Project")

    st.write("""

This application predicts the possibility of Heart Disease using Machine Learning.

Model Used

✅ K-Nearest Neighbors (KNN)

Accuracy

84.24%

Dataset

Heart Disease Dataset

""")

    st.success("Educational Project")

    st.info("""

⚠️

This prediction should NOT replace professional medical diagnosis.

Always consult your physician.

""")

# =========================
# Patient Information
# =========================

st.markdown("<div class='card'>",unsafe_allow_html=True)

st.subheader("👤 Patient Information")

left,right=st.columns(2)

with left:

    age=st.number_input(
        "Age",
        1,
        120,
        40
    )

    sex=st.selectbox(
        "Sex",
        ["Female","Male"]
    )

    sex=0 if sex=="Female" else 1

with right:

    dataset=0

    cp=st.selectbox(

        "Chest Pain Type",

        [

            "Asymptomatic",

            "Atypical Angina",

            "Non-anginal",

            "Typical Angina"

        ]

    )

    cp_dict={

        "Asymptomatic":0,

        "Atypical Angina":1,

        "Non-anginal":2,

        "Typical Angina":3

    }

    cp=cp_dict[cp]

st.markdown("</div>",unsafe_allow_html=True)

# =========================
# Clinical Measurements
# =========================

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("🩺 Clinical Measurements")

col1, col2 = st.columns(2)

with col1:

    trestbps = st.number_input(
        "Resting Blood Pressure (mmHg)",
        min_value=50,
        max_value=250,
        value=120
    )

    chol = st.number_input(
        "Serum Cholesterol (mg/dL)",
        min_value=50,
        max_value=700,
        value=200
    )

    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dL",
        ["False", "True"]
    )

    fbs = 0 if fbs == "False" else 1

    restecg = st.selectbox(
        "Resting ECG",
        [
            "LV Hypertrophy",
            "Normal",
            "ST-T Abnormality"
        ]
    )

    restecg_dict = {
        "LV Hypertrophy": 0,
        "Normal": 1,
        "ST-T Abnormality": 2
    }

    restecg = restecg_dict[restecg]

with col2:

    thalch = st.number_input(
        "Maximum Heart Rate",
        min_value=50,
        max_value=250,
        value=150
    )

    exang = st.selectbox(
        "Exercise Induced Angina",
        ["False", "True"]
    )

    exang = 0 if exang == "False" else 1

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    slope = st.selectbox(
        "ST Segment Slope",
        [
            "Downsloping",
            "Flat",
            "Upsloping"
        ]
    )

    slope_dict = {
        "Downsloping": 0,
        "Flat": 1,
        "Upsloping": 2
    }

    slope = slope_dict[slope]

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button(
    "🔍 Predict Heart Disease",
    use_container_width=True,
    type="primary"
)

# =====================================================
# Prediction Dashboard
# =====================================================

if predict:

    # Animation
    with st.spinner("🩺 AI is analyzing the patient's data..."):
        time.sleep(2)

    st.success("✅ Prediction Completed Successfully")

    # ترتيب الـ Features كما تم تدريب الموديل
    features = np.array([[
        age,
        sex,
        dataset,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalch,
        exang,
        oldpeak,
        slope
    ]])

    # Scaling
    features = scaler.transform(features)

    # Prediction
    prediction = model.predict(features)[0]

    # Probability
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(features)[0][1]
    else:
        probability = 0.80 if prediction == 1 else 0.20

    risk = int(probability * 100)

    st.divider()
    st.markdown("## 📊 Prediction Dashboard")

    # -----------------------------
    # Gauge Chart
    # -----------------------------
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk,
        title={'text': "Heart Disease Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 35], 'color': "#8BE28B"},
                {'range': [35, 70], 'color': "#FFD966"},
                {'range': [70, 100], 'color': "#FF8C8C"}
            ]
        }
    ))

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Risk Level
    # -----------------------------
    if risk < 35:
        level = "🟢 Low Risk"

    elif risk < 70:
        level = "🟡 Moderate Risk"

    else:
        level = "🔴 High Risk"

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Prediction",
                  "Heart Disease" if prediction == 1 else "No Heart Disease")

    with c2:
        st.metric("Risk Score",
                  f"{risk}%")

    with c3:
        st.metric("Risk Level",
                  level)

    st.progress(risk / 100)

    # =====================================================
    # Result
    # =====================================================

    if prediction == 1:

        st.markdown("""
        <div class='result-bad'>
        <h2>🔴 High Risk of Heart Disease</h2>

        <h4>
        The model predicts that the patient may have Heart Disease.
        </h4>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class='result-good'>
        <h2>🟢 Low Risk of Heart Disease</h2>

        <h4>
        The model predicts that the patient is unlikely to have Heart Disease.
        </h4>

        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # =====================================================
    # Recommendations
    # =====================================================

    st.subheader("💡 Recommendations")

    if prediction == 1:

        st.warning("""

✅ Consult a Cardiologist.

✅ Perform ECG and additional laboratory investigations.

✅ Monitor blood pressure regularly.

✅ Follow a healthy balanced diet.

✅ Reduce cholesterol intake.

✅ Stop smoking.

✅ Exercise only after medical advice.

✅ Follow your doctor's recommendations.

""")

    else:

        st.success("""

✅ Maintain a healthy diet.

✅ Exercise regularly.

✅ Drink enough water.

✅ Sleep 7–8 hours daily.

✅ Continue regular medical check-ups.

✅ Maintain a healthy body weight.

""")

    st.divider()

    # =====================================================
    # Patient Summary
    # =====================================================

    st.subheader("📋 Patient Summary")

    summary = pd.DataFrame({

        "Feature":[

            "Age",

            "Sex",

            "Chest Pain",

            "Blood Pressure",

            "Cholesterol",

            "Fasting Blood Sugar",

            "Rest ECG",

            "Maximum Heart Rate",

            "Exercise Angina",

            "Oldpeak",

            "Slope"

        ],

        "Value":[

            age,

            "Male" if sex else "Female",

            ["Asymptomatic",
             "Atypical Angina",
             "Non-anginal",
             "Typical Angina"][cp],

            trestbps,

            chol,

            "True" if fbs else "False",

            ["LV Hypertrophy",
             "Normal",
             "ST-T Abnormality"][restecg],

            thalch,

            "True" if exang else "False",

            oldpeak,

            ["Downsloping",
             "Flat",
             "Upsloping"][slope]

        ]

    })

    st.dataframe(summary, use_container_width=True)

    st.divider()

    # =====================================================
    # Model Information
    # =====================================================

    with st.expander("🤖 Model Information"):

        st.write("### Machine Learning Model")

        st.write("**Algorithm:** K-Nearest Neighbors (KNN)")

        st.write("**Accuracy:** 84.24%")

        st.write("**Features Used:** 12")

        st.write("**Dataset:** Heart Disease Dataset")

        st.write(
            "This model predicts the likelihood of Heart Disease based on clinical measurements."
        )

    st.divider()

    # =====================================================
    # Disclaimer
    # =====================================================

    st.subheader("⚠️ Medical Disclaimer")

    st.info("""

This application is intended for educational purposes only.

The prediction generated by this Machine Learning model should NOT be considered
a medical diagnosis.

Always consult a qualified healthcare professional before making any medical decision.

""")

    st.divider()

    # =====================================================
    # New Prediction
    # =====================================================

    if st.button("🔄 New Prediction", use_container_width=True):

        st.rerun()

# =====================================================
# Footer
# =====================================================

st.markdown("""

<br><br>

<hr>

<div class="footer">

<h3>🩺 Heart Disease Prediction System</h3>

Developed by <b>Alya</b>

<br>

Machine Learning Graduation Project

<br><br>

© 2026 All Rights Reserved

</div>

""", unsafe_allow_html=True)
