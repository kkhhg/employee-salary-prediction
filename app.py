import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ---------------- UI Layout ----------------
st.set_page_config(page_title="Income Prediction App", page_icon="💰", layout="centered")

st.markdown("""
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #2F4F4F;
        text-align: center;
        padding: 10px;
    }
    .subtitle {
        font-size: 20px;
        text-align: center;
        color: #555;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">💼 Employee Income Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict whether an individual earns more than ₹50,000/year based on their profile.</div>', unsafe_allow_html=True)

st.write("---")

# ---------------- Input Form ----------------
with st.form("income_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 30)
        fnlwgt = st.number_input("Final Weight (fnlwgt)", min_value=10000, max_value=1000000, value=300000)
        educational_num = st.slider("Education Number", 1, 16, 10)
        capital_gain = st.number_input("Capital Gain", value=0)
        capital_loss = st.number_input("Capital Loss", value=0)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)

    with col2:
        workclass = st.selectbox("Workclass", label_encoders['workclass'].classes_)
        education = st.selectbox("Education", label_encoders['education'].classes_)
        marital_status = st.selectbox("Marital Status", label_encoders['marital-status'].classes_)
        occupation = st.selectbox("Occupation", label_encoders['occupation'].classes_)
        relationship = st.selectbox("Relationship", label_encoders['relationship'].classes_)
        race = st.selectbox("Race", label_encoders['race'].classes_)
        gender = st.selectbox("Gender", label_encoders['gender'].classes_)
        native_country = st.selectbox("Native Country", label_encoders['native-country'].classes_)

    submit = st.form_submit_button("🔍 Predict Income")

# ---------------- Prediction ----------------
if submit:
    input_dict = {
        'age': age,
        'workclass': label_encoders['workclass'].transform([workclass])[0],
        'fnlwgt': fnlwgt,
        'education': label_encoders['education'].transform([education])[0],
        'educational-num': educational_num,
        'marital-status': label_encoders['marital-status'].transform([marital_status])[0],
        'occupation': label_encoders['occupation'].transform([occupation])[0],
        'relationship': label_encoders['relationship'].transform([relationship])[0],
        'race': label_encoders['race'].transform([race])[0],
        'gender': label_encoders['gender'].transform([gender])[0],
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': label_encoders['native-country'].transform([native_country])[0]
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = label_encoders['income'].inverse_transform([prediction])[0]

    st.success(f"💡 **Predicted Income:** {result}")
    if result == ">50K":
        st.balloons()
