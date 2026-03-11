import streamlit as st
import joblib
import numpy as np

model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction")

age = st.slider("Age",20,80,45)
chol = st.slider("Cholesterol",100,400,200)
max_hr = st.slider("Max Heart Rate",60,200,150)

input_data = np.array([[age,1,0,120,chol,0,0,max_hr,0,0,0,0,0]])

scaled = scaler.transform(input_data)

if st.button("Predict"):
    
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1]

    if pred == 1:
        st.error(f"High Risk ({round(prob*100,2)}%)")
    else:
        st.success(f"Low Risk ({round(prob*100,2)}%)")
