import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")
st.divider()

st.write("Please enter the values and hit the predict button for getting a prediction.")
st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)
tenure = st.number_input("Enter Tenure", min_value=0, max_value=130, value=10)
monthlycharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)
gender = st.selectbox("Enter the Gender", ["Male", "Female"])
st.divider()

predictbutton = st.button("Predict!")

if predictbutton:
    st.balloons()
    gender_sel = 1 if gender == "Female" else 0
    
    X= [age, gender_sel, tenure, monthlycharge]
    X1= np.array(X)
    
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    Predicted = "YES" if prediction == 1 else "NO"
    st.write(f"Predicted: {Predicted}")
    

else:
    st.write("Please click the predict button to get a prediction.")
