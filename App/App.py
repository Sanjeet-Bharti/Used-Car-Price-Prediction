import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model/model.pkl", "rb"))

st.title("🚗 Used Car Price Prediction App")

year = st.number_input("Year")
km = st.number_input("Kilometers Driven")

fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

car_age = 2026 - year

input_data = np.array([[car_age, km]])

if st.button("Predict Price"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Price: ₹ {prediction[0]:,.2f}")
