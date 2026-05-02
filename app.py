import streamlit as st
import pickle
import numpy as np

# Load model
with open("models/saved_models.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Smart Internet Usage Productivity Predictor")

st.write("Enter your internet usage details:")

# Inputs
website_category = st.selectbox(
    "Website Category",
    ["Education", "Coding", "Social Media", "Entertainment", "Gaming", "Shopping", "Messaging", "News", "Streaming", "Research"]
)

time_spent = st.slider("Time Spent (minutes)", 10, 300, 60)
frequency = st.slider("Frequency (visits/day)", 1, 10, 3)
day_type = st.selectbox("Day Type", ["Weekday", "Weekend"])

# Manual encoding (same as training)
category_map = {
    "Coding":0,"Education":1,"Entertainment":2,"Gaming":3,
    "Messaging":4,"News":5,"Research":6,"Shopping":7,
    "Social Media":8,"Streaming":9
}

day_map = {"Weekday":0, "Weekend":1}

# Convert input
input_data = np.array([[ 
    category_map[website_category],
    time_spent,
    frequency,
    day_map[day_type]
]])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    
    label_map = {
        0: "Productive",
        1: "Neutral",
        2: "Unproductive"
    }
    
    st.success(f"Predicted Productivity: {label_map[prediction]}")
