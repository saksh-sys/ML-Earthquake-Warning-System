import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load your trained model and scaler
model = load('XG_Boost_model.joblib')
scaler = load('scaler.joblib')  # Assuming you have this saved now

def predict_earthquake(features):
    # Scale features
    features_scaled = scaler.transform([features])
    # Predict
    prediction = model.predict(features_scaled)
    return prediction

# Streamlit page configuration
st.title('Earthquake Magnitude Prediction')
st.write('This application predicts the magnitude of earthquakes based on input features.')

# Input fields
latitude = st.number_input('Latitude', value=0.0)
longitude = st.number_input('Longitude', value=0.0)
depth = st.number_input('Depth', value=0.0)
year = st.number_input('Year', min_value=2000, max_value=2023, value=2021)
day_of_year = st.number_input('Day of the Year', min_value=1, max_value=365, value=1)
hour = st.number_input('Hour of the Day', min_value=0, max_value=23, value=12)

# Button to make prediction
if st.button('Predict Magnitude'):
    features = [latitude, longitude, depth, year, day_of_year, hour]
    prediction = predict_earthquake(features)
    st.write(f'Predicted Magnitude: {prediction[0]}')


st.header("*Contributors*")
        contributors = {
            "ANSH VARSHNEY": {"email": "anshvarshney3@gmail.com", 
                              "github_link": "https://github.com/ansh0707",
                              "details":{
                                  "Reg No" : "12006893",
                                  "University":"Lovely Professional University"
                              }},
            "SARA BORA": {"email": "sarabora420@gmail.com", 
                          "github_link": "https://github.com/sara-bora",
                          "details":{
                              "Reg No" : "12013194",
                              "University":"Lovely Professional University"
                          }},
            "SARTHAK MISHRA": {"email": "sam4sarthak@gmail.com",
                                "github_link": "https://github.com/SarthakMishra0307",
                                "details":{
                                    "Reg No" : "12018433",
                                    "University":"Lovely Professional University"
                                }},
            "SATYAM DUBEY": {"email": "satyamdubey2988@gmail.com",
                             "github_link": "https://github.com/dubeysatyam2002",
                             "details":{
                                    "Reg No" : "12014267",
                                    "University":"Lovely Professional University"
                            }},
        }
