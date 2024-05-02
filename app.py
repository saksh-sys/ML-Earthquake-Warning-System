import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import folium_static
import numpy as np  # Add numpy import

# Define and load your scaler and model
scaler = load('scaler.joblib')
model = load('XG_Boost_model.joblib')

def predict_earthquake(features):
    # Scale features
    features_scaled = scaler.transform([features])
    # Predict
    prediction = model.predict(features_scaled)
    return prediction

# Streamlit page configuration
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a page', ['Model', 'Contributors'])

if page == 'Model':
    st.title('Earthquake Magnitude Prediction')
    st.write('This application predicts the magnitude of earthquakes based on input features.')

    # Input fields
    latitude = st.number_input('Latitude', value=0.0)
    longitude = st.number_input('Longitude', value=0.0)
    depth = st.number_input('Depth', value=0.0)
    year = st.number_input('Year', min_value=2000, max_value=2023, value=2021)
    day_of_year = st.number_input('Day of the Year', min_value=1, max_value=365, value=1)
    hour = st.number_input('Hour of the Day', min_value=0, max_value=23, value=12)

    # Create a Folium map
