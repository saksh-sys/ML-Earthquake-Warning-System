import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import folium
from streamlit_folium import folium_static

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

# Create a Folium map
m = folium.Map(location=[latitude, longitude], zoom_start=6)

# Add marker for selected location
#folium.Marker([latitude, longitude], popup='Selected Location').add_to(m)

# Define a function to handle the click event
def handle_click(event):
    # Get the latitude and longitude coordinates of the clicked location
    lat, lon = event.latlng
    # Display the coordinates in Streamlit
    st.write(f'Clicked Latitude: {lat}, Clicked Longitude: {lon}')

# Add a click event to the map using ClickForMarker
folium.ClickForMarker(popup='Click to get coordinates', callback=handle_click).add_to(m)

# Display the map in Streamlit app
folium_static(m)

# Button to make prediction
if st.button('Predict Magnitude'):
    features = [latitude, longitude, depth, year, day_of_year, hour]
    prediction = predict_earthquake(features)
    st.write(f'Predicted Magnitude: {prediction[0]}')
