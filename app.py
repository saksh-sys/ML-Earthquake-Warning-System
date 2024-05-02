import streamlit as st
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import folium  # Import Folium for map visualization

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

# Create a map for location selection
st.header('Select Location on Map')
map_center = [0, 0]  # Initial map center
map_zoom = 2  # Initial zoom level

# Display map
# my_map = folium.Map([longitude, latitude], tiles='stamentoner',   zoom_start=12)
# open(map_path, 'wb').write(m.repr_html())
# #my_map = folium.Map(location=map_center, zoom_start=map_zoom)
# folium.Marker(location=map_center, popup='Selected Location').add_to(my_map)
# st.markdown(my_map._repr_html_(), unsafe_allow_html=True)
# st.markdown('<iframe src="/map.html"> </iframe>')

# Get location from user
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
