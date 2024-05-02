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
with st.sidebar:
        selected =option_menu(
            menu_title=None,
            options=["TEXT", "IMAGE", "CONTACT"],
            icons=["cursor-text","card-image","person-lines-fill"],
            default_index=0,
        )


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
    m = folium.Map(location=[latitude, longitude], zoom_start=6, max_bounds=True)

    # Add marker for selected location
    folium.Marker([latitude, longitude], popup='Selected Location').add_to(m)

    # Display the map in Streamlit app
    folium_static(m)

    # Button to make prediction
    if st.button('Predict Magnitude'):
        features = [latitude, longitude, depth, year, day_of_year, hour]
        prediction = predict_earthquake(features)
        st.write(f'Predicted Magnitude: {prediction[0]}')

elif page == 'Contributors':
    st.title('Contributors')
    contributors = {
        "Saksham Raj Gupta": {
            "email": "sakshamrajg@gmail.com",
            "github_link": "https://github.com/sakshsys",
            "details": {
                "Reg No": "12016513",
                "University": "Lovely Professional University"
            }
        },
        "Amardeep Singh Gujraal": {
            "email": "gujraal2006@gmail.com",
            "github_link": "https://github.com/amartist",
            "details": {
                "Reg No": "12006933",
                "University": "Lovely Professional University"
            }
        },
        "Royal Chaudhary": {
            "email": "roychaudhary1999@icloud.com",
            "github_link": "https://github.com/Royal-Chaudhary",
            "details": {
                "Reg No": "12016265",
                "University": "Lovely Professional University"
            }
        },
    }

    for name, info in contributors.items():
        st.markdown(
            '''
            <style>
            div[data-testid="stExpander"] details div[data-testid="stExpanderContent"] summary {
                font-size: 1.2rem;
                color: blue;
                /* Add any other styles you want */
            }
            </style>
            ''',
            unsafe_allow_html=True
        )

        with st.expander(name):
            st.write(f"**Email:** {info['email']}", unsafe_allow_html=True)
            st.write(f"**GitHub:** [{name}]({info['github_link']})")
            st.write(f"**Reg No:** {info['details']['Reg No']}")
            st.write(f"**University:** {info['details']['University']}")
