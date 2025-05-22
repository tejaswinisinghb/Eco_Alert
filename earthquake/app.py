import streamlit as st
import pandas as pd
import joblib
import requests
import ee
from twilio.rest import Client

# Authenticate and initialize the Earth Engine library
ee.Authenticate()
ee.Initialize(project='ee-srifiles4')

# Load the pre-trained model with error handling
try:
    rf = joblib.load('random.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure that 'random.joblib' exists.")
    st.stop()

# Title for the web app
st.title('Earthquake Prediction App')

# Collect user input for the earthquake location
st.header('Enter Earthquake Location')
latitude = st.number_input('Latitude', value=34.05)
longitude = st.number_input('Longitude', value=-118.25)

# Function to fetch earthquake data from USGS API
def fetch_earthquake_data(latitude, longitude, radius=10):
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        'format': 'geojson',
        'latitude': latitude,
        'longitude': longitude,
        'maxradius': radius,
        'orderby': 'time',
        'limit': 10
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching earthquake data: {response.status_code}")
        return None

# Fetch earthquake data based on user input
earthquake_data = fetch_earthquake_data(latitude, longitude)

# Initialize lists to collect MMI and CDI values
mmi_values = []
cdi_values = []

# Display earthquake data if available
if earthquake_data:
    st.subheader('Recent Earthquake Data')
    for feature in earthquake_data['features']:
        properties = feature['properties']
        geometry = feature['geometry']

        # Collect MMI and CDI values if available
        if properties.get('mmi') is not None:
            mmi_values.append(properties['mmi'])
        if properties.get('cdi') is not None:
            cdi_values.append(properties['cdi'])

# Calculate means for MMI and CDI if available
mean_mmi = sum(mmi_values) / len(mmi_values) if mmi_values else 30  # Default to 30 if no MMI data
mean_cdi = sum(cdi_values) / len(cdi_values) if cdi_values else 20  # Default to 20 if no CDI data

# Function to extract specified features from GEE and return a tuple
def extract_selected_data(latitude, longitude):
    aoi = ee.Geometry.Point([longitude, latitude])
    
    # Load datasets and extract mean values
    topography = ee.Image('USGS/SRTMGL1_003').select('elevation')
    deforestation = ee.Image('UMD/hansen/global_forest_change_2018_v1_6').select('loss')
    urbanization = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
        .filterBounds(aoi) \
        .mean() \
        .select('avg_rad')
    
    # Extract values
    topography_value = topography.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=1000,
        maxPixels=1e9
    ).get('elevation').getInfo()
    
    deforestation_value = deforestation.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=1000,
        maxPixels=1e9
    ).get('loss').getInfo()
    
    urbanization_value = urbanization.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=1000,
        maxPixels=1e9
    ).get('avg_rad').getInfo()
    
    return (topography_value, deforestation_value, urbanization_value)

# Extract features from GEE
gee_data = extract_selected_data(latitude, longitude)

# Prepare input data for the model in the required order
input_data = pd.DataFrame({
    'latitude': [latitude],
    'longitude': [longitude],
    'depth': [10.0],  # Example placeholder; replace with actual depth if available
    'sig': [earthquake_data['features'][0]['properties']['sig']] if earthquake_data else [500],
    'mmi': [mean_mmi],
    'cdi': [mean_cdi],
})

st.subheader('Input Data for Prediction')
st.write(input_data)

# Twilio configuration
account_sid = 'Your_Account'  # Replace with your Account SID
auth_token = 'Your_Token'    # Replace with your Auth Token
client = Client(account_sid, auth_token)

def send_sms_alert(phone_number, message):
    """
    Send an SMS alert to a specific phone number.
    
    :param phone_number: The recipient's phone number (e.g., '+1234567890')
    :param message: The message to send
    """
    try:
        message = client.messages.create(
            body=message,
            from_='+12512732373',  # Replace with your Twilio number
            to=phone_number
        )
        print(f"Message sent to {phone_number}: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS alert: {e}")

# Button to make prediction
if st.button('Predict'):
    # Make predictions with the loaded model
    predicted_magnitude = rf.predict(input_data)
    st.subheader('Predicted Magnitude')
    st.write(f"{predicted_magnitude[0]:.2f}")  # Format the output to two decimal places

    # Determine the severity based on the predicted magnitude
    severity = ""
    message = ""
    if predicted_magnitude[0] < 4:
        severity = "light"
        message = f"Light earthquake predicted with a magnitude of {predicted_magnitude[0]:.2f}. Stay alert."
    elif 4 <= predicted_magnitude[0] <= 6.5:
        severity = "moderate"
        message = f"Moderate earthquake predicted with a magnitude of {predicted_magnitude[0]:.2f}. Please take necessary precautions."
    else:
        severity = "severe"
        message = f"Severe earthquake predicted with a magnitude of {predicted_magnitude[0]:.2f}. Evacuate to a safe location immediately."

    st.subheader('Severity')
    st.write(severity.capitalize())

    # Send SMS alert with the severity message
    send_sms_alert('+919030342481', message)
