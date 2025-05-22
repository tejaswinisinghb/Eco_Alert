import streamlit as st
import pandas as pd
import joblib
import numpy as np
import ee

# Initialize the Earth Engine library
ee.Initialize(project='ee-srifiles4')  # Removed ee.Authenticate() to ensure proper authentication
flood_data = pd.read_csv('flood.csv')
# Load the pre-trained model
try:
    model = joblib.load('best.joblib')
except FileNotFoundError:
    st.error("Model file not found. Please ensure that 'best.joblib' exists.")
    st.stop()

# Title for the web app
st.title('Flood Prediction App')

# Collect user input for the area of interest (AOI)
st.header('Enter the Area of Interest (AOI)')
latitude = st.number_input('Latitude', value=20.5937)  # Default to central India
longitude = st.number_input('Longitude', value=78.9629)  # Default to central India

# Create AOI geometry from user input
aoi = ee.Geometry.Point([longitude, latitude])

# Function to extract specified features from GEE
import ee

# Initialize the Earth Engine library
ee.Initialize()

# Define the area of interest (AOI) as a point (India)
aoi = ee.Geometry.Point([78.9629, 20.5937])

# Function to print band names for debugging
def print_band_names(image):
    bands = image.bandNames().getInfo()
    print(f"Band names: {bands}")

# Function to extract specified features
def extract_selected_data(aoi):
    # Load datasets for each specified feature
    topography = ee.Image('USGS/SRTMGL1_003').select('elevation')
    deforestation = ee.Image('UMD/hansen/global_forest_change_2018_v1_6').select('loss')
    urbanization = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG') \
        .filterBounds(aoi) \
        .mean() \
        .select('avg_rad')
    ineffective_disaster_preparedness = ee.Image('USGS/GFSAD1000_V1').select('landcover')
    encroachments = urbanization  # Using urbanization as a proxy
    drainage_systems = topography  # Using topography as a proxy
    landslides = ee.Image('JRC/GHSL/P2016/BUILT_LDSMT_GLOBE_V1').select('built')
    deteriorating_infrastructure = ineffective_disaster_preparedness  # Using a placeholder dataset

    # Extract mean values for each selected feature at the AOI
    extracted_data = {
        'Topography': topography.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('elevation'),

        'Deforestation': deforestation.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('loss'),

        'Urbanization': urbanization.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('avg_rad'),

        'IneffectiveDisasterPreparedness': ineffective_disaster_preparedness.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('landcover'),

        'Encroachments': encroachments.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('avg_rad'),

        'DrainageSystems': drainage_systems.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('elevation'),

        'Landslides': landslides.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('built'),

        'DeterioratingInfrastructure': deteriorating_infrastructure.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=1000,
            maxPixels=1e9
        ).get('landcover')  # Use the correct band for this dataset
    }

    # Convert dictionary values to a tuple
    return tuple(extracted_data[key].getInfo() if extracted_data[key] is not None else None for key in extracted_data)


# Execute the function and print the results as a tuple
result = extract_selected_data(aoi)
print('Extracted Data:', result)


# Extract GEE data
result = extract_selected_data(aoi)

# Convert extracted data to a DataFrame-compatible format
mean_values = {
    'TopographyDrainage': result[0]/100,
    'Deforestation': result[1],
    'Urbanization': result[2]*10,
    'IneffectiveDisasterPreparedness': result[3],
    'Encroachments': result[4],
    'DrainageSystems': result[5],
    'Landslides': result[6],
    'DeterioratingInfrastructure': result[7],
    'ClimateChange': flood_data['ClimateChange'].mean(),  # Replace placeholder with actual mean
    'DamsQuality': flood_data['DamsQuality'].mean(),       # Replace placeholder with actual mean
    'Siltation': flood_data['Siltation'].mean(),           # Replace placeholder with actual mean
    'AgriculturalPractices': flood_data['AgriculturalPractices'].mean(),  # Replace placeholder with actual mean
    'CoastalVulnerability': flood_data['CoastalVulnerability'].mean(),     # Replace placeholder with actual mean
    'Watersheds': flood_data['Watersheds'].mean(),           # Replace placeholder with actual mean
    'PopulationScore': flood_data['PopulationScore'].mean(),   # Replace placeholder with actual mean
    'WetlandLoss': flood_data['WetlandLoss'].mean(),           # Replace placeholder with actual mean
    'InadequatePlanning': flood_data['InadequatePlanning'].mean(),  # Replace placeholder with actual mean
    'PoliticalFactors': flood_data['PoliticalFactors'].mean(),   # Replace placeholder with actual mean
    'MonsoonIntensity': flood_data['MonsoonIntensity'].mean(),    # Replace placeholder with actual mean
    'RiverManagement': flood_data['RiverManagement'].mean() 
}

# Create a DataFrame for the input data
input_data = pd.DataFrame({
    'MonsoonIntensity': [mean_values['MonsoonIntensity']],
    'TopographyDrainage': [mean_values['TopographyDrainage']],
    'RiverManagement': [mean_values['RiverManagement']],
    'Deforestation': [mean_values['Deforestation']],
    'Urbanization': [mean_values['Urbanization']],
    'ClimateChange': [mean_values['ClimateChange']],
    'DamsQuality': [mean_values['DamsQuality']],
    'Siltation': [mean_values['Siltation']],
    'AgriculturalPractices': [mean_values['AgriculturalPractices']],
    'Encroachments': [mean_values['Encroachments']],
    'IneffectiveDisasterPreparedness': [mean_values['IneffectiveDisasterPreparedness']],
    'DrainageSystems': [mean_values['DrainageSystems']],
    'CoastalVulnerability': [mean_values['CoastalVulnerability']],
    'Landslides': [mean_values['Landslides']],
    'Watersheds': [mean_values['Watersheds']],
    'DeterioratingInfrastructure': [mean_values['DeterioratingInfrastructure']],
    'PopulationScore': [mean_values['PopulationScore']],
    'WetlandLoss': [mean_values['WetlandLoss']],
    'InadequatePlanning': [mean_values['InadequatePlanning']],
    'PoliticalFactors': [mean_values['PoliticalFactors']]
})

# Display the input data
st.subheader('Input Data')
st.write(input_data)

# Button to make prediction
if st.button('Predict'):
    # Make predictions with the loaded model
    prediction = model.predict(input_data)
    
    # Interpret the prediction

    st.subheader('Prediction Result')
    st.write("Probability of Flood: .",prediction[0]*100)
    
