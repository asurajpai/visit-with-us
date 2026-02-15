import streamlit as st
import pandas as pd
import joblib
import numpy as np
from huggingface_hub import hf_hub_download

# Page Config
st.set_page_config(page_title="Tourism Package Prediction", layout="wide")

st.title("Visit with Us: Wellness Package Prediction")
st.write("Enter customer details to predict the likelihood of purchasing the Wellness Tourism Package.")

# Load Model
@st.cache_resource
def load_model():
    # OPTION 1: Load local (if running locally)
    try:
        model = joblib.load("tourism_model.pkl")
    except:
        # OPTION 2: Download from HF Hub (Update repo_id)
        REPO_ID = "asurajpai/visit-with-us-app" 
        FILENAME = "tourism_model.pkl"
        model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        model = joblib.load(model_path)
    return model

model = load_model()

# Input Form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    monthly_income = st.number_input("Monthly Income", min_value=0, value=20000)
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Free Lancer'])
    gender = st.selectbox("Gender", ['Female', 'Male'])
    marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Unmarried'])

with col2:
    duration_of_pitch = st.number_input("Duration of Pitch (min)", min_value=0, value=15)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, value=3)
    product_pitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
    preferred_property_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
    type_of_contact = st.selectbox("Type of Contact", ['Self Enquiry', 'Company Invited'])

with col3:
    number_of_trips = st.number_input("Number of Trips", min_value=0, value=2)
    passport = st.selectbox("Has Passport?", [0, 1])
    own_car = st.selectbox("Owns Car?", [0, 1])
    number_of_children = st.number_input("Children Visiting", min_value=0, value=0)
    designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP'])
    number_of_person = st.number_input("Total Person Visiting", min_value=1, value=1)

# Create DataFrame for Prediction
input_data = pd.DataFrame({
    'Age': [age],
    'TypeofContact': [type_of_contact],
    'CityTier': [city_tier],
    'DurationOfPitch': [duration_of_pitch],
    'Occupation': [occupation],
    'Gender': [gender],
    'NumberOfPersonVisiting': [number_of_person],
    'NumberOfFollowups': [number_of_followups],
    'ProductPitched': [product_pitched],
    'PreferredPropertyStar': [preferred_property_star],
    'MaritalStatus': [marital_status],
    'NumberOfTrips': [number_of_trips],
    'Passport': [passport],
    'PitchSatisfactionScore': [pitch_satisfaction_score],
    'OwnCar': [own_car],
    'NumberOfChildrenVisiting': [number_of_children],
    'Designation': [designation],
    'MonthlyIncome': [monthly_income]
})

if st.button("Predict Purchase"):
    # Note: Pipeline handles imputation/encoding automatically
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.success(f"Likely to Purchase! (Probability: {probability:.2f})")
    else:
        st.warning(f"Unlikely to Purchase. (Probability: {probability:.2f})")
