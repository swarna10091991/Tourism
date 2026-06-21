
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load model
model_path = hf_hub_download(
    repo_id="SwarnaP/Tourism",
    filename="best_tourism_model_v1.joblib"
)

model = joblib.load(model_path)

# Streamlit UI
st.title("Tourism Package Purchase Prediction")

st.write("""
This application predicts whether a customer is likely to purchase a tourism package
based on demographic and behavioral information.
Please enter the customer details below.
""")

# Categorical inputs
typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced"]
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

# Numeric inputs
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

citytier = st.number_input(
    "City Tier",
    min_value=1,
    max_value=3,
    value=1
)

durationofpitch = st.number_input(
    "Duration Of Pitch",
    min_value=0,
    value=15
)

numberofpersonvisiting = st.number_input(
    "Number Of Persons Visiting",
    min_value=1,
    value=2
)

numberoffollowups = st.number_input(
    "Number Of Followups",
    min_value=0,
    value=2
)

preferredpropertystar = st.number_input(
    "Preferred Property Star",
    min_value=1,
    max_value=5,
    value=3
)

numberoftrips = st.number_input(
    "Number Of Trips",
    min_value=0,
    value=2
)

passport = st.selectbox(
    "Passport Available",
    [0, 1]
)

pitchsatisfactionscore = st.number_input(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

owncar = st.selectbox(
    "Own Car",
    [0, 1]
)

numberofchildrenvisiting = st.number_input(
    "Number Of Children Visiting",
    min_value=0,
    value=0
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=1000,
    value=30000
)

# Create input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": citytier,
    "DurationOfPitch": durationofpitch,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "NumberOfFollowups": numberoffollowups,
    "PreferredPropertyStar": preferredpropertystar,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "MonthlyIncome": monthlyincome,
    "TypeofContact": typeofcontact,
    "Occupation": occupation,
    "Gender": gender,
    "ProductPitched": productpitched,
    "MaritalStatus": maritalstatus,
    "Designation": designation
}])

# Prediction
if st.button("Predict"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Customer is likely to purchase the tourism package.")
    else:
        st.error("Customer is unlikely to purchase the tourism package.")
