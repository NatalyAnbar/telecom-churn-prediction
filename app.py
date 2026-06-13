import streamlit as st
import requests

# =====================================================================
# Page setup
# =====================================================================
st.set_page_config(page_title="Churn Prediction", page_icon="📊")
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details below to predict churn likelihood.")

# =====================================================================
# Input fields 
# =====================================================================
contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["Fiber optic", "DSL", "No"]
)

online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=150.0, value=70.0, step=0.5)
tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=72, value=12, step=1)

# =====================================================================
# API connection settings
# =====================================================================

API_URL = "http://127.0.0.1:8000/predict"
API_KEY = "churnpredict@123"  # same value as your .env file (apiKey)

# =====================================================================
# Predict button
# =====================================================================
if st.button("Predict Churn"):

    # Build payload exactly as the Customer model in main.py
    payload = {
        "contract": contract,
        "payment_method": payment_method,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "tech_support": tech_support,
        "monthly_charges": monthly_charges,
        "tenure_months": tenure_months
    }

    headers = {
        "X-API-Key": API_KEY
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code == 200:
            result = response.json()
            if result["Churn"] == "Yes":
                st.error("⚠️ This customer is likely to CHURN")
            else:
                st.success("✅ This customer is likely to STAY")
        elif response.status_code == 401:
            st.error("Authentication failed. Check your API key.")
        else:
            st.error(f"Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure FastAPI is running: `uvicorn main:app --reload`")
