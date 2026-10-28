import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Telecom Churn Prediction", page_icon="📊", layout="centered")
st.title("📊 Telecom Customer Churn Prediction")
st.write("Predict whether a customer is likely to leave the telecom service.")

BASE = Path(__file__).parent


@st.cache_resource
def load_artifacts():
    return joblib.load(BASE / "rf_model.pkl"), joblib.load(BASE / "scaler.pkl")


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

# LabelEncoder alphabetical order mein encode karta hai
YES_NO = {"No": 0, "Yes": 1}
THREE = {"No": 0, "No internet service": 1, "Yes": 2}

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.sidebar.selectbox("Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Dependents", ["No", "Yes"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple = st.sidebar.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
device = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
tech = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"],
)
monthly_charges = st.sidebar.slider("Monthly Charges", 0.0, 150.0, 70.0)
total_charges = st.sidebar.number_input("Total Charges", 0.0, 10000.0, 840.0)

if st.button("Predict Churn", type="primary"):
    row = {
        "gender": {"Female": 0, "Male": 1}[gender],
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": YES_NO[partner],
        "Dependents": YES_NO[dependents],
        "tenure": tenure,
        "PhoneService": YES_NO[phone],
        "MultipleLines": {"No": 0, "No phone service": 1, "Yes": 2}[multiple],
        "InternetService": {"DSL": 0, "Fiber optic": 1, "No": 2}[internet],
        "OnlineSecurity": THREE[security],
        "OnlineBackup": THREE[backup],
        "DeviceProtection": THREE[device],
        "TechSupport": THREE[tech],
        "StreamingTV": THREE[tv],
        "StreamingMovies": THREE[movies],
        "Contract": {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract],
        "PaperlessBilling": YES_NO[paperless],
        "PaymentMethod": {
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1,
            "Electronic check": 2,
            "Mailed check": 3,
        }[payment],
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    # Wahi columns, wahi order jo training mein the (extra columns apne aap hat jayenge)
    features = pd.DataFrame([row]).reindex(columns=scaler.feature_names_in_)

    try:
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]

        if prediction == 1:
            st.error("⚠️ Customer is likely to churn.")
        else:
            st.success("✅ Customer is likely to stay.")
        st.write(f"Churn probability: {proba * 100:.2f}%")
    except Exception as e:
        st.error(f"Prediction failed: {e}")