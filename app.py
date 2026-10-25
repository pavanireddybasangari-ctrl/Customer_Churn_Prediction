import streamlit as st
import joblib
import pandas as pd

# Load model and feature names
model = joblib.load("churn_model.pkl")
model_features = joblib.load("model_features.pkl")

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# Title
st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

st.divider()

# Customer information
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=100,
        value=12
    )

with col2:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=800.0
)

st.divider()
# Prediction
if st.button("🔍 Predict Churn", use_container_width=True):

    # Create input with all model features
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )

    # Add customer values
    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = total_charges

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Calculate churn probability
    probability = model.predict_proba(input_data)[0][1]
    probability_percent = probability * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🔴 Customer is likely to churn.")
    else:
        st.success("🟢 Customer is not likely to churn.")

    # Show probability
    st.metric(
        "Churn Probability",
        f"{probability_percent:.2f}%"
    )

    # Show risk level
    if probability_percent >= 70:
        st.error("🚨 Risk Level: HIGH")
    elif probability_percent >= 40:
        st.warning("⚠️ Risk Level: MEDIUM")
    else:
        st.success("🟢 Risk Level: LOW")

    # Probability bar
    st.progress(probability)
    # Feature Importance
st.divider()
st.subheader("📊 Feature Importance")

if hasattr(model, "coef_"):
    importance = pd.DataFrame({
        "Feature": model_features,
        "Importance": model.coef_[0]
    })

    importance["Absolute Importance"] = importance["Importance"].abs()

    importance = importance.sort_values(
        "Absolute Importance",
        ascending=False
    )

    st.bar_chart(
        importance.set_index("Feature")["Absolute Importance"].head(10)
    )

else:
    st.info(
        "Feature importance is available for Logistic Regression."
    )
# Project Information
st.divider()
st.subheader("ℹ️ About This Project")

st.write(
    """
    This Customer Churn Prediction project uses Machine Learning
    to predict whether a customer is likely to leave a service.

    The project uses customer information such as tenure,
    monthly charges and total charges.

    Machine Learning models used:
    • Logistic Regression
    • Random Forest

    The best-performing model is selected for prediction.
    """)
st.write(
    """
    This Customer Churn Prediction project uses Machine Learning
    to predict whether a customer is likely to leave a service.

    The project uses customer information such as tenure,
    monthly charges and total charges.

    Machine Learning models used:
    • Logistic Regression
    • Random Forest

    The best-performing model is selected for prediction.
    """
)
st.write(
    """
    This Customer Churn Prediction project uses Machine Learning
    to predict whether a customer is likely to leave a service.

    The project uses customer information such as tenure,
    monthly charges and total charges.

    Machine Learning models used:
    • Logistic Regression
    • Random Forest

    The best-performing model is selected for prediction.
    """
)
st.write(
    """
    This Customer Churn Prediction project uses Machine Learning
    to predict whether a customer is likely to leave a service.

    The project uses customer information such as tenure,
    monthly charges and total charges.

    Machine Learning models used:
    • Logistic Regression
    • Random Forest

    The best-performing model is selected for prediction.
    """
)
st.write(
    """
    This Customer Churn Prediction project uses Machine Learning
    to predict whether a customer is likely to leave a service.

    The project uses customer information such as tenure,
    monthly charges and total charges.

    Machine Learning models used:
    • Logistic Regression
    • Random Forest

    The best-performing model is selected for prediction.
    """
)