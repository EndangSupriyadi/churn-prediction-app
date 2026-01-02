import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction App", layout="centered")

st.title("📊 Customer Churn Prediction")
st.write("Model: Random Forest + Preprocessing Pipeline")

# =====================
# Load pipeline & model
# =====================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("preprocessing_pipeline.joblib")
    model = joblib.load("./model/model.pkl")
    return pipeline, model

pipeline, model = load_artifacts()

# =====================
# User Input (RAW DATA)
# =====================
st.subheader("🧾 Customer Information")

# Numeric
tenure = st.number_input("Tenure (months)", min_value=0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Binary / Categorical
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["No", "Yes"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])

phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# =====================
# Predict
# =====================
if st.button("🔮 Predict Churn"):
    input_df = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,

        "PhoneService": phone,
        "MultipleLines": multiple_lines,

        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,

        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,

        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        X_processed = pipeline.transform(input_df)
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]  # Probabilitas CHURN

        # Fokus pada CHURN
        st.write(f"⚠️ Peluang CHURN: {probability:.2%}")

        # Pie Chart
        import matplotlib.pyplot as plt

        labels = ['Stay', 'Churn']
        sizes = [1 - probability, probability]
        colors = ['#66b3ff', '#ff6666']
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Churn')

        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error("Pastikan semua input sudah benar dan lengkap.")
