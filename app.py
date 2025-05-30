import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model and encoder
try:
    model = joblib.load("hr_risk_model.pkl")
    encoder = joblib.load("encoder.pkl")
except Exception as e:
    st.error(f"Model or encoder could not be loaded: {e}")
    st.stop()

st.title("🌾 Almond Hull Rot Risk Predictor")

# --- Input fields ---
st.subheader("Enter Orchard and Weather Information")
orchard = st.text_input("Enter Orchard Name", value="Belvedere")
variety = st.selectbox("Select Variety", ["NP", "Monterey", "Carina", "Price", "Unknown"])

# Weather inputs
col1, col2 = st.columns(2)
with col1:
    rain_total = st.number_input("Total Rain Since Jan (ml)", value=20.0)
    rain_jan = st.number_input("Rain in January (mm)", value=10.0)
    rain_feb = st.number_input("Rain in February (mm)", value=5.0)
    rain_event = st.number_input("Rain in Last Event (<5 days) (mm)", value=2.0)

with col2:
    days_35 = st.number_input("Days Over 35°C", value=8)
    days_40 = st.number_input("Days Over 40°C", value=3)
    rain_days = st.number_input("No. of Rainy Days Since Jan", value=5)

# --- Predict button ---
if st.button("Predict HR Risk"):
    # Format input for encoder
    cat_input = pd.DataFrame([[orchard, variety]], columns=["Orchard", "Variety"])
    cat_encoded = encoder.transform(cat_input)
    num_input = np.array([[days_40, days_35, rain_total, rain_days, rain_event, rain_jan, rain_feb]])

    X_input = np.hstack([cat_encoded, num_input])
    prob = model.predict_proba(X_input)[0][1]  # Probability of class = 1

    # Display result
    st.subheader("Prediction Results")
    st.markdown(f"**Predicted HR Risk Probability:** `{prob:.2f}`")

    # Traffic light indicator
    if prob < 0.2:
        st.success("🟢 Low Risk")
    elif prob < 0.5:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # Optionally export result
    result = {
        "Orchard": orchard,
        "Variety": variety,
        "Predicted Risk Probability": round(prob, 3),
        "Risk Level": "Low" if prob < 0.2 else "Medium" if prob < 0.5 else "High"
    }
    result_df = pd.DataFrame([result])
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Prediction Result", csv, "hr_risk_prediction.csv", "text/csv")