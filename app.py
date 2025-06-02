import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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
orchard = st.selectbox("Select Orchard", ['Belvedere', 'Bennett', 'Big River Produce', 'Caernarvon', 'Carina', 'Century Loxton', 
                                          'Chatfield', 'Cmv Lp', 'Domenic Cavallaro', 'Egan Rd', 'Freeman Farming', 'Gary Thorpe', 
                                          'Gone Nuts', 'Jubilee', 'Lake Powell', 'Margooya', 'Mclaren Dr', 'Meilman', 'Mullroo', 
                                          'Nick Pezzaniti', 'Nutwood', 'Outback', 'Vince Ruggiero', 'Walker Flat Almonds', 'Wemen', 
                                          'Yilgah'])
variety = st.selectbox("Select Variety", ['Carina', 'Carmel', 'Fritz', 'Independence', 'Johnson', 'Keane', 'Maxima', 'Monterey', 
                                          'Ne Plus', 'Np', 'Peerless', 'Price', 'Vella', 'Wild Type', 'Wood Colony'])

# Weather inputs
col1, col2 = st.columns(2)
with col1:
    rain_event_5plus = st.number_input("Rain in Last Event (>5 days before assessment) (ml)", value=3.0)
    rain_3rd_week_jan = st.number_input("Rain in 3rd Week of January (mm)", value=5.0)
    rain_within_10_days = st.number_input("Rain within 10 Days of Assessment (mm)", value=7.0)


with col2:
    rain_days_since_jan = st.number_input("No. of Rain Days Since Jan", value=5)
    rain_feb = st.number_input("Total Rain in February (mm)", value=10.0)
    rain_days_feb = st.number_input("No. of Rain Days in February", value=3)


# --- Predict button ---
if st.button("Predict HR Risk"):
    # Format input for encoder
    cat_input = pd.DataFrame([[orchard, variety]], columns=["Orchard", "Variety"])
    cat_encoded = encoder.transform(cat_input)
    num_input = np.array([[rain_event_5plus, rain_3rd_week_jan, rain_within_10_days,
                           rain_days_since_jan, rain_feb, rain_days_feb]])

    X_input = np.hstack([cat_encoded, num_input])
    prob = model.predict_proba(X_input)[0][1]  # Probability of class = 1

    # Display result
    st.subheader("Prediction Results")
    st.markdown(f"**Predicted HR Risk Probability:** `{prob:.2f}`")

    # Traffic light indicator
    if prob < 0.3:
        st.success("🟢 Low Risk")
    elif prob < 0.6:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")

    # --- Visualisation: Risk Gauge ---
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "HR Risk (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 60], 'color': "gold"},
                {'range': [60, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': prob * 100
            }
        }
    ))
    st.plotly_chart(fig)

    # Export result
    result = {
        "Orchard": orchard,
        "Variety": variety,
        "Rain in Last Event (>5d)": rain_event_5plus,
        "Rain 3rd Week Jan": rain_3rd_week_jan,
        "Rain within 10d": rain_within_10_days,
        "Rain Days Since Jan": rain_days_since_jan,
        "Rain in Feb": rain_feb,
        "Rain Days in Feb": rain_days_feb,
        "Predicted Risk Probability": round(prob, 3),
        "Risk Level": "Low" if prob < 0.2 else "Medium" if prob < 0.6 else "High"
    }

    result_df = pd.DataFrame([result])
    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Prediction Result", csv, "hr_risk_prediction.csv", "text/csv")

    