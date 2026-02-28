import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Energy Forecast Interface", layout="wide")

FEATURES = [
    'Global_active_power',
    'Global_reactive_power',
    'Voltage',
    'Global_intensity',
    'Sub_metering_1',
    'Sub_metering_2',
    'Sub_metering_3',
    'Energy_kWh_min',
    'Hourly_Consumption_kWh',
    'Daily_Consumption_kWh',
    'Weekly_Consumption_kWh'
]

WINDOW = 24
N_FEATURES = 11

# -------------------------------
# LOAD MODEL + SCALERS
# -------------------------------
@st.cache_resource
def load_artifacts():

    model = load_model("energy_model.h5", compile=False)
    
    scaler = joblib.load("scaler.pkl")
    
    target_scaler = joblib.load("scaler.pkl")
    
    return model, scaler, target_scaler

model, scaler, target_scaler = load_artifacts()

# -------------------------------
# INVERSE TARGET SCALING
# -------------------------------
def inverse_targets(pred_scaled):

    return target_scaler.inverse_transform(pred_scaled)

# -------------------------------
st.title("Energy Consumption Forecast (CNN–BiLSTM–LSTM)")

mode = st.radio("Input Mode", ["Upload 24x11 CSV", "Manual Single Row (Repeat to 24)"])

input_window = None

if mode == "Upload 24x11 CSV":

    file = st.file_uploader("Upload CSV with 24 rows and 11 columns", type=["csv"]) 

    if file is not None:
        df = pd.read_csv(file)

        if df.shape != (WINDOW, N_FEATURES):
            st.error(f"CSV must be 24 x 11. Got {df.shape}")

        else:
            df.columns = FEATURES
            input_window = df.values
            st.success("Valid input window loaded")
            st.dataframe(df)

else:

    st.subheader("Enter one timestep values (repeated to 24)")

    cols = st.columns(3)
    vals = []

    for i, f in enumerate(FEATURES):
        with cols[i % 3]:
            vals.append(st.number_input(f, value=0.0, step=0.01))

    row = np.array(vals).reshape(1, -1)
    input_window = np.repeat(row, WINDOW, axis=0)

# -------------------------------
# PREDICTION
# -------------------------------
if input_window is not None:

    scaled = scaler.transform(input_window)

    X = scaled.reshape(1, WINDOW, N_FEATURES)

    if st.button("Predict"):

        preds_scaled = model.predict(X)

        preds_actual = inverse_targets(preds_scaled)

        hourly, daily, weekly = preds_actual[0]

        c1, c2, c3 = st.columns(3)

        c1.metric("Hourly kWh", float(hourly))
        c2.metric("Daily kWh", float(daily))
        c3.metric("Weekly kWh", float(weekly))

# -------------------------------
# SHAP EXPLANATION
# -------------------------------
    with st.expander("Explain Hourly Prediction (SHAP)"):

        try:

            X_flat = X.reshape(1, -1)

            try:
                background = np.load("background.npy")
            except:
                background = np.repeat(X_flat, 50, axis=0)

            def predict_hourly(data):

                data = data.reshape(data.shape[0], WINDOW, N_FEATURES)

                return model.predict(data)[:,0]

            explainer = shap.KernelExplainer(predict_hourly, background)

            shap_vals = explainer.shap_values(X_flat)

            shap_ts = shap_vals.reshape(1, WINDOW, N_FEATURES)

            shap_2d = np.mean(shap_ts, axis=1)

            X_2d = np.mean(X.reshape(1, WINDOW, N_FEATURES), axis=1)

            fig1 = plt.figure()
            shap.summary_plot(shap_2d, X_2d, feature_names=FEATURES, show=False)
            st.pyplot(fig1)

            fig2 = plt.figure()
            shap.summary_plot(shap_2d, X_2d, feature_names=FEATURES, plot_type="bar", show=False)
            st.pyplot(fig2)

        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")

st.markdown("---")
st.caption("Run using: streamlit run app.py")
