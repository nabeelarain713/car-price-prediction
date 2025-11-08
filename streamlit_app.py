import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Load model and preprocessor
model = joblib.load('car_price_xgb_model.pkl')
preprocessor = joblib.load('xgb_preprocessor.pkl')

st.title("🚗 Car Price Prediction App")

# -------------------------
# Upload CSV for batch prediction
# -------------------------
st.header("Upload Test Data CSV for Batch Prediction and Visualization")
uploaded_file = st.file_uploader("Upload CSV (with Actual Price column)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Actual Price' not in df.columns:
        st.error("CSV must contain 'Actual Price' column for comparison")
    else:
        try:
            X_input = df.drop('Actual Price', axis=1)
            X_input_processed = preprocessor.transform(X_input)

            preds = model.predict(X_input_processed)
            df['Predicted Price'] = preds

            st.subheader("Predictions Table")
            st.dataframe(df)

            # -------------------------
            # Bar chart visualization (first 20 samples)
            # -------------------------
            n_samples = min(20, len(df))
            bar_width = 0.35
            indices = range(n_samples)

            plt.figure(figsize=(14, 6))
            plt.bar(indices, df['Actual Price'][:n_samples], width=bar_width, label='Actual', color='skyblue')
            plt.bar([i + bar_width for i in indices], df['Predicted Price'][:n_samples], width=bar_width, label='Predicted', color='orange')

            plt.xlabel("Sample Index")
            plt.ylabel("Car Price")
            plt.title("Actual vs Predicted Car Prices")
            plt.xticks([i + bar_width/2 for i in indices], [f"Car {i+1}" for i in indices])
            plt.legend()
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            st.pyplot(plt)

            # -------------------------
            # Top 5 largest errors
            # -------------------------
            df['Absolute Error'] = abs(df['Actual Price'] - df['Predicted Price'])
            top_errors = df.sort_values(by='Absolute Error', ascending=False).head(5)
            st.subheader("Top 5 Largest Errors")
            st.dataframe(top_errors)
        except Exception as e:
            st.error(f"Error in batch prediction: {e}")

