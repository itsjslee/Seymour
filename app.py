import streamlit as st
from datetime import date
import matplotlib.pyplot as plt
from backend import get_stock_data, calculate_bollinger_bands, predict_prices

# Streamlit App
st.title("Make smart trades with *Seymour*.")

# User Inputs
ticker = st.text_input("Enter Stock Ticker:").upper()
start_date = st.date_input("Start Date:", value=date(2024, 1, 1))

# Fetch and process data
if st.button("Analyze"):
    st.write(f"Fetching data for {ticker} from {start_date} to today...")

    # Get stock data
    data = get_stock_data(ticker, start_date)
    if data.empty:
        st.error("No data found for the given inputs.")
    else:
        # Calculate Bollinger Bands
        data = calculate_bollinger_bands(data)

        # Plot Bollinger Bands
        st.write("### Bollinger Bands Visualization")
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Close Price', color='#ffffff')  # White
        plt.plot(data['bb_high'], label='Upper Band', linestyle='--', color='#00ff41')  # Red
        plt.plot(data['bb_low'], label='Lower Band', linestyle='--', color='#ff4b4b')  # Green
        plt.fill_between(data.index, data['bb_low'], data['bb_high'], color='gray', alpha=0.1)
        plt.title(f"{ticker}")
        plt.legend()
        st.pyplot(plt)

        # Predict future prices
        st.write("### Future Price Prediction")
        forecast = predict_prices(data)
        st.write(forecast)

        # Plot predictions
        st.write("### Prediction Visualization")
        plt.style.use('dark_background')
        plt.figure(figsize=(12, 6))
        plt.plot(data['Close'], label='Historical Close', color='#ffffff')  # White
        plt.plot(forecast['Date'], forecast['Predicted_Close'], label='Predicted Close', color='#ffff00')  # Yellow
        plt.title(f"{ticker}")
        plt.legend()
        st.pyplot(plt)
