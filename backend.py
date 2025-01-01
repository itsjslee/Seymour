import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import date, timedelta

# Fetch stock data
def get_stock_data(ticker, start_date):
    today = date.today().strftime('%Y-%m-%d')
    data = yf.download(ticker, start=start_date, end=today)
    return data

# Calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    data['bb_middle'] = data['Close'].rolling(window=window).mean()
    data['bb_std'] = data['Close'].rolling(window=window).std()
    data['bb_high'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_low'] = data['bb_middle'] - 2 * data['bb_std']
    return data

# Predict future prices
def predict_prices(data, forecast_days=30):
    data = data.dropna()  # Ensure no NaN values
    data['Days'] = np.arange(len(data))  # Numeric days for model

    # Train linear regression model
    X = data['Days'].values.reshape(-1, 1)  # Ensure 2D for scikit-learn
    y = data['Close'].values  # Ensure 1D
    model = LinearRegression().fit(X, y)

    # Predict future prices
    future_days = np.arange(len(data), len(data) + forecast_days).reshape(-1, 1)
    future_prices = model.predict(future_days).flatten()  # Ensure 1D array

    # Construct DataFrame with 1D arrays
    forecast_dates = [data.index[-1] + timedelta(days=i) for i in range(1, forecast_days + 1)]
    return pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Close': future_prices
    })

