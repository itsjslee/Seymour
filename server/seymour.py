import pandas as pd
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import datetime

# Function to download stock data
def download_data(stock_symbol, start_date, end_date):
    return yf.download(stock_symbol, start=start_date, end=end_date, progress=False)

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    indicators = {}

    # Ensure 'Close' is 1D
    close_prices = data['Close']

    # Bollinger Bands
    bb_indicator = BollingerBands(close_prices)
    indicators['BollingerBands'] = {
        'High': bb_indicator.bollinger_hband(),
        'Low': bb_indicator.bollinger_lband()
    }

    # MACD
    indicators['MACD'] = MACD(close_prices).macd()

    # RSI
    indicators['RSI'] = RSIIndicator(close_prices).rsi()

    # SMA
    indicators['SMA'] = SMAIndicator(close_prices, window=14).sma_indicator()

    # EMA
    indicators['EMA'] = EMAIndicator(close_prices).ema_indicator()

    return indicators

# Function to make predictions using RandomForestRegressor
def random_forest_predict(data, forecast_days):
    scaler = StandardScaler()

    # Prepare the data
    df = data[['Close']]
    df['Preds'] = df['Close'].shift(-forecast_days)
    x = scaler.fit_transform(df.drop(['Preds'], axis=1).values[:-forecast_days])
    y = df['Preds'].values[:-forecast_days]

    # Last `forecast_days` for prediction
    x_forecast = scaler.transform(df.drop(['Preds'], axis=1).values[-forecast_days:])

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

    # Train the RandomForestRegressor
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Evaluate the model
    preds = model.predict(x_test)
    print(f"Model Performance:")
    print(f"R2 Score: {r2_score(y_test, preds)}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, preds)}")

    # Forecast
    forecast = model.predict(x_forecast)
    print(f"\n{forecast_days}-Day Forecast:")
    for day, prediction in enumerate(forecast, start=1):
        print(f"Day {day}: {prediction}")

# Main function
def main():
    # Input stock details
    stock_symbol = input("Enter the stock symbol (e.g., AAPL, TSLA): ").upper()
    duration_days = int(input("Enter the number of days for historical data: "))
    forecast_days = int(input("Enter the number of days to forecast: "))

    # Set up date range
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=duration_days)

    # Download data
    data = download_data(stock_symbol, start_date, today)

    # Check if data is valid
    if data.empty:
        print(f"No data found for {stock_symbol}. Please try again with a valid stock symbol.")
        return

    # Display technical indicators
    print("\nCalculating Technical Indicators...")
    indicators = calculate_technical_indicators(data)
    for indicator, values in indicators.items():
        print(f"\n{indicator}:")
        if isinstance(values, dict):
            for subkey, subvalues in values.items():
                print(f"{subkey}: {subvalues.tail()}")
        else:
            print(values.tail())

    # Perform predictions with RandomForestRegressor
    print("\nPerforming Predictions...")
    random_forest_predict(data, forecast_days)

if __name__ == "__main__":
    main()
