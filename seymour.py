import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys

def calculate_rsi(data, periods=14):
    """Calculate RSI using native pandas functions with proper index alignment"""
    # Calculate price changes
    delta = data.diff()
    
    # Create two copies of the price change series
    gains = delta.copy()
    losses = delta.copy()
    
    # Zero out the gains where price decreased
    gains[gains < 0] = 0
    # Zero out the losses where price increased and make positive
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate simple moving average of gains and losses
    avg_gains = gains.rolling(window=periods).mean()
    avg_losses = losses.rolling(window=periods).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gains / avg_losses
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def validate_ticker(ticker):
    """Validate if the ticker exists and has data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return True
    except:
        return False

def validate_date(date_str):
    """Validate if the date string is in correct format"""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

def run_trading_strategy(asset=None, start_date=None, params=None):
    """
    Run trading strategy with specified parameters
    """
    # Default parameters
    default_params = {
        'stop_loss': -0.7,  # 70% loss triggers complete exit
        'notify_loss': -0.3,  # 30% loss triggers trading halt
        'waitdays': 5,  # Trading halt duration in days
        'ema_period': 10,  # EMA period
        'rsi_period': 14,  # RSI period
        'rsi_oversold': 30,  # RSI oversold threshold
        'rsi_overbought': 70  # RSI overbought threshold
    }
    
    # Update defaults with provided parameters if any
    if params:
        default_params.update(params)
    params = default_params

    # Get user input if not provided
    while not asset:
        asset = input("Enter stock ticker symbol (e.g. AAPL): ").upper()
        if not validate_ticker(asset):
            print(f"Error: Could not find ticker {asset}. Please try again.")
            asset = None

    while not start_date:
        start_date = input("Enter start date (YYYY-MM-DD): ")
        if not validate_date(start_date):
            print("Error: Invalid date format. Please use YYYY-MM-DD.")
            start_date = None

    try:
        # Download data
        df = yf.download(asset, start=start_date)
        if df.empty:
            print(f"No data available for {asset} from {start_date}")
            return

        # Calculate indicators
        df['EMA10'] = df['Close'].ewm(span=params['ema_period'], adjust=False).mean()
        df['RSI'] = calculate_rsi(df['Close'], periods=params['rsi_period'])
        
        # Remove NaN values
        df = df.dropna()

        # Generate signals using .loc to ensure alignment
        df['Buy_Signal'] = False
        df['Sell_Signal'] = False
        df.loc[(df['Close'] <= df['EMA10']) & (df['RSI'] <= params['rsi_oversold']), 'Buy_Signal'] = True
        df.loc[(df['Close'] >= df['EMA10']) & (df['RSI'] >= params['rsi_overbought']), 'Sell_Signal'] = True

        # Initialize trading variables
        df['Holding'] = 0
        df['Bought'] = 0
        df['Sold'] = 0
        df['Buy_Price'] = 0.0
        df['Curr_Profit'] = 0.0
        df['Circuitbreaker'] = False
        
        # Conduct backtest
        open_pos = False
        buy_price = 0.0
        circuitbreaker = 0

        for i in df.index:
            if circuitbreaker > 0:
                df.loc[i, 'Circuitbreaker'] = True
                circuitbreaker -= 1
                continue

            if open_pos:
                df.loc[i, 'Holding'] = 1
                if buy_price == 0.0:
                    buy_price = df.loc[i, 'Open']
                df.loc[i, 'Buy_Price'] = buy_price
                df.loc[i, 'Curr_Profit'] = (df.loc[i, 'Open'] - buy_price)/buy_price

                # Check sell conditions
                if df.loc[i, 'Sell_Signal']:
                    df.loc[i, 'Sold'] = 1
                    open_pos = False
                elif df.loc[i, 'Curr_Profit'] < params['stop_loss']:
                    df.loc[i, 'Sold'] = 1
                    open_pos = False
                    print(f"Stop loss triggered on {i.strftime('%Y-%m-%d')}")
                    break
                elif df.loc[i, 'Curr_Profit'] < params['notify_loss']:
                    df.loc[i, 'Sold'] = 1
                    open_pos = False
                    circuitbreaker = params['waitdays']
                    print(f"Circuit breaker triggered on {i.strftime('%Y-%m-%d')}")

            else:
                df.loc[i, 'Holding'] = 0
                buy_price = 0.0
                if df.loc[i, 'Buy_Signal']:
                    df.loc[i, 'Bought'] = 1
                    open_pos = True

        # Shift signals for plotting
        df['Bought'] = df['Bought'].shift(1, fill_value=0)
        df['Sold'] = df['Sold'].shift(1, fill_value=0)

        # Plotting
        plt.figure(figsize=(15,7))
        plt.plot(df.index, df['Close'], label='Close', alpha=0.8)
        plt.plot(df.index, df['EMA10'], label=f'EMA{params["ema_period"]}', alpha=0.8)
        plt.scatter(df[df['Bought'] == 1].index, 
                   df[df['Bought'] == 1]['Open'], 
                   marker='^', color='g', s=100, label='Buy')
        plt.scatter(df[df['Sold'] == 1].index, 
                   df[df['Sold'] == 1]['Open'], 
                   marker='v', color='r', s=100, label='Sell')
        plt.title(f'{asset} Trading Strategy Results')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Calculate and display results
        trades = pd.concat([
            df[df['Bought'] == 1]['Open'].rename('Buys'),
            df[df['Sold'] == 1]['Open'].rename('Sells'),
            df[df['Circuitbreaker']].Circuitbreaker.rename('Circuitbreaker')
        ], axis=1)
        
        print("\nTrade Summary:")
        print(trades)

        # Calculate profits
        profits = trades.shift(-1).Sells - trades.Buys
        rel_profits = profits / trades.Buys
        total_return = np.prod(1 + rel_profits.dropna()) - 1
        
        print(f"\nStrategy Performance for {asset}:")
        print(f"Total Return: {total_return*100:.2f}%")
        print(f"Number of Trades: {len(profits.dropna())}")
        if len(profits.dropna()) > 0:
            print(f"Average Return per Trade: {rel_profits.mean()*100:.2f}%")
            print(f"Win Rate: {(rel_profits > 0).mean()*100:.2f}%")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    run_trading_strategy()