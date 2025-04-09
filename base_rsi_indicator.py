import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import yfinance as yf


def get_up_or_down(df):
    df['gain'] = 0.0
    df['loss'] = 0.0
    for i in range(1, len(df)):
        current_close = float(df['Close'].iloc[i])      # Ensure it's a float
        prev_close = float(df['Close'].iloc[i - 1])     # Ensure it's a float

        if current_close > prev_close:
            df.at[df.index[i], 'gain'] = current_close - prev_close
            df.at[df.index[i], 'loss'] = 0.0
        elif current_close < prev_close:
            df.at[df.index[i], 'gain'] = 0.0
            df.at[df.index[i], 'loss'] = prev_close - current_close
        else:
            df.at[df.index[i], 'gain'] = 0.0
            df.at[df.index[i], 'loss'] = 0.0
    return df



# Step 1: Download stock data
ticker = "TATAMOTORS.NS"
stock = yf.Ticker(ticker)
info = stock.info

print("Downloading stock data...")
data = yf.download(ticker, period="1y", interval="1d")
price_history = data.copy()

# Step 2: Check if data is valid
if price_history.empty or 'Close' not in price_history.columns or 'Open' not in price_history.columns:
    print("No data found or missing columns.")
else:
    print("Data downloaded successfully.")
    print("Last 5 rows:")
    print(price_history[['Open', 'Close']].tail())

    # Step 3: Plot Opening and Closing Prices
    plt.figure(figsize=(12, 6))
    plt.plot(price_history.index, price_history['Open'], label="Opening Price", color="orange")
    plt.plot(price_history.index, price_history['Close'], label="Closing Price", color="blue")
    plt.title("TATAMOTORS Opening and Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 4: Clean up
    price_history.drop(columns=['Dividends', 'Stock Splits'], inplace=True, errors='ignore')

    # Step 5: Calculate returns
    price_history['Returns'] = price_history['Close'].pct_change()
    price_history.dropna(inplace=True)

    # Step 6: Show return data
    print("Recent closing prices and returns:")
    print(price_history[['Close', 'Returns']].tail(10))

    # Step 7: Confidence interval
    mean_return = price_history['Returns'].mean()
    std_return = price_history['Returns'].std()
    n = len(price_history)
    conf_int = stats.norm.interval(0.95, loc=mean_return, scale=std_return / np.sqrt(n))
    print("Mean Return:", mean_return)
    print("95% Confidence Interval:", conf_int)

    # Step 8: Plot daily returns
    plt.figure(figsize=(10, 4))
    plt.plot(price_history.index, price_history['Returns'], label='Daily Returns')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("TATAMOTORS Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Step 9: Cumulative Return
    price_history['Cumulative Return'] = price_history['Close'] / price_history['Close'].iloc[0] - 1
    price_history['Cumulative Return'].plot(title=f'Cumulative Return for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()

    # Step 10: Rolling Volatility
    price_history['Daily Return'] = price_history['Close'].pct_change()
    price_history['Rolling Volatility'] = price_history['Daily Return'].rolling(window=20).std()
    price_history['Rolling Volatility'].plot(title='20-Day Rolling Volatility (Risk)')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

    # Step 11: Moving Averages
    price_history['MA20'] = price_history['Close'].rolling(window=20).mean()
    price_history['MA50'] = price_history['Close'].rolling(window=50).mean()
    plt.figure(figsize=(14, 6))
    plt.plot(price_history['Close'], label='Close Price')
    plt.plot(price_history['MA20'], label='20-Day MA')
    plt.plot(price_history['MA50'], label='50-Day MA')
    plt.title('Tata Motors Stock with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Step 12: Custom RSI using your function
    price_history = get_up_or_down(price_history)
    price_history['avg_gain'] = price_history['gain'].rolling(window=14).mean()
    price_history['avg_loss'] = price_history['loss'].rolling(window=14).mean()
    price_history['rs'] = price_history['avg_gain'] / price_history['avg_loss']
    price_history['RSI'] = 100 - (100 / (1 + price_history['rs']))

    # Step 13: Plot RSI
    plt.figure(figsize=(12, 5))
    price_history['RSI'].plot()
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Extract PE and PB Ratios
pe_ratio = info.get('trailingPE', 'N/A')
pb_ratio = info.get('priceToBook', 'N/A')

print("Pe ratio", pe_ratio)
print("Pb ratio", pb_ratio)

