import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Step 1: Download stock data
ticker = "TATAMOTORS.NS"
print("Downloading stock data...")
data = yf.download(ticker, period="1y", interval="1d")  # 1 year daily data
price_history = data.copy()

# Step 2: Check if data is valid
if price_history.empty or 'Close' not in price_history.columns or 'Open' not in price_history.columns:
    print("No data found or missing columns.")
else:
    print("Data downloaded successfully.")
    print("Last 5 rows:")
    print(price_history[['Open', 'Close']].tail())

    # Step 3: Plot both Opening and Closing prices
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

    # Step 8: Plot return trend
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

# Calculate daily cumulative return
price_history['Cumulative Return'] = price_history['Close'] / price_history['Close'].iloc[0] - 1

# Plot cumulative return
price_history['Cumulative Return'].plot(title=f'Cumulative Return for {ticker}')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.grid(True)
plt.show()

# Volatility rolling std
price_history['Daily Return'] = price_history['Close'].pct_change()

# Step 2: Calculate 20-day Rolling Standard Deviation (Volatility)
price_history['Rolling Volatility'] = price_history['Daily Return'].rolling(window=20).std()

# Step 3: Plot it
price_history['Rolling Volatility'].plot(title='20-Day Rolling Volatility (Risk)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.show()

#moving average
# Step 1: Calculate Moving Averages
price_history['MA20'] = price_history['Close'].rolling(window=20).mean()
price_history['MA50'] = price_history['Close'].rolling(window=50).mean()

# Step 2: Plot them with the original 'Close' price
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
