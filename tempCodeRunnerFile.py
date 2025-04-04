import yfinance as yf  # Yahoo Finance API to download stock data
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Step 1: Download stock data
ticker = "TATAMOTORS.NS"
data = yf.download(ticker)
price_history = data

# Step 2: Check and print data
if price_history.empty:
    print("No data available for the ticker.")
else:
    print(price_history)

# Step 3: Basic info
print(price_history.columns)
print(price_history.index)

# Step 4: Initial closing price plot
price_history.plot.line(y="Close", use_index=True)
plt.title("TATAMOTORS Closing Price")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# Step 5: Clean up unnecessary columns
if 'Dividends' in price_history.columns:
    del price_history['Dividends']
if 'Stock Splits' in price_history.columns:
    del price_history['Stock Splits']

# Step 6: Replot after cleaning
price_history.plot.line(y="Close", use_index=True)
plt.title("TATAMOTORS Closing Price (Cleaned)")
plt.ylabel("Close Price")
plt.grid(True)
plt.show()

# Step 7: Calculate daily returns
price_history['Returns'] = price_history['Close'].pct_change()
price_history = price_history.dropna()

# Step 8: Show recent return data
print(price_history[['Close', 'Returns']].tail(10))

# Step 9: Confidence Interval of returns
mean_return = price_history['Returns'].mean()
std_return = price_history['Returns'].std()
conf_int = stats.norm.interval(0.95, loc=mean_return, scale=std_return / np.sqrt(len(price_history)))

print("Mean Return:", mean_return)
print("95% Confidence Interval:", conf_int)

# Step 10: Plot return trend
plt.figure(figsize=(10, 4))
price_history['Returns'].plot(title="Daily Returns Trend")
plt.axhline(0, color='red', linestyle='--')  # Horizontal line at 0
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True)
plt.tight_layout()
plt.show()
