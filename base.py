import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

ticker = "TATAMOTORS.NS"
data = yf.download(ticker)

if data.empty:
    print("No data available for the ticker.")
else:
    print(data.head())

data.drop(columns=['Dividends', 'Stock Splits'], errors='ignore', inplace=True)
data['Date'] = data.index.map(pd.Timestamp.toordinal)
X = data[['Date', 'Open', 'High', 'Low', 'Volume']]
y = data['Close']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")

plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test):], y_test, label="Actual Prices", color='blue')
plt.plot(data.index[-len(y_test):], y_pred, label="Predicted Prices", color='red', linestyle="dashed")
plt.xlabel("Date")
plt.ylabel("Stock Closing Price")
plt.title(f"{ticker} Stock Price Prediction")
plt.legend()
plt.show()
