import yfinance as yf  #yahoo finance api to download daily stock indices
import matplotlib.pyplot as plt
ticker = yf.Ticker("^GSPC")
price_history = ticker.history(period="max")
if price_history.empty:
    print("No data available for the ticker.")
else:
    print(price_history)
print(price_history.columns)

print(price_history.index)

price_history.plot.line(y="Close",use_index=True)   