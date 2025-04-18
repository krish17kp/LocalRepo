import yfinance as yf  
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
plt.show()

if 'Dividends' in price_history.columns:
    del price_history['Dividends']
    
if 'Stock Splits' in price_history.columns:
    del price_history['Stock Splits']

price_history.plot.line(y="Close", use_index=True)
plt.show()
