import pandas as pd
import yfinance as yf  #yahoo finance api to download daily stock indices
import matplotlib.pyplot as plt
ticker = "TATAMOTORS.NS"
data=yf.download(ticker)
price_history = data
if price_history.empty:
    print("No data available for the ticker.")
else:
    print(price_history)

price_history['Adjacent Close Change']=price_history['Close'].diff()# add new column for day-day change in cp
print(price_history.head())


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