import yfinance as yf  #yahoo finance api to download daily stock indices
import matplotlib.pyplot as plt
ticker = "TATAMOTORS.NS"
data=yf.download(ticker)
price_history = data
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
price_history['Returns'] = price_history['Close'].pct_change() #pct = (current value - previous value)/ previous value
price_history = price_history.dropna()  
print(price_history[['Close', 'Returns']].tail(10)) #shows the percentage change in the closing price compared to the previous trading day.

import numpy as np
import scipy.stats as stats

#confidence interval and mean return
mean_return = price_history['Returns'].mean()
std_return = price_history['Returns'].std()
conf_int = stats.norm.interval(0.95, loc=mean_return, scale=std_return/np.sqrt(len(price_history)))
print("Mean Return:", mean_return)
print("95% Confidence Interval:", conf_int)


# return trend
price_history.drop(columns=['Dividends', 'Stock Splits'], inplace= True, errors="ignore")
price_history['Returns']=price_history['Close'].pct_change()
price_history=price_history.dropna()


# Don't recalculate returns again — just reuse what you already calculated
# And no need to drop Dividends/Stock Splits again since already done

# Plot Daily Return Trend
plt.figure(figsize=(10, 4))
price_history['Returns'].plot(title="Daily Returns Trend")
plt.axhline(0, color='red', linestyle='--')  # Add reference line
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.grid(True)
plt.tight_layout()
plt.show()
