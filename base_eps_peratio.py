import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

def get_up_or_down(df):
    df['gain'] = 0.0
    df['loss'] = 0.0
    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i].item()
        prev_close = df['Close'].iloc[i - 1].item()
        if current_close > prev_close:
            df.at[df.index[i], 'gain'] = current_close - prev_close
        elif current_close < prev_close:
            df.at[df.index[i], 'loss'] = prev_close - current_close
    return df

# Set your stock ticker here
ticker = "MAZDOCK.NS"
stock = yf.Ticker(ticker)
info = stock.info

print("Downloading stock data...")
data = yf.download(ticker, period="2y", interval="1d")
price_history = data.copy()

if price_history.empty or "Close" not in price_history.columns:
    print("No data found for this ticker.")
else:
    # Clean up data
    price_history.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors="ignore")
    price_history["Returns"] = price_history["Close"].pct_change()
    price_history.dropna(inplace=True)

    # Compute basic statistics
    mean_return = price_history["Returns"].mean()
    std_return = price_history["Returns"].std()
    n = len(price_history)
    conf_int = stats.norm.interval(0.95, loc=mean_return, scale=std_return / np.sqrt(n))

    # Compute technical indicators
    price_history["Cumulative Return"] = price_history["Close"] / price_history["Close"].iloc[0] - 1
    price_history["Daily Return"] = price_history["Close"].pct_change()
    price_history["Rolling Volatility"] = price_history["Daily Return"].rolling(window=20).std()
    price_history["MA20"] = price_history["Close"].rolling(window=20).mean()
    price_history["MA50"] = price_history["Close"].rolling(window=50).mean()

    price_history = get_up_or_down(price_history)
    price_history["avg_gain"] = price_history["gain"].rolling(window=14).mean()
    price_history["avg_loss"] = price_history["loss"].rolling(window=14).mean()
    price_history["rs"] = price_history["avg_gain"] / price_history["avg_loss"]
    price_history["RSI"] = 100 - (100 / (1 + price_history["rs"]))


    # Financial data: today's price, EPS, Book Value, then calculate ratios
    price_today = stock.history(period="1d")["Close"].iloc[0]
    eps = info.get("trailingEps", None)
    book_value = info.get("bookValue", None)

    if eps and eps != 0:
        pe_ratio = price_today / eps
    else:
        pe_ratio = "N/A"

    if book_value and book_value != 0:
        pb_ratio = price_today / book_value
    else:
        pb_ratio = "N/A"

    # Create a summary DataFrame
    summary_data = {
        "Stock Ticker": [ticker],
        "Latest Price": [price_today],
        "EPS": [eps],
        "P/E Ratio": [pe_ratio],
        "Book Value": [book_value],
        "P/B Ratio": [pb_ratio],
        "Mean Daily Return": [mean_return],
        "95% CI Low": [conf_int[0]],
        "95% CI High": [conf_int[1]]
    }
    result_df = pd.DataFrame(summary_data)

    # Save the summary to CSV
    file_name = ticker + "_summary.csv"
    result_df.to_csv(file_name, index=False)

    # Print the summary
    print("Summary for stock:")
    print(ticker)
    print("Latest Price:")
    print(price_today)
    print("EPS:")
    print(eps)
    print("P/E Ratio:")
    print(pe_ratio)
    print("Book Value per Share:")
    print(book_value)
    print("P/B Ratio:")
    print(pb_ratio)
    print("Mean Daily Return:")
    print(mean_return)
    print("95% Confidence Interval:")
    print(conf_int)
    print("Data saved to CSV file:")
    print(file_name)

    # --- PLOTTING SECTION ---

    # Plot Opening and Closing Prices
    plt.figure(figsize=(12, 6))
    plt.plot(price_history.index, price_history["Open"], label="Opening Price", color="orange")
    plt.plot(price_history.index, price_history["Close"], label="Closing Price", color="blue")
    plt.title(ticker + " Opening and Closing Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Daily Returns
    plt.figure(figsize=(10, 4))
    plt.plot(price_history.index, price_history["Returns"], label="Daily Returns")
    plt.axhline(0, color="red", linestyle="--")
    plt.title(ticker + " Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot Cumulative Return
    price_history["Cumulative Return"].plot(title="Cumulative Return for " + ticker)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Rolling Volatility
    price_history["Rolling Volatility"].plot(title="20-Day Rolling Volatility (Risk) for " + ticker)
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Moving Averages
    plt.figure(figsize=(14, 6))
    plt.plot(price_history["Close"], label="Close Price")
    plt.plot(price_history["MA20"], label="20-Day MA")
    plt.plot(price_history["MA50"], label="50-Day MA")
    plt.title(ticker + " Stock with Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot RSI
    plt.figure(figsize=(12, 5))
    price_history["RSI"].plot()
    plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
    plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
    plt.title("Relative Strength Index (RSI) for " + ticker)
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
