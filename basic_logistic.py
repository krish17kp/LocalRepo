import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler

def get_up_or_down(df):
    df['gain'] = 0.0
    df['loss'] = 0.0
    for i in range(1, len(df)):
        # Use .item() to extract the scalar value
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

print(df)
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
    # To prevent division by zero, you might add a small epsilon here if needed.
    price_history["rs"] = price_history["avg_gain"] / price_history["avg_loss"]
    price_history["RSI"] = 100 - (100 / (1 + price_history["rs"]))

    # Financial data
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

    # Summary DataFrame
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

    file_name = ticker + "_summary.csv"
    result_df.to_csv(file_name, index=False)

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

    # --- K-Fold with Lasso Regression ---
    model_data = price_history[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
    model_data['Next_Close'] = model_data['Close'].shift(-1)
    model_data.dropna(inplace=True)

    X = model_data[['Open', 'High', 'Low', 'Volume']]
    y = model_data['Next_Close']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = Lasso(alpha=0.1, max_iter=10000)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(lasso, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)

    print("Lasso Regression with K-Fold Cross Validation")
    print("RMSE scores for each fold:")
    print(rmse_scores)
    print("Average RMSE:")
    print(np.mean(rmse_scores))

    # --- PLOTTING SECTION ---
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

    price_history["Cumulative Return"].plot(title="Cumulative Return for " + ticker)
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    price_history["Rolling Volatility"].plot(title="20-Day Rolling Volatility (Risk) for " + ticker)
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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

    # --- LOGISTIC REGRESSION CLASSIFIER ---
    # Create a copy of the price_history DataFrame for classification
    data_for_classifier = price_history.copy()
    # Create a new column "Signal": 1 if next day's Close is higher than current day's Close, else 0
    data_for_classifier["Signal"] = (data_for_classifier["Close"].shift(-1) > data_for_classifier["Close"]).astype(int)
    # Remove rows with NaN values (e.g., last row after shift)
    data_for_classifier.dropna(inplace=True)
    
    # Define feature columns (you may adjust this list as needed)
    feature_cols = ["Returns", "Cumulative Return", "Rolling Volatility", "MA20", "MA50", "RSI"]
    
    X_clf = data_for_classifier[feature_cols]
    y_clf = data_for_classifier["Signal"]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_clf, y_clf, test_size=0.3, random_state=42)
    
    # Initialize and train the logistic regression classifier
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    print("Logistic Regression Classifier Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
