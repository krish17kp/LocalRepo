import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import statsmodels.api as sm

def get_up_or_down(df):
    """
    Calculate daily 'gain' and 'loss' based on consecutive Close prices.
    Ensures we use scalar values rather than entire Series.
    """
    # Make sure Close is numeric
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    
    # Initialize gain/loss columns
    df['gain'] = 0.0
    df['loss'] = 0.0
    
    for i in range(1, len(df)):
        current_close = df['Close'].iloc[i]
        prev_close = df['Close'].iloc[i - 1]
        
        # Skip if either is NaN
        if pd.isna(current_close) or pd.isna(prev_close):
            continue
        
        if current_close > prev_close:
            df.loc[df.index[i], 'gain'] = current_close - prev_close
        elif current_close < prev_close:
            df.loc[df.index[i], 'loss'] = prev_close - current_close
    
    return df

# ============ MAIN SCRIPT ============

if __name__ == "__main__":
    
    # Ticker symbol you want to analyze
    ticker = "MAZDOCK.NS"
    
    # Create a Ticker object
    stock = yf.Ticker(ticker)
    
    # Safely retrieve stock info (some versions use stock.info)
    try:
        info = stock.get_info()  # If older yfinance, you might need: info = stock.info
    except Exception as e:
        print("Error retrieving stock info:", e)
        info = {}
    
    # Download historical data
    # Adjust 'period' and 'interval' to your preference
    data = yf.download(ticker, period="2y", interval="1d")
    price_history = data.copy()
    
    # Basic check for data
    if price_history.empty or "Close" not in price_history.columns:
        print("No data found for", ticker)
    else:
        # Remove any columns not needed
        price_history.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors="ignore")
        
        # Ensure numeric for key columns
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            price_history[col] = pd.to_numeric(price_history[col], errors='coerce')
        
        # Drop rows where 'Close' is NaN
        price_history.dropna(subset=["Close"], inplace=True)
        
        # Compute daily returns; remove NaNs from the shift
        price_history["Returns"] = price_history["Close"].pct_change()
        price_history.dropna(inplace=True)
        
        # Mean, std, and sample size for the returns
        mean_return = price_history["Returns"].mean()
        std_return = price_history["Returns"].std()
        n = len(price_history)
        
        # 95% confidence interval for the mean (assuming normal)
        conf_int = stats.norm.interval(0.95, loc=mean_return, scale=std_return/np.sqrt(n))
        
        # Additional columns
        price_history["Cumulative Return"] = (
            price_history["Close"] / price_history["Close"].iloc[0] - 1
        )
        price_history["Daily Return"] = price_history["Close"].pct_change()
        price_history["Rolling Volatility"] = price_history["Daily Return"].rolling(window=20).std()
        price_history["MA20"] = price_history["Close"].rolling(window=20).mean()
        price_history["MA50"] = price_history["Close"].rolling(window=50).mean()
        
        # Calculate gain/loss and RSI
        price_history = get_up_or_down(price_history)
        price_history["avg_gain"] = price_history["gain"].rolling(window=14).mean()
        price_history["avg_loss"] = price_history["loss"].rolling(window=14).mean()
        
        epsilon = 1e-12
        price_history["rs"] = price_history["avg_gain"] / (price_history["avg_loss"] + epsilon)
        price_history["RSI"] = 100 - (100 / (1 + price_history["rs"]))
        
        # Retrieve today's price if available
        try:
            current_day_data = stock.history(period="1d")
            if current_day_data.empty:
                print("No '1d' data found for current price.")
                price_today = np.nan
            else:
                # Extract last close as float
                last_close_val = current_day_data["Close"].iloc[-1]
                try:
                    price_today = float(last_close_val)
                except Exception:
                    price_today = np.nan
        except Exception as e:
            print("Error retrieving current price:", e)
            price_today = np.nan
        
        # Try converting EPS, Book Value to float (some data can be None or unusual types)
        eps_raw = info.get("trailingEps", np.nan)
        try:
            eps = float(eps_raw)
        except Exception:
            eps = np.nan
        
        book_value_raw = info.get("bookValue", np.nan)
        try:
            book_value = float(book_value_raw)
        except Exception:
            book_value = np.nan
        
        # Safely compute P/E, P/B
        if not np.isnan(eps) and not np.isnan(price_today) and eps != 0:
            pe_ratio = price_today / eps
        else:
            pe_ratio = np.nan
        
        if not np.isnan(book_value) and not np.isnan(price_today) and book_value != 0:
            pb_ratio = price_today / book_value
        else:
            pb_ratio = np.nan
        
        # Add ratio columns
        price_history["PE Ratio"] = pe_ratio
        price_history["PB Ratio"] = pb_ratio
        price_history["95% CI Low"] = conf_int[0]
        price_history["95% CI High"] = conf_int[1]
        
        # Create summary data
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
        result_df.to_csv(ticker + "_summary.csv", index=False)
        
        # ===== LASSO REGRESSION =====
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
        
        # ===== PLOTTING =====
        
        # (1) Plot Open vs. Close
        plt.figure(figsize=(12, 6))
        plt.plot(price_history.index, price_history["Open"], label="Opening Price", color="orange")
        plt.plot(price_history.index, price_history["Close"], label="Closing Price", color="blue")
        plt.title(f"{ticker} Opening and Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # (2) Daily Returns with Confidence Interval lines
        plt.figure(figsize=(10, 5))
        plt.plot(price_history.index, price_history["Returns"], label="Daily Returns")
        plt.axhline(conf_int[0], color="green", linestyle="--", label="95% CI Low")
        plt.axhline(conf_int[1], color="red", linestyle="--", label="95% CI High")
        plt.axhline(0, color="black", linestyle=":")
        plt.legend()
        plt.title("Daily Returns with Confidence Interval")
        plt.tight_layout()
        plt.show()
        
        # (3) Histogram of Returns + Normal Fit
        plt.figure(figsize=(10, 5))
        sns.histplot(price_history["Returns"], kde=True, bins=50, stat='density')
        xmin, xmax = plt.xlim()
        x_vals = np.linspace(xmin, xmax, 100)
        p_vals = stats.norm.pdf(x_vals, mean_return, std_return)
        plt.plot(x_vals, p_vals, 'k', linewidth=2)
        plt.title("Histogram of Returns with Normal Fit")
        plt.tight_layout()
        plt.show()
        
        # (4) QQ Plot of Daily Returns
        sm.qqplot(price_history["Returns"], line='s')
        plt.title("QQ Plot of Daily Returns")
        plt.tight_layout()
        plt.show()
        
        # (5) Z-test: Is mean return = 0?
        z_stat = mean_return / (std_return / np.sqrt(n))
        p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        print(f"Z-test statistic = {z_stat:.4f}, p-value = {p_val:.4f}")
        if p_val < 0.05:
            print("Reject null hypothesis: Return is significantly different from 0.")
        else:
            print("Fail to reject null hypothesis: Return is not significantly different from 0.")
        
        # (6) RSI Plot
        plt.figure(figsize=(12, 5))
        price_history["RSI"].plot()
        plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
        plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
        plt.title("Relative Strength Index (RSI)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Print summary DataFrame and Lasso result
        print(result_df)
        print("Average RMSE (Lasso Regression):", np.mean(rmse_scores))
        
        # ===== LOGISTIC REGRESSION CLASSIFIER =====
        data_for_classifier = price_history.copy()
        # Create a Signal column: 1 if next day's Close is higher, else -1
        data_for_classifier["Signal"] = np.where(
            data_for_classifier["Close"].shift(-1) > data_for_classifier["Close"], 1, -1
        )
        
        feature_cols = [
            "Returns", "Cumulative Return", "Daily Return", "Rolling Volatility",
            "MA20", "MA50", "RSI", "PE Ratio", "PB Ratio", "95% CI Low", "95% CI High"
        ]
        
        # Ensure columns exist and are numeric
        for col in feature_cols + ["Signal"]:
            if col not in data_for_classifier.columns:
                data_for_classifier[col] = np.nan
            data_for_classifier[col] = pd.to_numeric(data_for_classifier[col], errors="coerce")
        
        # Drop rows with NaN in features or Signal
        data_for_classifier.dropna(subset=feature_cols + ["Signal"], inplace=True)
        
        X_clf = data_for_classifier[feature_cols]
        y_clf = data_for_classifier["Signal"]
        X_train, X_test, y_train, y_test = train_test_split(
            X_clf, y_clf, test_size=0.25, random_state=42
        )
        
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
