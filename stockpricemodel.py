import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score, confusion_matrix, classification_report, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import datetime


#   HELPER FUNCTION: RSI

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


#     STOCK DATA & PREPROCESSING

ticker = "TATAMOTORS.NS"
stock = yf.Ticker(ticker)
info = stock.info

# Get full historical data and remove unwanted columns
df = stock.history(period="max")
df.drop(columns=["Dividends", "Stock Splits"], inplace=True, errors="ignore")

# Create additional columns for next day's close and a binary Target column
df["Tomorrow"] = df["Close"].shift(-1)
df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

# Limit data to a specific period and drop NaN rows
df = df.loc["1990-01-01":].copy()


#     TECHNICAL INDICATORS

df["Returns"] = df["Close"].pct_change()
df["Cumulative Return"] = df["Close"] / df["Close"].iloc[0] - 1
df["Daily Return"] = df["Close"].pct_change()
df["Rolling Volatility"] = df["Daily Return"].rolling(window=20).std()
df["MA20"] = df["Close"].rolling(window=20).mean()
df["MA50"] = df["Close"].rolling(window=50).mean()

# RSI Calculation using the helper function
df = get_up_or_down(df)
df["avg_gain"] = df["gain"].rolling(window=14).mean()
df["avg_loss"] = df["loss"].rolling(window=14).mean()
df["rs"] = df["avg_gain"] / df["avg_loss"]
df["RSI"] = 100 - (100 / (1 + df["rs"]))
df.dropna(inplace=True)


#     FINANCIAL RATIOS & SUMMARY

price_today = stock.history(period="1d")["Close"].iloc[0]
eps = info.get("trailingEps", None)
book_value = info.get("bookValue", None)
pe_ratio = price_today / eps if eps and eps != 0 else "N/A"
pb_ratio = price_today / book_value if book_value and book_value != 0 else "N/A"

mean_return = df["Returns"].mean()
std_return = df["Returns"].std()
n = len(df)
conf_int = stats.norm.interval(0.95, loc=mean_return, scale=std_return / np.sqrt(n))

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
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(ticker + "_summary.csv", index=False)
print("\nStock Summary:")
print(summary_df)


#       LASSO REGRESSION (K-Fold)

model_data = df[['Open', 'High', 'Low', 'Volume', 'Close']].copy()
model_data['Next_Close'] = model_data['Close'].shift(-1)
model_data.dropna(inplace=True)
X = model_data[['Open', 'High', 'Low', 'Volume']]
y = model_data['Next_Close']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
lasso = Lasso(alpha=0.1, max_iter=10000)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lasso_scores = cross_val_score(lasso, X_scaled, y, cv=kf, scoring='neg_mean_squared_error')
print("\nLasso Regression K-Fold RMSEs:", np.sqrt(-lasso_scores))
print("Average RMSE:", np.mean(np.sqrt(-lasso_scores)))


#  LOGISTIC REGRESSION CLASSIFIER (K-Fold)

clf_data = df.copy()
clf_data["Signal"] = (clf_data["Close"].shift(-1) > clf_data["Close"]).astype(int)
clf_data.dropna(inplace=True)
feature_cols = ["Returns", "Cumulative Return", "Rolling Volatility", "MA20", "MA50", "RSI"]
X_clf = clf_data[feature_cols]
y_clf = clf_data["Signal"]
log_model = LogisticRegression(max_iter=1000)
kf_clf = KFold(n_splits=5, shuffle=True, random_state=42)
log_precisions = []
for train_i, test_i in kf_clf.split(X_clf):
    log_model.fit(X_clf.iloc[train_i], y_clf.iloc[train_i])
    preds = log_model.predict(X_clf.iloc[test_i])
    log_precisions.append(precision_score(y_clf.iloc[test_i], preds))
print("\nLogistic Regression K-Fold Precision Scores:", log_precisions)
print("Average Precision:", np.mean(log_precisions))


#    RANDOM FOREST (K-Fold) WITH ADDITIONAL FEATURES

horizons = [2, 5, 60, 250, 1000]
new_predictors = []
for h in horizons:
    df[f"Close_Ratio_{h}"] = df["Close"] / df["Close"].rolling(h).mean()
    df[f"Trend_{h}"] = df["Target"].shift(1).rolling(h).sum()
    new_predictors += [f"Close_Ratio_{h}", f"Trend_{h}"]
df.dropna(inplace=True)

rf_model = RandomForestClassifier(n_estimators=200, min_samples_split=50, random_state=1)
X_rf = df[new_predictors]
y_rf = df["Target"]
kf_rf = KFold(n_splits=5, shuffle=True, random_state=42)
rf_precisions = []
for train_idx, test_idx in kf_rf.split(X_rf):
    rf_model.fit(X_rf.iloc[train_idx], y_rf.iloc[train_idx])
    rf_preds = rf_model.predict(X_rf.iloc[test_idx])
    rf_precisions.append(precision_score(y_rf.iloc[test_idx], rf_preds))
print("\nRandom Forest K-Fold Precision Scores:", rf_precisions)
print("Average Precision:", np.mean(rf_precisions))


#       LINEAR REGRESSION (OLS)

X_lr = df[["Open", "High", "Low", "Volume"]].values
y_lr = df["Close"].values
X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=0)
lr_model = LinearRegression().fit(X_train_lr, y_train_lr)
y_pred_lr = lr_model.predict(X_test_lr)
print("\nLinear Regression Coefficients:", lr_model.coef_)
print("R² Score:", lr_model.score(X_test_lr, y_test_lr))
diff = abs(y_pred_lr - y_test_lr)
accuracy_lr = 100 - np.mean(100 * (diff / y_test_lr))
print("Linear Regression Approx Accuracy (%):", round(accuracy_lr, 2))
ols_result = sm.OLS(y_test_lr, X_test_lr).fit()
print("\nOLS Regression Summary:")
print(ols_result.summary())


#            PLOTTING SECTION

plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Open"], label="Opening Price", color="orange")
plt.plot(df.index, df["Close"], label="Closing Price", color="blue")
plt.title(ticker + " Opening and Closing Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Daily Returns
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["Returns"], label="Daily Returns")
plt.axhline(0, color="red", linestyle="--")
plt.title(ticker + " Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot Cumulative Return
df["Cumulative Return"].plot(title="Cumulative Return for " + ticker, figsize=(12,6))
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Rolling Volatility
df["Rolling Volatility"].plot(title="20-Day Rolling Volatility (Risk) for " + ticker, figsize=(12,6))
plt.xlabel("Date")
plt.ylabel("Volatility")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Moving Averages
plt.figure(figsize=(14, 6))
plt.plot(df["Close"], label="Close Price")
plt.plot(df["MA20"], label="20-Day MA")
plt.plot(df["MA50"], label="50-Day MA")
plt.title(ticker + " Stock with Moving Averages")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot RSI
plt.figure(figsize=(12, 5))
df["RSI"].plot()
plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
plt.title("Relative Strength Index (RSI) for " + ticker)
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Autocorrelation and Partial Autocorrelation of Returns
returns = df["Returns"].dropna()
plot_acf(returns)
plt.title("Autocorrelation of Returns")
plt.show()

plot_pacf(returns, method='ywm')
plt.title("Partial Autocorrelation of Returns")
plt.show()
from statsmodels.tsa.arima.model import ARIMA
prices = stock.history(start="2022-01-01")["Close"].dropna()

model = ARIMA(prices, order=(7, 1, 6))
result = model.fit()
print(result.summary())

# Forecast 1 step ahead
next_day_price = result.forecast(steps=1, alpha=0.05)
predicted_price = next_day_price.values[0]
print(f"The predicted price for tomorrow: {predicted_price}")

# Compare with today's price
today_price = prices.iloc[-1]
print(f"Today's price: {today_price}")

# [KEEPING EVERYTHING FROM LINE 1 TO 263 INTACT — NO CHANGES]
# Copy your full 263-line script here (as provided above)

# Now ADD this at the end:
# ------------------------------
# FINAL MODEL ON COMBINED FEATURES
# ------------------------------

# Select combined features
combined_features = df[["Open", "High", "Low", "Close", "Volume", "RSI", "Rolling Volatility", "MA20", "MA50"]].dropna()
combined_target = df.loc[combined_features.index, "Target"]

# Split data into train and test sets
X_train_comb, X_test_comb, y_train_comb, y_test_comb = train_test_split(
    combined_features, combined_target, test_size=0.2, random_state=42
)

# Train Random Forest on combined features
combined_model = RandomForestClassifier(n_estimators=200, random_state=42)
combined_model.fit(X_train_comb, y_train_comb)
y_pred_comb = combined_model.predict(X_test_comb)

# Evaluate model performance
accuracy_comb = accuracy_score(y_test_comb, y_pred_comb)
precision_comb = precision_score(y_test_comb, y_pred_comb)

# Print results
print("\nFinal Model with Combined Features")
print("Accuracy:", round(accuracy_comb * 100, 2), "%")
print("Precision:", round(precision_comb * 100, 2), "%")
