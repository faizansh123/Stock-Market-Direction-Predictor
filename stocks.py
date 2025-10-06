import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, precision_recall_curve

# ========= 1. Get Data =========
API_KEY = "HGJNX3K27Z7QBA96"
symbol = input("Enter stock ticker: ").upper()

url = "https://www.alphavantage.co/query"
params = {
    "function": "TIME_SERIES_DAILY",  # regular daily prices
    "symbol": symbol,
    "outputsize": "full",
    "apikey": API_KEY
}

print(f"\n🔍 Fetching data for {symbol} ...")
response = requests.get(url, params=params)
data = response.json()

# --- Handle common API responses ---
if "Error Message" in data:
    raise ValueError(f"❌ Invalid symbol '{symbol}'. Check ticker spelling.")

if "Note" in data or "Information" in data:
    print("⚠️ Alpha Vantage API rate limit reached — waiting 60s...")
    time.sleep(60)
    data = requests.get(url, params=params).json()

# --- Find correct key ---
time_series_key = next((k for k in data.keys() if "Time Series" in k), None)
if not time_series_key:
    print("⚠️ Unexpected API response, printing for debugging:")
    print(data)
    raise ValueError("❌ Could not find 'Time Series (Daily)' in API response.")

# --- Convert to DataFrame ---
df = pd.DataFrame(data[time_series_key]).T
df = df.rename(columns={
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume"
})
df = df[["open", "high", "low", "close", "volume"]].astype(float)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# --- Limit to last 10 years ---
df = df[df.index >= (pd.Timestamp.today() - pd.DateOffset(years=10))]

if len(df) < 200:
    raise ValueError(f"⚠️ Not enough data for {symbol} — only {len(df)} trading days found.")

print(f"✅ Loaded {len(df)} rows for {symbol}\n")

# ========= Dynamic Horizon Selection =========
def choose_horizons(n_rows: int):
    """
    Dynamically choose rolling-window horizons depending on how much data we have.
    Approx. 250 trading days = 1 year.
    """
    years = n_rows / 250
    if years < 2:
        return [2, 5, 15, 30]
    elif years < 6:
        return [2, 5, 15, 30, 60, 90]
    else:
        return [2, 5, 15, 30, 60, 90, 180]

horizons = choose_horizons(len(df))
print(f"📈 Using horizons {horizons} based on {len(df)} rows (~{len(df)/250:.1f} years of data).\n")

# ========= 2. Preprocess =========
df["tom"] = df["close"].shift(-1)
df["target"] = (df["tom"] > df["close"]).astype(int)
df = df.dropna()

# ========= 3. Features =========
predictors = ["open", "high", "low", "close", "volume"]

# --- Rolling ratio & trend features ---
for h in horizons:
    rolling_avg = df.rolling(h).mean()
    df[f"Close_Ratio_{h}"] = df["close"] / rolling_avg["close"]
    df[f"Trend_{h}"] = df["target"].shift(1).rolling(h).sum()

# --- Momentum & Volatility features ---
df["return_1d"] = df["close"].pct_change(1)
df["return_5d"] = df["close"].pct_change(5)
df["return_10d"] = df["close"].pct_change(10)
df["ma_5"] = df["close"].rolling(5).mean()
df["ma_20"] = df["close"].rolling(20).mean()
df["ma_ratio"] = df["ma_5"] / df["ma_20"]
df["volatility_5d"] = df["return_1d"].rolling(5).std()
df["volatility_20d"] = df["return_1d"].rolling(20).std()

# --- RSI ---
window = 14
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.rolling(window=window, min_periods=window).mean()
avg_loss = loss.rolling(window=window, min_periods=window).mean()
RS = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + RS))

df = df.dropna()

new_predictors = predictors + [
    "return_1d", "return_5d", "return_10d",
    "ma_5", "ma_20", "ma_ratio",
    "volatility_5d", "volatility_20d",
    "RSI"
] + [f"Close_Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons]

# ========= 4. Model =========
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ========= 5. Helper Functions =========
def find_best_threshold(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    best = np.nanargmax(f1)
    return thresholds[best] if best < len(thresholds) else 0.5

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["target"])
    train_probs = model.predict_proba(train[predictors])[:, 1]
    best_thresh = find_best_threshold(train["target"], train_probs)
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= best_thresh).astype(int)
    preds = pd.Series(preds, index=test.index, name="predictions")
    return pd.concat([test["target"], preds], axis=1)

def backtest(data, predictors, model, start=None, step=25):
    if start is None:
        start = max(100, int(len(data) * 0.4))  # flexible for newer tickers
    all_preds = []
    print(f"🧠 Running backtest ({len(data)} rows total)...")
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:(i + step)].copy()
        if len(test) == 0:
            continue
        combined = predict(train, test, predictors, model)
        all_preds.append(combined)
    if not all_preds:
        raise ValueError("⚠️ No predictions generated — not enough data.")
    return pd.concat(all_preds)

# ========= 6. Backtest =========
predictions = backtest(df, new_predictors, model)
precision = precision_score(predictions["target"], predictions["predictions"])
accuracy = accuracy_score(predictions["target"], predictions["predictions"])

print("\n=== Model Performance ===")
print(f"Precision: {precision:.4f}")
print(f"Accuracy:  {accuracy:.4f}")

# ========= 7. Predict Tomorrow =========
model.fit(df[new_predictors], df["target"])
latest = df.iloc[-1:][new_predictors]
prob_up = model.predict_proba(latest)[:, 1][0]
prob_down = 1 - prob_up
threshold = 0.5
prediction = "UP 📈" if prob_up >= threshold else "DOWN 📉"

print("\n=== Tomorrow's Prediction ===")
print(f"Chance of going UP:   {prob_up*100:.2f}%")
print(f"Chance of going DOWN: {prob_down*100:.2f}%")
print(f"Prediction: Market likely to go {prediction}")
