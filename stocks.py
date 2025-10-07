from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, precision_recall_curve
import os
import uvicorn

# ======================================================
# 🔹 Initialize FastAPI
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 🔹 Environment variable for API key
# ======================================================
API_KEY = os.getenv("API_KEY")

# ======================================================
# 🔹 Helper functions from your original code
# ======================================================
def choose_horizons(n_rows: int):
    years = n_rows / 250
    if years < 2:
        return [2, 5, 15, 30]
    elif years < 6:
        return [2, 5, 15, 30, 60, 90]
    else:
        return [2, 5, 15, 30, 60, 90, 180]

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
        start = max(100, int(len(data) * 0.4))
    all_preds = []
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

# ======================================================
# 🔹 Root route
# ======================================================
@app.get("/")
def home():
    return {"message": "Stock Predictor Backend running successfully 🚀"}

# ======================================================
# 🔹 Prediction route
# ======================================================
@app.get("/predict/{symbol}")
def predict_stock(symbol: str):
    try:
        # ========= 1. Get Data (Twelve Data) =========
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 5000,
            "apikey": API_KEY,
        }

        response = requests.get(url, params=params)
        data = response.json()

        if "values" not in data:
            raise HTTPException(status_code=400, detail=f"Twelve Data error: {data}")

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume"
        })
        df = df.astype(float, errors="ignore")
        df = df.sort_values("datetime")
        df = df.set_index("datetime")
        df = df[df.index >= (pd.Timestamp.today() - pd.DateOffset(years=10))]

        if len(df) < 200:
            raise HTTPException(status_code=400, detail=f"Not enough data for {symbol}")

        horizons = choose_horizons(len(df))

        # ========= 2. Preprocess =========
        df["tom"] = df["close"].shift(-1)
        df["target"] = (df["tom"] > df["close"]).astype(int)
        df = df.dropna()

        # ========= 3. Features =========
        predictors = ["open", "high", "low", "close", "volume"]

        for h in horizons:
            rolling_avg = df.rolling(h).mean()
            df[f"Close_Ratio_{h}"] = df["close"] / rolling_avg["close"]
            df[f"Trend_{h}"] = df["target"].shift(1).rolling(h).sum()

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

        new_predictors = (
            predictors
            + ["return_1d", "return_5d", "return_10d", "ma_5", "ma_20", "ma_ratio",
               "volatility_5d", "volatility_20d", "RSI"]
            + [f"Close_Ratio_{h}" for h in horizons]
            + [f"Trend_{h}" for h in horizons]
        )

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

        # ========= 5. Backtest =========
        predictions = backtest(df, new_predictors, model)
        precision = precision_score(predictions["target"], predictions["predictions"])
        accuracy = accuracy_score(predictions["target"], predictions["predictions"])

        # ========= 6. Predict Tomorrow =========
        model.fit(df[new_predictors], df["target"])
        latest = df.iloc[-1:][new_predictors]
        prob_up = model.predict_proba(latest)[:, 1][0]
        prob_down = 1 - prob_up
        threshold = 0.5
        prediction = "UP 📈" if prob_up >= threshold else "DOWN 📉"

        return {
            "symbol": symbol.upper(),
            "precision": round(precision, 4),
            "accuracy": round(accuracy, 4),
            "prob_up": round(prob_up * 100, 2),
            "prob_down": round(prob_down * 100, 2),
            "prediction": prediction,
            "rows_used": len(df),
            "horizons": horizons
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# 🔹 Run server (local or Render)
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
