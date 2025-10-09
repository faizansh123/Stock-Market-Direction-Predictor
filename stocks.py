from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from functools import lru_cache
import uvicorn

# ======================================================
# 🔹 Initialize FastAPI
# ======================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================
# 🔹 Environment variable for API key
# ======================================================
API_KEY = os.getenv("API_KEY")

# ======================================================
# 🔹 Cached data fetcher (fast repeated calls)
# ======================================================
@lru_cache(maxsize=64)
def fetch_data(symbol: str):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": 1000,  # ⬅️ reduced for speed (~4 years)
        "apikey": API_KEY,
    }
    response = requests.get(url, params=params)
    data = response.json()
    if "values" not in data:
        raise HTTPException(status_code=400, detail=f"Twelve Data error: {data}")
    return data["values"]

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
        # ====== 1️⃣ Get data (cached)
        data_values = fetch_data(symbol.upper())
        df = pd.DataFrame(data_values)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume"
        })
        df = df.astype(float, errors="ignore")
        df = df.sort_values("datetime").set_index("datetime")

        if len(df) < 100:
            raise HTTPException(status_code=400, detail=f"Not enough data for {symbol}")

        # ====== 2️⃣ Lightweight feature generation
        df["tomorrow"] = df["close"].shift(-1)
        df["target"] = (df["tomorrow"] > df["close"]).astype(int)
        df["return_1d"] = df["close"].pct_change(1)
        df["ma_5"] = df["close"].rolling(5).mean()
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_ratio"] = df["ma_5"] / df["ma_20"]
        df["volatility_5d"] = df["return_1d"].rolling(5).std()
        df.dropna(inplace=True)

        predictors = ["open", "high", "low", "close", "volume",
                      "return_1d", "ma_ratio", "volatility_5d"]

        # ====== 3️⃣ Train model once (no backtesting)
        model = RandomForestClassifier(
            n_estimators=100,      # smaller forest
            max_depth=8,           # controlled depth
            n_jobs=-1,
            random_state=42
        )
        model.fit(df[predictors], df["target"])

        # ====== 4️⃣ Predict tomorrow
        latest = df.iloc[-1:][predictors]
        prob_up = model.predict_proba(latest)[:, 1][0]
        prediction = "UP 📈" if prob_up >= 0.5 else "DOWN 📉"

        return {
            "symbol": symbol.upper(),
            "prob_up": round(prob_up * 100, 2),
            "prob_down": round((1 - prob_up) * 100, 2),
            "prediction": prediction,
            "rows_used": len(df),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ======================================================
# 🔹 Run server
# ======================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("stocks:app", host="0.0.0.0", port=port)
