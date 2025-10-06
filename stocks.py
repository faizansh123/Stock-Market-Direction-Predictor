# backend/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests, pandas as pd, numpy as np, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score, precision_recall_curve

app = FastAPI()

# allow frontend access (React/Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "HGJNX3K27Z7QBA96"

@app.get("/predict/{symbol}")
def predict_stock(symbol: str):
    try:
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "outputsize": "full",
            "apikey": API_KEY
        }
        response = requests.get(url, params=params)
        data = response.json()
        key = next((k for k in data.keys() if "Time Series" in k), None)
        if not key:
            raise HTTPException(status_code=400, detail="Invalid ticker or API limit hit")

        df = pd.DataFrame(data[key]).T
        df = df.rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume"
        })
        df = df.astype(float)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[df.index >= (pd.Timestamp.today() - pd.DateOffset(years=10))]

        # --- features ---
        df["tom"] = df["close"].shift(-1)
        df["target"] = (df["tom"] > df["close"]).astype(int)
        df = df.dropna()

        predictors = ["open", "high", "low", "close", "volume"]
        df["return_1d"] = df["close"].pct_change(1)
        df = df.dropna()
        predictors.append("return_1d")

        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(df[predictors], df["target"])

        latest = df.iloc[-1:][predictors]
        prob_up = model.predict_proba(latest)[:, 1][0]
        prob_down = 1 - prob_up
        prediction = "UP" if prob_up > 0.5 else "DOWN"

        return {
            "symbol": symbol.upper(),
            "prob_up": round(prob_up * 100, 2),
            "prob_down": round(prob_down * 100, 2),
            "prediction": prediction,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
