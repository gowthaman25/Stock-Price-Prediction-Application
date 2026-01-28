import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam

# ---------------- UI ----------------
st.set_page_config(page_title="Stock Prediction Agent", layout="centered")
st.title("📈 AI Stock Prediction Agent")
st.write("XGBoost + LSTM Ensemble for Tomorrow Direction")

symbol = st.text_input(
    "Enter Stock Symbol (Yahoo Finance format)",
    placeholder="e.g. AAPL, RELIANCE.NS"
)

WINDOW = st.slider("Window Size", 5, 20, 15)

# ---------------- DATA ----------------
@st.cache_data
def get_stock_data(symbol):
    return yf.download(symbol, period="3y", progress=False)

def prepare_data(df):
    df["Returns"] = df["Close"].pct_change()
    df["Future_Return"] = df["Returns"].shift(-1)

    threshold = 0.003
    df["Target"] = np.where(
        df["Future_Return"] > threshold, 1,
        np.where(df["Future_Return"] < -threshold, 0, np.nan)
    )

    df.dropna(inplace=True)
    return df

def make_supervised(df, window):
    data = df.copy()

    for i in range(1, window + 1):
        data[f"ret_lag_{i}"] = data["Returns"].shift(i)

    data.dropna(inplace=True)

    X_tab = data[[f"ret_lag_{i}" for i in range(1, window + 1)]].values
    X_seq = X_tab.reshape(len(X_tab), window, 1)
    y = data["Target"].values

    return X_tab, X_seq, y, data

# ---------------- PREDICTION ----------------
if st.button("Predict Tomorrow"):

    if not symbol.strip():
        st.warning("⚠️ Please enter a stock symbol")
        st.stop()
     
    df = get_stock_data(symbol)
    if df.empty:
            st.error("No data found.")
            st.stop()

    df = prepare_data(df)
    
    st.subheader("📈 Historical Stock Price Performance")

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df.index,
        df["Close"],
        label="Close Price",
        linewidth=2
    )

    ax.set_title(f"{symbol} – Historical Price Performance")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    
     
    with st.spinner(f"Processing {symbol}... ⏳"):

        
        X_tab, X_seq, y, data = make_supervised(df, WINDOW)

        # ---------- Time Series Split ----------
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, test_idx = list(tscv.split(X_tab))[-1]

        X_train_tab, X_test_tab = X_tab[train_idx], X_tab[test_idx]
        X_train_seq, X_test_seq = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---------- XGBoost ----------
        xgb = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            random_state=42
        )

        xgb.fit(X_train_tab, y_train)
        xgb_test_probs = xgb.predict_proba(X_test_tab)[:, 1]

        # ---------- LSTM ----------
        lstm = Sequential([
            LSTM(32, input_shape=(WINDOW, 1)),
            Dropout(0.3),
            Dense(1, activation="sigmoid")
        ])

        lstm.compile(
            optimizer=Adam(0.001),
            loss="binary_crossentropy"
        )

        lstm.fit(
            X_train_seq, y_train,
            epochs=20,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )

        lstm_test_probs = lstm.predict(X_test_seq).flatten()

        # ---------- ROC ----------
        roc_xgb = roc_auc_score(y_test, xgb_test_probs)
        roc_lstm = roc_auc_score(y_test, lstm_test_probs)

        # ---------- Tomorrow Prediction ----------
        latest_returns = df["Returns"].iloc[-WINDOW:].values[::-1]
        latest_tab = latest_returns.reshape(1, -1)
        latest_seq = latest_returns.reshape(1, WINDOW, 1)

        prob_xgb = xgb.predict_proba(latest_tab)[0][1]
        prob_lstm = lstm.predict(latest_seq)[0][0]
       
        alpha = 0.6
        ensemble_prob = alpha * prob_xgb + (1 - alpha) * prob_lstm

        # ---------- Signal ----------
        if ensemble_prob > 0.6:
            signal = "BUY"
        elif ensemble_prob < 0.4:
            signal = "SELL"
        else:
            signal = "HOLD"



    # ---------------- RESULTS ----------------
    st.subheader("📊 Prediction Result")

    st.metric("Tomorrow UP Probability", f"{ensemble_prob:.2%}")

    if signal == "BUY":
        st.success("📈 Trading Signal: BUY")
    elif signal == "SELL":
        st.error("📉 Trading Signal: SELL")
    else:
        st.warning("⏸ Trading Signal: HOLD")

    st.subheader("📐 Model Validation ROC-AUC")
    st.write(f"XGBoost ROC-AUC : **{roc_xgb:.3f}**")
    st.write(f"LSTM ROC-AUC    : **{roc_lstm:.3f}**")
