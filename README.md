# 📈 AI Stock Prediction Agent
XGBoost + LSTM Ensemble for Next-Day Market Direction

An interactive AI-powered stock movement prediction app built with Streamlit, combining XGBoost (tabular ML) and LSTM (deep learning for sequences) to forecast whether a stock is likely to go UP 📈 or DOWN 📉 tomorrow.

## 🚀 Live App Features

✔ Download 3 years of historical stock data from Yahoo Finance
✔ Feature engineering using rolling return windows
✔ Dual-model prediction system:

![Screenshot](https://github.com/gowthaman25/Stock-Price-Prediction-Application/blob/main/Sc1.png)

## 🌲 XGBoost for tabular learning

🧠 LSTM Neural Network for time-series pattern learning
✔ Ensemble probability-based trading signal
✔ Visual stock performance chart
✔ Model validation using ROC-AUC

## 🧠 How the Model Works

1️⃣ Data Processing

Historical stock prices are downloaded using yfinance

Daily returns are calculated

Future return is used to create a classification target

A threshold removes noisy small movements

2️⃣ Feature Engineering
For each day, the model uses the previous N days' returns (Window Size) as features.

Lag Feature	Meaning
ret_lag_1	Yesterday’s return
ret_lag_2	Return 2 days ago
...	...
ret_lag_N	Return N days ago

3️⃣ Models Used

🌲 XGBoost Classifier
Learns nonlinear relationships in tabular lag-return features.

## 🧠 LSTM Neural Network
Learns temporal patterns from sequences of stock returns.

4️⃣ Ensemble Prediction

Final Probability = 0.6 × XGBoost + 0.4 × LSTM

Probability	Signal
> 0.60	📈 BUY
< 0.40	📉 SELL
0.40 – 0.60	⏸ HOLD

## 📊 Model Validation

Performance is evaluated using ROC-AUC on a TimeSeriesSplit validation strategy.

This avoids data leakage and simulates real trading conditions.

## 🖥 App Interface

🔍 User Inputs

Stock symbol (Yahoo Finance format, e.g. AAPL, TSLA, RELIANCE.NS)

Window size for historical lookback

📈 Outputs

## Historical stock price chart

Tomorrow UP probability

Trading signal (BUY / SELL / HOLD)

## Model ROC-AUC scores
![Screenshot](https://github.com/gowthaman25/Stock-Price-Prediction-Application/blob/main/Sc2.png)
## ⚙️ Installation

## Clone the repository:
```bash
git clone https://github.com/gowthaman25/stock-prediction-agent.git
cd stock-prediction-agent
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the Streamlit app:
```bash
streamlit run app.py
```
## 📬 Contact

For questions or collaboration requests:

* 📧 Email: [ecsgowtham@gmail.com ](mailto:ecsgowtham@gmail.com )
* 🌐 GitHub: [gowthaman25](https://github.com/gowthaman25/)
  
---

> ✨ Built using [Streamlit](https://streamlit.io/), and LLaMA 3.3 models
