#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import yfinance as yfin
from scipy.stats import norm

# Load and preprocess data
def load_data(symbol, start_date, end_date):
    st.write(f"Fetching stock data for **{symbol}** from {start_date} to {end_date}...")
    data = yfin.download(symbol, start=start_date, end=end_date)
    data = data[['Close']]
    data.index.name = 'Date'
    return data

def preprocess_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=True),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_future_prices(model, last_data, scaler, window_size=60, steps=30):
    predictions = []
    last_window = last_data[-window_size:]
    scaled_last_window = scaler.transform(last_window)

    for _ in range(steps):
        scaled_last_window = np.reshape(scaled_last_window, (1, window_size, 1))
        pred = model.predict(scaled_last_window, verbose=0)[0][0]
        predictions.append(pred)
        scaled_last_window = np.append(scaled_last_window[0][1:], pred).reshape(window_size, 1)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions

def calculate_prediction_probability(predictions, actual, std_dev):
    probabilities = []
    for pred in predictions:
        prob = norm.cdf((pred - actual) / std_dev) * 100
        probabilities.append(prob)
    return probabilities

# Streamlit Dashboard
def main():
    st.title("📈 Stock Price Prediction with LSTM")
    st.markdown("This dashboard uses an LSTM model to predict future **stock closing prices** based on historical data.")

    symbol = st.text_input("Enter a stock symbol (e.g., AAPL, MSFT, TSLA):", value="AAPL")
    start_date = st.date_input("Select Start Date", value=pd.to_datetime("2018-01-01"))
    end_date = st.date_input("Select End Date", value=pd.to_datetime("2022-12-31"))

    if st.button("Fetch and Predict"):
        data = load_data(symbol, start_date, end_date)
        st.write("Data Preview:", data.tail())

        # Split data
        window_size = 60
        train_data = data.iloc[:-100]
        test_data = data.iloc[-100:]

        # Preprocess
        X_train, y_train, scaler = preprocess_data(train_data[['Close']], window_size)
        X_test, y_test, _ = preprocess_data(test_data[['Close']], window_size)

        # Train model
        model = build_model((X_train.shape[1], 1))
        with st.spinner("Training the LSTM model..."):
            model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        st.success("Model training complete.")

        # Predict on test set
        predictions = model.predict(X_test, verbose=0)
        predictions = scaler.inverse_transform(predictions)
        actual = scaler.inverse_transform(y_test.reshape(-1, 1))

        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))

        st.subheader("📊 Model Performance")
        st.write(f"**MAE:** {mae:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")

        # Predict next 30 days
        future_prices = predict_future_prices(model, test_data[['Close']], scaler, window_size, steps=30)

        std_dev = np.std(actual - predictions)
        probabilities = calculate_prediction_probability(future_prices, actual[-1][0], std_dev)

        # Results
        st.subheader("🔮 Future Predictions")
        st.write(f"**Next Day Prediction:** {future_prices[0][0]:.2f} (Probability: {probabilities[0]:.2f}%)")
        st.write(f"**Next Week:** {future_prices[:7].flatten()} (Avg Prob: {np.mean(probabilities[:7]):.2f}%)")
        st.write(f"**Next Month:** {future_prices.flatten()} (Avg Prob: {np.mean(probabilities):.2f}%)")

        # Plot
        st.subheader("📉 30-Day Forecast")
        future_dates = pd.date_range(test_data.index[-1] + pd.Timedelta(days=1), periods=30)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(future_dates, future_prices, label="Predicted Prices", color="blue")
        ax.set_title(f"{symbol} Future Price Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()

