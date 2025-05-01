import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ----------------- Load Data --------------------
st.write("App loaded!")

@st.cache_data
def load_data():
    # Replace with your real CSV files or data sources
    amzn = pd.read_csv('AMZN.csv')
    aapl = pd.read_csv('AAPL.csv')
    pypl = pd.read_csv('PYPL.csv')
    nke = pd.read_csv('NKE.csv')

    for df in [amzn, aapl]:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    return {'AMZN': amzn, 'AAPL': aapl}

data = load_data()

st.write("Data loaded successfully!")

# ----------------- Streamlit Sidebar --------------------

st.title("Stock Price Forecasting with ARIMA 📈")

selected_stock = st.sidebar.selectbox('Select Stock', list(data.keys()))
forecast_horizon = st.sidebar.selectbox('Forecast Horizon (days)', [1, 3, 7])

df = data[selected_stock]

st.write(f"### {selected_stock} Close Price History")
st.line_chart(df['Close'])

# ----------------- ARIMA Forecasting --------------------

st.subheader(f"ARIMA Forecast for {selected_stock} ({forecast_horizon} day(s) ahead)")

# Fit ARIMA model
model = ARIMA(df['Close'], order=(5,1,0))
model_fit = model.fit()

forecast_result = model_fit.get_forecast(steps=forecast_horizon)
forecast = forecast_result.predicted_mean
conf_int = forecast_result.conf_int()

# Create forecast DataFrame
forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_horizon)

forecast_df = pd.DataFrame({
    'Forecast': forecast.values,
    'Lower CI': conf_int.iloc[:, 0].values,
    'Upper CI': conf_int.iloc[:, 1].values
}, index=forecast_dates)

st.write("### Forecasted Close Prices")
st.dataframe(forecast_df)

# ----------------- Plot --------------------

plt.figure(figsize=(10,5))
plt.plot(df['Close'], label='Historical Close')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast', color='green')
plt.fill_between(forecast_df.index, forecast_df['Lower CI'], forecast_df['Upper CI'], color='gray', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(f'{selected_stock} Forecast')
plt.legend()
st.pyplot(plt)
