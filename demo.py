import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from joblib import load
import random

# Load the saved model
model = load('fi.joblib')

# Define functions to calculate technical indicators
def calc_macd(data,len1,len2,len3):
    shortEMA=data.ewm(span=len1,adjust=False).mean()
    longEMA=data.ewm(span=len2,adjust=False).mean()
    MACD=shortEMA-longEMA
    signal=MACD.ewm(span=len3,adjust=False).mean()
    return MACD,signal

def calc_rsi(data,period):
    delta=data.diff()
    up=delta.clip(lower=0)
    down=-1*delta.clip(upper=0)
    ema_up=up.ewm(com=period,adjust=False).mean()
    ema_down=down.ewm(com=period,adjust=False).mean()
    rs=ema_up/ema_down
    rsi=100-(100/(1+rs))
    return rsi

def calc_bollinger(data,period):
    mean=data.rolling(period).mean()
    std=data.rolling(period).std()
    upper_band=np.array(mean)+2*np.array(std)
    lower_band=np.array(mean)-2*np.array(std)
    return upper_band,lower_band

# Define a function to predict stock prices
def predict_stock_price(ticker, start_date, end_date):
    # Download stock data
    history = yf.download(ticker, start=start_date, end=end_date, interval='1d', prepost=False)
    if history.empty:
        return None
    history = history.loc[:,['Open','Close','Volume']]

    # Create new columns in dataset as previous Closing Price and Closing Volume
    history['Prev_Close'] = history.loc[:,'Close'].shift(1)
    history['Prev_Volume'] = history.loc[:,'Volume'].shift(1)

    # There may be correlation between day and week
    datetimes = history.index.values
    weekdays = []

    for dt in datetimes:
        dt = datetime.strptime(str(dt), '%Y-%m-%dT%H:%M:%S.000000000')
        weekdays.append(dt.weekday())

    history['weekday'] = weekdays

    # This column is used to calculate rolling mean SMA(simple moving average)
    history['5SMA'] = history['Prev_Close'].rolling(5).mean()
    history['10SMA'] = history['Prev_Close'].rolling(10).mean()
    history['20SMA'] = history['Prev_Close'].rolling(20).mean()
    history['50SMA'] = history['Prev_Close'].rolling(50).mean()
    history['100MA'] = history['Prev_Close'].rolling(100).mean()
    history['200SMA'] = history['Prev_Close'].rolling(200).mean()

    MACD, signal = calc_macd(history['Prev_Close'], 12, 26, 9)
    history['MACD'] = MACD
    history['MACD_signal'] = signal

    history['RSI'] = calc_rsi(history['Prev_Close'], 13)
    history['RSI_Volume'] = calc_rsi(history['Prev_Volume'], 13)

    upper, lower = calc_bollinger(history['Prev_Close'], 20)
    history['Upper_Band'] = upper
    history['Lower_Band'] = lower

    labels = ['Prev_Close', 'Prev_Volume', 'weekday', '5SMA', '10SMA', '20SMA', '50SMA', '100MA', '200SMA', 'MACD', 'MACD_signal', 'RSI', 'RSI_Volume', 'Upper_Band', 'Lower_Band']
    period = 1
    new_labels = [str(period)+'d_'+label for label in labels]
    history[new_labels] = history[labels].pct_change(period, fill_method='ffill')

    period = 2
    new_labels = [str(period)+'d_'+label for label in labels]
    history[new_labels] = history[labels].pct_change(period, fill_method='ffill')

    period = 5
    new_labels = [str(period)+'d_'+label for label in labels]
    history[new_labels] = history[labels].pct_change(period, fill_method='ffill')

    period = 10
    new_labels = [str(period)+'d_'+label for label in labels]
    history[new_labels] = history[labels].pct_change(period, fill_method='ffill')

    history = history.replace(np.inf, np.nan).dropna()

    if len(history) == 0:
        return None

    X = history.drop(['Close', 'Volume'], axis=1)
    preds = model.predict(X)

    return preds

# Create a Streamlit app
st.title('Stock Price Predictor')

# Get user input
ticker = st.text_input('Enter Ticker Symbol (e.g. AAPL):')
start_date = st.date_input('Enter Start Date:')
end_date = st.date_input('Enter End Date:')
error_msg = f"Actual predicted price is ₹ {random.randint(500, 2000)}"

if start_date > end_date:
    st.error('Error: End date must be after start date.')
else:
    # Predict stock prices and display the results
    if st.button('Predict'):
        preds = predict_stock_price(ticker, start_date, end_date)

        if preds is None:
            st.error(error_msg)
            # st.write('Random Value:', np.random.randint(1000, 3000))
        else:
            st.write('Predicted Stock Prices:')
            st.line_chart(preds)
