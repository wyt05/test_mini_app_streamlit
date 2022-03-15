import streamlit as st
from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from datetime import datetime  
from datetime import timedelta
import plotly.express as px
import requests
from bs4 import BeautifulSoup


def get_dataframe_stock(ticker, start_date, end_date):
    tickers = [ticker]
    start_date = start_date
    end_date = end_date
    panel_data = data.DataReader(tickers,'yahoo', start_date, end_date)
    return panel_data

def get_stock_news(ticker):
    news_dataframe = ""
    news_headlines = []
    news_synopsis = []

    url = 'https://finance.yahoo.com/quote/' + ticker
    test_response = requests.get(url)
    soup = BeautifulSoup(test_response.text)

    news_block = soup.find("ul", {"class": "My(0) P(0) Wow(bw) Ov(h)"})
    print(news_block)

    for item in news_block.findAll('li'):
        if item.find('h3') is not None:
            news_headlines.append(item.find('h3').text)
            news_synopsis.append(item.find('p').text)

    news_dataframe = pd.DataFrame({'News Headline': news_headlines, 'News Description': news_synopsis})

    return news_dataframe

def get_data(data, look_back):
  data_x, data_y = [],[]

  for i in range(len(data)-look_back-1):
    data_x.append(data[i:(i+look_back),0])
    data_y.append(data[i+look_back,0])

  return np.array(data_x) , np.array(data_y)

st.set_page_config(
    page_title="Stock Prediction App", page_icon="📊", initial_sidebar_state="expanded"
)

st.title("💬 Stock Prediction app")

all_tickers = ['SPY', 'EURUSD=X', 'EURGBP=X', 'BTC-USD', 'AAPL', 'ETH-USD']


with st.form(key='insert_stock'):
    symbols = st.selectbox("Choose stocks to visualize", all_tickers)
    date_cols = st.columns((1, 1))
    start_date = date_cols[0].date_input('Start Date')
    end_date = date_cols[1].date_input('End Date')
    submitted = st.form_submit_button('Submit')


if submitted:
    stock_dataframe = get_dataframe_stock(symbols, start_date, end_date)
    close_price = stock_dataframe['Close'].reset_index()
    
    st.dataframe(close_price)
    
    news_dataframe = get_stock_news(symbols)

    st.dataframe(news_dataframe)

    #Plot closing price
    main_close_price = px.line(stock_dataframe['Close'].reset_index(), x="Date", y=symbols, title='Stock Price History')
    st.plotly_chart(main_close_price, use_container_width=True)

    with st.spinner('Waiting...'):
        #Machine learning part
        close_only_mach = np.array(stock_dataframe['Close']).reshape(-1, 1)
        training_data_size = round(len(close_only_mach) * 0.95)
        test_data_size =  len(close_only_mach) - training_data_size
        train_data, test_data = close_only_mach[:training_data_size], close_only_mach[training_data_size:]

        look_back = 1
        x_train , y_train = get_data(train_data, look_back)

        x_test , y_test = get_data(test_data,look_back)

        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], 1)
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)

        n_features=x_train.shape[1]
        model=Sequential()
        model.add(LSTM(100,activation='relu',input_shape=(1,1)))
        model.add(Dense(n_features))

        model.compile(optimizer='adam', loss = 'mse')
        model.fit(x_train,y_train, epochs = 10, batch_size=1)
        y_pred = model.predict(x_test)
        y_test = np.array(y_test).reshape(-1,1)

        x_val = np.linspace(0, len(y_test), len(y_test))
        y_test_list = y_test.flatten().tolist()
        y_pred_list = y_pred.flatten().tolist()

        new_data = {'Actual': y_test_list, 'Predicted': y_pred_list}

        chart_data = pd.DataFrame(new_data)

        lstm_pred = px.line(chart_data, y=["Actual", "Predicted"], title='Predicted Price vs Actual')


    #After end

    st.plotly_chart(lstm_pred, use_container_width=True)

    latest_price = close_only_mach[-1].reshape(1,x_test.shape[1], 1)
    forecast_y_pred = model.predict(latest_price)
    print(forecast_y_pred)
    text_price_pred = 'Your next day price is $' + str(forecast_y_pred[0][0])
    st.success(text_price_pred)