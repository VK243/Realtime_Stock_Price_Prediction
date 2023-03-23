# Import libraries
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Flatten
import datetime


# Setting the page layout
st. set_page_config(layout='wide')

# Instialize the objects
Scale = StandardScaler()

stock_info = pd.read_csv('Dataset\Companies_Data.csv')

# Loading Dataset
@st.cache_data
def load_data(option):
    data = yf.Ticker(option).history(period='5y').reset_index()
    return data


## Data Preparation  
def dataprep (df, lookback, future, Scale):
    date_train = pd.to_datetime(df['Date'])
    df_train = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    df_train = df_train.astype(float)

    df_train_scaled = Scale.fit_transform(df_train)

    X, y = [],[]
    for i in range(lookback , len(df_train_scaled)-future+1):
        X.append(df_train_scaled[i-lookback:i, 0:df_train.shape[1]])
        y.append(df_train_scaled[i+future-1:i+future, 0])
    
    return np.array(X), np.array(y), df_train, date_train
    
   
# LSTM Models
def Lstm(X,y):
    model = Sequential()

    model.add(LSTM(64, activation='relu', input_shape=(X.shape[1],X.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y.shape[1], activation='relu'))

    opt = tf.keras.optimizer.Adam(lr = 0.001, decay=1e-6)
    model.compile(loss = 'mse', optimizer = opt)

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model.fit(X , y, epochs = 100, verbos = 1, callbacks=[es], validation_split= 0.1, batch_size=16)
    return model

def Lstm_mod1(X,y):
    model = Sequential()

    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(15))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(16, activation ='relu'))
    model.add(Dense(1))

    adam = optimizers.Adam(0.001)

    model.compile(optimizer = adam, loss = 'mean_squared_error')

    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)

    model.fit(X,y,epochs = 100, validation_split = 0.2, batch_size = 64, verbose = 1, callbacks = [es])
    return model


def Lstm_mod2(X,y):
    regressor = Sequential()

    regressor.add(LSTM(20, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))

    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
    es = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=15, restore_best_weights=True)
    regressor.fit(X,y,epochs = 100, validation_split = 0.1, batch_size = 64, verbose = 1, callbacks = [es])
    return regressor

## Prediction
def prediction(model, date_train, Lstm_x, df_train, future, Scale):
    forcasting_dates = pd.date_range(list(date_train)[-1], periods=future, freq='1d').tolist()
    predicted = model.predict(Lstm_x[-future:])
    predicted1 = np.repeat(predicted, df_train.shape[1], axis = -1)
    predicted_descaled = Scale.inverse_transform(predicted1)[:,0]
    return predicted_descaled, forcasting_dates

## Output Preparation
def outputprep(forcasting_dates, predicted_descaled):
    dates = []
    for i in forcasting_dates:
        dates.append(i.date())
    df_final = pd.DataFrame(columns=['Date', 'Open'])
    df_final['Date'] = pd.to_datetime(dates)
    df_final['Open'] = predicted_descaled
    return df_final


def result1 (df, lookback, future, Scale):
    Lstm_x, Lstm_y, df_train, date_train = dataprep(df, lookback, future, Scale)
    model = Lstm_mod1(Lstm_x,Lstm_y)
    loss = pd.DataFrame(model.history.history)
    loss.plot()
    future = 30
    predicted_descaled, forcasting_dates = prediction(model, date_train, Lstm_x, df_train, future, Scale)
    results = outputprep(forcasting_dates, predicted_descaled)
    return results
    
def result2 (df, lookback, future, Scale):
    Lstm_x, Lstm_y, df_train, date_train = dataprep(df, lookback, future, Scale)
    model = Lstm_mod2(Lstm_x,Lstm_y)
    loss = pd.DataFrame(model.history.history)
    loss.plot()
    future = 30
    predicted_descaled, forcasting_dates = prediction(model, date_train, Lstm_x, df_train, future, Scale)
    results = outputprep(forcasting_dates, predicted_descaled)
    return results

# Loading AI model output
@st.cache_resource
def AI_Prediction_Model():
    r1 = result1(df, 30, 1, Scale)
    r2 = result2(df, 30, 1, Scale)
    return r1, r2

# Title of the page

st.title('Realtime - Stock Price Prediction')
st.subheader('Project by - Varshith Kumar')
st.subheader('Predict the next 30 days price of the stock by clicking on the Predict Button below')

left_col , right_col = st.columns(2)

# Selection of Stock

company_selected = left_col.selectbox('Choose a Stock: ', options=stock_info['Company Name'])

# Storing the dataset as Data Frame
company_info = stock_info.loc[stock_info["Company Name"] == company_selected]
df = load_data(company_info["Symbol"].values[0])
data_chart = load_data(company_info["Symbol"].values[0])

a,b,c = st.columns(3)
a.subheader(company_info['Company Name'].values[0])
b.write('Symbol: '+company_info['Symbol'].values[0])
c.write('Industry: '+company_info['Industry'].values[0])


# Adding filter for the stock visualization
y_range = right_col.slider('Range',1,15)
data_chart['Date'] = pd.to_datetime(df['Date'])
cur_year = datetime.datetime.now().year
year = cur_year - y_range
data_chart = data_chart[data_chart['Date'].dt.year >= year]

# Visualization of the Stock by Line chart
st.line_chart(data_chart, x = 'Date', y = 'Open',width=0, use_container_width=True)


# Checkbox to show the Raw Dataset
if st.checkbox('Show Raw Data'):
    st.dataframe(df)


# Spliting of Training and testing Data   
Lstm_x, Lstm_y, df_train, date_train = dataprep(df, 30, 1, Scale)


# Column division for the display of Prediction
left, right = st.columns(2)

# Prediction button
if st.button('Predict', type='primary'): 
    r1,r2 = AI_Prediction_Model()
    right.line_chart(r1, x = 'Date', y = 'Open')
    left.line_chart(r2, x = 'Date', y = 'Open')