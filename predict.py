import torch
from pandas_datareader import data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error
from models.LSTM import LSTM
from utils.split_data import split_data
from sklearn.model_selection import train_test_split
import joblib
import pylab as plt

def predict(which_model):
    yf.pdr_override()
    scaler = MinMaxScaler(feature_range=(-1, 1))  
    stock_data = data.get_data_yahoo('0066.HK',start="2010-01-01", end="2020-06-30").reset_index()
    if which_model == 'lstm':
        mtr = stock_data
        lookback = 100
        mtr = scaler.fit_transform(mtr['Close'].values.reshape(-1,1))
        
        x_train, y_train, x_test, y_test = split_data(mtr, lookback)

        x_test = torch.from_numpy(x_test).type(torch.Tensor)
        y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
        x_train = torch.from_numpy(x_train).type(torch.Tensor)
        y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

        model = torch.load('pthfile/lstm_model_complete.pth')


        mtr = scaler.inverse_transform(mtr)
        
        test_pred = model(x_test)
        test_pred = pd.DataFrame((test_pred.detach().numpy()))
        train_pred = model(x_train)
        train_pred = pd.DataFrame((train_pred.detach().numpy()))

        trainPredictPlot=np.empty_like(mtr)
        trainPredictPlot[:,:]=np.nan
        trainPredictPlot[lookback:len(scaler.inverse_transform(train_pred))+lookback] = scaler.inverse_transform(train_pred)

        testPredictPlot=np.empty_like(mtr)
        testPredictPlot[:,:] = np.nan
        testPredictPlot[len(train_pred)+(lookback):len(mtr)] = scaler.inverse_transform(test_pred)

        trainPredictPlot = pd.DataFrame({'Date':stock_data['Date'],'Close':trainPredictPlot.flatten()})
        testPredictPlot = pd.DataFrame({'Date':stock_data['Date'],'Close':testPredictPlot.flatten()})
        mtr = pd.DataFrame({'Date':stock_data['Date'],'Close':mtr.flatten()})
        # plt.figure(figsize=(12,8))
        # plt.plot(mtr)

        # plt.plot(testPredictPlot)
        # plt.plot(trainPredictPlot)

        # plt.show()

        # criterion = torch.nn.MSELoss(reduction='mean')


        mse_test = mean_squared_error(y_test_lstm.detach().numpy(),test_pred)
        mse_train = mean_squared_error(y_train_lstm.detach().numpy(),train_pred)



        return mse_train,mse_test,testPredictPlot,trainPredictPlot,mtr

    elif which_model == 'linearregression':

        predict_data = data.get_data_yahoo('0066.HK', start="2021-01-01", end="2021-04-30").reset_index()
        predict_data.set_index('Date', inplace=True)
        predict_x = predict_data[['Open', 'High', 'Low', 'Volume']]
        predict_y = predict_data['Close']

        # print(predict_x)
        predict_y = predict_y.sort_index()
        predict_x = predict_x.sort_index()
        model = joblib.load('pthfile/linear_regression_model.joblib')
        predict_result = model.predict(predict_x)
        mse = mean_squared_error(predict_y, predict_result)

        pred = pd.DataFrame({'Date':pd.to_datetime(predict_x.index),'Close':predict_result})
        actual = pd.DataFrame({'Date':pd.to_datetime(predict_x.index),'Close':predict_y.values})

        


        return mse,actual,pred



# predict('linearregression')