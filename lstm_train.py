from pandas_datareader import data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import pylab as plt
import torch
import torch.nn as nn
import time
from sklearn.metrics import mean_squared_error
from models.LSTM import LSTM
from utils.split_data import split_data
def lstm_train():
    yf.pdr_override()

    #######参数配置
    input_dim = 1
    hidden_dim = 16
    num_layers = 3
    output_dim = 1
    num_epochs = 50
    lookback = 100


    scaler = MinMaxScaler(feature_range=(-1, 1))  
    mtr = data.get_data_yahoo('0066.HK',start="2010-01-01", end="2020-06-30").reset_index()
    mtr['Date'] = pd.to_datetime(mtr['Date'])
    mtr.set_index('Date', inplace=True)
    mtr = scaler.fit_transform(mtr['Close'].values.reshape(-1,1))

    x_train, y_train, x_test, y_test = split_data(mtr, lookback)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape,mtr.shape

    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_train.shape

    y_train_lstm = torch.from_numpy(y_train).type(torch.Tensor)

    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_test_lstm = torch.from_numpy(y_test).type(torch.Tensor)
    y_train_lstm.detach




    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    criterion = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


    hist = np.zeros(num_epochs)
    start_time = time.time()
    lstm = []
    #####训练
    for t in range(num_epochs):
        y_train_pred = model(x_train)
        
        loss = criterion(y_train_pred, y_train_lstm)
        print("Epoch ", t, "MSE: ", loss.item())
        hist[t] - loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(model, 'pthfile/lstm_model_complete.pth')


# lstm_train()