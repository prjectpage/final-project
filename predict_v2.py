# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:37:36 2024

@author: hyt
"""

import torch
from pandas_datareader import data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from models.LSTM import LSTM
from utils.split_data import split_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from torch.nn import GRU
import matplotlib.pyplot as plt
import io
from itertools import product
from statsmodels.tsa.arima.model import ARIMA



def predict(which_model):
    
    yf.pdr_override()
    scaler = MinMaxScaler(feature_range=(-1, 1))  
    stock_data = data.get_data_yahoo('0066.HK', start="2010-01-01", end="2020-06-30").reset_index()

    # stock_data = pd.read_csv('FSV.csv' )
    if which_model == 'lstm':
        mtr = stock_data
        lookback = 100
        mtr = scaler.fit_transform(mtr['Close'].values.reshape(-1,1))
        Open = stock_data['Open'].values.reshape(-1,1)
        
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

        ####整体predict序列
        whole_predict = np.empty_like(mtr)
        whole_predict[:,:] = np.nan
        whole_predict[lookback:len(scaler.inverse_transform(train_pred))+lookback] = scaler.inverse_transform(train_pred)
        whole_predict[len(train_pred)+(lookback):len(mtr)] = scaler.inverse_transform(test_pred)
        # actual = mtr[lookback:]

        whole_predict = pd.DataFrame({'Date':stock_data['Date'],'pred':whole_predict.flatten(),'actual':mtr.flatten(),'open':Open.flatten()})

        ###trading
        total_yield = 0
        total_basis = 0
        yesterday = whole_predict['actual'].iloc[0]
        start = yesterday
        for i,row in whole_predict.iterrows():
            today = row['pred']
            if today > yesterday:
                today_yield = row['actual']-row['open']
                total_yield = total_yield + today_yield
                # print('DATE:'+ str(row['Date'])+' buy at '+ str(row['open']) + '  sell at ' + str(row['actual']) + '  total yield:' + str(total_yield))
            yesterday = row['actual']

            today_basis = row['actual']-row['open']
            total_basis = total_basis + today_basis
            # print('DATE:'+ str(row['Date'])+' buy at '+ str(row['open']) + '  sell at ' + str(row['actual']) + '  todayield:' + str(today_basis) + 'total yield:' + str(total_yield))
            with open("log.txt", "a") as file:
                file.write('DATE:'+ str(row['Date'])+' buy at '+ str(row['open']) + '  sell at ' + str(row['actual']) + '  todayield:' + str(today_basis) + 'total yield:' + str(total_basis) + "\n")  # 写入日志消息并添加换行符
        
        print('basis yield:' + str(total_basis))
        
        



        trainPredictPlot = pd.DataFrame({'Date':stock_data['Date'],'Close':trainPredictPlot.flatten()})
        testPredictPlot = pd.DataFrame({'Date':stock_data['Date'],'Close':testPredictPlot.flatten()})
        mtr = pd.DataFrame({'Date':stock_data['Date'],'Close':mtr.flatten()})

        
   



        mse_test = mean_squared_error(y_test_lstm.detach().numpy(),test_pred)
        mse_train = mean_squared_error(y_train_lstm.detach().numpy(),train_pred)



        return mse_train,mse_test,testPredictPlot,trainPredictPlot,mtr,whole_predict

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

        
    elif which_model == 'arima':
        stock_data['Date'] = pd.to_datetime(stock_data['Date'])
        stock_data.set_index('Date', inplace=True)
        close_prices = stock_data['Close']

        # 3. 模型选择
        # AIC (Akaike Information Criterion)最小化来选择ARIMA的参数
        # 这里使用简单的网格搜索，实际应用中可能需要更复杂的方法
        p = d = q = range(0, 2)  # 取值可以根据数据不同进行调整
        pdq_combinations = list(product(p, d, q))
        best_aic = float("inf")
        best_pdq = None

        # 遍历不同的ARIMA参数组合
        for combination in pdq_combinations:
            try:
                model = ARIMA(close_prices, order=combination)
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_pdq = combination
            except:
                continue

        print(f"Best ARIMA{best_pdq} model AIC: {best_aic}")

        # 4. 滚动回测
        # 设置初始训练数据集的长度
        train_size = int(len(close_prices) * 0.8)
        train, test = close_prices[0:train_size], close_prices[train_size:]

        history = [x for x in train]
        predictions = list()

        # 滚动预测每个测试集数据点
        for t in range(len(test)):
            model = ARIMA(history, order=best_pdq)
            model_fit = model.fit()

            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)


        

        # 计算并打印性能指标
        error = mean_squared_error(test, predictions)
        # print(f'Test MSE: {error}')

        # # 绘制测试集观测值与预测值
        # plt.figure(figsize=(12, 6))
        # plt.plot(test.index, test, label='Observed')
        # plt.plot(test.index, predictions, label='Predicted')
        # plt.legend()
        # plt.show()


        return error,test,predictions

        return mse,actual,pred
        
    if which_model == 'randomforest':
        lookback = 100                
        csv_file_path = 'SSE_Index.csv'  # 替换为你的 CSV 文件路径
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        mtr = scaler.fit_transform(df.values)
        # 构建训练集和测试集
        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(len(mtr) - lookback):
            x_train.append(mtr[i:i+lookback])
            y_train.append(mtr[i+lookback])
            for i in range(len(mtr) - lookback, len(mtr)):
                x_test.append(mtr[i-lookback:i])
                y_test.append(mtr[i])
        # 转换为 numpy 数组并重塑为二维数组
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        # Load pre-trained Random Forest model
        model = torch.load('pthfile/randomforest_model.pth')
        
        #model = RandomForestRegressor(n_estimators=100, random_state=42)
        #model.fit(x_train, y_train.ravel())

        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        
        mse_train = mean_squared_error(y_train, train_pred)
        mse_test = mean_squared_error(y_test, test_pred)
        # 打印均方误差
        print('MSE ON TRAIN:', mse_train)
        print('MSE ON TEST:', mse_test)   
        
        # # 绘制训练集的预测结果
        # plt.figure(figsize=(12, 6))
        # plt.plot(y_train, label='Actual Train')
        # plt.plot(train_pred, label='Predicted Train')
        # plt.legend()
        # plt.title('Random Forest Predictions on Train Data')
        # plt.show()

        # # 绘制测试集的预测结果
        # plt.figure(figsize=(12, 6))
        # plt.plot(y_test, label='Actual Test')
        # plt.plot(test_pred, label='Predicted Test')
        # plt.legend()
        # plt.title('Random Forest Predictions on Test Data')
        # plt.show()
        
        # No prediction plot for Random Forest model, so return None for plot variables
        return mse_train, mse_test, y_train, train_pred, y_test, test_pred, mtr


    elif which_model == 'svm':
        lookback = 100   
        csv_file_path = 'SSE_Index.csv'  # 替换为你的 CSV 文件路径
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # 对数据进行标准化
        scaler = MinMaxScaler(feature_range=(-1, 1))
        mtr = scaler.fit_transform(df.values)

        # 构建训练集和测试集
        x_train, y_train, x_test, y_test = [], [], [], []
        for i in range(len(mtr) - lookback):
            x_train.append(mtr[i:i+lookback])
            y_train.append(mtr[i+lookback])
        for j in range(len(mtr) - lookback, len(mtr)):
            x_test.append(mtr[j-lookback:j])
            y_test.append(mtr[j])

        # 转换为 numpy 数组并重塑为二维数组
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)

        # Load pre-trained SVM model
        model = torch.load('pthfile/svm_model.pth')

        # 进行预测
        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)
        
        # 计算均方误差
        mse_train = mean_squared_error(y_train, train_pred)
        mse_test = mean_squared_error(y_test, test_pred)
        
        # 打印均方误差
        print('MSE ON TRAIN:', mse_train)
        print('MSE ON TEST:', mse_test)

        # 绘制预测结果
        plt.plot(y_train, label='Actual Train')
        plt.plot(train_pred, label='Predicted Train')
        plt.legend()
        plt.title('SVM Predictions on Train Data')
        plt.show()
        
        plt.plot(y_test, label='Actual Test')
        plt.plot(test_pred, label='Predicted Test')
        plt.legend()
        plt.title('SVM Predictions on Test Data')
        plt.show()

        return mse_train, mse_test, y_train, train_pred, y_test, test_pred, mtr


    elif which_model == 'gru':
        # 数据准备
        def prepare_data(data, lookback):
            x, y = [], []
            for i in range(len(data) - lookback):
                x.append(data[i:(i + lookback)])
                y.append(data[i + lookback])
            return np.array(x), np.array(y)

        # 加载数据
        csv_file_path = 'SSE_Index.csv'
        df = pd.read_csv(csv_file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # 数据预处理
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = scaler.fit_transform(df.values)

        # 定义超参数
        lookback = 100

        # 加载预训练的GRU模型
        loaded_model = torch.jit.load('pthfile/gru_model_script.pth')

        # 准备测试集
        train_data_scaled = scaler.transform(df.values)
        x_train, y_train = prepare_data(train_data_scaled, lookback)
        x_train = torch.from_numpy(x_train).type(torch.Tensor)  # 将 ndarray 转换为 Tensor
        
        test_data_scaled = scaler.transform(df.values)
        x_test, y_test = prepare_data(test_data_scaled, lookback)
        x_test = torch.from_numpy(x_test).type(torch.Tensor)  # 将 ndarray 转换为 Tensor

        # 使用加载的模型进行预测
        loaded_model.eval()
        train_pred = loaded_model(x_train)
        train_pred = scaler.inverse_transform(train_pred.detach().numpy())
        test_pred = loaded_model(x_test)
        test_pred = scaler.inverse_transform(test_pred.detach().numpy())

        # 可视化预测结果
        # 绘制训练集预测结果
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[lookback:len(train_pred) + lookback], df['Close'][lookback:len(train_pred) + lookback], label='Actual Train')
        plt.plot(df.index[lookback:len(train_pred) + lookback], train_pred, label='Predicted Train')
        plt.legend()
        plt.title('GRU Predictions on Train Data')
        plt.show()

        
        plt.figure(figsize=(12, 6))
        plt.plot(df.index[lookback:], df['Close'][lookback:], label='Actual Test')
        plt.plot(df.index[lookback:], test_pred, label='Predicted Test (GRU Model)')
        plt.legend()
        plt.title('GRU Predictions on Test Data')
        plt.show()

        # 计算均方误差
        mse_train = mean_squared_error(y_train, train_pred)
        mse_test = mean_squared_error(y_test, test_pred)#df['Close'][lookback:]
        print('MSE ON TRAIN:', mse_train)
        print('MSE ON TEST:', mse_test) 

        return mse_train, mse_test, y_train, train_pred, y_test, test_pred, scaled_data

    else:
        raise ValueError("Invalid model specified.")


