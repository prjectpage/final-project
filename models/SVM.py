# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 16:36:33 2024

@author: hyt
"""

from pandas_datareader import data
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error

def svm_train():
    yf.pdr_override()
    lookback = 100
    
    # 指定 CSV 文件的路径
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
    for i in range(len(mtr) - lookback, len(mtr)):
        x_test.append(mtr[i-lookback:i])
        y_test.append(mtr[i])

    # 转换为 numpy 数组并重塑为二维数组
    x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    # 训练 SVM 模型
    model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model.fit(x_train, y_train.ravel())
    
    # 保存模型
    torch.save(model, 'pthfile/svm_model.pth')
    
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

# 调用 SVM 训练函数
svm_train()
