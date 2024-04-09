# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:54:49 2024

@author: hyt
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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
input_dim = 1
hidden_dim = 32
output_dim = 1
num_layers = 2
lookback = 100
num_epochs = 100
learning_rate = 0.001

# 准备训练集和测试集
train_size = int(len(scaled_data) * 0.80)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size], scaled_data[train_size:]

train_x, train_y = prepare_data(train_data, lookback)
test_x, test_y = prepare_data(test_data, lookback)

# 转换为张量
train_x = torch.from_numpy(train_x).type(torch.Tensor)
train_y = torch.from_numpy(train_y).type(torch.Tensor)
test_x = torch.from_numpy(test_x).type(torch.Tensor)
test_y = torch.from_numpy(test_y).type(torch.Tensor)

# 定义模型
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型
model = GRUModel(input_dim, hidden_dim, output_dim, num_layers)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    outputs = model(train_x)
    optimizer.zero_grad()
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 模型评估
model.eval()
train_pred = model(train_x)
test_pred = model(test_x)
train_pred = scaler.inverse_transform(train_pred.detach().numpy())
train_y = scaler.inverse_transform(train_y.detach().numpy())
test_pred = scaler.inverse_transform(test_pred.detach().numpy())
test_y = scaler.inverse_transform(test_y.detach().numpy())

# 计算均方误差
mse_train = mean_squared_error(train_y, train_pred)
mse_test = mean_squared_error(test_y, test_pred)
print('MSE ON TRAIN:', mse_train)
print('MSE ON TEST:', mse_test)

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(train_y, label='Actual Train')
plt.plot(train_pred, label='Predicted Train')
plt.legend()
plt.title('GRU Predictions on Train Data')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(test_y, label='Actual Test')
plt.plot(test_pred, label='Predicted Test')
plt.legend()
plt.title('GRU Predictions on Test Data')
plt.show()

# 保存模型
torch.jit.save(torch.jit.script(model), 'pthfile/gru_model_script.pth')

# 加载模型
loaded_model = torch.jit.load('pthfile/gru_model_script.pth')

# 使用加载的模型进行预测
loaded_model.eval()
test_pred_loaded_model = loaded_model(test_x)
test_pred_loaded_model = scaler.inverse_transform(test_pred_loaded_model.detach().numpy())

# 绘制加载的模型的预测结果
plt.figure(figsize=(12, 6))
plt.plot(test_y, label='Actual Test')
plt.plot(test_pred_loaded_model, label='Predicted Test (Loaded Model)')
plt.legend()
plt.title('GRU Predictions on Test Data (Loaded Model)')
plt.show()

