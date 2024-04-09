from predict_v2 import predict
from lstm_train import lstm_train
import pylab as plt
import pandas as pd
import joblib
import matplotlib.dates as mdates

#####LSTM
########此文件供前后端直接调用，来对模型进行训练和预测
# lstm_train()
mse_train, mse_test, y_train, train_pred, y_test, test_pred, mtr = predict('randomforest')

plt.figure(figsize=(12, 6))
plt.plot(y_train, label='Actual Train')
plt.plot(train_pred, label='Predicted Train')
plt.legend()
plt.title('Random Forest Predictions on Train Data')
plt.show()

# 绘制测试集的预测结果

####显示好像有问题？？？
# plt.figure(figsize=(12, 6))
# plt.plot(y_test, label='Actual Test')
# plt.plot(test_pred, label='Predicted Test')
# plt.legend()
# plt.title('Random Forest Predictions on Test Data')
# plt.show()




# plt.figure(figsize=(12,8))

# plt.plot(pd.to_datetime(data['Date']),data['Close'],label = 'data')
# plt.plot(pd.to_datetime(testPredictPlot['Date']),testPredictPlot['Close'],label = 'test_pred')
# plt.plot(pd.to_datetime(trainPredictPlot['Date']),trainPredictPlot['Close'],label = 'train_pred')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.gcf().autofmt_xdate()
# plt.title('Predictions vs Actual Data')
# plt.show()


# print('!MSE ON TRAIN:'+str(mse_train))
# print('!MSE ON TEST:'+str(mse_test))





# #####线性回归
# mse,actual,pred = predict('linearregression')


# plt.plot(actual['Date'],actual['Close'], label ='actual')
# plt.plot(pred['Date'],pred['Close'],label = 'pred' )
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
# plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
# plt.xlabel('Date')
# plt.ylabel('Value')
# plt.gcf().autofmt_xdate()
# plt.title('Predictions vs Actual Data')
# plt.show()
# print("Mean Squared Error at prediction:", mse)

