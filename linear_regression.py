import pandas as pd
from pandas_datareader import data
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pylab as plt
import joblib

yf.pdr_override()

# Load the stock price data
mtr = data.get_data_yahoo('0066.HK', start="2010-01-01", end="2020-06-30").reset_index()

# Preprocess the data
mtr['Date'] = pd.to_datetime(mtr['Date'])
mtr.set_index('Date', inplace=True)

# Select the features and target variable
X = mtr[['Open', 'High', 'Low', 'Volume']]  # Replace with your own features
y = mtr['Close']  # Replace with your own target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'pthfile/linear_regression_model.joblib')
# Make predictions on the test set
X_test_sorted = X_test.sort_index()
y_pred = model.predict(X_test_sorted)
y_test_sorted = y_test.sort_index()
# Evaluate the model
mse = mean_squared_error(y_test_sorted, y_pred)

# # print(y_test_sorted.index)
pred = pd.DataFrame({'Date':pd.to_datetime(y_test_sorted.index),'Close':y_pred})
actual = pd.DataFrame({'Date':pd.to_datetime(y_test_sorted.index),'Close':y_test_sorted.values})
plt.plot(actual['Date'],actual['Close'], label ='actual')
plt.plot(pred['Date'],pred['Close'],label = 'pred' )
plt.show()

print("Mean Squared Error at test:", mse)


predict_data = data.get_data_yahoo('0066.HK', start="2021-01-01", end="2021-04-30").reset_index()
predict_data.set_index('Date', inplace=True)
predict_x = predict_data[['Open', 'High', 'Low', 'Volume']]
predict_y = predict_data['Close']

# print(predict_x)
predict_y = predict_y.sort_index()
predict_x = predict_x.sort_index()
predic
t_result = model.predict(predict_x)
mse = mean_squared_error(y_test_sorted, y_pred)

pred = pd.DataFrame({'Date':pd.to_datetime(predict_x.index),'Close':predict_result})
actual = pd.DataFrame({'Date':pd.to_datetime(predict_x.index),'Close':predict_y.values})
plt.plot(actual['Date'],actual['Close'], label ='actual')
plt.plot(pred['Date'],pred['Close'],label = 'pred' )
plt.show()

print("Mean Squared Error at prediction:", mse)