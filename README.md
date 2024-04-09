# Work report up to now
So far, we have completed the training and deployment of six time series models: LSTM, linear regression, ARIMA, random forest, SVM, and GRU, and used them for training and prediction on stock data.
The next step is to add corresponding trading strategies to each model, in order to achieve the goal of backtesting and calculating returns on historical stock data.

The most obvious difficulty currently encountered is that the training methods of each model are not exactly the same, and the requirements for data are also not the same. However, in the end, it is necessary to achieve unified input and format in web page display. We will find suitable methods



To test the model, directly run test.ipynb to use the model saved in pth file to predict the stock price and draw the plot and get the MSE of each.
