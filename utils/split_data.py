import numpy as np

def split_data(stock, lookback):
    data_raw = stock
    data = []
    for index in range(len(data_raw) - lookback):
        data.append(data_raw[index: index + lookback])
        
    data = np.array(data)
    test_set_size = int(np.round(0.2*stock.shape[0]))
    train_set_size = stock.shape[0] - test_set_size

    x_train = data[:train_set_size, :-1,:]
    y_train = data[:train_set_size, -1, :]

    x_test = data[train_set_size:, :-1, :]
    y_test = data[train_set_size:, -1, :]
    
    return (x_train, y_train, x_test, y_test)

