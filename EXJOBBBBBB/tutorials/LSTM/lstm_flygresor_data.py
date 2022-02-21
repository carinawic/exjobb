# chapter 2 of https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
import pandas as pd
import matplotlib.pyplot as plt
import numpy 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime


def prepare_data():

    # read dataset
    df_complete = pd.read_csv('Time.csv')

    # mask between certain dates
    mask3 = (df_complete.iloc[:, 0] > '2020-02-19') & (df_complete.iloc[:, 0] <= '2021-12-01')

    # only pick date and click_outs
    df_date_and_clicks = df_complete[['date', 'clicks_out']]

    # apply mask
    df_date_and_clicks_masked = df_date_and_clicks.loc[mask3]

    print(df_date_and_clicks_masked)

    df_date_and_clicks_masked.to_csv('date_and_clicks.csv', index=False)


def workingexample():
    # split a multivariate sequence into samples
    def split_sequences(sequences, n_steps):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)
        return numpy.array(X[:,10]), numpy.array(X[10,:]), numpy.array(y[:,10]), numpy.array(y[10,:])

    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])

        return numpy.array(dataX), numpy.array(dataY)
    # fix random seed for reproducibility
    numpy.random.seed(7)
    # load the dataset
    dataframe = pd.read_csv('date_and_clicks.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    print(dataset)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    # reshape into X=t and Y=t+1
    look_back = 5

    print("ABC")
    print(train)
    # with look_back = 5, then X (previous) is 
    # trainX = [10,15,20,25,30] and trainY = [35]
    # test is formatted the same way
    
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]


    trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    



    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()

if __name__ == "__main__":
    #prepare_data()
    workingexample()
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 