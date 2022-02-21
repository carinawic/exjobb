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


click_outs = []
week = []

def prepare_data():

    global click_outs, week

    # read dataset
    df_complete = pd.read_csv('Time.csv')

    # mask between certain dates DURING COVID
    mask3 = (df_complete.iloc[:, 0] > '2020-02-19') & (df_complete.iloc[:, 0] <= '2021-12-01')
 
    click_outs = numpy.array(df_complete['clicks_out'][mask3].values)
    week = numpy.array(df_complete['week'][mask3].values)

    

def workingexample():
    # split a multivariate sequence into samples
    def split_sequences(sequences, n_steps, trainTestLimit):
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
        return numpy.array(X[:trainTestLimit]), numpy.array(X[trainTestLimit:]), numpy.array(y[:trainTestLimit]), numpy.array(y[trainTestLimit:])

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

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets

    look_back = 2

    # define input sequence
    in_seq1 = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    in_seq2 = numpy.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    out_seq = numpy.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

    # convert to [rows, columns] structure
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))

    # horizontally stack columns
    dataset_stacked = numpy.hstack((in_seq1, in_seq2, out_seq))

    Xtrain, Xtest, ytrain, ytest = split_sequences(dataset_stacked, look_back, 5)
    print("shapes")
    print(Xtrain.shape, ytrain.shape)
    print(Xtest.shape, ytest.shape)
    # summarize the data


    # the dataset knows the number of features, e.g. 2
    n_features = 2


    # we want data like 
    # [in1a, in1b]
    # [in2a, in2b]
    # ...

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(2, n_features))) #input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model

    model.fit(Xtrain, ytrain, epochs=50, batch_size=1, verbose=2)

    trainPredict = model.predict(Xtrain)
    testPredict = model.predict(Xtest)

    # invert predictions WHY??
    trainPredict = scaler.inverse_transform(trainPredict)
    ytrain = scaler.inverse_transform([ytrain])
    testPredict = scaler.inverse_transform(testPredict)
    ytest = scaler.inverse_transform([ytest])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(ytrain[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(ytest[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

    """
    
    # make predictions
    trainPredict = model.predict(Xtrain)
    testPredict = model.predict(Xtest)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    ytrain = scaler.inverse_transform([ytrain])
    testPredict = scaler.inverse_transform(testPredict)
    ytest = scaler.inverse_transform([ytest])
    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(ytrain[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(ytest[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))
    # shift train predictions for plotting
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # shift test predictions for plotting
    
    #testPredictPlot = numpy.empty_like(dataset)
    #testPredictPlot[:, :] = numpy.nan
    #testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    
    # plot baseline and predictions
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    #plt.plot(testPredictPlot)
    plt.show()
    
    """

if __name__ == "__main__":
    #prepare_data()
    prepare_data()
    workingexample()
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 