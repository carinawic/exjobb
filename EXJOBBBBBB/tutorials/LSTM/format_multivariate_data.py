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
	return numpy.array(X), numpy.array(y)

if __name__ == "__main__":
    #prepare_data()

    """
    
    An LSTM model needs sufficient context to learn a mapping from an input sequence to an output value. LSTMs can support parallel input time series as separate variables or features. Therefore, we need to split the data into samples maintaining the order of observations across the two input sequences.
    
    """
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
    # choose a number of time steps
    n_steps = 2
    # convert into input/output
    X, y = split_sequences(dataset_stacked, n_steps)
    print(X.shape, y.shape)
    # summarize the data
    for i in range(len(X)):
        print(X[i], y[i])

    # the dataset knows the number of features, e.g. 2
    n_features = X.shape[2]
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=50, verbose=0)
    # demonstrate prediction
    x_input = numpy.array([[80, 85], [90, 95], [100, 105]])
    x_input = x_input.reshape((1, n_steps, n_features))

    print("x_input")
    print(x_input)
    yhat = model.predict(x_input, verbose=0)
    print(yhat)