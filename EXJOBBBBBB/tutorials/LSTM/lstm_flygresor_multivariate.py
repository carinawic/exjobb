# chapter 2 of https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
import pandas as pd
import matplotlib.pyplot as plt 
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime

click_outs = []
week = []
clicks_media = []
impr_media = []
trainTestLimit = 1000

# TODO: 
# 1. add marketing (and maybe week) to click_outs and put graphs into document
# do for covid and post-covid separately
# 
# 2. get weather dataset
# plot weather next to clickouts
# 
# calculate p-value for every extreme weather effect 
# "extreme snow" - clickouts, "not extreme snow" - clickouts
#
# add weather as input to model

np.random.seed(0)

def unpickle_data_as_Impr():

    df = pd.read_pickle('Media.p').impr
    return df

def unpickle_data_as_clicks():

    df = pd.read_pickle('Media.p').clicks
    return df

def prepare_data():

    global click_outs, week, clicks_media, impr_media

    # read dataset
    df_complete = pd.read_csv('Time.csv')

    # read media_clicks_df as pd df and add sum_total col
    media_clicks_df = pd.read_csv('media_clicks.csv')
    media_clicks_df['sum_total'] = media_clicks_df.sum(axis=1)

    media_imprs_df = pd.read_csv('media_imprs.csv')
    media_imprs_df['sum_total'] = media_imprs_df.sum(axis=1)

    precovid_startdate = '2016-01-01'
    precovid_enddate = '2020-02-19'
    postcovid_startdate = '2020-02-20'
    postcovid_enddate = '2021-12-01'
    
    # mask between certain dates DURING COVID
    mask_clickouts = (df_complete.iloc[:, 0] > precovid_startdate) & (df_complete.iloc[:, 0] <= precovid_enddate)
    mask_media_clicks = (media_clicks_df.iloc[:, 0] > precovid_startdate) & (media_clicks_df.iloc[:, 0] <= precovid_enddate)
    mask_media_imprs = (media_imprs_df.iloc[:, 0] > precovid_startdate) & (media_imprs_df.iloc[:, 0] <= precovid_enddate)
 
    click_outs = np.array(df_complete['clicks_out'][mask_clickouts].values)
    week = np.array(df_complete['week'][mask_clickouts].values)

    clicks_media = np.array(media_clicks_df["sum_total"][mask_media_clicks].values)
    impr_media = np.array(media_clicks_df["sum_total"][mask_media_imprs].values)

    # len is 1510 "pre covid"
    # len is 651 "post covid"
    #print(len(clicks_media))
    #print(len(impr_media))
    #print(len(click_outs))
    #print(len(week))

    #x = clicksMedia['Media_Bing_lowerfunnel_search_brand']
    #print(x[mask_media1])

def split_sequences(sequences, n_steps, trainTestLimit):
        X, y = list(), list()
        for i in range(len(sequences)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the dataset
            if end_ix > len(sequences):
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix-1, -1]
            X.append(seq_x)
            y.append(seq_y)

        y.pop(0)
        X.pop(-1)

        return np.array(X[:trainTestLimit]), np.array(X[trainTestLimit:]), np.array(y[:trainTestLimit]), np.array(y[trainTestLimit:])

def workingexample():
    # split a multivariate sequence into samples
    

    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset
    dataframe = pd.read_csv('date_and_clicks.csv', usecols=[1], engine='python')
    dataset = dataframe.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets

    look_back = 5
    n_features = 1

    # define input sequence
    #in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    #in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    #out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

    
    #click_outs = []
    #week = []
    #clicks_media = []
    #impr_media = []
        
    #in_seq1 = clicks_media# clicks_media
    #in_seq2 = impr_media# impr_media
    in_seq2 = week# impr_media
    out_seq = click_outs

    #pop(0)
    #in_seq1 = in_seq1[1:]
    in_seq2 = in_seq2[1:]
    out_seq = out_seq[:-1]

    # convert to [rows, columns] structure
    #in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    out_seq = out_seq.reshape((len(out_seq), 1))

    # horizontally stack columns
    #dataset_stacked = np.hstack((in_seq1, in_seq2, out_seq))
    dataset_stacked = np.hstack((in_seq2, out_seq))

    Xtrain, Xtest, ytrain, ytest = split_sequences(dataset_stacked, look_back, trainTestLimit)
    #print("shapes")
    #print(Xtrain.shape, ytrain.shape)
    #print(Xtest.shape, ytest.shape)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, n_features+1))) 
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # fit model
    model.fit(Xtrain, ytrain, epochs=1, batch_size=1, verbose=2)
    trainPredict = model.predict(Xtrain)
    testPredict = model.predict(Xtest)

    # invert predictions in case we used minmaxscaler earlier
    #trainPredict = scaler.inverse_transform(trainPredict)
    #ytrain = scaler.inverse_transform([ytrain])
    #testPredict = scaler.inverse_transform(testPredict)
    #ytest = scaler.inverse_transform([ytest])

    trainPredict = np.array(trainPredict,dtype=float)
    ytrain = np.array(ytrain,dtype=float)
    testPredict = np.array(testPredict,dtype=float)
    ytest = np.array(ytest,dtype=float)
    
    print(Xtrain)
    print(ytrain) 
    # [ 45.  65.  85. 105. 125.]
    print(testPredict)
    """
    [[148.57341003]
    [170.15072632]
    [191.95950317]
    """
    print(ytest) 
    # [145. 165. 185.]

    # calculate root mean squared error
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 
    trainScore = math.sqrt(mean_squared_error(ytrain, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(ytest, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    # plotting training data ytrain
    plt.plot(range(len(ytrain)),ytrain,'-', label="training data ytrain")
    
    # plotting training prediction trainPredict
    trainPredictFlatten = trainPredict.flatten()
    plt.plot(range(len(trainPredictFlatten)),trainPredictFlatten,'-', label="train prediction")
    
    #plotting testing data ytest
    plt.plot(
        range(len(ytrain), len(ytest) + len(ytrain)),
        ytest,'-', label="testing data ytest")
        
    #plotting testing prediction testPredict
    testPredictFlatten = testPredict.flatten()
    plt.plot(range(len(ytrain),len(testPredictFlatten)+len(ytrain)),testPredictFlatten,'-', label="test prediction")

    plt.legend()
    plt.show()

if __name__ == "__main__":
    prepare_data()
    workingexample()
