# chapter 2 of https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
import pandas as pd
import matplotlib.pyplot as plt 
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, minmax_scale 
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error
import datetime

click_outs = []
week = []
clicks_media = []
impr_media = []
trainTestLimit = 1145 # 430 or 1000

clicks_Search = []
clicks_Inactive = []
clicks_Active = []

"""
Search:
['Media_Bing_lowerfunnel_search_brand',
 'Media_Bing_midfunnel_search_midbrand',
 'Media_Bing_upperfunnel_search_nobrand',
 'Media_Google_lowerfunnel_search_brand',
 'Media_Google_midfunnel_search_midbrand',
 'Media_Google_upperfunnel_search_nobrand']
Inactive:
['Media_Google_video_lowerfunnel_Youtube',
 'Media_Google_video_upperfunnel_Youtube',
 'Media_Youtube_Masthead_upperfunnel_video',
 'Media_Online_radio_upperfunnel',
 'Media_Radio_upperfunnel',
 'Media_TV_upperfunnel',
 'Media_DBM_upperfunnel_video',
 'Media_DC_DBM_upperfunnel_video',
 'Media_MediaMath_upperfunnel_video',
 'Media_Snapchat_upperfunnel_video',
 'Media_Tiktok_upperfunnel_video',
 'Media_Eurosize_upperfunnel_OOH_JCD',
 'Media_Eurosize_upperfunnel_OOH_VA']
Active:
['Media_Adwell_upperfunnel_native',
 'Media_DBM_lowerfunnel_display',
 'Media_DBM_midfunnel_display',
 'Media_DBM_upperfunnel_display',
 'Media_Facebook_lowerfunnel_display',
 'Media_Facebook_lowerfunnel_video',
 'Media_Facebook_upperfunnel_display',
 'Media_Facebook_upperfunnel_video',
 'Media_Flygstart_upperfunnel_newsletter',
 'Media_Google_lowerfunnel_display',
 'Media_Google_midfunnel_display',
 'Media_Google_upperfunnel_display',
 'Media_HejSenior_upperfunnel_newsletter',
 'Media_Instagram_lowerfunnel_display',
 'Media_Instagram_lowerfunnel_video',
 'Media_Instagram_upperfunnel_display',
 'Media_Instagram_upperfunnel_video',
 'Media_LinkedIn_upperfunnel_display', <- ?
 'Media_Newsletter_lowerfunnel',
 'Media_Newsner_midfunnel_native',
 'Media_Secreteescape_midfunnel_display',
 'Media_Smarter_Travel_upperfunnel_affiliate',
 'Media_Snapchat_upperfunnel_display',
 'Media_Sociomantic_lowerfunnel_retarg_display',
 'Media_Sociomantic_upperfunnel_prospecting_display',
 'Media_TradeTracker_upperfunnel_affiliate']
"""

np.random.seed(0)

def unpickle_data_as_Impr():

    df = pd.read_pickle('Media.p').impr
    return df

def unpickle_data_as_clicks():

    df = pd.read_pickle('Media.p').clicks
    return df

def prepare_data():

    global click_outs, week, clicks_media, impr_media, clicks_Search, clicks_Inactive, clicks_Active, clicks_Extreme

    # read dataset
    df_complete = pd.read_csv('Time.csv')

    # read media_clicks_df as pd df and add sum_total col
    media_clicks_df = pd.read_csv('media_clicks.csv')
    media_imprs_df = pd.read_csv('media_imprs.csv')
    
    media_clicks_df['sum_total'] = media_clicks_df.sum(axis=1)

    media_clicks_df['media_clicks_SEARCH_df'] = media_clicks_df['Media_Bing_lowerfunnel_search_brand'] + media_clicks_df['Media_Bing_midfunnel_search_midbrand'] + media_clicks_df['Media_Bing_upperfunnel_search_nobrand'] + media_clicks_df['Media_Google_lowerfunnel_search_brand'] + media_clicks_df['Media_Google_midfunnel_search_midbrand'] + media_clicks_df['Media_Google_upperfunnel_search_nobrand'] 
    
    media_clicks_df['media_clicks_INACTIVE_df'] = media_imprs_df['Media_Google_video_lowerfunnel_Youtube'] + media_imprs_df['Media_Google_video_upperfunnel_Youtube'] + media_imprs_df['Media_Online_radio_upperfunnel'] + media_imprs_df['Media_Radio_upperfunnel'] + media_imprs_df['Media_TV_upperfunnel'] + media_imprs_df['Media_DBM_upperfunnel_video'] + media_imprs_df['Media_DC_DBM_upperfunnel_video'] + media_imprs_df['Media_MediaMath_upperfunnel_video'] + media_imprs_df['Media_Snapchat_upperfunnel_video'] + media_imprs_df['Media_Tiktok_upperfunnel_video'] + media_imprs_df['Media_Eurosize_upperfunnel_OOH_JCD'] + media_imprs_df['Media_Eurosize_upperfunnel_OOH_VA'] 

    # should also have Media_Youtube_Masthead_upperfunnel_video
    media_clicks_df['media_clicks_ACTIVE_df'] =  media_clicks_df['Media_Adwell_upperfunnel_native'] + media_clicks_df['Media_DBM_lowerfunnel_display'] + media_clicks_df['Media_DBM_midfunnel_display'] + media_clicks_df['Media_DBM_upperfunnel_display'] + media_clicks_df['Media_Facebook_lowerfunnel_display'] + media_clicks_df['Media_Facebook_lowerfunnel_video'] + media_clicks_df['Media_Facebook_upperfunnel_display'] + media_clicks_df['Media_Facebook_upperfunnel_video'] + media_clicks_df['Media_Flygstart_upperfunnel_newsletter'] + media_clicks_df['Media_Google_lowerfunnel_display'] + media_clicks_df['Media_Google_midfunnel_display'] + media_clicks_df['Media_Google_upperfunnel_display'] + media_clicks_df['Media_HejSenior_upperfunnel_newsletter'] + media_clicks_df['Media_Instagram_lowerfunnel_display'] + media_clicks_df['Media_Instagram_lowerfunnel_video'] + media_clicks_df['Media_Instagram_upperfunnel_display'] + media_clicks_df['Media_Instagram_upperfunnel_video'] + media_clicks_df['Media_Newsletter_lowerfunnel'] + media_clicks_df['Media_Newsner_midfunnel_native'] + media_clicks_df['Media_Secreteescape_midfunnel_display'] + media_clicks_df['Media_Smarter_Travel_upperfunnel_affiliate'] + media_clicks_df['Media_Snapchat_upperfunnel_display'] + media_clicks_df['Media_Sociomantic_lowerfunnel_retarg_display'] + media_clicks_df['Media_Sociomantic_upperfunnel_prospecting_display'] + media_clicks_df['Media_TradeTracker_upperfunnel_affiliate'] 
    
    

    precovid_startdate = '2016-01-01'
    precovid_enddate = '2020-02-19'
    postcovid_startdate = '2020-02-20'
    postcovid_enddate = '2021-12-01'
    
    # mask between certain dates DURING COVID
    mask_clickouts = (df_complete.iloc[:, 0] > precovid_startdate) & (df_complete.iloc[:, 0] <= precovid_enddate)
    mask_media_clicks = (media_clicks_df.iloc[:, 0] > precovid_startdate) & (media_clicks_df.iloc[:, 0] <= precovid_enddate)
    #mask_media_imprs = (media_imprs_df.iloc[:, 0] > precovid_startdate) & (media_imprs_df.iloc[:, 0] <= precovid_enddate)
 
    click_outs = np.array(df_complete['clicks_out'][mask_clickouts].values)
    week = np.array(df_complete['week'][mask_clickouts].values)

    clicks_Search = np.array(media_clicks_df["media_clicks_SEARCH_df"][mask_media_clicks].values)
    clicks_Inactive = np.array(media_clicks_df["media_clicks_INACTIVE_df"][mask_media_clicks].values)
    clicks_Active = np.array(media_clicks_df["media_clicks_ACTIVE_df"][mask_media_clicks].values)
    clicks_Extreme = np.array(media_clicks_df['Media_Youtube_Masthead_upperfunnel_video'][mask_media_clicks].values)

    #scaling the inputs
    #clicks_Search = minmax_scale(clicks_Search, feature_range=(0,1))
    #clicks_Inactive = minmax_scale(clicks_Inactive, feature_range=(0,1))
    #clicks_Active = minmax_scale(clicks_Active, feature_range=(0,1))
    #clicks_Extreme = minmax_scale(clicks_Extreme, feature_range=(0,1))
    
    clicks_Search = minmax_scale(clicks_Search, feature_range=(0,500))
    clicks_Inactive = minmax_scale(clicks_Inactive, feature_range=(0,500))
    clicks_Active = minmax_scale(clicks_Active, feature_range=(0,500))
    clicks_Extreme = minmax_scale(clicks_Extreme, feature_range=(0,500))

    

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

    look_back = 7
    n_features = 4

    # define input sequence
    #in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    #in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    #in_seq3 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    #out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

        
    in_seq1 = clicks_Search
    in_seq2 = clicks_Active
    in_seq3 = clicks_Inactive
    
    in_seq4 = clicks_Extreme

    out_seq = click_outs

    # remove the first item of each data because the [0-3] first clicks match with the [1-4] first clickouts
    """in_seq1 = in_seq1[1:]
    in_seq2 = in_seq2[1:]
    in_seq3 = in_seq3[1:]
    in_seq4 = in_seq4[1:]

    out_seq = out_seq[:-1]"""

    in_seq1 = in_seq1
    in_seq2 = in_seq2
    in_seq3 = in_seq3
    in_seq4 = in_seq4

    out_seq = out_seq



    # convert to [rows, columns] structure
    #in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    in_seq2 = in_seq2.reshape((len(in_seq2), 1))
    in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    in_seq4 = in_seq3.reshape((len(in_seq4), 1))

    out_seq = out_seq.reshape((len(out_seq), 1))

    # horizontally stack columns
    #dataset_stacked = np.hstack((in_seq1, in_seq2, out_seq))
    dataset_stacked = np.hstack((in_seq1, in_seq2, in_seq3, in_seq4, out_seq))

    Xtrain, Xtest, ytrain, ytest = split_sequences(dataset_stacked, look_back, trainTestLimit)
    #print("shapes")
    #print(Xtrain.shape, ytrain.shape)
    #print(Xtest.shape, ytest.shape)

    model = Sequential()
    model.add(LSTM(32, activation='relu', input_shape=(look_back, n_features+1))) 
    model.add(Dense(16, activation='relu')) 
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    """
    
    
    64-64
    64-32
    32-64

    128 - 64
    64 - 128

    64-16-1
    16-64-1
    16-16-1
    64-64-1
    16-16-16-1
    
    """


    
    from tensorflow.keras.callbacks import EarlyStopping
    callback=EarlyStopping(monitor="loss",patience=30)

    # fit model
    history = model.fit(Xtrain, ytrain, epochs=400, batch_size=1, verbose=2, callbacks=[callback]) # epochs=200
    trainPredict = model.predict(Xtrain)
    testPredict = model.predict(Xtest)

    #print(history)

    # invert predictions in case we used minmaxscaler earlier
    #trainPredict = scaler.inverse_transform(trainPredict)
    #ytrain = scaler.inverse_transform([ytrain])
    #testPredict = scaler.inverse_transform(testPredict)
    #ytest = scaler.inverse_transform([ytest])

    trainPredict = np.array(trainPredict,dtype=float)
    ytrain = np.array(ytrain,dtype=float)
    testPredict = np.array(testPredict,dtype=float)
    ytest = np.array(ytest,dtype=float)
    
    #print(Xtrain)
    #print(ytrain) 
    # [ 45.  65.  85. 105. 125.]
    #print(testPredict)
    """
    [[148.57341003]
    [170.15072632]
    [191.95950317]
    """
    #print(ytest) 
    # [145. 165. 185.]

    # calculate root mean squared error
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 
    trainScore = math.sqrt(mean_squared_error(ytrain, trainPredict))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(ytest, testPredict))
    print('Test Score: %.2f RMSE' % (testScore))

    # calculate root mean squared error
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 
    trainScore = mean_absolute_error(ytrain, trainPredict)
    print('Train Score: %.2f MAE' % (trainScore))
    testScore = mean_absolute_error(ytest, testPredict)
    print('Test Score: %.2f MAE' % (testScore))

    
    # calculate root mean squared error
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 
    trainScore = mean_absolute_percentage_error(ytrain, trainPredict)
    print('Train Score: %.2f MAPE' % (trainScore))
    testScore = mean_absolute_percentage_error(ytest, testPredict)
    print('Test Score: %.2f MAPE' % (testScore))

    # plotting training data ytrain
    plt.plot(range(len(ytrain)),ytrain,'-', label="training data ytrain", color="red")
    
    # plotting training prediction trainPredict
    trainPredictFlatten = trainPredict.flatten()
    plt.plot(range(len(trainPredictFlatten)),trainPredictFlatten,'-', label="train prediction", color="green")
    
    #plotting testing data ytest
    plt.plot(
        range(len(ytrain), len(ytest) + len(ytrain)),
        ytest,'-', label="testing data ytest", color='purple')
        
    #plotting testing prediction testPredict
    testPredictFlatten = testPredict.flatten()
    plt.plot(range(len(ytrain),len(testPredictFlatten)+len(ytrain)),testPredictFlatten,'-', label="test prediction", color="blue")

    
    plt.xlabel('days')
    plt.ylabel('flights')
    plt.title('Forecast using LSTM')

    plt.legend()
    plt.show()

if __name__ == "__main__":
    prepare_data()
    workingexample()
