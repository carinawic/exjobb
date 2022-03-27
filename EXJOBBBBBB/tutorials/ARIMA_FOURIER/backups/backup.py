import pandas as pd
import numpy as np
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima.utils import c, diff
# chapter 2 of https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from pmdarima.datasets import load_lynx
from pmdarima.arima.utils import nsdiffs

import matplotlib.pyplot as plt 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, minmax_scale 
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error
import datetime

# https://stackoverflow.com/questions/49235508/statsmodel-arima-multiple-input
# https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
# https://www.quora.com/How-does-multivariate-ARIMA-work

click_outs = []
week = []
click_outs_df = ""

clicks_Search = []
clicks_Inactive = []
clicks_Active = []

def prepare_data():

    global click_outs, week, clicks_Search, clicks_Inactive, clicks_Active, click_outs_df

    # read dataset
    df_complete = pd.read_csv('Time.csv')

    # read media_clicks_df as pd df and add sum_total col
    media_clicks_df = pd.read_csv('media_clicks.csv')
    
    media_clicks_df['media_clicks_SEARCH_df'] = media_clicks_df['Media_Bing_lowerfunnel_search_brand'] + media_clicks_df['Media_Bing_midfunnel_search_midbrand'] + media_clicks_df['Media_Bing_upperfunnel_search_nobrand'] + media_clicks_df['Media_Google_lowerfunnel_search_brand'] + media_clicks_df['Media_Google_midfunnel_search_midbrand'] + media_clicks_df['Media_Google_upperfunnel_search_nobrand'] 
    
    media_clicks_df['media_clicks_INACTIVE_df'] = media_clicks_df['Media_Google_video_lowerfunnel_Youtube'] + media_clicks_df['Media_Google_video_upperfunnel_Youtube'] + media_clicks_df['Media_Youtube_Masthead_upperfunnel_video'] + media_clicks_df['Media_Online_radio_upperfunnel'] + media_clicks_df['Media_Radio_upperfunnel'] + media_clicks_df['Media_TV_upperfunnel'] + media_clicks_df['Media_DBM_upperfunnel_video'] + media_clicks_df['Media_DC_DBM_upperfunnel_video'] + media_clicks_df['Media_MediaMath_upperfunnel_video'] + media_clicks_df['Media_Snapchat_upperfunnel_video'] + media_clicks_df['Media_Tiktok_upperfunnel_video'] + media_clicks_df['Media_Eurosize_upperfunnel_OOH_JCD'] + media_clicks_df['Media_Eurosize_upperfunnel_OOH_VA'] 

    media_clicks_df['media_clicks_ACTIVE_df'] =  media_clicks_df['Media_Adwell_upperfunnel_native'] + media_clicks_df['Media_DBM_lowerfunnel_display'] + media_clicks_df['Media_DBM_midfunnel_display'] + media_clicks_df['Media_DBM_upperfunnel_display'] + media_clicks_df['Media_Facebook_lowerfunnel_display'] + media_clicks_df['Media_Facebook_lowerfunnel_video'] + media_clicks_df['Media_Facebook_upperfunnel_display'] + media_clicks_df['Media_Facebook_upperfunnel_video'] + media_clicks_df['Media_Flygstart_upperfunnel_newsletter'] + media_clicks_df['Media_Google_lowerfunnel_display'] + media_clicks_df['Media_Google_midfunnel_display'] + media_clicks_df['Media_Google_upperfunnel_display'] + media_clicks_df['Media_HejSenior_upperfunnel_newsletter'] + media_clicks_df['Media_Instagram_lowerfunnel_display'] + media_clicks_df['Media_Instagram_lowerfunnel_video'] + media_clicks_df['Media_Instagram_upperfunnel_display'] + media_clicks_df['Media_Instagram_upperfunnel_video'] + media_clicks_df['Media_Newsletter_lowerfunnel'] + media_clicks_df['Media_Newsner_midfunnel_native'] + media_clicks_df['Media_Secreteescape_midfunnel_display'] + media_clicks_df['Media_Smarter_Travel_upperfunnel_affiliate'] + media_clicks_df['Media_Snapchat_upperfunnel_display'] + media_clicks_df['Media_Sociomantic_lowerfunnel_retarg_display'] + media_clicks_df['Media_Sociomantic_upperfunnel_prospecting_display'] + media_clicks_df['Media_TradeTracker_upperfunnel_affiliate'] 
    
    
    precovid_startdate = '2016-01-01'
    precovid_enddate = '2020-02-19'
    postcovid_startdate = '2020-02-20'
    postcovid_enddate = '2021-12-01'
    
    # mask between certain dates DURING COVID
    mask_clickouts = (df_complete.iloc[:, 0] > precovid_startdate) & (df_complete.iloc[:, 0] <= precovid_enddate)
    mask_media_clicks = (media_clicks_df.iloc[:, 0] > precovid_startdate) & (media_clicks_df.iloc[:, 0] <= precovid_enddate)

    click_outs = np.array(df_complete['clicks_out'][mask_clickouts].values)
    week = np.array(df_complete['week'][mask_clickouts].values)

    click_outs_df = df_complete['clicks_out'][mask_clickouts]
 
    clicks_Search = np.array(media_clicks_df["media_clicks_SEARCH_df"][mask_media_clicks].values)
    clicks_Inactive = np.array(media_clicks_df["media_clicks_INACTIVE_df"][mask_media_clicks].values)
    clicks_Active = np.array(media_clicks_df["media_clicks_ACTIVE_df"][mask_media_clicks].values)

    clicks_Search = minmax_scale(clicks_Search, feature_range=(0,500))
    clicks_Inactive = minmax_scale(clicks_Inactive, feature_range=(0,500))
    clicks_Active = minmax_scale(clicks_Active, feature_range=(0,500))


def arima():
    

    y = click_outs_df
    y_to_train = y.iloc[:(len(y)-365)]
    y_to_test = y.iloc[(len(y)-365):] # last year for testing

  
    #y_to_train = y.iloc[:(len(y)-365)]
    #y_to_test = y.iloc[(len(y)-365):] # last year for testing


    # without exog:
    # arima_model = auto_arima(y_to_train, seasonal=True, m=7)
    # y_arima_forecast = arima_model.predict(n_periods=365)

    # prepare Fourier terms
        
    exog = pd.DataFrame({'date': y.index})
    exog = exog.set_index(pd.PeriodIndex(exog['date'], freq='D'))
        
    exog['sin365'] = np.sin(  np.pi * exog.index.dayofyear / 365.25)
    exog['cos365'] = np.cos(  np.pi * exog.index.dayofyear / 365.25)
    exog['sin365_2'] = np.sin(2 * np.pi * exog.index.dayofyear / 365.25)
    exog['cos365_2'] = np.cos(2 * np.pi * exog.index.dayofyear / 365.25)

    exog = exog.drop(columns=['date'])
    exog_to_train = exog.iloc[:(len(y)-365)]
    exog_to_test = exog.iloc[(len(y)-365):]

    

    # Fit model
    arima_exog_model = sm.tsa.statespace.SARIMAX(y_to_train.values, exog=exog, trend='n', order=(0,1,0), seasonal_order=(1,1,1,52))

    # Forecast
    y_arima_exog_forecast = arima_exog_model.predict(n_periods=365, exog=exog_to_test)
    
    
    print('MAE', np.mean(np.abs(y_arima_exog_forecast - y_to_test)))
    print('RMSE', np.sqrt(np.sum((y_arima_exog_forecast - y_to_test)**2)/len(y_to_test)))
    # solve this for multivariate 

    print(arima_exog_model.summary())


    plt.plot( range(len(y_to_train)), y_to_train, color='green')
    plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='blue')
    plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_arima_exog_forecast, color="red")

    #print("forecast:")
    #print(y_arima_exog_forecast)
    pm.plot_acf(y_arima_exog_forecast)


if __name__ == "__main__":
    prepare_data()
    arima()
