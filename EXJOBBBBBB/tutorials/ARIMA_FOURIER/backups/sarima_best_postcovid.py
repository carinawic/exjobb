import itertools
from turtle import width
import pandas as pd
import numpy as np

import math
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import auto_arima
from itertools import cycle, islice
from statsmodels.tsa.seasonal import seasonal_decompose
from pandas import read_csv
from matplotlib import pyplot
from numpy import polyfit
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error

from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from scipy.optimize.minpack import curve_fit

from sklearn.preprocessing import MinMaxScaler, minmax_scale 

# https://stackoverflow.com/questions/49235508/statsmodel-arima-multiple-input
# https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
# https://www.quora.com/How-does-multivariate-ARIMA-work


click_outs = []
week = []
clicks_media = []
impr_media = []

clicks_Search = []
clicks_Inactive = []
clicks_Active = []
clicks_Extreme = []

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
    mask_clickouts = (df_complete.iloc[:, 0] > postcovid_startdate) & (df_complete.iloc[:, 0] <= postcovid_enddate)
    mask_media_clicks = (media_clicks_df.iloc[:, 0] > postcovid_startdate) & (media_clicks_df.iloc[:, 0] <= postcovid_enddate)

 
    click_outs = np.array(df_complete['clicks_out'][mask_clickouts].values)
    week = np.array(df_complete['week'][mask_clickouts].values)


    clicks_Search = np.array(media_clicks_df["media_clicks_SEARCH_df"][mask_media_clicks].values)
    clicks_Inactive = np.array(media_clicks_df["media_clicks_INACTIVE_df"][mask_media_clicks].values)
    clicks_Active = np.array(media_clicks_df["media_clicks_ACTIVE_df"][mask_media_clicks].values)
    clicks_Extreme = np.array(media_clicks_df['Media_Youtube_Masthead_upperfunnel_video'][mask_media_clicks].values)

    #scaling the inputs
    clicks_Search = minmax_scale(clicks_Search, feature_range=(0,500))
    clicks_Inactive = minmax_scale(clicks_Inactive, feature_range=(0,500))
    clicks_Active = minmax_scale(clicks_Active, feature_range=(0,500))
    clicks_Extreme = minmax_scale(clicks_Extreme, feature_range=(0,500))

    """
    # plotting the media investment

    plt.xlabel('days')
    plt.ylabel('clicks')
    plt.title('Media invesment')

    plt.plot(range(len(clicks_Search)), clicks_Active, 'red', label='clicks active')
    plt.plot(range(len(clicks_Search)), clicks_Inactive, 'green', label='impressions inactive')
    plt.plot(range(len(clicks_Search)), clicks_Search, 'blue', label='clicks search')
    plt.plot(range(len(clicks_Search)), clicks_Extreme, 'orange', label='clicks campaign')
    plt.legend()

    plt.show()
    """
    

def working():
    #df = pd.read_csv('kaggle_sales.csv')
    #df = df[(df['store'] == 1) & (df['item'] == 1)] # item 1 in store 1
    #df = df.set_index('date')
    #y = df['sales']

    df = pd.read_csv('Time.csv')
    #df = df.set_index('date')

    precovid_startdate = '2016-01-01'
    precovid_enddate = '2020-02-19'
    postcovid_startdate = '2020-02-20'
    postcovid_enddate = '2021-12-01'

    mask_clickouts = (df.iloc[:, 0] > postcovid_startdate) & (df.iloc[:, 0] <= postcovid_enddate)

    y = df['clicks_out'][mask_clickouts]

    print("HEEEEEEEEEERRRRRRRRRRRREEEEEEEEEEEE")
    
    print("len(y)")
    print(len(y))
    #renamed = df.rename(columns={'clicks_out': 'decomposition of data'})
    #y = renamed['decomposition of data'][mask_clickouts]
    
    y_to_train = y.iloc[:(len(y)-365)]
    y_to_test = y.iloc[(len(y)-365):] # last year for testing

    
    print("len(y_to_train)")
    print(len(y_to_train))
    
    print("len(y_to_test)")
    print(len(y_to_test))
    

    
    # creating the 8-degree curve polnomial curve
    X = [i%365 for i in range(0, len(y.values))]
    y_vals = y.values
    degree = 16 # 8 looked reasonable
    coef = polyfit(X, y_vals, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)

    
    decompose_result = seasonal_decompose(y_to_train, model="additive", period=365)
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid
    #decompose_result.plot()
    #plt.plot( range(len(trend)), trend, color='purple')
    trend = [x for x in trend if not math.isnan(x)] # remove nan values
    
    m,b = np.polyfit(range(len(trend)),trend,1) # f(x) = m*i + b

    def func(x, a, tau, c):
        return a * np.exp(-x/tau) + c

    popt, pcov = curve_fit(func, np.array(range(len(trend))), trend)

    #coefficients = np.polyfit(np.log(range(1,len(trend))), trend[:-1], 1)

    coefficients = np.polyfit(range(len(trend)), trend, 2)
    p = np.poly1d(coefficients)
    #print("values")
    #print(p(0))
    #print(p(1))

    

    print("HERE")
    print(coefficients)

    logestimate = []
    linestimate = []
    #for i in range(len(y_vals)):
        #logestimate.append(p(i))
    #    logestimate.append(func(np.array(range(len(y_vals))), *popt))
    
    logestimate = (func(np.array(range(len(y_vals))), *popt))
    
    for i in range(len(trend)):
        linestimate.append(m*i+b)
    
    
    #plt.plot(range(len(trend)),m*range(len(trend))+b, color='purple')
    #plt.plot(range(len(trend)),m*np.log(range(len(trend)))+b, color='green')
    
    clickouts = np.array(y.values)

    
    print("len(clickouts)")
    print(len(clickouts))

    clickouts_wo_trend = []
    clickouts_wo_trend_or_seasonality = []
    
    seasonal = [x for x in seasonal if not x is(None)] # remove None values

    # creating the datasets without seasonality or trend
    for i,real_val in enumerate(clickouts):
        
        #clickouts_wo_trend.append(real_val - (m*i)) # centered around b now!
        clickouts_wo_trend.append(real_val - logestimate[i]) # centered around b now!
        clickouts_wo_trend_or_seasonality.append(clickouts_wo_trend[i] - curve[i])

    
    print("len(clickouts_wo_trend_or_seasonality)")
    print(len(clickouts_wo_trend_or_seasonality))

    ### SARIMAX model making forecast without seasonality or trend ###

    cl_train = clickouts_wo_trend_or_seasonality[:len(clickouts_wo_trend_or_seasonality)-365]
    cl_test = clickouts_wo_trend_or_seasonality[len(clickouts_wo_trend_or_seasonality)-365:]

    #arima_exog_model = auto_arima(y=cl_train, seasonal=False, m=7) ##TODO set true
    #y_arima_exog_forecast = arima_exog_model.predict(n_periods=365)

    clicks_Search_test = clicks_Search[-365:]
    clicks_Search_train = clicks_Search[:len(clicks_Search)-365]


    clicks_Active_test = clicks_Active[-365:]
    clicks_Active_train = clicks_Active[:len(clicks_Search)-365]

    clicks_Inactive_test = clicks_Inactive[-365:]
    clicks_Inactive_train = clicks_Inactive[:len(clicks_Search)-365]

    clicks_Extreme_test = clicks_Extreme[-365:]
    clicks_Extreme_train = clicks_Extreme[:len(clicks_Search)-365]

    """
    exog zeros:

    Test Score: 2545.85 RMSE
    Test Score: 1982.96 MAE
    Test Score: 0.09 MAPE


    
    """


    # Create figure
    #fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,8))
    
    # Plot the ACF of df_store_2_item_28_timeon ax1
    #plot_acf(cl_train,lags=7, zero=False, ax=ax1)

    # Plot the PACF of df_store_2_item_28_timeon ax2
    #plot_pacf(cl_train,lags=7, zero=False, ax=ax2)

    #plt.show()

    #model = SARIMAX(cl_train, order=(2,0,0), seasonal_order=(2,0,0,7), exog = np.zeros(len(cl_train)))

    # removed clicks_Inactive_train, clicks_Extreme_train
    exog_train = np.column_stack((clicks_Search_train, clicks_Active_train))
    exog_test = np.column_stack((clicks_Search_test,clicks_Active_test))



    model = SARIMAX(cl_train, order=(1,1,2), seasonal_order=(1,1,2,7), exog = exog_train, period = 360)
    model_fit = model.fit()
    #model_fit.plot_diagnostics()
    print(model_fit.summary())
    """
    Non-seasonal part:
    p = autoregressive order
    d = differencing (used for comparing the current timestep with a previous one at offset d in order to even out the trend)
    q = moving average order
    Seasonal part:
    P = seasonal AR order
    D = seasonal differencing
    Q = seasonal MA order
    S = length of seasonal pattern
    with including marketing investment as input
    (3,1,2)(3,1,2,7)
    est Score: 2647.43 RMSE
    Test Score: 2071.88 MAE
    Test Score: 0.09 MAPE
    (1,1,1)(1,1,1,7)
    Test Score: 2986.49 RMSE
    Test Score: 2461.09 MAE
    Test Score: 0.11 MAPE
    (1,1,1)(0,1,1,7)
    -> 2200
    seasonal:
    1,0,1 -> shit
    1,1,0 -> 2335

    (0,1,2)(1,0,1)
    """
    # change trend to accurately only take into account the training data in the decomposition!
    # 3,1,2 -> BEST 2890.00 RMSE
    #print(len(y_to_train))
    # one-step out-of sample forecast
    #y_arima_exog_forecast = model_fit.predict(1145,1509)
    y_arima_exog_forecast = model_fit.forecast(steps=365, exog=[[exog_test]])
    y_arima_exog_forecast_with_trend = []
    y_arima_exog_forecast_with_trend_and_seasonality = []
    #print(arima_exog_model.summary())
    i_test_values = range(len(y_vals)-365, len(y_vals))
    
    curve_test = curve[-365:]
    
    
    #y_arima_exog_forecast = cl_test # best case scenario forecast, the test data without trend or seasonality
    for i in range(len(y_arima_exog_forecast)): # go through the forecast
        #y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + m*i_test_values[i]) # linear
        y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + logestimate[i_test_values[i]])
        y_arima_exog_forecast_with_trend_and_seasonality.append(y_arima_exog_forecast_with_trend[i] + curve_test[i])
    # curve_with_trend is the curve with offset is added just for plotting nicely
    offset = 23000
    curve_with_trend = []
    #print(" b is ")
    #print( b)
    
    for i in range(len(curve)):
        #curve_with_trend.append(curve[i] + m*i - offset)
        curve_with_trend.append(curve[i] + logestimate[i] -offset)
    
    # good plots
    plt.plot( range(len(y_vals)), y_vals, color='green')
    plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='purple')
    plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_arima_exog_forecast_with_trend_and_seasonality, color='blue')
    #plt.plot( range(len(y_vals)), trend_line, color='orange')
    plt.plot( range(len(curve)), curve_with_trend, color='black')
    plt.plot( range(len(logestimate)), logestimate, color='red')
    #plt.plot( range(len(trend)), trend, color='red')
    


    # calculate root mean squared error
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 
    testScore = math.sqrt(mean_squared_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality))
    print('Test Score: %.2f RMSE' % (testScore))

    testScore = mean_absolute_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality)
    print('Test Score: %.2f MAE' % (testScore))

    testScore = mean_absolute_percentage_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality)
    print('Test Score: %.2f MAPE' % (testScore))
    
    plt.xlabel('days')
    plt.ylabel('flights')
    plt.title('Forecast using SARIMAX')

    plt.show()
    #pm.plot_acf(y_arima_exog_forecast)
if __name__ == "__main__":
    prepare_data()
    working()

