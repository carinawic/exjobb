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

# https://stackoverflow.com/questions/49235508/statsmodel-arima-multiple-input
# https://medium.com/intive-developers/forecasting-time-series-with-multiple-seasonalities-using-tbats-in-python-398a00ac0e8a
# https://www.quora.com/How-does-multivariate-ARIMA-work



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

    mask_clickouts = (df.iloc[:, 0] > precovid_startdate) & (df.iloc[:, 0] <= precovid_enddate)

    y = df['clicks_out'][mask_clickouts]

    #renamed = df.rename(columns={'clicks_out': 'decomposition of data'})
    #y = renamed['decomposition of data'][mask_clickouts]
    
    y_to_train = y.iloc[:(len(y)-365)]
    y_to_test = y.iloc[(len(y)-365):] # last year for testing

    
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

    
    print("len(y_vals)")
    print(len(y_vals))
    
    
    # printing the 8-degree curve polynomial
    #plt.plot( range(len(y_to_train)), y_to_train, color='green')
    #plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='green')
    #plt.plot( range(len(curve)), curve, color='black', linewidth=3)
       
    
    
    decompose_result = seasonal_decompose(y_to_train, model="additive", period=365)
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid
    #decompose_result.plot()
    #plt.plot( range(len(trend)), trend, color='purple')
    trend = [x for x in trend if not math.isnan(x)] # remove nan values, then linear approx is a bit meh :)
    #plt.plot( range(len(trend)), trend, color='black')
    #plt.plot( range(len(seasonal)), seasonal, color='green')
    #plt.plot( range(365,365*2),seasonal[:365], color='blue')

    m,b = np.polyfit(range(len(trend)),trend,1) # f(x) = m*i + b
    coefficients = np.polyfit(range(len(trend)),trend,2) # f(x) = m*i + b

    p = np.poly1d(coefficients)
    print("values")
    print(p(0))
    print(p(1))

    logestimate = []
    linestimate = []
    for i in range(len(trend)):
        logestimate.append(p(i))
    
    for i in range(len(trend)):
        linestimate.append(m*i+b)
    
    plt.plot(range(len(trend)), trend, 'black')
    plt.plot(range(len(trend)), logestimate, 'red')
    plt.plot(range(len(trend)), linestimate, 'blue')
    plt.show()
    
    

    
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
        
        clickouts_wo_trend.append(real_val - (m*i)) # centered around b now!
        clickouts_wo_trend_or_seasonality.append(clickouts_wo_trend[i] - curve[i])

    
    print("len(clickouts_wo_trend_or_seasonality)")
    print(len(clickouts_wo_trend_or_seasonality))

    ### SARIMAX model making forecast without seasonality or trend ###

    cl_train = clickouts_wo_trend_or_seasonality[:len(clickouts_wo_trend_or_seasonality)-365]
    cl_test = clickouts_wo_trend_or_seasonality[len(clickouts_wo_trend_or_seasonality)-365:]

    #arima_exog_model = auto_arima(y=cl_train, seasonal=False, m=7) ##TODO set true
    #y_arima_exog_forecast = arima_exog_model.predict(n_periods=365)

    #model = SARIMAX(cl_train, order=(2,0,0), seasonal_order=(2,0,0,7), exog = np.zeros(len(cl_train)))
    model = SARIMAX(cl_train, order=(3,1,2), seasonal_order=(3,1,2,7), exog = np.zeros(len(cl_train)))
    model_fit = model.fit()

    # 2 0 0  -> big to small amplotude, follows max-ampl 5145.37 RMSE
    # 0 2 0  -> trend enourmosly skewed upwards, inf RMSE
    # 0 0 2  -> seasonality dies, just one line, 5000 RMSE
    # 2 0 2  -> looks reasonable, 2900 RMSE
    # 2 1 2  -> same as above but seem to float upwards, 3000 RMSE
    # 2 2 2  -> trend enourmosly skews away downwards
    # 2 0 1  -> 2646 RMSE
    # 1 0 2  -> 3133 RMSE
    # 3 0 1  -> 2763 RMSE
    # change trend to accurately only take into account the training data in the decomposition!
    # 3,1,2 -> BEST 2890.00 RMSE

    print(len(y_to_train))
    # one-step out-of sample forecast
    #y_arima_exog_forecast = model_fit.predict(1145,1509)
    y_arima_exog_forecast = model_fit.forecast(steps=365, exog=[[np.zeros(len(cl_test))]])

    

    print("length of y_arima_exog_forecast")
    print(len(y_arima_exog_forecast))


    y_arima_exog_forecast_with_trend = []
    y_arima_exog_forecast_with_trend_and_seasonality = []

    #print(arima_exog_model.summary())


    i_test_values = range(len(y_vals)-365, len(y_vals))
    


    curve_test = curve[-365:]

    
    
    #y_arima_exog_forecast = cl_test # this is the best case scenario forecast, the test data without trend or seasonality
    # (will give 0 RMSE)

    #minus_twenty = clickouts_wo_trend[0] - curve[0]
    #y_arima_exog_forecast = np.ones(365)*minus_twenty

    for i in range(len(y_arima_exog_forecast)): # go through the forecast
        y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + m*i_test_values[i])
        y_arima_exog_forecast_with_trend_and_seasonality.append(y_arima_exog_forecast_with_trend[i] + curve_test[i])

        #y_arima_exog_forecast_with_trend_and_seasonality.append(curve_test[i] + y_arima_exog_forecast[i])
       
    print("len(y_arima_exog_forecast_with_trend_and_seasonality)")
    print(len(y_arima_exog_forecast_with_trend_and_seasonality))

    print("len(y_arima_exog_forecast_with_trend_and_seasonality)")
    print(len(y_arima_exog_forecast_with_trend_and_seasonality))

    print("len(y_to_test)")
    print(len(y_to_test))

    #plt.plot( range(len(clickouts_wo_trend_or_seasonality)), clickouts_wo_trend_or_seasonality, color='blue')
    #plt.plot( range(len(cl_train), len(cl_train)+len(cl_test)), y_arima_exog_forecast, color='green')
    

    trend_line = []
    for i in range(len(y_vals)):
        trend_line.append(i*m+b)

    
    offset = 5000
    curve_with_trend = []
    print("len(curve)")
    print(len(curve))
    
    for i in range(len(curve)):
        curve_with_trend.append(curve[i] + m*i - offset)
        

    # good plots
    plt.plot( range(len(y_vals)), y_vals, color='green')
    plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='purple')
    #plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_arima_exog_forecast_with_trend_and_seasonality, color='blue')
    plt.plot( range(len(y_vals)), trend_line, color='orange')
    plt.plot( range(len(curve)), curve_with_trend, color='black')
    

    # also good plots
    #plt.plot( range(len(cl_train), len(cl_train)+len(cl_test)), y_arima_exog_forecast, color='red')
    #plt.plot( range(len(cl_train), len(cl_train)+len(cl_test)), cl_test, color='green')
    #plt.plot( range(len(cl_train)), cl_train, color='yellow')

    
    ### SARIMAX model compensating with added seasonality and trend ###

    # plotting data without seasonality or trend!
    """
    fig, axs = plt.subplots(3)
    fig.suptitle('Removing trend and seasonality')
    axs[0].plot( range(len(y_to_train)), y_to_train, color='green')
    axs[0].plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='green')
    axs[0].set_title('original flight data')
    axs[1].plot(range(len(clickouts_wo_trend)), clickouts_wo_trend, color='black')
    axs[1].set_title('data without trend')
    axs[2].plot(range(len(clickouts_wo_trend_or_seasonality)), clickouts_wo_trend_or_seasonality, color='purple')
    axs[2].set_title('data without seasonality')
    fig.tight_layout()

    for ax in axs.flat:
        ax.set(xlabel='days', ylabel='clicks')

    
    """


    
    # calculate root mean squared error
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 
    testScore = math.sqrt(mean_squared_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality))
    print('Test Score: %.2f RMSE' % (testScore))

    testScore = mean_absolute_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality)
    print('Test Score: %.2f MAE' % (testScore))

    testScore = mean_absolute_percentage_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality)
    print('Test Score: %.2f MAPE' % (testScore))
    
    plt.show()
    #pm.plot_acf(y_arima_exog_forecast)
if __name__ == "__main__":
    working()

