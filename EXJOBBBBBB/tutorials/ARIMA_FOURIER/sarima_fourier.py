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


    
    y_to_train = y.iloc[:(len(y)-365)]
    y_to_test = y.iloc[(len(y)-365):] # last year for testing

    #a = np.zeros(len(y))

    #for n in range(len(y)):
    #    a[n]=15*np.sin(np.deg2rad(n-90))+20


    #plt.plot( range(len(y_to_train)), y_to_train, color='green')
    #plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='blue')

    #plt.show()

    # prepare Fourier terms
    #exog = pd.DataFrame({'date': y.index})
    #exog = exog.set_index(pd.PeriodIndex(exog['date'], freq='D'))
    #exog['weekly'] = a
    #exog = exog.drop(columns=['date'])
    #exog_to_train = exog.iloc[:(len(y)-365)]
    #exog_to_test = exog.iloc[(len(y)-365):]

    # Fit model
    #arima_exog_model = auto_arima(y=y_to_train, exogenous=exog_to_train, seasonal=True, m=7)
    # Forecast
    #y_arima_exog_forecast = arima_exog_model.predict(n_periods=365, exogenous=exog_to_test)

    #print(arima_exog_model.summary())
   
    X = [i%365 for i in range(0, len(y.values))]
    y = y.values
    degree = 4
    coef = polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)
    # plot curve over original data
  
    
    #plt.plot( range(len(curve)), curve, color='black')

    #exog = pd.DataFrame({'date': y.index})
    #exog = exog.set_index(pd.PeriodIndex(exog['date'], freq='D'))
    #exog['passengers'] = y

    decompose_result = seasonal_decompose(y, model="additive", period=365)
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid



    #plt.plot( range(len(trend)), trend, color='purple')
    
    plt.plot( range(len(trend)), trend, color='black')
    #plt.plot( range(len(seasonal)), seasonal, color='green')
    #plt.plot( range(365,365*2),seasonal[:365], color='blue')

    m,b = np.polyfit(range(trend),trend,1)
    plt.plot(range(trend),m*range(trend)+b,'-', color='purple')
    

    #plt.plot( range(len(y_to_train)), y_to_train, color='green')
    #plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='blue')
    #plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_arima_exog_forecast, color="red")
    
    #pyplot.plot(curve, color='black', linewidth=3)

    #plt.plot( a, color="yellow")
    plt.show()
    #pm.plot_acf(y_arima_exog_forecast)
if __name__ == "__main__":
    working()

