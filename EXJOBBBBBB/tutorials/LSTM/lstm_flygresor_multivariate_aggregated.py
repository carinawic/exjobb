"""


Test Score: 2279.96 RMSE - Test Score: 2243.47 RMSE

We preprocess the input data to be in the same range as one another. Scaling (0,1) which is common in ml works poorly in SARIMAX. An intuitive decision was made to scale (0,500) due to the ranges of stocks and weather input data.

near24th

scale (0,500) gave 
2240.48 ; 1707.78 ; 0.07

scale(0,2) gave 
2240.56 ; 1707.97 ; 0.07

scale (0,1) gave 
test score: 2679.66 ; 2006.33 ; 0.08


"""

import itertools
from turtle import width
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, minmax_scale 
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error
import datetime
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping

from keras.models import load_model
import math
import matplotlib.pyplot as plt
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
"""
note to self

- if something breaks regarding accuracy, remove the mega_offset!
- if accuracy is low, remove the scaling

for deciding on parameters, run 10 times and take the median
use numpy to calculate the mean and standard deviation
save rmse in a list and plot it

read about loss and f1 score also
f1 score for ARIMA?

"""

week = []
clicks_media = []
impr_media = []

clicks_Search = []
clicks_Inactive = []
clicks_Active = []
clicks_Extreme = []

open_exchange_SEK_EUR = []
open_exchange_SEK_USD = []
open_nasdaq = []
near25th = []
near25thcurve = []
salaryday = []
near24th = []
lufttemp = []
lufttemp_8years = []
rain = []
regn_8years = []

OMSX30 = []
OMSXPI = []
SP500 = []
MSCI = []
VIX = []
OIL = []

logestimate_eight_years = False

def create_OIL():
    global OIL
    df = read_csv('stonks/Oil_WTI_ORDERED.csv')
    stgng = np.array(df['Open'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    OIL = stgng_scaled

def create_VIX():
    global VIX
    df = read_csv('stonks/VIX_ORDERED.csv')
    stgng = np.array(df[' Open'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    VIX = stgng_scaled #[-1511:]

def create_MSCI():
    global MSCI
    df = read_csv('stonks/MSCI_ORDERED.csv')
    stgng = np.array(df['Öppen'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    MSCI = stgng_scaled # [-1511:]

def create_SP500():
    global SP500
    df = read_csv('stonks/SP500_ORDERED.csv')
    stgng = np.array(df[' Open'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    SP500 = stgng_scaled#[-1511:]

def create_OMSXPI():
    global OMSXPI
    df = read_csv('stonks/OMSXPI_ORDERED.csv')
    stgng = np.array(df['Stängn.kurs'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    OMSXPI = stgng_scaled#[-1511:]

def create_OMSX30():
    global OMSX30
    df = read_csv('stonks/OMSX30_ORDERED.csv')
    stgng = np.array(df['Stängn.kurs'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    #print(stgng)
    #print(len(stgng))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    OMSX30 = stgng_scaled

    print("len(stgng_scaled)")
    print(len(stgng_scaled))
    OMSX30 = stgng_scaled # [-1511:] # just add [-1511:] to get the 4 year data
    

def create_neder_8years():

    global regn_8years

    neder_full = pd.read_csv("Observatoriekullen_alldata\\neder.csv",sep=';')
    
    neder_full = neder_full.drop(['Från Datum Tid (UTC)','Till Datum Tid (UTC)','Kvalitet'], axis=1)

    #print(neder_full)
    
    neder_full_values = neder_full['Nederbördsmängd'].values
    values_in_list = 0
    regn_consecutive = []
    for regn_magnitude in neder_full_values:

        min_rain = 1
        days_in_row_threshold = 10
        
        if regn_magnitude > min_rain:
            counter += 1
            if counter >= days_in_row_threshold:
                regn_consecutive.append(10)
                values_in_list = values_in_list + 1
                continue
        else: 
            counter = 0
        regn_consecutive.append(0)

    print(values_in_list, " is HERE ")

    regn_8years = regn_consecutive

def createNeder():

    global rain

    neder_full = pd.read_csv("Observatoriekullen\\nederbördsmängd_corrected.csv",sep=';')

    neder_full = neder_full.drop(['Från Datum Tid (UTC)','Till Datum Tid (UTC)','Kvalitet'], axis=1)

    
    neder_full_values = neder_full['Nederbördsmängd'].values

    regn_consecutive = []
    for regn_magnitude in neder_full_values:

        min_rain = 2 # we are below x degrees difference from expected temp
        days_in_row_threshold = 2
        
        if regn_magnitude > min_rain:
            counter += 1
            if counter >= days_in_row_threshold:
                regn_consecutive.append(500)
                continue
        else: 
            counter = 0
        regn_consecutive.append(0)

    rain = regn_consecutive
    """
    plt.plot(range(len(neder_full_values)), neder_full_values, color='blue', label='rain magnitude')
    plt.plot(range(len(regn_consecutive)), regn_consecutive, color='red', label='rain consecutive')
    plt.legend()
    plt.xlabel('days')
    plt.ylabel('rain magnitude')
    plt.show()
    """

def create_lufttemp_8years():

    global lufttemp_8years

    lufttemp_full = pd.read_csv("Observatoriekullen_alldata\\lufttemp.csv",sep=';',parse_dates=['Representativt dygn'])
    #lufttemp_latest = pd.read_csv("lufttemperatur_latest.csv",sep=';',parse_dates=['Representativt dygn'])
    #lufttemp_full = pd.concat([lufttemp_corrected, lufttemp_latest], ignore_index=True)

    lufttemp_full = lufttemp_full.drop(['Från Datum Tid (UTC)','Till Datum Tid (UTC)','Kvalitet'], axis=1)

    #print(lufttemp_full)
    
    #lufttemp_full = lufttemp_full.set_index('Representativt dygn')
    lufttemp_full.index = pd.to_datetime(lufttemp_full['Representativt dygn'])
    new_date_range = pd.date_range(start="2013-01-01", end="2020-02-19", freq="D")
    lufttemp_full = lufttemp_full.reindex(new_date_range, fill_value=None)
    lufttemp_full = lufttemp_full.fillna(method='ffill')

    #print(lufttemp_full)

    #lufttemp_full['mean_month'] = lufttemp_full.resample('M', on='Från Datum Tid (UTC)').mean()
    lufttemp_full['MonthlyAverage'] = (lufttemp_full.groupby(lufttemp_full['Representativt dygn'].dt.to_period('M'))['Lufttemperatur'].transform('mean'))

    lufttemp_full['Dev_from_avg'] = lufttemp_full['Lufttemperatur'] - lufttemp_full['MonthlyAverage']

    #g = lufttemp_full.groupby(pd.Grouper(key='Representativt dygn', freq='M'))
    # groups to a list of dataframes with list comprehension
    
    x = range(len(lufttemp_full['Lufttemperatur'].values))
    
    y = lufttemp_full['Lufttemperatur'].values

    time = np.array(range(len(y)))
    sinwave = np.sin(2 * np.pi * time/365 - np.deg2rad(110)) * 10 + 9.4
    
    deviation_from_sine = y - sinwave

    deviation_consecutive = []
    month_list = []
    
    for i in lufttemp_full['Representativt dygn'].values:
        month_list.append(i.astype('datetime64[M]').astype(int) % 12 + 1)

    counter = 0

    for devpoint in deviation_from_sine:

        # we are below x degrees difference from expected temp
        at_least_x_degrees_under_expected = 5
        days_in_row_threshold = 8
        

        if devpoint > at_least_x_degrees_under_expected:
            counter += 1
            if counter >= days_in_row_threshold:
                deviation_consecutive.append(10)
                continue
        else: 
            counter = 0
        
        deviation_consecutive.append(0)
        

    def remove_values_during_season(list_to_be_filtered, endmonth=9,startmonth=5):
        # remove value if not summer
        
        if endmonth > startmonth:

            for i in range(len(list_to_be_filtered)):
                if(month_list[i] <= endmonth) and (month_list[i] >= startmonth):
                    list_to_be_filtered[i] = 0
        else:
            for i in range(len(list_to_be_filtered)):
                if((month_list[i] <= endmonth) or (month_list[i] >= startmonth)):
                   list_to_be_filtered[i] = 0
                    
                    
    
    lufttemp_8years = deviation_consecutive
    lufttemp_8years = minmax_scale(lufttemp_8years, feature_range=(0,500))
    #print("len(lufttemp_8years)")
    #print(len(lufttemp_8years))

def createLufttemp_4years():

    global lufttemp

    lufttemp_corrected = pd.read_csv("Observatoriekullen\lufttemperatur_corrected.csv",sep=';',parse_dates=['Representativt dygn'])
    lufttemp_latest = pd.read_csv("Observatoriekullen\lufttemperatur_latest.csv",sep=';',parse_dates=['Representativt dygn'])
    lufttemp_full = pd.concat([lufttemp_corrected, lufttemp_latest], ignore_index=True)

    lufttemp_full = lufttemp_full.drop(['Från Datum Tid (UTC)','Till Datum Tid (UTC)','Kvalitet'], axis=1)

    lufttemp_full.index = pd.to_datetime(lufttemp_full['Representativt dygn'])
    new_date_range = pd.date_range(start="2016-01-01", end="2020-02-19", freq="D")
    lufttemp_full = lufttemp_full.reindex(new_date_range, fill_value=None)
    lufttemp_full = lufttemp_full.fillna(method='ffill')

    lufttemp_full['MonthlyAverage'] = (lufttemp_full.groupby(lufttemp_full['Representativt dygn'].dt.to_period('M'))['Lufttemperatur'].transform('mean'))

    lufttemp_full['Dev_from_avg'] = lufttemp_full['Lufttemperatur'] - lufttemp_full['MonthlyAverage']

    
    y = lufttemp_full['Lufttemperatur'].values

    time = np.array(range(len(y)))
    sinwave = np.sin(2 * np.pi * time/365 - np.deg2rad(110)) * 10 + 9.4
    
    deviation_from_sine = y - sinwave

    deviation_consecutive = []
    month_list = []
    
    for i in lufttemp_full['Representativt dygn'].values:
        month_list.append(i.astype('datetime64[M]').astype(int) % 12 + 1)

    counter = 0

    for devpoint in deviation_from_sine:

        at_least_x_degrees_under_expected = 1 # we are below x degrees difference from expected temp
        days_in_row_threshold = 1
        

        if devpoint < at_least_x_degrees_under_expected:
            counter += 1
            if counter >= days_in_row_threshold:
                deviation_consecutive.append(10)
                continue
        else: 
            counter = 0
        
        deviation_consecutive.append(0)
        

    def remove_values_during_season(list_to_be_filtered, endmonth=9,startmonth=5):
        # remove value if not summer
        
        if endmonth > startmonth:

            for i in range(len(list_to_be_filtered)):
                if(month_list[i] <= endmonth) and (month_list[i] >= startmonth):
                    list_to_be_filtered[i] = 0
        else:
            for i in range(len(list_to_be_filtered)):
                if((month_list[i] <= endmonth) or (month_list[i] >= startmonth)):
                   list_to_be_filtered[i] = 0
                    
                    
    
    #remove_values_during_season(deviation_consecutive, 5,10)
    
    lufttemp = deviation_consecutive
    lufttemp = minmax_scale(lufttemp, feature_range=(0,500))
    print("len(lufttemp)")
    print(len(lufttemp))


def createSalaryDay():

    global salaryday

    # df should be the one containing clickouts, but only for the good time range 2016-2020
    rng = pd.date_range('2016-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["IsSalaryDay"] = 0
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.day == 25, 1)



    """
    2013
    Dec 24
    Aug 23
    May 24
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2013-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2013-12-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2013-08-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2013-08-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2013-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2013-05-24'), 1)
    
    """
    2014
    Dec 24
    Oct 24
    May 23
    Jan 24

    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-12-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-10-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-10-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-05-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-01-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2014-01-24'), 1)
    
    """
    2015
    Dec 24
    Oct 23
    Jul 24
    apr 24
    Jan 23
    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-12-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-10-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-10-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-07-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-07-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-04-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-04-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-01-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2015-01-23'), 1)
    
    """
    2016
    Dec 24
    Sep 23
    Jun 24
    Mar 24
    """
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-09-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-09-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-03-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-03-24'), 1)
    
    """
    2017
    Dec 22
    Nov 24
    Jun 23
    May 24
    Mar 24
    Feb 24

    """
     
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-22'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-24'), 1)
    
    """
    2018
    Dec 24
    Nov 23
    Aug 24
    Mar 23
    Feb 23

    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-11-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-11-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-08-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-08-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-03-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-03-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-02-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-02-23'), 1)

    """
    2019
    Dec 24
    Aug 23
    May 24

    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-24'), 1)
    
    """
    2020
    Jan 24
    """
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-01-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-01-24'), 1)

    salaryday = np.array(df['IsSalaryDay'].values)
    salaryday = minmax_scale(salaryday, feature_range=(0,500))


def createNear25thcurve():

    global near25thcurve
    # create an array of 5 dates starting at '2015-02-24', one per day
    rng = pd.date_range('2013-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 

    """
    df["Is25th"] = 1
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 25, 10)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 26, 7)
    #df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 27, 6)
    #df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 28, 3)
    #df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 30, 2)
    #df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 31, 1)
    print(df)
    """

    df["Is25th"] = 0
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 1, 18177.72972972973)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 2, 18918.945945945947)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 3, 18970.391891891893)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 4, 19051.756756756757)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 5, 18705.283783783783)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 6, 18989.18918918919)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 7, 18781.108108108107)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 8, 18638.95945945946)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 9, 18710.824324324323)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 10, 18897.824324324323)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 11, 18844.81081081081)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 12, 18630.54054054054)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 13, 18413.08108108108)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 14, 18175.932432432433)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 15, 18202.027027027027)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 16, 18183.202702702703)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 17, 18344.85135135135)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 18, 18269.716216216217)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 19, 17904.905405405407)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 20, 17723.424657534248)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 21, 17601.972602739726)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 22, 17658.465753424658)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 23, 17581.424657534248)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 24, 17642.58904109589)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 25, 18965.72602739726)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 26, 18651.479452054795)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 27, 18639.424657534248)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 28, 18466.13698630137)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 30, 17668.268656716416)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 31, 16779.837209302324)
    print(df)

    near25thcurve = np.array(df['Is25th'].values)
    near25thcurve = minmax_scale(near25thcurve, feature_range=(0,500))

    print(near25thcurve) # copied from statistics.py

    #print(len(near25thcurve))


def createNear25th():

    global near25th
    # create an array of 5 dates starting at '2015-02-24', one per day
    rng = pd.date_range('2013-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["Is25th"] = 0
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 25, 1)
    print(df)
    
    near25th = np.array(df['Is25th'].values)
    near25th = minmax_scale(near25th, feature_range=(0,500))

def createNear24th():

    global near24th
    # create an array of 5 dates starting at '2015-02-24', one per day
    rng = pd.date_range('2013-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["Is25th"] = 0
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 24, 1)
    print(df)
    
    near24th = np.array(df['Is25th'].values)
    near24th = minmax_scale(near24th, feature_range=(0,500))

def prepare_exchange_rates_USD():
    global open_exchange_SEK_USD

    df = read_csv('SEK_USD_FILLED.csv')
    exchange_SEK_USD = np.array(df['Open'].values)

    
    exchange_SEK_USD_rounded = []

    for i in exchange_SEK_USD:
        exchange_SEK_USD_rounded.append(int(float(i*10000)))

    open_exchange_SEK_USD = exchange_SEK_USD_rounded

    #scaling?
    open_exchange_SEK_USD = minmax_scale(open_exchange_SEK_USD, feature_range=(0,500))


    

def prepare_exchange_rates_EUR():
    global open_exchange_SEK_EUR

    df = read_csv('SEK_EUR_FILLED.csv')
    exchange_SEK_EUR = np.array(df['Öppen'].values)
    #print(exchange_SEK_EUR)
    #print(len(exchange_SEK_EUR))  

    
    exchange_SEK_EUR_rounded = []

    for i in exchange_SEK_EUR:
        exchange_SEK_EUR_rounded.append(int(float(i.replace(",", "."))*10000))

    open_exchange_SEK_EUR = exchange_SEK_EUR_rounded

    #scaling?
    open_exchange_SEK_EUR = minmax_scale(open_exchange_SEK_EUR, feature_range=(0,500))


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
    
    # mask between certain dates DURING COVID
    mask_clickouts = (df_complete.iloc[:, 0] >= precovid_startdate) & (df_complete.iloc[:, 0] <= precovid_enddate)
    mask_media_clicks = (media_clicks_df.iloc[:, 0] >= precovid_startdate) & (media_clicks_df.iloc[:, 0] <= precovid_enddate)
    #mask_media_imprs = (media_imprs_df.iloc[:, 0] > precovid_startdate) & (media_imprs_df.iloc[:, 0] <= precovid_enddate)
 
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

    print("len(clicks_Active)")
    print(len(clicks_Active))
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


def working():

    df = pd.read_csv('Time.csv')

    precovid_startdate = '2013-01-01'
    #precovid_startdate = '2016-01-01'
    precovid_enddate = '2020-02-19'

    mask_clickouts = (df.iloc[:, 0] >= precovid_startdate) & (df.iloc[:, 0] <= precovid_enddate)

    y = df['clicks_out'][mask_clickouts]

    #renamed = df.rename(columns={'clicks_out': 'decomposition of data'})
    #y = renamed['decomposition of data'][mask_clickouts]
    
    y_to_train = y.iloc[:(len(y)-365)]
    y_to_test = y.iloc[(len(y)-365):] # last year for testing
    y_vals = y.values
    
    print("len(y_to_test)")
    print(len(y_to_test))

    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset
    #dataframe = pd.read_csv('date_and_clicks.csv', usecols=[1], engine='python')
    #dataset = dataframe.values
    #dataset = dataset.astype('float32')

    # normalize the dataset
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #dataset = scaler.fit_transform(dataset)
    # split into train and test sets



    """
    HIGHSCORE:
    
    32-32
    Train Score: 1595.36 RMSE
    Test Score: 1872.24 RMSE

    """
    # define input sequence
    #in_seq1 = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])
    #in_seq2 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    #in_seq3 = np.array([15, 25, 35, 45, 55, 65, 75, 85, 95])
    #out_seq = np.array([in_seq1[i]+in_seq2[i] for i in range(len(in_seq1))])

    
    train_len = len(clicks_Search)-365

    clicks_Search_test = clicks_Search[-365:]
    clicks_Search_train = clicks_Search[:train_len]

    clicks_Active_test = clicks_Active[-365:]
    clicks_Active_train = clicks_Active[:train_len]

    clicks_Inactive_test = clicks_Inactive[-365:]
    clicks_Inactive_train = clicks_Inactive[:train_len]

    clicks_Extreme_test = clicks_Extreme[-365:]
    clicks_Extreme_train = clicks_Extreme[:train_len]


    # all data!!!

    
    near25th_test = near25th[-365:]
    near25th_train = near25th[:len(near25th)-365]

    near24th_test = near24th[-365:]
    near24th_train = near24th[:len(near24th)-365]
    
    open_exchange_SEK_EUR_test = open_exchange_SEK_EUR[-365:]
    open_exchange_SEK_EUR_train = open_exchange_SEK_EUR[:len(open_exchange_SEK_EUR)-365]

    open_exchange_SEK_USD_test = open_exchange_SEK_USD[-365:]
    open_exchange_SEK_USD_train = open_exchange_SEK_USD[:len(open_exchange_SEK_USD)-365]
    
    #lufttemp = np.array(lufttemp, int)
    lufttemp_test = lufttemp[-365:]
    lufttemp_train = lufttemp[:len(lufttemp)-365]
    
    rain_test = rain[-365:]
    rain_train = rain[:len(rain)-365]

    print("salaryday")
    print(salaryday)
    
    salaryday_test = salaryday[-365:]
    salaryday_train = salaryday[:len(salaryday)-365]

    dayBeforeSalaryday = np.append(salaryday[1:],0)

    dayBeforeSalaryday_test = dayBeforeSalaryday[-365:]
    dayBeforeSalaryday_train = dayBeforeSalaryday[:len(dayBeforeSalaryday)-365]
    
    lufttemp_8years_test = lufttemp_8years[-365:]
    lufttemp_8years_train = lufttemp_8years[:len(lufttemp_8years)-365]

    regn_8years_test = regn_8years[-365:]
    regn_8years_train = regn_8years[:len(regn_8years)-365]

    OMSX30_test = OMSX30[-365:]
    OMSX30_train = OMSX30[:len(OMSX30)-365]
    
    OMSXPI_test = OMSXPI[-365:]
    OMSXPI_train = OMSXPI[:len(OMSXPI)-365]

    SP500_test = SP500[-365:]
    SP500_train = SP500[:len(SP500)-365]

    MSCI_test = MSCI[-365:]
    MSCI_train = MSCI[:len(MSCI)-365]

    VIX_test = VIX[-365:]
    VIX_train = VIX[:len(VIX)-365]

    OIL_test = OIL[-365:]
    OIL_train = OIL[:len(OIL)-365]

    near25thcurve_test = near25thcurve[-365:]
    near25thcurve_train = near25thcurve[:len(near25thcurve)-365]

    
    #in_seq1 = np.array([1,2,3,4,5,6,7])
    #in_seq2 = np.array([1,2,3,4,5,6,7])
    #in_seq3 = np.array([1,2,3,4,5,6,7])
    #in_seq4 = np.array([1,2,3,4,5,6,7])
    #out_seq = np.array([4,5,6,7,8,9,10])

    # remove the first item of each data because the [0-3] first clicks match with the [1-4] first clickouts
    """in_seq1 = in_seq1[1:]
    in_seq2 = in_seq2[1:]
    in_seq3 = in_seq3[1:]
    in_seq4 = in_seq4[1:]

    out_seq = out_seq[:-1]"""

    

    OMSX30_n = np.array(OMSX30)
    #dayBeforeSalaryday = np.array(dayBeforeSalaryday)
    
    out_seq_n = np.array(y_vals)

    # convert to [rows, columns] structure
    #in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    #in_seq1 = OMSX30_n.reshape((len(OMSX30_n), 1))
    #in_seq2 = in_seq2.reshape((len(dayBeforeSalaryday), 1))
    #in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    #in_seq4 = in_seq3.reshape((len(in_seq4), 1))

    out_seq = out_seq_n.reshape((len(out_seq_n), 1))


    #dataset_stacked = np.hstack((in_seq1, in_seq2, in_seq3, in_seq4, out_seq))
    #dataset_stacked = np.hstack((in_seq1, out_seq))
    
    look_back = 7
    n_features = 0

    #Xtrain, Xtest, ytrain, ytest = split_sequences(dataset_stacked, look_back, len(y_vals)-365)
    Xtrain, Xtest, ytrain, ytest = split_sequences(out_seq, look_back, len(y_vals)-365)
    
    first_dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]
    second_dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    dropoutdict = {}

    for firsts in first_dropout_list:
        for seconds in second_dropout_list:
            for iterrr in range(10):
                model = Sequential()
                model.add(LSTM(32, recurrent_dropout=firsts, activation='relu',return_sequences=True, input_shape=(look_back, n_features+1)))
                #model.add(LSTM(32, activation='relu', input_shape=(look_back, n_features+1)))
                model.add(LSTM(32, recurrent_dropout=seconds, activation='relu'))
                #
                #model.add(LSTM(16, activation='relu'))
                #model.add(Dropout(0.1))

                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mse')
                


                callback=EarlyStopping(monitor="val_loss", mode='min', patience=30, restore_best_weights=True)

                # fit model
                history = model.fit(Xtrain, ytrain, validation_split=0.5, epochs=200, batch_size=1, verbose=2,  callbacks=[callback]) # 

                #print(history.history)
                """
                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plt.show()
                # save model
                """
                

                trainPredict = model.predict(Xtrain)
                testPredict = model.predict(Xtest)

                
                trainPredict = np.array(trainPredict,dtype=float)
                ytrain = np.array(ytrain,dtype=float)
                testPredict = np.array(testPredict,dtype=float)
                ytest = np.array(ytest,dtype=float)
                
                # calculate root mean squared error
                # 22.93 RMSE means an error of about 23 passengers (in thousands) 
                #trainScore1 = math.sqrt(mean_squared_error(ytrain, trainPredict))
                #print('Train Score: %.2f RMSE' % (trainScore1))
                testScore1 = math.sqrt(mean_squared_error(ytest, testPredict))
                #print('Test Score: %.2f RMSE' % (testScore1))

                # calculate root mean squared error
                # 22.93 RMSE means an error of about 23 passengers (in thousands) 
                #trainScore2 = mean_absolute_error(ytrain, trainPredict)
                #print('Train Score: %.2f MAE' % (trainScore2))
                #testScore2 = mean_absolute_error(ytest, testPredict)
                #print('Test Score: %.2f MAE' % (testScore2))

                
                # calculate root mean squared error
                # 22.93 RMSE means an error of about 23 passengers (in thousands) 
                #trainScore3 = mean_absolute_percentage_error(ytrain, trainPredict)
                #print('Train Score: %.2f MAPE' % (trainScore3))
                #testScore3 = mean_absolute_percentage_error(ytest, testPredict)
                #print('Test Score: %.2f MAPE' % (testScore3))

                dropoutdict["["+str(firsts)+","+str(seconds)+"]"+"iteration:"+str(iterrr)] = str(testScore1)
                print(dropoutdict)

    """

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

    """
def working_preprocessed():

    global logestimate_eight_years 

    df = pd.read_csv('Time.csv')
    
    logestimate_eight_years = True
    precovid_startdate = '2013-01-01'
    #precovid_startdate = '2016-01-01'
    precovid_enddate = '2020-02-19'

    mask_clickouts = (df.iloc[:, 0] >= precovid_startdate) & (df.iloc[:, 0] <= precovid_enddate)

    y = df['clicks_out'][mask_clickouts]

    #renamed = df.rename(columns={'clicks_out': 'decomposition of data'})
    #y = renamed['decomposition of data'][mask_clickouts]
    
    y_to_train = y.iloc[:(len(y)-365)]
    y_to_test = y.iloc[(len(y)-365):] # last year for testing
    y_vals = y.values
    
    print("len(y_to_test)")
    print(len(y_to_test))

    # fix random seed for reproducibility
    np.random.seed(7)

    
    # creating the 8-degree curve polnomial curve
    # can you beat 2900
    X = [i%365 for i in range(0, len(y_to_train))]
    y_to_train
    degree = 50 # 16 in results for 2016-2020
    coef = polyfit(X, y_to_train, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()

    X_counter = [i%365 for i in range(0, len(y.values))]

    skottdatgar = 0
    for i in range(len(X_counter)):
        value = coef[-1]
        for d in range(degree):
            value += X_counter[i]**(degree-d) * coef[d]
        curve.append(value)
        if (i!=0 and i%(365*4)==0):
            curve.append(value)
            skottdatgar = skottdatgar+1
    
    curve = curve[:-skottdatgar]

    decompose_result = seasonal_decompose(y_to_train, model="additive", period=365)
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid

    
    trend = [x for x in trend if not math.isnan(x)] # remove nan values

    print("len(better_trend)")
    #print(len(better_trend))
    print("len(trend)")
    print(len(trend))
    
    #m,b = np.polyfit(range(len(trend)),trend,1) # f(x) = m*i + b

    def func(x, a, tau, c):
        return a * np.exp(-x/tau) + c

    popt, pcov = curve_fit(func, np.array(range(len(trend))), trend)

    logestimate = []
    #linestimate = []
    #for i in range(len(y_vals)):
        #logestimate.append(p(i))
    #    logestimate.append(func(np.array(range(len(y_vals))), *popt))
    
    logestimate = (func(np.array(range(len(y.values))), *popt))

    if logestimate_eight_years:
        logestimate = []
        not_log_coeffs = np.polyfit(range(len(trend)),trend,3)

        deg = 3
        for i in range(len(y_vals)):
            value = 0#not_log_coeffs[-1]
            for d in range(deg):
                value += i**(deg-d) * not_log_coeffs[d]
            logestimate.append(value)


    clickouts = np.array(y.values)

    
    print("len(clickouts)")
    print(len(clickouts))

    clickouts_wo_trend = []
    clickouts_wo_trend_or_seasonality = []
    
    seasonal = [x for x in seasonal if not x is(None)] # remove None values

    mega_offset = 0 # 40000 #
    # creating the datasets without seasonality or trend
    for i,real_val in enumerate(clickouts):
        
        #clickouts_wo_trend.append(real_val - (m*i)) # centered around b now!
        clickouts_wo_trend.append(real_val - logestimate[i]) # centered around b now!
        clickouts_wo_trend_or_seasonality.append(clickouts_wo_trend[i] - curve[i] + mega_offset)

    
    print("len(clickouts_wo_trend_or_seasonality)")
    print(len(clickouts_wo_trend_or_seasonality))

    cl_train = clickouts_wo_trend_or_seasonality[:len(clickouts_wo_trend_or_seasonality)-365]
    cl_test = clickouts_wo_trend_or_seasonality[len(clickouts_wo_trend_or_seasonality)-365:]


    train_len = len(clicks_Search)-365

    clicks_Search_test = clicks_Search[-365:]
    clicks_Search_train = clicks_Search[:train_len]

    clicks_Active_test = clicks_Active[-365:]
    clicks_Active_train = clicks_Active[:train_len]

    clicks_Inactive_test = clicks_Inactive[-365:]
    clicks_Inactive_train = clicks_Inactive[:train_len]

    clicks_Extreme_test = clicks_Extreme[-365:]
    clicks_Extreme_train = clicks_Extreme[:train_len]


    # all data!!!

    
    near25th_test = near25th[-365:]
    near25th_train = near25th[:len(near25th)-365]

    near24th_test = near24th[-365:]
    near24th_train = near24th[:len(near24th)-365]
    
    open_exchange_SEK_EUR_test = open_exchange_SEK_EUR[-365:]
    open_exchange_SEK_EUR_train = open_exchange_SEK_EUR[:len(open_exchange_SEK_EUR)-365]

    open_exchange_SEK_USD_test = open_exchange_SEK_USD[-365:]
    open_exchange_SEK_USD_train = open_exchange_SEK_USD[:len(open_exchange_SEK_USD)-365]
    
    #lufttemp = np.array(lufttemp, int)
    lufttemp_test = lufttemp[-365:]
    lufttemp_train = lufttemp[:len(lufttemp)-365]
    
    rain_test = rain[-365:]
    rain_train = rain[:len(rain)-365]

    print("salaryday")
    print(salaryday)
    
    salaryday_test = salaryday[-365:]
    salaryday_train = salaryday[:len(salaryday)-365]

    dayBeforeSalaryday = np.append(salaryday[1:],0)

    dayBeforeSalaryday_test = dayBeforeSalaryday[-365:]
    dayBeforeSalaryday_train = dayBeforeSalaryday[:len(dayBeforeSalaryday)-365]
    
    lufttemp_8years_test = lufttemp_8years[-365:]
    lufttemp_8years_train = lufttemp_8years[:len(lufttemp_8years)-365]

    regn_8years_test = regn_8years[-365:]
    regn_8years_train = regn_8years[:len(regn_8years)-365]

    OMSX30_test = OMSX30[-365:]
    OMSX30_train = OMSX30[:len(OMSX30)-365]
    
    OMSXPI_test = OMSXPI[-365:]
    OMSXPI_train = OMSXPI[:len(OMSXPI)-365]

    SP500_test = SP500[-365:]
    SP500_train = SP500[:len(SP500)-365]

    MSCI_test = MSCI[-365:]
    MSCI_train = MSCI[:len(MSCI)-365]

    VIX_test = VIX[-365:]
    VIX_train = VIX[:len(VIX)-365]

    OIL_test = OIL[-365:]
    OIL_train = OIL[:len(OIL)-365]

    near25thcurve_test = near25thcurve[-365:]
    near25thcurve_train = near25thcurve[:len(near25thcurve)-365]

    
    #in_seq1 = np.array([1,2,3,4,5,6,7])
    #in_seq2 = np.array([1,2,3,4,5,6,7])
    #in_seq3 = np.array([1,2,3,4,5,6,7])
    #in_seq4 = np.array([1,2,3,4,5,6,7])
    #out_seq = np.array([4,5,6,7,8,9,10])

    # remove the first item of each data because the [0-3] first clicks match with the [1-4] first clickouts
    """in_seq1 = in_seq1[1:]
    in_seq2 = in_seq2[1:]
    in_seq3 = in_seq3[1:]
    in_seq4 = in_seq4[1:]

    out_seq = out_seq[:-1]"""

    

    OMSX30_n = np.array(OMSX30)
    #dayBeforeSalaryday = np.array(dayBeforeSalaryday)
    
    out_seq_n = np.array(y_vals)

    # convert to [rows, columns] structure
    #in_seq1 = in_seq1.reshape((len(in_seq1), 1))
    #in_seq1 = OMSX30_n.reshape((len(OMSX30_n), 1))
    #in_seq2 = in_seq2.reshape((len(dayBeforeSalaryday), 1))
    #in_seq3 = in_seq3.reshape((len(in_seq3), 1))
    #in_seq4 = in_seq3.reshape((len(in_seq4), 1))

    out_seq = out_seq_n.reshape((len(out_seq_n), 1))


    #dataset_stacked = np.hstack((in_seq1, in_seq2, in_seq3, in_seq4, out_seq))
    #dataset_stacked = np.hstack((in_seq1, out_seq))
    
    look_back = 7
    n_features = 0

    #Xtrain, Xtest, ytrain, ytest = split_sequences(dataset_stacked, look_back, len(y_vals)-365)
    Xtrain, Xtest, ytrain, ytest = split_sequences(out_seq, look_back, len(y_vals)-365)
   
    model = Sequential()
    model.add(LSTM(32, activation='relu', return_sequences=True, input_shape=(look_back, n_features+1)))
    #model.add(LSTM(32, activation='relu', input_shape=(look_back, n_features+1)))
    model.add(LSTM(32,  activation='relu'))
    #
    #model.add(LSTM(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    

    """
    saved
    Train Score: 2786.01 RMSE
    Test Score: 3127.24 RMSE
    """

    #model = load_model('my_model.h5')

    callback=EarlyStopping(monitor="val_loss", mode='min', patience=30, restore_best_weights=True)


    # fit model
    history = model.fit(Xtrain, ytrain, validation_split=0.5, epochs=200, batch_size=1, verbose=2,  callbacks=[callback]) # 

    print(history.history)

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    # save model

    
    
    #model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
    #del model  # deletes the existing model
    
    # returns a compiled model
    # identical to the previous one

    #json_string = model.to_json()
    #import json
    #out_file = open("myfile.json", "w")
    #json.dump(json_string, out_file)

    trainPredict = model.predict(Xtrain)
    testPredict = model.predict(Xtest)

    #y_arima_exog_forecast = model_fit.forecast(steps=365)
    y_arima_exog_forecast = testPredict
    y_arima_exog_forecast_train = trainPredict

    y_arima_exog_forecast_with_trend = []
    y_arima_exog_forecast_with_trend_and_seasonality = []
    
    y_arima_exog_forecast_with_trend_train = []
    y_arima_exog_forecast_with_trend_and_seasonality_train = []
    #print(arima_exog_model.summary())
    i_test_values = range(len(y_vals)-365, len(y_vals))
    curve_test = curve[-365:]

    for i in range(len(y_arima_exog_forecast)): # go through the forecast

        #y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + m*i_test_values[i]) # linear
        y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + logestimate[i_test_values[i]] - mega_offset)
        y_arima_exog_forecast_with_trend_and_seasonality.append(y_arima_exog_forecast_with_trend[i] + curve_test[i])
    
    for i in range(len(y_arima_exog_forecast_train)): # go through the forecast

        #y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + m*i_test_values[i]) # linear
        y_arima_exog_forecast_with_trend_train.append(y_arima_exog_forecast_train[i] + logestimate[i] - mega_offset)
        y_arima_exog_forecast_with_trend_and_seasonality_train.append(y_arima_exog_forecast_with_trend_train[i] + curve[i])
    
    
    
    #trainPredict = trainPredict.reshape(-1, 1)
    #testPredict = testPredict.reshape(-1, 1)
    #trainPredict = scaler.inverse_transform(trainPredict)
    #testPredict = scaler.inverse_transform(testPredict)

    print("successfully inverse_transform")
    
    #trainPredict = scaler.inverse_transform(trainPredict)
    #testPredict = scaler.inverse_transform(testPredict)

    #print(history)

    # invert predictions in case we used minmaxscaler earlier
    #trainPredict = scaler.inverse_transform(trainPredict)
    #ytrain = scaler.inverse_transform([ytrain])
    #testPredict = scaler.inverse_transform(testPredict)
    #ytest = scaler.inverse_transform([ytest])

    trainPredict = np.array(y_arima_exog_forecast_train,dtype=float)
    ytrain = np.array(ytrain,dtype=float)
    testPredict = np.array(y_arima_exog_forecast,dtype=float)
    ytest = np.array(ytest,dtype=float)
    
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

def handle_model():
    import json
    model = load_model('my_model.h5')
    json_string = model.to_json()
    out_file = open("model_1833.json", "w")
    json.dump(json_string, out_file)
    
if __name__ == "__main__":
    #prepare_data()
    #create_OMSX30()
    #handle_model()
    working()
    #working_preprocessed()
