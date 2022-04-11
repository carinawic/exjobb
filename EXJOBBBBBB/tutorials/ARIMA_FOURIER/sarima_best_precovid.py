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
"""
note to self

- if something breaks regarding accuracy, remove the mega_offset!
- if accuracy is low, remove the scaling

"""

click_outs = []
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
salaryday = []
dayBeforeSalaryday = []
near24th = []
lufttemp = []
rain = []

OMSX30 = []
OMSXPI = []
SP500 = []
MSCI = []
VIX = []
OIL = []

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
    VIX = stgng_scaled

def create_MSCI():
    global MSCI
    df = read_csv('stonks/MSCI_ORDERED.csv')
    stgng = np.array(df['Öppen'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    MSCI = stgng_scaled

def create_SP500():
    global SP500
    df = read_csv('stonks/SP500_ORDERED.csv')
    stgng = np.array(df[' Open'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    SP500 = stgng_scaled

def create_OMSXPI():
    global OMSXPI
    df = read_csv('stonks/OMSXPI_ORDERED.csv')
    stgng = np.array(df['Stängn.kurs'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))
    OMSXPI = stgng_scaled

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
    OMSX30 = stgng_scaled


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


def createLufttemp():

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

def createDayBeforeSalaryDay():

    global dayBeforeSalaryday

    # df should be the one containing clickouts, but only for the good time range 2016-2020
    rng = pd.date_range('2016-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["IsSalaryDay"] = 0
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.day == 24, 1)

    """
    2016
    Dec 23 
    Sep 23
    Jun 23
    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-22'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-11-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-11-22'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-22'), 1)

    """
    2017
    Dec 22
    Nov 24
    Jun 22
    Maj 24
    Mar 24
    Feb 24
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-21'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-21'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-23'), 1)
    
    """ 
    2018
    Dec 21
    Nov 23
    Aug 24
    Mar 23
    Feb 23
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-20'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-11-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-11-22'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-08-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-08-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-03-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-03-22'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-02-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-02-22'), 1)

    """
    2019
    Dec 23
    Aug 23
    Maj 24
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-22'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-22'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-23'), 1)

    """
    2020
    Jan 24
    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2020-01-24'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2020-01-23'), 1)

    
    dayBeforeSalaryday = np.array(df['IsSalaryDay'].values)
    dayBeforeSalaryday = minmax_scale(dayBeforeSalaryday, feature_range=(0,500))

def createSalaryDay():

    global salaryday

    # df should be the one containing clickouts, but only for the good time range 2016-2020
    rng = pd.date_range('2016-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["IsSalaryDay"] = 0
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.day == 25, 1)

    """
    2016
    Dec 23 
    Sep 23
    Jun 23
    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-11-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-11-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-23'), 1)

    """
    2017
    Dec 22
    Nov 24
    Jun 22
    Maj 24
    Mar 24
    Feb 24
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-22'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-22'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-24'), 1)
    
    """ 
    2018
    Dec 21
    Nov 23
    Aug 24
    Mar 23
    Feb 23
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-21'), 1)

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
    Dec 23
    Aug 23
    Maj 24
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-24'), 1)

    """
    2020
    Jan 24
    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2020-01-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2020-01-24'), 1)

    
    salaryday = np.array(df['IsSalaryDay'].values)
    salaryday = minmax_scale(salaryday, feature_range=(0,500))

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
    
logestimate_eight_years = False

def working():
    global logestimate_eight_years 
    #df = pd.read_csv('kaggle_sales.csv')
    #df = df[(df['store'] == 1) & (df['item'] == 1)] # item 1 in store 1
    #df = df.set_index('date')
    #y = df['sales']

    df = pd.read_csv('Time.csv')
    #df = df.set_index('date')

    # switch this boolean when you change the precovid_startdate variable
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

    """
    degree for all data?
    2016-2020 = deg 16


    """
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
    #decompose_result.plot()
    #plt.plot( range(len(trend)), trend, color='purple')
    """
    better_trend = []
    for i,val in enumerate(trend):
        
        if np.isfinite(val):
            better_trend.append(val)
        else:
            better_trend.append(better_trend[i-1])
            
    """
    trend = [x for x in trend if not math.isnan(x)] # remove nan values

    print("len(better_trend)")
    #print(len(better_trend))
    print("len(trend)")
    print(len(trend))
    
    #m,b = np.polyfit(range(len(trend)),trend,1) # f(x) = m*i + b

    def func(x, a, tau, c):
        return a * np.exp(-x/tau) + c

    popt, pcov = curve_fit(func, np.array(range(len(trend))), trend)

    #coefficients = np.polyfit(np.log(range(1,len(trend))), trend[:-1], 1)

    #coefficients = np.polyfit(range(len(trend)), trend, 2)
    #p = np.poly1d(coefficients)
    #print("values")
    #print(p(0))
    #print(p(1))

    

    #print("HERE")
    #print(coefficients)

    logestimate = []
    #linestimate = []
    #for i in range(len(y_vals)):
        #logestimate.append(p(i))
    #    logestimate.append(func(np.array(range(len(y_vals))), *popt))
    
    logestimate = (func(np.array(range(len(y.values))), *popt))

    # for the FULL DATA, the logestimate might not even be a log!

    if logestimate_eight_years:
        logestimate = []
        not_log_coeffs = np.polyfit(range(len(trend)),trend,3)

        deg = 3
        for i in range(len(y_vals)):
            value = 0#not_log_coeffs[-1]
            for d in range(deg):
                value += i**(deg-d) * not_log_coeffs[d]
            logestimate.append(value)

    #for i in range(len(trend)):
    #    linestimate.append(m*i+b)
    
    # plot linear fit
    #plt.plot(range(len(trend)),m*range(len(trend))+b, color='purple')
    #plt.plot(range(len(trend)),m*np.log(range(len(trend)))+b, color='green')
    #plt.show()
    
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

    ### SARIMAX model making forecast without seasonality or trend ###

    cl_train = clickouts_wo_trend_or_seasonality[:len(clickouts_wo_trend_or_seasonality)-365]
    cl_test = clickouts_wo_trend_or_seasonality[len(clickouts_wo_trend_or_seasonality)-365:]

    #arima_exog_model = auto_arima(y=cl_train, seasonal=False, m=7) ##TODO set true
    #y_arima_exog_forecast = arima_exog_model.predict(n_periods=365)

    train_len = len(clicks_Search)-365

    clicks_Search_test = clicks_Search[-365:]
    clicks_Search_train = clicks_Search[:train_len]

    clicks_Active_test = clicks_Active[-365:]
    clicks_Active_train = clicks_Active[:train_len]

    clicks_Inactive_test = clicks_Inactive[-365:]
    clicks_Inactive_train = clicks_Inactive[:train_len]

    clicks_Extreme_test = clicks_Extreme[-365:]
    clicks_Extreme_train = clicks_Extreme[:train_len]

    near25th_test = near25th[-365:]
    near25th_train = near25th[:train_len]

    near24th_test = near24th[-365:]
    near24th_train = near24th[:train_len]

    open_exchange_SEK_EUR_test = open_exchange_SEK_EUR[-365:]
    open_exchange_SEK_EUR_train = open_exchange_SEK_EUR[:train_len]

    open_exchange_SEK_USD_test = open_exchange_SEK_USD[-365:]
    open_exchange_SEK_USD_train = open_exchange_SEK_USD[:train_len]

    salaryday_test = salaryday[-365:]
    salaryday_train = salaryday[:train_len]

    dayBeforeSalaryday_test = dayBeforeSalaryday[-365:]
    dayBeforeSalaryday_train = dayBeforeSalaryday[:train_len]

    #lufttemp = np.array(lufttemp, int)
    lufttemp_test = lufttemp[-365:]
    lufttemp_train = lufttemp[:train_len]
    
    rain_test = rain[-365:]
    rain_train = rain[:train_len]

    # all data!!!
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


    #combined_lufttemp_rain = np.logical_and(np.array(lufttemp) , np.array(rain))
    #combined_lufttemp_rain = 500*combined_lufttemp_rain # converts bool to int (0 or 500)
    #combined_lufttemp_rain_test = combined_lufttemp_rain[-365:]
    #combined_lufttemp_rain_train = combined_lufttemp_rain[:train_len]

    
    randlist = np.random.randint(500, size=len(clicks_Search))
    randlist_test = randlist[-365:]
    randlist_train = randlist[:train_len]



    """
    # Plot only graphs for euro and usd
    open_exchange_SEK_EUR_plotonly = [x*100 - 35000 for x in open_exchange_SEK_EUR]
    open_exchange_SEK_USD_plotonly = [x*100 - 35000 for x in open_exchange_SEK_USD]
    near25th_plotonly = [x*10 - 35000 for x in near25th]

    plt.plot( range(len(clickouts_wo_trend_or_seasonality)), clickouts_wo_trend_or_seasonality, color='green', label='flights')
    plt.plot( range(len(open_exchange_SEK_EUR)), open_exchange_SEK_EUR_plotonly, color='red', label='SEK_EUR')
    plt.plot( range(len(open_exchange_SEK_USD)), open_exchange_SEK_USD_plotonly, color='black', label='SEK_USD')
    #plt.plot( range(len(near25th)), near25th_plotonly, color='blue', label='salary')

    
    plt.xlabel('days')
    plt.ylabel('flights minus expected seasonality')
    plt.legend()
    plt.title('scaled features for display')

    plt.show()
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
    """
    only inactive and near25th => 2279.05 RMSE and good looking graph
    the correlation graph tells us that the factors clicks_Active_train, clicks_Inactive_test, clicks_Extreme_test have high p values and are not relevant. We can not find a good linear help from them. So we could leave them out. clicks_Search_train has very high correlation, and so does near25th_test. So we should only keep near25th_test. When evaluating features, it's almost cheating to include clicks_Search_train because it correlates so much. In order to investigate near25th_test's impact, should we remove the super highly correlating clicks_Search_test?

    
    """

    # all marketing
    #exog_train = np.column_stack((clicks_Search_train, clicks_Active_train, clicks_Inactive_train, clicks_Extreme_train))
    #exog_test = np.column_stack((clicks_Search_test,clicks_Active_test, clicks_Inactive_test, clicks_Extreme_test))
    
    #exog_train = np.column_stack((clicks_Search_train, clicks_Active_train))
    #exog_test = np.column_stack((clicks_Search_test,clicks_Active_test))
    
    #exog_train = np.column_stack((salaryday_train, dayBeforeSalaryday_train))
    #exog_test = np.column_stack((salaryday_test, dayBeforeSalaryday_test))
    
    #exog_train = open_exchange_SEK_USD_train
    #exog_test = open_exchange_SEK_USD_test
    
    """
    exog_train = np.column_stack((open_exchange_SEK_EUR_train, near25th_train, randlist_train, open_exchange_SEK_USD_train))
    exog_test = np.column_stack((open_exchange_SEK_EUR_test, near25th_test, randlist_test, open_exchange_SEK_USD_test))
    """
    # EXOG_HERE

    #exog_train = np.column_stack((salaryday_train, dayBeforeSalaryday_train))
    #exog_test = np.column_stack((salaryday_test, dayBeforeSalaryday_test))
    
    dict_settings = {}
    
    from tensorflow.keras.callbacks import EarlyStopping
    from itertools import product

    def permutation():
        """
        permutations_012 = list(product(range(2), repeat=4))
        permutations_12 = list(product(range(1,3), repeat=2))
        for onetwothree in permutations_012:
        for onetwo in permutations_12:
        
        try:
            AR = onetwo[0]
            I = onetwothree[0]
            MA = onetwo[1]
            SAR = onetwothree[1]
            SI = onetwothree[2]
            SMA = onetwothree[3]

            
        listToStr = ''.join([str(x) for x in [AR, I, MA, SAR, SI, SMA]])
        dict_settings[listToStr] = testScore
        dict_printme = dict(sorted(dict_settings.items(), key=lambda item: item[1]))
        print(dict_printme)

            """
    """

    permutations_012 = list(product(range(2), repeat=4))
    permutations_12 = list(product(range(1,3), repeat=2))
    for onetwothree in permutations_012:
        for onetwo in permutations_12:
        
            try:
                AR = onetwo[0]
                I = onetwothree[0]
                MA = onetwo[1]
                SAR = onetwothree[1]
                SI = onetwothree[2]
                SMA = onetwothree[3]

    """
    #model = SARIMAX(cl_train, order=(1,1,1), seasonal_order=(1,1,1,7), exog = exog_train, period = 7)
    #model = SARIMAX(cl_train, order=(AR,I,MA), seasonal_order=(SAR,SI,SMA,7), period = 7)
    #model = SARIMAX(cl_train, order=(1,1,1), seasonal_order=(1,1,1,7), period = 7, exog = exog_train)
    model = SARIMAX(cl_train, order=(1,1,1), seasonal_order=(1,1,1,7), period = 7)
    """
    212,1117 -> 2332.42
    211,1117 -> 2307.93 
    111,1117 -> 2327.42
    """
    callback=EarlyStopping(monitor="loss",patience=60) #30
    model_fit = model.fit(maxiter=200, callbacks=[callback]) # increase maxiter otherwise encounter convergence error
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
    """
    
    y_arima_exog_forecast = model_fit.forecast(steps=365)
    #y_arima_exog_forecast = model_fit.forecast(steps=365, exog=[[exog_test]])
    y_arima_exog_forecast_with_trend = []
    y_arima_exog_forecast_with_trend_and_seasonality = []
    #print(arima_exog_model.summary())
    i_test_values = range(len(y_vals)-365, len(y_vals))
    curve_test = curve[-365:]
    
    #y_arima_exog_forecast = cl_test # best case scenario forecast, the test data without trend or seasonality
    for i in range(len(y_arima_exog_forecast)): # go through the forecast

        #y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + m*i_test_values[i]) # linear
        y_arima_exog_forecast_with_trend.append(y_arima_exog_forecast[i] + logestimate[i_test_values[i]] - mega_offset)
        y_arima_exog_forecast_with_trend_and_seasonality.append(y_arima_exog_forecast_with_trend[i] + curve_test[i])
    # curve_with_trend is the curve with offset is added just for plotting nicely
    
    offset_for_plotting_only = 23000
    curve_with_trend = []
    #print(" b is ")
    #print( b)
    for i in range(len(curve)):
        #curve_with_trend.append(curve[i] + m*i - offset_for_plotting_only)
        curve_with_trend.append(curve[i] + logestimate[i] -offset_for_plotting_only)           

    # good plots
    plt.plot( range(len(y_vals)), y_vals, color='green', label = 'training data')
    plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_to_test, color='purple', label = 'testing data')
    plt.plot( range(len(y_to_train), len(y_to_train)+len(y_to_test)), y_arima_exog_forecast_with_trend_and_seasonality, color='blue', label = 'forecast')          
    #plt.plot( range(len(y_vals)), y_vals, color='green', label = 'flights')
    #plt.plot( range(len(curve_with_trend)), [x + 18000 for x in curve_with_trend], color='black', label = 'estimated yearly seasonality')
    #plt.plot( range(len(clickouts_wo_trend_or_seasonality)), clickouts_wo_trend_or_seasonality, color='blue', label = 'flights without trend or yearly seasonality')
    #plt.legend()
    #plt.show()           
    # #plt.plot( range(len(curve_with_trend)), curve_with_trend, color='black',  label = 'estimated yearly seasonality')
    #plt.plot( range(len(logestimate)), logestimate, color='blue', label = "estimated trend")
    #plt.plot( range(len(trend)), trend, color='red', label = "training data trend")
    #plt.legend()
    #plt.show()
    """
    from statsmodels.tsa.stattools import adfuller
    #result = adfuller(clickouts_wo_trend_or_seasonality, autolag='AIC')
    result = adfuller(clickouts_wo_trend_or_seasonality, autolag='AIC')
    print(f'ADF Statistic: {result[0]}')
    print(f'n_lags: {result[1]}')
    print(f'p-value: {result[1]}')
    for key, value in result[4].items():
        print('Critial Values:')
        print(f'   {key}, {value}')    
    """
    #PRINTING ACCURACY
    # calculate root mean squared error
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 
    testScore = math.sqrt(mean_squared_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality))
    print('Test Score: %.2f RMSE' % (testScore))
   
    testScore = mean_absolute_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality)
    print('Test Score: %.2f MAE' % (testScore))
    testScore = mean_absolute_percentage_error(y_to_test, y_arima_exog_forecast_with_trend_and_seasonality)
    print('Test Score: %.2f MAPE' % (testScore))
    
    """
    print("number of elements in rain is: ", np.count_nonzero(np.array(rain)))
    print("number of elements in lufttemp is: ", np.count_nonzero(np.array(lufttemp)))
    print("number of elements in combined_lufttemp_rain is: ", np.count_nonzero(np.array(combined_lufttemp_rain)))
    """
    # PLOTIING GRAPH
    #plt.xlabel('days')
    #plt.ylabel('flights')
    #plt.title('Forecast using SARIMAX')
    #plt.legend()
    #plt.show()
    #pm.plot_acf(y_arima_exog_forecast)
    
    #listToStr = ''.join([str(x) for x in [AR, I, MA, SAR, SI, SMA]])
    #dict_settings[listToStr] = testScore
    #dict_printme = dict(sorted(dict_settings.items(), key=lambda item: item[1]))
    #print(dict_printme)


if __name__ == "__main__":
    #prepare_data() # marketing
    #prepare_exchange_rates_EUR()
    #prepare_exchange_rates_USD()
    #createNear25th()
    #createSalaryDay()
    #createDayBeforeSalaryDay()
    #createNear24th()
    #createLufttemp()
    #createNeder
    # all data
    #create_OMSX30() 
    #create_OMSXPI()
    #create_SP500()
    #create_MSCI()
    #create_VIX()
    #create_OIL()
    working()