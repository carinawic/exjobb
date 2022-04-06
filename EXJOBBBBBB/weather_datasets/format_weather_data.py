from time import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

lufttemp = []
rain = []

def format_lufttemp():

    global lufttemp

    lufttemp_corrected = pd.read_csv("Observatoriekullen\lufttemperatur_corrected.csv",sep=';',parse_dates=['Representativt dygn'])
    lufttemp_latest = pd.read_csv("Observatoriekullen\lufttemperatur_latest.csv",sep=';',parse_dates=['Representativt dygn'])
    lufttemp_full = pd.concat([lufttemp_corrected, lufttemp_latest], ignore_index=True)

    lufttemp_full = lufttemp_full.drop(['Från Datum Tid (UTC)','Till Datum Tid (UTC)','Kvalitet'], axis=1)

    #print(lufttemp_full)
    
    #lufttemp_full = lufttemp_full.set_index('Representativt dygn')
    lufttemp_full.index = pd.to_datetime(lufttemp_full['Representativt dygn'])
    new_date_range = pd.date_range(start="2016-01-01", end="2020-02-19", freq="D")
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

    """
    from numpy import polyfit
    # creating the 8-degree curve polnomial curve
    X = [i%365 for i in range(0, len(y))]
    y_vals = y
    degree = 8 # 8 looked reasonable
    coef = polyfit(X, y_vals, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve2 = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve2.append(value)
    """

    time = np.array(range(len(y)))
    sinwave = np.sin(2 * np.pi * time/365 - np.deg2rad(110)) * 10 + 9.4
    
    deviation_from_sine = y - sinwave

    deviation_consecutive = []
    month_list = []
    
    for i in lufttemp_full['Representativt dygn'].values:
        month_list.append(i.astype('datetime64[M]').astype(int) % 12 + 1)

    counter = 0

    for devpoint in deviation_from_sine:

        at_least_x_degrees_under_expected = 9 # we are below x degrees difference from expected temp
        days_in_row_threshold = 1
        

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
                    
                    
    
    #remove_values_during_season(deviation_consecutive, 5,10)
    lufttemp = deviation_consecutive

    #print(lufttemp)

    # get x value of each new month for plotting vertical lines
    vertical_line_here = []
    lastmonth = 0
    for i,month_num in enumerate(month_list):
        
        if month_num != lastmonth:
            if (month_num == 5 or month_num == 10):
                vertical_line_here.append(i)
            lastmonth = month_num
        

    
    plt.plot(x, y, color='green', label='temperatures')
    plt.plot(range(len(deviation_from_sine)), deviation_from_sine, color='blue', label='deviation from sin estimate')
    plt.plot(range(len(sinwave)), sinwave, color='black', label='sin estimate')
    plt.plot(range(len(deviation_consecutive)), deviation_consecutive, color='orange', label='< 0 deg for 3 consecutive days')

    for xc in vertical_line_here:
        plt.axvline(x=xc)

    plt.xlabel('days')
    plt.ylabel('degrees celcius')
    plt.legend()
   
    plt.show()
    
def format_neder():

    global rain

    neder_corrected = pd.read_csv("Observatoriekullen\\nederbördsmängd_corrected.csv",sep=';')
    neder_latest = pd.read_csv("Observatoriekullen\\nederbördsmängd_latest.csv",sep=';')
    neder_full = pd.concat([neder_corrected, neder_latest], ignore_index=True)

    neder_full = neder_full.drop(['Från Datum Tid (UTC)','Till Datum Tid (UTC)','Kvalitet'], axis=1)

    print(neder_full)
    
    neder_full_values = neder_full['Nederbördsmängd'].values
    values_in_list = 0
    regn_consecutive = []
    for regn_magnitude in neder_full_values:

        min_rain = 6
        days_in_row_threshold = 2
        
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

    rain = regn_consecutive
    plt.plot(range(len(neder_full_values)), neder_full_values, color='blue', label='rain magnitude')
    plt.plot(range(len(regn_consecutive)), regn_consecutive, color='red', label='rain consecutive')
    #plt.axhline(y = 1, color = 'r', linestyle = '-')
    #plt.axhline(y = 2, color = 'r', linestyle = '-')
    #plt.axhline(y = 3, color = 'r', linestyle = '-')
    #plt.axhline(y = 6, color = 'r', linestyle = '-')
    #plt.axhline(y = 10, color = 'r', linestyle = '-')
    #plt.axhline(y = 15, color = 'r', linestyle = '-')
    plt.legend()
    plt.xlabel('days')
    plt.ylabel('rain magnitude')
    plt.show()

if __name__ == "__main__":
    format_neder()