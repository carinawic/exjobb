import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def format_lufttemp():
    lufttemp_corrected = pd.read_csv("Observatoriekullen\lufttemperatur_corrected.csv",sep=';',parse_dates=['Representativt dygn'])
    lufttemp_latest = pd.read_csv("Observatoriekullen\lufttemperatur_latest.csv",sep=';',parse_dates=['Representativt dygn'])
    lufttemp_full = pd.concat([lufttemp_corrected, lufttemp_latest], ignore_index=True)

    lufttemp_full = lufttemp_full.drop(['Från Datum Tid (UTC)','Till Datum Tid (UTC)','Kvalitet'], axis=1)

    print(lufttemp_full)
    
    #lufttemp_full = lufttemp_full.set_index('Representativt dygn')
    lufttemp_full.index = pd.to_datetime(lufttemp_full['Representativt dygn'])
    new_date_range = pd.date_range(start="2016-01-01", end="2020-02-19", freq="D")
    lufttemp_full = lufttemp_full.reindex(new_date_range, fill_value=None)
    lufttemp_full = lufttemp_full.fillna(method='ffill')

    print(lufttemp_full)

    #lufttemp_full['mean_month'] = lufttemp_full.resample('M', on='Från Datum Tid (UTC)').mean()
    lufttemp_full['MonthlyAverage'] = (lufttemp_full.groupby(lufttemp_full['Representativt dygn'].dt.to_period('M'))['Lufttemperatur'].transform('mean'))

    lufttemp_full['Dev_from_avg'] = lufttemp_full['Lufttemperatur'] - lufttemp_full['MonthlyAverage']

    g = lufttemp_full.groupby(pd.Grouper(key='Representativt dygn', freq='M'))
    # groups to a list of dataframes with list comprehension
    dfs = [group for _,group in g] 

    fixed_temperatures_all_days = []
    step_temperatures_all_days = []

    # for each dataframe (each containing one month) except last month
    for i,month_group in enumerate(dfs[:-1]):
        
        current_month_avg_values = dfs[i]['MonthlyAverage'].values
        current_month_avg = current_month_avg_values[0]

        next_month_avg_values = dfs[i+1]['MonthlyAverage'].values
        next_month_avg = next_month_avg_values[0]

        delta = current_month_avg - next_month_avg / len(current_month_avg_values)

        # for each day in the current month
        for num,day in enumerate(current_month_avg_values):
            fixed_temperatures_all_days.append(current_month_avg + num*delta)
            step_temperatures_all_days.append(current_month_avg)

    #    print(fixed_temperatures_all_days)

    
    x = range(len(lufttemp_full['Lufttemperatur'].values))
    
    y = lufttemp_full['Lufttemperatur'].values

    from numpy import polyfit
    # creating the 8-degree curve polnomial curve
    X = [i%365 for i in range(0, len(y))]
    y_vals = y
    degree = 8 # 8 looked reasonable
    coef = polyfit(X, y_vals, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i]**(degree-d) * coef[d]
        curve.append(value)

    plt.plot(range(len(curve)), curve, color='red')
    plt.show()
    
    #plt.plot(range(len(fixed_temperatures_all_days)), fixed_temperatures_all_days, color='green')
    #plt.plot(x, y, color='black')
    
    

def format_neder():
    neder_corrected = pd.read_csv("Observatoriekullen\\nederbördsmängd_corrected.csv",sep=';')
    neder_latest = pd.read_csv("Observatoriekullen\\nederbördsmängd_latest.csv",sep=';')
    


if __name__ == "__main__":
    format_lufttemp()