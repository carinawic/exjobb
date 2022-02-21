import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import warnings
import datetime
from pathlib import Path  
import csv
import datetime

def unpickle_data_as_Impr():

    df = pd.read_pickle('Media.p').impr
    return df

def unpickle_data_as_clicks():

    df = pd.read_pickle('Media.p').clicks
    return df

def save_media_impr_as_csv():
    df = unpickle_data_as_Impr()
    
    filepath = Path('media_imprs.csv')  
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath)

def save_media_clicks_as_csv():
    df = unpickle_data_as_clicks()
    
    filepath = Path('media_clicks.csv')    
    filepath.parent.mkdir(parents=True, exist_ok=True)  
    df.to_csv(filepath)


def sum_all_media_data_and_plot():
    df = pd.read_csv('media_clicks.csv')
    # df['Media_Bing_lowerfunnel_search_brand'] + df['Media_Bing_midfunnel_search_midbrand'] + df['Media_Bing_upperfunnel_search_nobrand'] + 
    df['sum_total'] = df.sum(axis=1)

    # Plot
    plt.figure(figsize=(6.8, 4.2))
    x = range(len(df['Media_Adwell_upperfunnel_native']))
    plt.plot(x, df['sum_total'])
    #plt.xticks(x, df['Media_Adwell_upperfunnel_native'].index.values)

    plt.xlabel('days after 2016-01-01')
    plt.ylabel('media data')

    plt.show()

def divide_media_data_and_plot():
    #divide into Search, Inactive and Active

    df = pd.read_csv('media_clicks.csv')

    #df = unpickle_data_as_Impr()

    # df['Media_Bing_lowerfunnel_search_brand'] + df['Media_Bing_midfunnel_search_midbrand'] + df['Media_Bing_upperfunnel_search_nobrand'] + 
    df['upperfunnel_total'] = df['Media_Adwell_upperfunnel_native']+ df['Media_DBM_lowerfunnel_display']+ df['Media_Google_lowerfunnel_display']

    # Plot
    plt.figure(figsize=(6.8, 4.2))
    x = range(len(df['Media_Adwell_upperfunnel_native']))
    plt.plot(x, df['upperfunnel_total'])
    #plt.xticks(x, df['Media_Adwell_upperfunnel_native'].index.values)

    plt.xlabel('days after 2016-01-01')
    plt.ylabel('media data')

    plt.show()

def plot_media_and_clickouts_together():

    # plot overlapping dates:
    # media csv = 2016-01-01 to 2021-12-01
    # time csv = 2012-12-31 to 2022-12-31
    # hence, overlapping time is
    # 2016-01-01 to 2021-12-01

    df1 = pd.read_csv('media_clicks.csv')
    df1['sum_total'] = df1.sum(axis=1)

    df2 = pd.read_csv('media_imprs.csv')
    df2['sum_total'] = df2.sum(axis=1)


    df3 = pd.read_csv('Time.csv')
    mask1 = (df1.iloc[:, 0] > '2016-01-01') & (df1.iloc[:, 0] <= '2021-12-01') # 2021-12-01
    mask2 = (df2.iloc[:, 0] > '2016-01-01') & (df2.iloc[:, 0] <= '2021-12-01') # 2020-02-19
    mask3 = (df3.iloc[:, 0] > '2016-01-01') & (df3.iloc[:, 0] <= '2021-12-01')
    
    print("df1")
    print(df1.loc[mask1]['sum_total'].shape)
    print("df2")
    print(df2.loc[mask2]['sum_total'].shape)
    print("df3")
    print(df3.loc[mask3]['clicks_out'].shape)


    fig1 = plt.figure()
    
    ax1 = fig1.add_subplot(311)
    ax2 = fig1.add_subplot(312)
    ax3 = fig1.add_subplot(313)

    default_x_ticks = range(2161)

    ax1.plot(default_x_ticks,df1.loc[mask1]['sum_total'])
    ax2.plot(default_x_ticks,df2.loc[mask2]['sum_total'])
    ax3.plot(default_x_ticks,df3.loc[mask3]['clicks_out'])

    #ax1.title.set_text('First Plot')
    #ax2.title.set_text('Second Plot')
    #ax3.title.set_text('Third Plot')


    plt.show()

    #ind = datetime.datetime.strptime("2021-11-24", format)  .between(startdate, enddate)].tolist()
    #print(df.iloc[ind])
    #df_between_dates = df[df['date']==startdate] #.between(startdate, enddate)]

    #print(df['date'])
    #print(df_between_dates.shape)

def get_clickouts_from_csv():

    
    df = pd.read_csv('Time.csv')

    # Plot
    plt.figure(figsize=(6.8, 4.2))
    x = range(len(df['week']))
    plt.plot(x, df['clicks_out'])

    plt.xlabel('days after 2012-12-31')
    plt.ylabel('clickouts')

    plt.show()

if __name__ == "__main__":
    plot_media_and_clickouts_together()
    