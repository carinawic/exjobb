from pandas import read_csv
import numpy as np
import pandas as pd
import datetime

def prepare_exchange_rates():
    # put nasdaq data in array
    df = read_csv('SEK_EUR_ORDERED.csv')
    #df = df.reindex(index=df.index[::-1]) # reverse
    #df.reset_index(drop=True, inplace=True)
    # 1079 wo weekends
    #df.to_csv("SEK_EUR_ORDERED.csv",index=False)
    
    #df.set_index('Datum', drop=True, append=False, inplace=False)

    #df.index = pd.DatetimeIndex(df.index)

    df = df.set_index('Datum')
    
    # to_datetime() method converts string
    # format to a DateTime object
    df.index = pd.to_datetime(df.index)

    new_date_range = pd.date_range(start="2016-01-01", end="2020-02-19", freq="D")
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')

    #idx = pd.date_range('2016-01-01','2020-02-19' )

    #df = df.reindex(idx)
    #df['Datum'].fillna(0, inplace=True)

    print(len(df))
    print(df.head(10))
    
    
if __name__ == "__main__":
    prepare_exchange_rates()