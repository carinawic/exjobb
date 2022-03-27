from pandas import read_csv
import numpy as np
import pandas as pd
import datetime

def create_exchange_rates():
    df = read_csv('SEK_EUR_ORDERED.csv')
    #df = df.reindex(index=df.index[::-1]) # reverse
    df = df.set_index('Datum')
    df.index = pd.to_datetime(df.index)
    new_date_range = pd.date_range(start="2016-01-01", end="2020-02-19", freq="D")
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')
    df.to_csv("SEK_EUR_FILLED.csv")
    
def prepare_exchange_rates():
    df = read_csv('SEK_EUR_FILLED.csv')
    exchange_SEK_EUR = np.array(df['Öppen'].values)
    print(exchange_SEK_EUR)
    print(len(exchange_SEK_EUR))
    
if __name__ == "__main__":
    prepare_exchange_rates()