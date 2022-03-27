from pandas import read_csv
import numpy as np
import pandas as pd

def prepare_NASDAQ():
    # put nasdaq data in array
    nasdaq_df = read_csv('NASDAQ.csv')
    

    nasdaq_df.index = pd.DatetimeIndex(s.index)

    idx = pd.date_range('01/01/2016', '19/02/2020')

    s = s.reindex(idx, fill_value=0)

    precovid_startdate = '01/01/2016'
    precovid_enddate = '19/02/2020'
    mask_nasdaq = (nasdaq_df.iloc[:, 0] > precovid_startdate) & (nasdaq_df.iloc[:, 0] <= precovid_enddate)

    open_nasdaq = np.array(nasdaq_df['Open'][mask_nasdaq].values)

    print(len(open_nasdaq))
    print(open_nasdaq)
    
if __name__ == "__main__":
    prepare_NASDAQ()