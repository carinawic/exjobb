from tkinter.ttk import Separator
from pandas import read_csv
import numpy as np
import pandas as pd


def prepare_OMSX30():
    df = read_csv('stonks\OMSX30.csv', sep=";")
    df = df.reindex(index=df.index[::-1]) # reverse
    df = df.set_index('Datum')
    df.index = pd.to_datetime(df.index)
    new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')
    df.drop(columns=['Högstakurs','Lägstakurs','Genomsnittspris','Tot.vol.','Oms',df.columns[-1]], axis=1, inplace=True)
    print(df.head(10))
    header = ["Stängn.kurs"]
    df.to_csv("stonks/OMSX30_ORDERED.csv",columns = header)

def create_OMSX30():
    df = read_csv('stonks/OMSX30_ORDERED.csv')
    stgng = np.array(df['Stängn.kurs'].values)
    print(stgng)
    print(len(stgng))

if __name__ == "__main__":
    create_OMSX30()