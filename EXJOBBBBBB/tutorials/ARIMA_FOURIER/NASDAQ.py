from tkinter.ttk import Separator
from pandas import read_csv
import numpy as np
import pandas as pd

"""
note that the csv files are creates with a few extra dates at the top, just manually remove those!
"""
def prepare_OMSXPI():
    df = read_csv('stonks\OMSXPI.csv', sep=";", decimal=',')
    df = df.reindex(index=df.index[::-1]) # reverse
    df = df.set_index('Datum')
    df.index = pd.to_datetime(df.index)
    new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')
    df.drop(columns=['Högstakurs','Lägstakurs','Genomsnittspris','Tot.vol.','Oms',df.columns[-1]], axis=1, inplace=True)
    print(df.head(10))
    header = ["Stängn.kurs"]
    df.to_csv("stonks/OMSXPI_ORDERED.csv",columns = header)


def prepare_OMSX30():
    df = read_csv('stonks\OMSX30.csv', sep=";", decimal=',')
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

def prepare_SP500():
    df = read_csv('stonks\SP500.csv')
    #df[' Open']=df[' Open'].str.replace(',','')
    #df = df.reindex(index=df.index[::-1]) # reverse
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    new_date_range = pd.date_range(start="28/12/2012", end="02/19/2020", freq="D")
    #new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    print(df.head(10))
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')
    df.drop(columns=[' High',' Low',' Close',df.columns[-1]], axis=1, inplace=True)
    header = [" Open"]
    df.to_csv("stonks/SP500_ORDERED.csv",columns = header)

def prepare_MSCI():
    df = read_csv('stonks\MSCI.csv')
    df['Öppen']=df['Öppen'].str.replace('.','')
    df['Öppen']=df['Öppen'].str.replace(',','.')
    df = df.reindex(index=df.index[::-1]) # reverse
    df = df.set_index('Datum')
    df.index = pd.to_datetime(df.index)
    new_date_range = pd.date_range(start="28/12/2012", end="02/19/2020", freq="D")
    #new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    print(df.head(10))
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')
    df.drop(columns=["Senaste","Högst","Lägst","Vol.","+/- %",df.columns[-1]], axis=1, inplace=True)
    header = ["Öppen"]
    df.to_csv("stonks/MSCI_ORDERED.csv",columns = header)

#problematic
def prepare_CCI():
    df = read_csv('stonks\CCI.csv')
    #df = df.reindex(index=df.index[::-1])
    df = df.set_index('TIME')
    df.index = pd.to_datetime(df.index, format="%Y-%m")
    new_date_range = pd.date_range(start="2013-01", end="2020-02", freq="M")
    #new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    df.drop(columns=["LOCATION","INDICATOR","SUBJECT","MEASURE","FREQUENCY","Flag Codes",df.columns[-1]], axis=1, inplace=True)
    print(df.head(10))
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')
    print(df.head(10))
    df.to_csv("stonks/CCI_ORDERED.csv")


def prepare_VIX():
    df = read_csv('stonks\VIX.csv')
    #df[' Open']=df[' Open'].str.replace(',','')
    #df = df.reindex(index=df.index[::-1]) # reverse
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    new_date_range = pd.date_range(start="28/12/2012", end="02/19/2020", freq="D")
    #new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    print(df.head(10))
    df = df.reindex(new_date_range, fill_value=None)
    df = df.fillna(method='ffill')
    df.drop(columns=[' High',' Low',' Close',df.columns[-1]], axis=1, inplace=True)
    header = [" Open"]
    df.to_csv("stonks/VIX_ORDERED.csv",columns = header)


def prepare_Oil_WTI():
    df = read_csv('stonks\Oil_WTI_two.csv')
    
    df.drop(columns=['Close','High','Low','Volume','Dunno'], axis=1, inplace=True)

    df = df.reindex(index=df.index[::-1]) # reverse
    print(df.head(10))
    #df[' Open']=df[' Open'].str.replace(',','')
    #df = df.reindex(index=df.index[::-1]) # reverse
    df['Date'] = pd.to_datetime(df["Date"],infer_datetime_format=True)
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    
    #new_date_range = pd.date_range(start="2012-12-28", end="2014-09-22", freq="D")
    #new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    print(df.head(10))
    #df = df.reindex(new_date_range, fill_value=None)
    #df = df.fillna(method='ffill')
    #print(df.head(10))
    df.to_csv("stonks/Oil_WTI_ORDERED_two.csv")



def prepare_Oil_WTI_full():
    df1 = read_csv('stonks\Oil_WTI_ORDERED_top.csv')
    df2 = read_csv('stonks\Oil_WTI_ORDERED_bottom.csv')
    
    df = pd.concat([df1, df2])

    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    new_date_range = pd.date_range(start="2012-12-28", end="2020-02-19", freq="D")
    
    df = df.reindex(new_date_range, fill_value=None)
    df['Open'] = df['Open'].replace(0.0,None) # sometimes there was price 0.0 so we override with previous data
    df = df.fillna(method='ffill')

    #print(df.head(10))
    print(df.head(10))
    df.to_csv("stonks/Oil_WTI_ORDERED.csv")
    

### copy to sarima

def create_OIL():
    global VIX
    df = read_csv('stonks/Oil_WTI_ORDERED.csv')
    stgng = np.array(df['Open'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    #stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))



def create_VIX():
    global VIX
    df = read_csv('stonks/VIX_ORDERED.csv')
    stgng = np.array(df[' Open'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    #stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))



def create_MSCI():
    global MSCI
    df = read_csv('stonks/MSCI_ORDERED.csv')
    stgng = np.array(df['Öppen'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    #stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))

def create_SP500():
    global SP500
    df = read_csv('stonks/SP500_ORDERED.csv')
    stgng = np.array(df[' Open'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    #stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))


def create_OMSX30():
    global OMSX30
    df = read_csv('stonks/OMSX30_ORDERED.csv')
    stgng = np.array(df['Stängn.kurs'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    #stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))


def create_OMSXPI():
    global OMSXPI
    df = read_csv('stonks/OMSXPI_ORDERED.csv')
    stgng = np.array(df['Stängn.kurs'].values)
    fixed_list = []
    for value in stgng:
        fixed_list.append(int(value*100))
    #stgng_scaled = minmax_scale(fixed_list, feature_range=(0,500))

if __name__ == "__main__":
    prepare_Oil_WTI_full()