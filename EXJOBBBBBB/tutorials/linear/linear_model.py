# chapter 2 of https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
import pandas as pd
import matplotlib.pyplot as plt 
import math
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import datetime


click_outs = []
week = []

def prepare_data():

    global click_outs, week

    # read dataset
    df_complete = pd.read_csv('Time.csv')

    # mask between certain dates DURING COVID
    mask3 = (df_complete.iloc[:, 0] > '2020-02-19') & (df_complete.iloc[:, 0] <= '2021-12-01')
 
    click_outs = np.array(df_complete['clicks_out'][mask3].values)
    week = np.array(df_complete['week'][mask3].values)

    

def workingexample():
    
    #plt.plot(range(len(click_outs),click_outs,'-', label="click_outs")
    #plt.show()
    pass

if __name__ == "__main__":
    prepare_data()
    workingexample()
    # 22.93 RMSE means an error of about 23 passengers (in thousands) 