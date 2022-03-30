import pandas as pd
import numpy as np

"""
datum där lönen INTE inträffar den 25e:
(25:e är röd dag) - notera fredagen innan, om den är röd så gå bakåt till föregående icke-röda dag

2020
Jan 24
2019
Dec 23
Aug 23
Maj 24

2018
Dec 21
Nov 23
Aug 24
Mar 23
Feb 23
2017
Dec 22
Nov 24
Jun 22
Maj 24
Mar 24
Feb 24
2016
Dec 23 
Sep 23
Jun 23

"""

def createSalaryDay():
    rng = pd.date_range('2016-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["IsSalaryDay"] = 0
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.day == 25, 1)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-01-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-01-23'), 1)

    print(df['Date'].dt.date)
    print(df.head(60))

def createNear25th():
    global near25th
    # create an array of 5 dates starting at '2015-02-24', one per day
    rng = pd.date_range('2016-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["Is25th"] = 1
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 24, 1)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 25, 1)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 26, 1)
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 27, 1)
    print(df)


    near25th = np.array(df['Is25th'].values)

    print("len is ", near25th)
    print("len is ", len(near25th))

if __name__ == "__main__":
    createSalaryDay()
