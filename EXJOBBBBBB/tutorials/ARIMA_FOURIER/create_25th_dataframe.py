import pandas as pd
import numpy as np


def createSalaryDay():

    # df should be the one containing clickouts, but only for the good time range 2016-2020
    rng = pd.date_range('2016-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["IsSalaryDay"] = 0
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.day == 25, 1)

    """
    2016
    Dec 23 
    Sep 23
    Jun 23
    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-12-23'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-11-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-11-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2016-06-23'), 1)

    """
    2017
    Dec 22
    Nov 24
    Jun 22
    Maj 24
    Mar 24
    Feb 24
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-12-22'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-11-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-06-22'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-05-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-03-24'), 1)
    
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2017-02-24'), 1)
    
    """ 
    2018
    Dec 21
    Nov 23
    Aug 24
    Mar 23
    Feb 23
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-12-21'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-11-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-11-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-08-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-08-24'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-03-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-03-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-02-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2018-02-23'), 1)

    """
    2019
    Dec 23
    Aug 23
    Maj 24
    """

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-12-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-08-23'), 1)

    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2019-05-24'), 1)

    """
    2020
    Jan 24
    """
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2020-01-25'), 0)
    df["IsSalaryDay"] = df['IsSalaryDay'].mask(df['Date'].dt.date == pd.to_datetime('2020-01-24'), 1)

    # ?? df["values_when_is_salary_date"] = df['clickouts'].mask(df['IsSalaryDay'] == 1)


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
