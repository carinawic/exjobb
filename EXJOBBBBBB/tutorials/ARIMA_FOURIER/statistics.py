
import numpy as np
import pandas as pd
from pandas import read_csv


def calc():

    
    df = read_csv('Time_training.csv')
    
    graph_means = []

    
    #twentifour = df.loc[ pd.to_datetime(df['date']).dt.day == 24, 'clicks_out'].mean()
    #twentifive = df.loc[ pd.to_datetime(df['date']).dt.day == 25, 'clicks_out'].mean()
    #thirty = df.loc[ pd.to_datetime(df['date']).dt.day == 30, 'clicks_out'].mean()
    #twenty = df.loc[ pd.to_datetime(df['date']).dt.day == 20, 'clicks_out'].mean()
    #third = df.loc[ pd.to_datetime(df['date']).dt.day == 3, 'clicks_out'].mean()
    #means = df['clicks_out'].mean()

    for i in range(1,32):
        sumofclicks = df.loc[ pd.to_datetime(df['date']).dt.day == i, 'clicks_out'].mean()
        graph_means.append(sumofclicks)
        print("month: ", i, " mean: ", sumofclicks)
    

    

    import matplotlib.pyplot as plt
    plt.plot(graph_means)
    plt.ylabel('mean flights')
    plt.xlabel('day of the month')
    plt.show()

    """
    # create an array of 5 dates starting at '2015-02-24', one per day
    rng = pd.date_range('2013-01-01', '2020-02-19', freq='D')
    df = pd.DataFrame({ 'Date': rng}) 
    df["Is25th"] = 0
    df["Is25th"] = df['Is25th'].mask(df['Date'].dt.day == 25, 1)
    print(df)
    
    near25th = np.array(df['Is25th'].values)
    """
if __name__ == "__main__":
    calc()