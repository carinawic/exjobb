import pandas as pd
import numpy as np

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
    createNear25th()
