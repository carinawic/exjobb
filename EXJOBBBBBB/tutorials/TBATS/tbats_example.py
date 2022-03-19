from tbats import TBATS
import pandas as pd
import numpy as np

def myfun():
    df = pd.read_csv('kaggle_sales.csv')
    df = df[(df['store'] == 1) & (df['item'] == 1)] # item 1 in store 1
    df = df.set_index('date')
    y = df['sales']
    y_to_train = y.iloc[:(len(y)-365)]
    y_to_test = y.iloc[(len(y)-365):] # last year for testing

    ## stack multivariate data
    feature1_train = np.array(range(y_to_train.size))
    feature1_test = np.array(range(y_to_test.size))

    # make input into 2 time series
    data = np.dstack([y_to_train.values, feature1_train])

    print("data")
    print(data)


    # Fit the model
    estimator = TBATS(seasonal_periods=(7, 365.25))
    model = estimator.fit(y_to_train)
    # Forecast 365 days ahead
    y_forecast = model.forecast(steps=365)
        # Summarize fitted model
    print(model.summary())


    #print('MAE', np.mean(np.abs(y_forecast - y_to_test)))
    # multivariate RMSE?
    
if __name__ == '__main__':
    myfun()