from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot

from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot

from sklearn.metrics import mean_squared_error
from math import sqrt

# tutorial from https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/

def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

def print_dataset():
    series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    print(series.head())
    series.plot()
    pyplot.show()

def calculate_correlation():
    series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
    autocorrelation_plot(series)
    pyplot.show()

def arima():
    series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
    series.index = series.index.to_period('M')
    # fit model
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    # summary of fit model
    print(model_fit.summary())
    # line plot of residuals
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    pyplot.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    pyplot.show()
    # summary stats of residuals
    print(residuals.describe())

def arima_rolling_forecast():
    series = read_csv('shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
    series.index = series.index.to_period('M')
    # split into train and test sets
    X = series.values
    size = int(len(X) * 0.66)
    train, test = X[0:size], X[size:len(X)]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        model = ARIMA(history, order=(5,1,0))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))
    # evaluate forecasts
    rmse = sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()
if __name__ == "__main__":
    arima_rolling_forecast()