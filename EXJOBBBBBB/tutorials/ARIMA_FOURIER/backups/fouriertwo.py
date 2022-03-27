import bornly as bns
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

import matplotlib.pyplot as plt


    
df_complete = pd.read_csv('Time.csv')

precovid_startdate = '2016-01-01'
precovid_enddate = '2020-02-19'
postcovid_startdate = '2020-02-20'
postcovid_enddate = '2021-12-01'

mask_clickouts = (df_complete.iloc[:, 0] > precovid_startdate) & (df_complete.iloc[:, 0] <= precovid_enddate)

click_outs = np.array(df_complete['clicks_out'][mask_clickouts].values)

#click_outs_df["t"] = np.array(np.arange(len(click_outs_df)))

my_frame_df = pd.DataFrame(data={'clicks_out':click_outs,'t':np.array(np.arange(len(click_outs)))})

#flights = bns.load_dataset("flights")

#flights["t"] = np.arange(len(flights))
#print(flights)
PERIOD = 365 # known period
n_steps = 50 # predict this number of days
train = my_frame_df.iloc[:-n_steps].copy()
test = my_frame_df.iloc[-n_steps:].copy()


def get_fourier_features(n_order, period, values):
    fourier_features = pd.DataFrame(
        {
            f"fourier_{func}_order_{order}": getattr(np, func)(
                2 * np.pi * values * order / period
            )
            for order in range(1, n_order + 1)
            for func in ("sin", "cos")
        }
    )
    return fourier_features


best_aicc = None
best_n_order = None

for n_order in range(1, 2):
    train_fourier_features = get_fourier_features(n_order, PERIOD, train["t"])
    arima_exog_model = auto_arima(
        y=np.log(train["clicks_out"]),
        exogenous=train_fourier_features,
        seasonal=False,
    )
    if best_aicc is None or arima_exog_model.aicc() < best_aicc:
        best_aicc = arima_exog_model.aicc()
        best_norder = n_order

train_fourier_features = get_fourier_features(best_norder, PERIOD, train["t"])
arima_exog_model = auto_arima(
    y=np.log(train["clicks_out"]),
    exogenous=train_fourier_features,
    seasonal=False
)
test_fourier_features = get_fourier_features(best_norder, PERIOD, test["t"])
y_arima_exog_forecast = arima_exog_model.predict(
    n_periods=n_steps,
    exogenous=test_fourier_features,
)
test["forecast"] = np.exp(y_arima_exog_forecast)

plt.plot( range(len(y_arima_exog_forecast)), y_arima_exog_forecast, color='green')
plt.show()
