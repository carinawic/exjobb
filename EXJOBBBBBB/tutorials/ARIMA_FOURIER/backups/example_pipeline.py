"""
=========================
Pipelines with auto_arima
=========================


Like scikit-learn, ``pmdarima`` can fit "pipeline" models. That is, a pipeline
constitutes a list of arbitrary length comprised of any number of
``BaseTransformer`` objects strung together ordinally, and finished with an
``AutoARIMA`` object.

The benefit of a pipeline is the ability to condense a complex sequence of
stateful transformations into a single object that can call ``fit``,
``predict`` and ``update``. It can also be serialized into *one* pickle file,
which greatly simplifies your life.

.. raw:: html

   <br/>
"""
import pandas as pd
import numpy as np
from pmdarima.pipeline import Pipeline
from pmdarima.preprocessing import BoxCoxEndogTransformer
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import auto_arima
from pmdarima.utils import c, diff
# chapter 2 of https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

from pmdarima.datasets import load_lynx
from pmdarima.arima.utils import nsdiffs

import matplotlib.pyplot as plt 
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler, minmax_scale 
from sklearn.metrics import mean_squared_error,  mean_absolute_error, mean_absolute_percentage_error
import datetime

click_outs = []
week = []
click_outs_df = ""

clicks_Search = []
clicks_Inactive = []
clicks_Active = []

def prepare_data():

    global click_outs, week, clicks_Search, clicks_Inactive, clicks_Active, click_outs_df

    # read dataset
    df_complete = pd.read_csv('Time.csv')

    # read media_clicks_df as pd df and add sum_total col
    media_clicks_df = pd.read_csv('media_clicks.csv')
    
    media_clicks_df['media_clicks_SEARCH_df'] = media_clicks_df['Media_Bing_lowerfunnel_search_brand'] + media_clicks_df['Media_Bing_midfunnel_search_midbrand'] + media_clicks_df['Media_Bing_upperfunnel_search_nobrand'] + media_clicks_df['Media_Google_lowerfunnel_search_brand'] + media_clicks_df['Media_Google_midfunnel_search_midbrand'] + media_clicks_df['Media_Google_upperfunnel_search_nobrand'] 
    
    media_clicks_df['media_clicks_INACTIVE_df'] = media_clicks_df['Media_Google_video_lowerfunnel_Youtube'] + media_clicks_df['Media_Google_video_upperfunnel_Youtube'] + media_clicks_df['Media_Youtube_Masthead_upperfunnel_video'] + media_clicks_df['Media_Online_radio_upperfunnel'] + media_clicks_df['Media_Radio_upperfunnel'] + media_clicks_df['Media_TV_upperfunnel'] + media_clicks_df['Media_DBM_upperfunnel_video'] + media_clicks_df['Media_DC_DBM_upperfunnel_video'] + media_clicks_df['Media_MediaMath_upperfunnel_video'] + media_clicks_df['Media_Snapchat_upperfunnel_video'] + media_clicks_df['Media_Tiktok_upperfunnel_video'] + media_clicks_df['Media_Eurosize_upperfunnel_OOH_JCD'] + media_clicks_df['Media_Eurosize_upperfunnel_OOH_VA'] 

    media_clicks_df['media_clicks_ACTIVE_df'] =  media_clicks_df['Media_Adwell_upperfunnel_native'] + media_clicks_df['Media_DBM_lowerfunnel_display'] + media_clicks_df['Media_DBM_midfunnel_display'] + media_clicks_df['Media_DBM_upperfunnel_display'] + media_clicks_df['Media_Facebook_lowerfunnel_display'] + media_clicks_df['Media_Facebook_lowerfunnel_video'] + media_clicks_df['Media_Facebook_upperfunnel_display'] + media_clicks_df['Media_Facebook_upperfunnel_video'] + media_clicks_df['Media_Flygstart_upperfunnel_newsletter'] + media_clicks_df['Media_Google_lowerfunnel_display'] + media_clicks_df['Media_Google_midfunnel_display'] + media_clicks_df['Media_Google_upperfunnel_display'] + media_clicks_df['Media_HejSenior_upperfunnel_newsletter'] + media_clicks_df['Media_Instagram_lowerfunnel_display'] + media_clicks_df['Media_Instagram_lowerfunnel_video'] + media_clicks_df['Media_Instagram_upperfunnel_display'] + media_clicks_df['Media_Instagram_upperfunnel_video'] + media_clicks_df['Media_Newsletter_lowerfunnel'] + media_clicks_df['Media_Newsner_midfunnel_native'] + media_clicks_df['Media_Secreteescape_midfunnel_display'] + media_clicks_df['Media_Smarter_Travel_upperfunnel_affiliate'] + media_clicks_df['Media_Snapchat_upperfunnel_display'] + media_clicks_df['Media_Sociomantic_lowerfunnel_retarg_display'] + media_clicks_df['Media_Sociomantic_upperfunnel_prospecting_display'] + media_clicks_df['Media_TradeTracker_upperfunnel_affiliate'] 
    
    
    precovid_startdate = '2016-01-01'
    precovid_enddate = '2020-02-19'
    postcovid_startdate = '2020-02-20'
    postcovid_enddate = '2021-12-01'
    
    # mask between certain dates DURING COVID
    mask_clickouts = (df_complete.iloc[:, 0] > precovid_startdate) & (df_complete.iloc[:, 0] <= precovid_enddate)
    mask_media_clicks = (media_clicks_df.iloc[:, 0] > precovid_startdate) & (media_clicks_df.iloc[:, 0] <= precovid_enddate)

    click_outs = np.array(df_complete['clicks_out'][mask_clickouts].values)
    week = np.array(df_complete['week'][mask_clickouts].values)

    click_outs_df = df_complete['clicks_out'][mask_clickouts]
 
    clicks_Search = np.array(media_clicks_df["media_clicks_SEARCH_df"][mask_media_clicks].values)
    clicks_Inactive = np.array(media_clicks_df["media_clicks_INACTIVE_df"][mask_media_clicks].values)
    clicks_Active = np.array(media_clicks_df["media_clicks_ACTIVE_df"][mask_media_clicks].values)

    clicks_Search = minmax_scale(clicks_Search, feature_range=(0,500))
    clicks_Inactive = minmax_scale(clicks_Inactive, feature_range=(0,500))
    clicks_Active = minmax_scale(clicks_Active, feature_range=(0,500))


def arimax():
    print(__doc__)

    # Author: Taylor Smith <taylor.smith@alkaline-ml.com>

    import numpy as np
    import pmdarima as pm
    from pmdarima import pipeline
    from pmdarima import model_selection
    from pmdarima import preprocessing as ppc
    from pmdarima import arima
    from matplotlib import pyplot as plt


    # Load the data and split it into separate pieces
    data = pm.datasets.load_wineind()
    train, test = model_selection.train_test_split(data, train_size=150)


    data = click_outs_df
    train = data.iloc[:(len(data)-365)]
    test = data.iloc[(len(data)-365):] # last year for testing


    # Let's create a pipeline with multiple stages... the Wineind dataset is
    # seasonal, so we'll include a FourierFeaturizer so we can fit it without
    # seasonality
    pipe = pipeline.Pipeline([
        ("fourier", ppc.FourierFeaturizer(m=12, k=4)),
        ("arima", arima.AutoARIMA(stepwise=True, trace=1, error_action="ignore",
                                seasonal=False,  # because we use Fourier
                                suppress_warnings=True))
    ])

    pipe.fit(train)
    print("Model fit:")
    print(pipe)

    # We can compute predictions the same way we would on a normal ARIMA object:
    preds, conf_int = pipe.predict(n_periods=365, return_conf_int=True)
    print("\nForecasts:")
    print(preds)

    # Let's take a look at the actual vs. the predicted values:
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    fig.tight_layout()

    # Visualize goodness of fit
    in_sample_preds, in_sample_confint = \
        pipe.predict_in_sample(X=None, return_conf_int=True)

    n_train = train.shape[0]

    

    plt.plot( range(len(train)), train, color='green')
    plt.plot( range(len(train), len(train)+len(test)), test, color='blue')
    plt.plot( range(len(train), len(train)+len(test)), preds, color="red")

    """
    x0 = np.arange(n_train)
    axes[0].plot(x0, train, alpha=0.75)
    axes[0].scatter(x0, in_sample_preds, alpha=0.4, marker='-')
    axes[0].fill_between(x0, in_sample_confint[:, 0], in_sample_confint[:, 1],
                        alpha=0.1, color='b')
    axes[0].set_title('Actual train samples vs. in-sample predictions')
    axes[0].set_xlim((0, x0.shape[0]))

    # Visualize actual + predicted
    x1 = np.arange(n_train + preds.shape[0])
    axes[1].plot(x1[:n_train], train, alpha=0.75)
    # axes[1].scatter(x[n_train:], preds, alpha=0.4, marker='o')
    axes[1].scatter(x1[n_train:], test[:preds.shape[0]], alpha=0.4, marker='-')
    axes[1].fill_between(x1[n_train:], conf_int[:, 0], conf_int[:, 1],
                        alpha=0.1, color='b')
    axes[1].set_title('Actual test samples vs. forecasts')
    axes[1].set_xlim((0, data.shape[0]))

    # We can also call `update` directly on the pipeline object, which will update
    # the intermittent transformers, where necessary:
    newly_observed, still_test = test[:15], test[15:]
    pipe.update(newly_observed, maxiter=10)

    # Calling predict will now predict from newly observed values
    new_preds = pipe.predict(still_test.shape[0])
    print(new_preds)

    x2 = np.arange(data.shape[0])
    n_trained_on = n_train + newly_observed.shape[0]

    axes[2].plot(x2[:n_train], train, alpha=0.75)
    axes[2].plot(x2[n_train: n_trained_on], newly_observed, alpha=0.75, c='orange')
    # axes[2].scatter(x2[n_trained_on:], new_preds, alpha=0.4, marker='o')
    axes[2].scatter(x2[n_trained_on:], still_test, alpha=0.4, marker='-')
    axes[2].set_title('Actual test samples vs. forecasts')
    axes[2].set_xlim((0, data.shape[0]))

    """
    plt.show()


if __name__ == "__main__":
    prepare_data()
    arimax()
