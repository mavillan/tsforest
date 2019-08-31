import numpy as np
import pandas as pd
from itertools import count
import collections
from fbprophet import Prophet

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
logging.getLogger("fbprophet").setLevel(logging.WARNING)


def moving_average(data, window_size):
    """ 
    Computes moving average using discrete linear convolution 
    of two one dimensional sequences.
    Parameters
    ----------
            data (pandas.Series): independent variable
            window_size (int): rolling window size
    Returns
    ---------
            np.ndarray of linear convolution
    """
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(data, window, 'same')

def anomaly_detector(y, window_size=10, sigma=2.0):
    """ 
    Identifies anomalies in TS data through rolling std

    Parameters
    ----------
    y: pandas.Series 
        independent variable
    window_size: int 
        rolling window size
    sigma : int 
        value for standard deviation
    Returns
    --------
    a dict (dict of 'standard_deviation': int, 'anomalies_dict': (index: value))
    containing information about the points indentified as anomalies
    """
    avg = moving_average(y, window_size)
    avg_list = avg.tolist()
    residual = y - avg
    # calculate the variation in the distribution of the residual
    testing_std = residual.rolling(window_size).std()
    testing_std_as_df = pd.DataFrame(testing_std)
    rolling_std = testing_std_as_df.replace(np.nan,
                                  testing_std_as_df.loc[window_size - 1]).round(3).iloc[:,0].tolist()
    ret = collections.OrderedDict([(index, y_i) for 
                                   index, y_i, avg_i, rs_i in zip(count(),y,avg_list,rolling_std) 
                                   if (y_i > avg_i + (sigma * rs_i)) | (y_i < avg_i - (sigma * rs_i))])
    return ret
    
def ts_cleaning(df):
    """
    Parameters
    -----------
    df: pd.DataFrame
        Dataframe with columns "ds" and "y" containing the TS
    Returns
    -----------
    pd.Dataframe
        Dataframe in the same format as input, with outliers removed 
        from the TS
    """
    df = df.copy()
    # days with no sales are removed
    df = df[df.y!=0].reset_index(drop=True)
    # detecting outliers with a rolling window std
    anomaly_dict = anomaly_detector(df.y, window_size=10, sigma=2.)
    idx = list(anomaly_dict.keys())
    idx.sort()
    # removing outlier from dataframe
    df.drop(idx, axis=0, inplace=True)
    return df

def train_model(df, kwargs=None, kwargs_extra=None, ts_preproc=True):
    """
    It receives the dataset and Prophet's model parameters 
    and returns a trained instance of the model
    """
    # time-series preprocessing 
    if ts_preproc: df = ts_cleaning(df)
        
    # setting the model parameters
    if kwargs is None: prophet_model = Prophet()
    else: prophet_model = Prophet(**kwargs) 
        
    if kwargs_extra is not None:
        if "wk_prior_scale" in kwargs_extra:
            prophet_model.add_seasonality(name="weekly", period=7, 
                                          fourier_order=kwargs_extra["wk_fourier_order"],
                                          prior_scale=kwargs_extra["wk_prior_scale"])
        if "mt_prior_scale" in kwargs_extra:
            prophet_model.add_seasonality(name='monthly', period=30.5, 
                                          fourier_order=kwargs_extra["mt_fourier_order"],
                                          prior_scale=kwargs_extra["mt_prior_scale"])
        if "yr_prior_scale" in kwargs_extra:
            prophet_model.add_seasonality(name="yearly", period=365.25, 
                                          fourier_order=kwargs_extra["yr_fourier_order"],
                                          prior_scale=kwargs_extra["yr_prior_scale"])
    # model fitting
    prophet_model.fit(df)    
    return prophet_model

def compute_prophet_trend(prophet_model, future_dataframe=None, periods=30, include_history=False):
    if future_dataframe is None:
        future_dataframe = prophet_model.make_future_dataframe(periods=periods, 
                                                               include_history=include_history)
        trend = prophet_model.predict_trend(future_dataframe)
    else:
        future_dataframe = prophet_model.setup_dataframe(future_dataframe.copy())
        trend = prophet_model.predict_trend(future_dataframe)
    return trend
