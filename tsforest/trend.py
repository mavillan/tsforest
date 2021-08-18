import numpy as np
import pandas as pd
from fbprophet import Prophet
from tsforest.config import default_prophet_kwargs
from joblib import Parallel, delayed
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
logging.getLogger("fbprophet").setLevel(logging.WARNING)


def fit_prophet_model(data, kwargs=dict()):
    """
    It receives the dataset and Prophet's model parameters 
    and returns a trained instance of the model
    """ 
    kwargs_extra = dict()
    keys_extra = ["wk_fourier_order", "wk_prior_scale", "mt_fourier_order", 
                  "mt_prior_scale", "yr_fourier_order", "yr_prior_scale"]
    for key in keys_extra:
        if key in kwargs.keys(): kwargs_extra[key] = kwargs.pop(key)

    if len(kwargs) == 0: prophet_model = Prophet()
    else: prophet_model = Prophet(**kwargs) 
        
    if len(kwargs_extra) > 0:
        if "wk_prior_scale" in kwargs_extra:
            prophet_model.add_seasonality(name="weekly", period=7, 
                                          fourier_order=kwargs_extra["wk_fourier_order"],
                                          prior_scale=kwargs_extra["wk_prior_scale"])
        if "mt_prior_scale" in kwargs_extra:
            prophet_model.add_seasonality(name="monthly", period=30.5, 
                                          fourier_order=kwargs_extra["mt_fourier_order"],
                                          prior_scale=kwargs_extra["mt_prior_scale"])
        if "yr_prior_scale" in kwargs_extra:
            prophet_model.add_seasonality(name="yearly", period=365.25, 
                                          fourier_order=kwargs_extra["yr_fourier_order"],
                                          prior_scale=kwargs_extra["yr_prior_scale"])
    # model fitting
    prophet_model.fit(data)    
    return prophet_model

def compute_prophet_trend(prophet_model, predict_dataframe=None, periods=30, include_history=False):
    if predict_dataframe is None:
        predict_dataframe = prophet_model.make_predict_dataframe(periods=periods, 
                                                                 include_history=include_history)
        trend = prophet_model.predict_trend(predict_dataframe)
    else:
        predict_dataframe = prophet_model.setup_dataframe(predict_dataframe.copy())
        trend = prophet_model.predict_trend(predict_dataframe)
    return trend

class TrendModel():
    """
    Parameters
    ----------
    model_kwargs: dict
        Prophet model params
    """

    def __init__(self, model_kwargs):
        self.model_kwargs = model_kwargs

    def fit(self, data):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with columns 'ds' and 'y'.
        """
        prophet_kwargs ={**default_prophet_kwargs, **self.model_kwargs}
        trend_model = fit_prophet_model(data, prophet_kwargs)
        self.trend_model = trend_model

    def predict(self, predict_period):
        """
        Parameters
        ----------
        predict_period: pandas.DataFrame
            Dataframe with column 'ds' containing the time period to be predicted.
        """
        trend = compute_prophet_trend(self.trend_model, predict_dataframe=predict_period)
        trend_dataframe = pd.DataFrame({"ds":predict_period.ds.values, "trend":trend.values})
        return trend_dataframe

def compute_trend_model(train_data_chunk, key, model_kwargs):
    trend_model = TrendModel(model_kwargs=model_kwargs)
    trend_model.fit(train_data_chunk.loc[:, ["ds", "y"]])
    return {key:trend_model}

def compute_trend_models(data, valid_index=pd.Index([]), ts_uid_columns=None, 
                         model_kwargs=dict(), n_jobs=-1):
    train_data = data.drop(valid_index, axis=0)
    if ts_uid_columns is None:
        train_data["ts_uid"] = 0
        ts_uid_columns = ["ts_uid"]
    ts_uid_values = train_data.loc[:, ts_uid_columns].drop_duplicates()

    train_data_split = list()
    for _,row in ts_uid_values.iterrows():
        key = tuple([item for _,item in row.iteritems()])
        query_string = " & ".join([f"{col_name}=={value}" for col_name,value in row.iteritems()])
        train_data_chunk = train_data.query(query_string)
        train_data_split.append((train_data_chunk, key))
    
    with Parallel(n_jobs=n_jobs) as parallel:
        delayed_func = delayed(compute_trend_model)
        with tqdm(train_data_split) as tqdm_train_data_split:
            trend_models = parallel(
                delayed_func(*data_chunk, model_kwargs)
                for data_chunk in tqdm_train_data_split
            )
        tqdm_train_data_split.close()

    trend_models_out = {}
    for trend_model in trend_models: trend_models_out.update(trend_model)
    return trend_models_out
