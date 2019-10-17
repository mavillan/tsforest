import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

import numpy as np
import pandas as pd
from zope.interface import implementer
from catboost import CatBoostRegressor, Pool 

from tsforest.config import cat_parameters
from tsforest.forecaster_base import ForecasterBase
from tsforest.forecaster_interface import ForecasterInterface


@implementer(ForecasterInterface)
class CatBoostForecaster(ForecasterBase):
    """
    Parameters
    ----------
    model_params : dict
        Dictionary containing the parameters of the specific boosting model 
    features: list
        List of features to be included
    categorical_features: list
        List of names of categorical features
    calendar_anomaly: list
        List of names of calendar features affected by an anomaly
    detrend: bool
        Whether or not to remove the trend from time series
    response_scaling:
        Whether or not to perform scaling of the reponse variable
    lags: list
        List of integer lag values
    window_sizes: list
        List of integer window sizes values
    window_functions: list
        List of string names of the window functions
    """
    def __init__(self, model_params=dict(), features=["calendar", "calendar_cyclical"], 
                 categorical_features=list(), calendar_anomaly=False, detrend=True, 
                 response_scaling=False, lags=None, window_sizes=None, window_functions=None):

        if lags is not None and "lag" not in features:
            features.append("lag")
        if (window_sizes is not None and window_functions is not None) and "rw" not in features:
            features.append("rw")

        self.model_params = model_params
        self.features = features
        self._categorical_features = categorical_features
        self.calendar_anomaly = calendar_anomaly
        self.detrend = detrend
        self.response_scaling = response_scaling
        self.lags = lags
        self.window_sizes = window_sizes
        self.window_functions = window_functions
        self._validate_inputs()

    def _cast_dataframe(self, features_dataframe, categorical_features):
        """
        Parameters
        ----------
        features_dataframe: pandas.DataFrame
            dataframe containing all the features
        categorical_features: list
            list of names of categorical features
        Returns
        ----------
        features_dataframe_casted: catboost.Pool
            features dataframe casted to CatBoost dataframe format
        """
        dataset_params = {"data":features_dataframe.loc[:, self.input_features],
                          "cat_features":categorical_features}
        if "weight" in features_dataframe.columns:
            dataset_params["weight"] = features_dataframe.weight.values
        if self.target in features_dataframe.columns:
            dataset_params["label"] = features_dataframe.loc[:, self.target]
        features_dataframe_casted = Pool(**dataset_params)
        return features_dataframe_casted  

    def fit(self, train_data, valid_period=None):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            dataframe with at least columns 'ds' and 'y'
        valid_period: pandas.DataFrame
            dataframe (with column 'ds') indicating the validation period
        """
        self._validate_fit_inputs(train_data, valid_period)
        train_features,categorical_features = super()._prepare_train_features(train_data)

        if valid_period is not None:
            valid_features = super()._prepare_valid_features(valid_period, train_features)
            valid_start_time = valid_features.ds.min()
            # removes validation period from train data
            train_data = train_data.query("ds < @valid_start_time")
            train_features = train_features.query("ds < @valid_start_time")
        
        train_features["y_hat"] = super()._prepare_train_response(train_features)
        if valid_period is not None:
            valid_features["y_hat"] = super()._prepare_valid_response(valid_features)
        
        train_features_casted = self._cast_dataframe(train_features, categorical_features)
        valid_features_casted = self._cast_dataframe(valid_features, categorical_features) \
                                if valid_period is not None else None

        self.train_data = train_data
        self.categorical_features = categorical_features
        self.train_features = train_features
        self.train_features_casted = train_features_casted

        self.valid_period = valid_period
        self.valid_features = valid_features if valid_period is not None else None
        self.valid_features_casted = valid_features_casted

        # model_params overwrites default params of model
        model_params = {**cat_parameters, **self.model_params}
        training_params = {"X":train_features_casted,
                           "verbose":False}
        if valid_period is not None:
            training_params["eval_set"] = valid_features_casted
        elif "early_stopping_rounds" in training_params:
            del training_params["early_stopping_rounds"]

        # model training
        model = CatBoostRegressor(**model_params)
        model.fit(**training_params)
        self.model = model
        self.best_iteration = model.best_iteration_ if model.best_iteration_ is not None else model.tree_count_

    def _predict(self, model, predict_features, trend_dataframe):
        """
        Parameters
        ----------
        model: catboost.CatBoostRegressor 
            Trained CatBoostRegressor model.
        predict_features: pandas.DataFrame
            Datafame containing the features for the prediction period.
        trend_dataframe: pandas.DataFrame
            Dataframe containing the trend estimation over the prediction period.
        """
        y_train = self.train_features.y.values
        y_valid = self.valid_features.y.values \
                  if self.valid_period is not None else np.array([])
        y = np.concatenate([y_train, y_valid])

        prediction = list()
        for idx in range(predict_features.shape[0]):
            if "lag" in self.features:
                for lag in self.lags:
                    predict_features.loc[idx, f"lag_{lag}"] = y[-lag]
            if "rw" in self.features:
                for window_func in self.window_functions:
                    for window in self.window_sizes:
                        predict_features.loc[idx, f"{window_func}_{window}"] = getattr(np, window_func)(y[-window:])
            y_pred = model.predict(predict_features.loc[[idx], self.input_features])
            prediction.append(y_pred.copy())
            if self.response_scaling:
                y_pred *= self.y_std
                y_pred += self.y_mean
            if self.detrend:
                y_pred += trend_dataframe.loc[idx, "trend"]
            y = np.append(y, [y_pred])
        return np.asarray(prediction).ravel()

    def predict(self, predict_data):
        """
        Parameters
        ----------
        predict_data: pandas.DataFrame
            Datafame containing the features for the prediction period.
            Contains the same columns as 'train_data' except for 'y'.
        Returns
        ----------
        prediction_dataframe: pandas.DataFrame
            Dataframe containing the dates 'ds' and predictions 'y_pred'.
        """
        self._validate_predict_inputs(predict_data) 
        predict_features = super()._prepare_predict_features(predict_data)
        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(predict_data.loc[:, ["ds"]])
        else:
            trend_dataframe = None

        if "lag" in self.features or "rw" in self.features:
            prediction = self._predict(self.model, predict_features, trend_dataframe)
        else:
            predict_features_casted = self._cast_dataframe(predict_features, self.categorical_features)
            prediction = self.model.predict(predict_features_casted)
        
        if self.response_scaling:
            prediction *= self.y_std
            prediction += self.y_mean
        if self.detrend:
            prediction += trend_dataframe.trend.values
        if "zero_response" in predict_features.columns:
            zero_response_mask = predict_features["zero_response"]==1
            prediction[zero_response_mask] = 0

        self.predict_features = predict_features
            
        prediction_dataframe = pd.DataFrame({"ds":predict_data.ds, "y_pred":prediction})
        return prediction_dataframe 