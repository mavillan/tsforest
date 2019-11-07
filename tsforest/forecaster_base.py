import numpy as np
import pandas as pd
import category_encoders as ce
from inspect import getmembers, isfunction

from tsforest.features import FeaturesGenerator
from tsforest.trend import TrendEstimator
from tsforest import metrics

# available methods in pandas.core.window.Rolling as of pandas 0.25.1 
AVAILABLE_RW_FUNCTIONS = ["count", "sum", "mean", "median", "var", "std", "min", 
                          "max", "corr", "cov", "skew", "kurt", "quantile"]

AVAILABLE_METRICS = [member[0].split('_')[1] for member in getmembers(metrics) 
                     if isfunction(member[1])]

class ForecasterBase(object):
    """
    Parameters
    ----------
    model_params : dict
        Dictionary containing the parameters of the specific boosting model.
    features: list
        List of features to be included.
    exclude_features: list
        List of features to be excluded from training dataframe.
    categorical_features: list
        List of names of categorical features.
    categorical_encoding: str
        String name of categorical encoding to use.
    calendar_anomaly: list
        List of names of calendar features affected by an anomaly.
    detrend: bool
        Whether or not to remove the trend from time series.
    response_scaling:
        Whether or not to perform scaling of the reponse variable.
    lags: list
        List of integer lag values.
    window_sizes: list
        List of integer window sizes values.
    window_functions: list
        List of string names of the window functions.
    """
    def __init__(self, model_params=dict(), features=["calendar", "calendar_cyclical"], exclude_features=list(),
                 categorical_features=list(), categorical_encoding="default", calendar_anomaly=list(), 
                 detrend=True, response_scaling=False, lags=None, window_sizes=None, window_functions=None):

        if lags is not None and "lag" not in features:
            features.append("lag")
        if (window_sizes is not None and window_functions is not None) and "rw" not in features:
            features.append("rw")

        self.model = None
        self.model_params = model_params
        self.features = features
        self.exclude_features = ["ds", "y", "y_hat", "weight", "fold_column",
                                 "zero_response", "calendar_anomaly"] + exclude_features
        self._categorical_features = categorical_features
        self.categorical_encoding = categorical_encoding
        self.calendar_anomaly = calendar_anomaly
        self.detrend = detrend
        self.response_scaling = response_scaling
        self.lags = lags
        self.window_sizes = window_sizes
        self.window_functions = window_functions
        self._validate_inputs()

    def _validate_inputs(self):
        """
        Validates the inputs
        """
        if not isinstance(self.model_params, dict):
            raise TypeError("Parameter 'model_params' should be of type 'dict'.")

        if not isinstance(self.features, list):
            raise TypeError("Parameter 'features' should be of type 'list'.")
        else:
            if any([x not in ["calendar", "calendar_cyclical", "lag", "rw"] for x in self.features]):
                raise ValueError("Values in 'features' should be any of: ['calendar', 'calendar_cyclical', 'lag', 'rw'].")
        
        if not isinstance(self._categorical_features, list):
            raise TypeError("Parameter 'categorical_features' should be of type 'list'.")

        if not isinstance(self.calendar_anomaly, list):
            raise TypeError("Parameter 'calendar_anomaly' should be of type 'list'.")

        if not isinstance(self.detrend, bool):
            raise TypeError("Parameter 'detrend' should be of type 'bool'.")

        if not isinstance(self.response_scaling, bool):
            raise TypeError("Parameter 'response_scaling' should be of type 'bool'.")

        if self.lags is not None:
            if not isinstance(self.lags, list):
                raise TypeError("Parameter 'lags' should be of type 'list'.")
            else:
                if any([type(x)!=int for x in self.lags]):
                    raise ValueError("Values in 'lags' should be integers.")
                elif any([x<1 for x in self.lags]):
                    raise ValueError("Values in 'lags' should be integers greater or equal to 1.")
        
        if self.window_sizes is not None:
            if not isinstance(self.window_sizes, list):
                raise TypeError("Parameter 'window_sizes' should be of type 'list'.")
            else:
                if any([type(x)!=int for x in self.window_sizes]):
                    raise ValueError("Values in 'window_sizes' should be integers.")
                elif any([x<1 for x in self.window_sizes]):
                    raise ValueError("Values in 'window_sizes' should be integers greater or equal to 1.")
        
        if self.window_functions is not None:
            if not isinstance(self.window_functions, list):
                raise TypeError("Parameter 'window_functions' should be of type list.")
            else:
                if any([type(x)!=str for x in self.window_functions]):
                    raise ValueError("Values in 'window_functions' should be string names.")
                elif any([x not in AVAILABLE_RW_FUNCTIONS for x in self.window_functions]):
                    raise ValueError(f"Values in 'window_functions' should be any of: {AVAILABLE_RW_FUNCTIONS}.")
    
    def _validate_fit_inputs(self, train_data, valid_period):
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("Parameter 'train_data' should be of type pandas.DataFrame.")
        elif not ({"ds", "y"} <= set(train_data.columns.values)):
            raise ValueError("'train_data' should contain columns 'ds' and 'y'.")
        
        if valid_period is not None:
            if not isinstance(valid_period, pd.DataFrame):
                raise TypeError("Parameter 'valid_period' should be of type pandas.DataFrame.")
            elif {"ds"} != set(valid_period.columns.values):
                raise ValueError("'valid_period' should contain only the column 'ds'.")
    
    def _validate_predict_inputs(self, predict_data):
        if not isinstance(predict_data, pd.DataFrame):
            raise TypeError("Parameter 'predict_data' should be of type pandas.DataFrame.")
        elif not (set(self.train_data.columns) - set(predict_data.columns) == {'y'}):
            raise ValueError("'predict_data' shoud have the same columns as 'train_data' except for 'y'.")
    
    def _validate_evaluate_inputs(self, eval_data, metric):
        if not isinstance(eval_data, pd.DataFrame):
            raise TypeError("'eval_data' should be of type pandas.DataFrame.")
        elif not (set(self.train_data.columns) == set(eval_data.columns)):
            raise ValueError("'eval_data' should have the same columns as 'train_data'.")

        if not isinstance(metric, str):
            raise TypeError("'metric' should be of type str.")
        elif metric not in AVAILABLE_METRICS:
            raise ValueError(f"'metric' should be any of these: {AVAILABLE_METRICS}")

    def _apply_encoding(self, train_features, input_features, categorical_features, categorical_encoding):
        train_features = train_features.copy()
        encoder_class = getattr(ce, categorical_encoding)
        encoder = encoder_class(cols=categorical_features)
        encoder.fit(train_features.loc[:, input_features], train_features.loc[:, 'y'].values)
        train_features.loc[:, input_features] = encoder.transform(train_features.loc[:, input_features])
        return train_features,encoder
      
    def _prepare_train_features(self, train_data):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            Dataframe with at least columns 'ds' and 'y'.
        """
        features_generator = FeaturesGenerator(include_features=self.features,
                                               lags=self.lags,
                                               window_sizes=self.window_sizes,
                                               window_functions=self.window_functions)
        train_features,categorical_features = features_generator.compute_train_features(train_data)
        self.features_generator = features_generator

        categorical_features = categorical_features + self._categorical_features
        self.raw_features = train_features.columns
        self.input_features = [feature for feature in train_features.columns
                               if feature not in self.exclude_features]

        if "zero_response" in train_features.columns:
            train_features = train_features.query("zero_response != 1")
        if "calendar_anomaly" in train_features.columns:
            assert len(self.calendar_anomaly) != 0, \
                "'calendar_anomaly' column found, but names of affected features were not provided."
            assert set(self.calendar_anomaly) <= set(train_features.columns), \
                f"Calendar anomaly affected columns: {set(self.calendar_anomaly)-set(train_features.columns)} are not present in 'train_features'."
            idx = train_features.query("calendar_anomaly == 1").index
            train_features.loc[idx, self.calendar_anomaly] = np.nan
        if self.categorical_encoding != "default":
            train_features,encoder = self._apply_encoding(train_features, self.input_features, categorical_features, self.categorical_encoding)
            self.encoder = encoder
        return train_features,categorical_features
    
    def _prepare_valid_features(self, valid_period, train_features):
        """
        valid_period : pandas.DataFrame
            Dataframe with column 'ds' indicating the validation period.
        train_features: pandas.DataFrame
            Dataframe containing the training features.
        """
        valid_features = pd.merge(valid_period, train_features, how="inner", on=["ds"])
        assert len(valid_features) > 0, \
            "None of the dates in valid_period are in train_features."
        return valid_features

    def _prepare_predict_features(self, predict_data):
        """
        Parameters
        ----------
        predict_data: pandas.DataFrame
            Dataframe with the same columns as self.train_data (except for 'y') 
            containing the prediction period.
        Returns
        ----------
        predict_features: pandas.DataFrame
            Dataframe containing all the features for making predictions with 
            the trained model.
        """
        features_generator = self.features_generator
        predict_features,_ = features_generator.compute_predict_features(predict_data, ignore_const_cols=False)
        if "calendar_anomaly" in predict_features.columns:
            assert len(self.calendar_anomaly) != 0, \
                "'calendar_anomaly' column found, but no names of affected features were provided."
            assert set(self.calendar_anomaly) <= set(train_features.columns), \
                f"Calendar anomaly affected columns: {set(self.calendar_anomaly)-set(train_features.columns)} are not present in 'train_features'."
            idx = predict_features.query("calendar_anomaly == 1").index
            predict_features.loc[idx, self.calendar_anomaly] = np.nan
        if self.categorical_encoding != "default":
            predict_features.loc[:, self.input_features] = self.encoder.transform(predict_features.loc[:, self.input_features])
        features_to_keep = [feature for feature in predict_features.columns if feature in self.raw_features]
        return predict_features.loc[:, features_to_keep]
    
    def _prepare_train_response(self, train_features):
        """
        Prepares the train response variable

        Parameters
        ----------
        train_features: pd.DataFrame
            Dataframe containing the columns 'ds' and 'y'.
        """
        y_hat = train_features.y.copy()

        if self.detrend:
            trend_estimator = TrendEstimator()
            trend_estimator.fit(data=train_features.loc[:, ["ds", "y"]])
            trend_dataframe = trend_estimator.predict(train_features.loc[:, ["ds"]])
            y_hat -= trend_dataframe.trend.values
            self.trend_estimator = trend_estimator
        if self.response_scaling:
            y_mean = y_hat.mean()
            y_std = y_hat.std()
            y_hat -= y_mean
            y_hat /= y_std 

        self.y_mean = y_mean if "y_mean" in locals() else None
        self.y_std  = y_std if "y_std" in locals() else None
        self.target = "y_hat"
        return y_hat
    
    def _prepare_valid_response(self, valid_features):
        """
        Prepares the validation response variable

        Parameters
        ----------
        valid_features: pd.DataFrame
            dataframe containing the columns 'ds' and 'y'
        """
        y_hat = valid_features.y.copy()

        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(valid_features.loc[:, ["ds"]])
            y_hat -= trend_dataframe.trend.values
        if self.response_scaling:
            y_hat -= self.y_mean
            y_hat /= self.y_std 
            
        return y_hat

    def _prepare_features(self, train_data, valid_period=None):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            Dataframe with at least columns 'ds' and 'y'.
        valid_period: pandas.DataFrame
            Dataframe (with column 'ds') indicating the validation period.
        """
        train_features,categorical_features = self._prepare_train_features(train_data)

        if valid_period is not None:
            valid_features = self._prepare_valid_features(valid_period, train_features)
            valid_start_time = valid_features.ds.min()
            # removes validation period from train data
            train_data = train_data.query("ds < @valid_start_time")
            train_features = train_features.query("ds < @valid_start_time")
        
        train_features["y_hat"] = self._prepare_train_response(train_features)
        if valid_period is not None:
            valid_features["y_hat"] = self._prepare_valid_response(valid_features)
        
        self.train_data = train_data
        self.train_features = train_features
        self.valid_period = valid_period
        self.valid_features = valid_features if valid_period is not None else None
        self.categorical_features = categorical_features if self.categorical_encoding=="default" else list()
        return self.train_features,self.valid_features

    def fit(self, train_data, valid_period=None):
        """
        Parameters
        ----------
        train_data: pandas.DataFrame
            Dataframe with at least columns 'ds' and 'y'.
        valid_period: pandas.DataFrame
            Dataframe (with column 'ds') indicating the validation period.
        """
        self._validate_fit_inputs(train_data, valid_period)
        train_features,valid_features = self._prepare_features(train_data, valid_period)
        self.model.fit(train_features, valid_features, self.input_features, self.target, self.categorical_features)
        self.best_iteration = self.model.best_iteration

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
        predict_features = self._prepare_predict_features(predict_data)
        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(predict_data.loc[:, ["ds"]])
        else:
            trend_dataframe = None

        if "lag" in self.features or "rw" in self.features:
            prediction = self._predict(self.model, predict_features, trend_dataframe)
        else:
            prediction = self.model.predict(predict_features)
        
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

    def _predict(self, model, predict_features, trend_dataframe):
        """
        Parameters
        ----------
        model: ...
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

    def evaluate(self, eval_data, metric="rmse"):
        """
        Parameters
        ----------
        eval_data: pandas.DataFrame
            Dataframe with the same columns as 'train_data'.
        metric: string
            Possible values: 'mae', 'mape', 'mse', 'rmse', 'smape'.
        Returns
        ----------
        error: float
            error of predictions according to the error measure
        """
        self._validate_evaluate_inputs(eval_data, metric)
        eval_data = eval_data.copy()
        y_real = eval_data.pop("y")
        y_pred = self.predict(eval_data)["y_pred"].values
        error_func = getattr(metrics, f"compute_{metric}")
        error = error_func(y_real, y_pred)
        return error
