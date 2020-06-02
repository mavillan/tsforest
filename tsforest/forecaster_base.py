import numpy as np
import pandas as pd
import category_encoders as ce
from inspect import getmembers, isfunction
from tsforest import metrics
from tsforest.trend import TrendModel
from tsforest.features import (compute_train_features, 
                               compute_predict_features)
from tsforest.config import (calendar_features_names,
                             calendar_cyclical_features_names)
from sklearn.base import TransformerMixin

# available time feature
AVAILABLE_TIME_FEATURES = ["year", "quarter", "month", "days_in_month",
                           "year_week", "year_day", "month_day", "week_day",
                           "hour", "minute", "second", "microsecond", "millisecond"
                           "nanosecond", "month_progress", "second_cos", "second_sin",
                           "minute_cos", "minute_sin", "hour_cos", "hour_sin", 
                           "week_day_cos", "week_day_sin", "year_day_cos", "year_day_sin",
                           "year_week_cos", "year_week_sin", "month_cos", "month_sin"]

# available methods in pandas.core.window.Rolling as of pandas 0.25.1 
AVAILABLE_RW_FUNCTIONS = ["count", "sum", "mean", "median", "var", "std", "min", 
                          "max", "corr", "cov", "skew", "kurt", "quantile"]

AVAILABLE_METRICS = [member[0].split('_')[1] for member in getmembers(metrics) 
                     if isfunction(member[1])]

AVAILABLE_ENCODERS = [member[1] for member in getmembers(ce)
                      if member[0]=="__all__"][0] + ["default"]

class ForecasterBase(object):
    """
    Parameters
    ----------
    model_params : dict
        Dictionary containing the parameters of the specific boosting model.
    time_features: list
        Time attributes to include as features.
    categorical_features: dict
        Dict with the name of the categorical feature as keys, and the name
        of the class in 'category_encoders' to be used for encoding as values.
    calendar_anomaly: list
        List of names of calendar features affected by an anomaly.
    ts_uid_columns: list
        List of columns names that are unique identifiers for time series.
    trend_models: dict
        Dictionary with ts_uid as key and TrendModel instances as value.
    target_scalers: dict
        Dictionary with ts_uid as key and sklearn.base.TransformerMixin as
        value, a trained sklearn.base.TransformerMixin object.
    lags: list
        List of integer lag values to include as features.
    window_shifts: list
        List of integer window shift values.
    window_sizes: list
        List of integer window sizes values to include as features.
    window_functions: list
        List of string names of the window functions to include as features.
    copy: bool
        If True performs a copy over the dataframes provided, if False performs
        inplace operations.
    """
    def __init__(self, model_params=dict(), time_features=[], exclude_features=[], 
                 categorical_features=dict(), calendar_anomaly=list(), ts_uid_columns=list(), 
                 trend_models=dict(), target_scalers=dict(), lags=list(), window_shifts=[1], 
                 window_sizes=list(), window_functions=list(), copy=False):
        self.model = None
        self.model_params = model_params
        self.time_features = time_features
        self.exclude_features = ["ds", "y", "y_raw", "weight", "fold_column", "trend",
                                 "zero_response", "calendar_anomaly"] + exclude_features
        self.categorical_features = categorical_features.copy()
        self.calendar_anomaly = calendar_anomaly
        self.ts_uid_columns = ts_uid_columns
        for ts_uid_column in ts_uid_columns:  
            if ts_uid_column in categorical_features: continue
            self.categorical_features[ts_uid_column] = "default"
        self.trend_models = trend_models
        self.target_scalers = target_scalers
        self.lags = lags
        self.window_shifts = window_shifts
        self.window_sizes = window_sizes
        self.window_functions = window_functions
        self.copy = copy
        self._features_already_prepared = False
        self._validate_inputs()
    
    def set_params(self, model_params):
        self.model_params = model_params
        self.model.set_params(model_params)
    
    def get_params(self):
        return self.model.get_params()
    
    def _validate_inputs(self):
        """
        Validates the inputs
        """
        if not isinstance(self.model_params, dict):
            raise TypeError("Parameter 'model_params' should be of type 'dict'.")
    
        if not isinstance(self.time_features, list):
            raise TypeError("Parameter 'time_features' should be of type 'list'.")
        else:
            if any([feature not in AVAILABLE_TIME_FEATURES for feature in self.time_features]):
                raise ValueError(f"Values in 'time_features' should by any of: {AVAILABLE_TIME_FEATURES}")
        
        if not isinstance(self.exclude_features, list):
            raise TypeError("Parameter 'exclude_features' should be of type 'list'.")
        
        if not isinstance(self.categorical_features, dict):
            raise TypeError("Parameter 'categorical_features' should be of type 'dict'.")
        else:
            if any([encoding not in AVAILABLE_ENCODERS for encoding in self.categorical_features.values()]):
                raise ValueError(f"Values in 'categorical_features' should be any of: {AVAILABLE_ENCODERS}")

        if not isinstance(self.calendar_anomaly, list):
            raise TypeError("Parameter 'calendar_anomaly' should be of type 'list'.")
        
        if not isinstance(self.ts_uid_columns, list):
            raise TypeError("Parameter 'ts_uid_columns' should be of type 'list'.")

        if not isinstance(self.trend_models, dict):
            raise TypeError("Parameter 'trend_models' should be of type 'dict'.")
        elif not all([isinstance(model, TrendModel) for model in self.trend_models.values()]):
            raise ValueError("Values in 'trend_models' shoud be instances of 'TrendModel'.")
        
        if not isinstance(self.target_scalers, dict):
            raise TypeError("Parameter 'target_scalers' should be of type 'dict'.")
        elif not all([isinstance(scaler, TransformerMixin) for scaler in self.target_scalers.values()]):
            raise ValueError("Values in 'target_scalers' shoud be instances of 'TransformerMixin'.")

        if not isinstance(self.lags, list):
            raise TypeError("Parameter 'lags' should be of type 'list'.")
        else:
            if any([type(x)!=int for x in self.lags]):
                raise ValueError("Values in 'lags' should be integers.")
            elif any([x<1 for x in self.lags]):
                raise ValueError("Values in 'lags' should be integers greater or equal to 1.")

        if not isinstance(self.window_shifts, list):
            raise TypeError("Parameter 'window_shifts' should be of type 'list'.")
        else:
            if any([type(x)!=int for x in self.window_shifts]):
                raise ValueError("Values in 'window_shifts' should be integers.")
            elif any([x<1 for x in self.window_shifts]):
                raise ValueError("Values in 'window_shifts' should be integers greater or equal to 1.")
        
        if not isinstance(self.window_sizes, list):
            raise TypeError("Parameter 'window_sizes' should be of type 'list'.")
        else:
            if any([type(x)!=int for x in self.window_sizes]):
                raise ValueError("Values in 'window_sizes' should be integers.")
            elif any([x<1 for x in self.window_sizes]):
                raise ValueError("Values in 'window_sizes' should be integers greater or equal to 1.")
        
        if not isinstance(self.window_functions, list):
            raise TypeError("Parameter 'window_functions' should be of type list.")
        else:
            if any([type(x)!=str for x in self.window_functions]):
                raise ValueError("Values in 'window_functions' should be string names.")
            elif any([x not in AVAILABLE_RW_FUNCTIONS for x in self.window_functions]):
                raise ValueError(f"Values in 'window_functions' should be any of: {AVAILABLE_RW_FUNCTIONS}.")
    
    def _validate_input_data(self, train_data, valid_index):
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("Parameter 'train_data' should be of type pandas.DataFrame.")
        elif not ({"ds", "y"} <= set(train_data.columns.values)):
            raise ValueError("'train_data' should contain columns 'ds' and 'y'.")
        
        if not isinstance(valid_index, pd.Index):
            raise TypeError("Parameter 'valid_index' should be of type 'pandas.Index'.")
        elif not (set(valid_index) <= set(train_data.index)):
            raise ValueError("Parameter 'valid_index' should only contain index values present in 'train_data.index'.") 

        if not set(self.ts_uid_columns) <= set(train_data.columns):
            raise ValueError(f"Parameter 'train_data' has missing ts_uid_columns: {set(self.ts_uid_columns)-set(train_data.columns)}.")
        
        if {"internal_ts_uid", "_internal_ts_uid"} in set(train_data.columns):
            raise ValueError("Columns 'internal_ts_uid' and '_internal_ts_uid' is reserved for internal usage and can not be in 'train_data' dataframe.")
    
    def _validate_predict_data(self, predict_data):
        if not isinstance(predict_data, pd.DataFrame):
            raise TypeError("Parameter 'predict_data' should be of type pandas.DataFrame.")
        elif not (set(self.raw_train_columns) - set(predict_data.columns) == {"y"}):
            raise ValueError("'predict_data' shoud have the same columns as 'train_data' except for 'y'.")
    
    def _validate_evaluate_data(self, eval_data, metric):
        if not isinstance(eval_data, pd.DataFrame):
            raise TypeError("'eval_data' should be of type pandas.DataFrame.")
        elif not (set(self.raw_train_columns) == set(eval_data.columns)):
            raise ValueError("'eval_data' should have the same columns as 'train_data'.")

        if not isinstance(metric, str):
            raise TypeError("'metric' should be of type str.")
        elif metric not in AVAILABLE_METRICS:
            raise ValueError(f"'metric' should be any of these: {AVAILABLE_METRICS}")
    
    def _encode_categorical_features(self, train_features):
        categorical_encoders = dict()
        for feature,encoding in self.categorical_features.items():
            if encoding == "default": 
                if not np.issubdtype(train_features[feature].dtype, np.number):
                    encoding = "OrdinalEncoder"
                else: continue
            encoder_class = getattr(ce, encoding)
            encoder = encoder_class(cols=[feature])
            encoder.fit(train_features.loc[:, [feature]], train_features.loc[:, "y_raw"].values)
            transformed = encoder.transform(train_features.loc[:, [feature]])
            if feature in self.ts_uid_columns:
                train_features["_"+feature] = transformed.values
                self.exclude_features.append(feature)
            else:
                del train_features[feature]
                train_features[feature] = transformed.values
            categorical_encoders[feature] = encoder
        return train_features,categorical_encoders
    
    def _prepare_predict_features(self, predict_data):
        """
        Parameters
        ----------
        predict_data: pandas.DataFrame
            Dataframe with the same columns as train_data (except for 'y') 
            containing the prediction period.
        Returns
        ----------
        predict_features: pandas.DataFrame
            Dataframe containing all the features for making predictions with 
            the trained model.
        """
        predict_features = compute_predict_features(data=predict_data,
                                                    ts_uid_columns=self.ts_uid_columns,
                                                    time_features=self.time_features,
                                                    lags=self.lags,
                                                    window_shifts=self.window_shifts,
                                                    window_sizes=self.window_sizes,
                                                    window_functions=self.window_functions,
                                                    ignore_const_cols=False)

        if "calendar_anomaly" in predict_features.columns:
            assert len(self.calendar_anomaly) != 0, \
                "'calendar_anomaly' column found, but no names of affected features were provided."
            assert set(self.calendar_anomaly) <= set(self.train_features.columns), \
                f"Calendar anomaly affected columns: {set(self.calendar_anomaly)-set(self.train_features.columns)} are not present in 'train_features'."
            idx = predict_features.query("calendar_anomaly == 1").index
            predict_features.loc[idx, self.calendar_anomaly] = np.nan
        if len(self.categorical_encoders) > 0:
            for feature,encoder in self.categorical_encoders.items():
                transformed = encoder.transform(predict_features.loc[:, [feature]])
                if feature in self.ts_uid_columns:
                    predict_features["_"+feature] = transformed.values
                else:
                    del predict_features[feature]
                    predict_features[feature] = transformed.values
        features_to_keep = [feature for feature in predict_features.columns if feature in self.raw_features]
        return predict_features.loc[:, features_to_keep]
    
    def prepare_target(self, train_data):
        """
        Prepares the target variable

        Parameters
        ----------
        train_data: pd.DataFrame
            Dataframe containing the columns 'ds' and 'y'.
        """
        train_data["y_raw"] = train_data["y"].copy()
        ts_uid_values = train_data.loc[:, self.ts_uid_columns].drop_duplicates()
        all_dataframes = list()
        for _,row in ts_uid_values.iterrows():
            key = tuple([item for _,item in row.iteritems()])
            query_string = " & ".join([f"{col_name}=={value}" for col_name,value in row.iteritems()])
            train_data_slice = train_data.query(query_string).copy()

            trend_model = self.trend_models[key] if len(self.trend_models) > 0 else None
            target_scaler = self.target_scalers[key] if len(self.target_scalers) > 0 else None
            if trend_model is not None:
                trend_dataframe = trend_model.predict(train_data_slice.loc[:, ["ds"]]) 
                train_data_slice["trend"] = trend_dataframe.trend.values
                train_data_slice.loc[:, "y"] -= trend_dataframe.trend.values
            if target_scaler is not None:
                train_data_slice.loc[:, "y"] = target_scaler.transform(train_data_slice.y.values.reshape(-1,1)).ravel()
            all_dataframes.append(train_data_slice)
        
        return pd.concat(all_dataframes)

    def prepare_train_features(self, train_data):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            Dataframe with at least columns 'ds' and 'y'.
        """
        train_features = compute_train_features(data=train_data,
                                                ts_uid_columns=self.ts_uid_columns,
                                                time_features=self.time_features,
                                                lags=self.lags,
                                                window_shifts=self.window_shifts,
                                                window_sizes=self.window_sizes,
                                                window_functions=self.window_functions,
                                                ignore_const_cols=True)
        if "zero_response" in train_features.columns:
            train_features = train_features.query("zero_response != 1")
        if "calendar_anomaly" in train_features.columns:
            assert len(self.calendar_anomaly) != 0, \
                "'calendar_anomaly' column found, but names of affected features were not provided."
            assert set(self.calendar_anomaly) <= set(train_features.columns), \
                f"Calendar anomaly affected columns: {set(self.calendar_anomaly)-set(train_features.columns)} are not present in 'train_features'."
            idx = train_features.query("calendar_anomaly == 1").index
            train_features.loc[idx, self.calendar_anomaly] = np.nan
        return train_features
    
    def prepare_features(self, train_data, valid_index=pd.Int64Index([])):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            Dataframe with at least columns 'ds' and 'y'.
        valid_index: pandas.Index
            Array with indexes from train_data to be used for validation.
        """
        self._validate_input_data(train_data, valid_index)
        self.raw_train_columns = list(train_data.columns)
        if self.copy:
            train_data = train_data.copy(deep=True)
        if len(self.ts_uid_columns) == 0:
            # adds a dummy '_internal_ts_uid' column in case of single ts data
            train_data["_internal_ts_uid"] = 0
            self.ts_uid_columns = ["_internal_ts_uid"]
        train_data = train_data.sort_values(self.ts_uid_columns+["ds"], axis=0)
        
        if len(self.trend_models) > 0 or len(self.target_scalers) > 0:
            train_data = self.prepare_target(train_data)
        else:
            train_data["y_raw"] = train_data["y"].values
        train_features = self.prepare_train_features(train_data)
        # needed to keep track of the rows used in valid_index
        train_features.set_index(train_data.index, inplace=True)

        if len(valid_index) > 0:
            valid_features = train_features.loc[valid_index, :]
            train_features = train_features.drop(valid_index, axis=0)
 
        # performs the encoding of categorical features
        if len(self.categorical_features) > 0:
            train_features,categorical_encoders = self._encode_categorical_features(train_features)
            if len(valid_index) > 0:
                for feature,encoder in categorical_encoders.items():
                    transformed = encoder.transform(valid_features.loc[:, [feature]])
                    if feature in self.ts_uid_columns:
                        valid_features["_"+feature] = transformed.values
                    else:
                        del train_features[feature]
                        valid_features[feature] = transformed.values
            self.categorical_encoders = categorical_encoders
        else:
            self.categorical_encoders = dict()
        # categorical features to be encoded by the tree/boosting model
        _categorical_features = [feature for feature,encoder in self.categorical_features.items() 
                                 if encoder == "default"]
        self._categorical_features = _categorical_features

        self.raw_features = train_features.columns
        self.input_features = [feature for feature in train_features.columns
                               if feature not in self.exclude_features]
        self.train_data = train_data
        self.train_features = train_features
        self.valid_features = valid_features if len(valid_index) > 0 else None
        self._features_already_prepared = True
        return self.train_features, self.valid_features

    def fit(self, train_data=None, valid_index=pd.Int64Index([]), fit_kwargs=dict()):
        """
        Parameters
        ----------
        train_data: pandas.DataFrame
            Dataframe with at least columns 'ds' and 'y'.
        valid_index: pandas.Index
            Array with indexes from train_data to be used for validation.
        fit_kwargs: dict
            Extra arguments passed to the fit/train call of the model. 
        """
        if not self._features_already_prepared:
            train_features,valid_features = self.prepare_features(train_data, valid_index)
        else:
            train_features = self.train_features
            valid_features = self.valid_features
        kwargs = {"train_features":train_features, 
                  "valid_features":valid_features,
                  "input_features":self.input_features,
                  "categorical_features":self._categorical_features,
                  "target":"y",
                  "fit_kwargs":fit_kwargs}
        self.model.fit(**kwargs)
        self.best_iteration = self.model.best_iteration

    def predict(self, predict_data, recursive=False, return_trend=False, bias_corr_func=None):
        """
        Parameters
        ----------
        predict_data: pandas.DataFrame
            Datafame containing the features for the prediction period.
            Contains the same columns as 'train_data' except for 'y'.
        recursive: boolean
            If True, performs recursive one-step-ahead predicion for the
            lag and/or rolling window features.
        return_trend: boolean
            If True, the returning dataframe will contain the trend 
            estimation for each time series.
        bias_corr_func: function
            Function to perform bias correction on recursive prediction. 
            It receives a 1-dimensional array 'x' an return an 1-dimensional 
            array of the same lenght. 
        Returns
        ----------
        prediction_dataframe: pandas.DataFrame
            Dataframe containing the dates 'ds' and predictions 'y_pred'.
        """
        self._validate_predict_data(predict_data)
        if self.copy:
            predict_data = predict_data.copy(deep=True)
        if set(self.ts_uid_columns) == {"_internal_ts_uid"}:
            # adds a dummy '_internal_ts_uid' column in case of single ts data
            predict_data["_internal_ts_uid"] = 0
        
        ts_uid_in_predict = predict_data.loc[:, self.ts_uid_columns].drop_duplicates()
        ts_uid_in_train = self.train_data.loc[:, self.ts_uid_columns].drop_duplicates()
        missing_ts_uid = (pd.merge(ts_uid_in_predict, ts_uid_in_train, how="left", indicator=True)
                          .query("_merge == 'left_only'"))
        assert len(missing_ts_uid) == 0, \
            "There are ts_uid values in 'predict_features' not present in training data."

        predict_features = self._prepare_predict_features(predict_data)
        if not recursive:
            prediction = self.model.predict(predict_features)
            prediction_dataframe = (predict_data
                                    .loc[:, ["ds"]+self.ts_uid_columns]
                                    .assign(y_pred = prediction))
        else:
            _prediction_dataframe = self.recursive_predict(predict_features, bias_corr_func)
            prediction_dataframe = pd.merge(predict_data.loc[:, ["ds"]+self.ts_uid_columns],
                                            _prediction_dataframe, 
                                            how="left", on=["ds"]+self.ts_uid_columns)
        
        if len(self.trend_models) > 0 or len(self.target_scalers) > 0:
            if len(self.trend_models) > 0 and return_trend:
                prediction_dataframe["trend"] = None
            ts_uid_values = prediction_dataframe.loc[:, self.ts_uid_columns].drop_duplicates()
            for _,row in ts_uid_values.iterrows():
                key = tuple([item for _,item in row.iteritems()])
                query_string = " & ".join([f"{col_name}=={value}" for col_name,value in row.iteritems()])
                slice_idx = prediction_dataframe.query(query_string).index
                
                if len(self.target_scalers) > 0:
                    target_scaler = self.target_scalers[key]
                    target_scaled = prediction_dataframe.loc[slice_idx,"y_pred"].values.reshape(-1,1)
                    target_unscaled = target_scaler.inverse_transform(target_scaled).ravel()
                    prediction_dataframe.loc[slice_idx, "y_pred"] = target_unscaled
                if len(self.trend_models) > 0:
                    trend_model = self.trend_models[key]
                    trend_dataframe = trend_model.predict(prediction_dataframe.loc[slice_idx, ["ds"]])
                    if return_trend:
                        prediction_dataframe.loc[slice_idx, "trend"] = trend_dataframe.trend.values
                    prediction_dataframe.loc[slice_idx, "y_pred"] += trend_dataframe.trend.values

        if "zero_response" in predict_features.columns:
            zero_response_idx = predict_features.query("zero_response == 1").index
            prediction_dataframe.loc[zero_response_idx, "y_pred"] = 0

        if "_internal_ts_uid" in prediction_dataframe.columns:
            prediction_dataframe.drop("_internal_ts_uid", axis=1, inplace=True)

        self.predict_features = predict_features
        return prediction_dataframe

    def recursive_predict(self, predict_features, bias_corr_func):
        min_predict_time = predict_features.ds.min()
        max_predict_time = predict_features.ds.max()
        max_train_time = self.train_data.ds.max()
        time_freq = (self.train_data.loc[:, ["ds"]]
                     .drop_duplicates()
                     .sort_values("ds")
                     .ds.diff(1).median())
        assert (min_predict_time - max_train_time) <= time_freq, \
            "predict time period cannot have time gaps from last time-step in training data."

        ts_uid_in_predict = predict_features.loc[:, self.ts_uid_columns].drop_duplicates()
        max_offset = max(0 if len(self.lags)==0 else max(self.lags), \
                         0 if len(self.window_sizes)==0 else max(self.window_sizes)+max(self.window_shifts))
        max_offset_time = min_predict_time - pd.DateOffset(max_offset+1)
        train_temp = (self.train_data
                      .loc[:, self.ts_uid_columns+["ds","y"]]
                      .merge(ts_uid_in_predict, how="inner")
                      .query("@max_offset_time <= ds < @min_predict_time")
                      .copy(deep=True))

        # todo: raise warning for missing ts_uid
        predict_features = predict_features.copy(deep=True)
        predict_features.sort_values(["ds"] + self.ts_uid_columns, axis=0, inplace=True)
        predict_features.set_index(["ds"] + self.ts_uid_columns, drop=False, inplace=True)

        for time_step in np.sort(predict_features.ds.unique()):

            for lag in self.lags: 
                lag_values = train_temp.groupby(self.ts_uid_columns)["y"].apply(lambda x: x.iloc[-lag])
                predict_features.loc[time_step].loc[lag_values.index, f"lag{lag}"] = lag_values.values
            
            for window_shift in self.window_shifts:
                for window_func in self.window_functions:
                    for window in self.window_sizes:
                        lidx = -(window + window_shift-1)
                        ridx = -(window_shift-1) if window_shift > 1 else None
                        rw_values = (train_temp.groupby(self.ts_uid_columns)["y"]
                                     .apply(lambda x: getattr(np, window_func)(x.iloc[lidx:ridx])))
                        feature_name = f"{window_func}{window}_shift{window_shift}"
                        predict_features.loc[time_step].loc[rw_values.index, feature_name] = rw_values.values
            
            _prediction = self.model.predict(predict_features.loc[time_step])
            if bias_corr_func is not None: 
                _prediction = bias_corr_func(_prediction)

            _prediction_dataframe = predict_features.loc[time_step, ["ds"]+self.ts_uid_columns].reset_index(drop=True)
            _prediction_dataframe["y"] = _prediction
            train_temp = pd.concat([train_temp, _prediction_dataframe], axis=0, ignore_index=True)

        prediction_dataframe = train_temp.query("@min_predict_time <= ds <= @max_predict_time")
        prediction_dataframe.rename({"y":"y_pred"}, axis=1, inplace=True)
        return prediction_dataframe 

    def evaluate(self, eval_data, metric="rmse", recursive=False):
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
        self._validate_evaluate_data(eval_data, metric)
        eval_data = eval_data.copy()
        y_real = eval_data.pop("y")
        y_pred = self.predict(eval_data, recursive)["y_pred"].values
        error_func = getattr(metrics, f"compute_{metric}")
        error = error_func(y_real, y_pred)
        return error

    def save_model(self, fname, **kwargs):
        if self.model is not None:
            self.model.save_model(fname, **kwargs)

    def load_model(self, fname, **kwargs):
        self.model.load_model(fname, **kwargs)
