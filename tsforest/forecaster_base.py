import numpy as np
import pandas as pd
from inspect import getmembers, isfunction

from tsforest.features import FeaturesGenerator
from tsforest.trend import TrendEstimator
from tsforest import metrics

# available methods in pandas.core.window.Rolling as of pandas 0.25.1 
AVAILABLE_RW_FUNCTIONS = ['count', 'sum', 'mean', 'median', 'var', 'std', 'min', 
                          'max', 'corr', 'cov', 'skew', 'kurt', 'quantile']

AVAILABLE_METRICS = [member[0].split('_')[1] for member in getmembers(metrics) 
                     if isfunction(member[1])]

class ForecasterBase(object):

    def _validate_inputs(self):
        '''
        Validates the inputs
        '''
        if not isinstance(self.model_params, dict):
            raise TypeError("Parameter 'model_params' should be of type 'dict'.")

        if not isinstance(self.features, list):
            raise TypeError("Parameter 'features' should be of type 'list'.")
        else:
            if any([x not in ['calendar','calendar_cyclical', 'lag', 'rw'] for x in self.features]):
                raise ValueError("Values in 'features' should be any of: ['calendar', 'calendar_cyclical', 'lag', 'rw'].")
        
        if not isinstance(self._categorical_features, list):
            raise TypeError("Parameter 'categorical_features' should be of type 'list'.")

        if not isinstance(self.calendar_anomaly, bool):
            raise TypeError("Parameter 'calendar_anomaly' should be of type 'bool'.")

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
                raise TypeError("Parameter 'window_sizes' should be of type 'list.")
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
        elif not ({"ds","y"} <= set(train_data.columns.values)):
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
      
    def _prepare_train_features(self, train_data):
        '''
        Parameters
        ----------
        train_data : pandas.DataFrame
            dataframe with at least columns "ds" and "y"
        '''
        features_generator = FeaturesGenerator(include_features=self.features,
                                               lags=self.lags,
                                               window_sizes=self.window_sizes,
                                               window_functions=self.window_functions)
        train_features,categorical_features = features_generator.compute_train_features(train_data)
        categorical_features = categorical_features + self._categorical_features
        if 'zero_response' in train_features.columns:
            train_features = train_features.query('zero_response != 1')
        if 'calendar_anomaly' in train_features.columns:
            assert self.calendar_anomaly is not None, \
                '"calendar_anomaly" column found, but no names of affected features were provided'
            idx = train_features.query('calendar_anomaly == 1').index
            train_features.loc[idx, self.calendar_anomaly] = np.nan

        exclude_features = ['ds', 'y', 'y_hat', 'month_day', 'weight', 
                            'fold_column', 'zero_response', 'calendar_anomaly']
        self.input_features = [feature for feature in train_features.columns
                               if feature not in exclude_features]
        self.features_generator = features_generator
        return train_features,categorical_features
    
    def _prepare_valid_features(self, valid_period, train_features):
        '''
        valid_period : pandas.DataFrame
            dataframe with column "ds" indicating the validation period
        train_features: pandas.DataFrame
            dataframe containing the training features
        '''
        valid_features = pd.merge(valid_period, train_features, how='inner', on=['ds'])
        assert len(valid_features) > 0, \
            'none of the dates in valid_period are in train_features'
        return valid_features

    def _prepare_predict_features(self, predict_data):
        '''
        Parameters
        ----------
        predict_data: pandas.DataFrame
            Dataframe with the same columns as self.train_data (except for "y") 
            containing the prediction period.
        Returns
        ----------
        predict_features: pandas.DataFrame
            Dataframe containing all the features for making predictions with 
            the trained model.
        '''
        predict_features,_ = self.features_generator.compute_predict_features(predict_data)
        if 'calendar_anomaly' in predict_features.columns:
            assert self.calendar_anomaly is not None, \
                '"calendar_anomaly" column found, but no names of affected features were provided'
            idx = predict_features.query('calendar_anomaly == 1').index
            predict_features.loc[idx, self.calendar_anomaly] = np.nan
        return predict_features
    
    def _prepare_train_response(self, train_features):
        '''
        Prepares the train response variable

        Parameters
        ----------
        train_features: pd.DataFrame
            dataframe containing the columns "ds" and "y"
        '''
        y_hat = train_features.y.copy()

        if self.detrend:
            trend_estimator = TrendEstimator()
            trend_estimator.fit(data=train_features.loc[:, ['ds','y']])
            trend_dataframe = trend_estimator.predict(train_features.loc[:, ['ds']])
            y_hat -= trend_dataframe.trend.values
            self.trend_estimator = trend_estimator
        if self.response_scaling:
            y_mean = y_hat.mean()
            y_std = y_hat.std()
            y_hat -= y_mean
            y_hat /= y_std 

        self.y_mean = y_mean if 'y_mean' in locals() else None
        self.y_std  = y_std if 'y_std' in locals() else None
        self.target = 'y_hat'
        return y_hat
    
    def _prepare_valid_response(self, valid_features):
        '''
        Prepares the validation response variable

        Parameters
        ----------
        valid_features: pd.DataFrame
            dataframe containing the columns "ds" and "y"
        '''
        y_hat = valid_features.y.copy()

        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(valid_features.loc[:, ['ds']])
            y_hat -= trend_dataframe.trend.values
        if self.response_scaling:
            y_hat -= self.y_mean
            y_hat /= self.y_std 
            
        return y_hat

    def evaluate(self, eval_data, metric='rmse'):
        '''
        Parameters
        ----------
        eval_data: pandas.DataFrame
            dataframe with the same columns as "train_data"
        metric: string
            possible values: "mae", "mape", "mse", "rmse", "smape"
        Returns
        ----------
        error: float
            error of predictions according to the error measure
        '''
        self._validate_evaluate_inputs(eval_data, metric)
        eval_data = eval_data.copy()
        y_real = eval_data.pop("y")
        y_pred = self.predict(eval_data)["y_pred"].values
        error_func = getattr(metrics, f'compute_{metric}')
        error = error_func(y_real, y_pred)
        return error
