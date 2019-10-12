import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import numpy as np
import pandas as pd
from zope.interface import implementer
import lightgbm as lgb

from tsforest.config import lgbm_parameters
from tsforest.forecaster_base import ForecasterBase
from tsforest.forecaster_interface import ForecasterInterface


@implementer(ForecasterInterface)
class LightGBMForecaster(ForecasterBase):
    '''
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
    '''
    def __init__(self, model_params=dict(), features=['calendar', 'calendar_cyclical'], 
                 categorical_features=list(), calendar_anomaly=False, detrend=True, 
                 response_scaling=False, lags=None, window_sizes=None, window_functions=None):

        if lags is not None and 'lag' not in features:
            features.append('lag')
        if (window_sizes is not None and window_functions is not None) and 'rw' not in features:
            features.append('rw')

        self.model_params = model_params
        self.features = features
        self._categorical_features = categorical_features
        self.calendar_anomaly = calendar_anomaly
        self.detrend = detrend
        self.response_scaling = response_scaling
        self.lags = lags
        self.window_sizes = window_sizes
        self.window_functions = window_functions
        self.validate_inputs()

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
        features_dataframe_casted: lightgbm.basic.Dataset
            features dataframe casted to LightGBM dataframe format
        """
        dataset_params = {'data':features_dataframe.loc[:, self.input_features],
                          'categorical_feature':categorical_features,
                          'free_raw_data':False}
        if 'weight' in features_dataframe.columns:
            dataset_params['weight'] = features_dataframe.loc[:, 'weight']
        if self.target in features_dataframe.columns:
            dataset_params['label'] = features_dataframe.loc[:, self.target]
        features_dataframe_casted = lgb.Dataset(**dataset_params)
        return features_dataframe_casted     

    def fit(self, train_data, valid_period=None, early_stopping_rounds=20):
        '''
        Parameters
        ----------
        train_data : pandas.DataFrame
            dataframe with "at least" columns "ds" and "y"
        valid_period: pandas.DataFrame
            dataframe (with column "ds") indicating the validation period
        '''
        assert {"ds","y"} <= set(train_data.columns.values), \
            '"train_data" must contain columns "ds" and "y"'
        train_features,categorical_features = super()._prepare_train_features(train_data)

        if valid_period is not None:
            valid_features = super()._prepare_valid_features(valid_period, train_features)
            valid_start_time = valid_features.ds.min()
            # removes validation period from train data
            train_data = train_data.query('ds < @valid_start_time')
            train_features = train_features.query('ds < @valid_start_time')
        
        train_features['y_hat'] = super()._prepare_train_response(train_features)
        if valid_period is not None:
            valid_features['y_hat'] = super()._prepare_valid_response(valid_features)
        
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
        model_params = {**lgbm_parameters, **self.model_params}

        training_params = {'params':model_params,
                           'train_set':train_features_casted}
        if valid_period is not None:
            training_params['valid_sets'] = valid_features_casted
            training_params['early_stopping_rounds'] = early_stopping_rounds
            training_params['verbose_eval'] = False

        # model training
        model = lgb.train(**training_params)
        self.model = model
        self.best_iteration = model.best_iteration if model.best_iteration>0 else model.num_trees()
    
    def _predict(self, model, test_features, trend_dataframe):
        """
        Parameters
        ----------
        model: LightGBM 
            Trained model
        test_features: pandas.DataFrame
            datafame containing the features for the test period
        trend_dataframe: pandas.DataFrame
            dataframe containing the trend estimation over the test period
        """
        y_train = self.train_features.y.values
        y_valid = self.valid_features.y.values \
                  if self.valid_period is not None else np.array([])
        y = np.concatenate([y_train, y_valid])

        prediction = list()
        for idx in range(test_features.shape[0]):
            if 'lag' in self.features:
                for lag in self.lags:
                    test_features.loc[idx, f'lag_{lag}'] = y[-lag]
            if 'rw' in self.features:
                for window_func in self.window_functions:
                    for window in self.window_sizes:
                        test_features.loc[idx, f'{window_func}_{window}'] = getattr(np, window_func)(y[-window:])
            y_pred = model.predict(test_features.loc[[idx], self.input_features])
            prediction.append(y_pred.copy())
            if self.response_scaling:
                y_pred *= self.y_std
                y_pred += self.y_mean
            if self.detrend:
                y_pred += trend_dataframe.loc[idx, 'trend']
            y = np.append(y, [y_pred])
        return np.asarray(prediction).ravel()

    def predict(self, test_period):
        '''
        Parameters
        ----------
        test_period: pandas.DataFrame
            dataframe with the same columns as "train_data" except for "y"
        Returns
        ----------
        prediction_dataframe: pandas.DataFrame
            dataframe containing dates "ds" and predictions "y_pred"
        '''
        assert set(self.train_data.columns) - set(test_period.columns) == {'y'}, \
            '"test_period" must have the same columns as "train_data" except for "y"'
        
        test_features = super()._prepare_test_features(test_period)
        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(test_period.loc[:, ['ds']])
        else:
            trend_dataframe = None

        if 'lag' in self.features or 'rw' in self.features:
            prediction = self._predict(self.model, test_features, trend_dataframe)
        else:
            prediction = self.model.predict(test_features.loc[:, self.input_features])
        
        if self.response_scaling:
            prediction *= self.y_std
            prediction += self.y_mean
        if self.detrend:
            prediction += trend_dataframe.trend.values
        if 'zero_response' in test_features.columns:
            zero_response_mask = test_features['zero_response']==1
            prediction[zero_response_mask] = 0

        self.test_features = test_features
            
        prediction_dataframe = pd.DataFrame({'ds':test_period.ds, 'y_pred':prediction})
        return prediction_dataframe

    def show_variable_importance(self):
        pass

    def save_model(self):
        pass