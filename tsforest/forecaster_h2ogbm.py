import numpy as np
import pandas as pd
from zope.interface import implementer
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
# init the cluster if is not already up
if h2o.cluster() is None: h2o.init(nthreads=-1)

from tsforest.config import gbm_parameters
from tsforest.forecaster_base import ForecasterBase
from tsforest.forecaster_interface import ForecasterInterface

@implementer(ForecasterInterface)
class H2OGBMForecaster(ForecasterBase):
    '''
    Parameters
    ----------
    model_params : dict
        Dictionary containing the specific parameters of the boosting model 
    features: list
        List of features to be included
    categorical_features: list
        List of names of categorical features
    calendar_anomaly: list
        List of names of calendar features affected by an anomaly 
    detrend: bool
        Whether or not to remove the trend from time serie
    response_scaling:
        Whether or not to perform scaling of the reponse variable
    lags: list
        List of integer lag values
    window_sizes: list
        List of integer window sizes values
    window_functions: list
        List of string names of the window functions
    '''
    def __init__(self, model_params=dict(), features=['calendar', 'calendar_cylical'], 
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
        features_dataframe_casted: h2o.H2OFrame
            features dataframe casted to H2O dataframe format
        """
        features_types = {feature:'categorical' for feature in categorical_features
                          if feature in self.input_features}
        features_dataframe_casted = h2o.H2OFrame(features_dataframe, 
                                                 column_types=features_types)
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
        model_params = {**gbm_parameters, **self.model_params}

        training_params = {'training_frame':train_features_casted, 
                           'x':self.input_features, 
                           'y':self.target}
        if valid_period is not None:
            training_params['validation_frame'] = valid_features_casted
            model_params['stopping_rounds'] = early_stopping_rounds 
        if 'weight' in self.train_features.columns:
            training_params['weights_column']='weight'

        # model training
        model = H2OGradientBoostingEstimator(**model_params)
        model.train(**training_params)
        self.model = model
        self.best_iteration = int(model.summary()["number_of_trees"][0])

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
            test_features_casted = self._cast_dataframe(test_features.loc[[idx], :], 
                                                        self.categorical_features)
            _y_pred = model.predict(test_features_casted)
            y_pred = _y_pred.as_data_frame().values[:,0]
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
            '"test_period" must have the same columns as self.train_data except for "y"'
        
        test_features = super()._prepare_test_features(test_period)
        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(test_period.loc[:, ['ds']])
        else:
            trend_dataframe = None

        if 'lag' in self.features or 'rw' in self.features:
            prediction = self._predict(self.model, test_features, trend_dataframe)
        else:
            test_features_casted = self._cast_dataframe(test_features, 
                                                        self.categorical_features)
            _prediction = self.model.predict(test_features_casted)
            prediction = _prediction.as_data_frame().values[:,0]

        if self.response_scaling:
            prediction *= self.y_std
            prediction += self.y_mean
        if self.detrend:
            prediction += trend_dataframe.trend.values
        if "zero_response" in test_features.columns:
            zero_response_mask = test_features['zero_response']==1
            prediction[zero_response_mask] = 0
        
        self.test_features = test_features
            
        prediction_dataframe = pd.DataFrame({"ds":test_period.ds, "y_pred":prediction})
        return prediction_dataframe

    def show_variable_importance(self):
        pass

    def save_model(self):
        pass