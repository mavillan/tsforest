import numpy as np
import pandas as pd

import unittest

from tsforest.forecaster import LightGBMForecaster
from tsforest.utils import make_time_range

DATA_PATH = './tests/tests_data/data.csv'

class TestLightGMB(unittest.TestCase):

    def test_it_fit(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical'])
        fcaster.fit(train_data=data)
    
    def test_it_fit_with_valid_period(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical'])
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_with_lag_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical','lag'],
                                     lags=[1,2,3,4,5,6,7])
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_with_rw_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical','rw'],
                                     window_functions=['mean','median','min','max','sum'],
                                     window_sizes=[7,14,21,28])
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_predict(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        predict_data = make_time_range('2019-07-01', '2019-07-31', freq='D')

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical'])
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(predict_data)

    def test_it_fit_predict_with_lag_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        predict_data = make_time_range('2019-07-01', '2019-07-31', freq='D')

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical','lag'],
                                     lags=[1,2,3,4,5,6,7])
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(predict_data)

    def test_it_fit_predict_with_rw_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        predict_data = make_time_range('2019-07-01', '2019-07-31', freq='D')

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical','rw'],
                                     window_functions=['mean','median','min','max','sum'],
                                     window_sizes=[7,14,21,28])
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(predict_data)

    def test_it_fit_evaluate(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        eval_data = data.query('ds >= "2019-06-01"')

        fcaster = LightGBMForecaster(model_params={'num_iterations':30,
                                                   'learning_rate':0.3}, 
                                     features=['calendar', 'calendar_cyclical'])
        fcaster.fit(train_data=train_data)
        error = fcaster.evaluate(eval_data)

        assert type(error)==np.float64, \
            f'fcaster.evaluate returns {error} which is not of type numpy.float64'
