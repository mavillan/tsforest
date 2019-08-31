import numpy as np
import pandas as pd

import unittest

from tsforest.forecaster_lightgbm import LGBMForecaster
from tsforest.utils import make_time_range

DATA_PATH = './tests/tests_data/data.csv'

class TestLightGMB(unittest.TestCase):

    def test_it_fit(self):
        print("##################"*12)
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])

        fcaster = LGBMForecaster(model_params={'num_iterations':100,
                                                'learning_rate':0.2}, 
                                 features=['calendar_mixed'])
        fcaster.fit(train_data=data)
    
    def test_it_fit_with_valid_period(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        fcaster = LGBMForecaster(model_params={'num_iterations':100,
                                                'learning_rate':0.2}, 
                                 features=['calendar_mixed'])
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_with_lag_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        fcaster = LGBMForecaster(model_params={'num_iterations':100,
                                                'learning_rate':0.2}, 
                                 features=['calendar_mixed','lag'],
                                 lags=[1,2,3,4,5,6,7])
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_with_rw_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        fcaster = LGBMForecaster(model_params={'num_iterations':100,
                                                'learning_rate':0.2}, 
                                 features=['calendar_mixed','rw'],
                                 window_functions=['mean','median','min','max','sum'],
                                 window_sizes=[7,14,21,28])
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_predict(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        test_period = make_time_range('2019-07-01', '2019-07-31', freq='D')

        fcaster = LGBMForecaster(model_params={'num_iterations':100,
                                                'learning_rate':0.2}, 
                                 features=['calendar_mixed'])
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(test_period)

    def test_it_fit_predict_with_lag_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        test_period = make_time_range('2019-07-01', '2019-07-31', freq='D')

        fcaster = LGBMForecaster(model_params={'num_iterations':100,
                                                'learning_rate':0.2}, 
                                 features=['calendar_mixed','lag'],
                                 lags=[1,2,3,4,5,6,7])
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(test_period)

    def test_it_fit_predict_with_rw_features(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        test_period = make_time_range('2019-07-01', '2019-07-31', freq='D')

        fcaster = LGBMForecaster(model_params={'num_iterations':100,
                                                'learning_rate':0.2}, 
                                 features=['calendar_mixed','rw'],
                                 window_functions=['mean','median','min','max','sum'],
                                 window_sizes=[7,14,21,28])
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(test_period)



