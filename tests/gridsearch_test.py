import numpy as np
import pandas as pd
import unittest

from tsforest.forecaster_lightgbm import LightGBMForecaster
from tsforest.forecaster_h2ogbm import H2OGBMForecaster
from tsforest.forecaster_catboost import CatBoostForecaster
from tsforest.grid_search import GridSearch
from tsforest.utils import make_time_range

DATA_PATH = './tests/tests_data/data.csv'

class TestGridSearch(unittest.TestCase):

    def test_it_fit_with_valid_period_in_lightgbm(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        valid_period = make_time_range('2019-05-01', '2019-05-31', freq='D')
        hyperparams = {'num_leaves':[25,30]}
        hyperparams_fixed = {'num_iterations':30, 'learning_rate':0.3}
  
        gs = GridSearch(model_class=LightGBMForecaster,
                features=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=train_data, valid_period=valid_period)

    def test_it_fit_with_test_data_in_lightgbm(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        test_data = data.query('ds >= "2019-06-01"')
        hyperparams = {'num_leaves':[25,30]}
        hyperparams_fixed = {'num_iterations':30, 'learning_rate':0.3}
        
        gs = GridSearch(model_class=LightGBMForecaster,
                features=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(data, test_data=test_data)

    def test_it_fit_with_valid_period_in_h2ogbm(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        valid_period = make_time_range('2019-05-01', '2019-05-31', freq='D')
        hyperparams = {'max_depth':[5,7]}
        hyperparams_fixed = {'ntrees':30, 'learn_rate':0.3}

        gs = GridSearch(model_class=H2OGBMForecaster,
                features=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=train_data, valid_period=valid_period)

    def test_it_fit_with_test_data_in_h2ogbm(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        test_data = data.query('ds >= "2019-06-01"')
        hyperparams = {'max_depth':[5,7]}
        hyperparams_fixed = {'ntrees':30, 'learn_rate':0.3}

        gs = GridSearch(model_class=H2OGBMForecaster,
                features=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=train_data, test_data=test_data)

    def test_it_fit_with_valid_period_in_catboost(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        valid_period = make_time_range('2019-05-01', '2019-05-31', freq='D')
        hyperparams = {'depth':[5,6]}
        hyperparams_fixed = {'iterations':30, 'learning_rate':0.3}
  
        gs = GridSearch(model_class=CatBoostForecaster,
                features=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=train_data, valid_period=valid_period)

    def test_it_fit_with_test_data_in_catboost(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        test_data = data.query('ds >= "2019-06-01"')
        hyperparams = {'depth':[5,6]}
        hyperparams_fixed = {'iterations':30, 'learning_rate':0.3}
        
        gs = GridSearch(model_class=CatBoostForecaster,
                features=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(data, test_data=test_data)
