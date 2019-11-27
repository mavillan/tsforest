import numpy as np
import pandas as pd
import unittest

from tsforest.forecaster import (LightGBMForecaster, 
                                 H2OGBMForecaster,
                                 CatBoostForecaster,
                                 XGBoostForecaster)
from tsforest.grid_search import GridSearch
from tsforest.utils import make_time_range

DATA_PATH = './tests/tests_data/data_single_ts.csv'

class TestGridSearch(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        self.train_data = data.query("ds <= '2019-04-30'")
        self.valid_index1 = data.query("'2019-04-01' <= ds <= '2019-04-30'").index
        self.valid_index2 = data.query("'2019-03-01' <= ds <= '2019-04-30'").index

    def test_it_fit_with_valid_indexes_in_lightgbm(self):
        hyperparams = {'num_leaves':[25,30]}
        hyperparams_fixed = {'num_iterations':30, 'learning_rate':0.3}
  
        gs = GridSearch(model_class=LightGBMForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, valid_indexes=[self.valid_index1, self.valid_index2])

    def test_it_fit_with_valid_indexes_in_h2ogbm(self):
        hyperparams = {'max_depth':[5,7]}
        hyperparams_fixed = {'ntrees':30, 'learn_rate':0.3}

        gs = GridSearch(model_class=H2OGBMForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, valid_indexes=[self.valid_index1, self.valid_index2])

    def test_it_fit_with_valid_indexes_in_catboost(self):
        hyperparams = {'depth':[5,6]}
        hyperparams_fixed = {'iterations':30, 'learning_rate':0.3}
  
        gs = GridSearch(model_class=CatBoostForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, valid_indexes=[self.valid_index1, self.valid_index2])

    def test_it_fit_with_valid_indexes_in_xgboost(self):
        hyperparams = {'max_depth':[5,6]}
        hyperparams_fixed = {'num_boost_round':30, 'learning_rate':0.3}
  
        gs = GridSearch(model_class=XGBoostForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, valid_indexes=[self.valid_index1, self.valid_index2])
