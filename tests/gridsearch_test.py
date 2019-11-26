import numpy as np
import pandas as pd
import unittest

from tsforest.forecaster import LightGBMForecaster
from tsforest.forecaster import H2OGBMForecaster
from tsforest.forecaster import CatBoostForecaster
from tsforest.grid_search import GridSearch
from tsforest.utils import make_time_range

DATA_PATH = './tests/tests_data/data_single_ts.csv'

class TestGridSearch(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        self.train_data = data.query("ds <= '2019-04-30'")
        self.valid_index = data.query("'2019-04-01' <= ds <= '2019-04-30'").index
        self.eval_data = data.query("'2019-05-01' <= ds <= '2019-05-31'")
        self.predict_data = self.eval_data.drop("y", axis=1)

    def test_it_fit_with_valid_index_in_lightgbm(self):
        hyperparams = {'num_leaves':[25,30]}
        hyperparams_fixed = {'num_iterations':30, 'learning_rate':0.3}
  
        gs = GridSearch(model_class=LightGBMForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, valid_index=self.valid_index)

    def test_it_fit_with_eval_data_in_lightgbm(self):
        hyperparams = {'num_leaves':[25,30]}
        hyperparams_fixed = {'num_iterations':30, 'learning_rate':0.3}
        
        gs = GridSearch(model_class=LightGBMForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, eval_data=self.eval_data)

    def test_it_fit_with_valid_index_in_h2ogbm(self):
        hyperparams = {'max_depth':[5,7]}
        hyperparams_fixed = {'ntrees':30, 'learn_rate':0.3}

        gs = GridSearch(model_class=H2OGBMForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, valid_index=self.valid_index)

    def test_it_fit_with_eval_data_in_h2ogbm(self):
        hyperparams = {'max_depth':[5,7]}
        hyperparams_fixed = {'ntrees':30, 'learn_rate':0.3}

        gs = GridSearch(model_class=H2OGBMForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, eval_data=self.eval_data)

    def test_it_fit_with_valid_index_in_catboost(self):
        hyperparams = {'depth':[5,6]}
        hyperparams_fixed = {'iterations':30, 'learning_rate':0.3}
  
        gs = GridSearch(model_class=CatBoostForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, valid_index=self.valid_index)

    def test_it_fit_with_eval_data_in_catboost(self):
        hyperparams = {'depth':[5,6]}
        hyperparams_fixed = {'iterations':30, 'learning_rate':0.3}
        
        gs = GridSearch(model_class=CatBoostForecaster,
                feature_sets=['calendar', 'calendar_cyclical'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(train_data=self.train_data, eval_data=self.eval_data)
