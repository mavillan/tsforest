import numpy as np
import pandas as pd
import unittest

from tsforest.forecaster_lightgbm import LGBMForecaster
from tsforest.forecaster_gbm import GBMForecaster
from tsforest.grid_search import GridSearch

DATA_PATH = './tests/tests_data/data.csv'

class TestGridSearch(unittest.TestCase):

    def test_it_fit_with_valid_period_in_lightgbm(self):
        hyperparams = {'num_leaves':[25,30]}
        hyperparams_fixed = {'num_iterations':100, 'learning_rate':0.2}
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = data.tail(30).loc[:, ['ds']]
        
        gs = GridSearch(model_class=LGBMForecaster,
                features=['calendar_mixed'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(data, valid_period)

    def test_it_fit_with_valid_period_in_gbm(self):
        hyperparams = {'max_depth':[5,7]}
        hyperparams_fixed = {'ntrees':100, 'learn_rate':0.2}
        data = pd.read_csv(DATA_PATH, parse_dates=['ds'])
        valid_period = data.tail(30).loc[:, ['ds']]
        
        gs = GridSearch(model_class=GBMForecaster,
                features=['calendar_mixed'], 
                hyperparams=hyperparams,
                hyperparams_fixed=hyperparams_fixed,
                n_jobs=-1)
        gs.fit(data, valid_period)