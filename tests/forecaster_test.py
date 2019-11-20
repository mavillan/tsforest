import numpy as np
import pandas as pd
import unittest
from parameterized import parameterized_class
from tsforest.forecaster import (CatBoostForecaster,
                                 LightGBMForecaster,
                                 XGBoostForecaster,
                                 H2OGBMForecaster)
from tsforest.utils import make_time_range

def get_default_model_params(model_class):
    if model_class == CatBoostForecaster:
        return {'iterations':30,'learning_rate':0.3}
    elif model_class == LightGBMForecaster:
        return {'num_iterations':30, 'learning_rate':0.3}
    elif model_class == XGBoostForecaster:
        return {'num_boost_round':30, 'learning_rate':0.3}
    elif model_class == H2OGBMForecaster:
        return {'ntrees':30,'learn_rate':0.3}

TEST_DATA = ['./tests/tests_data/data_single_ts.csv',
             './tests/tests_data/data_many_ts.csv']
TEST_MODELS = [CatBoostForecaster,
               LightGBMForecaster,
               XGBoostForecaster,
               H2OGBMForecaster]

@parameterized_class([
    {"data_path":data_path, "model_class":model_class}
    for model_class in TEST_MODELS
    for data_path in TEST_DATA
])
class TestForecaster(unittest.TestCase):

    def test_it_fit(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical']}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=data)
    
    def test_it_fit_with_valid_period(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical']}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_with_lag_features(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical', 'lag'],
                        "lags":[1,2,3,4,5,6,7]}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_with_rw_features(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        valid_period = make_time_range('2019-06-01', '2019-06-30', freq='D')

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical', 'rw'],
                        "window_functions":['mean','median','min','max','sum'],
                        "window_sizes":[7,14,21,28]}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=data, valid_period=valid_period)

    def test_it_fit_predict(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        predict_data = make_time_range('2019-07-01', '2019-07-31', freq='D')
        if "ts_uid" in data.columns:
            predict_data = (pd.concat([predict_data.assign(ts_uid=1),
                                       predict_data.assign(ts_uid=2)])
                            .reset_index(drop=True))

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical']}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(predict_data)

    def test_it_fit_predict_with_lag_features(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        predict_data = make_time_range('2019-07-01', '2019-07-31', freq='D')
        if "ts_uid" in data.columns:
            predict_data = (pd.concat([predict_data.assign(ts_uid=1),
                                       predict_data.assign(ts_uid=2)])
                            .reset_index(drop=True))

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical', 'lag'],
                        "lags":[1,2,3,4,5,6,7]}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(predict_data)

    def test_it_fit_predict_with_rw_features(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        predict_data = make_time_range('2019-07-01', '2019-07-31', freq='D')
        if "ts_uid" in data.columns:
            predict_data = (pd.concat([predict_data.assign(ts_uid=1),
                                       predict_data.assign(ts_uid=2)])
                            .reset_index(drop=True))

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical', 'rw'],
                        "window_functions":['mean','median','min','max','sum'],
                        "window_sizes":[7,14,21,28]}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=data)
        prediction_dataframe = fcaster.predict(predict_data)

    def test_it_fit_evaluate(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        eval_data = data.query('ds >= "2019-06-01"')

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical']}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=train_data)
        error = fcaster.evaluate(eval_data)

        assert type(error)==np.float64, \
            f'fcaster.evaluate returns {error} which is not of type numpy.float64'
    
    def test_it_fit_evaluate_with_bounded_error(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        train_data = data.query('ds < "2019-06-01"')
        eval_data = data.query('ds >= "2019-06-01"')

        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "feature_sets":['calendar', 'calendar_cyclical']}
        if "ts_uid" in data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=train_data)
        error = fcaster.evaluate(eval_data)

        assert error <= 2, \
            f"fcaster.evaluate returns error=={error} which is greater than 2."
