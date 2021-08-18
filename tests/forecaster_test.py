import numpy as np
import pandas as pd
import unittest
from parameterized import parameterized_class
from tsforest.forecast import (CatBoostForecaster,
                               LightGBMForecaster,
                               XGBoostForecaster,
                               H2OGBMForecaster)
from tsforest.utils import make_time_range
from tsforest.trend import compute_trend_models

def get_default_model_params(model_class):
    if model_class == CatBoostForecaster:
        return {'iterations':30,'learning_rate':0.3}
    elif model_class == LightGBMForecaster:
        return {'num_iterations':30, 'learning_rate':0.3}
    elif model_class == XGBoostForecaster:
        return {'num_boost_round':30, 'learning_rate':0.3}
    elif model_class == H2OGBMForecaster:
        return {'ntrees':30,'learn_rate':0.3}

TEST_DATA = [
    './tests/tests_data/data_single_ts.csv',
    './tests/tests_data/data_many_ts.csv',
]
TEST_MODELS = [
    CatBoostForecaster,
    LightGBMForecaster,
    XGBoostForecaster,
    H2OGBMForecaster,
]

@parameterized_class([
    {"data_path":data_path, "model_class":model_class}
    for model_class in TEST_MODELS
    for data_path in TEST_DATA
])

class TestForecaster(unittest.TestCase):
    def setUp(self):
        data = pd.read_csv(self.data_path, parse_dates=['ds'])
        self.time_features = ["year", "quarter", "month", "year_week", "year_day",
                              "month_day", "week_day", "month_progress", "week_day_cos",
                              "week_day_sin", "year_day_cos", "year_day_sin", "year_week_cos",
                              "year_week_sin", "month_cos", "month_sin"]
        self.train_data = data.query("ds <= '2019-04-30'")
        self.valid_index = data.query("'2019-04-01' <= ds <= '2019-04-30'").index
        self.eval_data = data.query("'2019-05-01' <= ds <= '2019-05-31'")
        self.predict_data = self.eval_data.drop("y", axis=1)

    def test_it_fit(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
    
    def test_it_fit_with_valid_index(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data, valid_index=self.valid_index)

    def test_it_fit_with_lag_features(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features,
                        "lags":[1,2,3,4,5,6,7]}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data, valid_index=self.valid_index)

    def test_it_fit_with_rw_features(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features,
                        "window_functions":{
                            "mean": (None, [1,], [7,14,21,28]),
                            "median": (None, [1,], [7,14,21,28]),
                            "max": (None, [1,], [7,14,21,28]),
                            "min": (None, [1,], [7,14,21,28]),}
                        }
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data, valid_index=self.valid_index)

    def test_it_fit_predict(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        _ = fcaster.predict(self.predict_data)

    def test_it_fit_predict_with_lag_features(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features,
                        "lags":[1,2,3,4,5,6,7]}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        _ = fcaster.predict(self.predict_data, recursive=True)

    def test_it_fit_predict_with_rw_features(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features,
                        "window_functions":{
                            "mean": (None, [1,], [7,14,21,28]),
                            "median": (None, [1,], [7,14,21,28]),
                            "max": (None, [1,], [7,14,21,28]),
                            "min": (None, [1,], [7,14,21,28]),}
                        }
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        _ = fcaster.predict(self.predict_data, recursive=True)
    
    def test_it_fit_predict_with_detrend(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
            trend_models = compute_trend_models(self.train_data, ts_uid_columns=["ts_uid"])
        else:
            trend_models = compute_trend_models(self.train_data)
        model_kwargs["trend_models"] = trend_models

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        _ = fcaster.predict(self.predict_data)

    def test_it_fit_predict_with_scaling(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        model_kwargs["target_scaler"] = "standard"
        
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        _ = fcaster.predict(self.predict_data)

    def test_it_fit_predict_with_detrend_and_scaling(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
            trend_models = compute_trend_models(self.train_data, ts_uid_columns=["ts_uid"])
        else:
            trend_models = compute_trend_models(self.train_data)
        model_kwargs["trend_models"] = trend_models
        model_kwargs["target_scaler"] = "standard"
        
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        _ = fcaster.predict(self.predict_data)

    def test_it_fit_evaluate(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        
        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        error = fcaster.evaluate(self.eval_data)

        assert type(error)==np.float64, \
            f'fcaster.evaluate returns {error} which is not of type numpy.float64'
    
    def test_it_fit_evaluate_with_bounded_error(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        error = fcaster.evaluate(self.eval_data)

        assert error <= 3.0, \
            f"fcaster.evaluate returns error={error} which is greater than 3.0"

    def test_it_fit_evaluate_with_lag_features_with_bounded_error(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features,
                        "lags":[1,2,3,4,5,6,7]}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        error = fcaster.evaluate(self.eval_data, recursive=True)

        assert error <= 3.0, \
            f"fcaster.evaluate returns error={error} which is greater than 3.0"

    def test_it_fit_evaluate_with_rw_features_with_bounded_error(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features,
                        "window_functions":{
                            "mean": (None, [1,], [7,14,21,28]),
                            "median": (None, [1,], [7,14,21,28]),
                            "max": (None, [1,], [7,14,21,28]),
                            "min": (None, [1,], [7,14,21,28]),}
                        }
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        error = fcaster.evaluate(self.eval_data, recursive=True)

        assert error <= 3.0, \
            f"fcaster.evaluate returns error={error} which is greater than 3.0"

    def test_it_fit_evaluate_with_detrend_with_bounded_error(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
            trend_models = compute_trend_models(self.train_data, ts_uid_columns=["ts_uid"])
        else:
            trend_models = compute_trend_models(self.train_data)
        model_kwargs["trend_models"] = trend_models

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        error = fcaster.evaluate(self.eval_data)

        assert error <= 3.0, \
            f"fcaster.evaluate returns error={error} which is greater than 3.0"

    def test_it_fit_evaluate_with_scaling_with_bounded_error(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
        model_kwargs["target_scaler"] = "standard"

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        error = fcaster.evaluate(self.eval_data)

        assert error <= 3.0, \
            f"fcaster.evaluate returns error={error} which is greater than 3.0"

    def test_it_fit_evaluate_with_detrend_and_scaling_with_bounded_error(self):
        model_kwargs = {"model_params":get_default_model_params(self.model_class),
                        "time_features":self.time_features}
        if "ts_uid" in self.train_data.columns:
            model_kwargs["ts_uid_columns"] = ["ts_uid"]
            trend_models = compute_trend_models(self.train_data, ts_uid_columns=["ts_uid"])
        else:
            trend_models = compute_trend_models(self.train_data)
        model_kwargs["trend_models"] = trend_models
        model_kwargs["target_scaler"] = "standard"

        fcaster = self.model_class(**model_kwargs)
        fcaster.fit(train_data=self.train_data)
        error = fcaster.evaluate(self.eval_data)

        assert error <= 3.0, \
            f"fcaster.evaluate returns error={error} which is greater than 3.0"
