import h2o
from tsforest.forest import H2OGBMRegressor, LightGBMRegressor, CatBoostRegressor, XGBoostRegressor
from tsforest.forecaster_base import ForecasterBase

class H2OGBMForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(H2OGBMForecaster, self).__init__(*args, **kwargs)
        # init the cluster if is not already up
        if h2o.cluster() is None: h2o.init(nthreads=-1)
        self.model = H2OGBMRegressor(kwargs['model_params'])

class LightGBMForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(LightGBMForecaster, self).__init__(*args, **kwargs)
        self.model = LightGBMRegressor(kwargs['model_params'])

class CatBoostForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(CatBoostForecaster, self).__init__(*args, **kwargs)
        self.model = CatBoostRegressor(kwargs['model_params'])

class XGBoostForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(XGBoostForecaster, self).__init__(*args, **kwargs)
        self.model = XGBoostRegressor(kwargs['model_params'])
        if self.categorical_encoding == "default":
            self.categorical_encoding = "CatBoostEncoder"