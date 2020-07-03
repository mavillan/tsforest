import h2o
from tsforest.forest import (H2OGBMRegressor, 
                             LightGBMRegressor, 
                             CatBoostRegressor, 
                             XGBoostRegressor)
from tsforest.forecast_base import ForecasterBase
import category_encoders as ce

class H2OGBMForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(H2OGBMForecaster, self).__init__(*args, **kwargs)
        # init the cluster if is not already up
        if h2o.cluster() is None: h2o.init(nthreads=-1)
        model_params = kwargs["model_params"] if "model_params" in kwargs else dict()
        self.model = H2OGBMRegressor(model_params)
        for feature,encoding in self.categorical_features.items():
            if encoding == "default":
                self.categorical_features[feature] = ("y", ce.TargetEncoder, dict())

class LightGBMForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(LightGBMForecaster, self).__init__(*args, **kwargs)
        self.model = LightGBMRegressor(self.model_params)

class CatBoostForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(CatBoostForecaster, self).__init__(*args, **kwargs)
        self.model = CatBoostRegressor(self.model_params)

class XGBoostForecaster(ForecasterBase):
    def __init__(self, *args, **kwargs):
        super(XGBoostForecaster, self).__init__(*args, **kwargs)
        self.model = XGBoostRegressor(self.model_params)
        for feature,encoding in self.categorical_features.items():
            if encoding == "default":
                self.categorical_features[feature] = ("y", ce.TargetEncoder, dict())
