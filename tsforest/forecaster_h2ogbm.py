import numpy as np
import pandas as pd
from zope.interface import implementer
import h2o
from h2o.estimators import H2OGradientBoostingEstimator
# init the cluster if is not already up
if h2o.cluster() is None: h2o.init(nthreads=-1)

from tsforest.config import gbm_parameters
from tsforest.forecaster_base import ForecasterBase
from tsforest.forecaster_interface import ForecasterInterface

@implementer(ForecasterInterface)
class H2OGBMForecaster(ForecasterBase):

    def _cast_dataframe(self, features_dataframe, categorical_features):
        """
        Parameters
        ----------
        features_dataframe: pandas.DataFrame
            dataframe containing all the features
        categorical_features: list
            list of names of categorical features
        Returns
        ----------
        features_dataframe_casted: h2o.H2OFrame
            features dataframe casted to H2O dataframe format
        """
        features_types = {feature:"categorical" for feature in categorical_features
                          if feature in self.input_features}
        features_dataframe_casted = h2o.H2OFrame(features_dataframe, 
                                                 column_types=features_types)
        return features_dataframe_casted

    def fit(self, train_data, valid_period=None):
        train_features,valid_features = super()._prepare_features(train_data, valid_period)
        train_features_casted = self._cast_dataframe(train_features, self.categorical_features)
        valid_features_casted = self._cast_dataframe(valid_features, self.categorical_features) \
                                if valid_period is not None else None
        # model_params overwrites default params of model
        model_params = {**gbm_parameters, **self.model_params}
        training_params = {"training_frame":train_features_casted, 
                           "x":self.input_features, 
                           "y":self.target}
        if valid_period is not None:
            training_params["validation_frame"] = valid_features_casted
        elif "stopping_rounds" in model_params:
            del model_params["stopping_rounds"]
        if "weight" in self.train_features.columns:
            training_params["weights_column"] = "weight"
        # model training
        model = H2OGradientBoostingEstimator(**model_params)
        model.train(**training_params)
        self.model = model
        self.best_iteration = int(model.summary()["number_of_trees"][0])

    def _predict(self, model, predict_features, trend_dataframe):
        """
        Parameters
        ----------
        model: h2o.estimators.H2OGradientBoostingEstimator 
            Trained H2OGradientBoostingEstimator model.
        predict_features: pandas.DataFrame
            Datafame containing the features for the prediction period.
        trend_dataframe: pandas.DataFrame
            Dataframe containing the trend estimation over the prediction period.
        """
        y_train = self.train_features.y.values
        y_valid = self.valid_features.y.values \
                  if self.valid_period is not None else np.array([])
        y = np.concatenate([y_train, y_valid])

        prediction = list()
        for idx in range(predict_features.shape[0]):
            if "lag" in self.features:
                for lag in self.lags:
                    predict_features.loc[idx, f"lag_{lag}"] = y[-lag]
            if "rw" in self.features:
                for window_func in self.window_functions:
                    for window in self.window_sizes:
                        predict_features.loc[idx, f"{window_func}_{window}"] = getattr(np, window_func)(y[-window:])
            predict_features_casted = self._cast_dataframe(predict_features.loc[[idx], :], 
                                                        self.categorical_features)
            _y_pred = model.predict(predict_features_casted)
            y_pred = _y_pred.as_data_frame().values[:,0]
            prediction.append(y_pred.copy())
            if self.response_scaling:
                y_pred *= self.y_std
                y_pred += self.y_mean
            if self.detrend:
                y_pred += trend_dataframe.loc[idx, "trend"]
            y = np.append(y, [y_pred])
        return np.asarray(prediction).ravel()

    def predict(self, predict_data):
        """
        Parameters
        ----------
        predict_data: pandas.DataFrame
            Datafame containing the features for the prediction period.
            Contains the same columns as 'train_data' except for 'y'.
        Returns
        ----------
        prediction_dataframe: pandas.DataFrame
            dataframe containing dates 'ds' and predictions 'y_pred'
        """
        self._validate_predict_inputs(predict_data) 
        predict_features = super()._prepare_predict_features(predict_data)
        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(predict_data.loc[:, ["ds"]])
        else:
            trend_dataframe = None

        if "lag" in self.features or "rw" in self.features:
            prediction = self._predict(self.model, predict_features, trend_dataframe)
        else:
            predict_features_casted = self._cast_dataframe(predict_features, 
                                                        self.categorical_features)
            _prediction = self.model.predict(predict_features_casted)
            prediction = _prediction.as_data_frame().values[:,0]

        if self.response_scaling:
            prediction *= self.y_std
            prediction += self.y_mean
        if self.detrend:
            prediction += trend_dataframe.trend.values
        if "zero_response" in predict_features.columns:
            zero_response_mask = predict_features["zero_response"]==1
            prediction[zero_response_mask] = 0
        
        self.predict_features = predict_features
            
        prediction_dataframe = pd.DataFrame({"ds":predict_data.ds, "y_pred":prediction})
        return prediction_dataframe

    def show_variable_importance(self):
        pass

    def save_model(self):
        pass