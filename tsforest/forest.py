import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

import h2o
import lightgbm
import catboost
import xgboost
from tsforest.config import cat_parameters, lgbm_parameters, gbm_parameters, xgb_parameters
from tsforest.forest_base import BaseRegressor

class H2OGBMRegressor(BaseRegressor):
    def __init__(self, model_params):
        self.model_params = {**gbm_parameters, **model_params}

    def cast_dataframe(self, features_dataframe, input_features, target,  categorical_features):
        features_types = {feature:"categorical" for feature in categorical_features
                          if feature in input_features}
        features_dataframe_casted = h2o.H2OFrame(features_dataframe, 
                                                 column_types=features_types)
        return features_dataframe_casted
    
    def fit(self, train_features, valid_features, input_features, target, categorical_features):
        train_features_casted = self.cast_dataframe(train_features, input_features, target, categorical_features)
        valid_features_casted = self.cast_dataframe(valid_features, input_features, target, categorical_features) \
                                if valid_features is not None else None
        model_params = dict(self.model_params)
        training_params = {"training_frame":train_features_casted, 
                           "x":input_features, 
                           "y":target}
        if valid_features is not None:
            training_params["validation_frame"] = valid_features_casted
        elif "stopping_rounds" in model_params:
            del model_params["stopping_rounds"]
        if "weight" in train_features.columns:
            training_params["weights_column"] = "weight"
        # model training
        self.model = h2o.estimators.H2OGradientBoostingEstimator(**model_params)
        self.model.train(**training_params)
        self.best_iteration = int(self.model.summary()["number_of_trees"][0])
        self.input_features = input_features
        self.target = target
        self.categorical_features = categorical_features

    def predict(self, predict_features):
        predict_features_casted = self.cast_dataframe(predict_features, self.input_features, self.target, self.categorical_features)
        _prediction = self.model.predict(predict_features_casted)
        prediction = _prediction.as_data_frame().values[:,0]
        return prediction


class LightGBMRegressor(BaseRegressor):
    def __init__(self, model_params):
        self.model_params = {**lgbm_parameters, **model_params}

    def cast_dataframe(self, features_dataframe, input_features, target, categorical_features):
        dataset_params = {"data":features_dataframe.loc[:, input_features],
                          "categorical_feature":categorical_features,
                          "free_raw_data":False}
        if "weight" in features_dataframe.columns:
            dataset_params["weight"] = features_dataframe.loc[:, weight].values
        if target in features_dataframe.columns:
            dataset_params["label"] = features_dataframe.loc[:, target].values
        features_dataframe_casted = lightgbm.Dataset(**dataset_params)
        return features_dataframe_casted 
    
    def fit(self, train_features, valid_features, input_features, target, categorical_features):
        train_features_casted = self.cast_dataframe(train_features, input_features, target, categorical_features)
        valid_features_casted = self.cast_dataframe(valid_features, input_features, target, categorical_features) \
                                if valid_features is not None else None
        model_params = dict(self.model_params)
        training_params = {"train_set":train_features_casted}
        if valid_features is not None:
            training_params["valid_sets"] = valid_features_casted
            training_params["verbose_eval"] = False
        elif "early_stopping_rounds" in model_params:
            del model_params["early_stopping_rounds"]
        training_params["params"] = model_params
        # model training
        self.model = lightgbm.train(**training_params)
        self.best_iteration = self.model.best_iteration if self.model.best_iteration>0 else self.model.num_trees()
        self.input_features = input_features
        self.target = target
        self.categorical_features = categorical_features

    def predict(self, predict_features):
        prediction = self.model.predict(predict_features.loc[:, self.input_features])
        return prediction


class CatBoostRegressor(BaseRegressor):
    def __init__(self, model_params):
        self.model_params = {**cat_parameters, **model_params}

    def cast_dataframe(self, features_dataframe, input_features, target,  categorical_features):
        dataset_params = {"data":features_dataframe.loc[:, input_features],
                          "cat_features":categorical_features}
        if "weight" in features_dataframe.columns:
            dataset_params["weight"] = features_dataframe.loc[:, weight].values
        if target in features_dataframe.columns:
            dataset_params["label"] = features_dataframe.loc[:, target].values
        features_dataframe_casted = catboost.Pool(**dataset_params)
        return features_dataframe_casted
    
    def fit(self, train_features, valid_features, input_features, target, categorical_features):
        train_features_casted = self.cast_dataframe(train_features, input_features, target, categorical_features)
        valid_features_casted = self.cast_dataframe(valid_features, input_features, target, categorical_features) \
                                if valid_features is not None else None
        model_params = dict(self.model_params)
        training_params = {"X":train_features_casted,
                           "verbose":False}
        if valid_features is not None:
            training_params["eval_set"] = valid_features_casted
        elif "early_stopping_rounds" in model_params:
            del model_params["early_stopping_rounds"]
        # model training
        self.model = catboost.CatBoostRegressor(**model_params)
        self.model.fit(**training_params)
        self.best_iteration = self.model.best_iteration_ if self.model.best_iteration_ is not None else self.model.tree_count_
        self.input_features = input_features
        self.target = target
        self.categorical_features = categorical_features
    
    def predict(self, predict_features):
        predict_features_casted = self.cast_dataframe(predict_features, self.input_features, self.target, self.categorical_features)
        prediction = self.model.predict(predict_features_casted)
        return prediction


class XGBoostRegressor(BaseRegressor):
    def __init__(self, model_params):
        self.model_params = {**xgb_parameters, **model_params}

    def cast_dataframe(self, features_dataframe, input_features, target,  categorical_features):
        dataset_params = {"data":features_dataframe.loc[:, input_features]}
        if "weight" in features_dataframe.columns:
            dataset_params["weight"] = features_dataframe.loc[:, weight].values
        if target in features_dataframe.columns:
            dataset_params["label"] = features_dataframe.loc[:, target].values
        features_dataframe_casted = xgboost.DMatrix(**dataset_params)
        return features_dataframe_casted
    
    def fit(self, train_features, valid_features, input_features, target, categorical_features):
        train_features_casted = self.cast_dataframe(train_features, input_features, target, categorical_features)
        valid_features_casted = self.cast_dataframe(valid_features, input_features, target, categorical_features) \
                                if valid_features is not None else None
        model_params = dict(self.model_params)
        training_params = {"dtrain":train_features_casted,
                           "num_boost_round":model_params.pop("num_boost_round"),
                           "early_stopping_rounds":model_params.pop("early_stopping_rounds")}
        if valid_features is not None:
            training_params["evals"] = [(valid_features_casted,"eval"),]
            training_params["verbose_eval"] = False
        elif "early_stopping_rounds" in training_params:
            del training_params["early_stopping_rounds"]
        training_params["params"] = model_params
        # model training
        self.model = xgboost.train(**training_params)
        self.best_iteration = self.model.best_ntree_limit
        self.input_features = input_features
        self.target = target
        self.categorical_features = categorical_features
    
    def predict(self, predict_features):
        predict_features_casted = self.cast_dataframe(predict_features, self.input_features, self.target, self.categorical_features)
        prediction = self.model.predict(predict_features_casted, ntree_limit=self.best_iteration)
        return prediction
