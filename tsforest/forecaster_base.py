import pandas as pd
from inspect import getmembers, isfunction

from tsforest.config import gbm_parameters
from tsforest.features import FeaturesGenerator
from tsforest.trend import TrendEstimator
from tsforest import metrics

class ForecasterBase(object):
      
    def _prepare_train_features(self, train_data):
        '''
        Parameters
        ----------
        train_data : pandas.DataFrame
            dataframe with at least columns "ds" and "y"
        '''
        features_generator = FeaturesGenerator(include_features=self.features,
                                               lags=self.lags,
                                               window_sizes=self.window_sizes,
                                               window_functions=self.window_functions)
        train_features,features_types = features_generator.compute_train_features(train_data)
        features_types = {**features_types, **self.features_types}

        exclude_features = ['ds', 'y', 'y_hat', 'month_day', 'weight', 'fold_column']
        self.input_features = [feature for feature in train_features.columns
                               if feature not in exclude_features]
        self.features_generator = features_generator
        return train_features,features_types
    
    def _prepare_valid_features(self, valid_period, train_features):
        '''
        valid_period : pandas.DataFrame
            dataframe with column "ds" indicating the validation period
        train_features: pandas.DataFrame
            dataframe
        '''
        valid_features = pd.merge(valid_period, train_features, how='inner', on=['ds'])
        assert len(valid_features)==len(valid_period), \
            'valid_period must be contained in the time period of time_features'
        return valid_features

    def _prepare_test_features(self, test_period):
        '''
        Parameters
        ----------
        test_data: pandas.DataFrame
            dataframe with the same columns as self.train_data (except for "y") 
            containing the test period
        Returns
        ----------
        test_features: pandas.DataFrame
            Dataframe containing all the features for evaluating the trained model
        '''
        test_features,features_types = self.features_generator.compute_test_features(test_period)
        return test_features
    
    def _prepare_train_response(self, train_features):
        '''
        Prepares the train response variable

        Parameters
        ----------
        train_features: pd.DataFrame
            dataframe containing the columns "ds" and "y"
        '''
        y_hat = train_features.y.copy()

        if self.detrend:
            trend_estimator = TrendEstimator()
            trend_estimator.fit(data=train_features.loc[:, ['ds','y']])
            trend_dataframe = trend_estimator.predict(train_features.loc[:, ['ds']])
            y_hat -= trend_dataframe.trend.values
            self.trend_estimator = trend_estimator
        if self.response_scaling:
            y_mean = y_hat.mean()
            y_std = y_hat.std()
            y_hat -= y_mean
            y_hat /= y_std 

        self.y_mean = y_mean if 'y_mean' in locals() else None
        self.y_std  = y_std if 'y_std' in locals() else None
        self.target = 'y_hat'
        return y_hat
    
    def _prepare_valid_response(self, valid_features):
        '''
        Prepares the validation response variable

        Parameters
        ----------
        valid_features: pd.DataFrame
            dataframe containing the columns "ds" and "y"
        '''
        y_hat = valid_features.y.copy()

        if self.detrend:
            trend_estimator = self.trend_estimator
            trend_dataframe = trend_estimator.predict(valid_features.loc[:, ['ds']])
            y_hat -= trend_dataframe.trend.values
        if self.response_scaling:
            y_hat -= self.y_mean
            y_hat /= self.y_std 
            
        return y_hat

    def evaluate(self, test_data, metric='rmse'):
        '''
        Parameters
        ----------
        test_data: pandas.DataFrame
            dataframe with the same columns as "train_data"
        metric: string
            possible values: "mae", "mape", "mse", "rmse", "smape"
        Returns
        ----------
        error: float
            error of predictions according to the error measure
        '''
        assert set(self.train_data.columns) == set(test_data.columns), \
            '"test_data" must have the same columns as "train_data"'
        available_metrics = [member[0].split('_')[1] for member in getmembers(metrics) 
                             if isfunction(member[1])]
        assert metric in available_metrics, \
            f'"metric" must be any of these: {available_metrics}'
        test_data = test_data.copy()
        y_real = test_data.pop("y")
        y_pred = self.predict(test_data)["y_pred"].values
        error_func = getattr(metrics, f'compute_{metric}')
        error = error_func(y_real, y_pred)
        return error
