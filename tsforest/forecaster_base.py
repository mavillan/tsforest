import pandas as pd
from tsforest.config import gbm_parameters
from tsforest.features import FeaturesGenerator

class ForecasterBase(object):
      
    def _prepare_train_features(self, train_data):
        '''
        Parameters
        ----------
        train_data : pandas.DataFrame
            dataframe with at least columns "ds" and "y"
        '''
        features_generator = FeaturesGenerator(include_features=self.features+['prophet'],
                                               lags=self.lags,
                                               window_sizes=self.window_sizes,
                                               window_functions=self.window_functions)
        train_features,features_types = features_generator.compute_train_features(train_data)

        train_features['y_hat'] = train_features.y.copy()
        if self.detrend:
            train_features.loc[:,'y_hat'] = train_features.y_hat - train_features.prophet_trend
        if self.response_scaling:
            y_mean = train_features.y_hat.mean()
            y_std = train_features.y_hat.std()
            train_features.loc[:, 'y_hat'] = (train_features.y_hat-y_mean)/y_std

        self.y_mean = y_mean if 'y_mean' in locals() else None
        self.y_std  = y_std if 'y_std' in locals() else None
        self.features_generator = features_generator

        exclude_features = ['ds','y','y_hat',
                            'sequence_day','month_day','price',
                            'prophet_trend', 'weights','fold_column']
        self.input_features = [feature for feature in train_features.columns
                               if feature not in exclude_features]
        self.target = 'y_hat'

        return train_features,features_types
    
    def _prepare_valid_features(self, valid_period, train_features):
        '''
        valid_period : pandas.DataFrame
            dataframe with column "ds" indicating the validation period
        train_features: pandas.DataFrame
            dataframe
        '''
        #print(valid_period.head())
        #print("#"*80)
        #print(train_features.head())
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
        return test_features,features_types
