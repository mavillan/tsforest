import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import calendar
import h2o

from tsforest.config import (calendar_sequential_features_types,
                             calendar_cyclical_features_types,
                             all_features_types)

def fill_time_gaps(data):
    """
    Parameters
    ----------
    data: pandas.DataFrame
        dataframe with columns 'ds' (dtype datetime64) and 'y' 
    """
    assert set(['ds','y']) <= set(data.columns), \
        'data must contain the column "ds"'
    filled_data = (data
                   .resample("D", on='ds').y.mean()
                   .interpolate("linear")
                   .reset_index())
    filled_data = pd.merge(filled_data, data.drop('y', axis=1), on=['ds'], how='left')
    return filled_data

def compute_calendar_features(start_time, end_time, freq='D'):
    """
    Parameters
    ----------
    start_time: str
        start time for time range
    end_time: str
        end time for time range
    freq: str
        frequency string of pandas.date_range function
    """
    date_range = pd.date_range(start=start_time, end=end_time, freq=freq)
    
    calendar_data = pd.DataFrame()
    calendar_data['ds'] = date_range
    calendar_data['week_day'] = date_range.dayofweek
    calendar_data['month_day'] = date_range.day
    calendar_data['year_day'] = date_range.dayofyear
    calendar_data['year_week'] = date_range.weekofyear
    calendar_data['month'] = date_range.month
    calendar_data['year'] = date_range.year
    days_of_month = calendar_data.apply(lambda x: calendar.monthrange(x.year,x.month)[1], axis=1)
    calendar_data['month_progress'] = calendar_data.month_day/days_of_month

    # cyclical encodings: day of week
    calendar_data['week_day_cos'] = np.cos(calendar_data.week_day*(2.*np.pi/7))
    calendar_data['week_day_sin'] = np.sin(calendar_data.week_day*(2.*np.pi/7))
    # cyclical encodings: day of year
    calendar_data['year_day_cos'] = np.cos((calendar_data.year_day-1)*(2.*np.pi/366))
    calendar_data['year_day_sin'] = np.sin((calendar_data.year_day-1)*(2.*np.pi/366))
    # cyclical encodings: week of year
    calendar_data['year_week_cos'] = np.cos((calendar_data.year_week-1)*(2.*np.pi/52))
    calendar_data['year_week_sin'] = np.sin((calendar_data.year_week-1)*(2.*np.pi/52))
    # cyclical encodings: month of year
    calendar_data['month_cos'] = np.cos((calendar_data.month-1)*(2.*np.pi/12))
    calendar_data['month_sin'] = np.sin((calendar_data.month-1)*(2.*np.pi/12))
    # week_day shifted to 1-7
    calendar_data['week_day'] += 1
    return calendar_data

def compute_lag_features(data, lags):
    """
    data: pandas.Dataframe
        dataframe with column 'y' (response values)
    lags: list
        list of integer lag values
    """
    assert 'y' in data.columns, \
        'Missing "y" column in dataframe'
    features = [data.y.shift(lag) for lag in lags]
    features_names = [f'lag_{lag}' for lag in lags]
    lag_features = pd.concat([data.ds] + features, axis=1)
    lag_features.columns = ['ds'] + features_names
    return lag_features  

def compute_rw_features(data, window_sizes, window_functions):
    """
    data: pandas.Dataframe
        dataframe with column 'y' (response values)
    window_sizes: list
        list of integer window sizes values
    window_functions: list
        list of string names of the window functions
    """
    assert 'y' in data.columns, \
        'Missing "y" column in dataframe'
    # assert window functions in availabe funcs...
    shifted = data.y.shift(1)    
    features = [getattr(shifted.rolling(window), window_func).__call__()
                for window_func in window_functions
                for window in window_sizes]
    features_names = [f'{window_func}_{window}'
                      for window_func in window_functions
                      for window in window_sizes]
    rw_features = pd.concat([data.ds] + features, axis=1)
    rw_features.columns = ['ds'] + features_names
    return rw_features

class FeaturesGenerator():

    def __init__(self, include_features, lags=None, window_sizes=None, 
                 window_functions=None):
        """
        Parameters
        ----------
        include_features: list
            Features to included in computation
        lags: list
            List of integer lag values
        window_sizes: list
            List of integer window sizes values
        window_functions: list
            List of string names of the window functions
        """
        self.include_features = include_features
        self.lags = lags
        self.window_sizes = window_sizes
        self.window_functions = window_functions

    def compute_train_features(self, data):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with (at least) columns: 'ds' and 'y'
        Returns
        ----------
        all_features: pd.Dataframe
            Dataframe containing all the features for the time series
        """
        self.train_data = data
        # in case of time gaps
        filled_data = fill_time_gaps(data)

        # list with all the dataframes of features
        all_features_list = list()

        # generating the calendar features
        if {'calendar','calendar_cyclical'} & set(self.include_features):
            calendar_features = (compute_calendar_features(data.ds.min(), data.ds.max())
                                 .merge(data.loc[:, ['ds']], how='inner', on=['ds']))
        if "calendar" not in self.include_features:
            columns_to_drop = list(calendar_sequential_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        elif "calendar_cyclical" not in self.include_features:
            columns_to_drop = list(calendar_cyclical_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        all_features_list.append(calendar_features.set_index('ds'))
                
        if "lag" in self.include_features:
            lag_features = (compute_lag_features(filled_data, lags=self.lags)
                            .merge(data.loc[:, ['ds']], how='inner', on=['ds']))
            all_features_list.append(lag_features.set_index('ds'))

        if "rw" in self.include_features:
            rw_features = (compute_rw_features(filled_data, 
                                               window_sizes=self.window_sizes, 
                                               window_functions=self.window_functions)
                           .merge(data.loc[:, ['ds']], how='inner', on=['ds']))
            all_features_list.append(rw_features.set_index('ds'))

        # merging all features
        all_features_list.append(data.set_index("ds"))
        all_features = (pd.concat(all_features_list, axis=1)
                        .reset_index()
                        .rename({'index':'ds'}, axis=1)
                        .assign(y = lambda x: x['y'].fillna(0.)))
        categorical_features = [feature for feature,dtype in all_features_types.items()
                                if feature in all_features.columns and dtype=='categorical']
        return all_features,categorical_features
        
    def compute_test_features(self, data):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with (at least) column: 'ds'
        """
        self.test_data = data

        # list with all the dataframes of features
        all_features_list = list()

        # generating the calendar features
        if np.any(['calendar' in feat for feat in self.include_features]):
            calendar_features = (compute_calendar_features(data.ds.min(), data.ds.max())
                                 .merge(data.loc[:, ['ds']], how='inner', on=['ds']))
        if "calendar" not in self.include_features:
            columns_to_drop = list(calendar_sequential_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        elif "calendar_cyclical" not in self.include_features:
            columns_to_drop = list(calendar_cyclical_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        all_features_list.append(calendar_features.set_index('ds'))

        if "lag" in self.include_features:
            column_names = [f'lag_{lag}' for lag in self.lags]
            lag_features = pd.DataFrame(np.nan, index=data.ds, 
                                        columns=column_names)
            all_features_list.append(lag_features)
    
        if "rw" in self.include_features:
            column_names = [f'{window_func}_{window}'
                            for window_func in self.window_functions
                            for window in self.window_sizes]
            rw_features = pd.DataFrame(np.nan, index=data.ds,
                                       columns=column_names)
            all_features_list.append(rw_features)
        
        # merging all features
        all_features_list.append(data.set_index("ds"))
        all_features = pd.concat(all_features_list, axis=1)
        all_features.reset_index(inplace=True)
        categorical_features = [feature for feature,dtype in all_features_types.items()
                                if feature in all_features.columns and dtype=='categorical']
        return all_features,categorical_features
