import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import calendar
import h2o

from tsforest.config import (calendar_sequential_features_types,
                             calendar_cyclical_features_types,
                             all_features_types)

def fill_time_gaps(data, freq="D"):
    """
    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe with columns 'ds' (dtype datetime64) and 'y' 
    """
    assert set(["ds","y"]) <= set(data.columns), "Data must contain the column 'ds'."
    filled_data = (data
                   .resample(freq, on="ds").y.mean()
                   .interpolate("linear")
                   .reset_index())
    filled_data = pd.merge(filled_data, data.drop("y", axis=1), on=["ds"], how="left")
    return filled_data

def compute_calendar_features(date_range, ignore_const_cols=True):
    """
    Parameters
    ----------
    date_range: pandas.DatetimeIndex or pandas.TimedeltaIndex
        Ranges of date times.
    ignore_const_cols: bool
        Specify whether to ignore constant columns.
    """  
    calendar_data = pd.DataFrame()
    calendar_data["ds"] = date_range
    calendar_features = ["year", "quarter", "month", "days_in_month", 
                         "weekofyear", "dayofyear", "day", "dayofweek", 
                         "hour", "minute", "second", "microsecond", 
                         "nanosecond"]
    calendar_features_map = {"weekofyear":"year_week",
                             "dayofyear":"year_day",
                             "day":"month_day",
                             "dayofweek":"week_day"}

    for feature in calendar_features:
        feature_series =  getattr(date_range, feature)
        if feature_series.nunique() == 1 and ignore_const_cols: 
            continue
        feature = calendar_features_map[feature] if feature in calendar_features_map else feature
        calendar_data[feature] = feature_series
    
    # adds missing features
    if (pd.infer_freq(date_range) in ["L", "ms", "U", "us", "N"] 
        and "microsecond" in calendar_data.columns):
        calendar_data['millisecond'] = calendar_data.microsecond//1000
    if {"month_day", "days_in_month"} < set(calendar_data.columns):
        calendar_data["month_progress"] = calendar_data.month_day/calendar_data.days_in_month 
    # adds cyclical encodings
    if "second" in calendar_data.columns:
        calendar_data["second_cos"] = np.cos(calendar_data.second*(2.*np.pi/60))
        calendar_data["second_sin"] = np.sin(calendar_data.second*(2.*np.pi/60))
    if "minute" in calendar_data.columns:
        calendar_data["minute_cos"] = np.cos(calendar_data.minute*(2.*np.pi/60))
        calendar_data["minute_sin"] = np.sin(calendar_data.minute*(2.*np.pi/60))
    if "hour" in calendar_data.columns:
        calendar_data["hour_cos"] = np.cos(calendar_data.hour*(2.*np.pi/24))
        calendar_data["hour_sin"] = np.sin(calendar_data.hour*(2.*np.pi/24))
    if "week_day" in calendar_data.columns:
        calendar_data["week_day_cos"] = np.cos(calendar_data.week_day*(2.*np.pi/7))
        calendar_data["week_day_sin"] = np.sin(calendar_data.week_day*(2.*np.pi/7))
    if "year_day" in calendar_data.columns:
        calendar_data["year_day_cos"] = np.cos((calendar_data.year_day-1)*(2.*np.pi/366))
        calendar_data["year_day_sin"] = np.sin((calendar_data.year_day-1)*(2.*np.pi/366))
    if "year_week" in calendar_data.columns:
        calendar_data["year_week_cos"] = np.cos((calendar_data.year_week-1)*(2.*np.pi/52))
        calendar_data["year_week_sin"] = np.sin((calendar_data.year_week-1)*(2.*np.pi/52))
    if "month" in calendar_data.columns:
        calendar_data["month_cos"] = np.cos((calendar_data.month-1)*(2.*np.pi/12))
        calendar_data["month_sin"] = np.sin((calendar_data.month-1)*(2.*np.pi/12))
    # week_day shifted to 1-7
    if "week_day" in calendar_data.columns:
        calendar_data["week_day"] += 1
    return calendar_data

def compute_lag_features(data, lags):
    """
    data: pandas.Dataframe
        dataframe with column 'y' (response values)
    lags: list
        list of integer lag values
    """
    assert "y" in data.columns, "Missing 'y' column in dataframe."
    features = [data.y.shift(lag) for lag in lags]
    features_names = [f"lag_{lag}" for lag in lags]
    lag_features = pd.concat([data.ds] + features, axis=1)
    lag_features.columns = ["ds"] + features_names
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
    assert "y" in data.columns, "Missing 'y' column in dataframe"
    # assert window functions in availabe funcs...
    shifted = data.y.shift(1)    
    features = [getattr(shifted.rolling(window), window_func).__call__()
                for window_func in window_functions
                for window in window_sizes]
    features_names = [f"{window_func}_{window}"
                      for window_func in window_functions
                      for window in window_sizes]
    rw_features = pd.concat([data.ds] + features, axis=1)
    rw_features.columns = ["ds"] + features_names
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

    def compute_train_features(self, data, ignore_const_cols=True):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe with (at least) columns: 'ds' and 'y'
        ignore_const_cols: bool
            Specify whether to ignore constant columns.
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
        if {"calendar","calendar_cyclical"} & set(self.include_features):
            input_params = {"date_range":pd.DatetimeIndex(data.ds),
                            "ignore_const_cols":ignore_const_cols}
            calendar_features = compute_calendar_features(**input_params)
        if "calendar" not in self.include_features:
            columns_to_drop = list(calendar_sequential_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        elif "calendar_cyclical" not in self.include_features:
            columns_to_drop = list(calendar_cyclical_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        all_features_list.append(calendar_features.set_index("ds"))
                
        if "lag" in self.include_features:
            lag_features = (compute_lag_features(filled_data, lags=self.lags)
                            .merge(data.loc[:, ["ds"]], how="inner", on=["ds"]))
            all_features_list.append(lag_features.set_index("ds"))

        if "rw" in self.include_features:
            rw_features = (compute_rw_features(filled_data, 
                                               window_sizes=self.window_sizes, 
                                               window_functions=self.window_functions)
                           .merge(data.loc[:, ["ds"]], how="inner", on=["ds"]))
            all_features_list.append(rw_features.set_index("ds"))

        # merging all features
        all_features_list.append(data.set_index("ds"))
        all_features = (pd.concat(all_features_list, axis=1)
                        .reset_index()
                        .rename({"index":"ds"}, axis=1)
                        .assign(y = lambda x: x["y"].fillna(0.)))
        categorical_features = [feature for feature,dtype in all_features_types.items()
                                if feature in all_features.columns and dtype=="categorical"]
        return all_features,categorical_features
        
    def compute_predict_features(self, data, ignore_const_cols=True):
        """
        Parameters
        ----------
        data: pandas.DataFrame
            Dataframe with (at least) column: 'ds'
        ignore_const_cols: bool
            Specify whether to ignore constant columns.
        """
        self.predict_data = data

        # list with all the dataframes of features
        all_features_list = list()

        # generating the calendar features
        if np.any(["calendar" in feat for feat in self.include_features]):
            input_params = {"date_range":pd.DatetimeIndex(data.ds),
                            "ignore_const_cols":ignore_const_cols}
            calendar_features = compute_calendar_features(**input_params)
        if "calendar" not in self.include_features:
            columns_to_drop = list(calendar_sequential_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        elif "calendar_cyclical" not in self.include_features:
            columns_to_drop = list(calendar_cyclical_features_types.keys())
            calendar_features.drop(columns=columns_to_drop, inplace=True)
        all_features_list.append(calendar_features.set_index("ds"))

        if "lag" in self.include_features:
            column_names = [f"lag_{lag}" for lag in self.lags]
            lag_features = pd.DataFrame(np.nan, index=data.ds, 
                                        columns=column_names)
            all_features_list.append(lag_features)
    
        if "rw" in self.include_features:
            column_names = [f"{window_func}_{window}"
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
                                if feature in all_features.columns and dtype=="categorical"]
        return all_features,categorical_features
