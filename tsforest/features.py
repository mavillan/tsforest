import numpy as np
import pandas as pd
from tsforest.config import (calendar_features_names,
                             calendar_cyclical_features_names)

time_features_mapping = {"year_week":"weekofyear",
                         "year_day":"dayofyear",
                         "month_day":"day",
                         "week_day":"dayofweek"}

def compute_train_features(data, time_features, lags, window_sizes, 
                           window_functions, ignore_const_cols=True):
    """
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with (at least) columns: 'ds' and 'y'.
    time_features: list
        Time attributes to include as features.
    lags: list
        List of integer lag values to include as features.
    window_sizes: list
        List of integer window sizes values to include as features.
    window_functions: list
        List of string names of the window functions to include as features.
    ignore_const_cols: bool
        Specify whether to ignore constant columns.
    Returns
    ----------
    all_features: pd.Dataframe
        Dataframe containing all the features for the time series.
    """
    # list with all the dataframes of features
    all_features_list = list()

    # generating the time features
    if len(time_features) > 0:
        input_params = {"date_range":pd.DatetimeIndex(data.ds),
                        "time_features":time_features,
                        "ignore_const_cols":ignore_const_cols}
        calendar_features = compute_calendar_features(**input_params)
        all_features_list.append(calendar_features.set_index("ds"))

    # filling time gaps for 'lag' and 'rw' features
    if len(lags) > 0 or (len(window_sizes) > 0 & len(window_functions) > 0):
        filled_data = fill_time_gaps(data)

    if len(lags) > 0:
        lag_features = (compute_lag_features(filled_data, lags=lags)
                        .merge(data.loc[:, ["ds"]], how="inner", on=["ds"]))
        all_features_list.append(lag_features.set_index("ds"))

    if len(window_sizes) > 0 & len(window_functions) > 0:
        rw_features = (compute_rw_features(filled_data, 
                                            window_sizes=window_sizes, 
                                            window_functions=window_functions)
                        .merge(data.loc[:, ["ds"]], how="inner", on=["ds"]))
        all_features_list.append(rw_features.set_index("ds"))

    # merging all features
    all_features_list.append(data.set_index("ds"))
    all_features = (pd.concat(all_features_list, axis=1)
                    .reset_index()
                    .rename({"index":"ds"}, axis=1)
                    .assign(y = lambda x: x["y"].fillna(0.)))
    return all_features

def compute_predict_features(data, time_features, lags, window_sizes, 
                             window_functions, ignore_const_cols=True):
    """
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with (at least) columns: 'ds' and 'y'.
    time_features: list
        Time attributes to include as features.
    lags: list
        List of integer lag values to include as features.
    window_sizes: list
        List of integer window sizes values to include as features.
    window_functions: list
        List of string names of the window functions to include as features.
    ignore_const_cols: bool
        Specify whether to ignore constant columns.
    Returns
    ----------
    all_features: pd.Dataframe
        Dataframe containing all the features for the time series.
    """
    # list with all the dataframes of features
    all_features_list = list()

    # generating the time features
    if len(time_features) > 0:
        input_params = {"date_range":pd.DatetimeIndex(data.ds),
                        "time_features":time_features,
                        "ignore_const_cols":ignore_const_cols}
        calendar_features = compute_calendar_features(**input_params)
        all_features_list.append(calendar_features.set_index("ds"))

    if len(lags) > 0:
        column_names = [f"lag_{lag}" for lag in lags]
        lag_features = pd.DataFrame(np.nan, index=data.ds, 
                                    columns=column_names)
        all_features_list.append(lag_features)

    if len(window_sizes) > 0 & len(window_functions) > 0:
        column_names = [f"{window_func}_{window}"
                        for window_func in window_functions
                        for window in window_sizes]
        rw_features = pd.DataFrame(np.nan, index=data.ds,
                                    columns=column_names)
        all_features_list.append(rw_features)
    
    # merging all features
    all_features_list.append(data.set_index("ds"))
    all_features = pd.concat(all_features_list, axis=1)
    all_features.reset_index(inplace=True)
    return all_features

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

def compute_calendar_features(date_range, time_features, ignore_const_cols=True):
    """
    Parameters
    ----------
    date_range: pandas.DatetimeIndex or pandas.TimedeltaIndex
        Ranges of date times.
    time_features: List
        Time attributes to include as features.
    ignore_const_cols: bool
        Specify whether to ignore constant columns.
    """  
    calendar_data = pd.DataFrame()
    calendar_data["ds"] = date_range

    for feature in time_features:
        if feature in time_features_mapping.keys():
            _feature = time_features_mapping[feature]
        else:
            _feature = feature
        
        if hasattr(date_range, _feature):
            feature_series = getattr(date_range, _feature)
            if feature_series.nunique() == 1 and ignore_const_cols: 
                continue
            calendar_data[feature] = feature_series

    # other time features
    if "month_progress" in time_features:
        calendar_data["month_progress"] = date_range.day/date_range.days_in_month
    if "millisecond" in time_features:
        calendar_data["millisecond"] = date_range.microsecond//1000

    # cyclical time features
    if "second_cos" in time_features:
        calendar_data["second_cos"] = np.cos(date_range.second*(2.*np.pi/60))
    if "second_sin" in time_features:        
        calendar_data["second_sin"] = np.sin(date_range.second*(2.*np.pi/60))
    if "minute_cos" in time_features:
        calendar_data["minute_cos"] = np.cos(date_range.minute*(2.*np.pi/60))
    if "minute_sin" in time_features:
        calendar_data["minute_sin"] = np.sin(date_range.minute*(2.*np.pi/60))
    if "hour_cos" in time_features:
        calendar_data["hour_cos"] = np.cos(date_range.hour*(2.*np.pi/24))
    if "hour_sin" in time_features:
        calendar_data["hour_sin"] = np.sin(date_range.hour*(2.*np.pi/24))
    if "week_day_cos" in time_features:
        calendar_data["week_day_cos"] = np.cos(date_range.dayofweek*(2.*np.pi/7))
    if "week_day_sin" in time_features:
        calendar_data["week_day_sin"] = np.sin(date_range.dayofweek*(2.*np.pi/7))
    if "year_day_cos" in time_features:
        calendar_data["year_day_cos"] = np.cos((date_range.dayofyear-1)*(2.*np.pi/366))
    if "year_day_sin" in time_features:
        calendar_data["year_day_sin"] = np.sin((date_range.dayofyear-1)*(2.*np.pi/366))
    if "year_week_cos" in time_features:
        calendar_data["year_week_cos"] = np.cos((date_range.weekofyear-1)*(2.*np.pi/52))
    if "year_week_sin" in time_features:
        calendar_data["year_week_sin"] = np.sin((date_range.weekofyear-1)*(2.*np.pi/52))
    if "month_cos" in time_features:
        calendar_data["month_cos"] = np.cos((date_range.month-1)*(2.*np.pi/12))
    if "month_sin" in time_features:
        calendar_data["month_sin"] = np.sin((date_range.month-1)*(2.*np.pi/12))
    
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
