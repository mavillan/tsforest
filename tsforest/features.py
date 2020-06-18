import numpy as np
import pandas as pd
from tsforest.config import (calendar_features_names,
                             calendar_cyclical_features_names)
from joblib import Parallel, delayed

time_features_mapping = {"year_week":"weekofyear",
                         "year_day":"dayofyear",
                         "month_day":"day",
                         "week_day":"dayofweek"}

def compute_train_features(data, ts_uid_columns, time_features, lags, window_shifts,
                           window_sizes, window_functions, ignore_const_cols=True,
                           n_jobs=1):
    """
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with (at least) columns: 'ds' and 'y'.
    ts_uid_columns: list
        List of columns names that are unique identifiers for time series.
    time_features: list
        Time attributes to include as features.
    lags: list
        List of integer lag values to include as features.
    window_shifts: list
        List of integer window shift values.
    window_sizes: list
        List of integer window sizes values to include as features.
    window_functions: list
        List of string names of the window functions to include as features.
    ignore_const_cols: bool
        Specify whether to ignore constant columns.
    n_jobs: int
        Number of jobs to run in parallel when computing the lag/rw features.
    Returns
    ----------
    all_features: pd.Dataframe
        Dataframe containing all the features for the time series.
    """
    # list with all the dataframes of features
    all_features_list = list()
    all_features_list.append(data.reset_index(drop=True))

    # generating the time features
    if len(time_features) > 0:
        input_params = {"date_range":pd.DatetimeIndex(data.ds),
                        "time_features":time_features,
                        "ignore_const_cols":ignore_const_cols}
        calendar_features = compute_calendar_features(**input_params)
        all_features_list.append(calendar_features)

    lag_kwargs = [{"lag":lag} for lag in lags]  
    rw_kwargs =  [{"window_shift":window_shift, "window_func":window_func, "window_size":window_size}
                    for window_shift in window_shifts
                    for window_func in window_functions
                    for window_size in window_sizes]
    input_kwargs = lag_kwargs + rw_kwargs

    grouped = data.loc[:, ts_uid_columns+["y"]].groupby(ts_uid_columns)["y"]
    with Parallel(n_jobs=n_jobs) as parallel:
        delayed_func = delayed(compute_lagged_train_feature)
        lagged_features = parallel(delayed_func(grouped, **kwargs) for kwargs in input_kwargs)
        lagged_features = pd.DataFrame({feature.name:feature.values for feature in lagged_features})
        all_features_list.append(lagged_features)

    # merging all features
    all_features = pd.concat(all_features_list, axis=1)
    all_features.set_index(data.index, inplace=True)
    return all_features

def compute_predict_features(data, ts_uid_columns, time_features, lags, window_shifts,
                             window_sizes, window_functions, ignore_const_cols=True):
    """
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with (at least) columns: 'ds' and 'y'.
    ts_uid_columns: list
        List of columns names that are unique identifiers for time series.
    time_features: list
        Time attributes to include as features.
    lags: list
        List of integer lag values to include as features.
    window_shifts: list
        List of integer window shift values.
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
    all_features_list.append(data.reset_index(drop=True))

    # generating the time features
    if len(time_features) > 0:
        input_params = {"date_range":pd.DatetimeIndex(data.ds),
                        "time_features":time_features,
                        "ignore_const_cols":ignore_const_cols}
        calendar_features = compute_calendar_features(**input_params)
        all_features_list.append(calendar_features)

    if len(lags) > 0:
        column_names = [f"lag{lag}" for lag in lags]
        lag_features = pd.DataFrame(np.nan, index=range(len(data)), 
                                    columns=column_names)
        all_features_list.append(lag_features)

    if (len(window_sizes) > 0) & (len(window_functions) > 0):
        column_names = [f"{window_func}{window}_shift{window_shift}"
                        for window_shift in window_shifts
                        for window_func in window_functions
                        for window in window_sizes]
        rw_features = pd.DataFrame(np.nan, index=range(len(data)),
                                    columns=column_names)
        all_features_list.append(rw_features)
    
    # merging all features
    all_features = pd.concat(all_features_list, axis=1)
    all_features.set_index(data.index, inplace=True)
    return all_features

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

def compute_lagged_train_feature(grouped, lag=None, window_shift=None, window_func=None, window_size=None):
    """
    grouped: pandas.core.groupby.generic.SeriesGroupBy
        Groupby object containing the response variable "y"
        grouped by ts_uid_columns.
    lag: int
        integer lag value.
    window_shift: int
        integer window shift value.
    window_func: string
        string names of the window function.
    window_size: int
        integer window sizes value.
    """
    is_lag_feature = lag is not None
    is_rw_feature = (window_shift is not None) and (window_func is not None) and (window_size is not None)

    if is_lag_feature and not is_rw_feature:
        feature_values = grouped.shift(lag)
        feature_values.name = f"lag{lag}"
    elif is_rw_feature and not is_lag_feature:
        feature_values = grouped.apply(lambda x: getattr(x.shift(window_shift).rolling(window_size), window_func)())
        feature_values.name = f"{window_func}{window_size}_shift{window_shift}"
    else:
        raise ValueError("Invalid input parameters.")

    return feature_values

def compute_lagged_predict_feature(grouped, lag=None, window_shift=None, window_func=None, window_size=None):
    """
    grouped: pandas.core.groupby.generic.SeriesGroupBy
        Groupby object containing the response variable "y"
        grouped by ts_uid_columns.
    lag: int
        integer lag value.
    window_shift: int
        integer window shift value.
    window_func: string
        string names of the window function.
    window_size: int
        integer window sizes value.
    """
    is_lag_feature = lag is not None
    is_rw_feature = (window_shift is not None) and (window_func is not None) and (window_size is not None)
    if is_lag_feature and not is_rw_feature:
        feature_values = grouped.apply(lambda x: x.iloc[-lag])
        feature_values.name = f"lag{lag}"
    elif is_rw_feature and not is_lag_feature:
        lidx = -(window_size + window_shift-1)
        ridx = -(window_shift-1) if window_shift > 1 else None
        feature_values = grouped.apply(lambda x: getattr(x.iloc[lidx:ridx], window_func)())
        feature_values.name = f"{window_func}{window_size}_shift{window_shift}"
    else:
        raise ValueError("Invalid input parameters.")
        
    return feature_values
