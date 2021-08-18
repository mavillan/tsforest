# h2o gbm default parameters
gbm_parameters = {
    "ntrees":300,
    "max_depth":5,
    "nbins":20,
    "learn_rate":0.25,
    "stopping_rounds":30,
    "stopping_metric":"mse",
    "score_each_iteration":True,
    "categorical_encoding":"enum",
    "sample_rate":1.0,
    "col_sample_rate":1.0,
    "min_rows":5,
    "distribution":"gaussian"
}

# lightgbm default parameters
lgbm_parameters = {
    "boosting_type":"gbrt",
    "objective":"regression",
    "num_iterations":300,
    "num_leaves":31,
    "min_data_in_leaf":5,
    "learning_rate":0.25,
    "feature_fraction":1.0,
    "early_stopping_rounds":30
}

# catboost default parameters
cat_parameters = {
    "iterations":300,
    "learning_rate":0.25,
    "l2_leaf_reg":3.0,
    "depth":6,
    "has_time":True,
    "bootstrap_type":"No",
    "early_stopping_rounds":30
}

# xgboost default parameters
xgb_parameters = {
    "objective":"reg:squarederror",
    "learning_rate":0.25,
    "max_depth":6,
    "lambda":1,
    "num_boost_round":300,
    "early_stopping_rounds":30
}

# fbprophet default parameters 
default_prophet_kwargs = {
    "n_changepoints":120,
    "changepoint_range":0.9,
    "changepoint_prior_scale":0.05,
    "uncertainty_samples":0,
    "wk_fourier_order" : 3,
    "wk_prior_scale"   : 10.,
    "yr_fourier_order" : 10,
    "yr_prior_scale"   : 10.
} 

# names of calendar features
calendar_features_names = [
    "year",
    "quarter",
    "month",
    "days_in_month",
    "year_week",
    "year_day",
    "month_day",
    "week_day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
    "month_progress"
]

# names of cyclical calendar features
calendar_cyclical_features_names = [
    "second_cos",
    "second_sin",
    "minute_cos",
    "minute_sin",
    "hour_cos",
    "hour_sin",
    "week_day_cos",
    "week_day_sin",
    "year_day_cos",
    "year_day_sin",
    "year_week_cos",
    "year_week_sin",
    "month_cos",
    "month_sin"
]
