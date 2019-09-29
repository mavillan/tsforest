# h2o gbm default parameters
gbm_parameters = {
    'ntrees':300,
    'max_depth':5,
    'nbins':20,
    'learn_rate':0.25,
    'stopping_metric':'mse',
    'score_each_iteration':True,
    'categorical_encoding':'enum',
    'sample_rate':1.0,
    'col_sample_rate':1.0,
    'min_rows':5,
    'distribution':'gaussian'
}

# lightgbm default parameters
lgbm_parameters = {
    'boosting_type':'gbrt',
    'objective':'regression',
    'num_leaves':31,
    'min_data_in_leaf':5,
    'learning_rate':0.25,
    'feature_fraction':1.0,
    'num_iterations':300
}

cat_parameters = {
    'iterations':300,
    'learning_rate':0.25,
    'l2_leaf_reg':3.0,
    'depth':6,
    'has_time':True,
    'bootstrap_type':'No'
}

# fbprophet default parameters 
prophet_kwargs = {
    'n_changepoints':120,
    'changepoint_range':0.9,
    'changepoint_prior_scale':0.05
}
prophet_kwargs_extra = {
    'wk_fourier_order' : 3,
    'wk_prior_scale'   : 10.,
    'yr_fourier_order' : 10,
    'yr_prior_scale'   : 10.
} 

# datatype of sequential calendar features
calendar_sequential_features_types = {
    'week_day':'numeric',
    'month_day':'numeric',
    'year_day':'numeric',
    'year_week':'numeric',
    'month':'categorical',
    'year':'numeric',
    'month_progress':'numeric'
}

# datatype of cyclical calendar features
calendar_cyclical_features_types = {
    'week_day_cos':'numeric',
    'week_day_sin':'numeric',
    'year_day_cos':'numeric',
    'year_day_sin':'numeric',
    'year_week_cos':'numeric',
    'year_week_sin':'numeric',
    'month_cos':'numeric',
    'month_sin':'numeric'
}

# datatype of all features
all_features_types = {
    **calendar_sequential_features_types,
    **calendar_cyclical_features_types
}
