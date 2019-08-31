# h2o gbm default parameters
gbm_parameters = {
    'ntrees':500,
    'max_depth':5,
    'nbins':20,
    'nbins_top_level':1024,
    'learn_rate':0.1,
    'stopping_metric':'mse',
    'stopping_rounds':25,
    'score_each_iteration':True,
    'categorical_encoding':'enum',
    'sample_rate':1.0,
    'col_sample_rate':0.9,
    'min_rows':10,
    'seed':23
}

# lightgbm default parameters
lgbm_parameters = {
    'boosting_type':'gbrt',
    'objective':'regression',
    'num_leaves':31,
    'learning_rate':0.1,
    'feature_fraction':0.9,
    'num_iterations':500
}

# fbprophet default parameters 
prophet_kwargs = {
    'n_changepoints':120,
    'changepoint_range':0.97, 
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
    'year_day':'numeric',
    'year_week':'numeric',
    'month':'categorical'
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

#  datatype of general calendar features
calendar_general_features_types = {
    'month_day':'numeric',
    'month_progress':'numeric',
    'year':'numeric',
    'sequence_day':'numeric',
    'fsl_type':'categorical',
    'fsl_day':'categorical',
    'prev_fsl':'categorical',
    'post_fsl':'categorical'
}

# datatype of events features
events_features_types = {
    'closed':'numeric',
    'holiday':'categorical',
    'prev_christmas':'categorical',
    'prev_newyear':'categorical',
    'prev_18':'categorical',
    'prev_trabajo':'categorical',
    'mil_a_mil':'categorical',
    'bf':'categorical',
    'ss_fds':'categorical',
    'prev_ss_fds':'categorical',
    'prev_closed':'categorical', 
    'prev_holiday':'categorical'
}

# datatype of store opening features
store_opening_features_types = {
    'fecha':'numeric',
    'mts_sq_bodega':'numeric',
    'mts_sq_ekono':'numeric',
    'mts_sq_express':'numeric',
    'mts_sq_lider':'numeric',
    'mts_sq_mayorista':'numeric',
    'mts_sq_total':'numeric'
}

# datatype of all features
all_features_types = {
    **calendar_sequential_features_types,
    **calendar_cyclical_features_types,
    **calendar_general_features_types,
    **events_features_types,
    **store_opening_features_types
}
