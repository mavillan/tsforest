import itertools
import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", -1)
from joblib import Parallel, delayed
from tqdm import tqdm


def fit_evaluate(model_class, model_config, model_params, train_data, valid_indexes, metric):
    """
    Parameters
    ----------
    model_class_path: tuple
        Class path of the model: (module_name, class_name)
    """
    errors = dict()
    best_iterations = dict()
    model = model_class(model_params=model_params, **model_config)
    for i,valid_index in enumerate(valid_indexes):
        model.fit(train_data, valid_index)
        eval_data = train_data.loc[valid_index, :]
        errors[f"fold{i}"] = model.evaluate(eval_data, metric=metric)
        best_iterations[f"fold{i}"] = model.best_iteration
    return (model_params,
            np.mean(list(best_iterations.values())),
            best_iterations,
            np.mean(list(errors.values())),
            errors)

class GridSearch(object):
    """
    model_class: Class
        Class of the model to be instantiated.
    feature_sets: list
        List of feature sets to be included.
    exclude_features: list
        List of features to be excluded from training dataframe.
    categorical_features: dict
        Dict with the name of the categorical feature as keys, and the name
        of the class in 'category_encoders' to be used for encoding as values.
    calendar_anomaly: list
        List of names of calendar features affected by an anomaly.
    ts_uid_columns: list
        List of columns names that are unique identifiers for time series.
    detrend: bool
        Whether or not to remove the trend from time serie.
    target_scaler: str
        Class in sklearn.preprocessing to perform scaling of the target variable.
    target_scaler_kwargs: dict
        Extra arguments passed to the target_scaler class constructor when instantiating.
    lags: list
        List of integer lag values.
    window_sizes: list
        List of integer window sizes values.
    window_functions: list
        List of string names of the window functions.
    hyperparams: dict
        Dictionary of hyperparameters in the form: 'parameter_name':[param_value,].
    hyperparams_fixed: dict
        Dictionary of fixed hyperparameters in the form: {'param_name':'param_value',}.
    n_jobs: int
        Number of parallel jobs to run on grid search.
    """

    def __init__(self, model_class, feature_sets=["calendar", "calendar_cyclical"], 
                 exclude_features=list(), categorical_features=dict(), calendar_anomaly=list(), 
                 ts_uid_columns=list(), detrend=True, target_scaler="StandardScaler", 
                 target_scaler_kwargs=dict(), lags=None, window_sizes=None, window_functions=None, 
                 hyperparams=dict(), hyperparams_fixed=dict(), n_jobs=-1):
        self.model_class = model_class
        self.feature_sets = feature_sets
        self.exclude_features = exclude_features
        self.categorical_features = categorical_features
        self.calendar_anomaly = calendar_anomaly
        self.ts_uid_columns = ts_uid_columns
        self.detrend = detrend
        self.target_scaler = target_scaler
        self.target_scaler_kwargs = target_scaler_kwargs
        self.lags = lags
        self.window_sizes = window_sizes
        self.window_functions = window_functions
        self.hyperparams = hyperparams
        self.hyperparams_fixed = hyperparams_fixed
        self.n_jobs = n_jobs

    def fit(self, train_data, valid_indexes=None, metric="rmse"):
        """
        Parameters
        ----------
        train_data : pandas.DataFrame
            Dataframe with at least columns 'ds' and 'y'.
        valid_indexes: iterable list | numpy.ndarray | pandas.Index
            Iterable object contaning the indexes from 'train_data' to be used for validation.
        metric: string
            Name of the error metric to be used.
        Returns
        ----------
        """
        params_names = self.hyperparams.keys()
        params_values = self.hyperparams.values()
        hyperparams_fixed = self.hyperparams_fixed
        _hyperparams_list = [dict(zip(params_names,combination)) 
                             for combination in itertools.product(*params_values)]
        hyperparams_list = [{**hyperparams, **hyperparams_fixed} 
                            for hyperparams in _hyperparams_list]

        # parallel fit & evaluation of model on hyperparams
        model_config = {"feature_sets":self.feature_sets,
                        "exclude_features":self.exclude_features,
                        "categorical_features":self.categorical_features,
                        "calendar_anomaly":self.calendar_anomaly,
                        "ts_uid_columns":self.ts_uid_columns,
                        "detrend":self.detrend,
                        "target_scaler":self.target_scaler,
                        "target_scaler_kwargs":self.target_scaler_kwargs,
                        "lags":self.lags,
                        "window_sizes":self.window_sizes,
                        "window_functions":self.window_functions}
        kwargs = {"model_class":self.model_class,
                  "model_config":model_config,
                  "train_data":train_data,
                  "valid_indexes":valid_indexes,
                  "metric":metric}
        
        with Parallel(n_jobs=self.n_jobs) as parallel:
            delayed_func = delayed(fit_evaluate)
            with tqdm(hyperparams_list) as tqdm_hyperparams_list:
                _results = parallel(delayed_func(model_params=model_params, **kwargs)
                                    for model_params in tqdm_hyperparams_list)
            tqdm_hyperparams_list.close()
        # removes fixed hyperparams from results
        results = [({key:value for key,value in r[0].items() 
                     if key not in hyperparams_fixed.keys()}, r[1], r[2], r[3], r[4]) 
                    for r in _results]
        # sort the results by error
        results.sort(key = lambda x: x[-2])
        columns = ["hyperparams", "best_iteration", "best_iteration_by_fold", "error", "error_by_fold"]
        results_dataframe = pd.DataFrame(results, columns=columns)
        self.results = results
        self.results_dataframe = results_dataframe
        
    def get_grid(self, top_n=10):
        """
        Parameters
        ----------
        top_n: int
            number of best parameters to return
        """
        return self.results_dataframe.head(top_n)

    def get_best_params(self, top_n=1):
        """
        Parameters
        ----------
        top_n: int
            number of best parameters to return
        """
        assert top_n <= len(self.results), "'top_n' cannot be greater that the grid size"
        return [self.results[i][0] for i in range(top_n)]
