import itertools
import pandas as pd
pd.set_option('display.max_colwidth', -1)
from joblib import Parallel, delayed


def fit_evaluate(model_class, model_config, model_params, train_data, valid_period, eval_period, metric):
    '''
    Parameters
    ----------
    model_class_path: tuple
        Class path of the model: (module_name, class_name)
    '''
    model = model_class(model_params=model_params, **model_config)
    model.fit(train_data, valid_period)
    error = model.evaluate(eval_period, metric=metric)
    error = 100*error/eval_period.y.mean()
    return (model_params, model.best_iteration, error)

class GridSearch(object):
    '''
    model_class: Class
        class of the model to be instantiated
    features: list
        list of features to be included
    detrend: bool
        whether or not to remove the trend from time serie
    response_scaling:
        Whether or not to perform scaling of the reponse variable
    lags: list
        List of integer lag values
    window_sizes: list
        List of integer window sizes values
    window_functions: list
        List of string names of the window functions
    hyperparams: dict
        dictionary of hyperparameters in the form: "parameter_name" : [param_value,]
    hyperparams_fixed: dict
        dictionary of fixed hyperparameters in the form: {"param_name" : "param_value",}
    n_jobs: int
        number of parallel jobs to run on grid search
    '''

    def __init__(self, model_class, features=['calendar_mixed','events'], 
                 detrend=True, response_scaling=True, lags=None, 
                 window_sizes=None, window_functions=None, 
                 hyperparams=dict(), hyperparams_fixed=dict(), 
                 n_jobs=-1):
        self.model_class = model_class
        self.features = features
        self.detrend = detrend
        self.response_scaling = response_scaling
        self.lags = lags
        self.window_sizes = window_sizes
        self.window_functions = window_functions
        self.hyperparams = hyperparams
        self.hyperparams_fixed = hyperparams_fixed
        self.n_jobs = n_jobs

    def fit(self, train_data, valid_period=None, eval_period=None, metric='rmse'):
        '''
        Parameters
        ----------
        train_data : pandas.DataFrame
            dataframe with at least columns "ds" and "y"
        valid_period: pandas.Series or pandas.DataFrame
            series or dataframe (with column "ds") indicating the validation period
        eval_period : pandas.Series or pandas.DataFrame
            series or dataframe (with column "ds") indicating the validation period
        metric: string
            name of the error metric to be measured
        Returns
        ----------
        '''
        params_names = self.hyperparams.keys()
        params_values = self.hyperparams.values()
        hyperparams_fixed = self.hyperparams_fixed
        _hyperparams_list = [dict(zip(params_names,combination)) 
                             for combination in itertools.product(*params_values)]
        hyperparams_list = [{**hyperparams, **hyperparams_fixed} 
                            for hyperparams in _hyperparams_list]

        if (valid_period is None) and (eval_period is None):
            eval_period = train_data.loc[:,['ds','y']]
        elif eval_period is None:
            eval_period = pd.merge(train_data, valid_period, how='inner', on=['ds'])

        # parallel fit & evaluation of model on hyperparams
        model_config = {'features':self.features,
                        'detrend':self.detrend,
                        'response_scaling':self.response_scaling,
                        'lags':self.lags,
                        'window_sizes':self.window_sizes,
                        'window_functions':self.window_functions}
        kwargs = {'model_class':self.model_class,
                  'model_config':model_config,
                  'train_data':train_data,
                  'valid_period':valid_period,
                  'eval_period':eval_period,
                  'metric':metric}
        
        with Parallel(n_jobs=self.n_jobs) as parallel:
            delayed_func = delayed(fit_evaluate)
            _results = parallel(delayed_func(model_params=model_params, **kwargs)
                                for model_params in hyperparams_list)
        # removes fixed hyperparams from results
        results = [({key:value for (key,value) in r[0].items() 
                     if key not in hyperparams_fixed.keys()}, r[1], r[2]) 
                    for r in _results]
        # sort the results by error
        results.sort(key = lambda x: x[-1])
        results_dataframe = pd.DataFrame([(r[0],r[1],r[2]) for r in results],
                                columns=["hyperparams","best_iteration","error"])
        self.results = results
        self.results_dataframe = results_dataframe
        
    def get_grid(self, top_n=10):
        '''
        Parameters
        ----------
        top_n: int
            number of best parameters to return
        '''
        return self.results_dataframe.head(top_n)

    def get_best_params(self, top_n=1):
        '''
        Parameters
        ----------
        top_n: int
            number of best parameters to return
        '''
        assert top_n <= len(self.results), \
            '"top_n" cannot be greater that the grid size'
        return [self.results[i][0] for i in range(top_n)]
