import abc

class BaseRegressor(abc.ABC):
    @abc.abstractmethod
    def cast_dataframe(self, features_dataframe, input_features, target,  categorical_features):
        """
        Parameters
        ----------
        features_dataframe: pandas.DataFrame
            Dataframe containing all the features.
        input_features: list
            List of column names to use as predictor features.
        target: str
            Name of column to use as target variable.
        categorical_features: list
            List of column names of categorical features.
        Returns
        ----------
        features_dataframe_casted
            Features dataframe casted to the corresponding dataframe format.
        """
        pass
    
    @abc.abstractmethod
    def fit(self, train_features, valid_features, input_features, target, categorical_features):
        """
        Parameters
        ----------
        train_features: pandas.DataFrame
            Dataframe containing the features corresponding to the training data.
        valid_features: pandas.DataFrame
            Dataframe containing the features corresponding to the validation period.
        input_features: list
            List of column names to use as predictor features.
        target: str
            Name of column to use as target variable.
        categorical_features: list
            List of column names of categorical features.
        """
        pass

    @abc.abstractmethod
    def tune(self, train_features, valid_features, input_features, target, categorical_features):
        """
        Parameters
        ----------
        train_features: pandas.DataFrame
            Dataframe containing the features corresponding to the training data.
        valid_features: pandas.DataFrame
            Dataframe containing the features corresponding to the validation period.
        input_features: list
            List of column names to use as predictor features.
        target: str
            Name of column to use as target variable.
        categorical_features: list
            List of column names of categorical features.
        """
        pass

    @abc.abstractmethod
    def predict(self, predict_features):
        """
        Parameters
        ----------
        predict_features: pandas.DataFrame
            Dataframe containing the features corresponding to the prediction period.
        Returns
        ----------
        predict: numpy.ndarray
            Array with the predicted values.
        """
        pass
