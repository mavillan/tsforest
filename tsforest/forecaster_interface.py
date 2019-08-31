from zope.interface import Interface

class ForecasterInterface(Interface):
    def __init__(self, model_params, features, detrend):
        pass

    def _prepare_train_data(self, train_data):
        pass
    
    def _prepare_test_data(self, test_data):
        pass

    def _cast_dataframe(self, features_dataframe, features_types):
        pass

    def fit(self, train_data, valid_data=None, early_stopping_rounds=20):
        pass

    def predict(self, test_data):
        pass

    def evaluate(self, test_data, metric='rmse'):
        pass

    def show_variable_importance(self):
        pass

    def save_model(self):
        pass