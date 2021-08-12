import numpy as np
import pandas as pd

AVAILABLE_SCALERS = ["maxabs", "minmax", "robust", "standard"]

class StandardScaler():

    def __init__(self, ts_uid_columns):
        self.ts_uid_columns = ts_uid_columns

    def fit(self, dataframe):
        mean = dataframe.groupby(self.ts_uid_columns)["y"].mean().reset_index(name="y_mean")
        std =  dataframe.groupby(self.ts_uid_columns)["y"].std().reset_index(name="y_std")
        std.loc[std.query("y_std == 0").index, "y_std"] = 1
        self.params = pd.merge(mean, std, how="inner", on=self.ts_uid_columns)

    def transform(self, dataframe, target="y"):
        dataframe = pd.merge(dataframe, self.params, how="inner", on=self.ts_uid_columns)
        dataframe[target] = dataframe.eval(f"({target}-y_mean)/y_std")
        return dataframe

    def inverse_transform(self, dataframe, target="y"):
        dataframe = pd.merge(dataframe, self.params, how="inner", on=self.ts_uid_columns)
        dataframe[target] = dataframe.eval(f"({target}*y_std) + y_mean")
        return dataframe
