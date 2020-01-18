import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn import preprocessing

AVAILABLE_SCALERS = ["MaxAbsScaler", "MinMaxScaler", "Normalizer", 
                     "RobustScaler", "StandardScaler"]

def validate_inputs():
    if target_scaler is not None:
        if not isinstance(target_scaler, str):
            raise TypeError("Parameter 'target_scaler' should be of type 'str'.")
        elif target_scaler not in AVAILABLE_SCALERS:
            raise ValueError(f"Parameter 'target_scaler' should be any of: {AVAILABLE_SCALERS}.")
    
    if not isinstance(target_scaler_kwargs, dict):
        raise TypeError("Parameter 'target_scaler_kwargs' should be of type 'dict'.")


def compute_scaler(train_data_chunk, key, target_scaler, target_scaler_kwargs):
    scaler_class = getattr(preprocessing, target_scaler)
    scaler = scaler_class(**target_scaler_kwargs)
    scaler.fit(train_data_chunk.y.values.reshape(-1,1))
    return {key:scaler}

def compute_scalers(data, valid_index=pd.Index([]), ts_uid_columns=None, 
                    target_scaler="StandardScaler", target_scaler_kwargs=dict(), 
                    n_jobs=-1):
    train_data = data.drop(valid_index, axis=0)
    if ts_uid_columns is None:
        train_data["ts_uid"] = 0
        ts_uid_columns = ["ts_uid"]
    ts_uid_values = train_data.loc[:, ts_uid_columns].drop_duplicates()

    train_data_split = list()
    for _,row in ts_uid_values.iterrows():
        key = tuple([item for _,item in row.iteritems()])
        query_string = " & ".join([f"{col_name}=={value}" for col_name,value in row.iteritems()])
        train_data_chunk = train_data.query(query_string)
        train_data_split.append((train_data_chunk, key))

    with Parallel(n_jobs=n_jobs) as parallel:
        delayed_func = delayed(compute_scaler)
        with tqdm(train_data_split) as tqdm_train_data_split:
            scalers = parallel(delayed_func(*data_chunk, target_scaler=target_scaler, 
                                            target_scaler_kwargs=target_scaler_kwargs)
                               for data_chunk in tqdm_train_data_split)
        tqdm_train_data_split.close()

    scalers_out = {}
    for scaler in scalers: scalers_out.update(scaler)
    return scalers_out
