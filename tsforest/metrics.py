import numpy as np
import pandas as pd

def compute_mae(yreal, ypred):
    # not considering yreal and ypred values when yreal = 0
    mask = yreal>0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.mean(np.abs(yreal-ypred))

def compute_mse(yreal, ypred):
    # not considering yreal and ypred values when yreal = 0
    mask = yreal>0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.mean((yreal-ypred)**2)

def compute_rmse(yreal, ypred):
    # not considering yreal and ypred values when yreal = 0
    mask = yreal>0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.sqrt(np.mean((yreal-ypred)**2))

def compute_mape(yreal, ypred):
    # not considering yreal and ypred values when yreal = 0
    mask = yreal>0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.mean( np.abs(yreal-ypred)/yreal )

def compute_smape(yreal, ypred):
    # not considering yreal and ypred values when yreal = 0
    mask = yreal>0
    yreal = yreal[mask]
    ypred = ypred[mask]
    N = yreal.shape[0]
    return np.sum(np.abs(yreal-ypred)/(np.abs(yreal)+np.abs(ypred)))/N

def compute_accuracy(yreal, ypred):
    # not considering yreal and ypred values when yreal = 0
    mask = ypred>0
    ypred = ypred[mask]
    yreal = yreal[mask]
    N = yreal.shape[0]
    return np.mean( np.minimum(np.ones(N), \
                    np.maximum(np.zeros(N), 1-np.abs(ypred-yreal)/ypred)) )