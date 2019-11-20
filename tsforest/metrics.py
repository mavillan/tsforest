import numpy as np

def compute_mae(yreal, ypred):
    mask = yreal!=0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.mean(np.abs(yreal-ypred))

def compute_mse(yreal, ypred):
    mask = yreal!=0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.mean((yreal-ypred)**2)

def compute_rmse(yreal, ypred):
    mask = yreal!=0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.sqrt(np.mean((yreal-ypred)**2))

def compute_rmsle(yreal, ypred):
    mask = yreal!=0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.sqrt(np.mean((np.log(ypred+1) - np.log(yreal+1))**2))
    
def compute_mape(yreal, ypred):
    mask = yreal!=0
    yreal = yreal[mask]
    ypred = ypred[mask]
    return np.mean( np.abs(yreal-ypred)/yreal )

def compute_smape(yreal, ypred):
    mask = yreal!=0
    yreal = yreal[mask]
    ypred = ypred[mask]
    N = yreal.shape[0]
    return np.sum(np.abs(yreal-ypred)/(np.abs(yreal)+np.abs(ypred)))/N
