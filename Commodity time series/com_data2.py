# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 08:52:47 2016

@author: WZhao10
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

def calc_score(y_true, y_pred):
    residuals = [y_true[k]-y_pred[k] for k in range(len(y_true))]
    squared_residuals = [resi**2 for resi in residuals]
    sum_squared_residuals = reduce(lambda x,y: x+y, squared_residuals)
    
    y_mean = sum(y_true)/float(len(y_true))
    squares = [(data-y_mean)**2 for data in y_true]
    sum_squares = reduce(lambda x,y: x+y, squares)
    r2 = 1-sum_squared_residuals/sum_squares
    return r2

def moving_window_arma(df, varname, indexes, p, q):
    train_num = 60
    val_num = 10
    rmse_list = []
    for index in indexes:
        y_train = df[varname].values[index:index+train_num]
        y_val = df[varname].values[index+train_num:index+train_num+val_num]
        try:
            model_arma = ARMA(y_train, order = (p, q)).fit()
            y_hat = model_arma.predict(train_num, train_num+val_num)
            y_hat = y_hat[1:]
            rmse = calc_score(y_val, y_hat)
            rmse_list.append(rmse)
        except:
            pass
    return rmse_list
            
def best_arma(df, varname, indexes):
    bestModel = []
    bestScore = -1e10
    
    pq = [(4, 2), (7, 4), (10, 5)]
    
    for (p,q) in pq:
        rmse_list = moving_window_arma(df, varname, indexes, p, q)
        score = np.nanmean(rmse_list)
        if score > bestScore:
            bestModel = (p, q)
            bestScore = score
            
    return bestModel, bestScore
                
def moving_window_gpr(df, varname, indexes, gp):
    train_num = 60
    val_num = 10
    X_train = np.arange(0, train_num).reshape(-1,1)
    rmse_list = []
    for index in indexes:
        y_train = df[varname].values[index:index+train_num]
        y_val = df[varname].values[index+train_num:index+train_num+val_num]
        try:
            model_gpr = gp.fit(X_train, y_train)
            y_hat = model_gpr.predict(np.linspace(X_train.max()+1, X_train.max() + val_num, val_num)[:, np.newaxis])
            rmse = calc_score(y_val, y_hat)
            rmse_list.append(rmse)
        except:
            pass
    return rmse_list

def best_gpr(df, varname, indexes):
    bestModel = None
    bestScore = -1e10
    
    kernel_irreg = 0.5**2 * RationalQuadratic(length_scale=10, alpha=1)
    kernel_noise = 0.1**2 * RBF(length_scale=0.1) \
                    + WhiteKernel(noise_level=0.1**2,
                                  noise_level_bounds=(1e-3, np.inf))  
    
    periodicities = [5, 10, 20]
    length_scales = [100, 200, 300]
    for periodicity in periodicities:
        for length_scale in length_scales:
            kernel_longterm = 50**2 * RBF(length_scale=length_scale)
            kernel_season = 3**2 * RBF(length_scale=length_scale) \
                            * ExpSineSquared(length_scale=1, periodicity=10,
                                             periodicity_bounds="fixed")
            kernel = kernel_longterm + kernel_season + kernel_irreg + kernel_noise
            gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              optimizer=None, normalize_y=True)
            rmse_list = moving_window_gpr(df, varname, indexes, gp)
            score = np.nanmean(rmse_list)
            if score > bestScore:
                bestModel = gp
                bestScore = score
                
    return bestModel, bestScore
    
#def best_gpr(df, varname, indexes):
#    bestModel = None
#    bestScore = -1e10
#    
#    kernel_irreg = 0.5**2 * RationalQuadratic(length_scale=1, alpha=1)
#    kernel_noise = 0.1**2 * RBF(length_scale=0.1) \
#                    + WhiteKernel(noise_level=0.1**2,
#                                  noise_level_bounds=(1e-3, np.inf))  
#    
#    periodicities = [5, 10, 20]
#    length_scales = [100, 200, 300]
#    for periodicity in periodicities:
#        for length_scale in length_scales:
#            kernel_longterm = 50**2 * RBF(length_scale=length_scale)
#            kernel_season = 3**2 * RBF(length_scale=200) \
#                            * ExpSineSquared(length_scale=1, periodicity=10,
#                                             periodicity_bounds="fixed")
#            kernel = kernel_longterm + kernel_season + kernel_irreg + kernel_noise
#            gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
#                              optimizer=None, normalize_y=True)
#            rmse_list = moving_window_gpr(df, varname, indexes, gp)
#            score = np.nanmean(rmse_list)
#            if score > bestScore:
#                bestModel = gp
#                bestScore = score
#                
#    return bestModel, bestScore
    
#%% load data

os.chdir('D:\Documents\Code')
varnames = ['var1', 'var2', 'var3']
df = pd.read_excel('input_data.xlsx', header=None, names = varnames)
df.rename(columns = {0:'var1', 1:'var2', 2:'var3'}, inplace=True)

#%%
train_num = 60
val_num = 10

results = pd.DataFrame(columns = ['Variable', 'Model', 'Score', 'Directionality', 'Volatility'])
results['Variable'] = varnames

for varname in varnames:
    indexes = np.random.choice(range(len(df)-(train_num+val_num)), 20, replace=False)
    model_arma, score_arma = best_arma(df, varname, indexes)
    model_gpr, score_gpr = best_gpr(df, varname, indexes)
    
    best_score, best_model_str, best_model = max(zip([score_arma, score_gpr], 
                                                     ['arma', 'gpr'], 
                                                     [model_arma, model_gpr]))
    
    X_test = np.arange(0, train_num).reshape(-1,1)
    y_test = df.loc[len(df)-train_num:, varname].values

    if best_model_str is 'arma':
        p_arma = model_arma[0]; q_arma = model_arma[1];
        try:
            model = ARMA(y_test, order = (p_arma, q_arma)).fit()
            model_arma.predict(train_num, train_num+val_num)
        except:
            best_model_str = 'gpr'
            model = model_gpr.fit(X_test, y_test)
            y_pred = model_gpr.predict(np.linspace(X_test.max()+1, X_test.max() + val_num, val_num)[:, np.newaxis])
    else:
        model = model_gpr.fit(X_test, y_test)
        y_pred = model_gpr.predict(np.linspace(X_test.max()+1, X_test.max() + val_num, val_num)[:, np.newaxis])
    
    directionality = 1 if y_pred[-1]-y_pred[0]>0 else 0
    volatility = np.std(y_pred)
    
    result_index = results['Variable']==varname
    results.loc[result_index, 'Model'] = best_model_str
    results.loc[result_index, 'Score'] = best_score
    results.loc[result_index, 'Directionality'] = directionality
    results.loc[result_index, 'Volatility'] = volatility

print results

#%%


