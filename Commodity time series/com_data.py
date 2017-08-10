#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 19:43:21 2016

@author: Wenyu
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gplearn.genetic import SymbolicRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import zero_one_loss

def is_up_in_days(df, varname, no):
    var_diff = df[varname].diff(periods=no)[no:]
    is_up_in_days = np.array([True if np.sign(num)==1 else False for num in var_diff]).reshape(len(df)-no,1)
    is_up_in_days = np.vstack((is_up_in_days, np.nan*np.ones((no,1))))
    return is_up_in_days

def vol_in_days(df, varname, no):
    var_rolling_std = df[varname].rolling(window=no,center=False).std(ddof=0).values[no:].reshape(len(df)-no,1)
    vol_in_days = np.vstack((var_rolling_std, np.nan*np.ones((no,1))))
    return vol_in_days
    
def is_up_past_days(df, varname, no):
    var_diff = df[varname].diff(periods=no)[no:]
    is_up_in_days = np.array([True if np.sign(num)==1 else False for num in var_diff]).reshape(len(df)-no,1)
    is_up_past_days = np.vstack((np.nan*np.ones((no,1)), is_up_in_days))
    return is_up_past_days
    
def vol_past_days(df, varname, no):
    vol_past_days = df[varname].rolling(window=no,center=False).std(ddof=0).values
    return vol_past_days
    
def feat_past_days(df, varname, no, feat):
    var_rolling_window = df[varname].rolling(window=no, center=False)
    var_rolling_feat = getattr(var_rolling_window, feat)()
    return var_rolling_feat
    
def best_gp(X_train, y_train, X_val, y_val, classifier=True):
    bestModel = None
    bestScore = 0 if classifier else -1e10

    popSizes = [1000, 2000, 5000]
    highParams = range(4)
    
    for popSize in popSizes:
        for highParam in highParams:
            params = [0.1]*4
            params[highParam] = 0.7
            model_gp = SymbolicRegressor(population_size=popSize,
                                         generations=25, stopping_criteria=0.01,
                                         p_crossover=params[0], p_subtree_mutation=params[1],
                                         p_hoist_mutation=params[2], p_point_mutation=params[3],
                                         max_samples=0.9, verbose=0,
                                         parsimony_coefficient=0.01, random_state=42)
            model_gp.fit(X_train, y_train.values.ravel())
            if classifier:
                y_hat = model_gp.predict(X_val)
                y_hat = np.round(y_hat)
                score = 1 - zero_one_loss(y_val, y_hat)
                
            else:
                score = model_gp.score(X_val, y_val)
            
            if (score > bestScore):
                bestModel = model_gp
                bestScore = score
                
    return bestModel, bestScore
    
def best_nn(X_train, y_train, X_val, y_val, classifier=True):
    bestModel = None
    bestScore = 0 if classifier else -1e10

    layerSizes = [50, 100, 200]
    Solvers= ['adam', 'sgd']
    
    for layerSize in layerSizes:
        for solver in Solvers:
            if classifier:
                model_nn = MLPClassifier(hidden_layer_sizes=layerSize, solver=solver)
            else:
                model_nn = MLPRegressor(hidden_layer_sizes=layerSize, solver=solver)
            
            model_nn.fit(X_train, y_train.values.ravel())
            if classifier:
                y_hat = model_nn.predict(X_val)
                score = 1 - zero_one_loss(y_val, y_hat)
                
            else:
                score = model_nn.score(X_val, y_val)
            
            if (score > bestScore):
                bestModel = model_nn
                bestScore = score
                
    return bestModel, bestScore
    
def best_rf(X_train, y_train, X_val, y_val, classifier=True):
    bestModel = None
    bestScore = 0 if classifier else -1e10
    
    numEsts = [50, 100, 200]
    maxDepths = [3, 6, 9]
    
    for numEst in numEsts:
        for maxDepth in maxDepths:
            if classifier:
                model_rf = RandomForestClassifier(n_estimators=numEst, max_depth=maxDepth)
            else:
                model_rf = RandomForestRegressor(n_estimators=numEst, max_depth=maxDepth)
            
            model_rf.fit(X_train, y_train.values.ravel())
            if classifier:
                y_hat = model_rf.predict(X_val)
                score = 1 - zero_one_loss(y_val, y_hat)
                
            else:
                score = model_rf.score(X_val, y_val)
            
            if (score > bestScore):
                bestModel = model_rf
                bestScore = score
                
    return bestModel, bestScore

#%% load data

os.chdir('D:\Documents\Code')
varnames = ['var1', 'var2', 'var3']
df = pd.read_excel('input_data.xlsx', header=None, names = varnames)
df.rename(columns = {0:'var1', 1:'var2', 2:'var3'}, inplace=True)

#%% set up data

features = ['mean', 'median', 'var', 'min', 'max', 'cov', 'skew', 'kurt']
past_window = [5, 10, 15, 22, 44, 66, 132]

feature_set = []
for varname in varnames:
    for window in past_window:
        for feat in features:
            feat_name = '_'.join((varname, str(window), feat))
            df[feat_name] = feat_past_days(df, varname, window, feat)
            feature_set.append(feat_name)
        feat_name = '_'.join((varname, str(window), 'vol')); feature_set.append(feat_name)
        df[feat_name] = vol_past_days(df, varname, window)
        feat_name = '_'.join((varname, str(window), 'is_up_past')); feature_set.append(feat_name)
        df[feat_name] = is_up_past_days(df, varname, window)

        
pred_window = [1, 3, 5, 22, 66]
pred_set_direction = []
pred_set_vol = []
for varname in varnames:
    for window in pred_window:
        feat_name = '_'.join((varname, str(window), 'is_up_in')); pred_set_direction.append(feat_name)
        df[feat_name] = is_up_in_days(df, varname, window)
        feat_name = '_'.join((varname, str(window), 'vol_in')); pred_set_vol.append(feat_name)
        df[feat_name] = vol_in_days(df, varname, window)
        
print df.head(5)

#%%

results_direction = pd.DataFrame(columns = ['Output', 'Model', 'Score', 'Result', 'Prob'])
results_direction['Output'] = pred_set_direction

for pred_var in pred_set_direction:
    df_new = df.loc[:, feature_set + [pred_var]]
    df_new.dropna(inplace=True); df_new.reset_index(drop=True, inplace=True)
    X = df_new.loc[:, feature_set]
    y = df_new.loc[:, pred_var]
    
    train_ind = np.random.rand(len(df_new)) < 0.67
    X_train = df_new.loc[train_ind, feature_set]
    y_train = df_new.loc[train_ind, pred_var]
    
    X_val = df_new.loc[~train_ind, feature_set]
    y_val = df_new.loc[~train_ind, pred_var]
    
    model_gp, score_gp = best_gp(X_train, y_train, X_val, y_val)
    model_rf, score_rf = best_rf(X_train, y_train, X_val, y_val)
    model_nn, score_nn = best_nn(X_train, y_train, X_val, y_val)
    
    best_score, best_model_str, best_model = max(zip([score_gp, score_nn, score_rf], 
                                                     ['gp', 'nn', 'rf'], 
                                                     [model_gp, model_nn, model_rf]))
    
    X_test = df.loc[len(df)-1, feature_set].values.reshape(1, -1)
    y_test = best_model.predict(X_test)
    
    if best_model_str != 'gp':
        probability = best_model.predict_proba(X_test)[0][y_test[0]]
    else:
        probability = np.nan
    result_index = results_direction['Output']==pred_var
    results_direction.loc[result_index, 'Model'] = best_model_str
    results_direction.loc[result_index, 'Score'] = best_score
    results_direction.loc[result_index, 'Result'] = y_test
    results_direction.loc[result_index, 'Prob'] = probability
    
print results_direction

#%%

results_vol = pd.DataFrame(columns = ['Output', 'Model', 'Score', 'Result'])
results_vol['Output'] = pred_set_vol

for pred_var in pred_set_vol:
    df_new = df.loc[:, feature_set + [pred_var]]
    df_new.dropna(inplace=True); df_new.reset_index(drop=True, inplace=True)
    X = df_new.loc[:, feature_set]
    y = df_new.loc[:, pred_var]
    
    train_ind = np.random.rand(len(df_new)) < 0.67
    X_train = df_new.loc[train_ind, feature_set]
    y_train = df_new.loc[train_ind, pred_var]
    
    X_val = df_new.loc[~train_ind, feature_set]
    y_val = df_new.loc[~train_ind, pred_var]
    
    model_gp, score_gp = best_gp(X_train, y_train, X_val, y_val, classifier=False)
    model_rf, score_rf = best_rf(X_train, y_train, X_val, y_val, classifier=False)
    try:
        model_nn, score_nn = best_nn(X_train, y_train, X_val, y_val, classifier=False)
    except:
        model_nn = None
        score_nn = 0
    
    best_score, best_model_str, best_model = max(zip([score_gp, score_nn, score_rf], 
                                                     ['gp', 'nn', 'rf'], 
                                                     [model_gp, model_nn, model_rf]))
    
    X_test = df.loc[len(df)-1, feature_set].values.reshape(1, -1)
    y_test = best_model.predict(X_test)

    result_index = results_vol['Output']==pred_var
    results_vol.loc[result_index, 'Model'] = best_model_str
    results_vol.loc[result_index, 'Score'] = best_score
    results_vol.loc[result_index, 'Result'] = y_test
    
print results_vol

#%%

import matplotlib.patches as mpatches
results_direction['ResultColor'] = ['g' if is_up else 'r' for is_up in results_direction['Result']]
fig, ax1 = plt.subplots(figsize=(9, 7))
pos = np.arange(len(results_direction)) + 0.5
barplot = ax1.barh(pos, results_direction['Prob'],
                   align='center',
                   height=0.5, color=results_direction['ResultColor'],
                   tick_label=results_direction['Output']
                   )
ax1.plot(results_direction['Score'], pos, 'ks:', label='Score')
ax1.set_xlabel('Probability & Score')
legend_class = ['Up', 'Down', 'Score']; legend_colors = ['g', 'r', 'k']
recs = [mpatches.Rectangle((0,0),1,1,fc=legend_colors[i]) for i in range(len(legend_class))]
ax1.legend(recs,legend_class,loc='best')

#%%

Blues = plt.get_cmap('Blues')
import matplotlib.patches as mpatches
fig, ax1 = plt.subplots(2,1,figsize=(9, 14))
pos = np.arange(len(results_vol)) + 0.5
norm = plt.Normalize()
colors = plt.cm.jet(norm(results_vol['Score'].values))
colors = [Blues(k) for k in results_vol['Score'].values]
cax = ax1[1].barh(pos, results_vol['Result'],
               align='center',
               height=0.5, color=colors,
               tick_label=results_vol['Output'])
ax1[1].set_xlabel('Volatility')

cax = ax1[0].imshow(results_vol['Score'].values.reshape(1,-1), cmap='Blues')
cbar = fig.colorbar(cax, orientation='vertical')
cbar.ax.set_title('Score')

#%%





