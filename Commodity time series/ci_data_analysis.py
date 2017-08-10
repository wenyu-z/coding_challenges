# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 21:39:56 2016

@author: Wenyu
"""

import os
import pandas as pd
import numpy as np
%pylab inline

#%% reorder data

varnames = ['var1', 'var2', 'var3']
df = pd.read_excel('input_data.xlsx', header=None, names = varnames)
df['x'] = np.arange(len(df)-1, -1, -1)
df.sort_values(by=['x'], ascending=True, inplace=True)
df.reset_index(drop=True, inplace=True)

df.plot(x = ['x'], y = varnames, subplots = True, grid = True)

data_max_list = df[varnames].max().values

#%% figure out var2 and var3

print data_max_list

ci_files = os.listdir('./ci_data/')
for ci_file in ci_files:
    df_ci = pd.read_excel(os.path.join('./ci_data/', ci_file), skiprows=6)
    df_ci.dropna(axis=0, inplace=True, how='any')
    ci_max_list = df_ci.max().values[1:]
    max_is_in = any([k in ci_max_list for k in data_max_list])
    if max_is_in:
        print ci_file
        print df_ci.max()[1:]

#var2 is S&P GSCI Heating Oil ER @ PerformanceGraphExport-8.xls
#var3 is S&P GSCI Crude Oil ER @ PerformanceGraphExport-6.xls

#%% figure out var1

df_ci = pd.read_excel(os.path.join('./ci_data/', 'PerformanceGraphExport-8.xls'), skiprows=6)
df_ci.dropna(axis=0, inplace=True, how='any')
print np.where(df['var2']==df_ci['S&P GSCI Heating Oil ER'].values[0])

df_ci = pd.read_excel(os.path.join('./ci_data/', 'PerformanceGraphExport-6.xls'), skiprows=6)
df_ci.dropna(axis=0, inplace=True, how='any')
print np.where(df['var3']==df_ci['S&P GSCI Crude Oil ER'].values[0])

df_cut = df.ix[242:]
data_max_new = df_cut['var1'].max()

ci_files = os.listdir('./ci_data/')
for ci_file in ci_files:
    df_ci = pd.read_excel(os.path.join('./ci_data/', ci_file), skiprows=6)
    df_ci.dropna(axis=0, inplace=True, how='any')
    ci_max_list = df_ci.max().values[1:]
    max_is_in = float("%.2f" % data_max_new) in ci_max_list
    if max_is_in:
        print ci_file
        print df_ci.max()[1:]

#var1 is S&P GSCI Natural Gas ER @ PerformanceGraphExport-10.xls

#%%