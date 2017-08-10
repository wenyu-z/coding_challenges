# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 09:58:41 2016

@author: WZhao10
"""

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

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
    
#%%

X = np.arange(0, 50).reshape(-1,1)
y = 3*np.sin(100*X).reshape(-1,1)
plt.plot(X, y,'.:')


#%%

plt.plot(X, y,'.:')

k1 = 50**2 * RBF(length_scale=200)  # long term smooth rising trend
k2 = 3**2 * RBF(length_scale=200) \
    * ExpSineSquared(length_scale=1, periodicity=10,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=1, alpha=1)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf))  # noise terms
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              optimizer=None, normalize_y=True)

gp.fit(X, y)

#X_ = np.linspace(X.min(), X.max() + 25, len(X)+25)[:, np.newaxis]
#y_pred, y_std = gp.predict(X_, return_std=True)
#plt.plot(X_, y_pred)

X_ = np.linspace(X.max()+1, X.max() + 25, 25)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)
plt.plot(X_, y_pred, 'r*:')


#%%

X = np.arange(0, 50)
y = 3*np.sin(100*X)
plt.plot(X, y,'.:')

from sklearn.neural_network import MLPRegressor

model_nn = MLPRegressor()
model_nn.fit(y[:45], y[45:])

#%%

X = np.arange(0, 50).reshape(-1,1)
y = 3*np.sin(100*X).reshape(-1,1)
plt.plot(X, y,'.:')

from statsmodels.tsa.arima_model import ARMA

model_arma = ARMA(y, (1, 0)).fit()

#resid = model_arma.fittedvalues[1:]
#plt.plot(resid)

test = model_arma.predict(0, 61)
plt.plot(test[1:], '.:')
plt.plot(np.arange(50,61), test[51:], '*:')

#%%

X = np.arange(0, 50).reshape(-1,1)
y = 3*np.sin(100*X).reshape(-1,1)
plt.plot(X, y,'.:')

from statsmodels.tsa.arima_model import ARMA

model_arma = ARMA(y, (1, 1)).fit()

#resid = model_arma.fittedvalues[1:]
#plt.plot(resid)

test = model_arma.predict(0, 61)
plt.plot(test[1:], '.:')
plt.plot(np.arange(50,61), test[51:], '*:')


#%%

from statsmodels.tsa.arima_model import ARMA
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

os.chdir('F:\Code')
varnames = ['var1', 'var2', 'var3']
df = pd.read_excel('input_data.xlsx', header=None, names = varnames)

y = df['var1'].values[:60]
model_arma = ARMA(y, (10, 3)).fit()
y_new = df['var1'].values[:70]
plt.plot(y_new, 'b.:')

resid = model_arma.fittedvalues[1:]
plt.plot(resid, 'g.:')

test = model_arma.predict(60, 70)
plt.plot(np.arange(60, 70), test[1:], 'r*:')

#%%
import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

os.chdir('F:\Code')
varnames = ['var1', 'var2', 'var3']
df = pd.read_excel('input_data.xlsx', header=None, names = varnames)
    
k1 = 50**2 * RBF(length_scale=300)  # long term smooth rising trend
k2 = 3**2 * RBF(length_scale=300) \
    * ExpSineSquared(length_scale=1, periodicity=10,
                     periodicity_bounds="fixed")  # seasonal component
# medium term irregularities
k3 = 0.5**2 * RationalQuadratic(length_scale=10, alpha=1)
k4 = 0.1**2 * RBF(length_scale=0.1) \
    + WhiteKernel(noise_level=0.1**2,
                  noise_level_bounds=(1e-3, np.inf))  # noise terms
kernel = k1 + k2 + k3 + k4

gp = GaussianProcessRegressor(kernel=kernel, alpha=0,
                              optimizer=None, normalize_y=True)


X_train = np.arange(0, 60).reshape(-1,1)
y_train = df['var1'].values[:60]
gp.fit(X_train, y_train)

y_new = df['var1'].values[:70]

X_test = np.linspace(X_train.max()+1, X_train.max() + 10, 10)[:, np.newaxis]

y_val = gp.predict(X_train)

y_hat = gp.predict(X_test)

plt.figure(figsize=(8,4.5))
plt.plot(y_new, 'b.:')
plt.plot(X_train, y_val, 'g.:')
plt.plot(X_test, y_hat, 'r*:')
plt.legend(['Actual Values', 'Fitted Values', 'Predicted Values'], loc='best')

plt.figure(figsize=(4,4))
plt.plot(y_new[60:], y_hat, 'kd')
xlim = plt.xlim()
ylim = plt.ylim()
lim1 = min((xlim[0], ylim[0]))
lim2 = max((xlim[1], ylim[1]))
plt.xlim((lim1, lim2))
plt.ylim((lim1, lim2))
plt.plot([lim1, lim2], [lim1, lim2], 'r--', linewidth = 3)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
yhere = calc_score(y_new[60:], y_hat)
plt.title('R Squared %.3f' %yhere)
