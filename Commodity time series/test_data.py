# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 11:01:24 2016

@author: WZhao10
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

varnames = ['var1', 'var2', 'var3']
os.chdir('D:\Documents\Downloads')
df = pd.read_excel('input_data.xlsx', header=None, names = varnames)
df.rename(columns={0:'var1', 1:'var2', 2:'var3'}, inplace=True)

#%%

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

# Kernel with parameters given in GPML book
k1 = 66.0**2 * RBF(length_scale=67.0)  # long term smooth rising trend
k2 = 2.4**2 * RBF(length_scale=90.0) \
    * ExpSineSquared(length_scale=1.3, periodicity=1.0)  # seasonal component
# medium term irregularity
k3 = 0.66**2 \
    * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18**2 * RBF(length_scale=0.134) \
    + WhiteKernel(noise_level=0.19**2)  # noise terms
kernel_gpml = k1 + k2 + k3 + k4
kernel_gpml = k1 + k3

gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                              optimizer=None, normalize_y=True)

X = np.arange(0, len(df)).reshape(len(df),1)
y = df['var2'].values

gp.fit(X, y)

X_ = np.linspace(X.min(), X.max() + 30, len(X)+30)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

plt.scatter(X, y, c='k')
plt.plot(X_, y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std,
                 alpha=0.5, color='k')
plt.xlim(X_.min(), X_.max())
plt.xlabel("Year")
plt.ylabel(r"CO$_2$ in ppm")
plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
plt.tight_layout()
plt.show()

#%%

#Direction plot

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

os.chdir('D:\Documents\Code')
results_direction = pd.read_excel('test_results.xlsx', sheetname = 'Directionality')


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

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

os.chdir('D:\Documents\Code')
results_vol = pd.read_excel('test_results.xlsx', sheetname = 'Volatility')

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
Blues = plt.get_cmap('Blues')

fig, ax1 = plt.subplots(figsize=(8, 4.5))
pos = np.arange(len(results_vol)) + 0.5
norm = plt.Normalize()
colors = plt.cm.jet(norm(results_vol['Score'].values))
colors = [Blues(k) for k in results_vol['Score'].values]
cax = ax1.barh(pos, results_vol['Result'],
               align='center',
               height=0.5, color=colors,
               tick_label=results_vol['Output'])
ax1.set_xlabel('Volatility')

plt.figure()
cax = plt.imshow(results_vol['Score'].values.reshape(1,-1), cmap='Blues')
cbar = fig.colorbar(cax, orientation='vertical')
cbar.ax.set_title('Score')
