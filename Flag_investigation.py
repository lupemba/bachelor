#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 09:09:38 2018

@author: simon
"""

import swarmtoolkit as st
import numpy as np
import os
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from astropy.time import Time
import copy
os.chdir('/home/simon/Desktop/Bachelor_project/code')

import My_functions_old as my


days = 31

FAC_C = my.load_FAC('C',days)
fig,ax1=st.plot(FAC_C[0].values,abs(FAC_C[6].values),legends=[FAC_C[6].name+'_C'])

#%%

pos_nan = np.where(FAC_C[6].values != FAC_C[6].values)
FAC_C[6].values[pos_nan] = 0
outliers = np.where(abs(FAC_C[6].values) > 100)

FAC_C[6].values[pos_nan] = float('NaN')


#%% plot section
a = 423900
b = 424000

fig,ax1=st.plot(FAC_C[0].values[a:b],abs(FAC_C[6].values[a:b]),legends=[FAC_C[6].name+'_C'])


fig,ax1=st.plot(FAC_C[0].values[a:b],FAC_C[8].values[a:b],legends=[FAC_C[8].name])

fig,ax1=st.plot(FAC_C[0].values[a:b],FAC_C[9].values[a:b],legends=[FAC_C[9].name])

fig,ax1=st.plot(FAC_C[0].values[a:b],FAC_C[10].values[a:b],legends=[FAC_C[10].name])

fig,ax1=st.plot(FAC_C[0].values[a:b],FAC_C[11].values[a:b],legends=[FAC_C[11].name])

fig,ax1=st.plot(FAC_C[0].values[a:b],FAC_C[7].values[a:b],legends=[FAC_C[7].name])


#%%
st.plot_geo(FAC_C[1].values[outliers],FAC_C[2].values[outliers],abs(FAC_C[6].values[outliers]),
            s=20,projection='hammer',lon_0=0,figsize=(12,12))

#%% Fac_clean

FAC_clean = copy.deepcopy(FAC_C)
for i in range(len(FAC_clean)):
    FAC_clean[i].values = np.delete(FAC_clean[i].values, outliers)
fig,ax1=st.plot(FAC_clean[0].values,abs(FAC_clean[6].values),legends=[FAC_clean[6].name+'_C'])

#%%
st.plot_geo(FAC_clean[1].values[200::],FAC_clean[2].values[200::],abs(FAC_clean[6].values[200::]),
            s=10,projection='hammer',lon_0=0,figsize=(12,12))

#%%
f = figure()
plot(FAC_C[7].values[outliers],label=[FAC_C[7].name])
legend()