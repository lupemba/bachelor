#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:56:41 2018

@author: simon
"""

import swarmtoolkit as st
import numpy as np
import os
from astropy.time import Time

os.chdir('/home/simon/Desktop/Bachelor_project/code')

import My_functions_old as my
days = 10

#%% Load_ap
ap = my.load_ap(days)

fig,ax1=st.plot(ap[0],ap[2],legends=['ap'])

#%% Load_DNS

DNS = my.load_DNS(days)

fig,ax1=st.plot(DNS[0].values,DNS[5].values,legends=[DNS[5].name])


#%% orbit_nr test

orbits =my.orbit_nr(DNS[2])

fig,ax1=st.plot(DNS[0].values,orbits,legends=['DNS orbit nr'])

#%% Load_FAC_A

FAC_A = my.load_FAC('A',days)

fig,ax1=st.plot(FAC_A[0].values,abs(FAC_A[6].values),legends=[FAC_A[6].name+'_A'])

#%% mean_test

mean_test = my.orbit_means(FAC_A[0],FAC_A[1],FAC_A[6],'simple')


fig,ax1=st.plot(mean_test[0],mean_test[1],legends=[FAC_A[6].name+'_A simple'])

#%%
fig,ax1=st.plot(mean_test[0],mean_test[2],legends=['Nr mesuments'])

mean_test = my.orbit_means(FAC_A[0],FAC_A[1],FAC_A[6],'abs')


fig,ax1=st.plot(mean_test[0],mean_test[1],legends=[FAC_A[6].name+'_A abs'])
#%%
mean_test = my.orbit_means(FAC_A[0],FAC_A[1],FAC_A[6],'power')

fig,ax1=st.plot(mean_test[0],mean_test[1],legends=[FAC_A[6].name+'_A power'])



#%% Load_FAC_C
FAC_C = my.load_FAC('C',days)

fig,ax1=st.plot(FAC_C[0].values,abs(FAC_C[6].values),legends=[FAC_C[6].name+'_C'])


#%% mean_test

mean_test = my.orbit_means(FAC_C[0],FAC_C[1],FAC_C[6],'simple')


fig,ax1=st.plot(mean_test[0],mean_test[1],legends=[FAC_A[6].name+'_C simple'])

fig,ax1=st.plot(mean_test[0],mean_test[2],legends=['Nr mesuments'])

#%% mean_test
mean_test = my.orbit_means(DNS[0],DNS[2],DNS[5],'simple')


fig,ax1=st.plot(mean_test[0],mean_test[1],legends=[DNS[5].name+'_C simple'])

fig,ax1=st.plot(mean_test[0],mean_test[2],legends=['Nr mesuments'])

