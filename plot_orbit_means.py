#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:14:27 2018

@author: simon
"""

import swarmtoolkit as st
import numpy as np
import os
from astropy.time import Time

os.chdir('/home/simon/Desktop/Bachelor_project/code')

import My_functions_old as my

#%% load data and compute means

ap = my.load_ap()

# Density
DNS = my.load_DNS()
DNS_mean = my.orbit_means(DNS[0],DNS[2],DNS[5],'simple')
del DNS


#FAC_C
FAC_C = my.load_FAC('C')



FAC_C_abs = my.orbit_means(FAC_C[0],FAC_C[1],FAC_C[6],'abs')
FAC_C_power = my.orbit_means(FAC_C[0],FAC_C[1],FAC_C[6],'power')
FAC_C_simple = my.orbit_means(FAC_C[0],FAC_C[1],FAC_C[6],'simple')

del FAC_C
#%%


#%% Plot 
fig,ax1=st.plot(ap[0],ap[2],
                legends=['ap'],lhide=True)
ax2=st.plot_twinx(FAC_C_abs[0],FAC_C_abs[1],ax=ax1,
                  legends=['FAC_abs'])

#%%
fig,ax1=st.plot(DNS_mean[0],DNS_mean[1],
                legends=['DNS simple mean'],lhide=True)
ax2=st.plot_twinx(FAC_C_abs[0],FAC_C_abs[1],ax=ax1,
                  legends=['FAC_abs'])

#%%

fig,ax1=st.plot(DNS_mean[0],DNS_mean[1],
                legends=['DNS simple mean'],lhide=True)
ax2=st.plot_twinx(FAC_C_power[0],FAC_C_power[1],ax=ax1,
                  legends=['FAC_power'])

#%%

fig,ax1=st.plot(DNS_mean[0],DNS_mean[1],
                legends=['DNS simple mean'],lhide=True)
ax2=st.plot_twinx(FAC_C_simple[0],FAC_C_simple[1],ax=ax1,
                  legends=['FAC_simple'])

