#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 20:20:10 2018

@author: simon
"""


## Basic functionality
import swarmtoolkit as st
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.time import Time
# path of data
data_loc = "/home/simon/Desktop/Bachelor_project/data/"


#%% Load ap
path_ap = os.path.join(data_loc,'Kp_ap_1998.dat')

file = open(path_ap,"r") 
lines = file.readlines() # This is what you forgot 
file.close()


start = 50145  ## 1. marts 2015 kl 1:30
ap_days = 1
slut = start+8*ap_days
## time, Kp, ap
ap = [[0 for x in range(slut-start)],np.zeros(slut-start),np.zeros(slut-start)]
for i in range(start,slut):
    t = Time(float(lines[i].strip().split()[0])+51544.5, format='mjd', scale='utc')
    t.format = 'datetime'
    ap[0][i-start] = t.value
    ap[1][i-start] = float(lines[i].strip().split()[1])
    ap[2][i-start] = float(lines[i].strip().split()[2])


#%% Load fiels 

path_DNS = os.path.join(data_loc,'SW_OPER_DNSCWND_2__20150301T000000_20150301T235950_0101.ZIP')
path_FAC = os.path.join(data_loc,'SW_OPER_FACATMS_2F_20150301T000000_20150301T235959_0205.ZIP')


# Load DNS
DNS_time,DNS_alt,DNS_lat,DNS_long,DNS_solt,DNS_dens =st.getCDFparams(path_DNS,temp=True)
DNS_dens.values[DNS_dens.values > 1e30] = float('nan')
print('Parameter: {}, units: {}\nValues:\n{}'.format(DNS_dens.name,DNS_dens.unit,DNS_dens.values),end='\n\n')

# Load FAC
FAC_time,FAC_lat,FAC_long,FAC_R,FAC_IRC,FAC_IRC_Error,FAC_FAC, FAC_FAC_Error,FAC_flag,FAC_flagF,FAC_flagB,FAC_flagq =st.getCDFparams(path_FAC,temp=True)
print('Parameter: {}, units: {}\nValues:\n{}'.format(FAC_FAC.name,FAC_FAC.unit,FAC_FAC.values),end='\n\n')


#%% Plot DNS
fig,ax1=st.plot(DNS_time.values,DNS_dens.values,
                legends=[DNS_dens.name])

st.plot_geo(DNS_lat.values[::10],DNS_long.values[::10],DNS_dens.values[::10]*10**10*7,
            s=20,projection='hammer',lon_0=90,figsize=(12,12))

#%% Plot FAC
fig,ax1=st.plot(FAC_time.values,FAC_FAC.values,
                legends=[FAC_FAC.name])

st.plot_geo(FAC_lat.values[::100],FAC_long.values[::100],FAC_FAC.values[::100],
            s=20,projection='hammer',lon_0=90,figsize=(12,12))


#%% Plot FAC DNS
fig,ax1=st.plot(DNS_time.values,DNS_dens.values,
                legends=[DNS_dens.name],lhide=True)
ax2=st.plot_twinx(FAC_time.values,FAC_FAC.values,ax=ax1,
                  legends=[FAC_FAC.name])

#%% Plot FAC ap

fig,ax1=st.plot(ap[0],ap[2],
                legends=['ap'],lhide=True)
ax2=st.plot_twinx(FAC_time.values,FAC_FAC.values,ax=ax1,
                  legends=[FAC_FAC.name])



