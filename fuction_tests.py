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

import Load_data as load

#%% Load_ap
ap = load.ap()

fig,ax1=st.plot(ap[0],ap[2],legends=['ap'])

#%% Load_DNS

DNS = load.DNS()

fig,ax1=st.plot(DNS[0].values,DNS[5].values,legends=[DNS[5].name])


#%% Load_FAC_A

FAC_A = load.FAC('A')

fig,ax1=st.plot(FAC_A[0].values,abs(FAC_A[6].values),legends=[FAC_A[6].name+'_A'])


#%% Load_FAC_C
FAC_C = load.FAC('C')

fig,ax1=st.plot(FAC_C[0].values,abs(FAC_C[6].values),legends=[FAC_C[6].name+'_C'])


