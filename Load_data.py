#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:46:35 2018

@author: simon
"""

## Basic functionality
import swarmtoolkit as st
import numpy as np
import os
from astropy.time import Time
# path of data
data_loc = "/home/simon/Desktop/Bachelor_project/data/"

#%%

def ap(days = 31):
    path_ap = os.path.join(data_loc,'Kp_ap_1998.dat')
    file = open(path_ap,"r") 
    lines = file.readlines() 
    file.close()
    
    
    start = 50145  ## 1. marts 2015 kl 1:30
    slut = start+8*days
    ## time, Kp, ap
    ap = [[0 for x in range(slut-start)],np.zeros(slut-start),np.zeros(slut-start)]
    for i in range(start,slut):
        t = Time(float(lines[i].strip().split()[0])+51544.5, format='mjd', scale='utc')
        t.format = 'datetime'
        ap[0][i-start] = t.value
        ap[1][i-start] = float(lines[i].strip().split()[1])
        ap[2][i-start] = float(lines[i].strip().split()[2])
    
    ## ap[0]=time, ap[1]=Kp, ap[2]=ap
    return ap


#%%

def DNS(days=31):
    name = 'SW_OPER_DNSCWND_2__20150301T000000_20150301T235950_0101.ZIP'
    path = os.path.join(data_loc,name)
    DNS = st.getCDFparams(path,temp=True)
    
    for i in range(2,days+1):
        ## Gennerate a path name
        if i < 10:
            nr = '0'+str(i)
        else:
            nr = str(i)
        name = name[0:25]+nr + name[27:41]+nr +name[43:]
        path = os.path.join(data_loc,name)
        
        ##Load data
        try:
            temp_DNS = st.getCDFparams(path,temp=True)
        
            ## Append data
            for k in range(len(DNS)):
                DNS[k].values= np.append(DNS[k].values,temp_DNS[k].values)
        except:
            print(name+' Was not found')
    ## Fix unrealistic values / errors in density
    DNS[5].values[DNS[5].values > 1e30] = float('nan')    
    
    return DNS

#%%
    
def FAC(sat='A',days=31):
    if sat == 'A':
        name = 'SW_OPER_FACATMS_2F_20150301T000000_20150301T235959_0205.ZIP'
    elif sat =='C':
        name = 'SW_OPER_FACCTMS_2F_20150301T000000_20150301T235959_0205.ZIP' 
    path = os.path.join(data_loc,name)
    FAC = st.getCDFparams(path,temp=True)
    
    for i in range(2,days+1):
        ## Gennerate a path name
        if i < 10:
            nr = '0'+str(i)
        else:
            nr = str(i)
        name = name[0:25]+nr + name[27:41]+nr +name[43:]
        path = os.path.join(data_loc,name)
        
        ##Load data
        try:
            temp_FAC = st.getCDFparams(path,temp=True)
            
            ## Append data
            for k in range(len(FAC)):
                FAC[k].values= np.append(FAC[k].values,temp_FAC[k].values)
        except:
            print(name+' Was not found')
        
   
    return FAC
