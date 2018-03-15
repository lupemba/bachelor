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

def load_ap(days = 31):
    # This function loads the ap from a .dat file in data_loc folder
    # It loads for x days in march 2015
    # The output is a list with ap[0]=time, ap[1]=Kp, ap[2]=ap
    
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

def load_DNS(days=31):
    # This function loads the DNS zip files from data_loc
    # It loads for x days in march 2015
    # The output is a list with swarmtoolkit.sw_io.Parameter
   
    
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
    
def load_FAC(sat='A',days=31):
    # This function loads the FAC zip files from data_loc
    # It loads for x days in march 2015
    # The output is a list with swarmtoolkit.sw_io.Parameter
    # !!!Note that the 5 march of sat='A' is not in the folder!!!
    
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

#%%
    
def orbit_nr(latitude):
    """
    Output an np.array with the orbit nr of corrospondig to the latitude.
    The firt mesument is denoted 0 the next 1 and so on.
    Be aware that the latitude should be ordered in time. 
    !!! Orbit nr is sensitive to holds in time series. Orbit nr does not 
    align when there is holds!!!
    
    """
    # initial orbits 
    orbits = np.zeros(len(latitude.values))
    current_orbit = 0
    
    # go through all latitudes
    for i in range(1,len(latitude.values)):
        
        # If the Acending node is crossed. New orbit
        if latitude.values[i-1]<0 and latitude.values[i]>0:
            current_orbit += 1
        
        # sets orbit
        orbits[i] = current_orbit
        
    return orbits
            
#%%         

def orbit_means(time,latitude,data,mode='simple'):
    
    #Checks input
    if data.name =='FAC':
        if time.name !='Timestamp' or latitude.name !='Latitude':
            print('Error in input')
            return 0
    elif data.name =='density':
        if time.name !='time' or latitude.name != 'latitude':
            print('Error in input')
            return 0
    else:
        print('Error in data type')
        return 0
    
    if mode == 'simple':
        values = data.values
        
    if mode == 'abs':
        values = abs(data.values)
    
    if mode == 'power':
        values = np.power(data.values , 2)
    
    #get orbit nr.
    orbits = orbit_nr(latitude)
     
    #Intialize arrays for result
    means = np.zeros(int(orbits[-1]+1))
    mesuments = np.zeros(int(orbits[-1]))
    orbit_time = [0 for x in range(int(orbits[-1]))]
    
    # sets first orbit
    current_orbit = 0
    start_time = time.values[0]
    n = 0 
    
    # Go through data
    for i in range(len(orbits)):
        
        ## If new orbit 
        if current_orbit != orbits[i]:
            
            # Error code
            #print('current_ orbit:%d  orbits[i]:%d    i:%d'
            #          %(current_orbit,orbits[i],i))
            
            # Compute the mean of the old orbit
            means[current_orbit] = means[current_orbit]/n
            # record nr of mesuments
            mesuments[current_orbit] = n
            # compute the time difference
            delta_time = start_time-time.values[i]
            
            #print('orbit nr %d has dt= %d'
            #          %(current_orbit,delta_time.seconds))
            # print warning for strange periods 
            
            # Print varning if time difference is strange
            if abs(delta_time.seconds-80774) > 600 and current_orbit !=0:
                print('Warning: orbit nr %d has dt= %d'
                      %(current_orbit,delta_time.seconds))
            
            # set orbit  time in the midle
            orbit_time[current_orbit] = start_time + delta_time/2
            
            # Set to new orbit
            current_orbit += 1
            
            # re initialize
            start_time = time.values[i]
            n = 0
            
        
        # Check for nan values
        if values[i] == values[i]:
            # Add each mesument to the corresponding orbit        
            means[current_orbit] += values[i]
            
            n += 1
       

    if mode == 'power':
        means = np.sqrt(means)
        
    #Return all but the last and first orbit
    return orbit_time[1:], means[1:-1], mesuments[1:]
    