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
import pandas as pd
from apexpy import Apex

#%%

def load_DNS():
    # This function loads the DNS zip files from data_loc
    # It loads for x days in march 2015
    # The output is a list with swarmtoolkit.sw_io.Parameter
   
    # path of data
    data_loc = "/home/simon/Desktop/Bachelor_project/data/DNSCWND/"
    
    files=os.listdir(data_loc)
    #dates = np.array(range(1,31+1))
    
    error_count = 0
    
    for i in range(0,len(files)):
        file_name = files[i]
        ##Load data
        try:
            temp_DNS = st.getCDFparams(data_loc+file_name,temp=True)
            
            if i-error_count == 0:
                 DNS = temp_DNS 
            else:
                ## Append data
                for k in range(len(DNS)):
                    DNS[k].values= np.append(DNS[k].values,temp_DNS[k].values)
            #file_date = file_name[25:27]
            #dates[dates==int(file_date)] = 0
        except:
            print(file_name+': Could not be loaded')
            error_count += 1
    
    #fails = dates[dates != 0]
    #if len(fails) != 0:
    #    print('Error loading files for these dates:')
    #    print(fails)
    
    ## Fix unrealistic values / errors in density
    DNS[5].values[DNS[5].values > 1e30] = float('nan')    
    
    # Convert to panda data format
    dates=pd.to_datetime(DNS[0].values)
    data = [DNS[1].values]
    names = ['Altitude','Latitude','Longitude','Local_solar_time','Density']
    for i in range(2,len(DNS)):
        data.append(DNS[i].values)
    DNS = pd.DataFrame(np.transpose(data), index=dates, columns=names)
    
    DNS = DNS.sort_index()
    
    return DNS

#%%
    
def load_FAC(sat='dual'):
    # This function loads the FAC zip files from data_loc
    # It loads for x days in march 2015
    # The output is a list with swarmtoolkit.sw_io.Parameter
    # !!!Note that the 5 march of sat='A' is not in the folder!!!
    
    if sat == 'dual':
        # path of data
        data_loc = "/home/simon/Desktop/Bachelor_project/data/FAC_dual/"
    elif sat == 'A':
        # path of data
        data_loc = "/home/simon/Desktop/Bachelor_project/data/FAC_A/"
    elif sat =='C':
        # path of data
        data_loc = "/home/simon/Desktop/Bachelor_project/data/FAC_C/"
    error_count = 0
    
    files=os.listdir(data_loc)
    #dates = np.array(range(1,31+1))
    
    
    for i in range(0,len(files)):
        file_name = files[i]
        ##Load data
        try:
            temp_FAC = st.getCDFparams(data_loc+file_name,temp=True)
            
            if i-error_count == 0:
                 FAC = temp_FAC 
            else:
                ## Append data
                for k in range(len(FAC)):
                    FAC[k].values= np.append(FAC[k].values,temp_FAC[k].values)
     #       file_date = file_name[25:27]
     #       dates[dates==int(file_date)] = 0
        except:
            print(file_name+': Could not be loaded')
            error_count += 1
    
    #fails = dates[dates != 0]
    #if len(fails) != 0:
    #    print('Error loading files for these dates:')
    #    print(fails)
    
    # convert to pandas
    dates=pd.to_datetime(FAC[0].values)
    data = [FAC[1].values]
    names = [FAC[1].name]
    for i in range(2,len(FAC)):
        data.append(FAC[i].values)
        names.append(FAC[i].name)
    FAC = pd.DataFrame(np.transpose(data), index=dates, columns=names)
    FAC = FAC.sort_index()
    return FAC

#%%
    
def add_orbit(dataframe):
    """
    add column to dataframe with the orbit nr of corrospondig to the latitude.
    The firt mesument is denoted 0 the next 1 and so on.
    Be aware that the latitude should be ordered in time. 
    !!! Orbit nr is sensitive to gaps in time series. Orbit nr does not 
    align when there is gaps !!!
    
    """
    if ('Orbit_nr' in dataframe.columns):
        print('Orbit_nr is all ready in the dataframe')
        return None
    
    # initial orbits
    latitude = dataframe['Latitude']
    orbits = np.zeros(len(latitude))
    hemisphere = np.ones(len(latitude))
    current_orbit = 0
    
    # Check if first mesument is on the southen hemisphere
    if latitude[0]<0:
        hemisphere[0]=-1
        
    # go through all latitudes
    for i in range(1,len(latitude)):
        
        if latitude[i]>0:
            hemisphere[i]=1 # set to norhten hemiphere 
            # If the Acending node is crossed. New orbit
            if latitude[i-1]<0:
                current_orbit += 1
        else:
            hemisphere[i]=-1 # set to southen hemiphere 
        # sets orbit
        orbits[i] = current_orbit
        
    dataframe.loc[:,'Orbit_nr'] =  orbits
    dataframe.loc[:,'Hemisphere'] = hemisphere
    return None
            
#%%         

def orbit_means(dataframe,mode='abs'):
    
    if ('FAC' in dataframe.columns):
        pos = 'FAC'
    elif ('Density' in dataframe.columns):
        pos = 'Density'
    else:
        print('Error in data type')
        return 0
    
    if not 'Orbit_nr' in dataframe.columns:
        add_orbit(dataframe)


    #get orbit nr.
    orbit_nr = np.repeat(np.array(range(int(dataframe.Orbit_nr[-1]+1))),2)
    hemisphere = -1*np.ones(len(orbit_nr))
    hemisphere[::2]=-hemisphere[::2]
    #Intialize arrays for result
    values = np.zeros(len(orbit_nr))
    mesuments = np.zeros(len(orbit_nr))
    delta_time = [0 for x in range(len(orbit_nr))]
    dates = [0 for x in range(len(orbit_nr))]
    # sets first orbit
    
    # Go through data
    for i in range(len(orbit_nr)):
        # get the orbit
        df = dataframe[dataframe.Orbit_nr == orbit_nr[i]]
        # Get hemisphere
        df = df[df.Hemisphere == hemisphere[i]]
          
        mesuments[i] = len(df.loc[:,pos])
        if mesuments[i] == 0:
            values[i] = float('nan')
            delta_time[i] = 0
            dates[i] = dataframe.index[0]
        else:
            delta_time[i] = (df.index[-1]-df.index[0]).total_seconds()
            dates[i] = df.index[0] + (df.index[-1]-df.index[0])/2
            if mode == 'simple':
               values[i] = df.loc[:,pos].mean()
            if mode == 'abs':
                values[i] = df.loc[:,pos].apply(abs).mean()
    
            if mode == 'power':
                values[i] = np.sqrt(df.loc[:,pos].apply(lambda x: x**2).mean())
    data = [values,orbit_nr,hemisphere,mesuments,delta_time]
    names = [pos,'Orbit_nr','Hemisphere','Count','Delta_time']
    means = pd.DataFrame(np.transpose(data), index=dates, columns=names)
    return means

#%%         
def get_jets(FAC,window='120s'):
    """
    This Function takes FAC with orbit nr. and output the location of the 
    electro-jets without local Solar time.
    """
    if ('FAC' not in FAC.columns):
        print('Error in data type')
        return 0
   
    # Smooth the data
    smooth = FAC.loc[:,['FAC','Latitude','Longitude','Orbit_nr','Hemisphere']].copy()
    smooth.FAC = smooth.FAC.abs()
    smooth.FAC = smooth.FAC.rolling(window).mean()
    
    add_apex_coords(smooth)
    
    add_heading(smooth,latitude = 'mLatitude')
    
    # Get index of max FAC for every quater orbit.
    # quater orbit becuse the sattelite passes the jet on both side of
    # the pole
    idx= smooth.groupby(['Orbit_nr','Hemisphere','N_heading'])['FAC'].transform(max) == smooth['FAC']
    
    return smooth[idx]

#%%
    
def add_heading(dataframe, latitude = 'Latitude' ):
    """
    
    """
    if ('N_heading' in dataframe.columns):
        print('N_heading is all ready in the dataframe')
        return None
    
    # Crate an column to indicate if the sattelite is headed N or S
    N_heading = dataframe.loc[:,'Latitude'].values.copy()
    N_heading = np.diff(N_heading)
    N_heading = np.append(N_heading, N_heading[-1]) # make sure dimensions fit
    N_heading[N_heading>0] = 1  # If diff(lat)>0 the sat is noth_going
    N_heading[N_heading<0] = -1 # If !(diff(lat)>0) the sat is going south
    dataframe.loc[:,'N_heading'] =  N_heading
    
    return None

#%%
    
def add_apex_coords(dataframe,date = 'none', h = 450):
    """
    Add the geomagtic coordinates using apexPy 
    """
    if ('mLatitude' in dataframe.columns):
        print('mLatitude is all ready in the dataframe')
        return None    
    
    if date == 'none':
        date=dataframe.index[int(len(dataframe)/2)].date()
    
    model = Apex(date)
    
    ## Get the apex coordinates
    mlat, mlon = model.geo2apex(dataframe.Latitude,dataframe.Longitude,h)
    
    # adds to the dataframe
    dataframe.loc[:,'mLatitude'] =  mlat
    dataframe.loc[:,'mLongitude'] =  mlon
    
    
    return None         


def filter_FAC(FAC, dt = '10',Flags= None, Flags_F=None, Flags_B=None,Flags_q=None):
    """
    Remove oberservations with flags above the enetered values 
    and the obserevation with in +-dt seconds of the flaged observation.
    If the value is None the flag is not considered
    """
    if ('FAC' not in FAC.columns):
        print('Error in data type')
        return 0
    
    # Set the accepted flag values to max if not specified
    if Flags == None:
        Flags = max(FAC.Flags)
    
    if Flags_F == None:
        Flags_F = max(FAC.Flags_F)
        
    if Flags_B == None:
        Flags_B = max(FAC.Flags_B)
        
    if Flags_q== None:
        Flags_q = max(FAC.Flags_q)
        
    # Get the index of the flags    
    Flag_idx= (FAC.Flags>Flags)|(FAC.Flags_F>Flags_F)|(FAC.Flags_B>Flags_B)|(FAC.Flags_q>Flags_q)
    
    # Initialize the FAC_filter with FAC values
    FAC_filter = FAC.copy(deep=True)
    
    # Get the time intervals that should be removed
    flagtime = FAC[Flag_idx].index.values
    start_time= flagtime - np.timedelta64(dt, 's')
    end_time = flagtime + np.timedelta64(dt, 's')

    # Loop though the flags and remove set the corresponig interval to Nan
    for i in range(0,len(flagtime)):
        FAC_filter.FAC.loc[start_time[i]:end_time[i]] = float('Nan')
        
    # Drop the NaN values.
    FAC_filter=FAC_filter.dropna(how='any')

    print("%d observation out off %d are removed" % (len(FAC)-len(FAC_filter),len(FAC)))
    
    return FAC_filter







