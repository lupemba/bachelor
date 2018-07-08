#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Date: 23th Feb 2018

@author: Simon Kok Lupemba

contact: simon.kok@live.dk
"""

import swarmtoolkit as st
import numpy as np
import os
import pandas as pd
from apexpy import Apex
from pyamps import AMPS
from math import radians, sin, cos, asin

#%%
def load_DNS():
    """
    This function tried to load all DNS zip files from data_loc  the output is
    a pandas data frame.
    """
    # path of data
    data_loc = "/home/simon/Desktop/Bachelor_project/data/DNSCWND/"

    files=os.listdir(data_loc)

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
        except:
            print(file_name+': Could not be loaded')
            error_count += 1


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
    """
    This function tried to load all FAC zip files from data_loc  the output is
    a pandas data frame.
    """

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
        except:
            print(file_name+': Could not be loaded')
            error_count += 1

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
    add column to dataframe with the orbit no. and a column with hemisphere.
    The firt mesument is denoted 0 the next 1 and so on.
    Hemisphere=1 is the northen and Hemisphere=-1 is the southen
    Be aware that the latitude should be ordered in time.
    !!! Orbit nr is sensitive to gaps in time series. Orbit nr does not
    align when there is gaps !!!
    """

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
    """
    Computes a mean value of FAC or Density for each hemisphere in each orbit.
    Returns a dataframe
    """



    if ('FAC' in dataframe.columns):
        pos = 'FAC'
        alt = 'Radius'
    elif ('Density' in dataframe.columns):
        pos = 'Density'
        alt = 'Altitude'
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
    altitudes = np.zeros(len(orbit_nr))
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
            altitudes[i] = float('nan')
            delta_time[i] = 0
            dates[i] = dataframe.index[0]
        else:
            delta_time[i] = (df.index[-1]-df.index[0]).total_seconds()
            dates[i] = df.index[0] + (df.index[-1]-df.index[0])/2
            altitudes[i] = df.loc[:,alt].mean()
            if mode == 'simple':
               values[i] = df.loc[:,pos].mean()
            if mode == 'abs':
                values[i] = df.loc[:,pos].apply(abs).mean()

            if mode == 'power':
                values[i] = df.loc[:,pos].apply(lambda x: x**2).mean()
    data = [values,altitudes,orbit_nr,hemisphere,mesuments,delta_time]
    names = [pos,alt,'Orbit_nr','Hemisphere','Count','Delta_time']
    means = pd.DataFrame(np.transpose(data), index=dates, columns=names)
    return means

#%%
def get_jets(FAC,window='120s'):
    """
    This Function takes FAC with orbit nr. and output the location of the
    electro-jets without local Solar time. The fuction should be improved
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
    idx= smooth.groupby(['Orbit_nr','Hemisphere','mN_heading'])['FAC'].transform(max) == smooth['FAC']

    return smooth[idx]

#%%

def add_heading(dataframe, latitude = 'Latitude' ):
    """
    Adds a number to detect is the sattelite is heading north or south.

    latitude=
    'Latitude': uses geografic latitude
    'mLatitude': uses magnetic latitude
    """
    if (latitude == 'mLatitude') & (~('mLatitude' in dataframe.columns)):
        add_apex_coords(dataframe)

    # Crate an column to indicate if the sattelite is headed N or S
    N_heading = dataframe.loc[:,latitude].values.copy()
    N_heading = np.diff(N_heading)
    N_heading = np.append(N_heading, N_heading[-1]) # make sure dimensions fit
    N_heading[N_heading>0] = 1  # If diff(lat)>0 the sat is noth_going
    N_heading[N_heading<0] = -1 # If !(diff(lat)>0) the sat is going south

    if latitude == 'mLatitude':
        dataframe.loc[:,'mN_heading'] =  N_heading
    else:
        dataframe.loc[:,'N_heading'] =  N_heading

    return None

#%%

def add_apex_coords(dataframe,date = 'none', h = 470):
    """
    Add quasidipol geomagtic coordinates to the dataframe using apexPy
    """
    if ('mLatitude' in dataframe.columns):
        print('mLatitude is all ready in the dataframe')
        return None

    if date == 'none':
        date=dataframe.index[int(len(dataframe)/2)].date()

    model = Apex(date)

    ## Get the apex coordinates
    mlat, mlon = model.geo2qd(dataframe.Latitude,dataframe.Longitude,h)

    # adds to the dataframe
    dataframe.loc[:,'mLatitude'] =  mlat
    dataframe.loc[:,'mLongitude'] =  mlon


    return None

#%%

def filter_FAC(FAC, dt = 10,Flags= None, Flags_F=None, Flags_B=None,Flags_q=None):
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

#%%

def Color_map(df, start_time, N, latitude = 'Latitude',min_lat = 0,
              roll = None,whitespace=10,lat_tolerance=1):
    """
    Returns values to make a color map plot. Interpolates values to grid
    of latitudes for each orbit. Uses nearest value with a
    tolorance of max lat_tolerance degree
    """

    # Check data_type
    if ('FAC' in df.columns):
        pos = 'FAC'
    elif ('Density' in df.columns):
        pos = 'Density'
    else:
        print('Error in data type')
        return 0

    # Do rolling mean if needed.
    if roll != None:
        df.loc[:,pos+'_roll']  = abs(df.loc[:,pos]).rolling(roll).mean().values
        pos = pos+'_roll'

    ## Add the heading and orbit
    if latitude == 'Latitude':
        heading = 'N_heading'
        orbit = 'pOrbit_nr'

        if ~(heading in df.columns):
            add_heading(df, latitude)

        if ~(orbit in df.columns):
            # add pOrbit which is a orbit nr. that changes increment with 0.5 for each pole passages
            df.loc[:,orbit] = np.cumsum(np.append(0, abs(np.diff(df.N_heading.values))))/4

    elif latitude == 'mLatitude':
        heading = 'mN_heading'
        orbit = 'mOrbit_nr'

        if ~(heading in df.columns):
            add_heading(df, latitude)

        if ~(orbit in df.columns):
            # add mOrbit which is a orbit nr. that changes increment with 0.5 for each mag-pole passages
            df.loc[:,orbit] = np.cumsum(np.append(0, abs(np.diff(df.mN_heading.values))))/4
    else:
        print('Error in input latitude, Try Latitude or mLatitude')
        return 0

    # get the start orbit
    start_orbit = df[start_time].loc[:,orbit][0]
    # Create the Y index'ex
    if min_lat == 0:
        fig_index1 = np.arange(-90,90,0.5)
        fig_index2 = np.arange(90,-90.5,-0.5)
        fig_index = np.hstack([fig_index1,fig_index2])
    else:
        fig_index1 = np.hstack([np.arange(-90,-min_lat,0.01),np.linspace(-1,1,whitespace), np.arange(min_lat,90,0.01)])
        fig_index2 = np.hstack([np.arange(90,min_lat,-0.01),np.linspace(1,-1,whitespace), np.arange(-min_lat,-90,-0.01)])
        fig_index = np.hstack([fig_index1,fig_index2])

    # Initialize for x indexes
    fig_dates = np.empty(N, dtype='datetime64[s]')
    fig_orbit = np.zeros(N)

    # Initilaize matrix for the densiteis
    fig_values = np.zeros([len(fig_index),N])

    for i in range(N):
    # Get the densiteis for (start_orbit+i) orbit and make sure that data_column1 is the N heading part
    # and data_column2 is the south heading part.
        data_column1= df[abs(df.loc[:,orbit].values-(start_orbit+i))<0.01]
        if data_column1.loc[:,heading].mean() > 0:
            data_column2 = df[abs(df.loc[:,orbit].values-(start_orbit+i+0.5))<0.01]
        else:
            data_column1= df[abs(df.loc[:,orbit].values-(start_orbit+i+0.5))<0.01]
            data_column2 = df[abs(df.loc[:,orbit].values-(start_orbit+i+1))<0.01]
        # Store the orbti nr. and the start time of the orbit to create x-axis later
        fig_orbit[i] = data_column1.Orbit_nr[0]
        fig_dates[i] = np.datetime64(data_column1.index[0])
        # Set index to Latitude
        data_column1 = data_column1.set_index(latitude)
        data_column2 = data_column2.set_index(latitude)
        # interpolate the Densities to the Latitude given by fig_index
        data_column1 = data_column1.reindex(labels=fig_index1, method='nearest', tolerance=lat_tolerance).loc[:,pos].values
        data_column2 = data_column2.reindex(labels=fig_index2, method='nearest', tolerance=lat_tolerance).loc[:,pos].values
        # store the values in matrix
        fig_values[:,i] = np.hstack([data_column1,data_column2])
    fig_dates = pd.to_datetime(fig_dates)

    return fig_values,fig_index,fig_dates,fig_orbit

#%%

def add_pyamps_currents(df, h = 470):
    """
    Input: DataFrame with time index containing:
     [Bulk_speed,BY_GSM,BZ_GSM,F10_INDEX,Hemisphere]
    Get an estimate of the total FAC current at the time specified by the index
    at the pole given by the Hemisphere.
    The result is added to the DataFrame
    """

    # create model
    m = AMPS(df.Bulk_speed.values[0], # Solar wind velocity in km/s
         df.BY_GSM.values[0], # IMF By (GSM) in nT
         df.BZ_GSM.values[0], # IMF Bz (GSM) in nT,
         dipole_tilt_angle(df.index[0])/np.pi*180, # dipole tilt angle in degrees
         df.F10_INDEX.values[0], # F107_index
         height = h )

    # initial array for results
    N=len(df)
    Total_J = np.zeros(N)

    for i in range(N):
        # update model
        m.update_model(df.Bulk_speed.values[i], # Solar wind velocity in km/s
             df.BY_GSM.values[i], # IMF By (GSM) in nT
             df.BZ_GSM.values[i], # IMF Bz (GSM) in nT,
             dipole_tilt_angle(df.index[i])/np.pi*180, # dipole tilt angle in degrees
             df.F10_INDEX.values[i]) # F107_index
        J_up_n, J_down_n, J_up_s, J_down_s = m.get_integrated_upward_current()

        if df.Hemisphere.values[i]==1:
            Total_J[i] = abs(J_up_n)+abs(J_down_n)
        else:
            Total_J[i] = abs(J_up_s)+abs(J_down_s)

        df.loc[:,'PyAmps']= Total_J

    return None

#%%

def sun_unit_vector(t):
    """credit: Eelco Doornbos"""
    from apexpy.helpers import subsol
    lat, lon = subsol(t)
    colatrad = radians(90.0-lat)
    lonrad   = radians(lon)

    return np.array([sin(colatrad)*cos(lonrad), sin(colatrad)*sin(lonrad), cos(colatrad)])

def pole_unit_vector(t):
    """credit: Eelco Doornbos"""
    apexdate = t.year + t.dayofyear/365 # routine needs data as for example 2015.3
    A = Apex(date=apexdate)
    glat, glon = A.convert(90, 0, 'apex', 'geo', height=0)
    colatrad = radians(90.0-glat)
    lonrad = radians(glon)

    return np.array([sin(colatrad)*cos(lonrad), sin(colatrad)*sin(lonrad), cos(colatrad)])

def dipole_tilt_angle(t):
    """credit: Eelco Doornbos"""
    return asin(np.dot(sun_unit_vector(t), pole_unit_vector(t)))

#%%
def correlation(df,var1,var2,minlag = -5,maxlag = 15):
    """
    Checks correlation between var1 and var2 and several lags. The lags are
    specifed in half orbits. Does not include
    observations near a gap in time. The time shift is applied to var2
    Input:
    df: Half orbit means and orbit no. (dataframe)
    var1: name of varible in dataframe (str)
    var2: list of m names of varibelse in dataframe (list)

    returns:
    corr: numpy matrix with correlation between each var2 varible and var1 at
        the different lags
    N: array with no. of observations used to calculete the correlation for each
        lag
    lag_array: array with the different lags
    """

    # Initilaize arrays
    lag_array = np.array(range(minlag,maxlag))
    corr = np.zeros([len(var2),len(lag_array)])
    N = np.zeros(len(lag_array))
    dt_array = np.zeros(len(lag_array))

    for i in range(len(lag_array)):
        lag = lag_array[i]

        if lag < 0:
            # get time difference
            dt = df.index[-lag:]-df.index[:lag]
            # Get indx where the time gaps are small
            indx = abs(dt.values-np.median(dt.values))< -lag*np.timedelta64(5,'m')
            # Create DataFrame where var2 is shifted.
            data = np.hstack([df.loc[:,var1].values[:lag][:,np.newaxis], df.loc[:,var2].values[-lag:]])
            df_lag = pd.DataFrame(data, index=df.index[-lag:], columns=[var1]+var2)
            # get the avareges time difference and remove gaps
            dt_mean = -np.mean(dt[indx]).astype('float')

        if lag > 0:
            dt = df.index[lag:]-df.index[:-lag]
            indx = abs(dt.values-np.median(dt.values))< lag*np.timedelta64(5,'m')
            data = np.hstack([df.loc[:,var1].values[lag:][:,np.newaxis], df.loc[:,var2].values[:-lag]])
            df_lag = pd.DataFrame(data, index=df.index[lag:], columns=[var1]+var2)
            dt_mean = np.mean(dt[indx]).astype('float')

        if lag == 0:
            indx = np.ones(len(df)).astype(bool)
            data = np.hstack([df.loc[:,var1].values[:,np.newaxis], df.loc[:,var2].values])
            df_lag = pd.DataFrame(data, index=df.index, columns=[var1]+var2)
            dt_mean = 0

        # Compute and store the corelations
        corr[:,i] = df_lag[indx].corr().values[0,1:]
        N[i] = len(df_lag[indx])
        dt_array[i] = dt_mean/(10**9*60*60)

    return corr,N,lag_array,dt_array

#%%
def shift_time(df,var1,var2,lag):
    """Shift var2 lag places with respect to var1.
        Returns: DataFrame
    """

    dt = df.index[lag:]-df.index[:-lag]
    indx = abs(dt.values-np.median(dt.values))< lag*np.timedelta64(5,'m')

    if type(var2) == list:
        var = [var1]+var2
        data = np.hstack([df.loc[:,var1].values[lag:][:,np.newaxis], df.loc[:,var2].values[:-lag]])
    else:
        var = [var1]+[var2]
        data = np.vstack([df.loc[:,var1].values[lag:], df.loc[:,var2].values[:-lag]])
        data = np.transpose(data)

    Delayed_df = pd.DataFrame(data, index=df.index[lag:], columns=var)
    Delayed_df = Delayed_df[indx]
    dt_mean = np.mean(dt[indx])

    return Delayed_df,dt_mean

#%%

def add_NaN_gap(df,dt=np.timedelta64(2,'h')):
    """Input: DataFrame
    adds NaN values where there is a time gap greater than dt
    """
    gaps=(df.index.values[1:]-df.index.values[:-1])> dt
    gaps = np.append(False,gaps)
    t_NaN = df.index[gaps] - 0.9*dt
    gaps = pd.DataFrame(np.ones([len(t_NaN),len(df.columns)])*float('NaN'), index=t_NaN,columns=df.columns)
    df2 = df.append(gaps)
    df2 = df2.sort_index()
    return df2
