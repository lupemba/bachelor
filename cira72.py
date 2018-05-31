#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 18:20:23 2018

@author: simon
"""

import numpy as np
import math 

def aden(AMJD,Sun_rAscension,Sun_declination,
         rAscension,Declination,height,F10,F10_mean,Kp=0):
    
    """
    !!!! Remember Kp from 0.27 days earlier. !!!!!
    
    Comments
    @author: simon
    """
    Temp = [0,0]
    # Define varibles as in the CIRA 72
    sun = np.array([Sun_rAscension,Sun_declination])
    sat = np.array([rAscension,Declination,height])
    geo = np.array([F10,F10_mean,Kp])
    
    
    # The alpha are the thermal diffusion coefficients in equation (6)
    alpha =  np.zeros(5)
    alpha[4] = -0.38
    
    # ln(10)
    al10 = math.log(10)
    
    # The AMW are molecular weights in the order N2, O2, O, A, He,H
    AMW =  np.zeros(6)
    
    AMW[0] = 28.0134
    AMW[1] = 31.9988
    AMW[2] = 15.9994
    AMW[3] = 39.948
    AMW[4] =4.0026
    AMW[5] =1.00797
    
    # Advogadros number
    avogad = 6.02257E26
    
    # The frac are the assumed sea_level volume-fractions in the order 
    #N2,O2,A,He
    frac =  np.zeros(4)
    
    frac[0] = 0.78110
    frac[1] = 0.20955
    frac[2] = 9.3432E-3
    frac[3] = 6.1471E-6
    
    # Rstar is the universal gas-constant in mks-units
    Rstar = 8314.32
    
    # The R are values used to establish height-step-sizes in the regimes 
    #90-100 km, 100-500 km and above 500 km
    R1 = 0.01
    R2 = 0.025
    R3 = 0.075
    
    # The WT are the weights used for the Newton-cotes Five-point 
    # quadrature formula
    WT =  np.zeros(5)
    
    WT[0] = 0.31111111
    WT[1] = 1.4222222
    WT[2] = 0.53333333
    WT[3] = WT[1]
    WT[4] = WT[0] 
    
    # Equation (14)    
    T_c = 379.0 +3.24*geo[1]+1.3*(geo[0]-geo[1])

    # Equation (15)
    eta = 0.5*abs(sat[1]-sun[1])
    theta = 0.5*abs(sat[1]+sun[1])
    
    # Equation (16)
    H = sat[0]-sun[0]
    tau = H-0.64577182+0.10471976*np.sin(H+0.75049158)
    
    # Equation (17)
    DF=np.sin(theta)**2.2+(np.cos(eta)**2.2-np.sin(theta)**2.2)*abs(np.cos(0.5*tau))**3
    T_l=T_c*(1.0+0.3*DF)
    
    # Equation (18)
    dtg18=28.0*geo[2]+0.03*math.exp(geo[2])
    
    # Equation (20)
    dtg20 = 14.0*geo[2]+0.02*math.exp(geo[2])
    dlr20 = 0.012*geo[2]+1.2E-5*math.exp(geo[2])
    
    # The following statements effect a contininuous
    # transition from equation (20) at height well below 
    # 350 km to equation (18) at heights well above 350 km
    F = 0.5*(np.tanh(0.04*(sat[2]-350.0))+1)
    dlrgm = dlr20*(1-F)
    dtg = dtg20*(1-F)+dtg18*F
    
    # Tinf for out put
    Tinf = T_l +dtg
    Temp[0] = Tinf
    # Equation (9)
    T_x = 371.6678+0.0518806*Tinf-294.3505*np.exp(-0.00216222*Tinf)
    
    # Equation (11)
    G_x = 0.054285714*(T_x-183.0)
    
    #The TC array will be an argument in the call to Tlocal,
    #which evaluates eaquation (10) or (13)
    TC = [T_x, G_x, (Tinf-T_x)*2/math.pi,0]
    TC[3] = G_x/TC[2]
    
    
    # Equation (5)
    z1= 90.0
    z2 = min(sat[2],100)
    al = np.log(z2/z1)
    N = (al/R1).astype('int')+1
    zr = np.exp(al/N)
    ambar1 = Ambar(z1)
    T_loc1 = Tlocal(z1,TC)
    z_end = z1
    sum2 = 0
    Ain = ambar1*Grav(z1/T_loc1)
    for i in range(N):
        z = z_end
        z_end = zr*z
        dz = 0.25*(z_end-z)
        sum1 = WT[0]*Ain
        for j in range(1,5):
            z = z+dz
            ambar2 = Ambar(z)
            T_loc2 = Tlocal(z,TC)
            gravl = Grav(z)
            Ain = ambar2*gravl/T_loc2
            sum1 = sum1 + WT[j]*Ain
        sum2 = sum2 + dz*sum1
    fact1 = 1000.0/Rstar
    rho=3.46E-6*ambar2*T_loc1*np.exp(-fact1*sum2)/ambar1/T_loc2 
    
    
    # Equation (2) 
    ANM = avogad*rho
    AN =  ANM/ambar2
    
    # Equation (3)
    fact2 = ANM/28.960
    
    Aln = [0,0,0,0,0,0]
    
    Aln[0] = np.log(frac[0]*fact2)
    Aln[3] = np.log(frac[2]*fact2)
    Aln[4] = np.log(frac[3]*fact2)
   
    # Equation (4)
    Aln[1] = np.log(fact2*(1+frac[1])-AN)
    Aln[2] = np.log(2*(AN-fact2))
    
    if sat[2] >100:
        
        # Equation (6)
        z3 = min(sat[2],500)
        al = np.log(z3/z)
        N = (al/R2).astype('int')+1
        zr = np.exp(al/N)
        sum2 = 0
        Ain = gravl/T_loc2
        for i in range(N):
            z = z_end
            z_end = zr*z
            dz = 0.25*(z_end-z)
            sum1 = WT[0]*Ain
            for j in range(1,5):
                z = z+dz
                T_loc3 = Tlocal(z,TC)
                gravl = Grav(z)
                Ain = gravl/T_loc3
                sum1 = sum1 + WT[j]*Ain
            sum2 = sum2 + dz*sum1
        
        z4 = max(sat[2],500.0)
        al = np.log(z4/z)
        R = R2
        if sat[2] > 500.0:
            R = R3
        N = (al/R).astype('int')+1
        zr = np.exp(al/N)
        sum3 = 0
        for i in range(N):
            z = z_end
            z_end = zr*z
            dz = 0.25*(z_end-z)
            sum1 = WT[0]*Ain
            for j in range(1,5):
                z = z+dz
                T_loc4 = Tlocal(z,TC)
                gravl = Grav(z)
                Ain = gravl/T_loc4
                sum1 = sum1 + WT[j]*Ain
            sum3 = sum3 + dz*sum1
        if sat[2] > 500.0:
            # GOTO 8
            T500 = T_loc3
            Temp[1] = T_loc4
            Altr = np.log(T_loc4/T_loc2)
            fact2 = fact1*(sum2+sum3)
            Hsign = -1.0
        else:
            T500 = T_loc4
            Temp[1] = T_loc3
            Altr = np.log(T_loc3/T_loc2)
            fact2 = fact1*sum2
            Hsign = 1.0
        for i in range(5):
            Aln[i] = Aln[i]-(1+alpha[i])*Altr-fact2*AMW[i]
        
        # equation (7)
        Al10T5 = np.log10(T500)
        AlnH5=(5.5*Al10T5-39.40)*Al10T5+73.13
        Aln[5]=al10*(AlnH5+6.0)+Hsign*(np.log(T_loc4/T_loc3)+fact1*sum3 *AMW[5])
        
    else:
        Temp[1]=T_loc2
        # Put in neglible hydrogen for use in loop line ?????
        Aln[5]=Aln[4]-25.0
        # GOTO 11
    
    # Equation (23)
    capphi = ((AMJD-36204.0)/365.2422)%1.0
    
    # Equation (22)
    tau=capphi+0.09544*((0.5+0.5*np.sin(2*np.pi*capphi+6.035))**1.650-0.5)
    goft =0.02835+0.3817*(1.0+0.4671*np.sin(2*np.pi*tau+4.137))*np.sin(4*np.pi*tau+4.259)
    fofz=(5.876E-07*sat[2]**2.331+0.06328)*np.exp(-2.868E-03*sat[2])
     
    # Equation (21)
    dlrsa =  goft * fofz
    
    # Equation (24)
    dlrsl=0.014*(sat[2]-90.0)*np.exp(-0.0013*(sat[2]-90.0)**2)*np.sign(sat[1])*np.sin(2*np.pi*capphi+1.72)*np.sin(sat[1])**2
    
    # Sum the delta-log-rhos and apply to the number densities 
    dlr = al10*(dlrgm+dlrsa+dlrsl)
    Aln = Aln +dlr
    
    # Equation (25)
    dlnHe = 0.65*abs(sun[1]/0.4091609)*(np.sin(np.pi/4-0.5*sat[1]*np.sign(sun[1]))**3-np.sin(np.pi/4)**3)
    Aln[4]=Aln[4]+al10*dlnHe
    
    # Compute mass-density and mean-molecular-weight
    # and convert number-density logarithms from natural
    #to commen
    
    sumn = 0.0
    sumnm = 0.0
    for i in range(6):
        an = np.exp(Aln[i])
        sumn = sumn+an
        sumnm = sumnm +an*AMW[i]
    alion = Aln/al10
    AMMW = sumnm/sumn
    rho = sumnm/avogad
    
    return Temp,alion,AMMW,rho 

#%%


def Tlocal(z,TC):
   " Evaluates equation (10) or equation (13) depending on z"
   dz = z-125.0
   if dz<0:
       T_loc =((-9.8204695E-06*dz-7.3039742E-04)*dz**2+1.0)*dz*TC[1]+TC[0]
   else:
       T_loc = TC[0]+TC[2]*np.arctan(TC[3]*dz*(1.0+4.5E-06*dz**2.5))
        
   return T_loc

def Ambar(z):
    "Evaluates equation (1)"
    c =  np.zeros(7)
    
    c[0] = 28.82678
    c[1] = -7.40066E-02
    c[2] =-1.19407E-02
    c[3] = 4.51103E-04
    c[4] = -8.21895E-06
    c[5] = 1.07561E-05
    c[6] = -6.97444E-07
    
    dz = z-90
    amb = c[0] + c[1]*dz+c[2]*dz**2 + c[3]*dz**3 + c[4]*dz**4 + c[5]*dz**5 + c[6]*dz**6
        
    return amb


def Grav(z):
    "Evaluates equation (8)"
    return 9.80665/(1.0+z/6356.766)**2













