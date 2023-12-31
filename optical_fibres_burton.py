# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:28:34 2023

@author: burto
"""

import pandas as pd
from scipy.optimize import fsolve
from scipy.constants import pi
from scipy.special import jv, kv
import numpy as np
import matplotlib.pyplot as plt


#%%

"Defining constants/ given parameters"

pi= np.pi
a = 5e-6                                 # Core radius
n1 = 1.51                                # Core refractive index
n2 = 1.49                                # Cladding refractive index
wavelength = 1450e-9                     # Wavelength
k0 = (2 * pi)/ wavelength                   # Wavevector


#%%

"Task 1"

V = a * k0 * np.sqrt(n1**2 - n2**2)


#%%

"Task 2"

# We will use the LP modes as we are working with the assumption
# that n1=~n2


#%%

"Task 3"

# Create array of beta values with range based on physical limits

beta_min = n2*k0
beta_max = n1*k0
beta_test_array = np.linspace(beta_min, beta_max, 100000)
beta_test_array_valid = beta_test_array[1:-1]

# Create arrays of p and q from beta for use in Bessell functions

p_vals = np.sqrt(((n1**2) * (k0 ** 2)) - (beta_test_array_valid**2))

q_vals = np.sqrt((beta_test_array_valid**2) - ((n2**2) * (k0 ** 2)))

# Define function for left and right hand side

def get_lhs(l,p):
    lhs = jv(l, (p * a)) / (p * a * jv(l - 1, (p * a)))          
    return lhs

def get_rhs(l,q):
    rhs = - kv(l, (q * a)) / (q * a * kv(l - 1, (q * a)))
    return rhs

# Function for finding solutions for all l values between a min
# and max l value specified by "l_max" and "l_min"

def get_solutions(lmin,lmax):
    
    l_range = np.arange(lmin,lmax+1,1)
    
    for l in l_range:
        
        index_vals = []
    
        lhs_arr = get_lhs(l,p_vals)
        rhs_arr = get_rhs(l,q_vals)
        diff = abs(lhs_arr-rhs_arr)
    
        for i in np.arange(0, len(beta_test_array_valid)):

            if diff[i] < 0.00003:     
                index_vals.append(i)  
                print("For mode l=",l,"Beta equals",beta_test_array_valid[i])
    
    return index_vals



#%%
# Just run this to return the beta values, ignore the duplicates! 
get_solutions(0,4)


