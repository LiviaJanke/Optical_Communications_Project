# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:09:23 2023

@author: eichm
"""

import pandas as pd
from scipy.optimize import fsolve
from scipy.constants import pi
from scipy.special import jv, kv
import numpy as np
import matplotlib.pyplot as plt


#%%

# Given parameters
wavelength = 1450e-9  # Wavelength in meters
k0 = (2 * pi) / wavelength
n1 = 1.51
n2 = 1.49
a = 5e-6  # Core radius in meters



#%%

# create total beta values range
# n_2 * k_0 < beta < n_1 * k_0
beta_min = n2 * k0
beta_max = n1 * k0
beta_test_array = np.linspace(beta_min, beta_max, 100000)
beta_test_array_valid = beta_test_array[1:-1]

#%%

p_vals = np.sqrt(((n1**2) * (k0 ** 2)) - (beta_test_array_valid**2))

q_vals = np.sqrt((beta_test_array_valid**2) - ((n2**2) * (k0 ** 2)))


#%%

# Plot for m = 0:
    
lhs_m0 = jv(1, (p_vals * a)) / (p_vals * a * jv(0, (p_vals * a)))

rhs_m0 = - kv(1, (q_vals * a)) / (q_vals * a * kv(0, (q_vals * a)))

#not too sure what these LHS and RHS series should be plotted against to reproduce the graphs shown in the lectures?


plt.plot(beta_test_array_valid, lhs_m0)
plt.plot(beta_test_array_valid, rhs_m0)
plt.show()


plt.plot(p_vals * a, lhs_m0)
plt.plot(q_vals * a, rhs_m0)
plt.show()



#%%


index_vals = []
func_vals = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m0[i]
    
    rhs = rhs_m0[i]
    
    if np.abs(lhs - rhs) < 0.00001:
        
        index_vals.append(i)
        
        func_vals.append(lhs)
        
        


beta_val_m0 = beta_test_array_valid[index_vals[0]]



#%%


# Repeating for m = 1:

lhs_m1 = jv(2, (p_vals * a)) / (p_vals * a * jv(1, (p_vals * a)))

rhs_m1 = - kv(2, (q_vals * a)) / (q_vals * a * kv(1, (q_vals * a)))

index_vals = []
func_vals = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m1[i]
    
    rhs = rhs_m1[i]
    
    if np.abs(lhs - rhs) < 0.0001:
        
        index_vals.append(i)
        
        func_vals.append(lhs)
        


beta_val_m1 = beta_test_array_valid[index_vals[0]]


#%%


# Repeating for m = 2:

lhs_m2 = jv(3, (p_vals * a)) / (p_vals * a * jv(2, (p_vals * a)))

rhs_m2 = - kv(3, (q_vals * a)) / (q_vals * a * kv(2, (q_vals * a)))

index_vals = []
func_vals = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m2[i]
    
    rhs = rhs_m2[i]
    
    if np.abs(lhs - rhs) < 0.001:
        
        index_vals.append(i)
        
        func_vals.append(lhs)
        


beta_val_m2 = beta_test_array_valid[index_vals[0]]

#%%

# Repeating for m = 3:

lhs_m3 = jv(4, (p_vals * a)) / (p_vals * a * jv(3, (p_vals * a)))

rhs_m3 = - kv(4, (q_vals * a)) / (q_vals * a * kv(3, (q_vals * a)))

index_vals = []
func_vals = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m3[i]
    
    rhs = rhs_m3[i]
    
    if np.abs(lhs - rhs) < 0.001:
        
        index_vals.append(i)
        
        func_vals.append(lhs)
        


beta_val_m3 = beta_test_array_valid[index_vals[0]]

#%%

# Repeating for m = 4:

lhs_m4 = jv(5, (p_vals * a)) / (p_vals * a * jv(4, (p_vals * a)))

rhs_m4 = - kv(5, (q_vals * a)) / (q_vals * a * kv(4, (q_vals * a)))

index_vals = []
func_vals = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m4[i]
    
    rhs = rhs_m4[i]
    
    if np.abs(lhs - rhs) < 1:
        
        index_vals.append(i)
        
        func_vals.append(lhs)
        


beta_val_m4 = beta_test_array_valid[index_vals[0]]

plt.plot(beta_test_array_valid[37990:99997], lhs_m4[37990:99997])
plt.plot(beta_test_array_valid[37990:99997], rhs_m4[37990:99997])
plt.show()


plt.plot((p_vals * a)[37990:99997], lhs_m0[37990:99997])
plt.plot((q_vals * a)[37990:99997], rhs_m0[37990:99997])
plt.show()


# Mode 4 is not included; no valid solutions

#%%

# find propagation constant and effective index for each mode





















