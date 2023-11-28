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
    



















