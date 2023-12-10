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


index_vals_m0 = []
func_vals_m0 = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m0[i]
    
    rhs = rhs_m0[i]
    
    if np.abs(lhs - rhs) < 0.00001:
        
        index_vals_m0.append(i)
        
        func_vals_m0.append(lhs)
        
        
print(len(index_vals_m0))

beta_val_m0 = beta_test_array_valid[index_vals_m0[0]]

plt.plot(beta_test_array_valid[index_vals_m0[0] - 5: index_vals_m0[0] + 5], lhs_m0[index_vals_m0[0] - 5: index_vals_m0[0] + 5])
plt.plot(beta_test_array_valid[index_vals_m0[0] - 5: index_vals_m0[0] + 5], rhs_m0[index_vals_m0[0] - 5: index_vals_m0[0] + 5])
plt.show()

plt.plot(p_vals * a, lhs_m0)
plt.plot(q_vals * a, rhs_m0)
plt.show()


#%%


# Repeating for m = 1:

lhs_m1 = jv(2, (p_vals * a)) / (p_vals * a * jv(1, (p_vals * a)))

rhs_m1 = - kv(2, (q_vals * a)) / (q_vals * a * kv(1, (q_vals * a)))

index_vals_m1 = []
func_vals_m1 = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m1[i]
    
    rhs = rhs_m1[i]
    
    if np.abs(lhs - rhs) < 0.00003:
        
        index_vals_m1.append(i)
        
        func_vals_m1.append(lhs)
        
print(len(index_vals_m1))

beta_val_m1 = beta_test_array_valid[index_vals_m1[0]]

plt.plot(beta_test_array_valid[index_vals_m1[0] - 5: index_vals_m1[0] + 5], lhs_m1[index_vals_m1[0] - 5: index_vals_m1[0] + 5])
plt.plot(beta_test_array_valid[index_vals_m1[0] - 5: index_vals_m1[0] + 5], rhs_m1[index_vals_m1[0] - 5: index_vals_m1[0] + 5])
plt.show()


plt.plot(p_vals * a, lhs_m1)
plt.plot(q_vals * a, rhs_m1)
plt.show()



#%%


# Repeating for m = 2:

lhs_m2 = jv(3, (p_vals * a)) / (p_vals * a * jv(2, (p_vals * a)))

rhs_m2 = - kv(3, (q_vals * a)) / (q_vals * a * kv(2, (q_vals * a)))

index_vals_m2 = []
func_vals_m2 = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m2[i]
    
    rhs = rhs_m2[i]
    
    if np.abs(lhs - rhs) < 0.001:
        
        index_vals_m2.append(i)
        
        func_vals_m2.append(lhs)
        
print(len(index_vals_m2))

beta_val_m2 = beta_test_array_valid[index_vals_m2[0]]

plt.plot(beta_test_array_valid[index_vals_m2[0] - 500: index_vals_m2[0] + 500], lhs_m2[index_vals_m2[0] - 500: index_vals_m2[0] + 500])
plt.plot(beta_test_array_valid[index_vals_m2[0] - 500: index_vals_m2[0] + 500], rhs_m2[index_vals_m2[0] - 500: index_vals_m2[0] + 500])
plt.show()

plt.plot(p_vals * a, lhs_m2)
plt.plot(q_vals * a, rhs_m2)
plt.show()


# PLOT THESE AGAINST V TO REPRODUCE PLOTS FROM THE NOTES

#%%

# Repeating for m = 3:

# no solution for m = 3 or m = 4

# fibre has 3 modes  (0,1,2)

lhs_m3 = jv(4, (p_vals * a)) / (p_vals * a * jv(3, (p_vals * a)))

rhs_m3 = - kv(4, (q_vals * a)) / (q_vals * a * kv(3, (q_vals * a)))

index_vals_m3 = []
func_vals_m3 = []

for i in np.arange(0, len(beta_test_array_valid)):
    
    lhs = lhs_m3[i]
    
    rhs = rhs_m3[i]
    
    if np.abs(lhs - rhs) < 1:
        
        index_vals_m3.append(i)
        
        func_vals_m3.append(lhs)
        

print(len(index_vals_m3))

beta_val_m3 = beta_test_array_valid[index_vals_m3[0]]



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

# fibre has modes 0,1,2 - appropriate for V value close to 5

# find propagation constant and effective index for each mode

# beta is the propagation constant

# beta_bar = beta / k0 is the effective index

effective_index_m0 = beta_val_m0 / k0

effective_index_m1 = beta_val_m1 / k0

effective_index_m2 = beta_val_m2 / k0


#%%

print('Mode 0:')
print(beta_val_m0)
print(effective_index_m0)


print('Mode 1:')
print(beta_val_m1)
print(effective_index_m1)

print('Mode 2:')
print(beta_val_m2)
print(effective_index_m2)

#%%


# looking at mode 2 for tasks 5-8

# set A to unity

B = jv(3, (p_vals*a)) / (kv(3, q_vals * a))


#%%


def calc_Dw(pval, step=0.0001):
    c = 3e10
    h = wavelength * step
    
    wave0 = wavelength - wavelength * step
    wave1 = wavelength
    wave2 = wavelength + wavelength * step
    
    
    neff_0 = calc_neff(pval, (2 * np.pi) / (wave0 * n1))
    neff_1 = calc_neff(pval, (2 * np.pi) / (wave1 * n1))
    neff_2 = calc_neff(pval, (2 * np.pi) / (wave2 * n1))
    
    return - (wavelength / c) * ((neff_2 - (2 * neff_1) + neff_0) / h**2)

def calc_neff(pval, k):
    beta = np.sqrt(n1 ** 2 * k ** 2 - pval ** 2)
    return beta / k


    


#%%

D_w = calc_Dw(p_vals[4455])
print(D_w)























