# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 14:09:23 2023

@author: eichm
"""

import pandas as pd
from scipy.optimize import fsolve
from scipy.constants import pi, epsilon_0, mu_0
from scipy.special import jv, kv, jvp
import numpy as np
import matplotlib.pyplot as plt
from cmath import pi, e, polar
from numpy import linspace, vectorize, sin, cos

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
m = 2

beta = beta_val_m2

q = np.sqrt((beta**2) - ((n2**2) * (k0**2)))

p = np.sqrt(((n1**2) * (k0**2)) - (beta**2))


# set A to unity

A = 1

B = A *  jv(m, (p*a)) / (kv(m, q * a))



phi = np.linspace(0, (2 * np.pi), 1000)

r = np.linspace(-a, a, 1000)


omega = np.sqrt((k0 **2)/ (epsilon_0 * mu_0))







#%%


E_z = A * jv(2, (p * r))


plt.plot(r, E_z)
plt.show()


E_r = - ((1j * beta) / (p **2))  *  ((A * p * jvp(m, (p * r), n = 1)) + (1j * omega * (mu_0 * m / beta * r) * B * jv(m, (p * r))) )

plt.plot(r, E_r)
plt.show()


E_phi =  - ((1j * beta) / (p **2)) * (((1j * m / r) * A * jv(m, (p * r))) - ((omega * mu_0 / beta) * p * B * jvp(m, (p * r), n=1)))


plt.plot(r, E_phi)
plt.show()

#%%

l = 1
# l can also be m + 1 or m - 1

E_vals = E_r * np.exp((1j * l * E_phi) - (1j * beta * E_z))

plt.plot(r, E_vals)

#%%

# Creating a new figure and setting up the resolution
fig = plt.figure(dpi=200)

# Change the coordinate system from scaler to polar
ax = fig.add_subplot(projection='polar')


plt.polar(E_phi,E_r,marker='o')

# Displaying the plot
plt.plot()

#%%

# E_total in the x-y plane:
    
E_tot = E_r + E_phi

# Plotting Argand Diagrams

fig,ax = plt.subplots()

ax.scatter(E_tot.real,E_tot.imag, marker = '.')

#%%


# Plot numbers on polar projection
# Yeah this isn't working too great

# https://stackoverflow.com/questions/17445720/how-to-plot-complex-numbers-argand-diagram-using-matplotlib



#vect_polar = vectorize(polar)
#rho_theta = vect_polar(E_r)

#fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
#ax.stem(rho_theta[1], rho_theta[0])

# Get a number, find projections on axes

#for i in np.arange(0, len(rho_theta[0]), 1):
#    n = i
#    rho, theta = rho_theta[0][n], rho_theta[1][n]
#    a = cos(theta)
#    b = sin(theta)
#    rho_h, theta_h = abs(a)*rho, 0 if a >= 0 else -pi
#    rho_v, theta_v = abs(b)*rho, pi/2 if b >= 0 else -pi/2


    # Plot h/v lines on polar projection
#    ax.plot((theta_h, theta), (rho_h, rho), c='r', ls='--')
#    ax.plot((theta, theta_v), (rho, rho_v), c='g', ls='--')

#%%

# what about just multiplying into x-y?

# x = r cos theta
# y = r sin theta

#cmath.polar(x)
#Return the representation of x in polar coordinates. Returns a pair (r, phi) where r is the modulus of x and phi is the phase of x. polar(x) is equivalent to (abs(x), phase(x)).
x_vals = []
y_vals = []

x_vals_phi = []
y_vals_phi = []

r_vals = []
theta_vals = []

for i in np.arange(0, len(E_r), 1):
    E_r_polar = polar(E_r[i])
    r_vals.append(E_r_polar[0])
    #theta_vals.append(E_r_polar[1])
    x = E_r_polar[0] * np.cos(E_r_polar[1])
    y = E_r_polar[0] * np.sin(E_r_polar[1])
    x_vals.append(x)
    y_vals.append(y)
    
plt.plot(x_vals,y_vals)
plt.show()


for i in np.arange(0, len(E_phi), 1):
    E_r_polar = polar(E_phi[i])
    #r_vals.append(E_r_polar[0])
    theta_vals.append(E_r_polar[1])
    x = E_r_polar[0] * np.cos(E_r_polar[1])
    y = E_r_polar[0] * np.sin(E_r_polar[1])
    x_vals_phi.append(x)
    y_vals_phi.append(y)
    
plt.plot(x_vals_phi,y_vals_phi)
plt.show()

# this kinda works for plotting - not too sure about the equations though!

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(theta_vals, r_vals, linewidth = 0, marker = '.')
plt.show()

# Yeah that doesn't seem quite right
# Trying the cartesian equations?

#%%

# Intentsity distribution in polars

# amplitude from intensity formula

intensity_polar = (np.abs(E_r) ** 2) + (np.abs(E_phi **2))
plt.plot(r, intensity_polar)

#%%

amplitude_polar = np.sqrt((np.abs(E_r) ** 2) + (np.abs(E_phi **2)))
plt.plot(r, amplitude_polar)


#%%

# Cartesian version


r = np.linspace(-a, a, 1000)
r_clad_pos = np.linspace(a, 3 * a, 500)
r_clad_neg = np.linspace(-3 * a, - a, 500)
r_clad = np.hstack((r_clad_neg, r_clad_pos))
r_tot = np.hstack((r_clad_neg,r, r_clad_pos))

m = 2
beta = beta_val_m2

phi = np.linspace(0, (2 * np.pi), 1000)

#x_vals_core = r * np.cos(phi)
#y_vals_core = r * np.sin(phi)

#x_vals_clad = r_clad * np.cos(phi)
#y_vals_clad = r_clad * np.sin(phi)

pos_x_phi = np.hstack((phi[750:], phi[:250]))
neg_x_phi = phi[250:750]

phi_neg_to_pos = np.hstack((neg_x_phi, pos_x_phi))

x_vals_core = r * np.cos(phi_neg_to_pos)
x_vals_clad = r_clad * np.cos(phi_neg_to_pos)

y_vals_core = r * np.sin(phi_neg_to_pos)
y_vals_clad = r_clad * np.sin(phi_neg_to_pos)




#%%

# Y-polarised modes

# core r < a

l = m

A = 1
B = A *  jv(l, (p*a)) / (kv(l, q * a))

# l can also be m + 1 or m - 1
# this may be the 3 fields that the question mentioned?
# maybe

E_x_core_y_pol = 0 * r

E_y_core_y_pol = A * jv(l, (p * r))

E_z_core_y_pol = (p / beta) * (A / 2) * ((jv(l+1, (p * r)) * np.exp(1j * phi)) + (jv(l-1, (p * r)) * np.exp(-1j * phi)))


# amplitude in plane perpendicular to fibre axis - just E_y in this case since E_x is zero?
# how does E_z factor in?

# cladding r > a

E_x_clad_y_pol = 0 * r

E_y_clad_y_pol = B * kv(l, (q * np.abs(r_clad)))

E_z_clad_y_pol = (q / beta) * (B/2) * ((kv(l+1, (q * r_clad)) * np.exp(1j * phi)) - (kv(l-1, (q * r_clad)) * np.exp(-1j * phi)))


plt.plot(r, E_y_core_y_pol, label = 'E_y')
plt.plot(r, E_z_core_y_pol, label = 'E_z')
plt.plot(r, E_x_core_y_pol, label = 'E_x')
plt.plot(r_clad, E_y_clad_y_pol)
plt.plot(r_clad, E_z_clad_y_pol)
plt.plot(r_clad, E_x_clad_y_pol)
plt.legend()
plt.show()

# X - polarised modes

# core r < a

E_x_core_x_pol = A * jv(l, (p * r))

E_y_core_x_pol = r * 0

E_z_core_x_pol = 1j * (p / beta) * (A/2) * ((jv(l+1, p * r) * np.exp(1j * phi)) - (jv(l-1, p * r) * np.exp(-1j * phi)))

# cladding r > a

E_x_clad_x_pol = A * kv(l, (p * np.abs(r_clad)))

E_y_clad_x_pol = r * 0

E_z_clad_x_pol = 1j * (q/beta) * (B/2) * ((kv(l+1, q * r_clad) * np.exp(1j * phi)) + (kv(l-1, q * r_clad) * np.exp(-1j * phi)))

# something strange going on with plotting complex values here?



plt.plot(r, E_y_core_x_pol, label = 'E_y')
plt.plot(r, E_z_core_x_pol, label = 'E_z')
plt.plot(r, E_x_core_x_pol, label = 'E_x')
plt.plot(r_clad, E_y_clad_x_pol, label = 'E_y')
plt.plot(r_clad, E_z_clad_x_pol, label = 'E_z')
plt.plot(r_clad, E_x_clad_x_pol, label = 'E_x')
plt.legend()
plt.show()

amplitude_core_1 = np.sqrt((E_y_core_y_pol ** 2) + (E_x_core_x_pol ** 2))


amplitude_clad_1 = np.sqrt((E_y_clad_y_pol ** 2) + (E_x_clad_x_pol ** 2))


e_field_amp_1 = np.hstack((amplitude_core_1, amplitude_clad_1))

#%%

plt.plot(x_vals_core, E_x_core_x_pol)
#plt.show()

plt.plot(y_vals_core, E_y_core_y_pol)
plt.show()

x_y_coords = np.column_stack((x_vals_core, y_vals_core))

# x and y are the same
# only z is different
# god fucking damnit

E_z_core_tot =  E_z_core_x_pol + E_z_core_y_pol

plt.plot(r, E_z_core_tot)
plt.show()

z_vals = np.linspace(0,5,1000)


plt.plot(z_vals, np.abs(E_z_core_tot))

#%%



y,x = np.meshgrid(y_vals_core, x_vals_core)
z = np.column_stack((E_x_core_x_pol, E_y_core_y_pol))
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()

#%%

# plot heat map
# somehow
# idk



#%%

l = m + 1

A = 1
B = A *  jv(l, (p*a)) / (kv(l, q * a))

# Y-polarised

E_x_core_y_pol = 0 * r

E_y_core_y_pol = A * jv(l, (p * r))

E_z_core_y_pol = (p / beta) * (A / 2) * ((jv(l+1, (p * r)) * np.exp(1j * phi)) + (jv(l-1, (p * r)) * np.exp(-1j * phi)))


# cladding r > a


E_x_clad_y_pol = 0 * r

E_y_clad_y_pol = B * kv(l, (q * np.abs(r_clad)))

E_z_clad_y_pol = (q / beta) * (B/2) * ((kv(l+1, (q * r_clad)) * np.exp(1j * phi)) - (kv(l-1, (q * r_clad)) * np.exp(-1j * phi)))


plt.plot(r, E_y_core_y_pol, label = 'E_y')
plt.plot(r, E_z_core_y_pol, label = 'E_z')
plt.plot(r, E_x_core_y_pol, label = 'E_x')
plt.plot(r_clad, E_y_clad_y_pol, label = 'E_y')
plt.plot(r_clad, E_z_clad_y_pol, label = 'E_z')
plt.plot(r_clad, E_x_clad_y_pol, label = 'E_x')
plt.legend()
plt.show()

# X polarised

# core r < a

E_x_core_x_pol = A * jv(l, (p * r))

E_y_core_x_pol = B * r

E_z_core_x_pol = 1j * (p / beta) * (A/2) * ((jv(l+1, p * r) * np.exp(1j * phi)) - (jv(l-1, p * r) * np.exp(-1j * phi)))

# cladding r > a

E_x_clad_x_pol = A * kv(l, (p * np.abs(r_clad)))

E_y_clad_x_pol = 0 * r

E_z_clad_x_pol = 1j * (q/beta) * (B/2) * ((kv(l+1, q * r_clad) * np.exp(1j * phi)) + (kv(l-1, q * r_clad) * np.exp(-1j * phi)))

plt.plot(r, E_y_core_x_pol, label = 'E_y')
plt.plot(r, E_z_core_x_pol, label = 'E_z')
plt.plot(r, E_x_core_x_pol, label = 'E_x')
plt.plot(r_clad, E_y_clad_x_pol, label = 'E_y')
plt.plot(r_clad, E_z_clad_x_pol, label = 'E_z')
plt.plot(r_clad, E_x_clad_x_pol, label = 'E_x')
plt.legend()
plt.show()

amplitude_core_m_plus_1 = np.sqrt((E_y_core_y_pol ** 2) + (E_x_core_x_pol ** 2))


amplitude_clad_m_plus_1 = np.sqrt((E_y_clad_y_pol ** 2) + (E_x_clad_x_pol ** 2))


e_field_amp_m_plus_1 = np.hstack((amplitude_core_m_plus_1, amplitude_clad_m_plus_1))


#%%

l = m - 1

A = 1
B = A *  jv(l, (p*a)) / (kv(l, q * a))

# Y-polarised

E_x_core_y_pol = 0 * r

E_y_core_y_pol = A * jv(l, (p * r))

E_z_core_y_pol = (p / beta) * (A / 2) * ((jv(l+1, (p * r)) * np.exp(1j * phi)) + (jv(l-1, (p * r)) * np.exp(-1j * phi)))

# cladding r > a


E_x_clad_y_pol = 0 * r

E_y_clad_y_pol = B * kv(l, (q * np.abs(r_clad)))

E_z_clad_y_pol = (q / beta) * (B/2) * ((kv(l+1, (q * np.abs(r_clad))) * np.exp(1j * phi)) - (kv(l-1, (q * np.abs(r_clad))) * np.exp(-1j * phi)))


plt.plot(r, E_y_core_y_pol, label = 'E_y')
plt.plot(r, E_z_core_y_pol, label = 'E_z')
plt.plot(r_clad, E_y_clad_y_pol, label = 'E_y')
plt.plot(r_clad, E_z_clad_y_pol, label = 'E_z')
plt.legend()
plt.show()



# X - polarised

E_x_core_x_pol = A * jv(l, (p * r))

E_y_core_x_pol = r * 0

E_z_core_x_pol = 1j * (p / beta) * (A/2) * ((jv(l+1, p * r) * np.exp(1j * phi)) - (jv(l-1, p * r) * np.exp(-1j * phi)))

# cladding r > a

E_x_clad_x_pol = A * kv(l, (p * np.abs(r_clad)))

E_y_clad_x_pol = 0 * r

E_z_clad_x_pol = 1j * (q/beta) * (B/2) * ((kv(l+1, q * np.abs(r_clad)) * np.exp(1j * phi)) + (kv(l-1, q * np.abs(r_clad)) * np.exp(-1j * phi)))

plt.plot(r, E_y_core_x_pol, label = 'E_y')
plt.plot(r, E_z_core_x_pol, label = 'E_z')
plt.plot(r, E_x_core_x_pol, label = 'E_x')
plt.plot(r_clad, E_y_clad_x_pol, label = 'E_y')
plt.plot(r_clad, E_z_clad_x_pol, label = 'E_z')
plt.plot(r_clad, E_x_clad_x_pol, label = 'E_x')
plt.legend()
plt.show()

#%%


#plt.imshow(E_z_clad_x_pol, cmap='hot', interpolation='nearest')
#plt.show()


#fig,ax = plt.subplots()
#ax.scatter(E_z_core_x_pol.real,E_z_core_x_pol.imag, marker = '.')
#ax.scatter(E_z_clad_x_pol.real,E_z_core_x_pol.imag, marker = '.')
#plt.show()

# generate 2 2d grids for the x & y bounds
y,x = np.meshgrid(np.linspace(-3 * a, 3 * a, 100), np.linspace(-3 * a, 3 * a, 100))


z = (1 - x / 2. + x ** 5 + y ** 3) * np.exp(-x ** 2 - y ** 2)
# x and y are bounds, so z should be the value *inside* those bounds.
# Therefore, remove the last value from the z array.
z = z[:-1, :-1]
z_min, z_max = -np.abs(z).max(), np.abs(z).max()

fig, ax = plt.subplots()

c = ax.pcolormesh(x, y, z, cmap='RdBu', vmin=z_min, vmax=z_max)
ax.set_title('pcolormesh')
# set the limits of the plot to the limits of the data
ax.axis([x.min(), x.max(), y.min(), y.max()])
fig.colorbar(c, ax=ax)

plt.show()

#%%

# Task 6


# intensity is proportional to amlpitude squared
# question 6 gives intensity as ex^2 + e_y^2 
# so amplitude must be sqrt (ex^2 + ey^2)




amplitude_core_m_minus_1 = np.sqrt((E_y_core_y_pol ** 2) + (E_x_core_x_pol ** 2))

plt.plot(r, amplitude_core_m_minus_1)
plt.show()

amplitude_clad_m_minus_1 = np.sqrt((E_y_clad_y_pol ** 2) + (E_x_clad_x_pol ** 2))

plt.plot(r_clad, amplitude_clad_m_minus_1)
plt.show()

e_field_amp_m_minus_1 = np.hstack((amplitude_core_m_minus_1, amplitude_clad_m_minus_1))



#%%

r_outoforder = np.hstack((r, r_clad))

plt.plot(r_outoforder, e_field_amp_m_minus_1, linewidth = 0, marker = '.')
plt.show()


plt.plot(r_outoforder, e_field_amp_m_plus_1, linewidth = 0, marker = '.')
plt.show()


plt.plot(r_outoforder, e_field_amp_1, linewidth = 0, marker = '.')
plt.show()

#%%

# intensity distribution of the mode


plt.plot(r_outoforder, e_field_amp_1 **2, linewidth = 0, marker = '.')
plt.title('L = 2')
plt.grid()
plt.show()

#%%

plt.plot(r_outoforder, e_field_amp_m_minus_1 **2, linewidth = 0, marker = '.')
plt.title('L = 1')
plt.grid()
plt.show()

#%%

plt.plot(r_outoforder, e_field_amp_m_plus_1 **2, linewidth = 0, marker = '.')
plt.title('L = 3')
plt.grid()
plt.show()















