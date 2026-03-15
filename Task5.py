# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 12:00:10 2023

@author: eichm
"""

# Task 5 with E_r, E_phi, E_z

import pandas as pd
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.constants import pi, epsilon_0, mu_0
from scipy.special import jv, kv, jvp
from cmath import pi, e, polar
from numpy import linspace, vectorize, sin, cos

#%%

"Defining constants/ given parameters"

pi= np.pi
a = 5e-6                                 # Core radius
n1 = 1.51                                # Core refractive index
n2 = 1.49                                # Cladding refractive index
wavelength = 1450e-9                     # Wavelength
k0 = (2 * pi)/ wavelength                   # Wavevector
omega = np.sqrt((k0 **2)/ (epsilon_0 * mu_0))


#%%

"Task 1"

V = a * k0 * np.sqrt(n1**2 - n2**2)


"Task 2"

# We will use the LP modes as we are working with the assumption
# that n1=~n2


#%%

"Task 3"

# Create array of beta values with range based on physical limits

beta_min = n2*k0
beta_max = n1*k0
beta_test_array = np.linspace(beta_min, beta_max, 10000000)
beta_test_array_valid = beta_test_array[1:-1]

# Create arrays of p and q from beta for use in Bessell functions

p_vals = np.sqrt(((n1**2) * (k0 ** 2)) - (beta_test_array_valid**2))

q_vals = np.sqrt((beta_test_array_valid**2) - ((n2**2) * (k0 ** 2)))

pa_vals = p_vals * a

qa_vals = q_vals * a

# Define function for left and right hand side

def get_lhs(m,pa,HE=False):
    if HE == False:
        lhs = jv(m+1, (pa)) / (pa * jv(m, (pa))) 
    elif HE == True:
        lhs = jv(m-1, (pa)) / (pa * jv(m, (pa)))          
    return lhs

def get_rhs(m,qa,HE=False):
    if HE == False:
        rhs = -(kv(m+1, (qa))) / (qa * kv(m, (qa)))
    elif HE == True:
        rhs = (kv(m-1, (qa))) / (qa * kv(m, (qa)))
    return rhs

# Function for finding solutions for all l values between a min
# and max l value specified by "l_max" and "l_min"

def get_solutions(mmin,mmax,HE=False):
    
    index_vals = []
    
    m_range = np.arange(mmin,mmax+1,1)
    
    for m in m_range:
    
        lhs_arr = get_lhs(m,pa_vals,HE)
        rhs_arr = get_rhs(m,qa_vals,HE)
        diff = abs(lhs_arr-rhs_arr)
    
        for i in np.arange(0, len(beta_test_array_valid)):

            if diff[i] <= 0.000001:     
                index_vals.append(i)  
                print("For mode m=",m,"Beta equals",beta_test_array_valid[i])
    
    return index_vals


# Ignore the duplicates! They are there because two values close
# to the solution are less than 0.00003

EH_indexes = get_solutions(0,4)



HE_indexes = get_solutions(0,4,True)


#%%

pa_vals = p_vals * a

def pa_to_qa(pa):
    qa = np.sqrt((V)**2 - pa**2)
    return qa

qa_vals = pa_to_qa(pa_vals)


for m in np.arange(0,4,1):
    plt.figure()
    plt.title(f'Mode {m}')
    plt.plot(pa_vals,get_lhs(m,pa_vals),color='red')
    plt.plot(pa_vals,get_rhs(m,pa_to_qa(pa_vals)),color='blue')
    plt.ylim(-5,5)


#%%

"Task 5"

# Mode selection: m= 1 Beta equals 6488083.926444083,
# EH_indexes[7] (LP_21)

m = 1
l = m+1                                    # l=m+1 for EH modes
p = p_vals[EH_indexes[7]]
q = q_vals[EH_indexes[7]]
beta = beta_test_array_valid[EH_indexes[7]]
A = 1
B = (A*jv(l,p*a))/(kv(l,q*a))




# Cartesian to polar conversion functions

def get_r(x,y):
    r = np.sqrt(x**2+y**2)
    return r

def get_phi(x,y):
    phi = np.arctan(y/x)
    return phi




r = np.linspace(-a, a, 1000)
r_clad_pos = np.linspace(a, 3 * a, 500)
r_clad_neg = np.linspace(-3 * a, - a, 500)
r_clad = np.hstack((r_clad_neg, r_clad_pos))
r_tot = np.hstack((r_clad_neg,r, r_clad_pos))

m = 2
beta = 6488083.926444083

phi = np.linspace(0, (2 * np.pi), 1000)

#x_vals_core = r * np.cos(phi)
#y_vals_core = r * np.sin(phi)

#x_vals_clad = r_clad * np.cos(phi)
#y_vals_clad = r_clad * np.sin(phi)

pos_x_phi = np.hstack((phi[750:], phi[:250]))
neg_x_phi = phi[250:750]

phi_neg_to_pos = np.hstack((neg_x_phi, pos_x_phi))

x_vals_core = r * np.cos(phi)
x_vals_clad = r_clad * np.cos(phi)

y_vals_core = r * np.sin(phi_neg_to_pos)
y_vals_clad = r_clad * np.sin(phi)

#%%


r = np.linspace(-a, a, 1000)

def Ez_vals_c(r):
    Ez = A * jv(2, (p * r))
    return Ez

def Er_vals_c(r):
    Er = - ((1j * beta) / (p **2))  *  ((A * p * jvp(m, (p * r), n = 1)) + (1j * omega * (mu_0 * m / beta * r) * B * jv(m, (p * r))) )
    return Er

def Ephi_vals_c(r):
    Ephi = - ((1j * beta) / (p **2)) * (((1j * m / r) * A * jv(m, (p * r))) - ((omega * mu_0 / beta) * p * B * jvp(m, (p * r), n=1)))
    return Ephi


def Ez_vals(x, y):
    Ez = A * jv(2, (p * get_r(x,y)))
    return Ez

def Er_vals(x, y):
    Er = - ((1j * beta) / (p **2))  *  ((A * p * jvp(m, (p * get_r(x,y)), n = 1)) + (1j * omega * (mu_0 * m / beta * get_r(x,y)) * B * jv(m, (p * get_r(x,y)))) )
    return Er

def Ephi_vals(x, y):
    Ephi = - ((1j * beta) / (p **2)) * (((1j * m / get_r(x,y)) * A * jv(m, (p * get_r(x,y)))) - ((omega * mu_0 / beta) * p * B * jvp(m, (p * get_r(x,y)), n=1)))
    return Ephi

#%%

# Plotting mode bitmaps

#x = np.linspace(-5e-6,5e-6,1000)
#y = np.linspace(-5e-6,5e-6,1000)


x = x_vals_core
y = y_vals_core

# Creating bitmap

xx, yy = np.meshgrid(x,y)
EE = Ez_vals(xx,yy)

# Plotting

fig, ax = plt.subplots()

levels = np.linspace(EE.min(), EE.max(), 100)
#c = ax.pcolormesh(x, y, EE, cmap='RdBu')
c = plt.contourf(x, y, EE, levels=levels, cmap='viridis')

plt.title('E_z LP_21 mode')
plt.xlabel('Horizontal Position / m')
plt.ylabel('Vertical Position / m')
ax.axis([x.min(), x.max(), y.min(), y.max()])                                                  # Set limits of plot to data limits
fig.colorbar(c, ax=ax,label='E-field amplitude / V/m')

#%%


x = x_vals_core
y = y_vals_core

# Creating bitmap

xx, yy = np.meshgrid(x,y)
EE = Er_vals(xx,yy)

# Plotting

fig, ax = plt.subplots()

levels = np.linspace(EE.min(), EE.max(), 100)
#c = ax.pcolormesh(x, y, EE, cmap='RdBu')
c = plt.contourf(x, y, EE, levels=levels, cmap='viridis')

plt.title('E_r LP_21 mode')
plt.xlabel('Horizontal Position / m')
plt.ylabel('Vertical Position / m')
ax.axis([x.min(), x.max(), y.min(), y.max()])                                                  # Set limits of plot to data limits
fig.colorbar(c, ax=ax,label='E-field amplitude / V/m')

#%%

x = x_vals_core
y = y_vals_core

# Creating bitmap

xx, yy = np.meshgrid(x,y)
EE = Ephi_vals(xx,yy)

# Plotting

fig, ax = plt.subplots()

levels = np.linspace(EE.min(), EE.max(), 100)
#c = ax.pcolormesh(x, y, EE, cmap='RdBu')
c = plt.contourf(x, y, EE, levels=levels, cmap='viridis')

plt.title('E_phi LP_21 mode')
plt.xlabel('Horizontal Position / m')
plt.ylabel('Vertical Position / m')
ax.axis([x.min(), x.max(), y.min(), y.max()])                                                  # Set limits of plot to data limits
fig.colorbar(c, ax=ax,label='E-field amplitude / V/m')



#%%

def exp_func(phi):
    return np.exp((1j*m*phi)-(1j*beta*z))

z = 0
r_vals = r
phi_vals = phi

r, phi = np.meshgrid(r_vals, phi_vals)
Er_vals = np.array([[Er_vals_c(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])
Ephi_vals = np.array([[Ephi_vals_c(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])
Ez_vals = np.array([[Ez_vals_c(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])
# Set custom color maps
custom_cmap = 'plasma' 

# Plot polar heatmaps for Er, Ephi, and Ez with colorbars
fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))

# Plot Er heatmap
mesh1 = axs[0].pcolormesh(phi, r, np.real(Er_vals), cmap=custom_cmap)
axs[0].set_title('$E_r(r)$')
fig.colorbar(mesh1, ax=axs[0], orientation='vertical', label='Strength')

# Plot Ephi heatmap
mesh2 = axs[1].pcolormesh(phi, r, np.real(Ephi_vals), cmap=custom_cmap)
axs[1].set_title('$E_{\phi}(r)$')
fig.colorbar(mesh2, ax=axs[1], orientation='vertical', label='Strength')


# Plot Ez heatmap
mesh3 = axs[2].pcolormesh(phi, r, np.real(Ez_vals), cmap=custom_cmap)
axs[2].set_title('$E_z(r)$')
fig.colorbar(mesh3, ax=axs[2], orientation='vertical', label='Strength')

plt.tight_layout()
plt.show()



#%%



r, phi = np.meshgrid(r_vals, phi_vals)
Er_vals = np.array([[Er_vals_c(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])
Ephi_vals = np.array([[Ephi_vals_c(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])

intensity_vals = (Er_vals **2) + (Ephi_vals * 2)

# Set custom color maps
custom_cmap = 'plasma' 

# Plot polar heatmaps for Er, Ephi, and Ez with colorbars
fig, axs = plt.subplots(1, 1, figsize=(15, 5), subplot_kw=dict(polar=True))

# Plot Er heatmap
mesh1 = axs[0].pcolormesh(phi, r, intensity_vals, cmap=custom_cmap)
axs[0].set_title('$E_r(r)$')
fig.colorbar(mesh1, ax=axs[0], orientation='vertical', label='Strength')



plt.tight_layout()
plt.show()





































