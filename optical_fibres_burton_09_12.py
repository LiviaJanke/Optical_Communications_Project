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


#%%

# Ignore the duplicates! They are there because two values close
# to the solution are less than 0.00003

EH_indexes = get_solutions(0,4)


#%%

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


# y-polarised modes

def Ex_ypol(x,y):
    Ex = 0
    return Ex

def Ey_ypol(x,y):
    Ey = A * jv(l,p*get_r(x,y))
    return Ey

def Ez_ypol(x,y):
    Ez = ((p/beta)*(A/2)*
          ((jv(l+1,p*get_r(x,y))*np.exp(1j*get_phi(x,y)))+
           jv(l-1,p*get_r(x,y))*np.exp(-1j*get_phi(x,y))))
    return Ez


# x-polarised modes

def Ex_xpol(x,y):
    Ex = A * jv(l,p*get_r(x,y))
    return Ex

def Ey_xpol(x,y):
    Ey = 0 * get_r(x,y)
    return Ey

def Ez_xpol(x,y):
    Ez = (1j*(p/beta)*(A/2)*
          ((jv(l+1,p*get_r(x,y))*np.exp(1j*get_phi(x,y)))-
           jv(l-1,p*get_r(x,y))*np.exp(-1j*get_phi(x,y))))
    return Ez


# Plotting mode bitmaps

x = np.linspace(-5e-6,5e-6,1000)
y = np.linspace(-5e-6,5e-6,1000)


# Creating bitmap

xx, yy = np.meshgrid(x,y)
EE = Ez_ypol(xx,yy)

# Plotting

fig, ax = plt.subplots()

levels = np.linspace(EE.min(), EE.max(), 100)
#c = ax.pcolormesh(x, y, EE, cmap='RdBu')
c = plt.contourf(x, y, EE, levels=levels, cmap='viridis')

plt.title('E_z y-polarised LP_21 mode')
plt.xlabel('Horizontal Position / m')
plt.ylabel('Vertical Position / m')
ax.axis([x.min(), x.max(), y.min(), y.max()])                                                  # Set limits of plot to data limits
fig.colorbar(c, ax=ax,label='E-field amplitude / V/m')


#%%



#%%

"""    
def Ex_xpol(x,y):
    Ex_arr = []
    for i in range(len(x)):
        if np.any(get_r(x[i],y[i])) < a:
            Ex_arr.append(A * jv(l,p*get_r(x[i],y[i])))
        elif np.any(get_r(x[i],y[i])) >= a:
            Ex_arr.append(B * kv(l,q*get_r(x[i],y[i])))
    return Ex_arr

def Ez_xpol(x,y):
    if get_r(x,y) < a:
        Ez = (1j*(p/beta)*(A/2)*
              ((jv(l+1,p*get_r(x,y))*np.exp(1j*get_phi(x,y)))-
               jv(l-1,p*get_r(x,y))*np.exp(-1j*get_phi(x,y))))
    elif get_r(x,y) >= a:
        Ez = (1j*(q/beta)*(B/2)*
              ((kv(l+1,q*get_r(x,y))*np.exp(1j*get_phi(x,y)))+
               kv(l-1,q*get_r(x,y))*np.exp(-1j*get_phi(x,y))))
    return Ez
"""

"""
def Ey_ypol(x,y):
    if get_r(x,y) < a:
        Ey = A * jv(l,p*get_r(x,y))
    elif get_r(x,y) >= a:
        Ey = B * kv(l,q*get_r(x,y))
    return Ey

def Ez_ypol(x,y):
    if get_r(x,y) < a:
        Ez = ((p/beta)*(A/2)*
              ((jv(l+1,p*get_r(x,y))*np.exp(1j*get_phi(x,y)))+
               jv(l-1,p*get_r(x,y))*np.exp(-1j*get_phi(x,y))))
    elif get_r(x,y) >= a:
        Ez = ((q/beta)*(B/2)*
              ((kv(l+1,q*get_r(x,y))*np.exp(1j*get_phi(x,y)))-
               kv(l-1,q*get_r(x,y))*np.exp(-1j*get_phi(x,y))))
    return Ez
"""