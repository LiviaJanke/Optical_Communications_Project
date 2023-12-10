# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 11:38:51 2023

@author: Hidemaru Hashida
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:28:34 2023

@author: burto
"""

import pandas as pd
from scipy.optimize import fsolve
from scipy.constants import pi
from scipy.special import jv, kv, jvp, kvp
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import pi, epsilon_0, mu_0


#%%

"Defining constants/ given parameters"

pi= np.pi
a = 5e-6                                 # Core radius
n1 = 1.51                                # Core refractive index
n2 = 1.49                                # Cladding refractive index
wavelength = 1450e-9                     # Wavelength
k0 = (2 * pi)/ wavelength    
c = 3e8                                  # Wavevector



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

def get_solutions(mmin, mmax,HE=False):
    m_range = np.arange(mmin, mmax + 1, 1)
    
    index_vals = []
    for m in m_range:
        lhs_arr = get_lhs(m, pa_vals, HE)
        rhs_arr = get_rhs(m, qa_vals, HE)
        diff = np.abs(lhs_arr - rhs_arr)
        
        indices = np.where(diff <= 0.000001)[0]
        beta_vals = np.unique(np.round(beta_test_array_valid[indices]))
        print(f"For mode m = {m}, Beta equals: {beta_vals}")
        bets_vales = np.round(beta_vals)
        index_vals.append(beta_vals)
    
    return index_vals


#%%

get_solutions(0,4)


#%%

get_solutions(0,4,True)


#%%

pa_vals = p_vals * a

def pa_to_qa(pa):
    qa = np.sqrt((V)**2 - pa**2)
    return qa

qa_vals = pa_to_qa(pa_vals)


for l in np.arange(0,4,1):
    plt.figure()
    plt.title(f'Mode {l}')
    plt.plot(pa_vals,get_lhs(l,pa_vals),color='red')
    plt.plot(pa_vals,get_rhs(l,pa_to_qa(pa_vals)),color='blue')
    plt.ylim(-5,5)


#%%

"Task 5"


beta = 6530758
m = 1

eps0 = epsilon_0
mu0 = mu_0
z = 0
qa = np.sqrt(beta**2-n2**2*k0**2)*a 
pa = np.sqrt(V**2-qa**2)
p = pa/a
q = qa/a
omega = np.sqrt((k0 ** 2)/ (eps0 * mu0))

#set C = 1
C = 1

A = kv(m,(qa))/jv(m,(pa))   

#jvp(m,pa,1)

top = -A*((n1**2*(jvp(m,pa,1)/(pa*jv(m,pa))))+ (n2**2*(kvp(m,qa,1)/(qa*kv(m,qa)))))
bot = ((1j*beta*m)/(omega*eps0))*((1/pa**2)+(1/qa**2))
B = top/bot

D = B*jv(m,pa)/kv(m,qa)

def Er(r):
    if r<a:
        return (-1j*beta/p**2)*((A*p*jvp(m,p*r,1))+((1j*omega*mu0*m*B*jv(m,p*r))/(beta*r)))
    else:
        return (-1j*beta/q**2)*((A*q*kvp(m,q*r))+((1j*omega*mu0*m*B*kv(m,q*r))/(beta*r)))
        return (1j*beta/q**2)*((C*q*kvp(m,q*r,1))+((1j*omega*mu0*m*D*kv(m,q*r))/(beta*r)))
def Ephi(r):
    if r< a:
        return (-1j*beta/p**2)*((1j*m*A*jv(m,p*r)/r)-(omega*mu0*p*B*jvp(m,p*r,1)/beta))
    else:
        return (1j*beta/q**2)*((1j*m*C*kv(m,q*r)/r)-(omega*mu0*q*D*kvp(m,q*r,1)/beta))
    
def Ez(r):
    if r< a:
        return A*jv(m,p*r)
    else:
        return C*kv(m,q*r)

def exp_func(phi):
    return np.exp((1j*m*phi)-(1j*beta*z))


#%%

r_vals = np.linspace(0.01e-6, 2*a, 100) 
phi_vals = np.linspace(0, 2 * np.pi, 100)

r, phi = np.meshgrid(r_vals, phi_vals)
Er_vals = np.array([[Er(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])
Ephi_vals = np.array([[Ephi(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])
Ez_vals = np.array([[Ez(r[i][j])*exp_func(phi[i][j]) for j in range(len(r[i]))] for i in range(len(r))])
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

"Task 6"

n_forI = 6530758 / k0

# Intensity Distribution
I =  ((c * epsilon_0)/2) * (n_forI **2) * ((Er_vals * np.conjugate(Er_vals)) + (Ephi_vals * np.conjugate(Ephi_vals)))
fig, ax = plt.subplots( figsize=(15, 5), subplot_kw=dict(polar=True))

# Plot Intensity heatmap
mesh1 = ax.pcolormesh(phi, r, np.abs(I), cmap=custom_cmap)
ax.set_title('Electric Field Intensity')
fig.colorbar(mesh1, ax=ax, orientation='vertical', label='Intensity')

plt.tight_layout()
plt.show()


#%%

"Task 7"

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


#%%

"Task 8"

# Calculating 'b' and n_eff from the approximate formulae
b = (1.1428 - 0.996/V)**2
n_eff = n2 + b*(n1-n2)

#comparing to value from Step 3:
# using the beta value for m=1:
    
beta_LP01 = 6488084

eff_ind_LP01 = beta_LP01 / k0


print('effective index from formula:')
print(n_eff)

print('effective index from Step 3:')
print(eff_ind_LP01)

print('Difference:')
print(n_eff - eff_ind_LP01)

# Values are the same to 3sf



#%%

# Calculating approximate core power fraction
#gamma = (A - B/V)**2
#power_frac = abs(gamma)
#print("The fraction of power in the core is", power_frac)


# Calculating approximate core power fraction from formula

#w_p = 0.634 + ((1.619 * V)**(-3/2)) + ((2.879 * V)**(-6)) + ((1.561 * V) ** (-7))

#Gamma_V = (2 / ((V**2) * (w_p**2))) + b

#print("The fraction of power in the core is", Gamma_V)

# third version of the equation

gamma_v2 = (eff_ind_LP01 - n2) / (n1 - n2)

print("The fraction of power in the core is", gamma_v2)































