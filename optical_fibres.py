import pandas as pd
from scipy.optimize import fsolve
from scipy.constants import pi
from scipy.special import jv, kv
import numpy as np

# Given parameters
wavelength = 1450e-9  # Wavelength in meters
k0 = 2 * pi / wavelength
n1 = 1.51
n2 = 1.49
a = 5e-6  # Core radius in meters

data = {'m': [], 'beta_solution': [], 'mode_type': []}  # Dictionary to store data
tolerance = 1

for m in range(5):  # Iterate over different m values
    beta_solutions = set()
    def equation_to_solve(beta):
        p = np.sqrt((k0 * n1)**2 - beta**2)
        q = np.sqrt(beta**2 - (k0 * n2)**2)

        J_m_p = jv(m + 1, p * a)
        J_m = jv(m, p * a)
        K_m_q = kv(m + 1, q * a)
        K_m = kv(m, q * a)

        equation_left = J_m_p / (p * J_m)
        equation_right = - K_m_q / (q * K_m)

        return equation_left - equation_right

    for beta_guess in np.linspace(n2 * k0, n1 * k0, num=1000):  # Adjust the range as needed
        beta_solution = fsolve(equation_to_solve, beta_guess)
        if n2 * k0 < beta_solution < n1 * k0:
            beta_solution = round(beta_solution[0], -4)  # Round to a certain decimal precision
            if all(abs(sol - beta_solution) > tolerance for sol in beta_solutions):
                beta_solutions.add(beta_solution)
                data['m'].append(m)
                data['beta_solution'].append(beta_solution)
                data['mode_type'].append(f"m{m}")



# Create a DataFrame from the collected data
df = pd.DataFrame(data)
print(df)



#%%
m = 0  # Set m value to 0

beta_solutions = []  # Store beta solutions for m = 0
p_values = []  # Store p values for plotting
lhs_values = []  # Store LHS values for plotting
rhs_values = []  # Store RHS values for plotting
for beta_guess in np.linspace(n2 * k0, n1 * k0, num=1000):  # Adjust the range as needed
    beta_solution = fsolve(equation_to_solve, beta_guess)
    if n2 * k0 < beta_solution < n1 * k0:
        beta_solution = beta_solution[0]
        beta_solutions.append(beta_solution)
        p = np.sqrt((k0 * n1)**2 - beta_solution**2)
        q = np.sqrt(beta_solution**2 - (k0 * n2)**2)
        p_values.append(p)
        
        J_m_p = jv(m + 1, p * a)
        J_m = jv(m, p * a)
        K_m_q = kv(m + 1, q * a)
        K_m = kv(m, q * a)
        
        lhs = J_m_p / (p * J_m)
        rhs = - K_m_q / (q * K_m)
        
        lhs_values.append(lhs)
        rhs_values.append(rhs)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(p_values, lhs_values, label='LHS')
plt.plot(p_values, rhs_values, label='RHS')
plt.xlabel('p values')
plt.ylabel('Function values')
plt.title('Comparison of LHS and RHS for m=0')
plt.legend()
plt.grid(True)
plt.show()