import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%---------------Initial conditions----------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

### import polar data
airfoil = 'ARAD8pct_polar.csv'
data1=pd.read_csv(airfoil, header=0, names = ["alfa", "cl", "cd", "cm"],  sep=',')
polar_alpha = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]

### Discretisation of blade geometry
delta_r_R = 0.01
r_R = np.arange(0.25, 1, delta_r_R)

### Blade geometry
pitch = 46                                  # degrees
chord_distribution = 0.18 - 0.06*(r_R)      # m
twist_distribution = -50*(r_R) + 35 + pitch # degrees

### Flow conditions
Uinf = 60
J = 1.6
Radius = 0.7
rho = 1.007
n = Uinf / (2*J*Radius)
Omega = 2*np.pi*n
TSR = np.pi/J
NBlades = 6

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%----------------Initialisations------------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
a_ax = np.zeros(len(r_R))
a_tan = np.zeros(len(r_R))

f_ax = np.zeros(len(r_R))
f_tan = np.zeros(len(r_R))
gamma = np.zeros(len(r_R))
CT = np.zeros(len(r_R))
CQ = np.zeros(len(r_R))
Prandtl_tip = np.zeros(len(r_R))
Prandtl_root = np.zeros(len(r_R))
Prandtl = np.zeros(len(r_R))
alpha = np.zeros(len(r_R))
phi = np.zeros(len(r_R))
U_ax_rotor = np.zeros(len(r_R))
U_tan_rotor = np.zeros(len(r_R))
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%-------------------MAIN CODE---------------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i in range(len(r_R)-1):
    chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
    twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
    r1_R, r2_R = r_R[i], r_R[i+1]
    r_R_mid = (r1_R + r2_R)/2
    a_ax[:] = 0.6
    a_tan[:] = 0.6
    a_ax_new = 0.0
    a_tan_new = 0.0
    dr = (r2_R - r1_R)*Radius
    error = 1e-7
    r = r_R_mid * Radius
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2)
    # u_a =Uinf*(1+a_ax[i])
    while ((np.abs(a_ax[i] - a_ax_new) > error) and (np.abs(a_tan[i] - a_tan_new) > error)):
        U_ax_rotor[i] = Uinf*(1+a_ax[i])
        U_tan_rotor[i] = (1 - a_tan[i])*Omega*r
        Umag2 = U_ax_rotor[i]**2 + U_tan_rotor[i]**2
        phi[i] = np.arctan(U_ax_rotor[i] / U_tan_rotor[i])
        alpha[i] =  twist - phi[i]*180/np.pi  
        Cl = np.interp(alpha[i], polar_alpha, polar_cl)
        Cd = np.interp(alpha[i], polar_alpha, polar_cd)
        lift = 0.5 * rho * Umag2 * Cl * chord
        drag = 0.5 * rho * Umag2 * Cd * chord
        f_ax[i] = lift * np.cos(phi[i]) - drag * np.sin(phi[i])
        f_tan[i] = lift * np.sin(phi[i]) + drag * np.cos(phi[i])
        gamma[i] = 0.5 * np.sqrt(Umag2) * Cl * chord
        CT[i] = f_ax[i] * NBlades / ( rho * (Uinf**2)*( np.pi * r ))
        # CT[i] = Cl[i]*np.cos(phi[i]) - Cd[i]*np.sin(phi[i])
        a_ax_new = 0.5 * (1 - np.sqrt(1 - CT[i]))
        root_exp = (-NBlades/2) * ((1 - r_R_mid)/r_R_mid)*np.sqrt(1 + ((TSR**2)*(r_R_mid**2)/((1 + a_ax_new)**2)))
        Prandtl_root[i] = (2/np.pi) * np.arccos(np.exp(root_exp))
        tip_exp = (-NBlades/2) * ((r_R_mid-0.25)/r_R_mid)*np.sqrt(1 + ((TSR**2)*(r_R_mid**2)/((1 + a_ax_new)**2)))
        Prandtl_tip[i] = (2/np.pi) * np.arccos(np.exp(root_exp))
        Prandtl[i] = Prandtl_root[i] * Prandtl_tip[i]
        if (Prandtl[i] < 0.0001):
            Prandtl[i]=0.0001

        a_ax_new = a_ax_new/Prandtl[i]
        a_ax[i] = a_ax[i]*0.75 + a_ax_new*0.25

        a_tan[i] = f_tan[i] * NBlades / (2*rho*(2*np.pi * (r**2))*(Uinf)*(1 + a_ax_new) * TSR)   # here +a is taken
        a_tan_new = a_tan[i]/Prandtl[i]
        a_tan[i] = a_tan[i]*0.75 + a_tan_new*0.25

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%---------------------PLOTS-----------------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Polar Plots
# fig1, axs = plt.subplots(1,2, figsize=(12,6))
# axs[0].plot(polar_alpha, polar_cl)
# axs[0].set_xlim([-30,30])
# axs[0].set_xlabel(r'$\alpha$')
# axs[0].set_ylabel(r'$C_l$')
# axs[0].grid()
# axs[1].plot(polar_cd, polar_cl)
# axs[1].set_xlim([0,.1])
# axs[1].set_xlabel(r'$C_d$')
# axs[1].grid()

# # Prandtl Corrections
# fig1 = plt.figure(figsize=(12, 6))
# plt.plot(r_R, Prandtl, 'r-', label='Prandtl')
# plt.plot(r_R, Prandtl_tip, 'g.', label='Prandtl tip')
# plt.plot(r_R, Prandtl_root, 'b.', label='Prandtl root')
# plt.xlabel('r/R')
# plt.legend()
print(a_ax[i])

# fig1 = plt.figure(figsize=(12, 6))
# plt.plot(r_R, phi*180/np.pi, 'r-', label='phi')
# plt.plot(r_R, alpha, 'b-', label='alpha')
# plt.legend()
# plt.show()

