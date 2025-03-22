import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

#------------------------- FUNCTION DEFINITIONS -------------------------

def BladeElement():
    return

#------------------------- MAIN -------------------------

# Read polar data
airfoil = 'ARAD8pct_polar.csv'
data1=pd.read_csv(airfoil, header=0, names = ["alfa", "cl", "cd", "cm"],  sep=',')
polar_alfa = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]

# Given conditions
Vinf = 60           # [m/s]
J = np.array([1.6])
R = 0.7
rho = 1.007         # density at h=2000m [kg/m^3]
Nb = 6
tip_pos_R = 1
root_pos_R = 0.25

# Discretisation of blade geometry
nodes = 75
r_R = np.linspace(root_pos_R, tip_pos_R, nodes)

# Blade geometry
pitch = 46                          # [deg]
chord_dist = 0.18 - 0.06*(r_R)      # [m]
twist_dist = -50*(r_R) + 35 + pitch # [deg]

n = Vinf/(2*J*R)
Omega = 2*np.pi*n
TSR = np.pi/J

# Solving the BEM model
results =np.zeros([len(r_R)-1,6]) 

# Iteration inputs
tol = 1e-6  # convergence tolerance
n_iter = 2000
iter = 0
T = Q = np.zeros(len(J))
a = np.ones((len(J),len(r_R)-1))*(1/3)
a_b4_Pr = np.ones((len(J),len(r_R)-1))*(1/3)
b = np.zeros((len(J),len(r_R)-1))
Cl = np.zeros((len(J),len(r_R)-1))
alfa = np.zeros((len(J),len(r_R)-1))
phi = np.zeros((len(J),len(r_R)-1))
F_tot = np.zeros((len(J),len(r_R)-1))
F_tip = np.zeros((len(J),len(r_R)-1))
F_root = np.zeros((len(J),len(r_R)-1))

for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist)
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(R/2)
        dr = (r_R[i+1]-r_R[i])*R
        
        while True and (iter < n_iter):
            V_ax = Vinf*(1+a[j][i])
            V_tan = Omega[j]*r*(1-b[j][i])
            V_loc = np.sqrt(V_ax**2 + V_tan**2)

            phi[j][i] = np.arctan(V_ax/V_tan)
            alfa[j][i] = -(np.rad2deg(phi[j][i]) - twist)

            Cl[j][i] = np.interp(alfa[j][i], polar_alfa, polar_cl)
            Cd = np.interp(alfa[j][i], polar_alfa, polar_cd)
            
            sigma = (Nb*chord)/(2*np.pi*r)
            
            C_ax = Cl[j][i]*np.cos(phi[j][i])-Cd*np.sin(phi[j][i])
            C_tan = Cl[j][i]*np.sin(phi[j][i])+Cd*np.cos(phi[j][i])   
            gamma = 0.5*V_loc*Cl[j][i]*chord

            
            dT = (0.5*rho*V_loc**2)*chord*C_ax*Nb*dr
            dQ = (0.5*rho*V_loc**2)*chord*C_tan*Nb*r*dr

            a_new = np.max(np.roots([1,1,(-dT/(4*np.pi*r*rho*dr*Vinf**2))]))
            b_new = (dQ)/((4*np.pi*r**3)*Vinf*(1+a[j][i])*Omega[j]*dr)
            
            a_b4_Pr[j][i] = a_new

            # Prandtl Root and Tip Corrections
            # F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((1-(r/R))/(r/R))*(1/np.sin(phi[j][i])))))
            F_tip[j][i] = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((1-(r/R))/(r/R))*(np.sqrt(1 + ((TSR[j]*(r/R))**2)/((1+a_new)**2))))))
              
            # F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(1/np.sin(phi[j][i]))))  
            F_root[j][i] = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR[j]*(r/R))**2)/((1+a_new)**2)))))  
            
            F_tot[j][i] = F_tip[j][i]*F_root[j][i]

            if(F_tot[j][i] == 0) or (F_tot[j][i] is np.nan) or (F_tot[j][i] is np.inf):
                F_tot[j][i] = 0.00001

            a_new /= (F_tot[j][i])
            b_new /= (F_tot[j][i])

            if(np.abs(a[j][i]-a_new)<tol) and (np.abs(b[j][i]-b_new)<tol):
                T[j] += dT
                Q[j] += dQ
                a[j][i] = a_new
                b[j][i] = b_new
                break
            else:
                a[j][i] = 0.75*a[j][i] + 0.25*a_new
                b[j][i] = 0.75*b[j][i] + 0.25*b_new
                continue
        
        # results[i,:] = BladeElement()

CT = T/(rho*(n**2)*(2*R)**4)
print("CT: ", CT)

CQ = Q/(rho*(n**2)*(2*R)**5)
print("CQ: ", CQ)

plt.figure(1)
plt.plot(r_R[1:],a[0][:], label="a_corr")
plt.plot(r_R[1:],a_b4_Pr[0][:], label="a_uncorr")
plt.legend()

plt.figure(2)
plt.plot(r_R[1:],alfa[0][:], label="alfa")
plt.plot(r_R[1:],np.rad2deg(phi[0][:]), label="phi")
plt.legend()


plt.figure(3)
plt.plot(r_R[1:],F_tip[0][:], label="F_tip")
plt.plot(r_R[1:],F_root[0][:], label="F_root")
plt.plot(r_R[1:],F_tot[0][:], label="F_tot")
plt.legend()


plt.show()