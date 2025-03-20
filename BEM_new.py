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
T = Q = 0
a = np.ones((len(r_R)-1))*(1/3)
b = np.zeros((len(r_R)-1))

for j in range(len(J)):
    for i in range(len(r_R)-1):
        # a = 1/3     # axial induction factor
        # b = 0       # tangential induction factor
        
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist)
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(R/2)
        dr = (r_R[i+1]-r_R[i])*R
        
        while True:
            V_ax = Vinf*(1+a[i])
            V_tan = Omega[j]*r*(1-b[i])
            V_loc = np.sqrt(V_ax**2 + V_tan**2)

            phi = np.arctan(V_ax/V_tan)
            alfa = twist - phi

            Cl = np.interp(alfa, polar_alfa, polar_cl)
            Cd = np.interp(alfa, polar_alfa, polar_cd)
            C_ax = Cl*np.cos(phi)-Cd*np.sin(phi)
            C_tan = Cl*np.sin(phi)+Cd*np.cos(phi)   
            # gamma = 0.5*V_loc*Cl*chord

            dT = (0.5*rho*V_loc**2)*chord*C_ax*Nb*dr
            dQ = (0.5*rho*V_loc**2)*chord*C_tan*Nb*r*dr

            a_new = np.max(np.roots([1,1,(-dT/(4*np.pi*r*rho*dr*Vinf**2))]))
            b_new = (dQ)/((4*np.pi*r**3)*Vinf*(1+a[i])*Omega[j]*dr)
            
            if(np.abs(a[i]-a_new)<tol) or (np.abs(b[i]-b_new)<tol):
                T += dT
                Q += dQ
                a[i] = a_new
                b[i] = b_new
                break
            else:
                a[i] = 0.75*a[i] + 0.25*a_new
                b[i] = 0.75*b[i] + 0.25*b_new
                continue
        
        # results[i,:] = BladeElement()

CT = T/(rho*(n**2)*(2*R)**4)
print("CT: ", CT)

CQ = Q/(rho*(n**2)*(2*R)**5)
print("CQ: ", CQ)


plt.figure()
plt.plot(r_R[1:],a, label="a")
plt.plot(r_R[1:],b, label="a'")
plt.legend()
plt.show()