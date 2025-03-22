'''
TO DO:
1. How to calculate power coefficient
2. Figure out yawed case
3. How to do energy harvesting and optimisation
4. Finish Plotting Routines
'''

import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

#------------------------- FUNCTION DEFINITIONS -------------------------
def PrandtlCorrections(Nb, r, R, TSR, a, root_pos_R, dCT, dCQ):
    F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((1-(r/R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1+a)**2))))))
    F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1+a)**2)))))  
    F_tot = F_tip*F_root
    
    if(F_tot == 0) or (F_tot is np.nan) or (F_tot is np.inf):
        # handle exceptional cases for 0/NaN/inf value of F_tot
        # print("F total is 0/NaN/inf.")
        F_tot = 0.00001

    a_new = (1/2)*(1-np.sqrt(1-(dCT/F_tot)))    # axial induction factor, a
    b_new = (dCQ)/(4*F_tot*(1+a)*(TSR*(r/R)))   # tangential induction factor, a'

    return a_new, b_new, F_tot, F_tip, F_root

def BladeElementMethod(Vinf, TSR, n, rho, R, r, root_pos_R, dr, Omega, Nb, a, b, twist, chord, polar_alfa, polar_cl, polar_cd, tol):
    while True:
            V_ax = Vinf*(1+a)       # axial velocity at the propeller blade
            V_tan = Omega*r*(1-b)   # tangential veloity at the propeller blade
            V_loc = np.sqrt(V_ax**2 + V_tan**2)

            phi = np.arctan(V_ax/V_tan)     # inflow angle [rad]
            alfa = twist - np.rad2deg(phi)  # local angle of attack [deg]

            Cl = np.interp(alfa, polar_alfa, polar_cl)
            Cd = np.interp(alfa, polar_alfa, polar_cd)
            
            C_ax = Cl*np.cos(phi) - Cd*np.sin(phi)      # axial force coefficient  
            C_tan = Cl*np.sin(phi) + Cd*np.cos(phi)     # tangential force coefficient
            # gamma = 0.5*V_loc*Cl[j][i]*chord

            dCT = ((0.5*rho*V_loc**2)*chord*C_ax*Nb*dr)/(rho*(n**2)*(2*R)**4)       # blade element thrust coefficient
            dCQ = ((0.5*rho*V_loc**2)*chord*C_tan*Nb*r*dr)/(rho*(n**2)*(2*R)**5)    # blade element torque coefficient

            a_new, b_new, F_tot, F_tip, F_root = PrandtlCorrections(Nb, r, R, TSR, a, root_pos_R, dCT, dCQ)

            if(np.abs(a-a_new)<tol) and (np.abs(b-b_new)<tol):
                a = a_new
                b = b_new
                break
            else:
                # introduces relaxation to induction factors a and a' for easier convergence
                a = 0.75*a + 0.25*a_new
                b = 0.75*b + 0.25*b_new
                continue
    return a, b, Cl, Cd, alfa, phi, F_tot, F_tip, F_root, dCT, dCQ

#--------------------------------- MAIN ---------------------------------

# Read polar data
airfoil = 'ARAD8pct_polar.csv'
data1=pd.read_csv(airfoil, header=0, names = ["alfa", "cl", "cd", "cm"],  sep=',')
polar_alfa = data1['alfa'][:]
polar_cl = data1['cl'][:]
polar_cd = data1['cd'][:]

# Flow conditions
Vinf = 60                       # freestream velocity [m/s]
J = np.array([1.6, 2.0, 2.4])   # advance ratio
rho = 1.007                     # density at h=2000m [kg/m^3]

# Blade geometry
R = 0.7                             # Blade radius [m]
Nb = 6                              # number of blades
tip_pos_R = 1                       # normalised blade tip position (r_tip/R)
root_pos_R = 0.25                   # normalised blade root position (r_root/R)
pitch = 46                          # blade pitch [deg]

# Discretisation into blade elements
nodes = 100
r_R = np.linspace(root_pos_R, tip_pos_R, nodes)
chord_dist = 0.18 - 0.06*(r_R)                  # chord distribution [m]
twist_dist = -50*(r_R) + 35 + pitch             # twist distribution [deg]

# Dependent variables 
n = Vinf/(2*J*R)    # RPS [Hz]
Omega = 2*np.pi*n   # Angular velocity [rad/s]
TSR = np.pi/J       # tip speed ratio

# Iteration inputs
tol = 1e-6  # convergence tolerance

# Variable initialisation
CT, CQ = [np.zeros(len(J)) for i in range(2)]
a = np.ones((len(J),len(r_R)-1))*(1/3)
b, Cl, Cd, alfa, phi, F_tot, F_tip, F_root = [np.zeros((len(J),len(r_R)-1)) for i in range(8)]

# Solving BEM model
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist)
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(R/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*R        # length of the blade element
        
        a[j][i], b[j][i], Cl[j][i], Cd[j][i], alfa[j][i], phi[j][i], F_tot[j][i], F_tip[j][i], F_root[j][i], dCT, dCQ = BladeElementMethod(Vinf, TSR[j], n[j], rho, R, r, root_pos_R, dr, Omega[j], Nb, a[j][i], b[j][i], twist, chord, polar_alfa, polar_cl, polar_cd, tol)

        CT[j] += dCT    # thrust coefficient for given J
        CQ[j] += dCQ    # torque coefficient for given J

#------------------------------- RESULTS --------------------------------

print("CT: ", CT)
print("CQ: ", CQ)

# Plotting Routines

plt.figure(1)
plt.plot(r_R[1:],a[0][:], label="a")
plt.plot(r_R[1:],b[0][:], label="a'")
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