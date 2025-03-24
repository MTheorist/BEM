'''
TO DO:
1. How to calculate power coefficient   [check if it is correct]
4. Finish Plotting Routines
'''

import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

#------------------------- FUNCTION DEFINITIONS -------------------------
def PrandtlCorrections(Nb, r, R, TSR, a, b, root_pos_R, dCT, dCQ):
    F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((1-(r/R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1+a)**2))))))
    F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1+a)**2)))))  
    F_tot = F_tip*F_root
    
    if(F_tot == 0) or (F_tot is np.nan) or (F_tot is np.inf):
        # handle exceptional cases for 0/NaN/inf value of F_tot
        # print("F total is 0/NaN/inf.")
        F_tot = 0.00001

    # a_new = ((1/2)*(-1+np.sqrt(1+(dCT))))    # axial induction factor, a
    # b_new = (dCQ)/(4*F_tot*(1+a)*(TSR*(r/R)))   # tangential induction factor, a'
    # a_new = a/F_tot
    # b_new = b/F_tot

    return F_tot, F_tip, F_root

def BladeElementMethod(Vinf, TSR, n, rho, R, r, root_pos_R, dr, Omega, Nb, a, b, twist, chord, polar_alfa, polar_cl, polar_cd, tol, P_up):
    flag = 0
    while True and (flag<1000):
            V_ax = Vinf*(1+a)       # axial velocity at the propeller blade
            V_tan = Omega*r*(1-b)   # tangential veloity at the propeller blade
            V_loc = np.sqrt(V_ax**2 + V_tan**2)

            phi = np.arctan(V_ax/V_tan)     # inflow angle [rad]
            alfa = twist - np.rad2deg(phi)  # local angle of attack [deg]
            
            Cl = np.interp(alfa, polar_alfa, polar_cl)
            Cd = np.interp(alfa, polar_alfa, polar_cd)
            
            C_ax = Cl*np.cos(phi) - Cd*np.sin(phi)      # axial force coefficient
            F_ax = (0.5*rho*V_loc**2)*C_ax*chord

            C_tan = Cl*np.sin(phi) + Cd*np.cos(phi)     # tangential force coefficient
            F_tan = (0.5*rho*V_loc**2)*C_tan*chord
            # gamma = 0.5*V_loc*Cl[j][i]*chord

            # sigma = (Nb*chord)/(2*np.pi*r)            # solidity

            # dCT = ((0.5*rho*V_loc**2)*chord*C_ax*Nb*dr)/(rho*(n**2)*(2*R)**4)
            # dCT = ((0.5*rho*V_loc**2)*chord*C_ax*Nb*dr)/(0.5*rho*(Vinf**2)*2*np.pi*r*dr)       # blade element thrust coefficient
            # dCP = (dr*F_tan*(r/R)*Nb*R*Omega)/(0.5*Vinf**3*np.pi*R**2)              # blade element power coefficient
            # dCQ = ((0.5*rho*V_loc**2)*chord*C_tan*Nb*r*dr)/(rho*(n**2)*(2*R)**5)    # blade element torque coefficient
            # dCT = (F_ax*Nb*dr)/(0.5*rho*Vinf**2*2*np.pi*r*dr)
            # dCQ = (r*F_tan)/(0.5*rho*Vinf**2*R*2*np.pi*r*dr)
            # dCQ = F_tan*r*dr*Nb
            # dCP = dCQ*TSR
            dCT = F_ax * Nb * dr/(rho*(n**2)*(2*R)**4)
            dCT1 = F_ax * Nb / (rho * Vinf**2 * np.pi * r)
            dCQ = ((0.5*rho*V_loc**2)*chord*C_tan*Nb*r*dr)/(rho*(n**2)*(2*R)**5)
            dCP = dCQ*TSR

            
            a_new = ((1/2)*(-1+np.sqrt(1+(dCT1))))
            b_new = F_tan * Nb / (2*rho*(2*np.pi*r)*Vinf*(1+a_new)*Omega*r)
            if (flag==0):
                a_b4_Pr = a_new
            
            # a_new = (1/2)*(-1+np.sqrt(1+dCT))
            # b_new = (F_tan*Nb)/(2*rho*(2*np.pi*r)*Vinf**2*(1+a_new)*TSR*(r/R))

            # a_new, b_new, F_tot, F_tip, F_root = PrandtlCorrections(Nb, r, R, TSR, a_new, b_new, root_pos_R, dCT, dCQ)
            F_tot, F_tip, F_root = PrandtlCorrections(Nb, r, R, TSR, a, b, root_pos_R, dCT, dCQ)
            a_new = a_new/F_tot
            b_new=b_new/F_tot
        
            if(np.abs(a-a_new)<tol) and (np.abs(b-b_new)<tol):
                a = a_new
                b = b_new
                flag += 1
                break
            else:
                # introduces relaxation to induction factors a and a' for easier convergence
                a = 0.75*a + 0.25*a_new
                b = 0.75*b + 0.25*b_new
                flag += 1
                continue
    P0_down = P_up + F_ax/(2*np.pi*r)
    return a_b4_Pr, a, b, Cl, Cd, F_ax, F_tan, alfa, phi, F_tot, F_tip, F_root, dCT, dCQ, dCP, P0_down

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
CT, CP, CQ = [np.zeros(len(J)) for i in range(3)]
a_b4_Pr, a = [(np.ones((len(J),len(r_R)-1))*(1/3)) for i in range(2)]
chord, b, Cl, Cd, F_ax, F_tan, dCT, dCQ, dCP, alfa, phi, F_tot, F_tip, F_root, P0_down = [np.zeros((len(J),len(r_R)-1)) for i in range(15)]

# Pressure just Upwind and infinity upwind
Pamb = 79495.22 #Pa
P_up = np.ones((len(J),len(r_R)-1))*(Pamb + 0.5*rho*(Vinf**2))  #[Pa]


# Solving BEM model
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord[j][i] = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist) * R
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(R/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*R        # length of the blade element
        
        a_b4_Pr[j][i], a[j][i], b[j][i], Cl[j][i], Cd[j][i], F_ax[j][i], F_tan[j][i], alfa[j][i], phi[j][i], F_tot[j][i], F_tip[j][i], F_root[j][i], dCT[j][i], dCQ[j][i], dCP[j][i], P0_down[j][i] = BladeElementMethod(Vinf, TSR[j], n[j], rho, R, r, root_pos_R, dr, Omega[j], Nb, a[j][i], b[j][i], twist, chord[j][i], polar_alfa, polar_cl, polar_cd, tol, P_up[j][i])

        CT[j] += dCT[j][i]    # thrust coefficient for given J
        CP[j] += dCP[j][i]    # power coefficient for given J
        CQ[j] += dCQ[j][i]    # torque coefficient for given J

#------------------------------- RESULTS --------------------------------

print("CT: ", CT)
print("CP: ", CP)
print("CQ: ", CQ)

# Plotting Routines

for i in range(len(J)):
    plt.figure("Local Angle of Attack vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],alfa[i][:], label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Alpha [deg]")
    plt.grid(True)
    plt.legend()
    
    plt.figure("Inflow vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],np.rad2deg(phi[i][:]), label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Inflow Angle [deg]")
    plt.grid(True)
    plt.legend()

    plt.figure("Axial Induction vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],a[i][:], label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Axial Induction Factor, a")
    plt.grid(True)
    plt.legend()

    plt.figure("Axial Induction before Correction vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],a_b4_Pr[i][:], label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Axial Induction Factor, a")
    plt.grid(True)
    plt.legend()

    plt.figure("Tangential Induction vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],b[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Tangential Induction Factor, a'")
    plt.grid(True)
    plt.legend()

    plt.figure("Thrust Coefficient vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],dCT[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Blade Element Thrust Coefficient, CT")
    plt.grid(True)
    plt.legend()
    
    plt.figure("Azimuthal Loading vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],F_tan[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Azimuthal Loading, F_tan")
    plt.grid(True)
    plt.legend()

    plt.figure("Axial Loading vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],F_ax[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Axial Loading, F_ax")
    plt.grid(True)
    plt.legend()
    
    plt.figure("Power Coefficient vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],dCP[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Blade Element Power Coefficient, CP")
    plt.grid(True)
    plt.legend()

    plt.figure("Torque Coefficient vs Blade Location", figsize = (8,4.5))
    plt.plot(r_R[1:],dCQ[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Blade Element Torque Coefficient, CQ")
    plt.grid(True)
    plt.legend()

    plt.figure("Stagnation Pressure at Rotor (Downwind)", figsize=(8,4.5))
    plt.plot(r_R[1:], P0_down[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Stagnation Pressure, At the Rotor(Downwind Side) [Pa]")
    plt.grid(True)
    plt.legend()

    plt.figure("Stagnation Pressure at Rotor (Infinity Downwind)", figsize=(8,4.5))
    plt.plot(r_R[1:], P0_down[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Stagnation Pressure, At the Rotor(Infinity Downwind) [Pa]")
    plt.grid(True)
    plt.legend()

    plt.figure("L/D vs r/R", figsize=(8,4.5))
    plt.plot(r_R[1:], (Cl[i][:]/Cd[i][:]), label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("L/D")
    plt.grid(True)
    plt.legend()

    plt.figure("Cl vs Chord", figsize=(8,4.5))
    plt.plot(chord[i][:], Cl[i][:], label="J = " + str(J[i]))
    plt.xlabel("Chord")
    plt.ylabel("Local Lift Coefficient")
    plt.grid(True)
    plt.legend()


plt.figure("Stagnation Pressure at Rotor (Upwind side)", figsize=(8,4.5))
plt.plot(r_R[1:], P_up[i][:], label="J = " + str(J[i]))
plt.xlabel("Normalised radius, r/R")
plt.ylabel("Stagnation Pressure, At the Rotor(Upwind side) [Pa]")
plt.grid(True)
plt.legend()

plt.figure("Stagnation Pressure at Rotor (Infinity Upwind)", figsize=(8,4.5))
plt.plot(r_R[1:], P_up[i][:], label="J = " + str(J[i]))
plt.xlabel("Normalised radius, r/R")
plt.ylabel("Stagnation Pressure, At the Rotor(Infinity Upwind) [Pa]")
plt.grid(True)
plt.legend()


plt.figure("Thrust and Torque vs Advance Ratio", figsize = (8,4.5))
plt.plot(J,((rho*(n**2)*(2*R)**4)*CT), '-o', label="Thrust",)
plt.plot(J,((rho*(n**2)*(2*R)**5)*CQ), '-o', label="Torque")
plt.xlabel("Normalised radius, r/R")
plt.ylabel("Forces")
plt.grid(True)
plt.legend()

plt.figure("Prandtl Corrections", figsize = (8,4.5))
plt.plot(r_R[1:],F_tip[i][:], label="F_tip")
plt.plot(r_R[1:],F_root[i][:], label="F_root")
plt.plot(r_R[1:],F_tot[i][:], label="F_tot")
plt.xlabel("Normalised radius, r/R")
plt.ylabel("Prandtl Correction Factor")
plt.grid(True)
plt.legend()

plt.show()