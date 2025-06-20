import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

#------------------------- FUNCTION DEFINITIONS -------------------------
def PrandtlCorrections(Nb, r, R, TSR, a, b, root_pos_R, dCT, dCQ):
    F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((1-(r/R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2))))))
    F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2)))))  
    F_tot = F_tip*F_root
    
    if(F_tot == 0) or (F_tot is np.nan) or (F_tot is np.inf):
        # handle exceptional cases for 0/NaN/inf value of F_tot
        # print("F total is 0/NaN/inf.")
        F_tot = 0.00001
    
    a_new = a/F_tot
    b_new = b/F_tot

    return a_new, b_new, F_tot, F_tip, F_root

def BladeElementMethod(Vinf, TSR, n, rho, R, r, root_pos_R, dr, Omega, Nb, a, b, twist, chord, polar_alfa, polar_cl, polar_cd, tol):
    flag = 0
    a_b4_Pr = 0
    while True and (flag<1000):
            V_ax = Vinf*(1+a)       # axial velocity at the propeller blade
            V_tan = Omega*r*(1-b)   # tangential veloity at the propeller blade
            V_loc = np.sqrt(V_ax**2 + V_tan**2)

            phi = np.arctan(V_ax/V_tan)     # inflow angle [rad]
            alfa = twist - np.rad2deg(phi)  # local angle of attack [deg]

            Cl = np.interp(alfa, polar_alfa, polar_cl)
            Cd = np.interp(alfa, polar_alfa, polar_cd)
            
            C_ax = Cl*np.cos(phi) - Cd*np.sin(phi)      # axial force coefficient
            F_ax = (0.5*rho*V_loc**2)*C_ax*chord*Nb

            C_tan = Cl*np.sin(phi) + Cd*np.cos(phi)     # tangential force coefficient
            F_tan = (0.5*rho*V_loc**2)*C_tan*chord*Nb*r

            # gamma = 0.5*V_loc*Cl[j][i]*chord
            # sigma = (Nb*chord)/(2*np.pi*r)            # solidity

            # dCT = ((0.5*rho*V_loc**2)*chord*C_ax*Nb*dr)/(rho*(n**2)*(2*R)**4)       # blade element thrust coefficient
            dCT = F_ax/(rho*(n**2)*(2*R)**4)
            # dCT = (F_ax)/(0.5*rho*(Vinf**2)*2*np.pi*r*dr)
            # dCQ = ((0.5*rho*V_loc**2)*chord*C_tan*Nb*r*dr)/(rho*(n**2)*(2*R)**5)    # blade element torque coefficient
            # dCQ = (r*F_tan)/(0.5*rho*(Vinf**2)*r*2*np.pi*r*dr)
            # dCQ = F_tan*r*dr*Nb
            dCQ = F_tan/((rho*(n**2)*(2*R)**5))
            # dCP = (dr*F_tan*(r/R)*Nb*R*Omega)/(0.5*(Vinf**3)*np.pi*R**2)            # blade element power coefficient
            dCP = dCT*(1+a)

            a = (1/2)*(-1+np.sqrt(1+(dCT)))    # axial induction factor, a
            # b = (dCQ)/(4*(1+a)*(TSR*(r/R)))   # tangential induction factor, a'
            b = (F_tan*Nb)/(2*rho*(2*np.pi*r)*(Vinf**2)*(1+a)*TSR*(r/R))

            if (flag==1):
                a_b4_Pr = a

            a_new, b_new, F_tot, F_tip, F_root = PrandtlCorrections(Nb, r, R, TSR, a, b, root_pos_R, dCT, dCQ)

            if(np.abs(a-a_new)<tol) and (np.abs(b-b_new)<tol):
                a = a_new
                b = b_new
                flag += 1
                break
            else:
                # introduces relaxation to induction factors a and a' for easier convergence
                a = a_new
                b = b_new
                flag += 1
                continue
    return a_b4_Pr, a, b, Cl, Cd, F_tan, alfa, phi, F_tot, F_tip, F_root, dCT, dCQ, dCP

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
chord_dist = 0.18 - 0.06*(r_R)                  # chord distribution 
twist_dist = -50*(r_R) + 35 + pitch             # twist distribution [deg]

# Dependent variables 
n = Vinf/(2*J*R)    # RPS [Hz]
Omega = 2*np.pi*n   # Angular velocity [rad/s]
TSR = np.pi/J       # tip speed ratio

# Iteration inputs
tol = 1e-7  # convergence tolerance

# Variable initialisation
CT, CP, CQ = [np.zeros(len(J)) for i in range(3)]
a_b4_Pr, a = [(np.ones((len(J),len(r_R)-1))*(1/3)) for i in range(2)]
b, Cl, Cd, F_tan, dCT, dCQ, dCP, alfa, phi, F_tot, F_tip, F_root = [np.zeros((len(J),len(r_R)-1)) for i in range(12)]

# Solving BEM model
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist) * R # chord length of the blade element [m]
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(R/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*R        # length of the blade element
        
        a_b4_Pr[j][i], a[j][i], b[j][i], Cl[j][i], Cd[j][i], F_tan[j][i], alfa[j][i], phi[j][i], F_tot[j][i], F_tip[j][i], F_root[j][i], dCT[j][i], dCQ[j][i], dCP[j][i] = BladeElementMethod(Vinf, TSR[j], n[j], rho, R, r, root_pos_R, dr, Omega[j], Nb, a[j][i], b[j][i], twist, chord, polar_alfa, polar_cl, polar_cd, tol)

        CT[j] += dCT[j][i]    # thrust coefficient for given J
        CP[j] += dCP[j][i]    # power coefficient for given J
        CQ[j] += dCQ[j][i]    # torque coefficient for given J

#------------------------------- RESULTS --------------------------------

print("CT: ", CT)
print("CP: ", CP)
print("CQ: ", CQ)
'''
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

    plt.figure("CT vs Axial Induction", figsize = (8,4.5))
    plt.plot(a[i][:], dCT[i][:], label= "J = " + str(J[i]))
    plt.xlabel("Axial Induction Factor, a")
    plt.ylabel("Blade Element Thrust Coefficient, CT")
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
'''