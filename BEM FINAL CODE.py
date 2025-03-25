import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np  
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

#------------------------- FUNCTION DEFINITIONS -------------------------
def PrandtlCorrections(Nb, r, R, TSR, a, root_pos_R, tip_pos_R):
    F_tip = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((tip_pos_R-(r/R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2))))))
    F_root = (2/np.pi)*np.arccos(np.exp((-Nb/2)*(((r/R)-(root_pos_R))/(r/R))*(np.sqrt(1 + ((TSR*(r/R))**2)/((1-a)**2)))))  
    F_tot = F_tip*F_root
    
    if(F_tot == 0) or (F_tot is np.nan) or (F_tot is np.inf):
        # handle exceptional cases for 0/NaN/inf value of F_tot
        # print("F total is 0/NaN/inf.")
        F_tot = 0.00001

    return F_tot, F_tip, F_root

def BladeElementMethod(Vinf, TSR, n, rho, R, r, root_pos_R, tip_pos_R, dr, Omega, Nb, a, b, twist, chord, polar_alfa, polar_cl, polar_cd, tol, P_up):
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
            F_ax = (0.5*rho*V_loc**2)*C_ax*chord        # axial force [N/m]

            C_tan = Cl*np.sin(phi) + Cd*np.cos(phi)     # tangential force coefficient
            F_tan = (0.5*rho*V_loc**2)*C_tan*chord      # tangential force [N/m]
           
            dCT = (F_ax*Nb*dr)/(rho*(n**2)*(2*R)**4)        # blade element thrust coefficient                   
            dCQ = (F_tan*Nb*r*dr)/(rho*(n**2)*(2*R)**5)     # blade element torque coefficient
            dCP = (F_ax*Nb*dr*Vinf)/(rho*(n**3)*(2*R)**5)   # blade element power coefficient
            
            a_new = ((1/2)*(-1+np.sqrt(1+(F_ax * Nb / (rho * Vinf**2 * np.pi * r)))))
            b_new = F_tan * Nb / (2*rho*(2*np.pi*r)*Vinf*(1+a_new)*Omega*r)
            if (flag==0):
                a_b4_Pr = a_new
            
            F_tot, F_tip, F_root = PrandtlCorrections(Nb, r, R, TSR, a, root_pos_R, tip_pos_R)
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
Pamb = 79495.22                 # static pressure at h=2000m [Pa]
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
tol = 1e-7  # convergence tolerance

# Variable initialisation
CT, CP, CQ = [np.zeros(len(J)) for i in range(3)]
a_b4_Pr, a = [(np.ones((len(J),len(r_R)-1))*(1/3)) for i in range(2)]
chord, b, Cl, Cd, F_ax, F_tan, dCT, dCQ, dCP, alfa, phi, F_tot, F_tip, F_root, P0_down = [np.zeros((len(J),len(r_R)-1)) for i in range(15)]

P_up = np.ones((len(J),len(r_R)-1))*(Pamb + 0.5*rho*(Vinf**2))  # pressure upwind of rotor [Pa]

# Solving BEM model
for j in range(len(J)):
    for i in range(len(r_R)-1):    
        chord[j][i] = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_dist) * R
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_dist)
        
        r = (r_R[i+1]+r_R[i])*(R/2)     # radial distance of the blade element
        dr = (r_R[i+1]-r_R[i])*R        # length of the blade element
        
        a_b4_Pr[j][i], a[j][i], b[j][i], Cl[j][i], Cd[j][i], F_ax[j][i], F_tan[j][i], alfa[j][i], phi[j][i], F_tot[j][i], F_tip[j][i], F_root[j][i], dCT[j][i], dCQ[j][i], dCP[j][i], P0_down[j][i] = BladeElementMethod(Vinf, TSR[j], n[j], rho, R, r, root_pos_R, tip_pos_R, dr, Omega[j], Nb, a[j][i], b[j][i], twist, chord[j][i], polar_alfa, polar_cl, polar_cd, tol, P_up[j][i])

        CT[j] += dCT[j][i]    # thrust coefficient for given J
        CP[j] += dCP[j][i]    # power coefficient for given J
        CQ[j] += dCQ[j][i]    # torque coefficient for given J

#------------------------------- RESULTS --------------------------------

print("CT: ", CT)
print("CP: ", CP)
print("CQ: ", CQ)

# Plotting Routines

fs = (8,5)      # size of plots

for i in range(len(J)):
    plt.figure("Local Angle of Attack vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],alfa[i][:], label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Alpha [deg]")
    plt.grid(True)
    plt.legend()
    
    plt.figure("Inflow vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],np.rad2deg(phi[i][:]), label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Inflow Angle [deg]")
    plt.grid(True)
    plt.legend()

    plt.figure("Axial Induction vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],a[i][:], label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Axial Induction Factor, a")
    plt.grid(True)
    plt.legend()

    plt.figure("Axial Induction before Correction vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],a_b4_Pr[i][:], label= "J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Axial Induction Factor, a")
    plt.grid(True)
    plt.legend()

    plt.figure("Tangential Induction vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],b[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Tangential Induction Factor, a'")
    plt.grid(True)
    plt.legend()

    plt.figure("Thrust Coefficient vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],dCT[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Blade Element Thrust Coefficient, CT")
    plt.grid(True)
    plt.legend()
    
    plt.figure("Azimuthal Loading vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],F_tan[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Azimuthal Loading, F_tan")
    plt.grid(True)
    plt.legend()

    plt.figure("Axial Loading vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],F_ax[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Axial Loading, F_ax")
    plt.grid(True)
    plt.legend()
    
    plt.figure("Power Coefficient vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],dCP[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Blade Element Power Coefficient, CP")
    plt.grid(True)
    plt.legend()

    plt.figure("Torque Coefficient vs Blade Location", figsize = fs)
    plt.plot(r_R[1:],dCQ[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Blade Element Torque Coefficient, CQ")
    plt.grid(True)
    plt.legend()

    plt.figure("Stagnation Pressure Downwind of Rotor", figsize = fs)
    plt.plot(r_R[1:], P0_down[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Stagnation Pressure [Pa]")
    plt.grid(True)
    plt.legend()

    plt.figure("Stagnation Pressure Infitinely Downwind of Rotor", figsize = fs)
    plt.plot(r_R[1:], P0_down[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Stagnation Pressure, [Pa]")
    plt.grid(True)
    plt.legend()

    plt.figure("Aerodynamic Efficiency vs r/R", figsize = fs)
    plt.plot(r_R[1:], (Cl[i][:]/Cd[i][:]), label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("L/D")
    plt.grid(True)
    plt.legend()

    plt.figure("Cl vs Chord", figsize = fs)
    plt.plot(chord[i][:], Cl[i][:], label="J = " + str(J[i]))
    plt.xlabel("Chord")
    plt.ylabel("Blade Element Lift Coefficient")
    plt.grid(True)
    plt.legend()

    plt.figure("Stagnation Pressure Upwind of Rotor", figsize = fs)
    plt.plot(r_R[1:], P_up[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Stagnation Pressure, [Pa]")
    plt.grid(True)
    plt.legend()

    plt.figure("Stagnation Pressure Infinitely Upwind of Rotor", figsize = fs)
    plt.plot(r_R[1:], P_up[i][:], label="J = " + str(J[i]))
    plt.xlabel("Normalised radius, r/R")
    plt.ylabel("Stagnation Pressure, [Pa]")
    plt.grid(True)
    plt.legend()


plt.figure("Thrust and Torque vs Advance Ratio", figsize = fs)
plt.plot(J,((rho*(n**2)*(2*R)**4)*CT), '-o', label="Thrust",)
plt.plot(J,((rho*(n**2)*(2*R)**5)*CQ), '-o', label="Torque")
plt.xlabel("Advance Ratio, J")
plt.ylabel("Forces")
plt.grid(True)
plt.legend()

plt.figure("Prandtl Corrections", figsize = fs)
plt.plot(r_R[1:],F_tip[i][:], label="F_tip")
plt.plot(r_R[1:],F_root[i][:], label="F_root")
plt.plot(r_R[1:],F_tot[i][:], label="F_tot")
plt.xlabel("Normalised radius, r/R")
plt.ylabel("Prandtl Correction Factor")
plt.grid(True)
plt.legend()

plt.close('all')

# Create 3D figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Convert r_R to actual blade length
r_values = r_R * R  # Convert normalized radius to meters
chord_values = chord_dist * R  # Scale chord to actual size
twist_values = np.radians(twist_dist)  # Convert degrees to radians for rotation

# Create mesh grid for chord-wise variation
num_chord_points = nodes  # Number of points along chord
chord_grid = np.linspace(0, 1, num_chord_points)  # Chord runs from -0.5 to 0.5
r_mesh, chord_mesh = np.meshgrid(r_values, chord_grid)

# Compute 3D coordinates of the blade shape
X = r_mesh
Y = (chord_mesh.T * chord_values).T * np.cos(twist_values)  # Apply twist rotation
Z = (chord_mesh.T * chord_values).T * np.sin(twist_values)  # Height due to twist

# Plot the surface
ax.plot_surface(X, Y, Z, color='lightblue', alpha=1)

# Labels and view adjustments
ax.set_xlabel("Spanwise Position (r) [m]")
ax.set_ylabel("Chordwise Direction [m]")
ax.set_zlabel("Twist Effect [m]")
ax.set_title("3D Blade Geometry")

ax.view_init(elev=20, azim=45)  # Adjust viewing angle

print(chord_values)

plt.show()