import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import os

os.chdir(os.path.dirname(__file__))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%---------------FUNCTION DEFINITIONS--------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Prandtl root and tip corrections
def PrandtlCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1+axial_induction)**2))
    Ftip = np.array((2/np.pi)*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp2 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1+axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp2)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot

# Load calculations
def loadBladeElement(vnorm, vtan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd, rho):
    vmag2 = vnorm**2 + vtan**2
    phi = np.arctan2(vnorm,vtan)
    alpha = twist - phi*180/np.pi
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5*rho*vmag2*cl*chord
    drag = 0.5*rho*vmag2*cd*chord
    fax = lift*np.cos(phi)+drag*np.sin(phi)
    ftan = lift*np.sin(phi)-drag*np.cos(phi)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord

    return fax, ftan, gamma

# Streamtube
def solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd, rho):
    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2) # area streamtube
    r_R = (r1_R+r2_R)/2                              # centroid
    a = 0.3                                          # initial guess
    atan = 0.2                                       # initial guess
    N_iter = 500
    error = 1e-7
    for i in range(N_iter):
        # Calculate velocity and loads at the blade element
        U_ax_rotor = Uinf*(1+a)
        U_tan_rotor = (1 - atan)*Omega*r_R*Radius
        fax, ftan, gamma = loadBladeElement(U_ax_rotor, U_tan_rotor, r_R, chord, twist, polar_alpha, polar_cl, polar_cd, rho)
        load3Daxial =fax*Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
        load3Dtan =ftan*Radius*(r2_R-r1_R)*NBlades    # 3D force in tangential direction
        CT = load3Daxial/(0.5 * rho * Uinf**2 * Area )
        CQ = load3Dtan/(0.5 * rho * Uinf**2 * Area * r_R*Radius)
        anew_ax = 0.5*(-1 + np.sqrt(1 + CT))
        
        # Apply prandtl corrections
        Prandtl, Prandtltip, Prandtlroot = PrandtlCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew_ax)
        if (Prandtl < 0.0001).all(): 
            Prandtl = 0.0001 
        anew_ax = anew_ax/Prandtl
        a = a*0.75 + anew_ax*0.25
        atan = ftan*NBlades/(2*np.pi*Uinf*(1+a)*Omega*2*(r_R*Radius)**2)
        # atan =atan/Prandtl

        if (np.abs(a-anew_ax) < error): 
            break
    return [a , atan, r_R, fax , ftan, gamma]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%----------------------MAIN-----------------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# P = PrandtlCorrection(0.250001, 0.25, 1, 1.4, 6, 0.3)
# print(P)
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
TipLocation_R = 1
RootLocation_R = 0.25

# Solving the BEM model
results =np.zeros([len(r_R)-1,6]) 

for i in range(len(r_R)-1):
    chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
    twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
    
    results[i,:] = solveStreamtube(Uinf, r_R[i], r_R[i+1], RootLocation_R, TipLocation_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd, rho )


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%----------------------PLOTS-----------------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2
dr = (r_R[1:]-r_R[:-1])*Radius
CT = np.sum(dr*results[:,3]*NBlades/(0.5*Uinf**2*np.pi*Radius**2))
CP = np.sum(dr*results[:,4]*results[:,2]*NBlades*Radius*Omega/(0.5*Uinf**3*np.pi*Radius**2))


print("CT is ", CT)
print("CP is ", CP)