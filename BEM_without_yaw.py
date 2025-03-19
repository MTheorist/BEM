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
    fax = lift*np.cos(phi)-drag*np.sin(phi)
    ftan = lift*np.sin(phi)+drag*np.cos(phi)   
    gamma = 0.5*np.sqrt(vmag2)*cl*chord

    return fax, ftan, gamma

# Streamtube

def solveStreamtube1(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd, rho, pitch):
    r_R = (r1_R+r2_R)/2                              # centroid
    Area = 2*np.pi*((Radius**2)*r_R*(r2_R-r1_R))     # streamtube area
    a = 0.3                                          # initial guess
    atan = 0.2                                       # initial guess
    N_iter = 500
    error = 1e-7
    for i in range(N_iter):
        # Calculate velocity and loads at the blade element
        U_ax_rotor = Uinf*(1+a)
        U_tan_rotor = (1 - atan)*Omega*r_R*Radius
        fax, ftan, gamma = loadBladeElement(U_ax_rotor, U_tan_rotor, r_R, chord, twist, polar_alpha, polar_cl, polar_cd, rho)
        load3Daxial =fax *Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
        load3Dtan =ftan*(Radius**2)*(r2_R-r1_R)*NBlades    # 3D force in tangential direction
        CT = load3Daxial/(0.5*rho  * Uinf**2 * Area )
        CQ = load3Dtan/(0.5 *rho * Uinf**2 * Area * r_R*Radius)
        anew_ax = 0.5*(-1 + np.sqrt(1 + CT))
        
        # Apply prandtl corrections
        # Prandtl, Prandtltip, Prandtlroot = PrandtlCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew_ax)
        # if (Prandtl < 0.0001).all(): 
        #     Prandtl = 0.0001 
        # anew_ax = anew_ax/Prandtl
        a =  a*0.75 + anew_ax*0.25
        # atan = (ftan*NBlades)/(2*rho*(2*np.pi*r_R*Radius)*(Uinf**2)*(1+a)*TSR*r_R)
        atan = (ftan*NBlades)/(2*rho*2*np.pi*(Radius*r_R)**2*Uinf*(1+a)*Omega)
        # atan =atan/Prandtl

        if (np.abs(a-anew_ax) < error): 
            break
    return [a , atan, r_R, fax , ftan, gamma]

def solveStreamtube2(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd, rho, pitch):
    r_R = (r1_R+r2_R)/2                              # centroid
    r = r_R * Radius
    dr = (r2_R-r1_R)*Radius
    Area = 2*np.pi*(Radius*r*(r2_R-r1_R))            # streamtube area
    a = a_new = 0.3                                          # initial guess
    atan = atan_new = 0.2                                       # initial guess
    N_iter = 500
    error = 1e-7
    
    for i in range(N_iter):
        V_ax = Uinf*(1+a)
        V_tan = Omega*r*(1-atan)
        V_inf = np.sqrt(V_ax**2 + V_tan**2)
        
        fax, ftan, gamma = loadBladeElement(V_ax, V_tan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd, rho)

        dT = fax*NBlades*dr
        dQ = ftan*NBlades*r*dr

        a_roots = np.roots([1,1,(-dT/(4*np.pi*r*rho*V_inf**2))])
        
        if(a_roots[0]>a_roots[1]):
            a_new = a_roots[0]
        else:
            a_new = a_roots[1]
        
        Ftot, _, _ = PrandtlCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, a_new)
        if (Ftot < 0.0001).all():
            Ftot = 0.0001
        a_new /= Ftot

        atan_new = dQ/(4*np.pi*r**3*V_inf*(1+a_new)*Omega*dr)

        # atan_new /= Ftot

        if (np.abs(a-a_new) < error): 
            break
        else:
            a = a_new
            atan = atan_new

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
NBlades = 6
TipLocation_R = 1
RootLocation_R = 0.25

n = Uinf / (2*J*Radius)
Omega = 2*np.pi*n
TSR = 1/J

# Solving the BEM model
results =np.zeros([len(r_R)-1,6]) 

for i in range(len(r_R)-1):
    chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
    twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
    
    results[i,:] = solveStreamtube2(Uinf, r_R[i], r_R[i+1], RootLocation_R, TipLocation_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd, rho, pitch)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%----------------PLOTTING ROUTINE-----------------%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

areas = (r_R[1:]**2-r_R[:-1]**2)*np.pi*Radius**2
dr = (r_R[1:]-r_R[:-1])*Radius
CT = np.sum(dr*results[:,3]*NBlades/(0.5*Uinf**2*np.pi*Radius**2))
CP = np.sum(dr*results[:,4]*results[:,2]*NBlades*Radius*Omega/(0.5*Uinf**3*np.pi*Radius**2))
print("CT is ", CT)
print("CP is ", CP)

# Plotting the Prandtl tip and root correction
r_R = np.arange(0.25, 1, .01)
a = np.zeros(np.shape(r_R))+0.3
Prandtl, Prandtltip, Prandtlroot = PrandtlCorrection(r_R, 0.25, TipLocation_R, TSR, 6, a)
fig1 = plt.figure(figsize=(12, 6))
plt.plot(r_R, Prandtl, 'r-', label='Prandtl')
plt.plot(r_R, Prandtltip, 'g.', label='Prandtl tip')
plt.plot(r_R, Prandtlroot, 'b.', label='Prandtl root')
plt.xlabel('r/R')
plt.legend()


# Plotting the polars of the airfoil
fig2, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].plot(polar_alpha, polar_cl)
axs[0].set_xlim([-30,30])
axs[0].set_xlabel(r'$\alpha$')
axs[0].set_ylabel(r'$C_l$')
axs[0].grid()
axs[1].plot(polar_cd, polar_cl)
axs[1].set_xlim([0,.1])
axs[1].set_xlabel(r'$C_d$')
axs[1].grid()


# Plotting the axial and tangential inductions
fig1 = plt.figure(figsize=(12, 6))
plt.title('Axial and tangential induction')
plt.plot(results[:,2], results[:,0], 'r-', label=r'$a$')
plt.plot(results[:,2], results[:,1], 'g--', label=r'$a^,$')
plt.grid()
plt.xlabel('r/R')
plt.legend()


# Plotting the normal and tangential forces normalised
fig1 = plt.figure(figsize=(12, 6))
plt.title(r'Normal and tagential force, non-dimensioned by $\frac{1}{2} \rho U_\infty^2 R$')
plt.plot(results[:,2], results[:,3]/(0.5*Uinf**2*Radius), 'r-', label=r'Fnorm')
plt.plot(results[:,2], results[:,4]/(0.5*Uinf**2*Radius), 'g--', label=r'Ftan')
plt.grid()
plt.xlabel('r/R')
plt.legend()

# Plotting the Circulation
fig1 = plt.figure(figsize=(12, 6))
plt.title(r'Circulation distribution, non-dimensioned by $\frac{\pi U_\infty^2}{\Omega * NBlades } $')
plt.plot(results[:,2], results[:,5]/(np.pi*Uinf**2/(NBlades*Omega)), 'r-', label=r'$\Gamma$')
plt.grid()
plt.xlabel('r/R')
plt.legend()
plt.show()


