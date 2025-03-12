# BEM
Python code on the blade element momentum theory.
Instructions
Do this group assignment in groups of 3. You can make the groups yourself and/or use the Discussion board to find a group. This assignment counts for 15% of your grade.

Enroll for Group Assignments in the menu to create your group first!

**Overview**

In this assignment you are asked to program a BEM (Blade Element Momentum) model for one of two rotors:

A wind turbine, in axial flow (optional: yawed flow)
A propeller, in axial flow, in cruise (optional: yawed flow; optional: and energy harvesting)  
NOTE: choose one of the two rotor cases for the mandatory assignment, If you wish, you can do both rotors.

The BEM code should incorporate the Glauert correction for heavily loaded rotors for the wind turbine case, and the Prandtl tip and root corrections. Assume a constant airfoil along the span and use the provided polars of the:

DU airfoil for the wind turbine
ARA-D 8% airfoil for the propeller
 Discuss the different inaccuracies introduced into the solution by using a single airfoil polar only.

 **Propeller** 

1-Calculate the propeller performance at advance ratios of J=1.6, 2.0, and 2.4. (advance ratio J = Vinf / n*D)

2-Optional: Calculate the performance for the different yawed flow cases at advance ratio J = 2.0. 

3-Optional: During landing, a propeller can be used to slow down the aircraft. If the propeller is driven by an electric engine, power can be harvested from the flow in this condition. For the given operational specs and basic rotor specs (radius, number of blades), change the collective blade pitch or the chord distribution or the twist distribution in order to maximize the power coefficient in this regime. You can choose your own design approach. Compare with the expected result from actuator-disk theory. Discuss the rationale for your design, including the twist and chord distributions, and compare with the results obtained with the baseline rotor specs, comparing cruise operation and landing operation with energy harvesting and the impact of your design choices (assume the same operational specs).

**Baseline design of the propeller rotor**

![image](https://github.com/user-attachments/assets/7852674f-c9df-45ea-8caf-116f87b090cb)

The local blade pitch (defined as in Figure 1) is equal to the sum of the local blade twist angle (blade characteristic) plus the collective blade pitch at the reference location (operational characteristic). In this case, the reference location is chosen at r/R=0.70. The blade twist at this point then needs to be zero by definition.

![image](https://github.com/user-attachments/assets/324a46d1-4970-4796-923c-9643172264e4)


Code in .zip or .rar file. Make sure the code is ready to run, so include all relevant files.

**References**

[1] Burton T., Jenkins N., Sharpe D., Bossanyi E., "Wind Energy Handbook", sections 3.5 and 3.6

[2] Veldhuis, L. L. M., Propeller Wing Aerodynamic Interference, PhD thesis, Delft University of Technology, 2005, Appendix A.

[3] Sub-module 2.2.3 Programming a BEM model

