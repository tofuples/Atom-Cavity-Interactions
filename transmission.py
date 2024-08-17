from qutip import *
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt 
import math

# Systems Dimensions
N = 5 #Hilbert space dimension
idatomA = qeye(3) #atom identity operator
ida = qeye(N) # field identity operator
nloop = 500

# Hamiltonian parameters
g = 50.0 #atom-field coupling
E = 1.0 #probe field
O = 250.0 #Rabi Frequency

D1 = 0.0 #detuning atom-cavity
D2 = 0.0 #detuning atom-probe field
DP = 100.0 #detuning cavity-probe field
DPList = np.linspace(-DP,DP,nloop)
OList = np.linspace(0,O,nloop)

#Master Equation Parameters (decay and dephasing rates)
Gamma31 = 0.1 
Gamma32 = 0.1 
gamma2 = 0.0 
gamma3 = 0.0 
kappa = 1.0 #cavity decay

#Atomic Operators
s12=Qobj([[0,1,0],[0,0,0],[0,0,0]])
s13=Qobj([[0,0,1],[0,0,0],[0,0,0]])
s23=Qobj([[0,0,0],[0,0,1],[0,0,0]])

#ATOM A
S13A = tensor(ida,s13)
S23A = tensor(ida,s23)
S11A = S13A*S13A.dag()
S22A = S23A*S23A.dag()
S33A = S23A.dag()*S23A 

#Cavity Operators
a=tensor(destroy(N),idatomA)

#Colapse Operators
C1 = math.sqrt(2*kappa)*a #cavity mode
C31 = math.sqrt(2*Gamma31)*S13A
C32 = math.sqrt(2*Gamma32)*S23A
C22 = math.sqrt(2*gamma2)*S22A
C33 = math.sqrt(2*gamma3)*S33A 

C_list = [C1, C31, C32, C22, C33]

#Hamiltonian
H1=D1*(S33A)+D1*S22A - D2*(S22A) + g*S13A.dag()*a +g*a.dag()*S13A + E*a+E*a.dag()


#SIMULAÇÃO#
T=[]
for k in range(0,nloop):
    Oc=OList[k] 
    H= Oc*S23A + Oc*S23A.dag()+ H1 #Hamiltoniano total
    rhoss=steadystate(H, C_list) 
    Transmiss= expect(a.dag()*a, rhoss)/((E/kappa)**2) #Transmissão normalizada
    Tr= Transmiss.real*100
    T+=[Tr]
    


#PLOT#
plt.figure(1, dpi=300)
plt.plot(OList,T, linewidth=2 )
plt.yscale("log")
plt.show()