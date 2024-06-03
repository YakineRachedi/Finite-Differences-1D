import numpy as np
import matplotlib.pyplot as plt

v = 0.01 #Coeff de diffusion
gamma = 10 #Coeff de réaction
Nt = 100
Nx = 100
Dt = 1 / Nt
h = 1 / (Nx+1)
x = np.linspace(0,1,Nx+2)

def CI(x):
    return 0.25 * np.sin(np.pi * x) * np.exp(-20 * (x-0.5) ** 2)

def DF_FKKP(N_espace,N_temps):
    A = np.zeros((N_espace, N_temps))
    A = (1 + 2 * v * Dt /h**2 - gamma * Dt) * np.diag(np.ones(N_espace)) + (-v * Dt / h ** 2) * (np.diag(np.ones(N_espace-1),1) + np.diag(np.ones(N_espace-1),-1))
    F = np.zeros(N_temps)
    return A, F

(A,F) = DF_FKKP(Nx,Nt,CI(x))
u = CI(x)
u = u[1:-1]
T = np.zeros(Nt)

for i in range(Nt):
    Bn = (Dt * gamma * u) * np.eye(Nx)
    u = np.linalg.solve(A + Bn,u + Dt * F)
    T[i] = i * Dt 

u=np.append([0],u)
u=np.append(u,[0])
plt.plot(x,u)       
plt.plot(np.linspace(0,1,Nx+2),u,'r',label = 'Solution approchée')
plt.xlabel("valeurs de x ")
plt.ylabel("valeurs de u")
plt.grid(True)
plt.legend()
plt.savefig('Courbe_DF_EX1.png')
plt.show()
