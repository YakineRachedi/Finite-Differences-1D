import numpy as np
import matplotlib.pyplot as plt


v = 0.01 #Coeff de diffusion
gamma = 10 #Coeff de réaction
Nx = 100 
Nt = 100  

def CI(x):
    return 0.25 * np.sin(np.pi * x) * np.exp(-20 * (x-0.5) ** 2)


def etape_diffusion(U,N_espace,N_temps):
    h = 1 / (N_espace + 1)
    Dt = 1 / N_temps
    A_dif = np.zeros((N_espace,N_temps))
    A_dif = (1 + (Dt * v) / h**2) * np.diag(np.ones(N_espace)) - ((Dt * v) / (2 * h ** 2)) * np.diag(np.ones(N_espace-1),1) - ((Dt * v) / (2 * h ** 2)) * np.diag(np.ones(N_espace-1),-1)
    U_dif = U
    U_dif = np.linalg.solve(A_dif,U)
    return U_dif

def etape_réaction(U,N_temps):
    Dt = 1 / N_temps
    Delta = (1 - Dt * gamma) ** 2 + 4 * Dt * gamma * U
    U_reaction = (-(1 - Dt * gamma) + np.sqrt(Delta)) / (2 * (Dt * gamma))
    return U_reaction

def résolution_du_splitting(N_espace,N_temps):
    x = np.linspace(0,1,N_espace+2)
    T = np.linspace(0,1,N_temps)
    U_ini = CI(x)
    U_ini = U_ini[1:-1]
    u = U_ini
    Temps = np.zeros(N_temps)
    for i in range(N_temps):
        u = etape_diffusion(u,N_espace,N_temps)
        u = etape_réaction(u,N_temps)
        u = etape_diffusion(u,N_espace,N_temps)
        Temps[i] = T[i]
        #print('Temps a l\'instant',i,T[i])
    u = np.append(u,[0])
    u = np.append([0],u)
    return u,x

Solution_approchée,x = résolution_du_splitting(Nx,Nt)    
plt.plot(x,Solution_approchée,'b',label = 'Solution approchée en utilisant le Splitting de Strang')
plt.legend()
plt.grid(True)
plt.savefig('Courbe_DF_EX2.png')
plt.show()