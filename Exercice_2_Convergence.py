import numpy as np
import matplotlib.pyplot as plt

# Paramètres du problème
v = 0.01  # Coefficient de diffusion
gamma = 10  # Coefficient de réaction
Nx_ref = 1000
Nt_ref = 1000

# Condition initiale
def initial_condition(x):
    return 0.25 * np.sin(np.pi * x) * np.exp(-20 * (x - 0.5) ** 2)

# Étapes de diffusion
def diffusion_step(U, N_space, N_time):
    h = 1 / (N_space + 1)
    Dt = 1 / N_time
    A_dif = (1 + (Dt * v) / h**2) * np.diag(np.ones(N_space)) - ((Dt * v) / (2 * h**2)) * np.diag(np.ones(N_space-1), 1) - ((Dt * v) / (2 * h**2)) * np.diag(np.ones(N_space-1), -1)
    U_dif = np.linalg.solve(A_dif, U)
    return U_dif

# Étape de réaction
def reaction_step(U, N_time):
    Dt = 1 / N_time
    Delta = (1 - Dt * gamma) ** 2 + 4 * Dt * gamma * U
    U_reaction = (-(1 - Dt * gamma) + np.sqrt(Delta)) / (2 * (Dt * gamma))
    return U_reaction

# Résolution par splitting
def solve_splitting(N_space, N_time):
    x = np.linspace(0, 1, N_space + 2)
    T = np.linspace(0, 1, N_time)
    U_ini = initial_condition(x)
    U_ini = U_ini[1:-1]
    u = U_ini
    result = []
    
    for i in range(N_time):
        u = diffusion_step(u, N_space, N_time)
        u = reaction_step(u, N_time)
        u = diffusion_step(u, N_space, N_time)
        result.append(u)
    
    result = np.array(result)
    return result

# Solution de référence
U_ref = solve_splitting(Nx_ref, Nt_ref)

# Calcul de l'erreur pour différentes valeurs de N
error_norm_L2 = []
error_norm_inf = []
times = []

# Valeurs de N qui divisent Nt
N_values = [20, 50, 100, 200, 500, 1000]

for N in N_values:
    # Fixer Nx et varier Nt
    U_N = solve_splitting(Nx_ref, N)  
    U_restriction = U_ref[::int(Nt_ref / N), :]
    difference = U_restriction - U_N

    # Calcul des erreurs
    L_2 = (np.sum(difference ** 2, axis=0) / (Nx_ref + 1)) ** 0.5
    L_inf = np.linalg.norm(difference, np.inf, axis=0)

    max_error_L2 = np.max(L_2)
    max_error_inf = np.max(L_inf, axis=0)

    error_norm_L2.append(max_error_L2)
    error_norm_inf.append(max_error_inf)
    times.append(N)

# Affichage des courbes d'erreur
plt.plot(times, error_norm_L2, 'b', label='Max erreur norme 2')
plt.plot(times, error_norm_inf, 'r', label='Max erreur norme inf')
plt.legend()
plt.grid(True)
plt.savefig('Courbe_CNVRGNCE.png')
plt.show()
