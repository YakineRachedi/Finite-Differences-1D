import numpy as np
import matplotlib.pyplot as plt

# Paramètres du problème
v = 0.01  # Coefficient de diffusion
gamma = 10  # Coefficient de réaction
Nt_ref = 1000
Nx_ref = 1000

# Condition initiale
def initial_condition(x):
    return 0.25 * np.sin(np.pi * x) * np.exp(-20 * (x - 0.5) ** 2)

# Méthode de résolution du problème
def solve_FKPP(N_space, N_time):
    x = np.linspace(0, 1, N_space + 2)
    Dt = 1 / N_time
    h = 1 / (N_space + 1)
    u_0 = initial_condition(x)
    U_0 = u_0[1:-1]

    A = (1 + 2 * v * Dt / h ** 2 - gamma * Dt) * np.diag(np.ones(N_space)) + (-v * Dt / h ** 2) * (
            np.diag(np.ones(N_space - 1), 1) + np.diag(np.ones(N_space - 1), -1))

    u = U_0
    result = []

    for i in range(N_time):
        Bn = (Dt * gamma * u) * np.eye(N_space)
        u = np.linalg.solve(A + Bn, u)
        result.append(u)

    result = np.array(result)
    return result

# Solution de référence
U_ref = solve_FKPP(Nx_ref, Nt_ref)

error_norm_2 = []
error_norm_inf = []
times = []

# Valeurs de N qui divisent Nt
N_values = [20, 50, 100, 200, 500, 1000]

for N in N_values:
    # Fixer Nx et varier Nt
    U_N = solve_FKPP(Nx_ref, N)
    U_restriction = U_ref[::int(Nt_ref / N), :]
    difference = U_N - U_restriction

    # Calcul des erreurs faire la somme selon Nt
    L_2 = np.sqrt(np.sum(difference ** 2, axis=0) / (Nx_ref + 1))
    L_inf = np.linalg.norm(difference, np.inf, axis=0)

    max_error_norm_2 = np.max(L_2)
    max_error_norm_inf = np.max(L_inf, axis=0)

    error_norm_2.append(max_error_norm_2)
    error_norm_inf.append(max_error_norm_inf)
    times.append(N)

# Affichage des courbes d'erreur
plt.plot(times, error_norm_2, 'b', label='Max erreur norme 2')
plt.plot(times, error_norm_inf, 'r', label='Max erreur norme inf')
plt.xlabel("Time")
plt.ylabel("Solution")
plt.legend()
plt.grid(True)
plt.savefig('Courbe_CNVRGNCE_EX1.png')
plt.show()
