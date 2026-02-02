# matvec_col.py - Décomposition par COLONNES
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dimension du problème (Augmentée à 4096 pour voir le speedup réel)
N = 4096 

# 1. Calcul de Nloc (Colonnes par processus)
if N % size != 0:
    if rank == 0: 
        print("Erreur : N doit être divisible par le nombre de processus.")
    comm.Abort()
    
N_loc = N // size

# 2. Initialisation Locale de la Matrice A (Seulement les colonnes détenues)
# Chaque processus crée une matrice N x N_loc
# La formule originale est A_ij = (i+j) % N + 1. 
# Ici, l'indice global de la colonne 'j' est : j_local + (rank * N_loc)
A_local = np.empty((N, N_loc), dtype=np.float64)
offset_col = rank * N_loc

for i in range(N):
    for j_local in range(N_loc):
        j_global = j_local + offset_col
        A_local[i, j_local] = (i + j_global) % N + 1.0

# 3. Distribution du vecteur u
u_local = np.empty(N_loc, dtype=np.float64)
if rank == 0:
    # Crée le vecteur complet uniquement sur le rang 0
    u_full = np.array([i + 1.0 for i in range(N)], dtype=np.float64)
else:
    u_full = None

# Scatter : Distribue des morceaux de u vers u_local
comm.Scatter(u_full, u_local, root=0)

# Début de la mesure du temps (après setup)
comm.Barrier()
start_time = MPI.Wtime()

# 4. Produit Matrice-Vecteur Local
# (N x N_loc) dot (N_loc) -> Résulte en un vecteur de taille N
v_partial = np.dot(A_local, u_local)

# 5. Réduction (Somme) des résultats partiels
# Nous devons sommer les vecteurs partiels de tous pour obtenir le résultat final
v_final = np.empty(N, dtype=np.float64)
comm.Allreduce(v_partial, v_final, op=MPI.SUM)

end_time = MPI.Wtime()

if rank == 0:
    print(f"COLONNE - Processus : {size} | Temps : {end_time - start_time:.4f}s")
    # Vérification simple (premier élément)
    print(f"v[0] = {v_final[0]}")