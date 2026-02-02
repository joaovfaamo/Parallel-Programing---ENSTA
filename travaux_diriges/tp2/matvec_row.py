
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 4096 

# 1. Calcul de Nloc (Lignes par processus)
if N % size != 0:
    if rank == 0: 
        print("Erreur : N doit être divisible par le nombre de processus.")
    comm.Abort()

N_loc = N // size

# 2. Initialisation Locale de la Matrice A (Seulement les lignes détenues)
# Chaque processus crée une matrice N_loc x N
# L'indice global de la ligne 'i' est : i_local + (rank * N_loc)
A_local = np.empty((N_loc, N), dtype=np.float64)
offset_row = rank * N_loc

for i_local in range(N_loc):
    i_global = i_local + offset_row
    for j in range(N):
        A_local[i_local, j] = (i_global + j) % N + 1.0

# 3. Distribution du vecteur u
u = np.empty(N, dtype=np.float64)
if rank == 0:
    u[:] = np.array([i + 1.0 for i in range(N)], dtype=np.float64)

# Début de la mesure du temps
comm.Barrier()
start_time = MPI.Wtime()

# Broadcast : Tout le monde a besoin du vecteur u ENTIER
comm.Bcast(u, root=0)

# 4. Produit Matrice-Vecteur Local
# (N_loc x N) dot (N) -> Résulte en un vecteur de taille N_loc
v_local_slice = np.dot(A_local, u)

# 5. Gather (Rassemblement) des résultats
# Nous rassemblons les morceaux (N_loc) de chacun pour former le vecteur final (N)
v_final = np.empty(N, dtype=np.float64)
comm.Allgather(v_local_slice, v_final)

end_time = MPI.Wtime()

if rank == 0:
    print(f"LIGNE   - Processus : {size} | Temps : {end_time - start_time:.4f}s")
    print(f"v[0] = {v_final[0]}")