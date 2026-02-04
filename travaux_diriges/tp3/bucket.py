import numpy as np
from mpi4py import MPI
import time

# --- Configuration MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Paramètres
N = 1000000  
if N % size != 0:
    if rank == 0:
        print("Erreur: N doit être divisible par le nombre de processus.")
    comm.Abort()

# Étape 1 : Génération et Distribution (Process 0 -> Tous)
# Le rang 0 génère des nombres aléatoires uniformes dans [0, 1)
if rank == 0:
    print(f"Génération de {N} nombres aléatoires sur le rang 0...")
    raw_data = np.random.rand(N).astype(np.float64)
else:
    raw_data = None
local_chunk_size = N // size
local_data = np.empty(local_chunk_size, dtype=np.float64)

# Scatter : Distribue les données aléatoirement entre les processus
comm.Scatter(raw_data, local_data, root=0)
comm.Barrier()
start_time = MPI.Wtime()

# Étape 2 : Création des (Buckets) Locaux 
local_buckets = [[] for _ in range(size)]

for value in local_data:
    target_rank = int(value * size)
    if target_rank >= size: target_rank = size - 1 
    local_buckets[target_rank].append(value)

# Étape 3 : Échange Tout-vers-Tout (Alltoall) 
received_buckets = comm.alltoall(local_buckets)

# Étape 4 : Tri Local (Local Sort) ---
my_final_values = []
for bucket in received_buckets:
    my_final_values.extend(bucket)

# 2. Convertir en numpy et trier localement
my_final_values = np.array(my_final_values, dtype=np.float64)
my_final_values.sort()

# Étape 5 : Rassemblement (Gather) 
sorted_parts = comm.gather(my_final_values, root=0)

end_time = MPI.Wtime()

if rank == 0:
    # Concatène tous les morceaux triés
    final_sorted_array = np.concatenate(sorted_parts)
    
    print(f"Temps de calcul (Bucket Sort) : {end_time - start_time:.4f}s")
    
    # Vérifie si le tableau est correctement trié
    is_sorted = np.all(np.diff(final_sorted_array) >= 0)
    print(f"Vérification : Le tableau est trié ? {is_sorted}")
    print(f"Taille finale : {len(final_sorted_array)} (Attendu : {N})")
    print(f"Premières 10 valeurs : {final_sorted_array[:10]}")