from mpi4py import MPI
import random
import time

# Inicialização MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Total de pontos (10 milhões)
nb_points_total = 10000000
nb_points_local = nb_points_total // size

count_local = 0

# Semente aleatória baseada no rank para garantir variação
random.seed(time.time() + rank)

# Início da medição de tempo
start = MPI.Wtime()

# Monte Carlo Local
for _ in range(nb_points_local):
    x = random.uniform(-1.0, 1.0)
    y = random.uniform(-1.0, 1.0)
    
    if x*x + y*y <= 1.0:
        count_local += 1

# Redução: Soma todos os counts locais no processo 0
# O mpi4py detecta automaticamente os tipos, mas op=MPI.SUM é explícito
count_global = comm.reduce(count_local, op=MPI.SUM, root=0)

end = MPI.Wtime()

if rank == 0:
    pi = 4.0 * count_global / nb_points_total
    print(f"--- Resultado Python MPI ---")
    print(f"Processos: {size}")
    print(f"Pi estimé : {pi}")
    print(f"Temps : {end - start:.6f} s")