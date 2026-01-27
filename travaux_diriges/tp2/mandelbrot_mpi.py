import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

# --- Inicialização do MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class MandelbrotSet:
    def __init__(self, max_iterations: int, escape_radius: float = 2.):
        self.max_iterations = max_iterations
        self.escape_radius = escape_radius

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: np.ndarray, smooth=False, clamp=True) -> np.ndarray:
        value = self.count_iterations(c, smooth) / self.max_iterations
        return np.maximum(0.0, np.minimum(value, 1.0)) if clamp else value

    def count_iterations(self, c: np.ndarray, smooth=False) -> np.ndarray:
        # Inicializa arrays
        iter_counts = self.max_iterations * np.ones(c.shape, dtype=np.double)
        mask = (np.abs(c) >= 0.25) | (np.abs(c + 1.) >= 0.25)
        
        z = np.zeros(c.shape, dtype=np.complex128)
        
        # Loop principal de iterações (vetorizado)
        for it in range(self.max_iterations):
            z[mask] = z[mask] * z[mask] + c[mask]
            
            # Identifica quem divergiu nesta iteração
            has_diverged = np.abs(z) > self.escape_radius
            
            # Atualiza contagem para os que divergiram agora
            if has_diverged.size > 0:
                # Onde divergiu, salvamos o número da iteração atual
                # A lógica original do código vetorizado usa uma máscara complexa,
                # aqui simplificamos para atualizar iter_counts onde a máscara ainda é True
                current_diverged = has_diverged & mask
                iter_counts[current_diverged] = it
                
                # Remove os divergentes da máscara de cálculo
                mask = mask & ~has_diverged
            
            if not np.any(mask):
                break
                
        # Suavização (Smooth coloring)
        if smooth:
            # Recalcula quem divergiu no final para aplicar log
            has_diverged_final = np.abs(z) > 2
            # Evita log de zero ou valores inválidos filtrando indices
            valid_mask = has_diverged_final & (iter_counts < self.max_iterations)
            if np.any(valid_mask):
                iter_counts[valid_mask] += 1 - np.log(np.log(np.abs(z[valid_mask]))) / log(2)
                
        return iter_counts

# --- Parâmetros ---
mandelbrot_set = MandelbrotSet(max_iterations=200, escape_radius=2.)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

# --- Passo 1: Divisão do Trabalho ---
# Dividimos o eixo Y (altura) entre os processos
lines_per_process = height // size
start_y = rank * lines_per_process
end_y = (rank + 1) * lines_per_process if rank != size - 1 else height
local_height = end_y - start_y

# Array local para armazenar o pedaço da imagem deste processo
# O shape é (width, local_height) porque o código original preenche convergence[:, y]
local_convergence = np.empty((width, local_height), dtype=np.double)

# Início da medição de tempo
start_time = MPI.Wtime()

# --- Cálculo Local ---
for i, global_y in enumerate(range(start_y, end_y)):
    # Gera a linha de números complexos para todo o eixo X nesta altura Y
    c = np.array([complex(-2. + scaleX * x, -1.125 + scaleY * global_y) for x in range(width)])
    # Calcula e armazena na linha local 'i'
    local_convergence[:, i] = mandelbrot_set.convergence(c, smooth=True)

end_time = MPI.Wtime()
print(f"Processo {rank}: calculou linhas {start_y} a {end_y} em {end_time - start_time:.4f}s")

# --- Passo 2: Reunião dos Resultados (Gather) ---
# O gather vai retornar uma lista de arrays numpy no processo 0
all_blocks = comm.gather(local_convergence, root=0)

if rank == 0:
    # --- Passo 3: Montagem da Imagem ---
    # Concatenamos os blocos ao longo do eixo 1 (altura)
    final_convergence = np.concatenate(all_blocks, axis=1)
    
    total_time = MPI.Wtime() - start_time
    print(f"\n--- Tempo Total MPI ({size} processos): {total_time:.4f}s ---")
    
    # Salva a imagem
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_convergence.T) * 255))
    image.save("mandelbrot.png")
    print("Imagem salva como 'mandelbrot.png'")