import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

# --- Initialisation de MPI ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius:  float = 2.0

    def __contains__(self, c: complex) -> bool:
        return self.stability(c) == 1

    def convergence(self, c: np.ndarray, smooth=False, clamp=True) -> np.ndarray:
        value = self.count_iterations(c, smooth)/self.max_iterations
        return np.maximum(0.0, np.minimum(value, 1.0)) if clamp else value

    def count_iterations(self, c: np.ndarray,  smooth=False) -> np.ndarray:
        iter_counts = self.max_iterations * np.ones(c.shape, dtype=np.double)
        mask = (np.abs(c) >= 0.25) | (np.abs(c+1.) >= 0.25)
        z = np.zeros(c.shape, dtype=np.complex128)
        
        for it in range(self.max_iterations):
            z[mask] = z[mask]*z[mask] + c[mask]
            has_diverged = np.abs(z) > self.escape_radius
            if has_diverged.size > 0:
                iter_counts[has_diverged] = it
                mask = mask & ~has_diverged
            if not np.any(mask):
                break
        
        if smooth:
            has_diverged_final = np.abs(z) > 2
            valid_mask = has_diverged_final & (iter_counts < self.max_iterations)
            if np.any(valid_mask):
                iter_counts[valid_mask] += 1 - np.log(np.log(np.abs(z[valid_mask])))/log(2)
        return iter_counts

# --- Configuration ---
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

# --- Étape 1 : Calcul Cyclique ---
start_time = MPI.Wtime()

# Liste pour stocker les lignes calculées par ce processus
# Chaque élément sera un tuple : (indice_ligne_Y, tableau_de_données)
my_lines = []

# Boucle avec un pas de 'size' : Distribution Cyclique (Round-Robin)
# Ex : Avec 4 processus, le rang 0 traite les lignes 0, 4, 8... le rang 1 traite 1, 5, 9...
for y in range(rank, height, size):
    c = np.array([complex(-2. + scaleX*x, -1.125 + scaleY * y) for x in range(width)])
    data = mandelbrot_set.convergence(c, smooth=True)
    my_lines.append((y, data))

# --- Étape 2 : Rassemblement des résultats ---
# comm.gather retourne une liste où l'élément 'i' est le résultat envoyé par le rang 'i'
all_data_lists = comm.gather(my_lines, root=0)

end_time = MPI.Wtime()

if rank == 0:
    print(f"Temps total de calcul (MPI Cyclique) : {end_time - start_time:.4f}s")
    
    # Reconstruction de l'image
    final_convergence = np.empty((width, height), dtype=np.double)
    
    # all_data_lists est une liste de listes de tuples. Nous devons la déballer.
    for process_data in all_data_lists:     # Pour chaque processus
        for y_index, row_data in process_data: # Pour chaque ligne calculée par ce processus
            final_convergence[:, y_index] = row_data

    # Sauvegarder l'image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_convergence.T)*255))
    image.save("mandelbrot_cyclic.png")
    print("Image sauvegardée sous 'mandelbrot_cyclic.png'")