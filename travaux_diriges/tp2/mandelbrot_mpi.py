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
        value = self.count_iterations(c, smooth) / self.max_iterations
        return np.maximum(0.0, np.minimum(value, 1.0)) if clamp else value

    def count_iterations(self, c: np.ndarray, smooth=False) -> np.ndarray:
        # Initialisation des tableaux
        iter_counts = self.max_iterations * np.ones(c.shape, dtype=np.double)
        mask = (np.abs(c) >= 0.25) | (np.abs(c + 1.) >= 0.25)
        
        z = np.zeros(c.shape, dtype=np.complex128)
        
        # Boucle principale d'itérations (vectorisée)
        for it in range(self.max_iterations):
            z[mask] = z[mask] * z[mask] + c[mask]
            
            # Identifie ceux qui ont divergé à cette itération
            has_diverged = np.abs(z) > self.escape_radius
            
            # Met à jour le compteur pour ceux qui viennent de diverger
            if has_diverged.size > 0:
                current_diverged = has_diverged & mask
                iter_counts[current_diverged] = it
                
                # Retire les points divergents du masque de calcul
                mask = mask & ~has_diverged
            
            if not np.any(mask):
                break
                
        # Lissage (Smooth coloring)
        if smooth:
            has_diverged_final = np.abs(z) > 2
            valid_mask = has_diverged_final & (iter_counts < self.max_iterations)
            if np.any(valid_mask):
                iter_counts[valid_mask] += 1 - np.log(np.log(np.abs(z[valid_mask]))) / log(2)
                
        return iter_counts

# --- Configuration ---
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024

scaleX = 3./width
scaleY = 2.25/height

# --- Étape 1 : Division du travail (Partition par blocs) ---
# Nous divisons l'axe Y (hauteur) entre les processus
lines_per_process = height // size
start_y = rank * lines_per_process
end_y = (rank + 1) * lines_per_process if rank != size - 1 else height
local_height = end_y - start_y

# Tableau local pour stocker la partie de l'image de ce processus
local_convergence = np.empty((width, local_height), dtype=np.double)

# Début de la mesure du temps
start_time = MPI.Wtime()

# --- Calcul Local ---
for i, global_y in enumerate(range(start_y, end_y)):
    # Génère la ligne de nombres complexes pour tout l'axe X à cette hauteur Y
    c = np.array([complex(-2. + scaleX * x, -1.125 + scaleY * global_y) for x in range(width)])
    # Calcule et stocke dans la ligne locale 'i'
    local_convergence[:, i] = mandelbrot_set.convergence(c, smooth=True)

end_time = MPI.Wtime()
print(f"Processus {rank} : a calculé les lignes {start_y} à {end_y} en {end_time - start_time:.4f}s")

# --- Étape 2 : Rassemblement des résultats (Gather) ---
# Le gather retourne une liste de tableaux numpy sur le processus 0
# Comme la partition est par blocs contigus, l'ordre des rangs correspond à l'ordre de l'image
all_blocks = comm.gather(local_convergence, root=0)

if rank == 0:
    # --- Étape 3 : Assemblage de l'image ---
    # Nous concaténons les blocs le long de l'axe 1 (hauteur)
    final_convergence = np.concatenate(all_blocks, axis=1)
    
    total_time = MPI.Wtime() - start_time
    print(f"\n--- Temps Total MPI ({size} processus) : {total_time:.4f}s ---")
    
    # Sauvegarder l'image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_convergence.T) * 255))
    image.save("mandelbrot.png")
    print("Image sauvegardée sous 'mandelbrot.png'")