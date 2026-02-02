import numpy as np
from dataclasses import dataclass
from PIL import Image
from math import log
from time import time
import matplotlib.cm
from mpi4py import MPI

# Constantes de TAG pour la communication
TAG_WORK = 1 # Message contenant une ligne à calculer
TAG_DATA = 2 # Message contenant le résultat calculé
TAG_STOP = 3 # Message pour arrêter le travailleur

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

# Configuration
mandelbrot_set = MandelbrotSet(max_iterations=50, escape_radius=10)
width, height = 1024, 1024
scaleX = 3./width
scaleY = 2.25/height

# ==========================================
# LOGIQUE MAÎTRE (Rang 0)
# ==========================================
if rank == 0:
    print(f"Démarrage Maître-Esclave avec {size-1} travailleurs.")
    start_time = MPI.Wtime()
    
    final_convergence = np.empty((width, height), dtype=np.double)
    next_row = 0
    active_workers = size - 1 # Rangs 1 jusqu'à size-1

    # Boucle principale du Maître
    while active_workers > 0:
        # Reçoit N'IMPORTE QUEL message de N'IMPORTE QUELLE source
        status = MPI.Status()
        # Nous attendons un dictionnaire {'source': rank, 'row': y, 'result': array}
        data_recv = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source_rank = status.Get_source()
        tag = status.Get_tag()

        # Si nous recevons des données (TAG_DATA), nous les sauvegardons dans la matrice
        if tag == TAG_DATA:
            row_index = data_recv['row']
            result_array = data_recv['result']
            final_convergence[:, row_index] = result_array
        
        # S'il reste des lignes à calculer, nous envoyons la suivante
        if next_row < height:
            comm.send(next_row, dest=source_rank, tag=TAG_WORK)
            next_row += 1
        else:
            # Plus de lignes, nous disons au travailleur de s'arrêter
            comm.send(None, dest=source_rank, tag=TAG_STOP)
            active_workers -= 1

    end_time = MPI.Wtime()
    print(f"Temps total (Maître-Esclave Dynamique) : {end_time - start_time:.4f}s")
    
    # Sauvegarder l'image
    image = Image.fromarray(np.uint8(matplotlib.cm.plasma(final_convergence.T)*255))
    image.save("mandelbrot_dynamic.png")
    print("Image sauvegardée sous 'mandelbrot_dynamic.png'")

# ==========================================
# LOGIQUE ESCLAVE (Rang > 0)
# ==========================================
else:
    # Étape 1 : Informer le maître que nous sommes prêts
    comm.send({'row': None, 'result': None}, dest=0, tag=TAG_WORK)

    while True:
        # Reçoit l'ordre du maître
        status = MPI.Status()
        row_to_calc = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP:
            break # Fin du travail
        
        elif tag == TAG_WORK:
            # Calcule la ligne demandée
            y = row_to_calc
            c = np.array([complex(-2. + scaleX*x, -1.125 + scaleY * y) for x in range(width)])
            row_data = mandelbrot_set.convergence(c, smooth=True)
            
            # Envoie le résultat au maître
            result_packet = {'row': y, 'result': row_data}
            comm.send(result_packet, dest=0, tag=TAG_DATA)