import time
import csv
import os
import matplotlib.pyplot as plt
from genetico import algoritmo_genetico
from tsplib import leer_tsplib

# =====================
# PARAMETROS DE EXPERIMENTO
# =====================
INSTANCIAS = {
    "eil101": "data/eil101.tsp",
    "gr229": "data/gr229.tsp",
    "inventado": "data/inventado.tsp"
}

N_RUNS = 10  # número de corridas por instancia
RESULTS_DIR = "results"

# Parámetros del GA para cada instancia
PARAMS = {
    "eil101": dict(n_poblacion=300, n_iter=1000, porc_elite=0.02, porc_cruce=0.7, prob_mut=0.2, selec_method="torneo"),
    "gr229": dict(n_poblacion=500, n_iter=1500, porc_elite=0.02, porc_cruce=0.7, prob_mut=0.2, selec_method="torneo"),
    "inventado": dict(n_poblacion=200, n_iter=800, porc_elite=0.02, porc_cruce=0.7, prob_mut=0.2, selec_method="torneo")
}

# =====================
# FUNCION DE EXPERIMENTO
# =====================
def correr_experimento():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "ga_resultados.csv"), "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["instancia","run","n_poblacion","n_iter","mejor_distancia","tiempo_seg"])
        
        for nombre, archivo in INSTANCIAS.items():
            print(f"\n=== Ejecutando GA en instancia {nombre} ===")
            ciudades = leer_tsplib(archivo)
            params = PARAMS[nombre]
            
            for run in range(1, N_RUNS+1):
                t0 = time.time()
                mejor_ruta, mejor_dist, historial = algoritmo_genetico(
                    ciudades, return_all=False, **params
                )
                t1 = time.time()
                tiempo = t1 - t0
                print(f"Run {run}: distancia={mejor_dist:.2f}, tiempo={tiempo:.2f}s")
                
                # Guardar en CSV
                writer.writerow([nombre, run, params["n_poblacion"], params["n_iter"], mejor_dist, tiempo])
                
                # Guardar gráficas individuales de convergencia
                plt.figure()
                plt.plot(historial)
                plt.title(f"Convergencia GA - {nombre} (Run {run})")
                plt.xlabel("Iteración")
                plt.ylabel("Distancia")
                plt.savefig(os.path.join(RESULTS_DIR, f"GA_{nombre}_run{run}_convergencia.png"))
                plt.close()
                
                # Guardar la mejor ruta encontrada
                ruta_coords = [ciudades[i] for i in mejor_ruta] + [ciudades[mejor_ruta[0]]]
                plt.figure(figsize=(6,6))
                plt.plot([c[0] for c in ruta_coords], [c[1] for c in ruta_coords], '-o')
                plt.title(f"Mejor ruta GA - {nombre} (Run {run})")
                plt.savefig(os.path.join(RESULTS_DIR, f"GA_{nombre}_run{run}_ruta.png"))
                plt.close()

if __name__ == "__main__":
    correr_experimento()
