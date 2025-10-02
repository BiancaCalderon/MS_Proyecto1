import matplotlib.pyplot as plt
from genetico import algoritmo_genetico
from tsplib import leer_tsplib   # 👈 ahora importamos desde tsplib

if __name__ == "__main__":
    # Elige instancia: "eil101" o "gr229"
    instancia = "eil101"
    archivo = f"data/{instancia}.tsp"

    ciudades = leer_tsplib(archivo)

    mejor_ruta, mejor_distancia, historial = algoritmo_genetico(
        ciudades, n_poblacion=300, n_iter=1000, prob_mut=0.2
    )

    print(f"Instancia: {instancia}")
    print("Número de ciudades:", len(ciudades))
    print("Mejor distancia encontrada:", mejor_distancia)

    # Graficar la mejor ruta
    ruta_coords = [ciudades[i] for i in mejor_ruta] + [ciudades[mejor_ruta[0]]]
    plt.figure(figsize=(6,6))
    plt.plot([c[0] for c in ruta_coords], [c[1] for c in ruta_coords], '-o')
    plt.title(f"Mejor ruta - {instancia}")
    plt.show()

    # Graficar evolución
    plt.figure()
    plt.plot(historial)
    plt.title(f"Convergencia - {instancia}")
    plt.xlabel("Iteración")
    plt.ylabel("Distancia")
    plt.show()
