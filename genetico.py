# genetico.py
import numpy as np
import random
import time

# =====================
# FUNCIONES AUXILIARES
# =====================

def distancia(ciudad1, ciudad2):
    return np.sqrt((ciudad1[0] - ciudad2[0])**2 + (ciudad1[1] - ciudad2[1])**2)

def calcular_distancia_total(ruta, ciudades=None, dist_matrix=None):
    if dist_matrix is not None:
        total = 0.0
        n = len(ruta)
        for i in range(n):
            a = ruta[i]
            b = ruta[(i+1) % n]
            total += dist_matrix[a, b]
        return total
    total = 0.0
    for i in range(len(ruta)):
        a = ciudades[ruta[i]]
        b = ciudades[ruta[(i+1) % len(ruta)]]
        total += ((a[0]-b[0])**2 + (a[1]-b[1])**2) ** 0.5
    return total


# =====================
# POBLACIÓN / OPERADORES
# =====================

def crear_poblacion(n_ciudades, n_poblacion, seed=None):
    if seed is not None:
        np.random.seed(seed)
    poblacion = [list(np.random.permutation(n_ciudades)) for _ in range(n_poblacion)]
    return poblacion

def seleccion_torneo(poblacion, fitness, k=3):
    candidatos = random.sample(list(zip(poblacion, fitness)), k)
    candidatos.sort(key=lambda x: x[1])
    return candidatos[0][0][:]  # retornar copia

def seleccion_ruleta(poblacion, fitness):
    # fitness: menor es mejor -> convertir en probabilidades inversas
    vals = np.array(fitness, dtype=float)
    # evitar division por cero
    vals = vals + 1e-8
    inv = 1.0 / vals
    probs = inv / inv.sum()
    idx = np.random.choice(len(poblacion), p=probs)
    return poblacion[idx][:]

def cruce_OX(padre1, padre2):
    n = len(padre1)
    a, b = sorted(random.sample(range(n), 2))
    hijo = [None]*n
    # copiar segmento del padre1
    hijo[a:b+1] = padre1[a:b+1]
    # rellenar con el orden de padre2
    p2_idx = 0
    for i in range(n):
        if hijo[i] is None:
            while padre2[p2_idx] in hijo:
                p2_idx += 1
            hijo[i] = padre2[p2_idx]
    return hijo

def mutacion_swap(ruta):
    i, j = random.sample(range(len(ruta)), 2)
    ruta[i], ruta[j] = ruta[j], ruta[i]
    return ruta

def medir_diversidad(poblacion):
    # porcentaje de individuos únicos (por comparación directa)
    únicos = {tuple(ind) for ind in poblacion}
    return len(únicos) / len(poblacion)

# =====================
# ALGORITMO GENÉTICO (MEJORADO)
# =====================

def algoritmo_genetico(
    ciudades,
    n_poblacion=100,
    n_iter=500,
    porc_elite=0.02,
    porc_cruce=0.7,
    prob_mut=0.2,
    dist_matrix=None,
    selec_method="torneo",  # "torneo" o "ruleta"
    torneo_k=3,
    seed=None,
    return_all=False
):
    """
    Ejecuta un GA para TSP.
    - porc_elite: fracción de población conservada como élite (ej: 0.02)
    - porc_cruce: fracción de población creada por cruce (resto se rellena con mutaciones / elites)
    - prob_mut: probabilidad de mutación aplicada a cada hijo (si se aplica)
    - selec_method: "torneo" o "ruleta"
    - return_all: si True devuelve (mejor_ruta, mejor_dist, historial, historial_diversity, tiempos)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_ciudades = len(ciudades)
    poblacion = crear_poblacion(n_ciudades, n_poblacion, seed=seed)

    elite_size = max(1, int(porc_elite * n_poblacion))
    num_cruce = max(1, int(porc_cruce * n_poblacion))
    # Asegurar no exceder población
    num_cruce = min(num_cruce, n_poblacion - elite_size)

    mejor_ruta = None
    mejor_distancia = float("inf")
    historial = []
    historial_diversity = []
    tiempos = []

    for gen in range(n_iter):
        t0 = time.time()
        fitness = [calcular_distancia_total(ind, ciudades, dist_matrix) for ind in poblacion]
        # ordenar población por fitness ascendente
        pop_fit = sorted(zip(poblacion, fitness), key=lambda x: x[1])
        # actualizar mejor global
        if pop_fit[0][1] < mejor_distancia:
            mejor_distancia = pop_fit[0][1]
            mejor_ruta = pop_fit[0][0][:]
        historial.append(mejor_distancia)
        # diversidad
        div = medir_diversidad([p for p, f in pop_fit])
        historial_diversity.append(div)

        # formar nueva población: conservar élite
        nueva_poblacion = [pf[0][:] for pf in pop_fit[:elite_size]]

        # crear hijos por cruce
        while len(nueva_poblacion) < elite_size + num_cruce:
            # seleccionar padres
            if selec_method == "torneo":
                padre1 = seleccion_torneo(poblacion, fitness, k=torneo_k)
                padre2 = seleccion_torneo(poblacion, fitness, k=torneo_k)
            else:
                padre1 = seleccion_ruleta(poblacion, fitness)
                padre2 = seleccion_ruleta(poblacion, fitness)
            hijo = cruce_OX(padre1, padre2)
            if random.random() < prob_mut:
                hijo = mutacion_swap(hijo)
            nueva_poblacion.append(hijo)

        # rellenar el resto de la población: mutaciones de elites o permutaciones aleatorias
        while len(nueva_poblacion) < n_poblacion:
            if random.random() < 0.5:
                # mutar una elite
                base = random.choice(nueva_poblacion[:max(1, len(nueva_poblacion))])
                hijo = base[:]
                if random.random() < prob_mut:
                    hijo = mutacion_swap(hijo)
                nueva_poblacion.append(hijo)
            else:
                # individuo aleatorio
                nueva_poblacion.append(list(np.random.permutation(n_ciudades)))

        poblacion = nueva_poblacion[:n_poblacion]
        tiempos.append(time.time() - t0)

    if return_all:
        return mejor_ruta, mejor_distancia, historial, historial_diversity, tiempos
    return mejor_ruta, mejor_distancia, historial

# =====================
# PRUEBA RÁPIDA (si se ejecuta como script)
# =====================
if __name__ == "__main__":
    # pequeño test
    np.random.seed(0)
    random.seed(0)
    ciudades = np.random.rand(20,2) * 100
    best_route, best_dist, hist, div_hist, times = algoritmo_genetico(
        ciudades,
        n_poblacion=100,
        n_iter=200,
        porc_elite=0.02,
        porc_cruce=0.7,
        prob_mut=0.2,
        selec_method="torneo",
        torneo_k=3,
        seed=42,
        return_all=True
    )
    print("Mejor distancia:", best_dist)
    print("Diversidad inicio/fin:", div_hist[0], div_hist[-1])
