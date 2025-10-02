import numpy as np

def leer_tsplib(ruta_archivo):
    """
    Lee un archivo .tsp de TSPLIB y devuelve un arreglo numpy con coordenadas [(x,y), ...]
    """
    ciudades = []
    with open(ruta_archivo, 'r') as f:
        start = False
        for linea in f:
            linea = linea.strip()
            if linea == "NODE_COORD_SECTION":
                start = True
                continue
            if linea == "EOF":
                break
            if start:
                partes = linea.split()
                if len(partes) >= 3:
                    x, y = float(partes[1]), float(partes[2])
                    ciudades.append((x, y))
    return np.array(ciudades)
