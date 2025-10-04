# tsplib.py
import math
import numpy as np

def _geo_to_rad(v: float) -> float:
    # TSPLIB GEO: dd.mm -> deg + minutes/100, fórmula oficial
    deg = int(v)
    min_ = v - deg
    return math.pi * (deg + 5.0 * min_ / 3.0) / 180.0

def leer_tsplib(ruta_archivo: str):
    """
    Retorna dict con:
      name, edge_type ('EUC_2D'|'GEO'), coords (np.ndarray Nx2 en floats tal como vienen)
    """
    name = None
    edge_type = None
    coords = []
    start = False
    with open(ruta_archivo, 'r') as f:
        for linea in f:
            s = linea.strip()
            if s.startswith("NAME"):
                name = s.split(":")[-1].strip()
            if s.startswith("EDGE_WEIGHT_TYPE"):
                edge_type = s.split(":")[-1].strip()
            if s == "NODE_COORD_SECTION":
                start = True
                continue
            if s == "EOF":
                break
            if start:
                parts = s.split()
                if len(parts) >= 3:
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))
    return {"name": name, "edge_type": edge_type, "coords": np.array(coords, dtype=float)}

def build_distance_matrix(tsplib_obj):
    coords = tsplib_obj["coords"]
    et = (tsplib_obj["edge_type"] or "EUC_2D").upper()
    n = len(coords)
    D = np.zeros((n, n), dtype=float)
    if et == "EUC_2D":
        for i in range(n):
            xi, yi = coords[i]
            for j in range(i+1, n):
                xj, yj = coords[j]
                dij = math.hypot(xi - xj, yi - yj)
                D[i, j] = D[j, i] = dij
    elif et == "GEO":
        # Fórmula TSPLIB GEO oficial
        RRR = 6378.388
        lats = np.array([_geo_to_rad(c[0]) for c in coords])
        lons = np.array([_geo_to_rad(c[1]) for c in coords])
        for i in range(n):
            for j in range(i+1, n):
                q1 = math.cos(lons[i] - lons[j])
                q2 = math.cos(lats[i] - lats[j])
                q3 = math.cos(lats[i] + lats[j])
                dij = int(RRR * math.acos(0.5 * ((1+q1)*q2 - (1-q1)*q3)) + 1.0)
                D[i, j] = D[j, i] = float(dij)
    else:
        raise ValueError(f"EDGE_WEIGHT_TYPE no soportado: {et}")
    return D
