# lp_solver.py
import time
import math
import pulp
import argparse
from tsplib import leer_tsplib

def construir_y_resolver_mtz(ciudades, time_limit_seconds=None, msg=False):
    """
    Construye y resuelve el TSP por la formulación MTZ usando PuLP (CBC por defecto).
    Retorna: dict con keys: 'status','objective','route','n_vars','n_constraints','time'
    """
    n = len(ciudades)
    dist = [[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = ciudades[i][0] - ciudades[j][0]
                dy = ciudades[i][1] - ciudades[j][1]
                dist[i][j] = math.hypot(dx, dy)
            else:
                dist[i][j] = 0.0

    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)

    # variables x_ij binarios (i != j)
    x = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x[(i,j)] = pulp.LpVariable(f"x_{i}_{j}", cat="Binary")

    # variables u_i (MTZ), i = 1..n-1 (u_0 fija a 0)
    u = {}
    for i in range(1, n):
        u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n-1, cat="Continuous")

    # objetivo
    prob += pulp.lpSum(dist[i][j] * x[(i,j)] for (i,j) in x)

    # restricciones: salida == 1, entrada == 1
    for i in range(n):
        prob += pulp.lpSum(x[(i,j)] for j in range(n) if j != i) == 1, f"salida_{i}"
    for j in range(n):
        prob += pulp.lpSum(x[(i,j)] for i in range(n) if i != j) == 1, f"entrada_{j}"

    # MTZ subtour-elimination
    for i in range(1, n):
        for j in range(1, n):
            if i == j:
                continue
            prob += u[i] - u[j] + (n-1) * x[(i,j)] <= n-2, f"mtz_{i}_{j}"

    # resolver
    solver = pulp.PULP_CBC_CMD(msg=msg)
    if time_limit_seconds is not None:
        # CBC accepts timeLimit in seconds in pulp's wrapper (param name timeLimit)
        solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=int(time_limit_seconds))

    t0 = time.time()
    prob.solve(solver)
    t1 = time.time()

    status = pulp.LpStatus.get(prob.status, "Unknown")
    obj = pulp.value(prob.objective)

    # contar variables y restricciones
    n_vars = len(prob.variables())
    n_constraints = len(prob.constraints)

    # reconstruir ruta si solución factible
    route = []
    if status == "Optimal" or status == "Not Solved" or status == "Integer Feasible" or status == "Optimal Solution Found":
        # construir mapa de salida
        succ = {}
        for (i,j), var in x.items():
            val = var.varValue
            if val is not None and round(val) == 1:
                succ[i] = j
        # seguir desde 0
        if len(succ) == n:
            cur = 0
            visited = set()
            for _ in range(n):
                route.append(cur)
                visited.add(cur)
                cur = succ.get(cur, None)
                if cur is None or cur in visited:
                    break
            # si no completó, intentar reconstruir con ciclos
            if len(route) < n:
                # intentar encontrar ciclos y concatenarlos (fallback)
                remaining = set(range(n)) - set(route)
                for start in remaining:
                    cur = start
                    for _ in range(n):
                        route.append(cur)
                        cur = succ.get(cur, None)
                        if cur is None or cur in route:
                            break
                    if len(route) >= n:
                        break
            # recortar a n si sobró
            route = route[:n]
        else:
            route = []

    return {
        "status": status,
        "objective": obj,
        "route": route,
        "n_vars": n_vars,
        "n_constraints": n_constraints,
        "time": t1 - t0
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolver TSP por MTZ (PuLP).")
    parser.add_argument("archivo", type=str, help="ruta al .tsp (TSPLIB) archivo")
    parser.add_argument("--time_limit", type=int, default=None, help="limite de tiempo en segundos para el solver (opcional)")
    parser.add_argument("--msg", action="store_true", help="mostrar mensajes del solver")
    args = parser.parse_args()

    ciudades = leer_tsplib(args.archivo)
    print("Instancia:", args.archivo, "n_ciudades:", len(ciudades))
    res = construir_y_resolver_mtz(ciudades, time_limit_seconds=args.time_limit, msg=args.msg)
    print("Status:", res["status"])
    print("Objetivo (distancia):", res["objective"])
    print("Tiempo (s):", res["time"])
    print("Variables:", res["n_vars"])
    print("Restricciones:", res["n_constraints"])
    if res["route"]:
        print("Ruta (primeros 20 nodos):", res["route"][:20])
    else:
        print("No se pudo reconstruir una ruta completa desde la solución obtenida.")
