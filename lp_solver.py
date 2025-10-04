# lp_solver.py
import math
import argparse
from tsplib import leer_tsplib
import time, pulp
def construir_y_resolver_mtz_dist(dist_matrix, msg=False, time_limit_seconds=None):
    n = dist_matrix.shape[0]
    prob = pulp.LpProblem("TSP_MTZ", pulp.LpMinimize)
    x = {(i,j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
         for i in range(n) for j in range(n) if i != j}
    u = {i: pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n-1, cat="Continuous")
         for i in range(1, n)}

    prob += pulp.lpSum(dist_matrix[i, j] * x[(i,j)] for (i,j) in x)

    for i in range(n):
        prob += pulp.lpSum(x[(i,j)] for j in range(n) if j != i) == 1
    for j in range(n):
        prob += pulp.lpSum(x[(i,j)] for i in range(n) if i != j) == 1

    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + (n-1) * x[(i,j)] <= n-2

    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=int(time_limit_seconds) if time_limit_seconds else None)
    t0 = time.time(); prob.solve(solver); t1 = time.time()

    status = pulp.LpStatus.get(prob.status, "Unknown")
    obj = pulp.value(prob.objective)
    n_vars, n_constraints = len(prob.variables()), len(prob.constraints)

    # reconstruir ruta
    succ = {}
    for (i,j), var in x.items():
        if var.varValue is not None and round(var.varValue) == 1:
            succ[i] = j
    route = []
    if len(succ) == n:
        cur, seen = 0, set()
        for _ in range(n):
            route.append(cur); seen.add(cur)
            cur = succ.get(cur)
            if cur is None or cur in seen: break
        route = route[:n]

    return {"status": status, "objective": obj, "route": route,
            "n_vars": n_vars, "n_constraints": n_constraints, "time": t1 - t0}

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
