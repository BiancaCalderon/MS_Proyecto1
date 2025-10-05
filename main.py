import argparse
import os
import time
import matplotlib.pyplot as plt

from genetico import algoritmo_genetico

# Import flexible del lector TSPLIB y la matriz de distancias
from tsplib import leer_tsplib
try:
    from tsplib import build_distance_matrix  # si ya lo implementaste
except ImportError:
    build_distance_matrix = None

# LP: intentar usar versión con dist_matrix, si no, la clásica
LP_DIST_FN = None
LP_PLAIN_FN = None
try:
    from lp_solver import construir_y_resolver_mtz_dist as LP_DIST_FN
except ImportError:
    pass
try:
    from lp_solver import construir_y_resolver_mtz as LP_PLAIN_FN
except ImportError:
    pass


def _load_instance(path):
    """
    Carga la instancia y devuelve:
      coords (Nx2), edge_type (str|None), dist_matrix (o None)
    Soporta tanto el nuevo leer_tsplib (dict) como el antiguo (np.ndarray).
    """
    ts = leer_tsplib(path)
    if isinstance(ts, dict):
        coords = ts.get("coords")
        edge_type = (ts.get("edge_type") or "EUC_2D").upper()
    else:
        coords = ts
        edge_type = "EUC_2D"

    D = None
    if build_distance_matrix is not None:
        try:
            D = build_distance_matrix({"coords": coords, "edge_type": edge_type})
        except Exception:
            D = None
    return coords, edge_type, D


def run_ga(coords, D, args):
    """
    Ejecuta GA intentando primero con dist_matrix; si falla, cae a coords.
    Devuelve: (mejor_ruta, mejor_dist, historial, diversity_or_None)
    """
    ga_kwargs = dict(
        n_poblacion=args.n_poblacion,
        n_iter=args.n_iter,
        porc_elite=args.porc_elite,
        porc_cruce=args.porc_cruce,
        prob_mut=args.prob_mut,
        selec_method=args.selec_method,
        torneo_k=args.torneo_k,
        seed=args.seed,
        porc_mut=args.porc_mut,     # 👈 pasar al GA
    )

    # Intento 1: API nueva con dist_matrix + return_all
    try:
        start = time.time()
        res = algoritmo_genetico(
            ciudades=None,
            dist_matrix=D,
            return_all=True,
            **ga_kwargs,
        )
        elapsed = time.time() - start
        if len(res) == 5:
            mejor_ruta, mejor_dist, historial, diversidad, _ = res
        else:
            mejor_ruta, mejor_dist, historial = res
            diversidad = None
        return mejor_ruta, mejor_dist, historial, diversidad, elapsed
    except TypeError:
        # Intento 2: API antigua (solo coords, sin dist_matrix)
        start = time.time()
        res = algoritmo_genetico(coords, **ga_kwargs)
        elapsed = time.time() - start
        if isinstance(res, tuple) and len(res) >= 3:
            mejor_ruta, mejor_dist, historial = res[:3]
        else:
            mejor_ruta, mejor_dist, historial = res, None, None
        diversidad = None
        return mejor_ruta, mejor_dist, historial, diversidad, elapsed


def plot_route(coords, route, title, outpath=None):
    ruta_coords = [coords[i] for i in route] + [coords[route[0]]]
    plt.figure(figsize=(6, 6))
    plt.plot([c[0] for c in ruta_coords], [c[1] for c in ruta_coords], "-o")
    plt.title(title)
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_series(values, title, xlabel, ylabel, outpath=None):
    plt.figure()
    plt.plot(values)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def run_lp(coords, D, args):
    """
    Intenta resolver por LP usando dist_matrix; si no existe esa función,
    cae a la versión que reconstruye dist internamente con coords.
    """
    if LP_DIST_FN is not None and D is not None:
        res = LP_DIST_FN(D, msg=args.lp_msg, time_limit_seconds=args.time_limit)
    elif LP_PLAIN_FN is not None:
        res = LP_PLAIN_FN(coords, msg=args.lp_msg, time_limit_seconds=args.time_limit)
    else:
        raise RuntimeError("No encuentro funciones LP válidas en lp_solver.py")
    return res


def main():
    parser = argparse.ArgumentParser(description="TSP Proyecto: GA / LP (MTZ)")
    parser.add_argument("--metodo", choices=["ga", "lp", "ambos"], default="ga")
    parser.add_argument("--instancia", choices=["eil101", "gr229", "inventado"], default="eil101")
    parser.add_argument("--base", default="data", help="Carpeta base de instancias .tsp")
    parser.add_argument("--time_limit", type=int, default=3600, help="Límite de tiempo LP (s)")
    parser.add_argument("--lp-msg", action="store_true", help="Mensajes del solver LP")
    parser.add_argument("--porc_mut", type=float, default=None, help="fracción de población creada por mutación pura")
    # Hiperparámetros GA
    parser.add_argument("--n_poblacion", type=int, default=300)
    parser.add_argument("--n_iter", type=int, default=1000)
    parser.add_argument("--porc_elite", type=float, default=0.02)
    parser.add_argument("--porc_cruce", type=float, default=0.7)
    parser.add_argument("--prob_mut", type=float, default=0.2)
    parser.add_argument("--selec_method", choices=["torneo", "ruleta"], default="torneo")
    parser.add_argument("--torneo_k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--save-results", action="store_true",
                        help="Si se especifica, guarda PNGs en results/ en lugar de mostrar")
    args = parser.parse_args()

    instancia = args.instancia
    archivo = os.path.join(args.base, f"{instancia}.tsp")

    # Cargar instancia
    coords, edge_type, D = _load_instance(archivo)

    print(f"Instancia: {instancia}")
    print(f"Número de ciudades: {len(coords)}")
    print(f"EDGE_WEIGHT_TYPE: {edge_type}")
    print(f"Distancia: {'matriz GEO/EUC_2D OK' if D is not None else 'SIN matriz (modo coords)'}")

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # GA
    if args.metodo in ("ga", "ambos"):
        mejor_ruta, mejor_dist, historial, diversidad, t_ga = run_ga(coords, D, args)
        print(f"[GA] Mejor distancia: {mejor_dist:.6f}  | tiempo: {t_ga:.2f}s")

        # Graficar
        if mejor_ruta is not None:
            title = f"Mejor ruta GA - {instancia}"
            out = os.path.join(results_dir, f"GA_{instancia}_ruta.png") if args.save_results else None
            plot_route(coords, mejor_ruta, title, out)

        if historial is not None:
            title = f"Convergencia GA - {instancia}"
            out = os.path.join(results_dir, f"GA_{instancia}_convergencia.png") if args.save_results else None
            plot_series(historial, title, "Iteración", "Distancia", out)

        if diversidad is not None:
            title = f"Diversidad GA - {instancia}"
            out = os.path.join(results_dir, f"GA_{instancia}_diversidad.png") if args.save_results else None
            plot_series(diversidad, title, "Iteración", "Diversidad", out)

    # LP
    if args.metodo in ("lp", "ambos"):
        res = run_lp(coords, D, args)
        print(f"[LP] Status: {res['status']} | Objetivo: {res['objective']:.6f} | "
              f"Tiempo: {res['time']:.2f}s | Vars: {res['n_vars']} | Restricciones: {res['n_constraints']}")
        if res.get("route"):
            title = f"Ruta LP MTZ - {instancia} ({res['status']})"
            out = os.path.join(results_dir, f"LP_{instancia}_ruta.png") if args.save_results else None
            plot_route(coords, res["route"], title, out)


if __name__ == "__main__":
    main()
