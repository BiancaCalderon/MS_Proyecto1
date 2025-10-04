import os, csv, matplotlib.pyplot as plt
from tsplib import leer_tsplib, build_distance_matrix
from lp_solver import construir_y_resolver_mtz_dist

INSTANCIAS = {
    "eil101": "data/eil101.tsp",
    "gr229":  "data/gr229.tsp",
    "inventado": "data/inventado.tsp",
}
RESULTS_DIR = "results"

def correr_lp(time_limit=3600, msg=False):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outcsv = os.path.join(RESULTS_DIR, "lp_resultados.csv")
    with open(outcsv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instancia","status","objetivo","tiempo_seg","n_vars","n_constraints"])
        for nombre, archivo in INSTANCIAS.items():
            ts = leer_tsplib(archivo)
            D = build_distance_matrix(ts)
            res = construir_y_resolver_mtz_dist(D, msg=msg, time_limit_seconds=time_limit)
            w.writerow([nombre, res["status"], res["objective"], res["time"], res["n_vars"], res["n_constraints"]])

            # PNG de la ruta
            if res.get("route"):
                ruta = res["route"] + [res["route"][0]]
                xs = [ts["coords"][i][0] for i in ruta]
                ys = [ts["coords"][i][1] for i in ruta]
                plt.figure(figsize=(6,6)); plt.plot(xs, ys, "-o", markersize=3)
                plt.title(f"Ruta LP MTZ - {nombre} ({res['status']})")
                plt.savefig(os.path.join(RESULTS_DIR, f"LP_{nombre}_ruta.png"), bbox_inches="tight")
                plt.close()
    print("LP terminado ->", outcsv)

if __name__ == "__main__":
    correr_lp()
