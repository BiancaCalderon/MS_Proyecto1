# experimento_ga.py
import time, csv, os, glob
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio  # opcional para GIF
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

from genetico import algoritmo_genetico
from tsplib import leer_tsplib, build_distance_matrix

INSTANCIAS = {
    "eil101": "data/eil101.tsp",
    "gr229": "data/gr229.tsp",
    "inventado": "data/inventado.tsp",
}

N_RUNS = 10
RESULTS_DIR = "results"

# snapshots / gif
SNAPSHOT_EVERY = 50   
SNAPSHOT_RUN = 1       
PARAMS = {
    "eil101":   dict(n_poblacion=300, n_iter=1000, porc_elite=0.02, porc_cruce=0.68, porc_mut=0.30, prob_mut=0.2, selec_method="torneo"),
    "gr229":    dict(n_poblacion=500, n_iter=1500, porc_elite=0.02, porc_cruce=0.68, porc_mut=0.30, prob_mut=0.2, selec_method="torneo"),
    "inventado":dict(n_poblacion=200, n_iter=800,  porc_elite=0.02, porc_cruce=0.68, porc_mut=0.30, prob_mut=0.2, selec_method="torneo"),
}

def _get_coords(tsobj):
    return tsobj["coords"] if isinstance(tsobj, dict) else tsobj

def _plot_route(coords, route, title, outpath):
    ruta_coords = [coords[i] for i in (route + [route[0]])]
    xs = [c[0] for c in ruta_coords]; ys = [c[1] for c in ruta_coords]
    plt.figure(figsize=(6,6))
    plt.plot(xs, ys, "-o", markersize=2)
    plt.title(title)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def _make_gif(frame_glob, gif_path, fps=GIF_FPS):
    if not HAS_IMAGEIO:
        return False
    frames = sorted(glob.glob(frame_glob), key=lambda p: (len(p), p))
    if not frames:
        return False
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for p in frames:
            writer.append_data(imageio.imread(p))
    return True

def correr_experimento():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    outcsv = os.path.join(RESULTS_DIR, "ga_resultados.csv")

    with open(outcsv, "w", newline="") as csvfile:
        w = csv.writer(csvfile)
        w.writerow([
            "instancia","run","seed","n_poblacion","n_iter","porc_elite","porc_cruce","porc_mut",
            "prob_mut","selec_method","mejor_distancia","tiempo_seg"
        ])

        for nombre, archivo in INSTANCIAS.items():
            print(f"\n=== Ejecutando GA en instancia {nombre} ===")
            ts = leer_tsplib(archivo)
            coords = _get_coords(ts)
            try:
                D = build_distance_matrix(ts if isinstance(ts, dict) else {"coords": coords, "edge_type": "EUC_2D"})
            except Exception:
                D = None

            params = PARAMS[nombre]

            for run in range(1, N_RUNS + 1):
                seed = run
                t0 = time.time()

                # --- callback para snapshots (solo en run 1) ---
                def on_gen(gen, best_route, best_dist):
                    if run != SNAPSHOT_RUN or best_route is None:
                        return
                    if gen % SNAPSHOT_EVERY == 0 or gen == params["n_iter"] - 1:
                        snap_path = os.path.join(RESULTS_DIR, f"GA_{nombre}_run{run}_iter{gen:04d}_ruta.png")
                        _plot_route(coords, best_route, f"GA {nombre} - iter {gen} (best={best_dist:.2f})", snap_path)

                # --- ejecutar GA ---
                try:
                    res = algoritmo_genetico(
                        ciudades=None if D is not None else coords,
                        dist_matrix=D,
                        return_all=True,
                        seed=seed,
                        on_generation=on_gen, 
                        **params
                    )
                    if len(res) == 5:
                        mejor_ruta, mejor_dist, historial, div_hist, _ = res
                    else:
                        mejor_ruta, mejor_dist, historial = res[:3]
                        div_hist = None
                except TypeError:
                    # fallback API antigua
                    res = algoritmo_genetico(coords, seed=seed, **params)
                    if isinstance(res, tuple) and len(res) >= 3:
                        mejor_ruta, mejor_dist, historial = res[:3]
                    else:
                        mejor_ruta, mejor_dist, historial = res, None, None
                    div_hist = None

                tiempo = time.time() - t0
                print(f"Run {run}: distancia={mejor_dist:.6f}, tiempo={tiempo:.2f}s")

                # CSV
                w.writerow([
                    nombre, run, seed,
                    params["n_poblacion"], params["n_iter"], params["porc_elite"], params["porc_cruce"], params.get("porc_mut"),
                    params["prob_mut"], params["selec_method"],
                    mejor_dist, tiempo
                ])

                # Convergencia
                if historial is not None:
                    plt.figure(); plt.plot(historial)
                    plt.title(f"Convergencia GA - {nombre} (Run {run})")
                    plt.xlabel("Iteración"); plt.ylabel("Distancia")
                    plt.savefig(os.path.join(RESULTS_DIR, f"GA_{nombre}_run{run}_convergencia.png"), bbox_inches="tight")
                    plt.close()

                # Diversidad
                if div_hist is not None:
                    plt.figure(); plt.plot(div_hist)
                    plt.title(f"Diversidad GA - {nombre} (Run {run})")
                    plt.xlabel("Iteración"); plt.ylabel("Ind únicos / población")
                    plt.savefig(os.path.join(RESULTS_DIR, f"GA_{nombre}_run{run}_diversidad.png"), bbox_inches="tight")
                    plt.close()

                # Mejor ruta final
                if mejor_ruta is not None:
                    _plot_route(coords, mejor_ruta, f"Mejor ruta GA - {nombre} (Run {run})",
                                os.path.join(RESULTS_DIR, f"GA_{nombre}_run{run}_ruta.png"))

            # Al terminar la instancia, arma GIF de la corrida 1 (si hay frames)
            gif_ok = _make_gif(
                os.path.join(RESULTS_DIR, f"GA_{nombre}_run{SNAPSHOT_RUN}_iter*.png"),
                os.path.join(RESULTS_DIR, f"GA_{nombre}_evolucion.gif"),
                fps=GIF_FPS
            )
            if gif_ok:
                print(f"[GIF] Evolución de ruta: {os.path.join(RESULTS_DIR, f'GA_{nombre}_evolucion.gif')}")

    print("GA terminado ->", outcsv)

if __name__ == "__main__":
    correr_experimento()
