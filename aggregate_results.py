# aggregate_results.py
import os, math, pandas as pd, matplotlib.pyplot as plt

RESULTS_DIR = "results"
GA_CSV = os.path.join(RESULTS_DIR, "ga_resultados.csv")
LP_CSV = os.path.join(RESULTS_DIR, "lp_resultados.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "comparativo_ga_lp.csv")

assert os.path.exists(GA_CSV), f"No existe {GA_CSV}"
assert os.path.exists(LP_CSV), f"No existe {LP_CSV}"

ga = pd.read_csv(GA_CSV)
lp = pd.read_csv(LP_CSV)

# Top-3 por instancia (mejor distancia primero)
ga_top = (ga.sort_values(["instancia","mejor_distancia"])
            .groupby("instancia")
            .head(3)
            .reset_index(drop=True))
ga_top["rank"] = ga_top.groupby("instancia")["mejor_distancia"].rank(method="first").astype(int)
ga_top["metodo"] = ga_top["rank"].map({1:"GA_top1",2:"GA_top2",3:"GA_top3"})

# LP mejor por instancia
lp_best = (lp.sort_values(["instancia","objetivo"])
             .groupby("instancia")
             .head(1)
             .reset_index(drop=True))
lp_best["metodo"] = "LP"

# Gap % de cada GA_top frente al LP de su instancia
lp_map = lp_best.set_index("instancia")["objetivo"].to_dict()
ga_top["gap_pct_vs_LP"] = ga_top.apply(
    lambda r: 100.0 * (r["mejor_distancia"] - lp_map[r["instancia"]]) / lp_map[r["instancia"]],
    axis=1
)

# Armar tabla final
ga_out = ga_top[["instancia","metodo","mejor_distancia","tiempo_seg","n_poblacion","n_iter","gap_pct_vs_LP"]]
ga_out = ga_out.rename(columns={"mejor_distancia":"distancia","tiempo_seg":"tiempo"})
lp_out = lp_best[["instancia","metodo","objetivo","tiempo_seg","n_vars","n_constraints"]]
lp_out = lp_out.rename(columns={"objetivo":"distancia","tiempo_seg":"tiempo"})

final = pd.concat([ga_out, lp_out], ignore_index=True)
final.to_csv(OUT_CSV, index=False)
print("Escribí", OUT_CSV)

# Gráfica de tiempos (promedio GA vs LP)
ga_time = ga.groupby("instancia")["tiempo_seg"].mean().rename("tiempo_ga_prom")
merged = pd.merge(ga_time, lp_best.set_index("instancia")["tiempo_seg"].rename("tiempo_lp"), left_index=True, right_index=True)

plt.figure(figsize=(7,4))
merged.plot(kind="bar")
plt.title("Tiempo promedio GA vs LP por instancia")
plt.ylabel("segundos")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "tiempos_ga_vs_lp.png"))
plt.close()

# Gráfica de gaps (usa solo GA_top1 vs LP)
ga1 = ga_top[ga_top["metodo"]=="GA_top1"].set_index("instancia")
plt.figure(figsize=(7,4))
ga1["gap_pct_vs_LP"].plot(kind="bar")
plt.title("Gap % GA_top1 vs LP por instancia")
plt.ylabel("%")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gap_ga_top1_vs_lp.png"))
plt.close()
