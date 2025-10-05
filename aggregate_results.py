# aggregate_results.py
import os
import math
import pandas as pd
import matplotlib.pyplot as plt

# ==== Config ====
RESULTS_DIR = "results"
GA_CSV = os.path.join(RESULTS_DIR, "ga_resultados.csv")
LP_CSV = os.path.join(RESULTS_DIR, "lp_resultados.csv")
OUT_CSV = os.path.join(RESULTS_DIR, "comparativo_ga_lp.csv")

# Mapeo requerido por la rúbrica
N_CIUDADES = {"eil101": 101, "gr229": 229, "inventado": 80}

# ==== Cargas básicas ====
assert os.path.exists(GA_CSV), f"No existe {GA_CSV}"
assert os.path.exists(LP_CSV), f"No existe {LP_CSV}"

ga = pd.read_csv(GA_CSV)
lp = pd.read_csv(LP_CSV)

# Normalizaciones de tipos
for col in ["mejor_distancia", "tiempo_seg"]:
    if col in ga.columns:
        ga[col] = pd.to_numeric(ga[col], errors="coerce")

for col in ["objetivo", "tiempo_seg"]:
    if col in lp.columns:
        lp[col] = pd.to_numeric(lp[col], errors="coerce")

# ==== Top-3 GA por instancia ====
ga_top = (
    ga.sort_values(["instancia", "mejor_distancia"])
      .groupby("instancia")
      .head(3)
      .reset_index(drop=True)
)
ga_top["rank"] = ga_top.groupby("instancia")["mejor_distancia"].rank(method="first").astype(int)
ga_top["metodo"] = ga_top["rank"].map({1: "GA_top1", 2: "GA_top2", 3: "GA_top3"})

# ==== LP mejor por instancia ====
lp_best = (
    lp.sort_values(["instancia", "objetivo"])
      .groupby("instancia")
      .head(1)
      .reset_index(drop=True)
)
lp_best["metodo"] = "LP"

# ==== Gap % de cada GA_top frente a LP ====
lp_map = lp_best.set_index("instancia")["objetivo"].to_dict()
def _gap(row):
    inst = row["instancia"]
    if inst not in lp_map or pd.isna(lp_map[inst]) or lp_map[inst] == 0:
        return float("nan")
    return 100.0 * (row["mejor_distancia"] - lp_map[inst]) / lp_map[inst]

ga_top["gap_pct_vs_LP"] = ga_top.apply(_gap, axis=1)

# ==== Armar tabla final con n_ciudades ====
ga_out = ga_top[["instancia", "metodo", "mejor_distancia", "tiempo_seg", "n_poblacion", "n_iter", "gap_pct_vs_LP"]].rename(
    columns={"mejor_distancia": "distancia", "tiempo_seg": "tiempo"}
)
ga_out.insert(1, "n_ciudades", ga_out["instancia"].map(N_CIUDADES))  # ← NUEVO

lp_out = lp_best[["instancia", "metodo", "objetivo", "tiempo_seg", "n_vars", "n_constraints"]].rename(
    columns={"objetivo": "distancia", "tiempo_seg": "tiempo"}
)
lp_out.insert(1, "n_ciudades", lp_out["instancia"].map(N_CIUDADES))  # ← NUEVO

final = pd.concat([ga_out, lp_out], ignore_index=True)
final.to_csv(OUT_CSV, index=False)
print("Escribí", OUT_CSV)

alertas = []
for inst, sub in ga_top.groupby("instancia"):
    lp_val = lp_map.get(inst, None)
    if lp_val is None or pd.isna(lp_val):
        continue
    min_gap = sub["gap_pct_vs_LP"].min()
    if pd.notna(min_gap) and min_gap < -5.0:
        alertas.append(f"[AVISO] Gaps negativos grandes en {inst} (min {min_gap:.2f}%). "
                       f"Verifica que el GA use la misma EDGE_WEIGHT_TYPE que LP (e.g., GEO en gr229).")
if alertas:
    print("\n".join(alertas))

# ==== Gráfica de tiempos: promedio GA vs LP ====
ga_time_mean = ga.groupby("instancia")["tiempo_seg"].mean().rename("tiempo_ga_prom")
lp_time = lp_best.set_index("instancia")["tiempo"].rename("tiempo_lp")
merged = pd.concat([ga_time_mean, lp_time], axis=1)

ax = merged.plot(kind="bar", figsize=(7, 4))
ax.set_title("Tiempo promedio GA vs LP por instancia")
ax.set_ylabel("segundos")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "tiempos_ga_vs_lp.png"))
plt.close()

# ==== Gráfica de gaps: GA_top1 vs LP ====
ga1 = ga_top[ga_top["metodo"] == "GA_top1"].set_index("instancia")
ax = ga1["gap_pct_vs_LP"].plot(kind="bar", figsize=(7, 4))
ax.set_title("Gap % GA_top1 vs LP por instancia")
ax.set_ylabel("%")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "gap_ga_top1_vs_lp.png"))
plt.close()
