# Proyecto 1 — Modelación y Simulación 2025

## Algoritmo Genético (GA) + Programación Lineal (LP) para TSP

> **Problema elegido:** Traveling Salesman Problem (TSP) en 3 escenarios (eil101, gr229 y uno inventado).

---

## 1) Resumen del repositorio

```
biancacalderon-ms_proyecto1/
├── experimento_ga.py              # Corridas batch de GA (CSV + gráficas + GIF opcional)
├── experimento_lp.py              # Corridas batch de LP/MTZ (CSV + ruta)
├── genetico.py                    # Implementación del GA (selección, OX, swap, élite, porc_mut)
├── lp_solver.py                   # Modelo MTZ (PuLP/CBC), versión con matriz de distancias
├── main.py                        # CLI unificada: GA | LP | ambos (por instancia)
├── tsplib.py                      # Lector TSPLIB + distancias EUC_2D/GEO (build_distance_matrix)
├── data/
│   ├── eil101.tsp                 # TSPLIB (EUC_2D)
│   ├── gr229.tsp                  # TSPLIB (GEO)
│   ├── inventado.tsp              # Instancia inventada (EUC_2D)
│   └── generar_inventado.py       # Script para generar otra instancia inventada
└── results/                       # Salidas (CSV + PNG + GIF)
```

---

## 2) Requisitos y preparación del entorno

* **Python 3.10+ recomendado**
* Dependencias (archivo `requirements.txt`):

  ```
  numpy>=1.26
  matplotlib>=3.8
  pulp>=2.8
  pandas>=2.2
  imageio>=2.34   # para generar GIF de evolución (opcional)
  ```
* **CBC (solver)**: PuLP usa CBC. En la mayoría de entornos funciona de fábrica; si no, instala COIN-OR CBC desde tu gestor de paquetes del sistema.

**Crear y activar venv** (ejemplo en bash):

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3) Cómo ejecutar

### 3.1. CLI general (main.py)

Ejecuta GA, LP o ambos sobre una instancia:

```bash
# GA sobre eil101, guardando figuras en results/
python main.py --metodo ga --instancia eil101 --save-results

# LP sobre gr229 con límite de 1800s
python main.py --metodo lp --instancia gr229 --time_limit 1800 --save-results

# Ambos sobre inventado con hiperparámetros ajustados
python main.py --metodo ambos --instancia inventado \
  --n_poblacion 200 --n_iter 800 --prob_mut 0.25 --save-results
```

**Parámetros GA relevantes:**

* `--n_poblacion (N)`
* `--n_iter (maxIter)`
* `--porc_elite`, `--porc_cruce`, `--porc_mut` *(fracción de población creada por mutación pura)*
* `--prob_mut` *(probabilidad de mutación aplicada a cada hijo de cruce)*
* `--selec_method torneo|ruleta`, `--torneo_k`
* `--seed` *(reproducibilidad)*

### 3.2. Corridas batch de GA (experimento_ga.py)

Genera 10 corridas por instancia y guarda:

* `results/ga_resultados.csv`
* Curvas de **convergencia**, **diversidad** y **mejor ruta** por *run*.
* **Evolución de la ruta** vía snapshots cada *X* iteraciones y un GIF (si `imageio` disponible).

```bash
python experimento_ga.py
```

Configuración de snapshots (en el propio script):

* `SNAPSHOT_EVERY = 50` (guarda cada 50 iteraciones)
* `SNAPSHOT_RUN = 1` (solo la corrida 1 por instancia)
* produce `GA_<instancia>_evolucion.gif`

### 3.3. Corridas batch de LP (experimento_lp.py)

```bash
python experimento_lp.py   # por defecto time_limit=3600
```

Genera `results/lp_resultados.csv` y las rutas óptimas `LP_<instancia>_ruta.png`.

### 3.4. Agregador y gráficas comparativas (aggregate_results.py)

```bash
python aggregate_results.py
```

Genera:

* `results/comparativo_ga_lp.csv`
* `results/tiempos_ga_vs_lp.png`
* `results/gap_ga_top1_vs_lp.png`

---

## 4) Notas sobre distancias TSPLIB (EUC_2D vs GEO)

* `eil101` e `inventado` usan **EUC_2D** (euclidiana plana).
* `gr229` usa **GEO** (coordenadas geográficas).
* El lector `tsplib.py` expone `build_distance_matrix(...)` para obtener una **misma métrica** consistente para GA y LP.
* **Importante**: mezclar EUC y GEO en la misma instancia produce **gaps inválidos** en el comparativo.

---

## 5) Salidas generadas (para el informe)

### 5.1. CSVs

* **`results/ga_resultados.csv`** (por corrida):

  * `instancia, run, seed, n_poblacion, n_iter, porc_elite, porc_cruce, porc_mut, prob_mut, selec_method, mejor_distancia, tiempo_seg`
* **`results/lp_resultados.csv`** (por instancia):

  * `instancia, status, objetivo, tiempo_seg, n_vars, n_constraints`
* **`results/comparativo_ga_lp.csv`** (tabla final):

  * `instancia, n_ciudades, metodo, distancia, tiempo, ...`
  * Para GA_top1..3: incluye `n_poblacion, n_iter, gap_pct_vs_LP`
  * Para LP: incluye `n_vars, n_constraints`

### 5.2. Imágenes

* GA (por corrida):

  * `GA_<instancia>_run<k>_convergencia.png`
  * `GA_<instancia>_run<k>_diversidad.png`
  * `GA_<instancia>_run<k>_ruta.png`
  * `GA_<instancia>_evolucion.gif` (solo para `run=1` si hay `imageio`)
* LP (por instancia):

  * `LP_<instancia>_ruta.png`
* Comparativas:

  * `tiempos_ga_vs_lp.png` (barras: tiempo promedio GA vs LP)
  * `gap_ga_top1_vs_lp.png` (gap % del mejor GA vs LP)

---

## 6) Guía para el **informe técnico**

Incluye al menos:

1. **Diseño del GA**

   * Población inicial, selección (torneo/ruleta), cruce OX, mutación swap, élite.
   * Parámetros (`N`, `maxIter`, `porc_elite`, `porc_cruce`, `porc_mut`, `prob_mut`, `selec_method`, `torneo_k`).
   * Indicadores de **diversidad** y su comportamiento.
2. **Formulación LP (MTZ)**

   * Variables, función objetivo, restricciones (flujo + MTZ), tamaño del modelo (#vars/#restricciones) y escalabilidad.
3. **Resultados por escenario** (eil101, gr229, inventado)

   * Tabla por escenario con: `n_ciudades, N (GA), maxIter (GA), #vars LP, #restricciones LP, tiempo, distancia`.
   * Ruta final (GA y LP), **convergencia**, **diversidad**, **evolución de ruta (GIF/snapshots)**.
4. **Comparativa GA vs LP**

   * Top-3 GA vs LP óptimo: **gap %**, tiempos, discusión de calidad/costo.
   * Efecto de la métrica (EUC vs GEO) en gr229.
5. **Conclusiones**

   * Cuándo usar GA vs LP; trade-offs de tiempo/óptimo; impacto de parámetros; escalabilidad.

---

## 7) Recomendaciones operativas

* **Reproducibilidad**: usar `--seed` o seeds fijas por *run*.
* **Tiempo/cómputo**: `gr229` con MTZ puede tardar; ajusta `--time_limit` y documenta el `status` del solver.
* **Consistencia de métrica**: verifica que GA y LP usen la **misma dist_matrix** para cada instancia.
* **Diversidad**: inspecciona la curva; si cae rápido, aumenta `porc_mut` o `prob_mut`.

---

## 8) Ejemplos rápidos

```bash
# (1) Recalcular GA para gr229 con GEO y guardar figuras
python main.py --metodo ga --instancia gr229 --save-results

# (2) Correr los experimentos batch (GA y LP)
python experimento_ga.py
python experimento_lp.py

# (3) Regenerar comparativo y gráficas de informe
python aggregate_results.py
```

---

## 9) Licencia y crédito

Proyecto académico para **Modelación y Simulación 2025**. Código orientado a docencia/demostración.
