import numpy as np

def generar_inventado(n=80, radio=100, ruido=10, archivo="data/inventado.tsp"):
    angulos = np.linspace(0, 2*np.pi, n, endpoint=False)
    xs = radio * np.cos(angulos) + np.random.randn(n)*ruido
    ys = radio * np.sin(angulos) + np.random.randn(n)*ruido
    with open(archivo, "w") as f:
        f.write(f"NAME: inventado\nTYPE: TSP\nDIMENSION: {n}\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")
        for i, (x,y) in enumerate(zip(xs,ys), start=1):
            f.write(f"{i} {x:.6f} {y:.6f}\n")
        f.write("EOF\n")
    print(f"Instancia inventada guardada en {archivo} con {n} ciudades.")

if __name__ == "__main__":
    generar_inventado()
