"""Lee resultados/tiempos.csv y calcula speedup S = T1/TN y eficiencia E = S/N."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

datos = defaultdict(lambda: defaultdict(list))
with open("resultados/tiempos.csv") as f:
    for fila in csv.DictReader(f):
        datos[fila["metodo"]][int(fila["procesos"])].append(float(fila["tiempo_s"]))

nombres = {"gauss_seidel": "Gauss-Seidel (bloques)", "gauss_jordan": "Gauss-Jordan (filas)"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

print(f"{'Método':<24}{'P':>3}{'T media (s)':>13}{'Desv.':>9}{'Speedup':>9}{'Eficiencia':>12}")
for metodo, por_p in datos.items():
    ps = sorted(por_p)
    T = {p: np.mean(por_p[p]) for p in ps}
    S = [T[1] / T[p] for p in ps]
    E = [s / p for s, p in zip(S, ps)]
    for p, s, e in zip(ps, S, E):
        print(f"{nombres[metodo]:<24}{p:>3}{T[p]:>13.3f}{np.std(por_p[p]):>9.3f}{s:>9.2f}{e*100:>11.1f}%")
    ax1.plot(ps, S, "o-", label=nombres[metodo])
    ax2.plot(ps, [e * 100 for e in E], "o-", label=nombres[metodo])

ps_all = sorted({p for d in datos.values() for p in d})
ax1.plot(ps_all, ps_all, "k--", label="Ideal")
ax1.set(xlabel="Procesos (N)", ylabel="Speedup S = T1/TN", title="Speedup", xticks=ps_all)
ax2.axhline(60, color="r", ls="--", label="Meta 60 %")
ax2.set(xlabel="Procesos (N)", ylabel="Eficiencia E = S/N (%)", title="Eficiencia", xticks=ps_all)
for ax in (ax1, ax2):
    ax.grid(alpha=0.3)
    ax.legend()
plt.tight_layout()
plt.savefig("resultados/speedup_eficiencia.png", dpi=150)
print("\nGráfica guardada en resultados/speedup_eficiencia.png")
