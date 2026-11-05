"""Valida Newton-Raphson y Bisección contra SciPy (brentq) y soluciones analíticas.
Criterio RT1: error relativo <= 1e-6."""
import math
from scipy.optimize import brentq
from metodos.newton_raphson import newton_raphson
from metodos.biseccion import biseccion

LIMITE = 1e-6

casos = [
    # nombre, f, f', intervalo [a,b], x0, referencia (None = usar brentq)
    ("x^2 - 2", lambda x: x**2 - 2, lambda x: 2 * x, (1, 2), 1.5, math.sqrt(2)),
    ("cos(x) - x", lambda x: math.cos(x) - x, lambda x: -math.sin(x) - 1, (0, 1), 0.5, None),
    ("x^3 - x - 2", lambda x: x**3 - x - 2, lambda x: 3 * x**2 - 1, (1, 2), 1.5, None),
]

print(f"{'Función':<14}{'Método':<17}{'Iter':>6}{'Raíz':>18}{'Error relativo':>17}  Resultado")
for nombre, f, df, (a, b), x0, ref in casos:
    if ref is None:
        ref = brentq(f, a, b, xtol=1e-14)

    r = newton_raphson(f, x0, df=df)
    e = abs(r["raiz"] - ref) / abs(ref)
    ok = r["convergio"] and e <= LIMITE
    print(f"{nombre:<14}{'Newton-Raphson':<17}{r['iteraciones']:>6}{r['raiz']:>18.12f}{e:>17.3e}  {'OK' if ok else 'FALLA'}")

    r = biseccion(f, a, b)
    e = abs(r["raiz"] - ref) / abs(ref)
    ok = r["convergio"] and e <= LIMITE
    print(f"{nombre:<14}{'Bisección':<17}{r['iteraciones']:>6}{r['raiz']:>18.12f}{e:>17.3e}  {'OK' if ok else 'FALLA'}")

# Newton sin derivada explícita (diferencias centrales)
r = newton_raphson(lambda x: x**2 - 2, 1.5)
print("\nNewton sin derivada dada ->", r["estado"], f"({r['iteraciones']} iter, raíz {r['raiz']:.12f})")

# Casos de error controlado
print("Bisección sin cambio de signo ->", biseccion(lambda x: x**2 + 1, -1, 1)["estado"])
print("Newton con derivada cero      ->", newton_raphson(lambda x: x**2 + 1, 0.0, df=lambda x: 2 * x)["estado"])
