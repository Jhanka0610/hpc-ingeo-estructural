"""Edificio de cortante de 5 pisos (SUPUESTO ACADÉMICO, no es un diseño real).

Un grado de libertad lateral por piso. Unidades: N, m.
Solo se modelan cargas laterales; la carga muerta y la viva (verticales) no
generan desplazamiento lateral en este modelo simplificado (limitación).

Comportamiento no lineal de cada entrepiso (ablandamiento hiperbólico):
    V(d) = k0 * d / (1 + |d| / d_ref)
    rigidez tangente = k0 / (1 + |d| / d_ref)^2
"""
import numpy as np
from metodos.gauss_jordan import gauss_jordan
from metodos.newton_raphson_sistema import newton_raphson_sistema
from metodos.biseccion import biseccion

# ---- Supuestos (todos ajustables) ----
PISOS = 5
H_PISO = 3.0                      # altura de entrepiso [m]
E = 200e9                         # módulo de elasticidad del acero [Pa]
I_COL = 2.0e-4                    # inercia de cada columna [m^4] (supuesto)
N_COL = 4                         # columnas por entrepiso
FACTORES = np.array([1.0, 1.0, 0.9, 0.8, 0.7])   # rigidez relativa por entrepiso
V_BASE = 400e3                    # cortante basal de referencia [N] (supuesto)
D_REF = 0.02 * H_PISO             # deriva de referencia del ablandamiento [m]
LIM_DERIVA = 0.01                 # límite de deriva (1 %): VERIFICAR en NSR-10 Tabla A.6.4-1


def rigideces():
    k0 = N_COL * 12 * E * I_COL / H_PISO ** 3
    return k0 * FACTORES


def ensamblar(k):
    n = len(k)
    K = np.zeros((n, n))
    for i in range(n):
        K[i, i] += k[i]
        if i > 0:
            K[i - 1, i - 1] += k[i]
            K[i, i - 1] -= k[i]
            K[i - 1, i] -= k[i]
    return K


def derivas(u):
    return np.diff(np.concatenate(([0.0], u)))


def deriva_max(u):
    return float(np.max(np.abs(derivas(u))) / H_PISO)


def fuerza_interna(u):
    d = derivas(u)
    s = rigideces() * d / (1 + np.abs(d) / D_REF)   # cortante de cada entrepiso
    f = s.copy()
    f[:-1] -= s[1:]
    return f


def tangente(u):
    d = derivas(u)
    return ensamblar(rigideces() / (1 + np.abs(d) / D_REF) ** 2)


def escenarios():
    w = np.arange(1, PISOS + 1, dtype=float)
    tri = V_BASE * w / w.sum()                 # distribución triangular
    uni = np.full(PISOS, V_BASE / PISOS)       # distribución uniforme
    return {"Sismo X triangular": tri,
            "Sismo X uniforme": uni,
            "Combinación 1.0 tri + 0.3 uni": tri + 0.3 * uni}


def respuesta_no_lineal(F, lam=1.0):
    """Equilibrio no lineal con factor de carga lam (Newton + Gauss-Jordan)."""
    u0 = gauss_jordan(ensamblar(rigideces()), lam * F)["x"]   # inicio: solución lineal
    return newton_raphson_sistema(lambda u: fuerza_interna(u) - lam * F, tangente, u0)


def g_deriva(F, lam, limite=LIM_DERIVA):
    """f(lam) = deriva máxima - límite. Si Newton no converge se toma como
    pérdida de equilibrio (se considera que el límite se superó)."""
    r = respuesta_no_lineal(F, lam)
    return (deriva_max(r["x"]) - limite) if r["convergio"] else 1.0


def factor_critico(F, limite=LIM_DERIVA):
    """Bisección sobre el factor de carga hasta que la deriva máxima = límite."""
    hi = 1.0
    while g_deriva(F, hi, limite) < 0 and hi < 1e3:
        hi *= 2
    res = biseccion(lambda lam: g_deriva(F, lam, limite), 0.0, hi, tol=1e-10)
    return res, hi
