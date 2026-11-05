"""Método de Bisección (versión serial).

Entradas:
    f        : función f(x) continua en [a, b]
    a, b     : extremos del intervalo (f(a) y f(b) deben tener signo contrario)
    tol      : tolerancia sobre el ancho del intervalo
    max_iter : máximo de iteraciones

Salida (dict):
    raiz, iteraciones, convergio (bool), error_final, historial, estado
    estado: "ok", "sin_cambio_de_signo" o "sin_convergencia"
"""


def biseccion(f, a, b, tol=1e-12, max_iter=200):
    a, b = float(a), float(b)
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        return {"raiz": None, "iteraciones": 0, "convergio": False,
                "error_final": None, "historial": [],
                "estado": "sin_cambio_de_signo"}

    historial = []
    for k in range(1, max_iter + 1):
        c = (a + b) / 2
        fc = f(c)
        ancho = (b - a) / 2
        historial.append(ancho)
        if fc == 0 or ancho < tol:
            return {"raiz": c, "iteraciones": k, "convergio": True,
                    "error_final": ancho, "historial": historial, "estado": "ok"}
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc

    return {"raiz": (a + b) / 2, "iteraciones": max_iter, "convergio": False,
            "error_final": (b - a) / 2, "historial": historial,
            "estado": "sin_convergencia"}
