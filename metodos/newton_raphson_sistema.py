"""Newton-Raphson para sistemas no lineales R(u) = 0 (versión serial).

En cada iteración resuelve J(u) du = -R(u) con Gauss-Jordan (encadenamiento
de métodos del catálogo) y actualiza u <- u + du.

Entradas:
    residuo  : función R(u) -> vector
    tangente : función J(u) -> matriz jacobiana (rigidez tangente)
    u0       : estimación inicial
    tol      : tolerancia sobre el cambio relativo de u
    max_iter : máximo de iteraciones

Salida (dict): x, iteraciones, convergio, error_final, historial, estado
    estado: "ok", "jacobiana_singular" o "sin_convergencia"
"""
import numpy as np
from metodos.gauss_jordan import gauss_jordan


def newton_raphson_sistema(residuo, tangente, u0, tol=1e-10, max_iter=50):
    u = np.array(u0, dtype=float)
    historial = []
    err = np.inf
    for k in range(1, max_iter + 1):
        sol = gauss_jordan(tangente(u), -residuo(u))
        if sol["estado"] != "ok":
            return {"x": u, "iteraciones": k, "convergio": False,
                    "error_final": err, "historial": historial,
                    "estado": "jacobiana_singular"}
        du = sol["x"]
        u = u + du
        err = np.linalg.norm(du, np.inf) / max(np.linalg.norm(u, np.inf), 1e-30)
        historial.append(err)
        if err < tol:
            return {"x": u, "iteraciones": k, "convergio": True,
                    "error_final": err, "historial": historial, "estado": "ok"}
    return {"x": u, "iteraciones": max_iter, "convergio": False,
            "error_final": err, "historial": historial,
            "estado": "sin_convergencia"}
