"""Método de Newton-Raphson (versión serial).

Entradas:
    f        : función f(x) cuya raíz se busca
    x0       : estimación inicial
    df       : derivada f'(x) (opcional; si no se da, se aproxima por diferencias centrales)
    tol      : tolerancia sobre |f(x)| y sobre el cambio entre iteraciones
    max_iter : máximo de iteraciones

Salida (dict):
    raiz, iteraciones, convergio (bool), error_final, historial, estado
    estado: "ok", "derivada_cero" o "sin_convergencia"
"""


def newton_raphson(f, x0, df=None, tol=1e-12, max_iter=100):
    if df is None:
        def df(x, h=1e-7):
            return (f(x + h) - f(x - h)) / (2 * h)

    x = float(x0)
    historial = []

    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-14:
            return {"raiz": x, "iteraciones": k, "convergio": False,
                    "error_final": abs(fx), "historial": historial,
                    "estado": "derivada_cero"}
        x_nuevo = x - fx / dfx
        err = abs(x_nuevo - x)
        historial.append(err)
        x = x_nuevo
        if err < tol and abs(f(x)) < 1e-9:
            return {"raiz": x, "iteraciones": k, "convergio": True,
                    "error_final": err, "historial": historial, "estado": "ok"}

    return {"raiz": x, "iteraciones": max_iter, "convergio": False,
            "error_final": historial[-1], "historial": historial,
            "estado": "sin_convergencia"}
