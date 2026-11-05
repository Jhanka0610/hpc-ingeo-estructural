"""Caso estructural: encadena Gauss-Seidel, Gauss-Jordan, Newton-Raphson y Bisección.
Ejecutar con: python -m pruebas.caso_estructural"""
import numpy as np
from scipy.optimize import brentq
from metodos.gauss_seidel import gauss_seidel
from metodos.gauss_jordan import gauss_jordan
from modelo import edificio_cortante as ed

K = ed.ensamblar(ed.rigideces())
print(f"Rigidez por entrepiso [N/m]: {np.array2string(ed.rigideces(), precision=3)}")
print(f"Límite de deriva usado: {ed.LIM_DERIVA*100:.1f} % (verificar NSR-10)\n")

for nombre, F in ed.escenarios().items():
    print("=" * 78)
    print(f"ESCENARIO: {nombre}  (cortante basal = {F.sum()/1e3:.0f} kN)")

    # 1) Respuesta lineal: Gauss-Seidel, verificada con Gauss-Jordan y numpy
    ref = np.linalg.solve(K, F)
    gs = gauss_seidel(K, F, tol=1e-12, max_iter=100000)
    gj = gauss_jordan(K, F)
    e_gs = np.linalg.norm(gs["x"] - ref) / np.linalg.norm(ref)
    e_gj = np.linalg.norm(gj["x"] - ref) / np.linalg.norm(ref)
    e_cruz = np.linalg.norm(gs["x"] - gj["x"]) / np.linalg.norm(gj["x"])
    print(f"Lineal  | Gauss-Seidel: {gs['iteraciones']} iter, error vs numpy {e_gs:.2e}"
          f" | Gauss-Jordan: error vs numpy {e_gj:.2e} | GS vs GJ {e_cruz:.2e}")

    # 2) Respuesta no lineal con carga de servicio (lam = 1): Newton + Gauss-Jordan
    nl = ed.respuesta_no_lineal(F, 1.0)
    res_norm = np.linalg.norm(ed.fuerza_interna(nl["x"]) - F) / np.linalg.norm(F)
    print(f"No lineal (lam=1) | Newton: {nl['iteraciones']} iter, estado {nl['estado']},"
          f" residuo relativo {res_norm:.2e}")

    print(f"{'Piso':>5}{'u lineal [mm]':>16}{'u no lineal [mm]':>19}{'Deriva lineal [%]':>20}{'Deriva NL [%]':>16}")
    dl = ed.derivas(gj["x"]) / ed.H_PISO * 100
    dn = ed.derivas(nl["x"]) / ed.H_PISO * 100
    for i in range(ed.PISOS):
        print(f"{i+1:>5}{gj['x'][i]*1e3:>16.3f}{nl['x'][i]*1e3:>19.3f}{dl[i]:>20.4f}{dn[i]:>16.4f}")
    cumple = ed.deriva_max(nl["x"]) <= ed.LIM_DERIVA
    print(f"Deriva máx. no lineal = {ed.deriva_max(nl['x'])*100:.4f} %  -> {'CUMPLE' if cumple else 'NO CUMPLE'}")

    # 3) Carga crítica: Bisección sobre el factor de carga
    res, hi = ed.factor_critico(F)
    ref_b = brentq(lambda lam: ed.g_deriva(F, lam), 0.0, hi, xtol=1e-12)
    e_b = abs(res["raiz"] - ref_b) / ref_b
    print(f"Bisección | factor crítico = {res['raiz']:.6f} ({res['iteraciones']} iter, estado {res['estado']})"
          f" | brentq = {ref_b:.6f} | error relativo {e_b:.2e}")
    print(f"Factor de seguridad (carga crítica / carga de servicio) = {res['raiz']:.3f}"
          f"  -> cortante basal crítico = {res['raiz']*F.sum()/1e3:.0f} kN\n")
