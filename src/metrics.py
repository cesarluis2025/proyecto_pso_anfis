"""
metrics.py
Funciones de evaluación: RMSE, MAE, R² y prueba estadística de Wilcoxon.
"""

import numpy as np
from scipy.stats import wilcoxon


def calc_rmse(y_real, y_pred):
    """Error cuadrático medio (raíz cuadrada)."""
    return np.sqrt(np.mean((y_real - y_pred) ** 2))


def calc_mae(y_real, y_pred):
    """Error absoluto medio."""
    return np.mean(np.abs(y_real - y_pred))


def calc_r2(y_real, y_pred):
    """Coeficiente de determinación R²."""
    ss_res = np.sum((y_real - y_pred) ** 2)
    ss_tot = np.sum((y_real - np.mean(y_real)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-8)


def evaluar_modelo(nombre, y_real, y_pred):
    """Calcula las tres métricas, las imprime y devuelve un diccionario."""
    r = calc_rmse(y_real, y_pred)
    m = calc_mae (y_real, y_pred)
    r2 = calc_r2 (y_real, y_pred)
    print(f"  {nombre:<28} | RMSE: {r:.4f} | MAE: {m:.4f} | R²: {r2:.4f}")
    return {'modelo': nombre, 'RMSE': r, 'MAE': m, 'R2': r2}


def prueba_wilcoxon(errores_a, errores_b, nombre_a, nombre_b):
    """
    Prueba de Wilcoxon signed-rank (α = 0.05).
    errores_a, errores_b: listas de RMSE de múltiples corridas.
    """
    stat, p = wilcoxon(errores_a, errores_b)
    print(f"\nPrueba Wilcoxon: {nombre_a} vs {nombre_b}")
    print(f"  Estadístico W: {stat:.4f} | p-valor: {p:.4f}")
    if p < 0.05:
        print(f"  → Diferencia SIGNIFICATIVA (p < 0.05) — {nombre_a} es mejor")
    else:
        print(f"  → No hay diferencia significativa (p ≥ 0.05)")
    return p
