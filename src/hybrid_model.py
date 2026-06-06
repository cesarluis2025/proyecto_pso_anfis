"""
hybrid_model.py
Integración PSO + ANFIS.
PSO optimiza los centros y anchuras de las funciones de pertenencia del ANFIS.
"""

import numpy as np
import sys, os
sys.path.append(os.path.dirname(__file__))

from anfis_model import ANFIS
from pso import PSO
from metrics import calc_rmse


def crear_fitness(X_train, y_train, X_val, y_val, n_inputs=4, n_mf=3):
    """
    Crea la función de aptitud para el PSO.
    Recibe un vector de parámetros, configura el ANFIS,
    entrena los consecuentes con LSE y devuelve el RMSE en validación.
    """
    def fitness(params):
        modelo = ANFIS(n_inputs=n_inputs, n_mf=n_mf)
        modelo.set_params(params)

        # Solo ajustamos los consecuentes (LSE, rápido)
        mu = modelo.calcular_activaciones(X_train)
        w  = modelo.calcular_pesos_reglas(mu)
        wn = modelo.normalizar_pesos(w)
        A  = modelo.construir_matriz_consecuentes(X_train, wn)
        modelo.consecuentes, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)

        return calc_rmse(y_val, modelo.predecir(X_val))

    return fitness


def entrenar_hibrido(X_train, y_train, X_val, y_val,
                     n_inputs=4, n_mf=3, n_particulas=30, n_iter=100):
    """
    Ejecuta el sistema híbrido PSO-ANFIS completo.
    1) PSO encuentra los mejores parámetros de MF (minimizando RMSE en val)
    2) ANFIS final se entrena con esos parámetros + ajuste de consecuentes
    Devuelve el modelo ANFIS entrenado y el historial de convergencia del PSO.
    """
    dim = n_inputs * n_mf * 2   # centros + sigmas

    fitness_fn = crear_fitness(X_train, y_train, X_val, y_val, n_inputs, n_mf)

    print(f"Iniciando PSO: {n_particulas} partículas | {n_iter} iteraciones | dim={dim}")
    pso = PSO(fitness_fn=fitness_fn, dim=dim,
              n_particulas=n_particulas, n_iter=n_iter)

    mejores_params, mejor_rmse = pso.optimizar()
    print(f"PSO finalizado. Mejor RMSE validación: {mejor_rmse:.6f}")

    # Construir el modelo final con los mejores parámetros encontrados
    modelo_final = ANFIS(n_inputs=n_inputs, n_mf=n_mf)
    modelo_final.set_params(mejores_params)
    # Ajuste fino de consecuentes con entrenamiento completo
    modelo_final.entrenar(X_train, y_train, epochs=30, lr=0.005)

    return modelo_final, pso.historial
