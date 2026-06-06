"""
baselines.py
Modelos de referencia para comparar con PSO-ANFIS:
  - Regresión lineal múltiple (LR)
  - Random Forest (RF)
  - Red neuronal multicapa (MLP)
Todos usan la misma partición de datos (70/15/15).
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
import sys, os
sys.path.append(os.path.dirname(__file__))
from metrics import evaluar_modelo


def entrenar_lr(X_train, y_train, X_test, y_test):
    modelo  = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred  = modelo.predict(X_test)
    return evaluar_modelo("Regresión Lineal", y_test, y_pred), y_pred


def entrenar_rf(X_train, y_train, X_test, y_test):
    modelo  = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    modelo.fit(X_train, y_train)
    y_pred  = modelo.predict(X_test)
    return evaluar_modelo("Random Forest", y_test, y_pred), y_pred


def entrenar_mlp(X_train, y_train, X_test, y_test):
    modelo  = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu',
                           max_iter=300, random_state=42,
                           early_stopping=True, validation_fraction=0.1)
    modelo.fit(X_train, y_train)
    y_pred  = modelo.predict(X_test)
    return evaluar_modelo("MLP (Red Neuronal)", y_test, y_pred), y_pred


def entrenar_todos(X_train, y_train, X_test, y_test):
    """Entrena los tres modelos baseline y devuelve resultados y predicciones."""
    print("\n--- Modelos de referencia ---")
    res_lr,  pred_lr  = entrenar_lr (X_train, y_train, X_test, y_test)
    res_rf,  pred_rf  = entrenar_rf (X_train, y_train, X_test, y_test)
    res_mlp, pred_mlp = entrenar_mlp(X_train, y_train, X_test, y_test)
    return [res_lr, res_rf, res_mlp], [pred_lr, pred_rf, pred_mlp]
