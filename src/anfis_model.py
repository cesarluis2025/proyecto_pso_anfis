"""
anfis_model.py
Clase ANFIS — Sistema Neuro-Difuso Adaptativo (Takagi-Sugeno de primer orden).
Usa 3 funciones de pertenencia gaussianas por cada variable de entrada.
"""

import numpy as np


class ANFIS:
    def __init__(self, n_inputs=4, n_mf=3):
        """
        n_inputs : número de variables de entrada (4 en este proyecto)
        n_mf     : funciones de pertenencia por variable (3: Baja, Media, Alta)
        """
        self.n_inputs = n_inputs
        self.n_mf     = n_mf

        # Centros distribuidos uniformemente en [0.1, 0.9] (datos normalizados)
        self.centros = np.linspace(0.1, 0.9, n_mf * n_inputs).reshape(n_inputs, n_mf)
        # Anchuras iniciales iguales para todas las MF
        self.sigmas  = np.full((n_inputs, n_mf), 0.3)
        # Parámetros del consecuente (se aprenden con mínimos cuadrados)
        self.consecuentes = None

    # ── Funciones de pertenencia ──────────────────────────────────────────────

    def gaussiana(self, x, c, sigma):
        """μ(x) = exp( -0.5 * ((x - c) / sigma)² )"""
        return np.exp(-0.5 * ((x - c) / (sigma + 1e-8)) ** 2)

    def calcular_activaciones(self, X):
        """
        Calcula el grado de pertenencia de cada muestra a cada MF.
        X   : (n_muestras, n_inputs)
        mu  : (n_muestras, n_inputs, n_mf)
        """
        n  = X.shape[0]
        mu = np.zeros((n, self.n_inputs, self.n_mf))
        for i in range(self.n_inputs):
            for j in range(self.n_mf):
                mu[:, i, j] = self.gaussiana(X[:, i], self.centros[i, j], self.sigmas[i, j])
        return mu

    def calcular_pesos_reglas(self, mu):
        """
        Peso de cada regla = producto de los grados de pertenencia de sus antecedentes.
        Usamos n_mf reglas (regla k usa la MF k de cada variable).
        w : (n_muestras, n_mf)
        """
        n = mu.shape[0]
        w = np.ones((n, self.n_mf))
        for j in range(self.n_mf):
            for i in range(self.n_inputs):
                w[:, j] *= mu[:, i, j]
        return w

    def normalizar_pesos(self, w):
        """Normaliza para que los pesos sumen 1 en cada muestra."""
        return w / (w.sum(axis=1, keepdims=True) + 1e-8)

    def construir_matriz_consecuentes(self, X, w_norm):
        """
        Construye la matriz de diseño para los consecuentes lineales.
        Cada regla k: y_k = p0 + p1*x1 + p2*x2 + p3*x3 + p4*x4
        """
        bloques = []
        for k in range(self.n_mf):
            bloque = np.hstack([w_norm[:, k:k+1],
                                w_norm[:, k:k+1] * X])
            bloques.append(bloque)
        return np.hstack(bloques)

    # ── Predicción ────────────────────────────────────────────────────────────

    def predecir(self, X):
        mu     = self.calcular_activaciones(X)
        w      = self.calcular_pesos_reglas(mu)
        w_norm = self.normalizar_pesos(w)
        A      = self.construir_matriz_consecuentes(X, w_norm)
        return A @ self.consecuentes

    # ── Entrenamiento híbrido ─────────────────────────────────────────────────

    def entrenar(self, X_train, y_train, epochs=50, lr=0.01):
        """
        Algoritmo híbrido de Jang (1993):
        - Consecuentes (p, q, r): mínimos cuadrados (LSE) — paso exacto
        - Premisas (c, sigma): descenso de gradiente — paso aproximado
        """
        for _ in range(epochs):
            # Paso 1 — LSE para consecuentes
            mu     = self.calcular_activaciones(X_train)
            w      = self.calcular_pesos_reglas(mu)
            w_norm = self.normalizar_pesos(w)
            A      = self.construir_matriz_consecuentes(X_train, w_norm)
            self.consecuentes, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)

            # Paso 2 — Gradiente para centros y sigmas
            y_pred = A @ self.consecuentes
            error  = y_train - y_pred

            for i in range(self.n_inputs):
                for j in range(self.n_mf):
                    diff   = X_train[:, i] - self.centros[i, j]
                    dmu_dc = mu[:, i, j] * diff / (self.sigmas[i, j] ** 2 + 1e-8)
                    dmu_ds = mu[:, i, j] * diff ** 2 / (self.sigmas[i, j] ** 3 + 1e-8)

                    self.centros[i, j] -= lr * (-2 * np.mean(error * w_norm[:, j] * dmu_dc))
                    self.sigmas[i, j]  -= lr * (-2 * np.mean(error * w_norm[:, j] * dmu_ds))
                    self.sigmas[i, j]   = max(self.sigmas[i, j], 0.01)

    # ── Interfaz para el PSO ──────────────────────────────────────────────────

    def set_params(self, vector_params):
        """
        Asigna el vector de parámetros que viene del PSO.
        Formato: [centros (n_inputs*n_mf), sigmas (n_inputs*n_mf)]
        """
        total = self.n_inputs * self.n_mf
        self.centros = vector_params[:total].reshape(self.n_inputs, self.n_mf)
        self.sigmas  = np.abs(vector_params[total:].reshape(self.n_inputs, self.n_mf)) + 0.01
