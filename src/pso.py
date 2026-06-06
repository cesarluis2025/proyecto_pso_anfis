"""
pso.py
Clase PSO — Particle Swarm Optimization (Kennedy & Eberhart, 1995).
Optimiza los parámetros de las funciones de pertenencia del ANFIS.
"""

import numpy as np


class PSO:
    def __init__(self, fitness_fn, dim, n_particulas=30, n_iter=100,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0):
        """
        fitness_fn  : función que recibe un vector de params y devuelve el RMSE
        dim         : dimensión del espacio (24 = 4 vars × 3 MFs × 2 params)
        n_particulas: tamaño del enjambre
        n_iter      : número máximo de iteraciones
        w_max/w_min : inercia inicial y final (decrece linealmente)
        c1, c2      : coeficientes cognitivo y social
        """
        self.fitness_fn   = fitness_fn
        self.dim          = dim
        self.n_particulas = n_particulas
        self.n_iter       = n_iter
        self.w_max        = w_max
        self.w_min        = w_min
        self.c1           = c1
        self.c2           = c2

        # Límites del espacio de búsqueda
        # Primera mitad → centros en [0, 1] (datos normalizados)
        # Segunda mitad → sigmas en [0.05, 0.5]
        self.lb = np.array([0.0]  * (dim // 2) + [0.05] * (dim // 2))
        self.ub = np.array([1.0]  * (dim // 2) + [0.5]  * (dim // 2))

        # Inicializar posiciones aleatorias dentro de los límites
        self.posiciones  = np.random.uniform(0, 1, (n_particulas, dim))
        self.posiciones[:, dim//2:] = np.random.uniform(0.05, 0.4, (n_particulas, dim//2))
        self.velocidades = np.zeros((n_particulas, dim))

        # Mejor posición individual (pbest) y mejor global (gbest)
        self.pbest_pos = self.posiciones.copy()
        self.pbest_val = np.full(n_particulas, np.inf)
        self.gbest_pos = np.zeros(dim)
        self.gbest_val = np.inf

        # Historial del mejor RMSE en cada iteración (para la curva de convergencia)
        self.historial = []

    def optimizar(self):
        """Ejecuta el ciclo principal del PSO y devuelve la mejor solución encontrada."""

        # Evaluación inicial de todas las partículas
        for i in range(self.n_particulas):
            val = self.fitness_fn(self.posiciones[i])
            self.pbest_val[i] = val
            if val < self.gbest_val:
                self.gbest_val = val
                self.gbest_pos = self.posiciones[i].copy()
        self.historial.append(self.gbest_val)

        # Ciclo principal
        for t in range(self.n_iter):
            # Inercia decreciente: exploración al inicio, explotación al final
            w = self.w_max - (self.w_max - self.w_min) * (t / self.n_iter)

            for i in range(self.n_particulas):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)

                # Actualizar velocidad
                cognitivo          = self.c1 * r1 * (self.pbest_pos[i] - self.posiciones[i])
                social             = self.c2 * r2 * (self.gbest_pos    - self.posiciones[i])
                self.velocidades[i] = w * self.velocidades[i] + cognitivo + social

                # Actualizar posición y mantener dentro de los límites
                self.posiciones[i] += self.velocidades[i]
                self.posiciones[i]  = np.clip(self.posiciones[i], self.lb, self.ub)

                # Evaluar nueva posición
                val = self.fitness_fn(self.posiciones[i])

                # Actualizar pbest si mejoró
                if val < self.pbest_val[i]:
                    self.pbest_val[i] = val
                    self.pbest_pos[i] = self.posiciones[i].copy()

                # Actualizar gbest si mejoró
                if val < self.gbest_val:
                    self.gbest_val = val
                    self.gbest_pos = self.posiciones[i].copy()

            self.historial.append(self.gbest_val)

            if (t + 1) % 10 == 0:
                print(f"  Iteración {t+1:3d}/{self.n_iter} | Mejor RMSE: {self.gbest_val:.6f}")

        return self.gbest_pos, self.gbest_val
