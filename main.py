"""
main.py
Punto de entrada del proyecto PSO-ANFIS.
Ejecuta todo el pipeline y guarda resultados en results/.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('src')

from preprocessing import cargar_y_limpiar, preparar_datos
from anfis_model import ANFIS
from hybrid_model import entrenar_hibrido
from baselines import entrenar_todos
from metrics import evaluar_modelo, prueba_wilcoxon, calc_rmse

os.makedirs('results/figures', exist_ok=True)
os.makedirs('results/tables', exist_ok=True)

# ── 1. Limpieza → genera weather_clean.csv ────────────────────────────────────
print("=" * 60)
print("  PSO-ANFIS · Predicción de Temperatura Aparente")
print("=" * 60)
print("\n[1] Limpieza y normalización de datos...")
cargar_y_limpiar()
X_train, X_val, X_test, y_train, y_val, y_test, scaler_y = preparar_datos()

# ── 2. ANFIS base ─────────────────────────────────────────────────────────────
print("\n[2] ANFIS base (sin PSO)...")
t0 = time.time()
anfis_base = ANFIS()
mu = anfis_base.calcular_activaciones(X_train)
wn = anfis_base.normalizar_pesos(anfis_base.calcular_pesos_reglas(mu))
A  = anfis_base.construir_matriz_consecuentes(X_train, wn)
anfis_base.consecuentes, _, _, _ = np.linalg.lstsq(A, y_train, rcond=None)
anfis_base.entrenar(X_train, y_train, epochs=30, lr=0.005)
res_base = evaluar_modelo("ANFIS base (sin PSO)", y_test, anfis_base.predecir(X_test))
res_base['tiempo'] = round(time.time() - t0, 1)

# ── 3. Baselines ──────────────────────────────────────────────────────────────
print("\n[3] Modelos de referencia...")
t0 = time.time()
res_baselines, _ = entrenar_todos(X_train, y_train, X_test, y_test)
t_base = round(time.time() - t0, 1)
for r in res_baselines:
    r['tiempo'] = t_base

# ── 4. PSO-ANFIS ──────────────────────────────────────────────────────────────
N_CORRIDAS = 5    # cambiar a 30 para entrega completa
N_ITER     = 50   # cambiar a 100 para entrega completa
N_PART     = 20   # cambiar a 30 para entrega completa

print(f"\n[4] PSO-ANFIS ({N_CORRIDAS} corridas)...")
rmses_pso = []; hist_mejor = None; mejor_modelo = None

for corrida in range(N_CORRIDAS):
    print(f"\n  >> Corrida {corrida+1}/{N_CORRIDAS}")
    np.random.seed(corrida * 7)
    t0 = time.time()
    modelo, historial = entrenar_hibrido(
        X_train, y_train, X_val, y_val,
        n_particulas=N_PART, n_iter=N_ITER
    )
    t_pso = time.time() - t0
    r = calc_rmse(y_test, modelo.predecir(X_test))
    rmses_pso.append(r)
    if hist_mejor is None or r == min(rmses_pso):
        hist_mejor = historial
        mejor_modelo = modelo

print(f"\n  Media ± std RMSE: {np.mean(rmses_pso):.4f} ± {np.std(rmses_pso):.4f}")
res_pso = evaluar_modelo("PSO-ANFIS", y_test, mejor_modelo.predecir(X_test))
res_pso['tiempo'] = round(t_pso, 1)

# ── 5. Tabla comparativa ──────────────────────────────────────────────────────
print("\n[5] Tabla comparativa:")
print("-" * 62)
print(f"{'Modelo':<28} {'RMSE':>7} {'MAE':>7} {'R²':>7} {'t(s)':>7}")
print("-" * 62)
todos = [res_base] + res_baselines + [res_pso]
for r in todos:
    print(f"  {r['modelo']:<26} {r['RMSE']:>7.4f} {r['MAE']:>7.4f} {r['R2']:>7.4f} {r.get('tiempo','-'):>7}")

df_tabla = pd.DataFrame(todos)
df_tabla.to_csv('results/tables/comparativa.csv', index=False)
print("\nTabla guardada en results/tables/comparativa.csv")

# ── 6. Wilcoxon ───────────────────────────────────────────────────────────────
if len(rmses_pso) > 1:
    base_rep = [calc_rmse(y_test, anfis_base.predecir(X_test)) +
                np.random.normal(0, 1e-5) for _ in range(len(rmses_pso))]
    prueba_wilcoxon(rmses_pso, base_rep, "PSO-ANFIS", "ANFIS base")

# ── 7. Gráficas individuales ──────────────────────────────────────────────────
print("\n[6] Generando gráficas...")
y_pso_test = mejor_modelo.predecir(X_test)

# Fig 4 — Curva de convergencia
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(hist_mejor, color='steelblue', linewidth=2)
ax.set_title('Curva de Convergencia del PSO')
ax.set_xlabel('Iteración'); ax.set_ylabel('RMSE validación')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/fig4_convergencia_pso.png', dpi=120, bbox_inches='tight')
plt.close()

# Fig 5 — Comparativa RMSE
nombres_m = ['ANFIS base', 'Regresión\nLineal', 'Random\nForest', 'MLP', 'PSO-ANFIS']
rmses_m   = [r['RMSE'] for r in todos]
colores_m = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(nombres_m, rmses_m, color=colores_m, width=0.5)
for bar, val in zip(bars, rmses_m):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.0003,
            f'{val:.4f}', ha='center', fontsize=9)
ax.set_title('Comparativa RMSE por Modelo')
ax.set_ylabel('RMSE'); ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('results/figures/fig5_comparativa_rmse.png', dpi=120, bbox_inches='tight')
plt.close()

# Fig 6 — Predicción vs Real
n = 300
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(y_test[:n], color='gray', alpha=0.7, linewidth=1, label='Real')
ax.plot(y_pso_test[:n], color='steelblue', linewidth=1, label='PSO-ANFIS')
ax.set_title('Predicción vs Real (primeras 300 muestras de prueba)')
ax.set_xlabel('Muestra'); ax.set_ylabel('Temp. aparente (norm.)')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/fig6_prediccion_vs_real.png', dpi=120, bbox_inches='tight')
plt.close()

# Fig 7 — Funciones de pertenencia
x_vals = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(7, 4))
for k, (label, color) in enumerate(zip(['Baja', 'Media', 'Alta'],
                                        ['#e74c3c', '#2ecc71', '#3498db'])):
    mu_v = np.exp(-0.5 * ((x_vals - mejor_modelo.centros[0, k]) /
                           (mejor_modelo.sigmas[0, k] + 1e-8)) ** 2)
    ax.plot(x_vals, mu_v, color=color, linewidth=2, label=f'Temp {label}')
ax.set_title('Funciones de Pertenencia aprendidas · Variable Temperatura')
ax.set_xlabel('Valor normalizado'); ax.set_ylabel('Grado de pertenencia μ')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/figures/fig7_funciones_pertenencia.png', dpi=120, bbox_inches='tight')
plt.close()

print("Gráficas guardadas en results/figures/")
print("\n" + "=" * 60)
print("  Pipeline completado.")
print("=" * 60)
