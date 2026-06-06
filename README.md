# Sistema Híbrido PSO-ANFIS
## Predicción de Temperatura Aparente · Weather Szeged 2006–2016
**Inteligencia Computacional · Universidad de San Buenaventura Medellín**

---

### Estructura del repositorio

```
proyecto_pso_anfis_atmosfera/
├── data/
│   ├── raw/weatherHistory.csv        ← dataset original Kaggle
│   └── processed/weather_clean.csv  ← dataset limpio y normalizado
├── src/
│   ├── preprocessing.py             ← EDA, limpieza, normalización
│   ├── anfis_model.py               ← Clase ANFIS (Takagi-Sugeno)
│   ├── pso.py                       ← Clase PSO con pbest/gbest
│   ├── hybrid_model.py              ← PSO+ANFIS integrado
│   ├── baselines.py                 ← LR, RF, MLP (sklearn)
│   └── metrics.py                   ← RMSE, MAE, R², Wilcoxon
├── notebooks/
│   ├── 01_EDA.ipynb                 ← Análisis exploratorio
│   └── 02_experimento.ipynb        ← Pipeline completo
├── results/
│   ├── tables/comparativa.csv
│   └── figures/
├── Dockerfile                       ← Imagen Docker del proyecto
├── docker-compose.yml               ← Orquestación del contenedor
├── requirements.txt                 ← Dependencias Python
├── main.py                          ← Punto de entrada
└── README.md
```

---

### Cómo correr el proyecto con Docker

**Requisitos:** tener Docker Desktop instalado y corriendo.

**Paso 1** — Abrir una terminal en la carpeta del proyecto y construir la imagen:
```bash
docker-compose build
```

**Paso 2** — Levantar el contenedor:
```bash
docker-compose up
```

**Paso 3** — Abrir en el navegador:
```
http://localhost:8888
```

Ahí aparece Jupyter con todos los notebooks listos para correr.

**Para detener el contenedor:**
```bash
docker-compose down
```

---

### Cómo correr sin Docker

```bash
pip install -r requirements.txt
python main.py
```
