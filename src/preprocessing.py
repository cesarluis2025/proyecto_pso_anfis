"""
preprocessing.py
EDA, limpieza y normalización del dataset Weather Szeged 2006-2016.
Genera el archivo data/processed/weather_clean.csv con los datos listos para el modelo.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def cargar_y_limpiar(ruta_raw='data/raw/weatherHistory.csv',
                     ruta_clean='data/processed/weather_clean.csv'):
    """
    Lee el CSV original, hace la limpieza y guarda el dataset limpio.
    Devuelve el dataframe limpio.
    """
    df = pd.read_csv(ruta_raw)
    print(f"Dataset original: {df.shape[0]} filas, {df.shape[1]} columnas")

    # --- Eliminar columnas que no se usan ---
    # Loud Cover es constante (todos en 0), no aporta nada
    # Daily Summary y Summary son texto redundante
    # Wind Bearing y Precip Type no los usamos en el modelo
    # Formatted Date solo es el índice de tiempo
    df = df.drop(columns=[
        'Formatted Date', 'Summary', 'Daily Summary',
        'Loud Cover', 'Wind Bearing (degrees)', 'Precip Type'
    ])

    # --- Corregir Pressure = 0 (error de sensor, 517 registros) ---
    n_presion_cero = (df['Pressure (millibars)'] == 0).sum()
    print(f"Registros con Pressure = 0 (error sensor): {n_presion_cero}")
    mediana_presion = df[df['Pressure (millibars)'] > 0]['Pressure (millibars)'].median()
    df['Pressure (millibars)'] = df['Pressure (millibars)'].replace(0, mediana_presion)

    # --- Transformación raíz cuadrada en Wind Speed ---
    # La distribución de velocidad del viento está sesgada a la derecha
    # con sqrt la dejamos más simétrica
    df['Wind Speed (km/h)'] = np.sqrt(df['Wind Speed (km/h)'])

    # --- Renombrar columnas para facilidad de uso ---
    df = df.rename(columns={
        'Temperature (C)':        'temp',
        'Apparent Temperature (C)': 'temp_aparente',
        'Humidity':                'humedad',
        'Wind Speed (km/h)':      'viento',
        'Visibility (km)':        'visibilidad',
        'Pressure (millibars)':   'presion'
    })

    print(f"Dataset limpio: {df.shape[0]} filas, {df.shape[1]} columnas")
    print(f"Columnas: {df.columns.tolist()}")

    # Guardar el dataset limpio
    df.to_csv(ruta_clean, index=False)
    print(f"Dataset limpio guardado en: {ruta_clean}")

    return df


def preparar_datos(ruta_clean='data/processed/weather_clean.csv'):
    """
    Lee el dataset limpio, normaliza y divide en train/val/test.
    Devuelve los 6 conjuntos y el escalador de y para desnormalizar después.
    """
    df = pd.read_csv(ruta_clean)

    # Variables de entrada y variable objetivo
    features = ['temp', 'humedad', 'viento', 'presion']
    target   = 'temp_aparente'

    X = df[features].values
    y = df[target].values

    # Normalización MinMaxScaler → todo queda entre 0 y 1
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_norm = scaler_X.fit_transform(X)
    y_norm = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # División: 70% entrenamiento | 15% validación | 15% prueba
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_norm, y_norm, test_size=0.30, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42
    )

    print(f"Entrenamiento: {X_train.shape[0]} registros")
    print(f"Validación:    {X_val.shape[0]} registros")
    print(f"Prueba:        {X_test.shape[0]} registros")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_y


if __name__ == '__main__':
    cargar_y_limpiar()
    preparar_datos()
