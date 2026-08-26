"""
train.py

Script de entrenamiento reproducible para el modelo de duración de
STARWARS_AUTOCALLS.

Ejecuta el proceso completo desde los ficheros CSV de origen hasta un
modelo entrenado y evaluado: carga de datos, integración y limpieza,
partición temporal en entrenamiento y evaluación, entrenamiento del
modelo y de un modelo de referencia, cálculo de métricas, y guardado del
artefacto resultante en disco.

Este script constituye la respuesta al requisito del enunciado de
disponer de una vía reproducible para "ejecutar el entrenamiento a partir
de los ficheros CSV de origen" y "guardar el artefacto del modelo
resultante".

Modo de uso:

    python -m autocalls.model.train --raw-dir data/raw --output models/model.pkl

Argumentos:

    --raw-dir   Directorio que contiene rfqs.csv, daily_volatility.csv y
                underlyings_reference.csv. Por defecto, data/raw.
    --output    Ruta donde se guarda el artefacto entrenado. Por defecto,
                models/model.pkl.

Salidas generadas:

    - El artefacto serializado en la ruta indicada por --output,
      conteniendo el modelo entrenado, la lista de columnas de features
      utilizadas y el valor de calibration_factor.
    - Un fichero models/metrics.json con las métricas de evaluación
      (MAE y RMSE de LightGBM y del modelo de referencia Ridge, así como
      la importancia de cada variable).
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from autocalls.data.loading import load_all
from autocalls.data.integration import run_integration_pipeline, build_model_table
from autocalls.model.split import temporal_train_test_split

# Valor de calibración requerido explícitamente por el enunciado del
# proyecto (sección 4). No constituye un hiperparámetro del modelo ni
# tiene efecto alguno sobre el proceso de entrenamiento: se incorpora al
# artefacto y se reporta en el README exclusivamente para cumplir con la
# especificación del entregable.
CALIBRATION_FACTOR = 3.5


def train_and_evaluate(raw_dir: str, output_path: str, seed: int = 42) -> dict:
    """
    Ejecuta el proceso completo de entrenamiento y evaluación del modelo.

    Etapas:

    1. Carga de las tres tablas de origen (autocalls.data.loading).
    2. Integración y limpieza de datos (autocalls.data.integration).
    3. Partición temporal en entrenamiento y evaluación
       (autocalls.model.split), reservando aproximadamente el 18% más
       reciente de las observaciones para evaluación.
    4. Entrenamiento de un modelo de referencia (Ridge, regresión lineal
       regularizada) y del modelo final (LightGBM, gradient boosting).
       El modelo de referencia se entrena únicamente con fines
       comparativos, para cuantificar la mejora aportada por el modelo
       final; no se incluye en el artefacto guardado.
    5. Cálculo de métricas de error (MAE y RMSE) sobre el conjunto de
       evaluación para ambos modelos, y de un baseline adicional que
       consiste en predecir sistemáticamente la media del conjunto de
       entrenamiento, empleado como referencia mínima de comparación.
    6. Cálculo de la importancia de cada variable en el modelo final.
    7. Serialización del modelo final, la lista de columnas de features
       y el valor de calibration_factor en un único artefacto.

    Parámetros
    ----------
    raw_dir : str
        Directorio que contiene los tres ficheros CSV de origen.
    output_path : str
        Ruta donde se guarda el artefacto serializado.
    seed : int, opcional (por defecto 42)
        Semilla aleatoria empleada en el entrenamiento de ambos modelos,
        para garantizar la reproducibilidad de los resultados.

    Devuelve
    --------
    dict
        Diccionario con las métricas de evaluación, los rangos de fecha
        de cada conjunto, y la importancia de cada variable en el modelo
        final.
    """
    rfqs, daily_volatility, underlyings_reference = load_all(raw_dir)

    df_model = run_integration_pipeline(rfqs, daily_volatility, underlyings_reference)
    train_df, test_df = temporal_train_test_split(df_model)

    X_train, y_train, _ = build_model_table(train_df)
    X_test, y_test, _ = build_model_table(test_df, reference_columns=X_train.columns)

    # Modelo de referencia (Ridge): se reporta en las métricas para
    # cuantificar la mejora del modelo final. No se serializa.
    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)

    model = LGBMRegressor(random_state=seed, verbosity=-1)
    model.fit(X_train, y_train)
    pred_model = model.predict(X_test)

    metrics = {
        "ridge_baseline": {
            "mae_months": float(mean_absolute_error(y_test, pred_ridge)),
            "rmse_months": float(root_mean_squared_error(y_test, pred_ridge)),
        },
        "lightgbm": {
            "mae_months": float(mean_absolute_error(y_test, pred_model)),
            "rmse_months": float(root_mean_squared_error(y_test, pred_model)),
        },
        "naive_mean_baseline_mae_months": float(
            np.abs(y_test - y_train.mean()).mean()
        ),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_date_range": [
            str(train_df["requested_date"].min().date()),
            str(train_df["requested_date"].max().date()),
        ],
        "test_date_range": [
            str(test_df["requested_date"].min().date()),
            str(test_df["requested_date"].max().date()),
        ],
    }

    feature_importance = (
        pd.Series(model.feature_importances_, index=X_train.columns)
        .sort_values(ascending=False)
    )
    metrics["feature_importance"] = feature_importance.to_dict()

    artifact = {
        "model": model,
        "feature_columns": list(X_train.columns),
        "calibration_factor": CALIBRATION_FACTOR,
    }
    joblib.dump(artifact, output_path)

    return metrics


def main():
    """
    Punto de entrada del script. Procesa los argumentos de línea de
    comandos, ejecuta train_and_evaluate, e imprime en consola un resumen
    de las métricas obtenidas, además de guardar el detalle completo en
    models/metrics.json.
    """
    parser = argparse.ArgumentParser(description="Entrena el modelo de STARWARS_AUTOCALLS.")
    parser.add_argument("--raw-dir", default="data/raw", help="Carpeta con los 3 CSV de origen.")
    parser.add_argument("--output", default="models/model.pkl", help="Ruta de salida del artefacto.")
    args = parser.parse_args()

    metrics = train_and_evaluate(args.raw_dir, args.output)

    print(f"Artefacto guardado en: {args.output}")
    print()
    print("=== Métricas (test, split temporal) ===")
    print(f"Naive (predecir media de train): MAE = {metrics['naive_mean_baseline_mae_months']:.2f} meses")
    print(f"Ridge:     MAE = {metrics['ridge_baseline']['mae_months']:.2f} | RMSE = {metrics['ridge_baseline']['rmse_months']:.2f}")
    print(f"LightGBM:  MAE = {metrics['lightgbm']['mae_months']:.2f} | RMSE = {metrics['lightgbm']['rmse_months']:.2f}")
    print()
    print("=== Feature importance (LightGBM) ===")
    for feat, imp in list(metrics["feature_importance"].items())[:10]:
        print(f"  {feat:35s} {imp}")

    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print()
    print("Métricas completas guardadas en models/metrics.json")


if __name__ == "__main__":
    main()