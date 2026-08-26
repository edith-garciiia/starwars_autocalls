"""
predict.py

Módulo de inferencia del proyecto STARWARS_AUTOCALLS.

Contiene la lógica necesaria para cargar el artefacto de modelo
entrenado (generado por autocalls.model.train) y obtener una predicción
de avg_duration_months a partir de los datos de una RFQ nueva.

Esta lógica se aísla en un módulo independiente, en lugar de
incorporarse directamente en la API, para que pueda ser reutilizada por
cualquier otro proceso que necesite realizar inferencia (por ejemplo,
scripts de evaluación por lotes o pruebas automatizadas), evitando
duplicar la lógica de carga del modelo y de construcción de features en
más de un lugar del proyecto.
"""

from dataclasses import dataclass
from typing import List

import joblib
import pandas as pd

from autocalls.data.integration import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


@dataclass
class LoadedModel:
    """
    Representa un modelo entrenado ya cargado en memoria, junto con los
    metadatos necesarios para realizar inferencia de forma consistente
    con el entrenamiento.

    Atributos
    ---------
    model : object
        Instancia del modelo entrenado (LightGBM), con un método
        predict compatible con la interfaz de scikit-learn.
    feature_columns : list de str
        Lista ordenada de las columnas de features utilizadas durante el
        entrenamiento. Toda fila que se le pase al modelo debe respetar
        exactamente este mismo conjunto y orden de columnas.
    calibration_factor : float
        Valor de calibración del proyecto (metadato requerido por el
        enunciado; no interviene en el cálculo de la predicción).
    """

    model: object
    feature_columns: List[str]
    calibration_factor: float


def load_model(path: str) -> LoadedModel:
    """
    Carga el artefacto de modelo serializado por autocalls.model.train.

    Parámetros
    ----------
    path : str
        Ruta al fichero del artefacto (por defecto, models/model.pkl).

    Devuelve
    --------
    LoadedModel
        Objeto con el modelo, las columnas de features esperadas y el
        valor de calibration_factor.
    """
    artifact = joblib.load(path)
    return LoadedModel(
        model=artifact["model"],
        feature_columns=artifact["feature_columns"],
        calibration_factor=artifact["calibration_factor"],
    )


def build_features_from_raw_inputs(record: dict, feature_columns: List[str]) -> pd.DataFrame:
    """
    Construye la fila de features en el formato exacto que espera el
    modelo, a partir de los datos de una RFQ representados como
    diccionario.

    Aplica la misma codificación one-hot de las variables categóricas
    (product_type, basket_type) empleada durante el entrenamiento, y
    reindexa el resultado a feature_columns para garantizar que la fila
    resultante tenga exactamente las mismas columnas, en el mismo orden,
    que las utilizadas al entrenar el modelo. Esta correspondencia exacta
    es indispensable: un modelo entrenado no interpreta los nombres de
    las columnas en el momento de predecir, sino su posición; una fila
    con columnas en un orden distinto al de entrenamiento produciría una
    predicción incorrecta sin generar ningún error visible.

    Parámetros
    ----------
    record : dict
        Diccionario con los valores de la RFQ. Debe incluir una clave
        por cada variable definida en NUMERIC_FEATURES y en
        CATEGORICAL_FEATURES (autocalls.data.integration).
    feature_columns : list de str
        Columnas de features esperadas por el modelo, en el orden
        utilizado durante el entrenamiento.

    Devuelve
    --------
    pd.DataFrame
        DataFrame de una única fila, con las columnas alineadas a
        feature_columns.
    """
    row = pd.DataFrame([record])

    X_numeric = row[NUMERIC_FEATURES].copy()
    X_categorical = pd.get_dummies(row[CATEGORICAL_FEATURES], drop_first=False)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    X.columns = X.columns.str.replace(" ", "_")
    X = X.reindex(columns=feature_columns, fill_value=0)

    return X


def predict_duration(loaded: LoadedModel, record: dict) -> float:
    """
    Predice avg_duration_months para una única RFQ.

    Parámetros
    ----------
    loaded : LoadedModel
        Modelo cargado mediante load_model.
    record : dict
        Datos de la RFQ sobre la que se desea predecir.

    Devuelve
    --------
    float
        Duración media estimada, en meses.
    """
    X = build_features_from_raw_inputs(record, loaded.feature_columns)
    prediction = loaded.model.predict(X)[0]
    return float(prediction)
