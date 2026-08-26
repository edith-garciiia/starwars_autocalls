"""
test_integration.py

Tests de humo (smoke tests) del pipeline de integración de datos del
proyecto STARWARS_AUTOCALLS.

Estos tests no pretenden verificar exhaustivamente cada función del
pipeline, sino comprobar, de forma automatizada y reproducible, que el
proceso completo (lectura de los CSV de origen, limpieza, integración y
construcción de la matriz de modelado) se ejecuta sin errores sobre los
datos reales del proyecto y produce resultados con las propiedades
mínimas esperadas: sin nulos en el target, sin pérdida ni duplicación de
filas, y con consistencia de columnas entre entrenamiento y evaluación.

Ejecución:

    pytest tests/
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from autocalls.data.loading import load_all
from autocalls.data.integration import run_integration_pipeline, build_model_table
from autocalls.model.split import temporal_train_test_split

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def test_pipeline_end_to_end():
    """
    Verifica que el pipeline de integración, ejecutado de principio a
    fin sobre los ficheros CSV reales del proyecto, produce una tabla en
    la que todas las filas corresponden a RFQs ejecutadas, ninguna
    carece de variable objetivo, y ninguna carece de las variables de
    riesgo de cesta derivadas del cruce con los datos de mercado.
    """
    rfqs, vol, ref = load_all(RAW_DIR)
    df_model = run_integration_pipeline(rfqs, vol, ref)

    assert df_model["executed"].all()
    assert df_model["avg_duration_months"].isna().sum() == 0

    assert df_model["basket_max_realized_vol"].isna().sum() == 0
    assert df_model["basket_max_structural_vol"].isna().sum() == 0


def test_split_no_overlap():
    """
    Verifica que la partición temporal en entrenamiento y evaluación no
    presenta solapamiento de fechas (la fecha máxima de entrenamiento es
    estrictamente anterior a la fecha mínima de evaluación), y que la
    suma de observaciones de ambos conjuntos coincide exactamente con el
    total de la tabla integrada, sin pérdida ni duplicación de filas.
    """
    rfqs, vol, ref = load_all(RAW_DIR)
    df_model = run_integration_pipeline(rfqs, vol, ref)
    train_df, test_df = temporal_train_test_split(df_model)

    assert train_df["requested_date"].max() < test_df["requested_date"].min()
    assert len(train_df) + len(test_df) == len(df_model)


def test_build_model_table_consistent_columns():
    """
    Verifica que, al construir la matriz de features del conjunto de
    evaluación con reference_columns fijado a las columnas del conjunto
    de entrenamiento, ambas matrices resultan con exactamente el mismo
    conjunto y orden de columnas, y que el número de filas de X coincide
    con el de y en ambos conjuntos.
    """
    rfqs, vol, ref = load_all(RAW_DIR)
    df_model = run_integration_pipeline(rfqs, vol, ref)
    train_df, test_df = temporal_train_test_split(df_model)

    X_train, y_train, _ = build_model_table(train_df)
    X_test, y_test, _ = build_model_table(test_df, reference_columns=X_train.columns)

    assert list(X_train.columns) == list(X_test.columns)
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)