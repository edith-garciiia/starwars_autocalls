"""
integration.py

Pipeline de integración de las 3 tablas de origen (rfqs, daily_volatility,
underlyings_reference) para el proyecto STARWARS_AUTOCALLS.

Decisiones de limpieza y de negocio documentadas en el EDA (notebooks/EDA.ipynb,
secciones 1-13). Resumen:

- Solo las RFQs con executed=True tienen target y se usan para entrenar; las
  no ejecutadas no aportan señal (tasa de ejecución constante aproximada del 55% frente a
  cualquier característica de la RFQ).
- observation_frequency tenía 18 etiquetas sucias que colapsan a 6
  categorías reales; se normalizan a meses entre observaciones.
- Un subconjunto de filas (concentrado en el producto "Wretched Hive
  Digital") tiene avg_duration_months por encima de la duración nominal.
  Se mantiene el target sin modificar (probable liquidación diferida no
  capturada por ninguna columna disponible) y se documenta como limitación
  del modelo, no se filtra ni se recorta.
- autocall_barrier_pct fuera del rango "típico" está totalmente explicado
  por product_type (cada estructura tiene su propia convención de barrera):
  no requiere limpieza.
- La cesta de subyacentes (worst_of, 2-3 activos) se agrega a nivel de RFQ
  tomando el máximo de volatilidad realizada y estructural: en un worst_of
  basta con que un subyacente se mueva mucho para determinar el resultado.
  El sector se descartó como proxy de correlación porque la diversidad
  sectorial es máxima por diseño en el 100% de las cestas (no aporta señal
  independiente del tamaño de la cesta).
- El cruce con daily_volatility se hace por requested_date (evita leakage,
  nunca se usa información posterior a la cotización), con merge exacto
  porque el 100% de las requested_date coinciden con un día de mercado.
"""

import pandas as pd


_FREQ_MAP = {
    "1M": 1, "Monthly": 1, "mensual": 1, "1 month": 1, "M": 1,
    "2M": 2,
    "3M": 3, "Quarterly": 3, "trimestral": 3, "Q": 3, "3 months": 3,
    "6M": 6,
    "1Y": 12, "12M": 12, "Y": 12, "Annual": 12, "anual": 12,
    "1D": 1 / 30,
}

NUMERIC_FEATURES = [
    "autocall_barrier_pct",
    "protection_barrier_pct",
    "no_call_period_months",
    "quoted_implied_vol",
    "notional_credits",
    "obs_freq_months",
    "basket_max_realized_vol",
    "basket_max_structural_vol",
    "nominal_months",
    "n_underlyings",
]
CATEGORICAL_FEATURES = ["product_type", "basket_type"]
TARGET_COLUMN = "avg_duration_months"
ID_COLUMN = "rfq_id"


def filter_executed(rfqs: pd.DataFrame) -> pd.DataFrame:
    """
    Se queda solo con las RFQs ejecutadas (executed=True), que son las
    únicas que tienen avg_duration_months (target).
    """
    return rfqs[rfqs["executed"]].copy()


def add_derived_features(rfqs: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features derivadas de columnas ya existentes:
    - nominal_months: duración nominal (end_date - start_date), la
      variable individual más correlacionada con el target en el EDA
      (corr ~= 0.53).
    - n_underlyings: tamaño de la cesta (1, 2 o 3).
    """
    rfqs = rfqs.copy()
    rfqs["nominal_months"] = (rfqs["end_date"] - rfqs["start_date"]).dt.days / 30.44
    rfqs["n_underlyings"] = rfqs["underlyings"].str.count(r"\|") + 1
    return rfqs


def normalize_observation_frequency(rfqs: pd.DataFrame) -> pd.DataFrame:
    """
    Añade obs_freq_months a partir de observation_frequency, normalizando
    las ~18 etiquetas distintas a un valor numérico de meses entre
    observaciones. Lanza un error si aparece alguna etiqueta no
    contemplada, en vez de dejarla pasar como NaN silenciosamente.
    """
    rfqs = rfqs.copy()
    rfqs["obs_freq_months"] = rfqs["observation_frequency"].map(_FREQ_MAP)

    sin_mapear = rfqs.loc[rfqs["obs_freq_months"].isna(), "observation_frequency"].unique()
    if len(sin_mapear) > 0:
        raise ValueError(f"Etiquetas de observation_frequency sin mapear: {sin_mapear}")

    return rfqs


def explode_underlyings(rfqs: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte underlyings (string separado por '|') en una fila por cada
    subyacente de la cesta, manteniendo rfq_id como clave para poder
    volver a agregar más adelante.
    """
    rfqs = rfqs.copy()
    rfqs["underlying"] = rfqs["underlyings"].str.split("|")
    return rfqs.explode("underlying", ignore_index=True)


def merge_market_data(
    exploded: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    underlyings_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cruza la tabla explotada con daily_volatility (por underlying + fecha,
    requested_date == date) y con underlyings_reference (por underlying).
    Merge exacto por fecha: el 100% de las requested_date coinciden con un
    día de mercado presente en daily_volatility (comprobado en el EDA).
    """
    merged = exploded.merge(
        daily_volatility[["date", "underlying", "realized_vol_63d"]],
        left_on=["requested_date", "underlying"],
        right_on=["date", "underlying"],
        how="left",
    )
    merged = merged.merge(
        underlyings_reference[["underlying", "sector", "structural_base_vol"]],
        on="underlying",
        how="left",
    )

    faltantes = merged["realized_vol_63d"].isna().sum()
    if faltantes > 0:
        raise ValueError(
            f"{faltantes} filas sin dato de volatilidad tras el cruce — "
            "revisar cobertura de fechas/tickers."
        )

    return merged


def aggregate_basket_risk(merged: pd.DataFrame, rfqs: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega de vuelta a nivel de RFQ: máximo de realized_vol_63d y de
    structural_base_vol dentro de cada cesta.
    """
    basket_risk = merged.groupby("rfq_id").agg(
        basket_max_realized_vol=("realized_vol_63d", "max"),
        basket_max_structural_vol=("structural_base_vol", "max"),
    )

    result = rfqs.merge(basket_risk, on="rfq_id", how="left")

    faltantes = result[["basket_max_realized_vol", "basket_max_structural_vol"]].isna().any(axis=1).sum()
    if faltantes > 0:
        raise ValueError(f"{faltantes} RFQs sin features de cesta tras la agregación.")

    return result


def run_integration_pipeline(
    rfqs: pd.DataFrame,
    daily_volatility: pd.DataFrame,
    underlyings_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encadena el pipeline completo de integración:
    1. Filtra a solo RFQs ejecutadas.
    2. Añade features derivadas (nominal_months, n_underlyings).
    3. Normaliza observation_frequency.
    4. Explota underlyings a una fila por (RFQ, subyacente).
    5. Cruza con daily_volatility y underlyings_reference.
    6. Agrega de vuelta a nivel de RFQ.

    Devuelve una tabla a nivel de RFQ, lista para feature engineering y
    entrenamiento, con avg_duration_months como target.
    """
    rfqs_executed = filter_executed(rfqs)
    rfqs_derived = add_derived_features(rfqs_executed)
    rfqs_norm = normalize_observation_frequency(rfqs_derived)
    exploded = explode_underlyings(rfqs_norm)
    merged = merge_market_data(exploded, daily_volatility, underlyings_reference)
    final = aggregate_basket_risk(merged, rfqs_norm)

    return final


def build_model_table(df_model: pd.DataFrame, reference_columns=None):
    """
    A partir del resultado de run_integration_pipeline, construye X, y, ids.

    Columnas descartadas y por qué:
    - underlyings, observation_frequency: sustituidas por features
      derivadas (n_underlyings, basket_max_*, obs_freq_months).
    - counterparty, trader_id: sin relación con el target (varianza de
      medias ~0.04% y ~0.45% de la varianza total del target).
    - requested_date, start_date, end_date, executed: sin valor directo
      como feature (ya se extrajo nominal_months); executed es constante
      tras el filtro.

    Si reference_columns se proporciona, X se reindexa a esas columnas
    exactas (rellenando con 0 lo que falte), para que train y test queden
    con las mismas columnas tras el one-hot encoding.
    """
    X_numeric = df_model[NUMERIC_FEATURES].copy()
    X_categorical = pd.get_dummies(df_model[CATEGORICAL_FEATURES], drop_first=True)
    X = pd.concat([X_numeric, X_categorical], axis=1)
    X.columns = X.columns.str.replace(" ", "_")

    if reference_columns is not None:
        X = X.reindex(columns=reference_columns, fill_value=0)

    y = df_model[TARGET_COLUMN].copy()
    ids = df_model[ID_COLUMN].copy()

    return X, y, ids
