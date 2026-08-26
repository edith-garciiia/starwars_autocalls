"""
split.py

Split train/test para STARWARS_AUTOCALLS.
"""

import pandas as pd


def temporal_train_test_split(df_model: pd.DataFrame, test_frac: float = 0.18):
    """
    Divide df_model en train/test por fecha (requested_date), no de forma
    aleatoria. No se detectó cambio de régimen a lo largo del tiempo en el
    EDA (duración media, vol implícita y mezcla de product_type estables
    año a año), pero un split temporal sigue siendo la opción más
    rigurosa: reproduce cómo se usaría el modelo en producción (entrenar
    con el histórico, predecir sobre RFQs futuras).

    test_frac=0.18 deja aproximadamente el 18% más reciente como test.

    Devuelve (train_df, test_df), ambos con las mismas columnas que df_model.
    """
    df_sorted = df_model.sort_values("requested_date")
    cutoff = df_sorted["requested_date"].quantile(1 - test_frac)

    train_df = df_sorted[df_sorted["requested_date"] <= cutoff].copy()
    test_df = df_sorted[df_sorted["requested_date"] > cutoff].copy()

    return train_df, test_df
