"""
loading.py

Lectura de los 3 CSVs de origen del proyecto STARWARS_AUTOCALLS. Responsabilidad única: leer del disco y devolver DataFrames con los tipos correctos.
No aplica ninguna limpieza ni lógica de negocio — eso vive en integration.py.
"""

import pandas as pd

def load_rfqs(path: str) -> pd.DataFrame:
    """Lee rfqs.csv con las columnas de fecha parseadas como datetime."""
    return pd.read_csv(
        path,
        parse_dates=["requested_date", "start_date", "end_date"],
    )


def load_daily_volatility(path: str) -> pd.DataFrame:
    """Lee daily_volatility.csv con la columna date parseada como datetime."""
    return pd.read_csv(path, parse_dates=["date"])


def load_underlyings_reference(path: str) -> pd.DataFrame:
    """Lee underlyings_reference.csv (tabla estática, sin fechas)."""
    return pd.read_csv(path)


def load_all(raw_dir: str):
    """
    Lee las 3 tablas de origen desde un directorio raw_dir que debe contener rfqs.csv, daily_volatility.csv y underlyings_reference.csv.
    Devuelve (rfqs, daily_volatility, underlyings_reference).
    """
    rfqs = load_rfqs(f"{raw_dir}/rfqs.csv")
    daily_volatility = load_daily_volatility(f"{raw_dir}/daily_volatility.csv")
    underlyings_reference = load_underlyings_reference(f"{raw_dir}/underlyings_reference.csv")
    return rfqs, daily_volatility, underlyings_reference
