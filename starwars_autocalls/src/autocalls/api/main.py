"""
main.py

API de inferencia del proyecto STARWARS_AUTOCALLS.

Expone el modelo entrenado a través de dos endpoints HTTP, construidos
con FastAPI:

- GET /health: estado del servicio (disponibilidad del modelo, versión
  del paquete, y build_tag).
- POST /predict: predicción de avg_duration_months para una RFQ dada.

Esta API constituye la respuesta al requisito del enunciado de disponer
de "una API sencilla que permita hacer inferencia con el modelo ya
entrenado" y de "levantar la API de inferencia en local".

Modo de uso:

    uvicorn autocalls.api.main:app --reload

La documentación interactiva de la API (generada automáticamente por
FastAPI) queda disponible en http://127.0.0.1:8000/docs una vez el
servicio está en ejecución.

Nota de diseño: el endpoint /predict recibe las features ya calculadas
por el pipeline de integración (incluidas las variables de riesgo de
cesta derivadas del cruce con los paneles de mercado), en lugar de los
campos crudos de la RFQ. Esta decisión evita que la API tenga que cargar
en memoria el histórico completo de daily_volatility en cada arranque
únicamente para atender peticiones de inferencia. En un despliegue
productivo, un servicio previo —o el propio sistema de front-office—
invocaría autocalls.data.integration para calcular estas features a
partir de la RFQ en bruto antes de invocar a /predict.
"""

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from autocalls import __version__
from autocalls.model.predict import load_model, predict_duration

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.pkl")

# build_tag requerido explícitamente por el enunciado del proyecto
# (sección 4). No corresponde a un identificador real de compilación ni
# tiene efecto funcional: se expone tal cual se especifica.
BUILD_TAG = "fenetre-glissante-v2"

app = FastAPI(
    title="STARWARS_AUTOCALLS Inference API",
    version=__version__,
    description="Predicción de avg_duration_months para autocallables (Banco Imperial de Coruscant).",
)

_loaded_model = None


def get_model():
    """
    Devuelve el modelo cargado en memoria, cargándolo desde disco en la
    primera invocación (patrón de carga perezosa, para no leer el
    artefacto en cada petición).

    Excepciones
    -----------
    HTTPException (503)
        Si el artefacto del modelo no se encuentra en MODEL_PATH.
    """
    global _loaded_model
    if _loaded_model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503,
                detail=f"Modelo no encontrado en {MODEL_PATH}. Ejecuta primero el entrenamiento.",
            )
        _loaded_model = load_model(MODEL_PATH)
    return _loaded_model


class RFQFeatures(BaseModel):
    """
    Esquema de entrada del endpoint /predict.

    Representa las features de una RFQ ya procesadas por el pipeline de
    integración (autocalls.data.integration.run_integration_pipeline y
    build_model_table). FastAPI valida automáticamente, a partir de este
    esquema, que toda petición entrante contenga estos campos con el
    tipo de dato correcto, rechazando con un error explícito cualquier
    petición que no lo cumpla.
    """

    autocall_barrier_pct: float = Field(..., example=1.05)
    protection_barrier_pct: float = Field(..., example=0.70)
    no_call_period_months: int = Field(..., example=6)
    quoted_implied_vol: float = Field(..., example=0.28)
    notional_credits: float = Field(..., example=1_000_000)
    obs_freq_months: float = Field(..., example=3.0)
    basket_max_realized_vol: float = Field(..., example=0.30)
    basket_max_structural_vol: float = Field(..., example=0.32)
    nominal_months: float = Field(..., example=36.0)
    n_underlyings: int = Field(..., example=2)
    product_type: str = Field(..., example="Kessel Run Snowball")
    basket_type: str = Field(..., example="worst_of")


class PredictionResponse(BaseModel):
    """Esquema de salida del endpoint /predict."""

    predicted_avg_duration_months: float


@app.get("/health")
def health():
    """
    Endpoint de estado del servicio.

    Informa de si el artefacto del modelo está disponible en disco, de
    la versión del paquete (autocalls.__version__) y del build_tag
    requerido por el enunciado del proyecto.

    Devuelve
    --------
    dict
        status: "ok" si el modelo está disponible, "model_not_found" en
        caso contrario.
        build_tag: identificador de build requerido por el enunciado.
        version: versión del paquete (autocalls.__version__).
        model_loaded: booleano indicando si el artefacto existe en disco.
    """
    model_loaded = os.path.exists(MODEL_PATH)
    return {
        "status": "ok" if model_loaded else "model_not_found",
        "build_tag": BUILD_TAG,
        "version": __version__,
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(rfq: RFQFeatures):
    """
    Endpoint de inferencia.

    Recibe las features de una RFQ (validadas contra el esquema
    RFQFeatures) y devuelve la duración media estimada del producto, en
    meses.

    Parámetros
    ----------
    rfq : RFQFeatures
        Features de la RFQ sobre la que se desea predecir.

    Devuelve
    --------
    PredictionResponse
        Objeto con el campo predicted_avg_duration_months.
    """
    loaded = get_model()
    prediction = predict_duration(loaded, rfq.model_dump())
    return PredictionResponse(predicted_avg_duration_months=round(prediction, 2))