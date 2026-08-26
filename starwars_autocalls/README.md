# STARWARS_AUTOCALLS

Predicción de `avg_duration_months` (duración media real hasta cancelación
o vencimiento) para productos autocallables — Banco Imperial de Coruscant,
División de Estructuración Cuantitativa.

## Metadatos del proyecto

| Campo | Valor |
|---|---|
| Versión del paquete | `3.14.15` |
| calibration_factor | `3.5` |
| build_tag (API) | `fenetre-glissante-v2` |

## Estructura del repositorio

```
starwars_autocalls/
├── data/raw/                    # CSVs de origen (rfqs, daily_volatility, underlyings_reference)
├── notebooks/EDA.ipynb          # Exploración y justificación de las decisiones de limpieza/features
├── src/autocalls/
│   ├── __init__.py              # __version__ = "3.14.15"
│   ├── data/
│   │   ├── loading.py           # Lectura de los 3 CSV
│   │   └── integration.py       # Limpieza + integración + construcción de X/y
│   ├── model/
│   │   ├── split.py             # Split temporal train/test
│   │   ├── train.py             # Script de entrenamiento
│   │   └── predict.py           # Carga de artefacto + inferencia
│   └── api/
│       └── main.py              # API FastAPI (/predict, /health)
├── models/
│   ├── model.pkl                # Artefacto entrenado (incluido, no hace falta reentrenar)
│   └── metrics.json             # Métricas y feature importance del último entrenamiento
├── tests/test_integration.py    # Tests de humo del pipeline
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenar el modelo desde los CSV de origen

```bash
PYTHONPATH=src python -m autocalls.model.train --raw-dir data/raw --output models/model.pkl
```

Esto reproduce todo el pipeline (integración de las 3 tablas → split
temporal 82/18 → entrenamiento LightGBM) y guarda:
- `models/model.pkl`: el artefacto serializado (modelo + columnas de
  features + calibration_factor).
- `models/metrics.json`: MAE/RMSE de test (LightGBM y el baseline Ridge de
  referencia) y el feature importance completo.

No es necesario ejecutar este paso para probar la API: el artefacto ya
entrenado se incluye en el repositorio.

## Levantar la API de inferencia en local

```bash
PYTHONPATH=src uvicorn autocalls.api.main:app --reload
```

Documentación interactiva en `http://127.0.0.1:8000/docs`.

### GET /health

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "build_tag": "fenetre-glissante-v2", "version": "3.14.15", "model_loaded": true}
```

### POST /predict

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "autocall_barrier_pct": 1.0,
    "protection_barrier_pct": 0.70,
    "no_call_period_months": 6,
    "quoted_implied_vol": 0.28,
    "notional_credits": 1000000,
    "obs_freq_months": 3.0,
    "basket_max_realized_vol": 0.30,
    "basket_max_structural_vol": 0.32,
    "nominal_months": 36.0,
    "n_underlyings": 2,
    "product_type": "Kessel Run Snowball",
    "basket_type": "worst_of"
  }'
```

```json
{"predicted_avg_duration_months": 22.63}
```

**Nota de diseño**: el endpoint recibe las features ya calculadas por el
pipeline de integración (incluidas las de volatilidad de la cesta), en vez
de los campos crudos de la RFQ. Esto mantiene la API ligera, sin necesidad
de cargar en memoria el histórico completo de `daily_volatility` en cada
petición. En un despliegue productivo, un servicio previo (o el propio
front-office) invocaría `autocalls.data.integration` para calcular estas
features a partir de la RFQ cruda antes de llamar a `/predict`.

## Resultados

Split temporal (train: 2016-01-04 a 2022-12-13, test: 2022-12-14 a
2024-06-28), sin cambio de régimen detectado en el EDA a lo largo del
periodo.

| Modelo | MAE (meses) | RMSE (meses) |
|---|---|---|
| Naive (predecir la media de train) | 18.14 | — |
| Ridge (lineal regularizado) | 10.33 | 13.71 |
| **LightGBM (modelo final)** | **4.74** | **6.42** |

LightGBM reduce el error en un 74% frente al baseline naive y en un 54%
frente a un modelo lineal, indicando relaciones no lineales relevantes
entre las features y la duración (p. ej. interacción entre barrera de
autocall, frecuencia de observación y volatilidad).

### Variables más importantes (feature importance, LightGBM)

1. `nominal_months` — duración nominal del producto (end_date - start_date). Es el techo natural de la duración real y la variable individual más correlacionada con el target (corr ≈ 0.53 en el EDA).
2. `basket_max_realized_vol` / `basket_max_structural_vol` — riesgo de la cesta (máximo entre los subyacentes, coherente con la lógica worst-of: basta con que uno se mueva mucho).
3. `autocall_barrier_pct` — una barrera más baja facilita la cancelación anticipada, acortando la duración esperada.
4. `quoted_implied_vol`, `protection_barrier_pct`, `obs_freq_months` — importancia moderada, todas con lectura de negocio razonable (más volatilidad u observaciones más frecuentes → más oportunidades de autocall anticipado).

Detalle completo en `models/metrics.json` tras cada entrenamiento.

## Limitaciones conocidas

- **`Wretched Hive Digital`**: un 7.6% de las RFQs ejecutadas de este
  producto tienen `avg_duration_months` por encima de la duración
  nominal, lo cual no debería ser posible según la lógica de negocio del
  autocallable. El patrón sugiere una liquidación diferida tras
  vencimiento no capturada por ninguna columna disponible en los datos de
  origen. Se ha optado por mantener el target sin modificar (no filtrar
  ni recortar, para no alterar información real) y documentarlo aquí: el
  modelo puede subestimar ligeramente la duración en ese subconjunto.
- **Ausencia de datos de precio/nivel de los subyacentes**: el enunciado
  define la condición de autocall en términos del nivel del subyacente
  más débil de la cesta, pero las tablas de origen solo incluyen
  volatilidad (no precio). Las features de cesta (`basket_max_*`) son una
  aproximación de riesgo basada en volatilidad, no una réplica exacta de
  la lógica de negocio real del producto.
- **`counterparty` y `trader_id` se descartaron** por no mostrar relación
  con el target (varianza de medias ≈0.04% y ≈0.45% de la varianza total
  del target) — decisión documentada en el EDA, no un descuido.
- El modelo no ha sido sometido a una búsqueda de hiperparámetros ni a
  validación cruzada temporal (solo un split simple); ambas son mejoras
  naturales para una siguiente iteración.
