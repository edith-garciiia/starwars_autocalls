"""
autocalls

Paquete principal del proyecto STARWARS_AUTOCALLS (Banco Imperial de
Coruscant — estimación de duración de productos autocallables).

El proyecto se organiza en tres subpaquetes, cada uno con una única
responsabilidad:

- autocalls.data  — carga e integración de las tablas de origen.
- autocalls.model — split, entrenamiento e inferencia del modelo.
- autocalls.api   — servicio de inferencia (FastAPI).

__version__ sigue la convención estándar de Python para exponer la versión instalada del paquete (equivalente a pandas.__version__ o
numpy.__version__). Se centraliza aquí para que cualquier otro módulo del proyecto consulte un único origen de verdad, en lugar de duplicar el 
número de versión en varios archivos.
"""

__version__ = "3.14.15"
