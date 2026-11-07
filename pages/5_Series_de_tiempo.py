"""Página de análisis y pronóstico de series de tiempo."""

import streamlit as st

from src.datos import tarjeta_estudiante


# Se identifica al estudiante en la barra lateral de esta página.
tarjeta_estudiante()

# Se presenta el alcance de la página mientras se implementan sus herramientas.
st.title("Series de tiempo")
st.info("En construcción")
st.markdown(
    """
### Temas
- Componentes de una serie y promedios móviles.
- Índices estacionales y descomposición clásica.
- Métodos de referencia: último valor, promedio, deriva y método estacional.
- Validación temporal con MAE, RMSE y SMAPE.
- Pronóstico con AutoTS y comparación con métodos de referencia.
"""
)