"""Página de análisis exploratorio de datos."""

import streamlit as st

from src.datos import tarjeta_estudiante


# Se identifica al estudiante en la barra lateral de esta página.
tarjeta_estudiante()

# Se presenta el alcance de la página mientras se implementan sus herramientas.
st.title("EDA: Análisis Exploratorio de Datos")
st.info("En construcción")
st.markdown(
    """
### Temas
- Estadísticos descriptivos: centro, dispersión, cuantiles y forma.
- Distribuciones con histogramas y densidades.
- Diagramas de caja y detección exploratoria de valores atípicos.
- Relaciones entre variables, dispersión y correlaciones.
- Exploración temporal y resúmenes por grupos.
"""
)