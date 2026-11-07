"""Página de correlación, regresión lineal y diagnóstico."""

import streamlit as st

from src.datos import tarjeta_estudiante


# Se identifica al estudiante en la barra lateral de esta página.
tarjeta_estudiante()

# Se presenta el alcance de la página mientras se implementan sus herramientas.
st.title("Regresión")
st.info("En construcción")
st.markdown(
    """
### Temas
- Correlaciones de Pearson y Spearman.
- Regresión lineal simple por mínimos cuadrados ordinarios (MCO).
- Bondad de ajuste y análisis de varianza.
- Inferencia, intervalo de confianza y prueba sobre la pendiente.
- Diagnóstico de residuos: linealidad, normalidad, varianza e independencia.
"""
)