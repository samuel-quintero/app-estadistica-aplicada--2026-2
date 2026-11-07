"""Página de distribuciones de precipitación y modelos de secado."""

import streamlit as st

from src.datos import tarjeta_estudiante


# Se identifica al estudiante en la barra lateral de esta página.
tarjeta_estudiante()

# Se presenta el alcance de la página mientras se implementan sus herramientas.
st.title("Distribuciones")
st.info("En construcción")
st.markdown(
    """
### Temas
- Distribuciones Normal, Lognormal, Gamma, Weibull, Gumbel y GEV.
- Ajuste de máximos anuales de precipitación en 24 horas.
- Pruebas Kolmogorov-Smirnov y Anderson-Darling.
- Comparación mediante AIC, BIC y AICc.
- Cuantiles y periodo de retorno, riesgo e incertidumbre.
- Modelos de secado en capa delgada y comparación con AICc.
"""
)