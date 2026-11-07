"""Página de revisión y preparación de datos."""

import streamlit as st

from src.datos import tarjeta_estudiante


# Se identifica al estudiante en la barra lateral de esta página.
tarjeta_estudiante()

# Se presenta el alcance de la página mientras se implementan sus herramientas.
st.title("Datos")
st.info("En construcción")
st.markdown(
    """
### Temas
- Carga de archivos CSV y Excel.
- Vista previa, tipos de variables, unidades y validación de columnas.
- Revisión de valores faltantes, duplicados y valores fuera de rango.
- Preparación de datos sin modificar los archivos originales.
"""
)