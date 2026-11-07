"""Página de inicio de la app de Estadística Aplicada."""

import streamlit as st

from src.datos import tarjeta_estudiante


# Se configura el título y el ancho antes de crear cualquier elemento de la página.
st.set_page_config(
    page_title="Estadística Aplicada con Python y R",
    layout="wide",
)

# Se muestra la identidad del estudiante en todas las páginas desde la barra lateral.
tarjeta_estudiante()

# Se presenta el propósito general de la aplicación.
st.title("App de Estadística Aplicada — Ingeniería Agrícola, Unisucre")
st.write(
    "Un espacio de trabajo para explorar datos agrícolas y aplicar métodos "
    "estadísticos con criterios claros, resultados reproducibles y atención a "
    "las unidades y supuestos."
)

# Se resumen las herramientas disponibles en cada sección multipágina.
st.header("Páginas de la aplicación")
st.markdown(
    """
- **Datos:** carga, revisión y preparación de conjuntos de datos.
- **EDA:** resúmenes, distribuciones, valores atípicos y relaciones entre variables.
- **Distribuciones:** ajuste de modelos de precipitación y secado, y comparación de su desempeño.
- **Regresión:** correlaciones, modelo lineal por MCO, inferencia y diagnóstico.
- **Series de tiempo:** componentes, métodos de referencia, validación y pronóstico.
"""
)

# Se ofrecen enlaces directos para abrir cada página además del menú multipágina de Streamlit.
st.subheader("Acceso a las páginas")
columnas_paginas = st.columns(5)
columnas_paginas[0].page_link("pages/1_Datos.py", label="Datos")
columnas_paginas[1].page_link("pages/2_EDA.py", label="EDA")
columnas_paginas[2].page_link("pages/3_Distribuciones.py", label="Distribuciones")
columnas_paginas[3].page_link("pages/4_Regresion.py", label="Regresión")
columnas_paginas[4].page_link("pages/5_Series_de_tiempo.py", label="Series de tiempo")

# Se explica la secuencia didáctica de resolver primero con lápiz y papel.
st.header("Lápiz y Algoritmo")
st.write(
    "Primero calcula a mano una muestra pequeña para entender cada paso; después "
    "verifica el procedimiento y amplía el análisis con la app, Python y R."
)

# Se recuerda que la herramienta no sustituye el razonamiento del estudiante.
st.warning(
    "Esta app apoya el aprendizaje, pero la interpretación de los resultados "
    "y la justificación de las conclusiones son responsabilidad del estudiante."
)