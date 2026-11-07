"""Página para explorar los datos del proyecto o revisar un archivo propio."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.datos import (
    cargar_csv,
    tarjeta_estudiante,
    tipos_de_variables,
    valores_faltantes,
)


# Se documentan el contenido, las unidades y el uso de cada conjunto del proyecto.
DESCRIPCIONES = {
    "parcelas.csv": {
        "representa": "Parcelas simuladas con manejo, suelo y rendimiento del cultivo.",
        "unidades": (
            "dosis_N: kg/ha; lamina_riego: mm; materia_organica: %; "
            "ph: sin unidad; rendimiento: t/ha."
        ),
        "uso": "EDA y Regresión.",
    },
    "precip_max_anual.csv": {
        "representa": "Máximo anual de precipitación acumulada en 24 horas.",
        "unidades": "pmax24h_mm: mm; anio: año calendario.",
        "uso": "Distribuciones.",
    },
    "clima_mensual.csv": {
        "representa": "Precipitación y temperatura mensual del periodo simulado.",
        "unidades": "precipitacion_mm: mm; temperatura_c: °C; fecha: mes.",
        "uso": "EDA y Series de tiempo.",
    },
    "secado.csv": {
        "representa": "Razón de humedad observada durante el secado en capa delgada.",
        "unidades": "tiempo_min: minutos; MR: adimensional.",
        "uso": "Distribuciones, comparación de modelos de secado.",
    },
}


def leer_archivo_subido(archivo) -> pd.DataFrame:
    """Lee un CSV o Excel subido y convierte la columna fecha si está presente."""
    # Se inspecciona el formato para escoger el lector de pandas adecuado.
    extension = Path(archivo.name).suffix.lower()
    if extension == ".csv":
        archivo.seek(0)
        columnas = pd.read_csv(archivo, nrows=0).columns
        columnas_fecha = [columna for columna in columnas if columna.lower() == "fecha"]
        archivo.seek(0)
        datos = pd.read_csv(archivo, parse_dates=columnas_fecha)
    else:
        archivo.seek(0)
        datos = pd.read_excel(archivo)

    # Se interpretan también como fechas las columnas con ese nombre leídas como texto.
    for columna in datos.columns:
        if columna.lower() == "fecha" and not pd.api.types.is_datetime64_any_dtype(
            datos[columna]
        ):
            datos[columna] = pd.to_datetime(datos[columna], errors="coerce")

    return datos


# Se mantiene visible la información personalizada en la barra lateral.
tarjeta_estudiante()

# Se presenta la selección del conjunto del proyecto y la carga opcional del usuario.
st.title("Datos")
st.write("Explora un conjunto simulado o carga un archivo propio para revisar su estructura.")
archivo_seleccionado = st.selectbox(
    "Conjunto de datos del proyecto",
    options=list(DESCRIPCIONES),
    format_func=lambda nombre: nombre.removesuffix(".csv"),
)
archivo_subido = st.file_uploader(
    "Archivo propio opcional (CSV o Excel .xlsx)",
    type=["csv", "xlsx"],
)

# Se usan los datos del proyecto si no se ha seleccionado un archivo propio.
if archivo_subido is None:
    nombre_mostrado = archivo_seleccionado
    nombre_descarga = archivo_seleccionado
    try:
        datos = cargar_csv(archivo_seleccionado)
    except (OSError, ValueError, pd.errors.ParserError) as error:
        st.error(f"No se pudo leer el conjunto seleccionado: {error}")
        st.stop()
    descripcion = DESCRIPCIONES[archivo_seleccionado]
else:
    # Se carga el archivo subido y se muestran errores de formato de manera comprensible.
    nombre_mostrado = archivo_subido.name
    nombre_descarga = f"{Path(archivo_subido.name).stem}.csv"
    try:
        datos = leer_archivo_subido(archivo_subido)
    except Exception as error:
        st.error(f"No se pudo leer el archivo cargado: {error}")
        st.stop()
    descripcion = {
        "representa": "Archivo proporcionado por el usuario; su contenido depende de su origen.",
        "unidades": "No se infieren automáticamente; consultar la documentación del archivo.",
        "uso": "Datos; su análisis posterior depende de las variables disponibles.",
    }

# Se muestra el nombre de la fuente y el tamaño de la tabla en métricas separadas.
st.caption(f"Fuente activa: {nombre_mostrado}")
columna_filas, columna_variables = st.columns(2)
columna_filas.metric("Filas", f"{datos.shape[0]:,}")
columna_variables.metric("Columnas", f"{datos.shape[1]:,}")

# Se presenta la tabla completa y un botón para descargarla como CSV.
st.subheader("Tabla de datos")
st.dataframe(datos, width="stretch", hide_index=True)
st.download_button(
    "Descargar datos como CSV",
    data=datos.to_csv(index=False).encode("utf-8-sig"),
    file_name=nombre_descarga,
    mime="text/csv",
    width="stretch",
)

# Se clasifican las variables según su tipo de dato y escala estadística.
st.subheader("Tipos y clasificación estadística")
st.dataframe(tipos_de_variables(datos), width="stretch", hide_index=True)

# Se resumen faltantes para todas las columnas, incluso cuando su cantidad es cero.
st.subheader("Valores faltantes")
st.dataframe(valores_faltantes(datos), width="stretch", hide_index=True)

# Se documentan la interpretación, las unidades conocidas y las páginas relacionadas.
st.subheader("Descripción del archivo")
st.dataframe(
    pd.DataFrame(
        [
            {
                "Archivo": nombre_mostrado,
                "Qué representa": descripcion["representa"],
                "Unidades": descripcion["unidades"],
                "Página de uso": descripcion["uso"],
            }
        ]
    ),
    width="stretch",
    hide_index=True,
)