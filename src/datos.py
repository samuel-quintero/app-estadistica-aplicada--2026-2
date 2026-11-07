"""Funciones compartidas para cargar datos y mostrar la ficha del estudiante."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


# Se calcula la carpeta de datos desde este archivo para no depender de dónde se ejecute.
CARPETA_DATOS = Path(__file__).resolve().parents[1] / "datos"


@st.cache_data
def cargar_csv(nombre: str) -> pd.DataFrame:
    """Lee un CSV de datos y convierte a fecha la columna `fecha` cuando existe."""
    # Se limita la lectura a nombres de archivo para mantenerla dentro de datos/.
    ruta_relativa = Path(nombre)
    if ruta_relativa.name != nombre or ruta_relativa.is_absolute():
        raise ValueError("Indica solo el nombre de un archivo CSV dentro de datos/.")

    # Se verifica que el archivo exista antes de inspeccionar sus columnas.
    ruta = CARPETA_DATOS / ruta_relativa
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta.name}")

    # Se detectan las columnas de fecha en la cabecera y se parsean al leer el archivo.
    columnas = pd.read_csv(ruta, nrows=0).columns
    columnas_fecha = [columna for columna in columnas if columna.lower() == "fecha"]
    return pd.read_csv(ruta, parse_dates=columnas_fecha)


@st.cache_data
def cargar_metadatos() -> dict:
    """Carga el nombre, código y datos de personalización del estudiante."""
    # Se abre el JSON explícitamente en UTF-8 para conservar tildes y eñes.
    ruta = CARPETA_DATOS / "metadatos.json"
    with ruta.open("r", encoding="utf-8") as archivo:
        return json.load(archivo)


def tipos_de_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Resume el tipo de dato y la clasificación estadística de cada variable."""
    # Se clasifica cada columna respetando primero los tipos categóricos y temporales.
    filas = []
    for nombre, serie in df.items():
        tipo = serie.dtype
        if pd.api.types.is_datetime64_any_dtype(tipo):
            clasificacion = "cualitativa ordinal"
        elif isinstance(tipo, pd.CategoricalDtype):
            clasificacion = (
                "cualitativa ordinal" if tipo.ordered else "cualitativa nominal"
            )
        elif pd.api.types.is_bool_dtype(tipo):
            clasificacion = "cualitativa nominal"
        elif pd.api.types.is_integer_dtype(tipo):
            clasificacion = "cuantitativa discreta"
        elif pd.api.types.is_numeric_dtype(tipo):
            clasificacion = "cuantitativa continua"
        else:
            clasificacion = "cualitativa nominal"

        # Se guarda el nombre legible del tipo junto con la clasificación asignada.
        filas.append(
            {
                "Variable": nombre,
                "Tipo de dato": str(tipo),
                "Clasificación estadística": clasificacion,
            }
        )

    return pd.DataFrame(
        filas,
        columns=["Variable", "Tipo de dato", "Clasificación estadística"],
    )


def valores_faltantes(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la cantidad y el porcentaje de faltantes para cada variable."""
    # Se cuentan los valores nulos y se calcula su proporción sobre todas las filas.
    cantidad = df.isna().sum()
    porcentaje = df.isna().mean().mul(100).fillna(0)

    # Se conserva una fila por columna, incluso cuando no hay valores faltantes.
    return pd.DataFrame(
        {
            "Variable": df.columns,
            "Cantidad faltante": cantidad.to_numpy(),
            "Porcentaje faltante": porcentaje.round(2).to_numpy(),
        }
    )


def tarjeta_estudiante() -> None:
    """Muestra en la barra lateral la ficha del estudiante y sus datos simulados."""
    # Se obtiene la información común desde el archivo de metadatos.
    metadatos = cargar_metadatos()

    # Se presenta la ficha en la barra lateral de la página activa.
    st.sidebar.subheader("Estudiante")
    st.sidebar.markdown(f"**Nombre:** {metadatos['nombre']}")
    st.sidebar.markdown(f"**Código:** {metadatos['codigo']}")
    st.sidebar.markdown(f"**Cultivo:** {metadatos['cultivo']}")
    st.sidebar.markdown(f"**Municipio:** {metadatos['municipio']}")
    st.sidebar.caption(f"Huella: {metadatos['huella']}")