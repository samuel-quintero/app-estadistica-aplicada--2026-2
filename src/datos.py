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