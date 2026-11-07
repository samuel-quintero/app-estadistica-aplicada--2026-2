"""Descargas reutilizables de tablas, figuras y reportes Markdown desde Streamlit."""

from io import BytesIO
import re

import pandas as pd
import streamlit as st


def crear_excel(tablas: dict[str, pd.DataFrame]) -> bytes:
    """Escribe varias tablas en hojas de Excel dentro de un buffer en memoria."""
    # Se rechaza un libro sin tablas para que el estudiante reciba un mensaje entendible.
    if not tablas:
        raise ValueError("Agrega al menos una tabla antes de crear el archivo Excel.")

    # Se escribe cada DataFrame en una hoja y se mantiene el archivo completamente en memoria.
    buffer = BytesIO()
    nombres_usados = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as escritor:
        for nombre, tabla in tablas.items():
            # Excel limita nombres de hoja a 31 caracteres y no admite ciertos signos.
            base = re.sub(r"[\\/*?:\[\]]", "_", str(nombre))[:31] or "Datos"
            nombre_hoja = base
            sufijo = 1
            while nombre_hoja in nombres_usados:
                marca = f"_{sufijo}"
                nombre_hoja = f"{base[:31 - len(marca)]}{marca}"
                sufijo += 1
            nombres_usados.add(nombre_hoja)
            tabla.to_excel(escritor, sheet_name=nombre_hoja, index=False)

    # Se devuelve el contenido binario para conectarlo con st.download_button.
    buffer.seek(0)
    return buffer.getvalue()


def descargar_excel(
    tablas: dict[str, pd.DataFrame],
    nombre_archivo: str,
    etiqueta: str = "Descargar tablas en Excel",
    key: str | None = None,
) -> None:
    """Muestra un botón de descarga para un libro Excel con una hoja por tabla."""
    # Se genera el workbook y se asigna la extensión correcta al archivo descargado.
    nombre = nombre_archivo if nombre_archivo.lower().endswith(".xlsx") else f"{nombre_archivo}.xlsx"
    st.download_button(
        etiqueta,
        data=crear_excel(tablas),
        file_name=nombre,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


def descargar_figuras(figura, nombre_base: str, key: str | None = None) -> None:
    """Muestra botones para descargar una figura de Matplotlib en PNG y PDF."""
    # Se renderizan ambos formatos en memoria con resolución adecuada para informes.
    buffer_png = BytesIO()
    figura.savefig(buffer_png, format="png", dpi=200, bbox_inches="tight")
    buffer_png.seek(0)
    buffer_pdf = BytesIO()
    figura.savefig(buffer_pdf, format="pdf", bbox_inches="tight")
    buffer_pdf.seek(0)

    # Se presentan los formatos como dos acciones de descarga independientes.
    columna_png, columna_pdf = st.columns(2)
    with columna_png:
        st.download_button(
            "Descargar figura PNG",
            data=buffer_png.getvalue(),
            file_name=f"{nombre_base}.png",
            mime="image/png",
            key=f"{key}_png" if key else None,
        )
    with columna_pdf:
        st.download_button(
            "Descargar figura PDF",
            data=buffer_pdf.getvalue(),
            file_name=f"{nombre_base}.pdf",
            mime="application/pdf",
            key=f"{key}_pdf" if key else None,
        )


def descargar_reporte_markdown(
    titulo: str,
    resultados: str,
    nombre_archivo: str,
    key: str | None = None,
) -> None:
    """Muestra un botón para descargar un resumen Markdown con resultados principales."""
    # Se agrega un título al reporte para que el archivo tenga contexto al abrirse.
    contenido = f"# {titulo}\n\n{resultados.strip()}\n"
    nombre = nombre_archivo if nombre_archivo.lower().endswith(".md") else f"{nombre_archivo}.md"

    # UTF-8 conserva unidades, acentos y símbolos estadísticos del reporte.
    st.download_button(
        "Descargar reporte Markdown",
        data=contenido.encode("utf-8"),
        file_name=nombre,
        mime="text/markdown",
        key=key,
    )