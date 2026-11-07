"""Validaciones pedagógicas reutilizables para datos y ajustes estadísticos."""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Callable


def validar_archivo_vacio(datos, nombre: str) -> bool:
    """Avisa si un archivo o estructura está vacío y devuelve si puede analizarse."""
    # Se reconoce como vacío un objeto ausente, sin filas o sin columnas.
    if datos is None or getattr(datos, "empty", True):
        st.error(
            f"El archivo o conjunto **{nombre}** está vacío. Revisa que tenga filas y "
            "columnas antes de continuar."
        )
        return False
    return True


def cargar_datos_seguros(lector: Callable[..., Any], nombre: str, contexto: str):
    """Lee datos con manejo pedagógico de errores y rechaza resultados vacíos."""
    # Se captura desde la operación de lectura para que archivos truncados no muestren trazas.
    try:
        datos = lector(nombre)
    except Exception as error:
        mostrar_error_ajuste(
            error,
            f"la lectura de {nombre}",
            f"Verifica que el archivo exista, no esté vacío y tenga un formato válido para {contexto}.",
        )
        return None

    # Se valida el resultado para diferenciar una lectura correcta de una tabla sin registros.
    if not validar_archivo_vacio(datos, nombre):
        return None
    return datos


def validar_columnas_seleccionadas(
    datos: pd.DataFrame,
    columnas: list[str],
    contexto: str = "análisis",
) -> bool:
    """Comprueba que estén presentes las columnas requeridas o seleccionadas."""
    # Se comparan las columnas pedidas con las disponibles en el DataFrame.
    faltantes = [nombre for nombre in columnas if nombre not in datos.columns]
    if faltantes:
        st.error(
            f"No se puede realizar el {contexto}: faltan las columnas "
            f"{', '.join(faltantes)}. Selecciona un archivo que las incluya o corrige sus nombres."
        )
        return False
    return True


def validar_respuesta_numerica(
    datos: pd.DataFrame,
    variable: str,
    contexto: str = "análisis",
) -> bool:
    """Comprueba que una variable de respuesta exista y contenga números válidos."""
    # Se explica primero el caso de columna inexistente para no propagar un KeyError.
    if variable not in datos.columns:
        st.error(
            f"La variable de respuesta **{variable}** no está en la tabla. Elige una "
            "columna disponible antes de continuar."
        )
        return False

    # Se exige un tipo numérico real y al menos una observación finita.
    serie = datos[variable]
    if not pd.api.types.is_numeric_dtype(serie.dtype):
        st.error(
            f"La variable **{variable}** no es numérica para el {contexto}. Corrige su tipo "
            "o selecciona una variable cuantitativa; no se convertirá silenciosamente."
        )
        return False
    valores = pd.to_numeric(serie, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(valores).any():
        st.error(
            f"La variable **{variable}** no contiene observaciones numéricas válidas. "
            "Revisa los datos o elige otra respuesta."
        )
        return False
    return True


def advertir_datos_faltantes(
    datos: pd.DataFrame,
    columnas: list[str] | None = None,
    contexto: str = "análisis",
) -> bool:
    """Advierte faltantes por columna y recuerda investigar antes de excluir o imputar."""
    # Se limita la revisión a columnas concretas si el análisis ya las seleccionó.
    datos_revisados = datos[columnas] if columnas else datos
    conteos = datos_revisados.isna().sum()
    conteos = conteos[conteos > 0]
    if conteos.empty:
        return True

    # Se presenta cantidad de faltantes y una recomendación antes de analizarlos.
    detalle = "; ".join(f"{nombre}: {int(cantidad)}" for nombre, cantidad in conteos.items())
    st.warning(
        f"El {contexto} contiene valores faltantes ({detalle}). Revisa su origen y patrón; "
        "no los borres ni imputes sin justificar la decisión. Algunos cálculos pueden usar "
        "solo los casos completos."
    )
    return False


def validar_minimo_observaciones(
    datos,
    minimo: int,
    contexto: str,
    columnas: list[str] | None = None,
) -> bool:
    """Advierte cuando hay menos observaciones válidas que las requeridas por el método."""
    # Se cuentan filas completas para las columnas del modelo o valores finitos de una serie.
    if isinstance(datos, pd.DataFrame):
        subconjunto = datos[columnas] if columnas else datos
        cantidad = int(subconjunto.dropna().shape[0])
    else:
        valores = pd.to_numeric(pd.Series(datos), errors="coerce").to_numpy(dtype=float)
        cantidad = int(np.isfinite(valores).sum())

    # Se explica el mínimo esperado y qué puede hacer el estudiante para resolverlo.
    if cantidad < minimo:
        st.warning(
            f"Hay {cantidad} observaciones válidas para {contexto}, pero se requieren al menos "
            f"{minimo}. Añade datos adecuados o usa un método que corresponda al tamaño disponible."
        )
        return False
    return True


def validar_serie_mensual(serie: pd.Series, minimo: int = 24, contexto: str = "serie") -> bool:
    """Valida índice DatetimeIndex, fechas mensuales completas y longitud suficiente."""
    # Se exige fecha interpretable antes de revisar los meses y sus posibles huecos.
    if not isinstance(serie.index, pd.DatetimeIndex):
        st.error(
            f"No se puede analizar la {contexto}: sus fechas no son un índice temporal. "
            "Selecciona o convierte una columna de fechas válida."
        )
        return False
    datos = serie.sort_index()

    # Se comprueba que las fechas correspondan a inicios de mes y no tengan meses omitidos.
    calendario_esperado = pd.date_range(datos.index.min(), datos.index.max(), freq="MS")
    if not datos.index.equals(calendario_esperado):
        st.warning(
            f"La {contexto} tiene meses faltantes, duplicados o fechas que no inician en el mes. "
            "Corrige el calendario o documenta cómo tratar esos periodos antes de descomponerla."
        )
        return False

    # Se delega el conteo numérico y el mínimo didáctico a la validación común.
    return validar_minimo_observaciones(datos, minimo, contexto)


def mostrar_error_ajuste(error: Exception, contexto: str, sugerencia: str | None = None) -> None:
    """Muestra una explicación pedagógica y mantiene el detalle técnico en un expander."""
    # Se formula primero el problema y una acción concreta en lenguaje para estudiantes.
    recomendacion = sugerencia or (
        "Revisa que haya suficientes observaciones válidas, que las variables tengan unidades "
        "coherentes y que el modelo sea apropiado para estos datos."
    )
    st.error(f"No se pudo completar {contexto}. {recomendacion}")

    # El detalle técnico queda disponible para depuración, acompañado de la explicación anterior.
    with st.expander("Detalle técnico del ajuste"):
        st.code(f"{type(error).__name__}: {error}")