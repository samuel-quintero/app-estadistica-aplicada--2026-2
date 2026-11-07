"""Cálculos estadísticos reutilizables para el análisis exploratorio de datos."""

import math

import pandas as pd


def resumen_numerico(serie: pd.Series) -> pd.Series:
    """Calcula medidas descriptivas; Q1 y Q3 usan cuantiles tipo 7 (interpolación lineal)."""
    # Se convierten los datos a números y se excluyen faltantes para cada estadístico.
    valores = pd.to_numeric(serie, errors="coerce").dropna().astype(float)
    cantidad = int(valores.size)

    # Para una serie vacía se conserva el esquema y se devuelve n como cero.
    if cantidad == 0:
        return pd.Series(
            {
                "n": 0,
                "Media": math.nan,
                "Mediana": math.nan,
                "Desviación estándar muestral": math.nan,
                "Varianza muestral": math.nan,
                "CV (%)": math.nan,
                "Mínimo": math.nan,
                "Q1 (tipo 7)": math.nan,
                "Q3 (tipo 7)": math.nan,
                "Máximo": math.nan,
                "RIC": math.nan,
                "Asimetría": math.nan,
            },
            name="Valor",
        )

    # Se calculan medidas centrales, de dispersión y cuartiles tipo 7 de pandas/R.
    media = float(valores.mean())
    desviacion = float(valores.std(ddof=1)) if cantidad > 1 else math.nan
    varianza = float(valores.var(ddof=1)) if cantidad > 1 else math.nan
    q1 = float(valores.quantile(0.25, interpolation="linear"))
    q3 = float(valores.quantile(0.75, interpolation="linear"))
    coeficiente_variacion = (
        desviacion / abs(media) * 100
        if cantidad > 1 and media != 0
        else math.nan
    )

    # Se devuelve una serie indexada para presentar fácilmente cada estadístico.
    return pd.Series(
        {
            "n": cantidad,
            "Media": media,
            "Mediana": float(valores.median()),
            "Desviación estándar muestral": desviacion,
            "Varianza muestral": varianza,
            "CV (%)": coeficiente_variacion,
            "Mínimo": float(valores.min()),
            "Q1 (tipo 7)": q1,
            "Q3 (tipo 7)": q3,
            "Máximo": float(valores.max()),
            "RIC": q3 - q1,
            "Asimetría": float(valores.skew()) if cantidad >= 3 else math.nan,
        },
        name="Valor",
    )


def vallas_tukey(serie: pd.Series) -> tuple[float, float, pd.Series]:
    """Devuelve vallas de Tukey y valores atípicos, conservando sus índices originales."""
    # Se descartan faltantes y se calculan cuartiles tipo 7 para construir las vallas.
    valores = pd.to_numeric(serie, errors="coerce").dropna().astype(float)
    if valores.empty:
        return math.nan, math.nan, valores

    # Cada valla queda a 1.5 rangos intercuartílicos del cuartil correspondiente.
    q1 = float(valores.quantile(0.25, interpolation="linear"))
    q3 = float(valores.quantile(0.75, interpolation="linear"))
    rango_intercuartilico = q3 - q1
    limite_inferior = q1 - 1.5 * rango_intercuartilico
    limite_superior = q3 + 1.5 * rango_intercuartilico

    # Se conservan las etiquetas de índice para asociar cada valor a su parcela.
    atipicos = valores.loc[
        (valores < limite_inferior) | (valores > limite_superior)
    ]
    return limite_inferior, limite_superior, atipicos


def tabla_frecuencias(serie: pd.Series) -> pd.DataFrame:
    """Calcula frecuencias absolutas y relativas, incluyendo faltantes como categoría."""
    # Se cuentan categorías y se calculan sus proporciones sobre todas las observaciones.
    frecuencias = serie.value_counts(dropna=False, sort=True)
    relativas = serie.value_counts(dropna=False, sort=True, normalize=True)

    # Se rotulan los valores faltantes para que también sean visibles en la tabla.
    categorias = [
        "Faltante" if pd.isna(categoria) else str(categoria)
        for categoria in frecuencias.index
    ]
    return pd.DataFrame(
        {
            "Categoría": categorias,
            "Frecuencia absoluta": frecuencias.to_numpy(dtype=int),
            "Frecuencia relativa": relativas.to_numpy(dtype=float),
        }
    )