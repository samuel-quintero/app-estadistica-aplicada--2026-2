"""Funciones para descomponer y pronosticar series mensuales agrícolas."""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


# Se define una sola ruta para que el módulo funcione desde cualquier directorio actual.
CARPETA_DATOS = Path(__file__).resolve().parents[1] / "datos"


def cargar_serie(variable: str) -> pd.Series:
    """Carga una variable de clima_mensual.csv con índice mensual de frecuencia MS."""
    # Se abre el CSV con la fecha interpretada como datetime para construir su calendario.
    ruta = CARPETA_DATOS / "clima_mensual.csv"
    datos = pd.read_csv(ruta, parse_dates=["fecha"])
    if variable not in datos.columns:
        variables_disponibles = [nombre for nombre in datos.columns if nombre != "fecha"]
        raise ValueError(
            f"Variable {variable!r} no encontrada. Disponibles: {variables_disponibles}."
        )

    # Se ordena por fecha, establece el índice y fija los meses que pudieran faltar.
    serie = datos.set_index("fecha")[variable].sort_index().asfreq("MS")
    serie.name = variable
    return serie


def promedio_movil(serie: pd.Series, orden: int) -> pd.Series:
    """Calcula promedio móvil trailing; cada ventana termina en el mes de su etiqueta."""
    # Se valida el ancho de la ventana antes de aplicar rolling.
    if not isinstance(orden, (int, np.integer)) or orden < 1:
        raise ValueError("orden debe ser un entero positivo.")

    # rolling usa por defecto una ventana que termina en la observación actual.
    return serie.rolling(window=int(orden), min_periods=int(orden)).mean()


def indices_estacionales(serie: pd.Series) -> pd.Series:
    """Calcula índices aditivos mensuales: media del mes menos la media general."""
    # Se exige un índice de fechas porque el mes del año determina cada estación.
    if not isinstance(serie.index, pd.DatetimeIndex):
        raise TypeError("La serie debe tener un DatetimeIndex para obtener meses estacionales.")
    datos = serie.dropna()
    if datos.empty:
        raise ValueError("La serie no contiene observaciones válidas.")

    # Se calcula una media general y una media por mes calendario (1=enero, ..., 12=diciembre).
    promedio_general = float(datos.mean())
    promedios_mensuales = datos.groupby(datos.index.month).mean()
    indices = promedios_mensuales.reindex(range(1, 13)) - promedio_general
    indices.index.name = "mes"
    indices.name = "índice estacional aditivo"
    return indices


def descomponer(serie: pd.Series):
    """Descompone una serie mensual en tendencia, estacionalidad y residuo aditivos."""
    # Se eliminan faltantes y se verifica que haya al menos dos ciclos anuales.
    datos = serie.dropna()
    if len(datos) < 24:
        raise ValueError("La descomposición mensual requiere al menos 24 observaciones.")

    # El modelo aditivo representa observado = tendencia + estación + residuo.
    return seasonal_decompose(datos, model="additive", period=12)


def metodos_referencia(entrenamiento: pd.Series, h: int) -> pd.DataFrame:
    """Pronostica h meses con ingenuo, ingenuo estacional y media del entrenamiento."""
    # Se valida el horizonte y el historial antes de formar pronósticos.
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError("h debe ser un entero positivo.")
    datos = entrenamiento.dropna().astype(float).sort_index()
    if datos.empty:
        raise ValueError("El entrenamiento debe contener al menos un valor válido.")

    # El índice de salida continúa el calendario mensual del último dato de entrenamiento.
    if not isinstance(datos.index, pd.DatetimeIndex):
        raise TypeError("El entrenamiento debe tener un DatetimeIndex mensual.")
    calendario_esperado = pd.date_range(
        start=datos.index[0],
        periods=len(datos),
        freq="MS",
    )
    if not datos.index.equals(calendario_esperado):
        raise ValueError(
            "El entrenamiento debe ser mensual, sin huecos y al inicio de cada mes (MS)."
        )
    fechas_futuras = pd.date_range(
        start=datos.index[-1] + pd.offsets.MonthBegin(1),
        periods=int(h),
        freq="MS",
    )

    # El método ingenuo mantiene el último dato; el método promedio repite la media histórica.
    pronostico_ingenuo = np.repeat(datos.iloc[-1], int(h))
    pronostico_promedio = np.repeat(datos.mean(), int(h))

    # El ingenuo estacional repite el patrón de los últimos 12 meses en horizontes sucesivos.
    if len(datos) < 12:
        raise ValueError("Se necesitan al menos 12 meses para el método ingenuo estacional.")
    patron_estacional = datos.iloc[-12:].to_numpy()
    pronostico_estacional = np.resize(patron_estacional, int(h))

    # Se devuelve una columna por método y una fila por mes pronosticado.
    return pd.DataFrame(
        {
            "ingenuo": pronostico_ingenuo,
            "estacional_ingenuo": pronostico_estacional,
            "promedio": pronostico_promedio,
        },
        index=fechas_futuras,
    )


def metricas(real, pronostico) -> dict:
    """Calcula MAE, RMSE y SMAPE porcentual entre valores reales y pronosticados."""
    # Se convierten entradas a vectores y se verifica que cada real tenga pronóstico.
    valores_reales = np.asarray(real, dtype=float).reshape(-1)
    valores_pronosticados = np.asarray(pronostico, dtype=float).reshape(-1)
    if valores_reales.size != valores_pronosticados.size:
        raise ValueError("real y pronostico deben tener la misma longitud.")

    # Se excluyen pares no finitos para que faltantes no se conviertan en métricas engañosas.
    validos = np.isfinite(valores_reales) & np.isfinite(valores_pronosticados)
    valores_reales = valores_reales[validos]
    valores_pronosticados = valores_pronosticados[validos]
    if valores_reales.size == 0:
        raise ValueError("Se requiere al menos un par finito de valores para calcular métricas.")

    # MAE promedia errores absolutos; RMSE penaliza más los errores grandes.
    errores = valores_reales - valores_pronosticados
    mae = float(np.mean(np.abs(errores)))
    rmse = float(np.sqrt(np.mean(errores**2)))

    # SMAPE = 200·media(|e|/(|real|+|pronóstico|)); 0/0 aporta cero al promedio.
    denominador = np.abs(valores_reales) + np.abs(valores_pronosticados)
    contribuciones = np.divide(
        np.abs(errores),
        denominador,
        out=np.zeros_like(denominador, dtype=float),
        where=denominador != 0,
    )
    smape = float(200 * np.mean(contribuciones))
    return {"MAE": mae, "RMSE": rmse, "SMAPE": smape}


def concurso_referencia(muestra24: pd.DataFrame) -> pd.DataFrame:
    """Compara los tres métodos de referencia en los 18/6 meses de muestra_series24."""
    # Se verifica el esquema requerido y se ordenan filas según la fecha real.
    columnas_requeridas = {"fecha", "precipitacion_mm", "conjunto"}
    faltantes = columnas_requeridas.difference(muestra24.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}.")
    datos = muestra24.copy()
    datos["fecha"] = pd.to_datetime(datos["fecha"])
    datos = datos.sort_values("fecha")

    # Se separan por etiqueta los meses de entrenamiento y validación, sin mezclar el tiempo.
    entrenamiento = datos.loc[datos["conjunto"] == "entrenamiento"].set_index("fecha")[
        "precipitacion_mm"
    ]
    validacion = datos.loc[datos["conjunto"] == "validacion"].set_index("fecha")[
        "precipitacion_mm"
    ]
    if entrenamiento.empty or validacion.empty:
        raise ValueError("La muestra debe incluir meses de entrenamiento y validación.")

    # Se generan pronósticos solo desde entrenamiento para el horizonte de validación.
    pronosticos = metodos_referencia(entrenamiento, h=len(validacion))
    reales = validacion.reindex(pronosticos.index)
    if reales.isna().any():
        raise ValueError("Las fechas de validación no continúan mensualmente al entrenamiento.")

    # Se calcula cada métrica por método y se ordena de menor a mayor SMAPE.
    filas = []
    for metodo in pronosticos.columns:
        filas.append({"Método": metodo, **metricas(reales, pronosticos[metodo])})
    return pd.DataFrame(filas).sort_values("SMAPE").reset_index(drop=True)


if __name__ == "__main__":
    # Se carga la muestra didáctica de 24 meses, con fechas interpretadas como datetime.
    ruta_muestra = CARPETA_DATOS / "muestra_series24.csv"
    muestra = pd.read_csv(ruta_muestra, parse_dates=["fecha"])

    # Se imprime el concurso de referencia ordenado por SMAPE en el bloque principal.
    resultados = concurso_referencia(muestra)
    print(resultados.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
