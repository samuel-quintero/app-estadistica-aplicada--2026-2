"""Verifica a mano promedios, índices y referencias de muestra_series24.csv."""

# Path localiza la muestra pequeña desde la raíz del proyecto.
from pathlib import Path

# NumPy permite expresar los promedios y errores con sus fórmulas directas.
import numpy as np

# Pandas permite leer fechas, ordenar por tiempo y agrupar por mes calendario.
import pandas as pd


# Se abre la muestra y se ordena cronológicamente para respetar el orden de la serie.
RUTA_DATOS = Path(__file__).resolve().parents[1] / "datos" / "muestra_series24.csv"
datos = pd.read_csv(RUTA_DATOS, parse_dates=["fecha"]).sort_values("fecha")

# Se asigna PM3 con una ventana trailing que termina en el mes de cada fila.
datos["PM3"] = datos["precipitacion_mm"].rolling(window=3, min_periods=3).mean()

# Se calcula la media general de los 24 meses y la media de cada mes del año.
serie = datos["precipitacion_mm"].to_numpy(dtype=float)
media_general = np.sum(serie) / len(serie)
medias_mes = datos.groupby("mes")["precipitacion_mm"].mean().reindex(range(1, 13))
indices = medias_mes.to_numpy(dtype=float) - media_general

# Se separan las observaciones usando las etiquetas de entrenamiento y validación del archivo.
entrenamiento = datos.loc[datos["conjunto"] == "entrenamiento"].copy()
validacion = datos.loc[datos["conjunto"] == "validacion"].copy()
y_train = entrenamiento["precipitacion_mm"].to_numpy(dtype=float)
y_real = validacion["precipitacion_mm"].to_numpy(dtype=float)
h = len(y_real)

# El método ingenuo mantiene en cada horizonte el último valor de entrenamiento.
y_ingenuo = np.repeat(y_train[-1], h)

# El método estacional repite en orden los doce últimos meses conocidos.
y_estacional = np.resize(y_train[-12:], h)

# El método promedio repite la media aritmética de los 18 meses de entrenamiento.
y_promedio = np.repeat(np.sum(y_train) / len(y_train), h)
pronosticos = {
    "ingenuo": y_ingenuo,
    "estacional_ingenuo": y_estacional,
    "promedio": y_promedio,
}

# Se calcula MAE, RMSE y SMAPE=200·media(|e|/(|y|+|ŷ|)) para cada método.
metricas_metodos = {}
for nombre, y_pronosticado in pronosticos.items():
    error = y_real - y_pronosticado
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))
    denominador = np.abs(y_real) + np.abs(y_pronosticado)
    cociente = np.divide(
        np.abs(error),
        denominador,
        out=np.zeros_like(denominador),
        where=denominador != 0,
    )
    smape = 200 * np.mean(cociente)
    metricas_metodos[nombre] = {"MAE": mae, "RMSE": rmse, "SMAPE": smape}

# Se muestran los valores PM3 de 2025, que corresponden a los meses de validación.
print(f"Archivo: {RUTA_DATOS.name}")
print("PM3 de 2025 (mm):")
print("fecha       precipitación  PM3")
for fila in datos.loc[datos["anio"] == 2025].itertuples(index=False):
    print(f"{fila.fecha:%Y-%m-%d}  {fila.precipitacion_mm:.4f}  {fila.PM3:.4f}")

# Se imprimen los doce índices aditivos y su suma, que es cero salvo redondeo.
print(f"Media general 24 meses (mm): {media_general:.4f}")
print("Mes  índice_estacional_mm")
for mes, valor in enumerate(indices, start=1):
    print(f"{mes:02d}   {valor:.4f}")
print(f"Suma de índices estacionales: {np.sum(indices):.4f}")

# Se comparan los tres pronósticos de julio a diciembre de 2025 con sus observaciones reales.
print("fecha       real       ingenuo    estacional  promedio")
for fila, ingenuo, estacional, promedio in zip(
    validacion.itertuples(index=False), y_ingenuo, y_estacional, y_promedio
):
    print(
        f"{fila.fecha:%Y-%m-%d}  {fila.precipitacion_mm:.4f}  "
        f"{ingenuo:.4f}  {estacional:.4f}  {promedio:.4f}"
    )

# Se ordenan los métodos por SMAPE para identificar el mejor resultado de la muestra.
print("Método              MAE       RMSE      SMAPE")
for nombre, valores in sorted(metricas_metodos.items(), key=lambda item: item[1]["SMAPE"]):
    print(
        f"{nombre:<20} {valores['MAE']:.4f}  {valores['RMSE']:.4f}  "
        f"{valores['SMAPE']:.4f}"
    )
