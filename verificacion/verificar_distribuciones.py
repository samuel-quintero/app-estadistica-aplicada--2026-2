"""Verifica a mano posiciones de graficación, Gumbel por momentos y riesgo."""

# Path ubica la muestra desde la raíz para ejecutar el script desde cualquier carpeta.
from pathlib import Path

# NumPy permite aplicar fórmulas explícitas a los datos ordenados.
import numpy as np

# Pandas se utiliza únicamente para leer la tabla de diez máximos anuales.
import pandas as pd


# Se encuentra la muestra pequeña de máximos anuales de precipitación.
RUTA_DATOS = Path(__file__).resolve().parents[1] / "datos" / "muestra_dist10.csv"

# Se leen y ordenan las diez observaciones desde la menor hasta la mayor.
datos = pd.read_csv(RUTA_DATOS).sort_values("pmax24h_mm").reset_index(drop=True)
x = datos["pmax24h_mm"].to_numpy(dtype=float)
n = len(x)

# Se asignan rangos i=1,...,n y posición de Weibull p=i/(n+1).
i = np.arange(1, n + 1, dtype=float)
probabilidad = i / (n + 1)

# El periodo de retorno empírico usa T=1/(1-p), donde p es no excedencia.
periodo_retorno = 1 / (1 - probabilidad)

# Se estiman los parámetros de Gumbel por momentos con desviación estándar muestral.
media = np.sum(x) / n
desviacion_muestral = np.sqrt(np.sum((x - media) ** 2) / (n - 1))
beta = desviacion_muestral * np.sqrt(6) / np.pi
mu = media - 0.5772 * beta

# Se calcula x_T = μ - β·ln(-ln(1-1/T)) para los periodos de diseño pedidos.
periodos_diseno = np.array([10.0, 50.0, 100.0])
probabilidades_diseno = 1 - 1 / periodos_diseno
cuantiles_gumbel = mu - beta * np.log(-np.log(probabilidades_diseno))

# Se calcula el riesgo de al menos una excedencia en N años: 1-(1-1/T)^N.
periodo_riesgo = 50.0
vida_util = 25
riesgo = 1 - (1 - 1 / periodo_riesgo) ** vida_util

# Se imprimen posiciones y periodos empíricos con cuatro decimales.
print(f"Archivo: {RUTA_DATOS.name}")
print("anio  pmax24h_mm  i  p_Weibull  T_empirico_anios")
for anio, valor, rango, p, retorno in zip(
    datos["anio"], x, i, probabilidad, periodo_retorno
):
    print(f"{anio:d}  {valor:.4f}  {rango:.0f}  {p:.4f}  {retorno:.4f}")

# Se imprimen los parámetros por momentos usados en los cuantiles de Gumbel.
print(f"Media muestral (mm): {media:.4f}")
print(f"Desviación estándar muestral (mm): {desviacion_muestral:.4f}")
print(f"Gumbel por momentos, beta (mm): {beta:.4f}")
print(f"Gumbel por momentos, mu (mm): {mu:.4f}")

# Se presenta cada cuantil de diseño y el riesgo durante la vida útil indicada.
print("Cuantiles Gumbel por momentos:")
for periodo, cuantil in zip(periodos_diseno, cuantiles_gumbel):
    print(f"  T={periodo:.0f} años: x_T={cuantil:.4f} mm")
print(f"Riesgo para T=50 y N=25 años: {riesgo:.4f} ({riesgo * 100:.4f} %)")
