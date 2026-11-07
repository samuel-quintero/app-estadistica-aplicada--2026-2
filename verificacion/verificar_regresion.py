"""Verifica con fórmulas explícitas la regresión de muestra_reg8.csv."""

# Path localiza el CSV de ocho parcelas desde la raíz del proyecto.
from pathlib import Path

# NumPy realiza sumatorias, productos y cálculos vectorizados de MCO.
import numpy as np

# Pandas lee datos y calcula rangos promedio para Spearman.
import pandas as pd

# SciPy proporciona la distribución t para el crítico y el p-valor bilateral.
from scipy import stats


# Se carga la muestra pequeña y se seleccionan lámina de riego y rendimiento.
RUTA_DATOS = Path(__file__).resolve().parents[1] / "datos" / "muestra_reg8.csv"
datos = pd.read_csv(RUTA_DATOS).dropna(subset=["lamina_riego", "rendimiento"])
x = datos["lamina_riego"].to_numpy(dtype=float)
y = datos["rendimiento"].to_numpy(dtype=float)
n = len(x)

# Se calculan medias, diferencias centradas y sumas de cuadrados y productos cruzados.
x_media = np.sum(x) / n
y_media = np.sum(y) / n
dx = x - x_media
dy = y - y_media
sxx = np.sum(dx**2)
syy = np.sum(dy**2)
sxy = np.sum(dx * dy)

# Pearson se calcula como Sxy/raíz(Sxx·Syy); Spearman aplica la misma fórmula a rangos.
r_pearson = sxy / np.sqrt(sxx * syy)
rango_x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
rango_y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
dr_x = rango_x - rango_x.mean()
dr_y = rango_y - rango_y.mean()
rho_spearman = np.sum(dr_x * dr_y) / np.sqrt(np.sum(dr_x**2) * np.sum(dr_y**2))

# Se obtiene la pendiente y el intercepto usando las ecuaciones cerradas de MCO simple.
b1 = sxy / sxx
b0 = y_media - b1 * x_media
y_ajustado = b0 + b1 * x
residuo = y - y_ajustado

# SCE es error residual, SCT variación total y R²=1-SCE/SCT.
sce = np.sum(residuo**2)
scr = np.sum((y_ajustado - y_media) ** 2)
sct = np.sum((y - y_media) ** 2)
r_cuadrado = 1 - sce / sct

# RSE estima la desviación residual con n-2 grados de libertad.
grados_libertad = n - 2
rse = np.sqrt(sce / grados_libertad)

# El error estándar de la pendiente es RSE/raíz(Sxx) y t=b1/SE(b1).
se_b1 = rse / np.sqrt(sxx)
t_observado = b1 / se_b1

# Se calcula t crítico bilateral y p-valor usando la distribución t con n-2 grados de libertad.
t_critico = stats.t.ppf(0.975, df=grados_libertad)
p_valor = 2 * stats.t.sf(abs(t_observado), df=grados_libertad)

# El intervalo del 95 % se obtiene como b1 ± t crítico por el error estándar.
ic_inferior = b1 - t_critico * se_b1
ic_superior = b1 + t_critico * se_b1

# Se comunica la decisión de H0: β1=0 según el p-valor bilateral de cinco por ciento.
decision = "Se rechaza H0" if p_valor < 0.05 else "No se rechaza H0"

# Se imprimen todas las medidas solicitadas con cuatro decimales.
print(f"Archivo: {RUTA_DATOS.name}; n={n}")
print(f"Media x (mm): {x_media:.4f}")
print(f"Media y (t/ha): {y_media:.4f}")
print(f"Sxx: {sxx:.4f}")
print(f"Syy: {syy:.4f}")
print(f"Sxy: {sxy:.4f}")
print(f"r de Pearson: {r_pearson:.4f}")
print(f"rho de Spearman: {rho_spearman:.4f}")
print(f"b0: {b0:.4f}")
print(f"b1: {b1:.4f}")
print(f"SCE: {sce:.4f}")
print(f"SCR: {scr:.4f}")
print(f"SCT: {sct:.4f}")
print(f"R²: {r_cuadrado:.4f}")
print(f"RSE: {rse:.4f}")
print(f"SE(b1): {se_b1:.4f}")
print(f"t observado: {t_observado:.4f}")
print(f"t crítico bilateral (gl={grados_libertad}): {t_critico:.4f}")
print(f"p-valor bilateral: {p_valor:.4f}")
print(f"Decisión al 5 %: {decision}")
print(f"IC 95 % de b1: [{ic_inferior:.4f}, {ic_superior:.4f}]")
