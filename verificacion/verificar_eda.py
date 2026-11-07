"""Verifica a mano medidas descriptivas de muestra_eda10.csv."""

# Path permite ubicar los datos desde la raíz del proyecto sin depender del directorio actual.
from pathlib import Path

# NumPy calcula cuartiles tipo 7 mediante interpolación lineal y fórmulas explícitas.
import numpy as np

# Pandas se usa solo para leer el CSV y agrupar por condición de riego.
import pandas as pd


# Se encuentra la carpeta datos como hermana de este directorio de verificación.
RUTA_DATOS = Path(__file__).resolve().parents[1] / "datos" / "muestra_eda10.csv"

# Se carga la muestra pequeña para que los cálculos puedan revisarse a mano.
datos = pd.read_csv(RUTA_DATOS)

# Se toma el rendimiento como variable de análisis y se elimina cualquier faltante.
x = datos["rendimiento"].dropna().to_numpy(dtype=float)

# Se calcula n y la media aritmética usando sumatoria dividida entre el tamaño muestral.
n = len(x)
media = np.sum(x) / n

# Se ordenan los datos para obtener la mediana y los cuantiles interpolados tipo 7.
x_ordenado = np.sort(x)
mediana = float(np.median(x_ordenado))
q1, q3 = np.quantile(x_ordenado, [0.25, 0.75], method="linear")

# Se calcula la varianza muestral dividiendo la suma de desviaciones cuadradas por n-1.
varianza = np.sum((x - media) ** 2) / (n - 1)

# La desviación estándar es la raíz de la varianza y el CV expresa su proporción porcentual.
desviacion = np.sqrt(varianza)
cv = desviacion / abs(media) * 100

# El rango intercuartílico y las vallas se obtienen directamente de Q1 y Q3.
ric = q3 - q1
valla_inferior = q1 - 1.5 * ric
valla_superior = q3 + 1.5 * ric

# Se seleccionan como atípicos los rendimientos que quedan fuera de las vallas de Tukey.
mascara_atipicos = (datos["rendimiento"] < valla_inferior) | (
    datos["rendimiento"] > valla_superior
)
atipicos = datos.loc[mascara_atipicos, ["parcela", "rendimiento"]]

# Se imprimen las medidas con cuatro decimales para cotejarlas con la app.
print(f"Archivo: {RUTA_DATOS.name}")
print(f"n: {n}")
print(f"Media: {media:.4f}")
print(f"Mediana: {mediana:.4f}")
print(f"Varianza muestral: {varianza:.4f}")
print(f"Desviación estándar muestral: {desviacion:.4f}")
print(f"CV (%): {cv:.4f}")
print(f"Q1 (tipo 7): {q1:.4f}")
print(f"Q3 (tipo 7): {q3:.4f}")
print(f"RIC: {ric:.4f}")
print(f"Valla inferior de Tukey: {valla_inferior:.4f}")
print(f"Valla superior de Tukey: {valla_superior:.4f}")
print("Atípicos:")
if atipicos.empty:
    print("  Ninguno")
else:
    for fila in atipicos.itertuples(index=False):
        print(f"  {fila.parcela}: {fila.rendimiento:.4f}")

# Se comparan las medias de rendimiento para cada condición de riego observada.
print("Media de rendimiento por riego:")
for condicion, grupo in datos.groupby("riego", sort=True):
    promedio_grupo = grupo["rendimiento"].mean()
    print(f"  {condicion}: {promedio_grupo:.4f}")
