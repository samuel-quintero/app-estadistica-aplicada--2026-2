# Resultados de verificación manual

Completa las columnas **A mano**, **Python**, **R** y **App** al ejecutar y contrastar cada muestra. En **¿Coincide?**, marca Sí/No y anota brevemente la diferencia de redondeo si aplica.

## EDA — muestra_eda10.csv

| Medida | A mano | Python | R | App | ¿Coincide? |
|---|---:|---:|---:|---:|---|
| n |  |  |  |  |  |
| Media |  |  |  |  |  |
| Mediana |  |  |  |  |  |
| Varianza muestral |  |  |  |  |  |
| Desviación estándar muestral |  |  |  |  |  |
| CV (%) |  |  |  |  |  |
| Q1 (tipo 7) |  |  |  |  |  |
| Q3 (tipo 7) |  |  |  |  |  |
| RIC |  |  |  |  |  |
| Valla inferior de Tukey |  |  |  |  |  |
| Valla superior de Tukey |  |  |  |  |  |
| Parcelas atípicas |  |  |  |  |  |
| Media de rendimiento con riego Sí |  |  |  |  |  |
| Media de rendimiento con riego No |  |  |  |  |  |

## Distribuciones — muestra_dist10.csv

| Medida | A mano | Python | R | App | ¿Coincide? |
|---|---:|---:|---:|---:|---|
| Posiciones de Weibull |  |  |  |  |  |
| Periodos de retorno empíricos |  |  |  |  |  |
| Media muestral (mm) |  |  |  |  |  |
| Desviación estándar muestral (mm) |  |  |  |  |  |
| Gumbel por momentos: μ (mm) |  |  |  |  |  |
| Gumbel por momentos: β (mm) |  |  |  |  |  |
| x_T, T=10 años (mm) |  |  |  |  |  |
| x_T, T=50 años (mm) |  |  |  |  |  |
| x_T, T=100 años (mm) |  |  |  |  |  |
| Riesgo, T=50 y N=25 |  |  |  |  |  |
| Gumbel MLE en R (si los paquetes están disponibles) |  |  |  |  |  |

## Regresión — muestra_reg8.csv

| Medida | A mano | Python | R | App | ¿Coincide? |
|---|---:|---:|---:|---:|---|
| Sxx |  |  |  |  |  |
| Syy |  |  |  |  |  |
| Sxy |  |  |  |  |  |
| r de Pearson |  |  |  |  |  |
| rho de Spearman |  |  |  |  |  |
| b0 |  |  |  |  |  |
| b1 |  |  |  |  |  |
| SCE |  |  |  |  |  |
| R² |  |  |  |  |  |
| RSE |  |  |  |  |  |
| SE(b1) |  |  |  |  |  |
| t observado |  |  |  |  |  |
| Decisión H0: β1=0 al 5 % |  |  |  |  |  |
| IC 95 % de b1 |  |  |  |  |  |

## Series de tiempo — muestra_series24.csv

| Medida | A mano | Python | R | App | ¿Coincide? |
|---|---:|---:|---:|---:|---|
| PM3 de julio de 2025 (mm) |  |  |  |  |  |
| PM3 de agosto de 2025 (mm) |  |  |  |  |  |
| PM3 de septiembre de 2025 (mm) |  |  |  |  |  |
| PM3 de octubre de 2025 (mm) |  |  |  |  |  |
| PM3 de noviembre de 2025 (mm) |  |  |  |  |  |
| PM3 de diciembre de 2025 (mm) |  |  |  |  |  |
| Suma de índices estacionales |  |  |  |  |  |
| MAE del método ingenuo |  |  |  |  |  |
| RMSE del método ingenuo |  |  |  |  |  |
| SMAPE del método ingenuo |  |  |  |  |  |
| MAE del ingenuo estacional |  |  |  |  |  |
| RMSE del ingenuo estacional |  |  |  |  |  |
| SMAPE del ingenuo estacional |  |  |  |  |  |
| MAE del promedio |  |  |  |  |  |
| RMSE del promedio |  |  |  |  |  |
| SMAPE del promedio |  |  |  |  |  |
| Método ganador por SMAPE |  |  |  |  |  |

## ¿Se cumplió mi predicción previa?

Escribe aquí la predicción que anotaste antes de ejecutar los cálculos y explica si se cumplió.


## Diferencias encontradas y su causa

Describe aquí cualquier diferencia entre A mano, Python, R y App. Considera redondeo, tipo de cuantil, divisor muestral/poblacional, orden de las observaciones y tratamiento de valores faltantes.
