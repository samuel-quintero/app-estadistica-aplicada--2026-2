# Verifica PM3, índices aditivos y métodos de referencia con R base.

# Se detecta si R se ejecuta desde la raíz del proyecto o desde verificacion/.
carpeta_datos <- if (dir.exists("datos")) "datos" else file.path("..", "datos")

# Se abre la muestra temporal desde la carpeta detectada y se interpreta fecha.
ruta_datos <- file.path(carpeta_datos, "muestra_series24.csv")
datos <- read.csv(ruta_datos, stringsAsFactors = FALSE)
datos$fecha <- as.Date(datos$fecha)
datos <- datos[order(datos$fecha), ]

# Se ordenan filas por periodo y se calcula PM3 trailing con stats::filter alineado a la derecha.
# filter() con sides=1 usa el mes actual y los dos meses anteriores, sin datos futuros.
datos$PM3 <- as.numeric(stats::filter(datos$precipitacion_mm,
                                      filter = rep(1 / 3, 3), sides = 1))

# Se calcula la media general de los 24 meses y las medias mensuales por mes calendario.
media_general <- sum(datos$precipitacion_mm) / nrow(datos)
medias_mes <- tapply(datos$precipitacion_mm, datos$mes, mean)
indices <- medias_mes[as.character(1:12)] - media_general

# Se separan las 18 observaciones de entrenamiento y las seis de validación.
entrenamiento <- datos[datos$conjunto == "entrenamiento", ]
validacion <- datos[datos$conjunto == "validacion", ]
y_train <- entrenamiento$precipitacion_mm
y_real <- validacion$precipitacion_mm
h <- length(y_real)

# El pronóstico ingenuo repite el último valor observado durante todo el horizonte.
y_ingenuo <- rep(tail(y_train, 1), h)

# El ingenuo estacional repite los doce valores más recientes en su mismo orden anual.
y_estacional <- rep(tail(y_train, 12), length.out = h)

# El método promedio mantiene la media aritmética del entrenamiento en cada mes futuro.
y_promedio <- rep(sum(y_train) / length(y_train), h)
pronosticos <- list(
  ingenuo = y_ingenuo,
  estacional_ingenuo = y_estacional,
  promedio = y_promedio
)

# Se calcula MAE, RMSE y SMAPE porcentual con las fórmulas del ejercicio.
calcular_metricas <- function(real, pronostico) {
  error <- real - pronostico
  cociente <- ifelse(abs(real) + abs(pronostico) == 0, 0,
                     abs(error) / (abs(real) + abs(pronostico)))
  c(MAE = mean(abs(error)), RMSE = sqrt(mean(error^2)), SMAPE = 200 * mean(cociente))
}
metricas <- lapply(pronosticos, function(pronostico) calcular_metricas(y_real, pronostico))

# Se muestran PM3 de 2025, índices y su suma, todos con cuatro decimales.
cat(sprintf("Archivo: %s\n", basename(ruta_datos)))
cat("PM3 de 2025 (mm):\n")
for (fila in which(datos$anio == 2025)) {
  cat(sprintf("%s  %.4f  %.4f\n", format(datos$fecha[fila], "%Y-%m-%d"),
              datos$precipitacion_mm[fila], datos$PM3[fila]))
}
cat(sprintf("Media general (mm): %.4f\n", media_general))
cat("Mes  índice_estacional_mm\n")
for (mes in 1:12) cat(sprintf("%02d   %.4f\n", mes, indices[as.character(mes)]))
cat(sprintf("Suma de índices: %.4f\n", sum(indices)))

# Se comparan pronósticos de julio-diciembre de 2025 y los datos observados.
cat("Pronósticos y reales (mm):\n")
for (fila in seq_len(h)) {
  cat(sprintf("%s  real=%.4f  ingenuo=%.4f  estacional=%.4f  promedio=%.4f\n",
              format(validacion$fecha[fila], "%Y-%m-%d"), y_real[fila],
              y_ingenuo[fila], y_estacional[fila], y_promedio[fila]))
}

# Se ordenan métodos por SMAPE ascendente para identificar el ganador de la validación.
resumen_metricas <- do.call(rbind, metricas)
resumen_metricas <- resumen_metricas[order(resumen_metricas[, "SMAPE"]), , drop = FALSE]
cat("Método              MAE       RMSE      SMAPE\n")
for (metodo in rownames(resumen_metricas)) {
  cat(sprintf("%-20s %.4f  %.4f  %.4f\n", metodo,
              resumen_metricas[metodo, "MAE"], resumen_metricas[metodo, "RMSE"],
              resumen_metricas[metodo, "SMAPE"]))
}
