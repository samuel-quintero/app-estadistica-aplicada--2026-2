# Verifica medidas descriptivas de muestra_eda10.csv con fórmulas de R base.

# Se detecta si R se ejecuta desde la raíz del proyecto o desde verificacion/.
carpeta_datos <- if (dir.exists("datos")) "datos" else file.path("..", "datos")

# Se arma la ruta de la muestra usando la carpeta detectada.
ruta_datos <- file.path(carpeta_datos, "muestra_eda10.csv")

# Se lee el CSV y se toma rendimiento, eliminando faltantes si los hubiera.
datos <- read.csv(ruta_datos, stringsAsFactors = FALSE)
x <- datos$rendimiento
x <- x[is.finite(x)]

# Se calcula n, media, mediana, varianza muestral y desviación estándar muestral.
n <- length(x)
media <- sum(x) / n
mediana <- median(x)
varianza <- sum((x - media)^2) / (n - 1)
desviacion <- sqrt(varianza)
cv <- desviacion / abs(media) * 100

# quantile(type=7) usa la interpolación predeterminada de R y pandas.
cuartiles <- quantile(x, probs = c(0.25, 0.75), type = 7, names = FALSE)
q1 <- cuartiles[1]
q3 <- cuartiles[2]
ric <- q3 - q1

# Se calculan las vallas de Tukey y se identifican las observaciones exteriores.
valla_inferior <- q1 - 1.5 * ric
valla_superior <- q3 + 1.5 * ric
atipicos <- datos[datos$rendimiento < valla_inferior | datos$rendimiento > valla_superior,
                  c("parcela", "rendimiento")]

# Se imprime cada medida con cuatro decimales para cotejarla con los otros programas.
cat(sprintf("Archivo: %s\n", basename(ruta_datos)))
cat(sprintf("n: %d\n", n))
cat(sprintf("Media: %.4f\n", media))
cat(sprintf("Mediana: %.4f\n", mediana))
cat(sprintf("Varianza muestral: %.4f\n", varianza))
cat(sprintf("Desviación estándar muestral: %.4f\n", desviacion))
cat(sprintf("CV (%%): %.4f\n", cv))
cat(sprintf("Q1 (tipo 7): %.4f\n", q1))
cat(sprintf("Q3 (tipo 7): %.4f\n", q3))
cat(sprintf("RIC: %.4f\n", ric))
cat(sprintf("Valla inferior de Tukey: %.4f\n", valla_inferior))
cat(sprintf("Valla superior de Tukey: %.4f\n", valla_superior))
cat("Atípicos:\n")
if (nrow(atipicos) == 0) {
  cat("  Ninguno\n")
} else {
  for (fila in seq_len(nrow(atipicos))) {
    cat(sprintf("  %s: %.4f\n", atipicos$parcela[fila], atipicos$rendimiento[fila]))
  }
}

# Se calculan medias de rendimiento por riego con aggregate de R base.
medias_riego <- aggregate(rendimiento ~ riego, data = datos, FUN = mean)
cat("Media de rendimiento por riego:\n")
for (fila in seq_len(nrow(medias_riego))) {
  cat(sprintf("  %s: %.4f\n", medias_riego$riego[fila], medias_riego$rendimiento[fila]))
}
