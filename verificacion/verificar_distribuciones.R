# Verifica posiciones, Gumbel por momentos, cuantiles y riesgo con R base.

# Se detecta si R se ejecuta desde la raíz del proyecto o desde verificacion/.
carpeta_datos <- if (dir.exists("datos")) "datos" else file.path("..", "datos")

# Se arma la ruta de la muestra usando la carpeta detectada.
ruta_datos <- file.path(carpeta_datos, "muestra_dist10.csv")

# Se leen y ordenan los diez máximos anuales de menor a mayor.
datos <- read.csv(ruta_datos)
datos <- datos[order(datos$pmax24h_mm), ]
x <- datos$pmax24h_mm
n <- length(x)

# Se asigna el rango i y la posición Weibull p=i/(n+1).
i <- seq_len(n)
probabilidad <- i / (n + 1)

# El retorno empírico se calcula con T=1/(1-p), donde p es probabilidad de no excedencia.
periodo_retorno <- 1 / (1 - probabilidad)

# Se calcula la media y la desviación estándar muestral con denominador n-1.
media <- sum(x) / n
desviacion_muestral <- sqrt(sum((x - media)^2) / (n - 1))

# Los parámetros Gumbel por momentos son beta=s·raíz(6)/pi y mu=media-0.5772·beta.
beta <- desviacion_muestral * sqrt(6) / pi
mu <- media - 0.5772 * beta

# Se evalúa x_T=mu-beta·ln(-ln(1-1/T)) para T de 10, 50 y 100 años.
periodos_diseno <- c(10, 50, 100)
probabilidades_diseno <- 1 - 1 / periodos_diseno
cuantiles_gumbel <- mu - beta * log(-log(probabilidades_diseno))

# El riesgo de al menos una excedencia en N años es 1-(1-1/T)^N.
periodo_riesgo <- 50
vida_util <- 25
riesgo <- 1 - (1 - 1 / periodo_riesgo)^vida_util

# Se imprimen posiciones Weibull y periodos empíricos con cuatro decimales.
cat(sprintf("Archivo: %s\n", basename(ruta_datos)))
cat("anio  pmax24h_mm  i  p_Weibull  T_empirico_anios\n")
for (fila in seq_len(n)) {
  cat(sprintf("%d  %.4f  %d  %.4f  %.4f\n",
              datos$anio[fila], x[fila], i[fila], probabilidad[fila], periodo_retorno[fila]))
}

# Se muestran parámetros por momentos para comparar cálculos manuales.
cat(sprintf("Media muestral (mm): %.4f\n", media))
cat(sprintf("Desviación estándar muestral (mm): %.4f\n", desviacion_muestral))
cat(sprintf("Gumbel por momentos, beta (mm): %.4f\n", beta))
cat(sprintf("Gumbel por momentos, mu (mm): %.4f\n", mu))
cat("Cuantiles Gumbel por momentos:\n")
for (fila in seq_along(periodos_diseno)) {
  cat(sprintf("  T=%d años: x_T=%.4f mm\n", periodos_diseno[fila], cuantiles_gumbel[fila]))
}
cat(sprintf("Riesgo para T=50 y N=25 años: %.4f (%.4f %%)\n", riesgo, riesgo * 100))

# Posit Cloud: en la consola R puede instalarse el soporte opcional de máxima verosimilitud.
# Ejecuta install.packages(c("fitdistrplus", "evd")) una sola vez en el proyecto de Posit Cloud.
# fitdistrplus implementa el ajuste MLE y evd aporta la densidad Gumbel dgumbel.
if (requireNamespace("fitdistrplus", quietly = TRUE) && requireNamespace("evd", quietly = TRUE)) {
  # Se adjunta evd para que fitdist encuentre dgumbel por el nombre de la distribución.
  suppressPackageStartupMessages(library(evd))

  # Se usan los momentos como valores iniciales y se maximiza la verosimilitud Gumbel.
  ajuste_mle <- tryCatch(
    fitdistrplus::fitdist(
      data = x,
      distr = "gumbel",
      start = list(loc = mu, scale = beta),
      method = "mle"
    ),
    error = function(error) error
  )

  # Se imprimen los parámetros MLE o una explicación si la versión instalada no los admite.
  if (inherits(ajuste_mle, "error")) {
    cat(sprintf("No se pudo ajustar MLE con las versiones instaladas: %s\n", ajuste_mle$message))
  } else {
    estimaciones <- ajuste_mle$estimate
    cat(sprintf("Gumbel MLE, loc (mm): %.4f\n", estimaciones["loc"]))
    cat(sprintf("Gumbel MLE, scale (mm): %.4f\n", estimaciones["scale"]))
  }
} else {
  # Se explica cómo activar el cálculo MLE si los paquetes aún no están instalados.
  cat("Ajuste MLE opcional omitido: instala paquetes en Posit Cloud con:\n")
  cat("install.packages(c(\"fitdistrplus\", \"evd\"))\n")
}
