# Verifica regresión simple con fórmulas explícitas y contrasta con lm() y summary().

# Se detecta si R se ejecuta desde la raíz del proyecto o desde verificacion/.
carpeta_datos <- if (dir.exists("datos")) "datos" else file.path("..", "datos")

# Se arma la ruta de la muestra usando la carpeta detectada.
ruta_datos <- file.path(carpeta_datos, "muestra_reg8.csv")

# Se leen los ocho pares completos de lámina de riego y rendimiento.
datos <- read.csv(ruta_datos)
datos <- datos[complete.cases(datos[, c("lamina_riego", "rendimiento")]), ]
x <- datos$lamina_riego
y <- datos$rendimiento
n <- length(x)

# Se calculan medias, diferencias centradas y sumas Sxx, Syy y Sxy explícitamente.
x_media <- sum(x) / n
y_media <- sum(y) / n
dx <- x - x_media
dy <- y - y_media
Sxx <- sum(dx^2)
Syy <- sum(dy^2)
Sxy <- sum(dx * dy)

# Pearson se calcula por Sxy/raíz(Sxx·Syy); Spearman aplica la fórmula a rangos.
r <- Sxy / sqrt(Sxx * Syy)
rango_x <- rank(x, ties.method = "average")
rango_y <- rank(y, ties.method = "average")
rho <- sum((rango_x - mean(rango_x)) * (rango_y - mean(rango_y))) /
  sqrt(sum((rango_x - mean(rango_x))^2) * sum((rango_y - mean(rango_y))^2))

# Se calculan b1=Sxy/Sxx y b0=ȳ-b1·x̄, y después ajustados y residuos.
b1 <- Sxy / Sxx
b0 <- y_media - b1 * x_media
y_ajustado <- b0 + b1 * x
residuo <- y - y_ajustado

# Se calculan SCE, SCT y R²=1-SCE/SCT a partir de las observaciones y ajustados.
SCE <- sum(residuo^2)
SCT <- sum((y - y_media)^2)
SCR <- sum((y_ajustado - y_media)^2)
R2 <- 1 - SCE / SCT

# RSE usa n-2 grados de libertad y SE(b1)=RSE/raíz(Sxx).
gl <- n - 2
RSE <- sqrt(SCE / gl)
SE_b1 <- RSE / sqrt(Sxx)
t_observado <- b1 / SE_b1

# Se ajusta t y su inferencia bilateral mediante la distribución t con gl=n-2.
modelo_lm <- lm(rendimiento ~ lamina_riego, data = datos)
resumen_lm <- summary(modelo_lm)
coeficientes_lm <- resumen_lm$coefficients
intervalos_lm <- confint(modelo_lm, level = 0.95)

# El p-valor, decisión e intervalo de b1 se extraen de summary() y confint() de lm().
p_valor <- coeficientes_lm["lamina_riego", "Pr(>|t|)"]
decision <- if (p_valor < 0.05) "Se rechaza H0" else "No se rechaza H0"

# Se imprimen cálculos explícitos con cuatro decimales.
cat(sprintf("Archivo: %s; n=%d\n", basename(ruta_datos), n))
cat(sprintf("Sxx: %.4f\nSyy: %.4f\nSxy: %.4f\n", Sxx, Syy, Sxy))
cat(sprintf("r de Pearson: %.4f\nrho de Spearman: %.4f\n", r, rho))
cat(sprintf("b0: %.4f\nb1: %.4f\n", b0, b1))
cat(sprintf("SCE: %.4f\nSCR: %.4f\nSCT: %.4f\nR²: %.4f\n", SCE, SCR, SCT, R2))
cat(sprintf("RSE: %.4f\nSE(b1): %.4f\nt: %.4f\n", RSE, SE_b1, t_observado))
cat(sprintf("p-valor: %.4f\nDecisión al 5 %%: %s\n", p_valor, decision))
cat(sprintf("IC 95 %% de b1: [%.4f, %.4f]\n",
            intervalos_lm["lamina_riego", 1], intervalos_lm["lamina_riego", 2]))

# Se muestra summary(lm) completo para comparar coeficientes, errores estándar y pruebas.
cat("\nResumen de lm():\n")
print(resumen_lm)
