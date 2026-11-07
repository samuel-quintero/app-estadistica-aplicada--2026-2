# Plan de desarrollo: app de Estadística Aplicada

## Objetivo y criterios generales

Construir una aplicación multipágina en Streamlit para explorar datos y comparar métodos estadísticos aplicados a precipitación, secado y series de tiempo. La interfaz debe mostrar los supuestos, unidades, tamaño de muestra y advertencias relevantes junto con los resultados; un valor p o un ranking de modelos no debe presentarse como conclusión automática.

Flujo común recomendado para las páginas analíticas:

- Permitir cargar un archivo CSV o Excel o seleccionar un conjunto de datos de ejemplo cuando esté disponible.
- Validar columnas, tipos, unidades, valores faltantes, duplicados y tamaño mínimo requerido antes de calcular.
- Permitir elegir variables y opciones del análisis con controles claros.
- Mostrar un resumen interpretable, una tabla descargable y gráficas con etiquetas, unidades y leyendas.
- Explicar limitaciones y mostrar errores de validación sin ocultar el origen del problema.

## Páginas, entradas y resultados

| Página | Entradas principales | Tablas y gráficas | Funciones y cálculos |
|---|---|---|---|
| Inicio | Ninguna; opcionalmente selección de un ejemplo o acceso a un análisis reciente de la sesión. | Descripción breve de los módulos, pasos sugeridos y estado/resumen del conjunto de datos cargado. | Orientación y navegación; no realiza inferencia estadística. |
| Datos | Archivo CSV/Excel; separador, hoja y codificación si aplica; selección de columnas, tipos, unidades y valores faltantes. | Vista previa; tabla de tipos, faltantes y duplicados; estadísticas descriptivas básicas; perfil de variables y serie temporal cuando aplique. | Lectura, validación, conversión de tipos, identificación de problemas y preparación no destructiva de los datos. Mantener el original separado de cualquier versión transformada. |
| EDA: Análisis Exploratorio de Datos | Conjunto de datos cargado; variable o variables; agrupación opcional, filtros y periodo si hay fechas. | Tabla de resumen por variable o grupo; histogramas y densidades; diagramas de caja; dispersión y matriz de correlación para variables numéricas; gráficos temporales si hay fechas. | Conteos, media, mediana, desviación estándar, varianza, cuantiles, rango, asimetría y curtosis; frecuencias para categorías; correlaciones seleccionadas. No confundir exploración con prueba confirmatoria. |
| Distribuciones | Serie de máximos anuales de precipitación y unidades; distribución Normal, Lognormal, Gamma, Weibull, Gumbel o GEV; método de ajuste; nivel de significancia; periodos de retorno o horizonte de riesgo. Para secado: tiempo y razón de humedad observados, modelos candidatos y pesos/errores. | Tabla de parámetros, log-verosimilitud, KS, Anderson-Darling, AIC, BIC y AICc; densidad/histograma, CDF empírica y ajustadas, gráfico Q-Q y curva de cuantiles de retorno con incertidumbre. Para secado, tabla comparativa de parámetros y AICc, datos observados y curvas ajustadas/residuos. | Ajuste de distribuciones y estimación de parámetros; estadísticos KS y Anderson-Darling; criterios de información; cuantiles para periodos de retorno y riesgo en horizonte; intervalos de incertidumbre. Ajuste de modelos candidatos de secado y comparación con AICc. |
| Regresión | Dos variables cuantitativas, variable respuesta y explicativa; nivel de confianza; filtros/grupos opcionales. | Tabla de Pearson y Spearman; coeficientes del modelo, errores estándar, intervalos, estadísticos de prueba y medidas de ajuste; dispersión con recta e intervalo; residuos frente a ajustados, Q-Q y residuos por observación/orden. | Correlación lineal de Pearson y monótona de Spearman; recta por mínimos cuadrados ordinarios (MCO); $R^2$, $R^2$ ajustado, RMSE y análisis de varianza; prueba e intervalo de confianza para la pendiente; diagnósticos de linealidad, normalidad aproximada, varianza constante e independencia de residuos. |
| Series de tiempo | Fecha/periodo, variable, frecuencia y tratamiento de faltantes; longitud de ventana; periodo estacional; componente aditivo o multiplicativo; métodos de referencia; horizonte y esquema de validación; configuración de AutoTS. | Serie original; tendencia, estacionalidad y residuo; medias móviles e índices estacionales; pronósticos por método; tabla de MAE, RMSE y SMAPE por validación; gráfico de observado frente a pronosticado con horizonte e intervalos si el modelo los proporciona. | Identificación e interpretación de componentes; promedios móviles; índices estacionales; descomposición clásica; referencias como último valor, media, deriva y último valor estacional; validación temporal fuera de muestra; ajuste y pronóstico con AutoTS. |

## Detalle estadístico y cautelas

### Distribuciones y precipitación

- Ajustar las seis familias seleccionadas con un método explícito y consistente, registrando parámetros, log-verosimilitud y tratamiento de ceros o datos no positivos. Lognormal, Gamma y Weibull requieren soporte positivo; no transformar ni descartar observaciones silenciosamente.
- Reportar $AIC=2k-2\ell$, $BIC=k\log(n)-2\ell$ y $AICc=AIC+\frac{2k(k+1)}{n-k-1}$, donde $k$ cuenta todos los parámetros estimados y $\ell$ es la log-verosimilitud. AICc no está definido si $n\leq k+1$. Comparar criterios solo sobre las mismas observaciones y una definición de verosimilitud compatible.
- Mostrar KS y Anderson-Darling como diagnósticos de ajuste, no como una competencia aislada para declarar un modelo verdadero. Si los parámetros se estiman con la misma muestra, los valores críticos/p-valores estándar no aplican directamente; utilizar una corrección apropiada o bootstrap paramétrico y documentar el método.
- Para máximos anuales, usar la convención $T=1/[1-F(x_T)]$ y explicar que representa el periodo medio de excedencia bajo estacionariedad e independencia, no una periodicidad garantizada. El riesgo de al menos una excedencia en $m$ años es $1-(1-1/T)^m$ bajo esos mismos supuestos.
- Si se muestran incertidumbres de parámetros o cuantiles, distinguir intervalo de confianza del cuantíl de intervalo predictivo; especificar método (por ejemplo, bootstrap) y alcance. Verificar con especial cuidado la parametrización de GEV y los signos de sus convenciones.

### Modelos de secado en capa delgada

- Ajustar cada modelo candidato a la misma respuesta y observaciones, usando una función de error y una verosimilitud comparables. Presentar residuos y parámetros además de AICc; valores de AICc menores son evidencia relativa dentro del conjunto evaluado, no una prueba de superioridad absoluta.
- Documentar la definición de razón de humedad, restricciones físicas de parámetros, estrategia de ajuste no lineal y cómo se cuentan parámetros. No comparar modelos con distintos datos efectivos sin advertirlo.

### Regresión

- Pearson resume asociación lineal y es sensible a valores extremos; Spearman usa rangos y resume asociación monótona. Mostrar tamaño de muestra y advertir sobre faltantes y empates.
- Para MCO simple, informar intercepto y pendiente, sus errores estándar e intervalos, prueba $H_0:\beta_1=0$, valor p, grados de libertad y ajuste. Aclarar que la inferencia usual requiere supuestos sobre errores; correlación o significancia no implican causalidad.
- Los gráficos de residuos son diagnósticos, no validaciones definitivas. Si los supuestos fallan, explicar consecuencias y posibles extensiones sin presentar automáticamente el modelo simple como adecuado.

### Series de tiempo

- Mostrar tendencia, estacionalidad y residuo solo con frecuencia y longitud suficientes; la descomposición clásica necesita definir el periodo y puede perder observaciones en los extremos por los promedios móviles.
- Calcular métricas sobre pronósticos genuinamente futuros mediante partición temporal o validación de origen rodante, nunca mediante partición aleatoria. Definir $MAE$, $RMSE$ y la convención de $SMAPE$; reportar que SMAPE puede ser indefinido o inestable cuando observado y pronosticado son ambos cero o cercanos a cero.
- Tratar AutoTS como una dependencia opcional: fijar versión y opciones en una etapa posterior, restringir el coste de búsqueda y conservar una línea base para saber si mejora métodos sencillos. No prometer resultados reproducibles sin registrar configuración, semilla y datos.

## Organización modular propuesta

```text
app.py                         # entrada, configuración global y bienvenida
pages/
  1_Datos.py                   # carga, revisión y preparación de datos
  2_EDA.py                     # exploración univariada y bivariada
  3_Distribuciones.py          # precipitación y secado
  4_Regresion.py               # correlación, MCO e inferencia
  5_Series_de_tiempo.py        # descomposición, validación y pronóstico
src/
  carga_datos.py               # lectura, validación y preparación
  validacion.py                # validaciones reutilizables y mensajes
  distribuciones.py            # ajustes, pruebas, criterios y cuantiles
  secado.py                    # modelos de capa delgada y comparación
  regresion.py                 # correlaciones, MCO y diagnósticos
  series_tiempo.py             # componentes, referencias y métricas
  graficos.py                  # funciones de visualización reutilizables
  estado.py                    # estado compartido de Streamlit, si se necesita
datos/
  README.md                    # procedencia, unidades y diccionario de datos
  ejemplo/                     # datos pequeños y no sensibles, si se incorporan
verificacion/
  test_validacion.py            # entradas y casos límite
  test_distribuciones.py        # parámetros, soporte y fórmulas
  test_regresion.py             # casos con resultados conocidos
  test_series_tiempo.py         # métricas, frecuencia y validación temporal
```

- `app.py` debe ser delgado: configuración de página, navegación/introducción y estado compartido imprescindible. Streamlit descubre páginas en `pages/`; evitar duplicar lógica estadística allí.
- `src/` contiene funciones de cálculo independientes de la interfaz para facilitar pruebas y reutilización. Las páginas leen controles, llaman a esas funciones y presentan resultados.
- `datos/` contiene datos de ejemplo y documentación de procedencia; no guardar datos privados ni archivos cargados por usuarios en el repositorio. Mantener resultados calculados fuera de los datos fuente.
- `verificacion/` agrupa pruebas automatizadas de fórmulas, validaciones y casos límite. Añadirlas conforme se construye cada módulo, usando muestras sintéticas pequeñas con resultados esperados.

## Orden recomendado de construcción

1. **Definir contratos de datos y alcance estadístico.** Acordar nombres de columnas, unidades, frecuencia, supuestos, modelos de secado candidatos y tratamiento de datos inválidos. Esto evita que cada página interprete los datos de forma distinta.
2. **Implementar la página Datos y validaciones reutilizables.** Sin datos consistentes no se pueden interpretar los análisis posteriores; probar primero carga, tipos, faltantes y mensajes de error.
3. **Añadir Inicio y EDA.** Inicio organiza la navegación y EDA permite verificar visualmente los datos antes de aplicar modelos; es la forma más rápida de detectar problemas de unidades, atípicos o frecuencia.
4. **Construir Regresión.** Es un módulo acotado que establece cómo separar cálculo, visualización e inferencia, y permite probar fórmulas y diagnósticos con datos sintéticos conocidos.
5. **Construir Distribuciones por etapas.** Primero ajuste, parámetros, log-verosimilitud y criterios de información; después pruebas de ajuste con corrección por parámetros estimados; finalmente cuantiles de retorno, riesgo e incertidumbre. Incorporar el secado comparando AICc una vez validada la contabilidad de parámetros.
6. **Construir Series de tiempo desde referencias hasta AutoTS.** Primero frecuencia, componentes y descomposición; luego pronósticos base y validación temporal; después métricas y, por último, AutoTS. Las referencias aportan una comparación transparente antes de introducir una búsqueda de modelos más compleja.
7. **Verificar el flujo integrado y documentar límites.** Revisar navegación entre páginas, estado de datos, mensajes, unidades, casos límite y coherencia de resultados. La verificación final debe incluir ejemplos pequeños y conocidos, además de una ejecución con un conjunto real documentado.

En cada etapa, mantener interfaz y cálculos separados, probar el módulo antes de conectarlo a las páginas y registrar las decisiones estadísticas que afecten la interpretación.