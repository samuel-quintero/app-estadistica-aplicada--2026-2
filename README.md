# Estadística Aplicada con Python y R

## Datos del estudiante

Información leída de `datos/metadatos.json`:

| Campo | Valor |
|---|---|
| Nombre | Samuel David Quintero Palencia |
| Código | 1034285837 |
| Grupo | No indicado en los metadatos; completar por el estudiante |
| Cultivo | Plátano |
| Municipio | Corozal |
| Huella | F3C2B3E5 |

## Descripción

Aplicación web multipágina construida con Streamlit para explorar datos agrícolas y practicar Estadística Aplicada. Incluye datos simulados personalizados y herramientas para análisis exploratorio, ajuste de distribuciones, regresión y series de tiempo. Los cálculos están organizados en módulos de `src/`, separados de la interfaz, y las muestras pequeñas permiten contrastar resultados a mano, en Python, en R y en la aplicación.

La huella y la semilla de los datos se derivan del código estudiantil. Los archivos generados son reproducibles para este código, pero no representan mediciones de campo.

## Objetivo académico

Desarrollar la capacidad de formular, calcular, verificar e interpretar métodos estadísticos aplicados a problemas de Ingeniería Agrícola. La aplicación apoya la exploración y la comprobación de cálculos; no sustituye la justificación de supuestos ni el criterio del estudiante.

## Páginas

| Página | Temas principales |
|---|---|
| **Inicio** | Propósito, navegación, metodología “Lápiz y Algoritmo” y responsabilidad académica. |
| **Datos** | Selección de conjuntos, vista tabular, tipos y clasificación de variables, faltantes, descripción y descarga; admite archivos CSV y Excel. |
| **EDA** | Filtros por suelo y riego; resúmenes numéricos; histogramas, densidad y cajas; frecuencias; asociaciones; faltantes y atípicos; verificación con diez parcelas. |
| **Distribuciones** | Máximos anuales de precipitación; Normal, Lognormal, Gamma, Weibull, Gumbel y GEV; KS/Anderson-Darling, AIC/BIC/AICc, cuantiles de retorno, riesgo e incertidumbre; modelos de secado en capa delgada. |
| **Regresión** | Correlaciones Pearson/Spearman; MCO simple y múltiple; inferencia de la pendiente; predicción e intervalos; diagnóstico de residuos y verificación manual con ocho parcelas. |
| **Series de tiempo** | Promedios móviles; índices estacionales; descomposición clásica; referencias ingenua, estacional ingenua y promedio; validación MAE/RMSE/SMAPE; pronóstico con AutoTS. |

## Tecnologías

- Python y Streamlit para la aplicación y sus páginas multipágina.
- pandas y NumPy para manejar datos y realizar cálculos numéricos.
- SciPy y statsmodels para distribuciones, pruebas, regresión y series.
- Matplotlib y seaborn para las gráficas.
- AutoTS para búsquedas de pronóstico configurables.
- openpyxl para exportar tablas a Excel y pytest para las pruebas de equivalencia.
- R base y, opcionalmente, `fitdistrplus` y `evd` para verificación independiente en Posit Cloud.

Las dependencias Python están fijadas en `requirements.txt` a las versiones comprobadas en este entorno. NumPy y SciPy requieren Python 3.12 o posterior; para el despliegue se recomienda seleccionar **Python 3.12** en Community Cloud. `scikit-learn` se declara porque AutoTS lo requiere, y `pytest` porque el repositorio incluye pruebas automatizadas.

## Estructura del proyecto

```text
.
├── app.py
├── pages/
│   ├── 1_Datos.py
│   ├── 2_EDA.py
│   ├── 3_Distribuciones.py
│   ├── 4_Regresion.py
│   └── 5_Series_de_tiempo.py
├── src/
│   ├── datos.py
│   ├── eda.py
│   ├── distribuciones.py
│   ├── modelos_curvas.py
│   ├── regresion.py
│   ├── series.py
│   ├── validacion.py
│   └── exportar.py
├── datos/
│   ├── metadatos.json
│   ├── parcelas.csv
│   ├── precip_max_anual.csv
│   ├── clima_mensual.csv
│   ├── secado.csv
│   ├── muestra_eda10.csv
│   ├── muestra_dist10.csv
│   ├── muestra_reg8.csv
│   └── muestra_series24.csv
├── resultados/
│   ├── autots_pronostico.csv
│   ├── autots_top10_modelos.csv
│   └── autots_mejor_modelo.json
├── verificacion/
│   ├── verificar_eda.py / verificar_eda.R
│   ├── verificar_distribuciones.py / verificar_distribuciones.R
│   ├── verificar_regresion.py / verificar_regresion.R
│   ├── verificar_series.py / verificar_series.R
│   └── RESULTADOS.md
├── tests/
│   └── test_calculos.py
├── docs/
│   └── PLAN_DESARROLLO.md
├── pronostico_autots.py
├── requirements.txt
├── .github/copilot-instructions.md
├── .streamlit/config.toml
└── .vscode/settings.json
```

Streamlit descubre las páginas dentro de `pages/`. Los cálculos reutilizables están en `src/`; `verificacion/` contiene scripts independientes de la interfaz para contrastar las muestras pequeñas. `resultados/` almacena pronósticos ya ejecutados, y los tests verifican resultados de `src/` frente a los verificadores Python.

## Ejecutar en GitHub Codespaces

1. Abre el repositorio en un Codespace y abre una terminal en la raíz del proyecto.
2. Comprueba Python con `.venv/bin/python --version`. Si el entorno aún no existe, créalo con `python -m venv .venv`.
3. Instala las dependencias declaradas con `.venv/bin/pip install -r requirements.txt`.
4. Inicia la aplicación desde la raíz con `.venv/bin/streamlit run app.py`.
5. En la pestaña **Puertos**, abre el puerto `8501` en el navegador.
6. Para ejecutar las pruebas de equivalencia, usa `.venv/bin/python -m pytest -q`.
7. Los verificadores Python se ejecutan, por ejemplo, con `.venv/bin/python verificacion/verificar_eda.py`. Los scripts R se pueden abrir y ejecutar desde Posit Cloud; para el ajuste Gumbel opcional consulta las instrucciones de su propio script.

No publiques secretos en el repositorio. La configuración de Streamlit desactiva XSRF para el flujo de archivos en Codespaces; no reutilices esa configuración sin revisar la protección correspondiente antes de desplegar la app públicamente.

## App en línea

**Enlace público:** ______________________________________________

### Despliegue en Streamlit Community Cloud

1. Sube el repositorio a GitHub y confirma que `app.py`, `requirements.txt`, `pages/`, `src/` y `datos/` estén en la rama que vas a publicar. No subas `.venv/` ni secretos.
2. Entra en [Streamlit Community Cloud](https://share.streamlit.io/) e inicia sesión con la cuenta de GitHub que tiene acceso al repositorio.
3. Selecciona **Create app** y elige el repositorio, la rama y `app.py` como archivo principal. El punto de entrada es `app.py`; Streamlit descubre las demás páginas en `pages/`.
4. Abre **Advanced settings** y selecciona **Python 3.12**. No hace falta `runtime.txt`: Community Cloud permite elegir la versión en esta pantalla y su versión predeterminada documentada es 3.12.
5. No necesitas configurar secretos para el flujo actual. Si en el futuro agregas credenciales, introdúcelas en **Secrets** de la app y no en archivos versionados.
6. Pulsa **Deploy** y revisa los logs si la instalación o el inicio fallan. Cuando termine, copia la URL pública y pégala en el espacio de arriba.

### Configuración de Streamlit

El archivo `.streamlit/config.toml` define `headless = true`, apropiado para un servidor sin interfaz local. También contiene `enableXsrfProtection = false`, ajuste usado para el flujo de `st.file_uploader` en Codespaces. Community Cloud lee la configuración del repositorio: antes de publicar una app accesible públicamente, cambia esa opción a `true` o elimina la línea para conservar la protección XSRF. Si el cargador de archivos vuelve a fallar en Codespaces, conserva una configuración específica para ese entorno o una rama separada y valida el comportamiento en cada despliegue; no desactives la protección pública sin evaluar el riesgo.

### AutoTS y límites de la nube

AutoTS puede consumir tiempo y memoria al evaluar candidatos. Empieza con `model_list="superfast"`; si la búsqueda tarda demasiado, usa 2 generaciones y 1 validación (el formulario de la app permite reducir ambas). Para consultar el pronóstico previamente generado, abre **Resultados guardados**: esa sección carga los archivos de `resultados/` sin volver a ejecutar AutoTS. Asegúrate de que esos CSV/JSON estén incluidos en la rama si quieres que aparezcan en la app desplegada.

### Errores comunes

- **Fallo al instalar dependencias:** comprueba que Community Cloud use Python 3.12 y que `requirements.txt` esté en la raíz. Revisa el log de resolución de paquetes; las versiones están fijadas para evitar que una actualización futura cambie el entorno sin aviso.
- **`ModuleNotFoundError` o archivos no encontrados:** usa `app.py` como archivo principal y conserva la estructura `src/`, `pages/` y `datos/` con la misma capitalización. Verifica que los nueve CSV/JSON de datos necesarios estén subidos a GitHub; las rutas de los módulos parten de la ubicación de `__file__`, no del directorio de trabajo de una terminal.
- **AutoTS tarda o la app parece consumir mucha memoria:** selecciona `superfast`, reduce generaciones a 2 y validaciones a 1, evita ejecutar varias búsquedas simultáneas y usa **Resultados guardados** para revisar un pronóstico existente.
- **Problemas con `st.file_uploader` o XSRF:** conserva la protección habilitada en la app pública. Si el ajuste especial de Codespaces es necesario para el desarrollo, usa una configuración/ramificación específica y prueba el cargador por separado.
- **Cambios de dependencias no reflejados:** revisa los logs del despliegue y fuerza un reinicio o redeploy desde Community Cloud; no intentes corregirlo instalando paquetes manualmente en el servidor efímero.

## Metodología: Lápiz y Algoritmo

1. Define la pregunta, las variables, sus unidades y la población o periodo que representan.
2. Resuelve primero la muestra pequeña a mano y anota antes tu predicción o expectativa.
3. Ejecuta el verificador independiente en Python o R y compara cada paso, no solo el resultado final.
4. Repite el análisis en la aplicación para trabajar con el conjunto completo.
5. Si hay diferencias, revisa fórmulas, redondeo, tipo de cuantil, divisor muestral, faltantes, unidades y supuestos.
6. Escribe la interpretación con cifras y unidades, y distingue asociación, ajuste y causalidad.

## Limitaciones

- **Datos simulados:** parcelas, precipitaciones, clima y secado fueron generados para fines didácticos. El rendimiento incluye relaciones y ruido programados; no debe interpretarse como evidencia agronómica de campo.
- **Muestras pequeñas:** las muestras de 8, 10 o 24 observaciones permiten practicar procedimientos, pero no representan necesariamente la incertidumbre de estudios reales.
- **Distribuciones:** Normal, Lognormal, Gamma, Weibull, Gumbel y GEV son candidatas, no una lista exhaustiva. Las pruebas de ajuste tienen potencia limitada; AIC, BIC y AICc comparan modelos sobre los mismos datos, pero no prueban que el ganador sea verdadero. Los cuantiles de retorno y el riesgo suponen un proceso aproximadamente estacionario y años independientes. En SciPy, el parámetro de forma `c` de `genextreme` usa signo opuesto al parámetro GEV ξ.
- **Modelos de secado:** Lewis, Henderson-Pabis, Page, Logarítmico y Dos términos son modelos semiempíricos. AICc depende de que se ajusten a las mismas observaciones y del conteo de parámetros; el mejor ajuste relativo no garantiza que describa el mecanismo físico.
- **Regresión:** la inferencia MCO usual requiere supuestos sobre errores y diseño. Correlación y significancia no demuestran causalidad; extrapolar fuera del rango observado es especialmente incierto.
- **Series de tiempo:** los índices y la descomposición aditivos simplifican la estructura; la descomposición clásica necesita ciclos suficientes y puede dejar extremos sin tendencia. Los métodos de referencia son líneas base, no modelos universales.
- **AutoTS:** la búsqueda se limita a catálogos rápidos (`superfast` o `fast`) y a generaciones/validaciones acotadas. Puede descartar candidatos, y el modelo ganador depende de la serie, los parámetros y la versión de AutoTS; los intervalos no garantizan cobertura futura.
- **Datos faltantes y exportación:** la app los señala, pero la decisión de excluir, imputar o transformar corresponde al análisis y debe justificarse. Las descargas reproducen las tablas y figuras de la ejecución actual.

## Posibles mejoras

- Incorporar mediciones reales documentadas, diccionario de variables, unidades y procedencia.
- Versionar dependencias y registrar versiones, parámetros y semilla junto a cada ejecución de AutoTS.
- Ampliar la validación temporal con origen rodante y comparar intervalos predictivos.
- Añadir diagnósticos más completos para autocorrelación, heterocedasticidad y estabilidad de parámetros.
- Permitir que el estudiante guarde en `resultados/` reportes con configuración, fecha y fuente de datos.
- Ampliar los casos de prueba para errores de entrada, fechas incompletas y valores extremos.

## Hallazgos del EDA

(escrito por el estudiante, con cifras y unidades)

## Hallazgos de distribuciones

(escrito por el estudiante, con cifras y unidades)

## Hallazgos de la regresión

(escrito por el estudiante, con cifras y unidades)

## Hallazgos de series de tiempo

(escrito por el estudiante, con cifras y unidades)

## Referencias básicas

- Streamlit. *Documentación oficial*. https://docs.streamlit.io/
- pandas. *Documentación oficial*. https://pandas.pydata.org/docs/
- NumPy. *Documentación oficial*. https://numpy.org/doc/
- SciPy. *Documentación de estadística*. https://docs.scipy.org/doc/scipy/reference/stats.html
- statsmodels. *Documentación oficial*. https://www.statsmodels.org/stable/
- AutoTS. *Repositorio y documentación del proyecto*. https://github.com/winedarksea/AutoTS
- Hyndman, R. J. y Athanasopoulos, G. *Forecasting: Principles and Practice*. https://otexts.com/fpp3/
- R Core Team. *Documentación de R*. https://www.r-project.org/
- Delignette-Muller, M. L. y Dutang, C. *fitdistrplus: An R package for fitting distributions*. https://cran.r-project.org/package=fitdistrplus
- Stephenson, A. G. *evd: Extreme Value Distributions*. https://cran.r-project.org/package=evd
