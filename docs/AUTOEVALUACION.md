# Autoevaluación técnica del proyecto

**Fecha de revisión:** 7 de noviembre de 2026  
**Alcance:** inspección de la estructura, módulos `src/`, páginas Streamlit, verificadores, pruebas y resultados guardados. Esta autoevaluación documenta hallazgos; no modifica los cálculos ni representa una calificación académica.

## 1. Organización y modularidad

La separación principal es apropiada para un proyecto didáctico:

- `app.py` configura Inicio y enlaza las cinco páginas ubicadas en `pages/`.
- `pages/` reúne la interfaz, controles, tablas, gráficas, validaciones y descargas.
- `src/` separa cálculos de EDA, distribuciones, regresión, series, curvas de secado, validación y exportación.
- `datos/` contiene datos simulados personalizados, metadatos y cuatro muestras pequeñas.
- `verificacion/` contiene scripts independientes en Python y R, además de la plantilla de resultados.
- `tests/test_calculos.py` compara resultados de los módulos de `src/` con los verificadores Python mediante aserciones.
- `resultados/` conserva archivos de pronóstico AutoTS; `docs/` contiene el plan y esta evaluación.

La modularidad es funcional: las páginas consumen funciones de `src/` y los scripts manuales no dependen de la interfaz. Como mejora, las páginas largas podrían dividir la preparación de tablas y reportes en funciones de presentación sin trasladar ni duplicar fórmulas.

## 2. Comentarios y docstrings

Los módulos de cálculo y verificadores incluyen docstrings y comentarios en español que explican fórmulas y pasos. La interfaz mantiene textos, unidades y mensajes principalmente en español. Los comentarios facilitan el uso como material de clase.

La principal oportunidad de mejora es complementar los comentarios con documentación de contratos de entrada/salida: unidades esperadas, comportamiento ante valores no finitos, supuestos y excepciones posibles. Los comentarios no sustituyen esa especificación y algunos bloques largos de interfaz se beneficiarían de funciones auxiliares de presentación.

## 3. Correctitud estadística

| Aspecto revisado | Evidencia en el código | Evaluación |
|---|---|---|
| Varianza y desviación en EDA | `src/eda.py` usa `ddof=1` / varianza muestral. | Correcto: divisor $n-1$. |
| Cuartiles y Tukey | `src/eda.py` usa interpolación lineal para Q1/Q3, equivalente al tipo 7 de R; las vallas son $Q_1-1.5RIC$ y $Q_3+1.5RIC$. | Correcto; las pruebas también comparan vallas y atípicos con el verificador. |
| Parámetros en AIC/BIC/AICc | `src/distribuciones.py` cuenta forma, ubicación y escala, descontando `loc=0` en Lognormal, Gamma y Weibull. Normal, Gumbel: $k=2$; Lognormal, Gamma y Weibull con ubicación fija: $k=2$; GEV: $k=3$. | Correcto para los ajustes implementados. AICc se marca indefinido si $n\leq k+1$. |
| Signo de forma GEV | El módulo indica que en `scipy.stats.genextreme` el parámetro `c` tiene signo opuesto a $\xi$. | Correcto y explícitamente documentado. |
| Pruebas KS/Anderson-Darling | `src/distribuciones.py` usa `scipy.stats.goodness_of_fit` con réplicas Monte Carlo y separa parámetros conocidos de estimados; además calcula un KS ingenuo para comparación. | Correcto para abordar el efecto de estimar parámetros; el KS ingenuo está distinguido del p-valor corregido. |
| Regresión e inferencia | `src/regresion.py` calcula RSE con $n-2$, $SE(b_1)=RSE/\sqrt{S_{xx}}$, t bilateral, p-valor e IC de la pendiente. | Coincide con statsmodels en la muestra de ocho parcelas según el bloque principal y los tests. |
| SMAPE | `src/series.py` calcula $200\,\mathrm{media}(|y-\hat y|/(|y|+|\hat y|))$; el caso $0/0$ aporta cero. | Fórmula correcta y división por cero controlada. |
| Ingenuo estacional | `src/series.py` repite los últimos 12 meses con `datos.iloc[-12:]` y `np.resize`. | Correcto para el mismo mes del año anterior, también al extender horizontes mayores de 12. |
| Modelos de secado | `src/modelos_curvas.py` usa los parámetros libres de cada curva en AICc; la página aclara que $\chi^2$ reducido usa SCE/(n-k) al no haber varianzas de medición. | Comparación relativa consistente; la aclaración evita presentar ese diagnóstico como prueba formal de chi-cuadrado. |

No se detectó una discrepancia en los seis puntos estadísticos solicitados. Los criterios de información comparan modelos sobre los mismos datos, y el riesgo de falla se presenta bajo los supuestos de independencia y estacionariedad.

## 4. Funcionamiento de páginas y pruebas

Se ejecutó `.venv/bin/python -m pytest -q`: **4 passed**. Las pruebas comparan cálculos de EDA, distribuciones, regresión y series con los scripts independientes de `verificacion/`.

También se ejecutaron las páginas mediante Streamlit AppTest y se inició la aplicación real con `.venv/bin/streamlit run app.py --server.port 8502`. El servidor respondió **HTTP 200** en `/_stcore/health`.

| Página | Resultado de ejecución observado |
|---|---|
| Inicio | Sin excepciones; cinco enlaces directos a las páginas. |
| Datos | Sin excepciones; tabla, métricas y descarga. |
| EDA | Sin excepciones; 10 tablas y cuatro opciones de descarga en el expander. |
| Distribuciones | Sin excepciones; 10 tablas y cuatro opciones de descarga. Los ajustes Monte Carlo y de secado finalizaron con los datos del proyecto. |
| Regresión | Sin excepciones; 7 tablas y cuatro opciones de descarga. |
| Series de tiempo | Sin excepciones; 9 tablas, formulario AutoTS y cuatro opciones de descarga. |

AutoTS no se envió desde AppTest durante esta revisión para evitar repetir una búsqueda potencialmente larga. El repositorio sí contiene los resultados guardados de una ejecución anterior; la vista previa confirma que la página puede mostrar esos archivos sin ejecutar de nuevo el modelo. Algunos candidatos AutoTS pueden fallar y ser descartados por la librería; eso no impidió completar la búsqueda registrada.

## 5. Coincidencia de verificaciones

Los tests de `tests/test_calculos.py` ejecutan los scripts Python de `verificacion/` con `runpy` y cotejan resultados numéricos:

- **EDA:** media, mediana, varianza, desviación, CV, Q1/Q3 tipo 7, RIC, vallas, atípicos y medias por riego.
- **Distribuciones:** posiciones Weibull, periodos empíricos, parámetros Gumbel por momentos, cuantiles y riesgo.
- **Regresión:** Sxx/Syy/Sxy, Pearson/Spearman, coeficientes, SCE, R², RSE, SE, t, p e IC.
- **Series:** PM3, índices aditivos, tres métodos de referencia y MAE/RMSE/SMAPE.

La app utiliza esas mismas funciones de `src/` en sus secciones de verificación, por lo que la coincidencia entre `src/` y `verificacion/` respalda la ruta de cálculo mostrada en la interfaz. Las verificaciones R no se ejecutaron en este entorno; deben correrse en Posit Cloud para confirmar la paridad entre Python y R, en particular el ajuste Gumbel opcional con `fitdistrplus` y `evd`.

## 6. Correcciones priorizadas

### Crítica

- **Ninguna confirmada** en los cálculos revisados ni en la ejecución de páginas y pruebas.

### Importante

1. **Ampliar la cobertura automatizada de validación y exportación.** Las cuatro pruebas comparan casos válidos de las muestras pequeñas; no prueban sistemáticamente archivos vacíos, tipos incorrectos, fechas incompletas, mensajes pedagógicos, generación real de Excel/PNG/PDF ni formularios de AutoTS. Añadir tests de esos contratos reduciría el riesgo de regresiones de interfaz.
2. **Ejecutar los verificadores R en Posit Cloud.** En esta revisión solo se ejecutaron Python y la app. La rama opcional de MLE con `fitdistrplus`/`evd` está condicionada a disponibilidad y no cuenta con evidencia de ejecución aquí.

### Deseable

1. **Definir el caso de regresión perfectamente ajustada.** En `src/regresion.py`, si SCE es exactamente cero, `SE(b1)` también es cero y la operación `t=b1/SE(b1)` puede dividir por cero. Conviene definir y probar explícitamente el comportamiento inferencial en ese límite.
2. **Fijar y registrar versiones de dependencias.** `requirements.txt` no fija versiones; conservar las versiones de Python, SciPy, statsmodels y AutoTS junto a un análisis facilita reproducir valores p Monte Carlo y búsquedas AutoTS.
3. **Añadir verificación de origen para los CSV simulados.** El README ya distingue datos simulados de observaciones de campo; una futura capa de metadatos podría registrar semilla, fecha y versión del generador en cada exportación.
