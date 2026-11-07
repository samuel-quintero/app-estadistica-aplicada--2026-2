"""Análisis de distribuciones para máximos anuales de precipitación."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.datos import cargar_csv, tarjeta_estudiante
from src.distribuciones import (
    CANDIDATAS,
    ajustar,
    bootstrap_cuantil,
    bondad_ajuste,
    criterios_informacion,
    cuantil_retorno,
    gumbel_momentos,
    posiciones_graficas,
    riesgo_falla,
    tabla_comparativa,
)
from src.modelos_curvas import MODELOS, ajustar_curva, tabla_modelos


# Se fijan los periodos de retorno usados en las tablas y curvas de diseño.
PERIODOS_RETORNO = [2, 5, 10, 25, 50, 100]


@st.cache_data(show_spinner=False)
def tabla_comparativa_cacheada(serie: pd.Series, n_mc: int) -> pd.DataFrame:
    """Guarda la tabla costosa de pruebas Monte Carlo según datos y n_mc."""
    # La caché evita repetir los ajustes y remuestreos si la entrada no cambia.
    return tabla_comparativa(serie, n_mc=n_mc)


@st.cache_data(show_spinner=False)
def ajuste_cacheado(serie: pd.Series, nombre: str) -> tuple[tuple, int]:
    """Reutiliza el ajuste MLE de una distribución para gráficos y cuantiles."""
    # Los parámetros ajustados son iguales mientras se mantengan la serie y el modelo.
    return ajustar(serie, nombre)


@st.cache_data(show_spinner=False)
def bootstrap_cacheado(
    serie: pd.Series,
    nombre: str,
    periodo: float,
    semilla: int,
) -> tuple[float, float]:
    """Cachea el intervalo bootstrap de 500 réplicas para el cuantil seleccionado."""
    # Se conserva el B=500 predeterminado del módulo y se fija la semilla para reproducir.
    return bootstrap_cuantil(
        serie,
        nombre,
        periodo,
        B=500,
        semilla=semilla,
    )


def cuantiles_qq(serie: pd.Series, nombre: str, params: tuple) -> tuple[np.ndarray, np.ndarray, float]:
    """Calcula cuantiles teóricos/observados y correlación de apoyo para el Q-Q."""
    # Se ordena la muestra y se asignan probabilidades centrales para evitar extremos.
    observados = np.sort(serie.dropna().to_numpy(dtype=float))
    probabilidades = (np.arange(1, observados.size + 1) - 0.5) / observados.size

    # Se obtienen cuantiles de la candidata con sus parámetros ya ajustados.
    teoricos = CANDIDATAS[nombre].ppf(probabilidades, *params)
    finitos = np.isfinite(teoricos) & np.isfinite(observados)
    if finitos.sum() < 2:
        return teoricos, observados, math.nan

    # La correlación Q-Q se usa solo como una guía cuantitativa, no como prueba formal.
    correlacion = float(np.corrcoef(teoricos[finitos], observados[finitos])[0, 1])
    return teoricos, observados, correlacion


def interpretar_delta_aic(delta: float) -> str:
    """Asigna una interpretación relativa de soporte a cada diferencia de AIC."""
    # Se completan los intervalos intermedios de la regla usual de ΔAIC.
    if delta < 2:
        return "Soporte sustancial"
    if delta < 4:
        return "Soporte débil"
    if delta <= 7:
        return "Menor soporte"
    if delta <= 10:
        return "Soporte muy débil"
    return "Sin soporte"


# Se muestra la tarjeta personal y se cargan la serie anual y la muestra de cálculo manual.
tarjeta_estudiante()
datos = cargar_csv("precip_max_anual.csv")
muestra = cargar_csv("muestra_dist10.csv")
datos = datos.sort_values("anio").reset_index(drop=True)
serie = datos["pmax24h_mm"].dropna()

# A. Se presenta la serie de máximos anuales, un resumen y su distribución empírica.
st.title("Distribuciones de precipitación máxima anual")
st.header("Los datos")
columna_serie, columna_resumen = st.columns([1.5, 1])
with columna_serie:
    figura, eje = plt.subplots(figsize=(10, 4))
    sns.barplot(data=datos, x="anio", y="pmax24h_mm", color="#287c78", ax=eje)
    paso_etiquetas = max(1, len(datos) // 10)
    eje.set_xticks(
        np.arange(0, len(datos), paso_etiquetas),
        labels=datos["anio"].iloc[::paso_etiquetas].astype(str),
        rotation=45,
    )
    eje.set_xlabel("Año")
    eje.set_ylabel("Precipitación máxima en 24 h (mm)")
    eje.set_title("Máximo anual de precipitación en 24 horas")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

# Se calcula el resumen descriptivo y se grafica la distribución de los máximos.
with columna_resumen:
    resumen_datos = pd.DataFrame(
        {
            "Estadístico": ["n", "Media", "Mediana", "Desviación estándar muestral", "Mínimo", "Máximo"],
            "Valor": [
                len(serie),
                serie.mean(),
                serie.median(),
                serie.std(ddof=1),
                serie.min(),
                serie.max(),
            ],
        }
    )
    st.dataframe(
        resumen_datos,
        width="stretch",
        hide_index=True,
        column_config={"Valor": st.column_config.NumberColumn(format="%.4f")},
    )
    figura, eje = plt.subplots(figsize=(7, 4))
    sns.histplot(data=datos, x="pmax24h_mm", bins="auto", kde=True, ax=eje, color="#e7a34b")
    eje.set_xlabel("Precipitación máxima en 24 h (mm)")
    eje.set_ylabel("Frecuencia (años)")
    eje.set_title("Histograma de máximos anuales")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

# B. Se seleccionan candidatas, se comparan densidades y se introduce el respaldo teórico.
st.header("Candidatas y ajuste")
modelos_seleccionados = st.multiselect(
    "Distribuciones para comparar",
    options=list(CANDIDATAS),
    default=list(CANDIDATAS),
)
if not modelos_seleccionados:
    st.warning("Selecciona al menos una distribución para continuar.")
    st.stop()

# Se superponen las densidades MLE elegidas sobre el histograma observado.
figura, eje = plt.subplots(figsize=(10, 5))
sns.histplot(
    x=serie,
    stat="density",
    bins="auto",
    color="#a6b0ae",
    alpha=0.45,
    ax=eje,
)
eje.plot([], [], color="#a6b0ae", linewidth=8, alpha=0.45, label="Máximos observados")
rejilla = np.linspace(max(0, float(serie.min()) * 0.8), float(serie.max()) * 1.15, 300)
filas_parametros = []
for nombre in modelos_seleccionados:
    parametros, k = ajuste_cacheado(serie, nombre)
    densidad = CANDIDATAS[nombre].pdf(rejilla, *parametros)
    eje.plot(rejilla, densidad, linewidth=2, label=nombre)

    # Se escriben los parámetros junto con k para hacer visible el ajuste de cada curva.
    formas = (
        [parte.strip() for parte in CANDIDATAS[nombre].shapes.split(",")]
        if CANDIDATAS[nombre].shapes
        else []
    )
    nombres_parametros = formas + ["loc", "scale"]
    filas_parametros.append(
        {
            "Modelo": nombre,
            "Parámetros (SciPy)": ", ".join(
                f"{clave}={valor:.4f}"
                for clave, valor in zip(nombres_parametros, parametros)
            ),
            "k libre": k,
        }
    )
eje.set_xlabel("Precipitación máxima en 24 h (mm)")
eje.set_ylabel("Densidad de probabilidad")
eje.set_title("Densidades ajustadas por máxima verosimilitud")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)
st.dataframe(pd.DataFrame(filas_parametros), width="stretch", hide_index=True)
st.info(
    "Gumbel y GEV tienen respaldo teórico para máximos anuales por el teorema de valores "
    "extremos: bajo condiciones de independencia y regularidad, máximos de bloques "
    "normalizados convergen a una familia GEV; Gumbel es el caso límite tipo I. "
    "La teoría orienta la elección, pero no sustituye la revisión de ajuste y supuestos."
)

# C. Se calcula la bondad con Monte Carlo y se trazan Q-Q para todas las candidatas seleccionadas.
st.header("Bondad de ajuste")
n_mc = st.slider(
    "Réplicas Monte Carlo (KS y Anderson-Darling)",
    min_value=99,
    max_value=499,
    value=199,
    step=100,
)
with st.spinner(f"Calculando pruebas con {n_mc} réplicas; el primer cálculo puede tardar..."):
    comparacion_completa = tabla_comparativa_cacheada(serie, n_mc)
tabla_pruebas = comparacion_completa.loc[
    comparacion_completa["Modelo"].isin(modelos_seleccionados),
    [
        "Modelo",
        "KS",
        "p KS (Monte Carlo)",
        "KS ingenuo",
        "p KS ingenuo",
        "AD",
        "p AD (Monte Carlo)",
        "¿Se rechaza H0? (5 %)",
    ],
].reset_index(drop=True)
st.dataframe(
    tabla_pruebas,
    width="stretch",
    hide_index=True,
    column_config={
        "KS": st.column_config.NumberColumn(format="%.4f"),
        "p KS (Monte Carlo)": st.column_config.NumberColumn(format="%.4f"),
        "KS ingenuo": st.column_config.NumberColumn(format="%.4f"),
        "p KS ingenuo": st.column_config.NumberColumn(format="%.4f"),
        "AD": st.column_config.NumberColumn(format="%.4f"),
        "p AD (Monte Carlo)": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption(
    "Los p-valores Monte Carlo vuelven a estimar los parámetros en cada réplica y corrigen "
    "el uso ingenuo de KS tras estimar parámetros (problema de Lilliefors). "
    "La resolución aproximada con n_mc réplicas es 1/(n_mc+1)."
)

# Se crean Q-Q plots en una cuadrícula y se guarda su correlación como apoyo diagnóstico.
cantidad_columnas = 2
cantidad_filas = math.ceil(len(modelos_seleccionados) / cantidad_columnas)
figura, ejes = plt.subplots(
    cantidad_filas,
    cantidad_columnas,
    figsize=(12, 4.5 * cantidad_filas),
    squeeze=False,
)
correlaciones_qq = {}
for eje, nombre in zip(ejes.flat, modelos_seleccionados):
    parametros, _ = ajuste_cacheado(serie, nombre)
    cuantiles_teoricos, cuantiles_observados, correlacion = cuantiles_qq(
        serie, nombre, parametros
    )
    correlaciones_qq[nombre] = correlacion

    # Se dibujan puntos observados y la diagonal ideal de concordancia.
    sns.scatterplot(x=cuantiles_teoricos, y=cuantiles_observados, ax=eje, color="#287c78")
    limites = [
        float(np.nanmin([cuantiles_teoricos.min(), cuantiles_observados.min()])),
        float(np.nanmax([cuantiles_teoricos.max(), cuantiles_observados.max()])),
    ]
    eje.plot(limites, limites, linestyle="--", color="#b7473e", label="Concordancia ideal")
    eje.set_xlabel("Cuantiles teóricos (mm)")
    eje.set_ylabel("Cuantiles observados (mm)")
    eje.set_title(f"Q-Q {nombre} (r = {correlacion:.4f})")
    eje.legend()

# Se ocultan paneles vacíos cuando se selecciona una cantidad impar de modelos.
for eje in list(ejes.flat)[len(modelos_seleccionados):]:
    eje.set_visible(False)
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# D. Se ordenan criterios de información, se representa ΔAIC y se aplica el semáforo.
st.header("Criterios de información")
tabla_criterios = comparacion_completa.loc[
    comparacion_completa["Modelo"].isin(modelos_seleccionados)
].copy()
tabla_criterios["ΔAIC"] = tabla_criterios["AIC"] - tabla_criterios["AIC"].min()
tabla_criterios["Interpretación de ΔAIC"] = tabla_criterios["ΔAIC"].map(
    interpretar_delta_aic
)
tabla_criterios["r Q-Q"] = tabla_criterios["Modelo"].map(correlaciones_qq)
tabla_criterios = tabla_criterios.sort_values("AIC").reset_index(drop=True)
st.dataframe(
    tabla_criterios[
        ["Modelo", "k", "ℓ", "AIC", "BIC", "AICc", "ΔAIC", "Interpretación de ΔAIC", "r Q-Q"]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "ℓ": st.column_config.NumberColumn(format="%.4f"),
        "AIC": st.column_config.NumberColumn(format="%.4f"),
        "BIC": st.column_config.NumberColumn(format="%.4f"),
        "AICc": st.column_config.NumberColumn(format="%.4f"),
        "ΔAIC": st.column_config.NumberColumn(format="%.4f"),
        "r Q-Q": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se comparan diferencias de AIC y se marca la referencia convencional de dos puntos.
figura, eje = plt.subplots(figsize=(9, 4))
sns.barplot(data=tabla_criterios, x="Modelo", y="ΔAIC", color="#e7a34b", ax=eje)
eje.axhline(2, linestyle="--", color="#b7473e", label="ΔAIC = 2")
eje.set_xlabel("Distribución")
eje.set_ylabel("ΔAIC respecto del mínimo")
eje.set_title("Comparación relativa de AIC")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se permite confirmar visualmente qué Q-Q plots se consideran aceptables.
preseleccion_qq = [
    nombre
    for nombre in modelos_seleccionados
    if np.isfinite(correlaciones_qq[nombre]) and correlaciones_qq[nombre] >= 0.90
]
qq_aceptables = st.multiselect(
    "Modelos cuyo Q-Q consideras aceptable tras revisar los gráficos",
    options=modelos_seleccionados,
    default=preseleccion_qq,
)
st.caption(
    "La preselección usa r Q-Q ≥ 0.90 solo como referencia operativa; la aceptabilidad "
    "es visual y no constituye una prueba formal. Puedes cambiar la selección."
)

# Se construye el semáforo con respaldo físico, Q-Q, AD y ΔAIC pequeño (≤ 2).
tabla_criterios["Respaldo físico"] = tabla_criterios["Modelo"].isin(["Gumbel", "GEV"])
tabla_criterios["Q-Q aceptable"] = tabla_criterios["Modelo"].isin(qq_aceptables)
tabla_criterios["AD no rechaza H0"] = tabla_criterios["p AD (Monte Carlo)"] >= 0.05
tabla_criterios["ΔAIC pequeño"] = tabla_criterios["ΔAIC"] <= 2
st.dataframe(
    tabla_criterios[
        [
            "Modelo",
            "Respaldo físico",
            "Q-Q aceptable",
            "AD no rechaza H0",
            "ΔAIC pequeño",
        ]
    ],
    width="stretch",
    hide_index=True,
)

# Entre candidatas que pasan el semáforo se elige primero el menor número de parámetros.
modelos_recomendables = tabla_criterios.loc[
    tabla_criterios[
        ["Respaldo físico", "Q-Q aceptable", "AD no rechaza H0", "ΔAIC pequeño"]
    ].all(axis=1)
]
if modelos_recomendables.empty:
    st.warning(
        "Ninguna candidata seleccionada cumple todos los criterios del semáforo. "
        "Revisa los Q-Q plots, supuestos, datos y selección de modelos; no fuerces una elección."
    )
else:
    menor_k = modelos_recomendables["k"].min()
    modelos_equivalentes = modelos_recomendables.loc[modelos_recomendables["k"] == menor_k]
    recomendado = modelos_equivalentes.sort_values("AIC").iloc[0]
    st.success(
        f"Modelo recomendado: {recomendado['Modelo']}. Cumple respaldo físico, Q-Q "
        "aceptable, no rechazo de Anderson-Darling y ΔAIC ≤ 2; entre modelos equivalentes "
        f"se aplicó la preferencia por el menor k (k={int(recomendado['k'])}) entre las "
        "candidatas que superaron el semáforo."
    )

# E. Se presenta la tabla de cuantiles de retorno y el papel de Gumbel para diseño.
st.header("Período de retorno y diseño")
filas_cuantiles = []
for periodo in PERIODOS_RETORNO:
    fila = {"T (años)": periodo}
    for nombre in modelos_seleccionados:
        parametros, _ = ajuste_cacheado(serie, nombre)
        fila[nombre] = cuantil_retorno(nombre, parametros, periodo)
    filas_cuantiles.append(fila)
tabla_cuantiles = pd.DataFrame(filas_cuantiles)
st.subheader("Cuantiles de precipitación de diseño")
st.dataframe(
    tabla_cuantiles,
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.2f mm")
        for nombre in modelos_seleccionados
    },
)

# Se transforma T a la variable reducida y se trazan datos Gringorten y modelos ajustados.
posiciones = posiciones_graficas(serie)
probabilidades_gumbel = 1 - 1 / np.geomspace(2, 100, 250)
variable_reducida_curva = -np.log(-np.log(probabilidades_gumbel))
variable_reducida_datos = -np.log(-np.log(posiciones["p Gringorten"].to_numpy()))
figura, eje = plt.subplots(figsize=(10, 6))
eje.scatter(
    variable_reducida_datos,
    posiciones["pmax24h_mm"],
    color="#202f3c",
    label="Datos (Gringorten)",
    zorder=3,
)
for nombre in modelos_seleccionados:
    parametros, _ = ajuste_cacheado(serie, nombre)
    cuantiles_modelo = CANDIDATAS[nombre].ppf(probabilidades_gumbel, *parametros)
    eje.plot(variable_reducida_curva, cuantiles_modelo, linewidth=2, label=nombre)

# Las marcas del eje x muestran los periodos de retorno que generan la variable reducida.
marcas_periodo = np.asarray(PERIODOS_RETORNO, dtype=float)
marcas_reducidas = -np.log(-np.log(1 - 1 / marcas_periodo))
eje.set_xticks(marcas_reducidas, labels=[str(int(valor)) for valor in marcas_periodo])
eje.set_xlim(variable_reducida_datos.min(), marcas_reducidas.max())
eje.set_xlabel("Variable reducida -ln(-ln(1 - 1/T)); marcas = T (años)")
eje.set_ylabel("Precipitación máxima en 24 h (mm)")
eje.set_title("Papel de Gumbel y curvas de las distribuciones ajustadas")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se permite elegir un periodo/modelo y calcular su intervalo bootstrap al solicitarlo.
columna_modelo_ic, columna_periodo_ic = st.columns(2)
modelo_recomendado = (
    recomendado["Modelo"] if not modelos_recomendables.empty else modelos_seleccionados[0]
)
with columna_modelo_ic:
    modelo_ic = st.selectbox(
        "Modelo para el intervalo bootstrap",
        options=modelos_seleccionados,
        index=modelos_seleccionados.index(modelo_recomendado),
    )
with columna_periodo_ic:
    periodo_ic = st.selectbox(
        "Periodo de retorno para el intervalo (años)",
        options=PERIODOS_RETORNO,
        index=PERIODOS_RETORNO.index(10),
    )
clave_ic = (modelo_ic, periodo_ic, 500, 42)
if st.button("Calcular intervalo bootstrap del 95 %", key="calcular_bootstrap"):
    with st.spinner("Calculando bootstrap paramétrico (500 réplicas)..."):
        intervalo = bootstrap_cacheado(serie, modelo_ic, float(periodo_ic), semilla=42)
    st.session_state["intervalo_bootstrap"] = (clave_ic, intervalo)

# Se conserva visible el intervalo ya calculado mientras no cambie modelo ni periodo.
resultado_ic = st.session_state.get("intervalo_bootstrap")
if resultado_ic is not None and resultado_ic[0] == clave_ic:
    limite_ic_inferior, limite_ic_superior = resultado_ic[1]
    st.metric(
        f"IC bootstrap 95 % de x_T ({modelo_ic}, T={periodo_ic} años)",
        f"[{limite_ic_inferior:.2f}, {limite_ic_superior:.2f}] mm",
    )
else:
    st.caption("Pulsa el botón para calcular el intervalo del cuantil seleccionado.")

# Se calcula el riesgo acumulado de excedencia para la vida útil de una obra.
st.subheader("Calculadora de riesgo de falla")
columna_t_riesgo, columna_n_riesgo, columna_resultado_riesgo = st.columns(3)
periodo_riesgo = columna_t_riesgo.number_input(
    "Periodo de retorno T (años)", min_value=2, max_value=1000, value=50, step=1
)
vida_util = columna_n_riesgo.number_input(
    "Vida útil N (años)", min_value=1, max_value=100, value=25, step=1
)
riesgo = riesgo_falla(periodo_riesgo, vida_util)
columna_resultado_riesgo.metric("Riesgo de al menos una excedencia", f"{riesgo:.2%}")
st.caption(
    "Interpretación: probabilidad de que el nivel asociado a T sea excedido al menos "
    "una vez durante N años, suponiendo estacionariedad e independencia anual."
)

# F. Se verifica a mano la muestra de diez años con posiciones, Gumbel y riesgo.
st.header("Verificación del cálculo a mano (10 años)")
posiciones_muestra = posiciones_graficas(muestra["pmax24h_mm"])
muestra_ordenada = muestra.sort_values("pmax24h_mm").reset_index(drop=True)
tabla_posiciones = pd.DataFrame(
    {
        "anio": muestra_ordenada["anio"],
        "pmax24h_mm": posiciones_muestra["pmax24h_mm"],
        "i": posiciones_muestra["i"],
        "Posición Weibull": posiciones_muestra["p Weibull"],
        "T empírico (años)": posiciones_muestra["T Weibull (años)"],
    }
)
st.dataframe(
    tabla_posiciones,
    width="stretch",
    hide_index=True,
    column_config={
        "pmax24h_mm": st.column_config.NumberColumn(format="%.4f"),
        "Posición Weibull": st.column_config.NumberColumn(format="%.4f"),
        "T empírico (años)": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se calculan media y desviación estándar muestral de los diez máximos.
media_muestra = float(muestra["pmax24h_mm"].mean())
desviacion_muestra = float(muestra["pmax24h_mm"].std(ddof=1))
columna_media, columna_desviacion = st.columns(2)
columna_media.metric("Media (mm)", f"{media_muestra:.4f}")
columna_desviacion.metric("Desviación estándar muestral (mm)", f"{desviacion_muestra:.4f}")

# Se comparan los parámetros Gumbel por momentos con los de máxima verosimilitud.
parametros_momentos = gumbel_momentos(muestra["pmax24h_mm"])
parametros_mle, _ = ajustar(muestra["pmax24h_mm"], "Gumbel")
mu_mle, beta_mle = float(parametros_mle[-2]), float(parametros_mle[-1])
st.dataframe(
    pd.DataFrame(
        [
            {"Método": "Momentos", "μ (mm)": parametros_momentos["mu"], "β (mm)": parametros_momentos["beta"]},
            {"Método": "Máxima verosimilitud", "μ (mm)": mu_mle, "β (mm)": beta_mle},
        ]
    ),
    width="stretch",
    hide_index=True,
    column_config={
        "μ (mm)": st.column_config.NumberColumn(format="%.4f"),
        "β (mm)": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se calculan niveles de retorno de 10, 50 y 100 años con ambos métodos.
parametros_momentos_tupla = (parametros_momentos["mu"], parametros_momentos["beta"])
filas_niveles_muestra = []
for periodo in [10, 50, 100]:
    filas_niveles_muestra.append(
        {
            "T (años)": periodo,
            "x_T momentos (mm)": cuantil_retorno(
                "Gumbel", parametros_momentos_tupla, periodo
            ),
            "x_T máxima verosimilitud (mm)": cuantil_retorno(
                "Gumbel", parametros_mle, periodo
            ),
        }
    )
st.dataframe(
    pd.DataFrame(filas_niveles_muestra),
    width="stretch",
    hide_index=True,
    column_config={
        "x_T momentos (mm)": st.column_config.NumberColumn(format="%.4f"),
        "x_T máxima verosimilitud (mm)": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se resuelve el ejercicio de riesgo para T=50 y la vida útil N=25 años.
riesgo_ejercicio = riesgo_falla(50, 25)
st.write(
    f"Riesgo para T=50 años y N=25 años: **{riesgo_ejercicio:.4f}** "
    f"({riesgo_ejercicio:.2%})."
)

# G. Se comparan modelos semiempíricos con los datos de secado en capa delgada.
st.header("Ingeniería: modelos de secado (AICc)")
datos_secado = cargar_csv("secado.csv")
tiempo_secado = datos_secado["tiempo_min"]
mr_observada = datos_secado["MR"]

# Se cachea la tabla para no repetir los cinco ajustes en cada interacción.
@st.cache_data(show_spinner=False)
def tabla_secado_cacheada(tiempo: pd.Series, humedad: pd.Series) -> pd.DataFrame:
    """Reutiliza la comparación de modelos mientras los datos no cambien."""
    return tabla_modelos(tiempo, humedad)


# Se cachean también los parámetros usados para dibujar las curvas individuales.
@st.cache_data(show_spinner=False)
def ajuste_secado_cacheado(
    tiempo: pd.Series,
    humedad: pd.Series,
    nombre: str,
) -> np.ndarray:
    """Reutiliza los parámetros curve_fit de una curva de secado."""
    return ajustar_curva(tiempo, humedad, nombre)


with st.spinner("Ajustando modelos semiempíricos de secado..."):
    tabla_secado = tabla_secado_cacheada(tiempo_secado, mr_observada)

# Se dibujan puntos observados y las curvas ajustadas de todos los modelos.
figura, eje = plt.subplots(figsize=(10, 5))
sns.scatterplot(
    x=tiempo_secado,
    y=mr_observada,
    ax=eje,
    color="#202f3c",
    label="MR observada",
    zorder=3,
)
tiempo_suave = np.linspace(float(tiempo_secado.min()), float(tiempo_secado.max()), 400)
for nombre in MODELOS:
    parametros_secado = ajuste_secado_cacheado(tiempo_secado, mr_observada, nombre)
    funcion_modelo = MODELOS[nombre]["funcion"]
    humedad_ajustada = funcion_modelo(tiempo_suave, *parametros_secado)
    eje.plot(tiempo_suave, humedad_ajustada, linewidth=2, label=nombre)
eje.set_xlabel("Tiempo de secado (min)")
eje.set_ylabel("Razón de humedad MR (adimensional)")
eje.set_title("Curvas semiempíricas de secado en capa delgada")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se calcula ΔAICc respecto del menor AICc y se ordenan los modelos.
tabla_secado["ΔAICc"] = tabla_secado["AICc"] - tabla_secado["AICc"].min()
tabla_secado = tabla_secado.sort_values("AICc").reset_index(drop=True)
st.dataframe(
    tabla_secado[
        ["Modelo", "Parámetros", "k", "R²", "RMSE", "χ² reducido", "AIC", "AICc", "ΔAICc"]
    ],
    width="stretch",
    hide_index=True,
    column_config={
        "R²": st.column_config.NumberColumn(format="%.4f"),
        "RMSE": st.column_config.NumberColumn(format="%.4f"),
        "χ² reducido": st.column_config.NumberColumn(format="%.4f"),
        "AIC": st.column_config.NumberColumn(format="%.4f"),
        "AICc": st.column_config.NumberColumn(format="%.4f"),
        "ΔAICc": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se visualizan las diferencias de AICc para comparar el soporte relativo.
figura, eje = plt.subplots(figsize=(9, 4))
sns.barplot(data=tabla_secado, x="Modelo", y="ΔAICc", color="#e7a34b", ax=eje)
eje.set_xlabel("Modelo de secado")
eje.set_ylabel("ΔAICc respecto del mínimo")
eje.set_title("Comparación de modelos semiempíricos de secado")
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se grafica el residuo del modelo con el menor AICc en función del tiempo.
mejor_modelo_secado = tabla_secado.iloc[0]["Modelo"]
mejores_parametros_secado = ajuste_secado_cacheado(
    tiempo_secado,
    mr_observada,
    mejor_modelo_secado,
)
mejor_funcion_secado = MODELOS[mejor_modelo_secado]["funcion"]
mr_mejor_ajustada = mejor_funcion_secado(
    tiempo_secado.to_numpy(), *mejores_parametros_secado
)
residuos_secado = mr_observada.to_numpy() - mr_mejor_ajustada
figura, eje = plt.subplots(figsize=(9, 4))
sns.scatterplot(x=tiempo_secado, y=residuos_secado, ax=eje, color="#287c78")
eje.axhline(0, color="#b7473e", linestyle="--")
eje.set_xlabel("Tiempo de secado (min)")
eje.set_ylabel("Residuo MR observado - ajustado (adimensional)")
eje.set_title(f"Residuos del mejor modelo AICc: {mejor_modelo_secado}")
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se explica la penalización AICc y el chi-cuadrado reducido calculado sin varianzas.
st.info(
    "El AICc combina ajuste y complejidad: añadir parámetros puede reducir los residuos, "
    "pero cada parámetro recibe una penalización adicional, especialmente con muestras "
    "pequeñas. Es una navaja de Ockham estadística: no compraríamos una secadora más "
    "compleja si la sencilla seca igual. Aquí χ² reducido se calcula como SCE/(n-k), "
    "porque no se proporcionaron varianzas de medición por observación; se interpreta como "
    "diagnóstico de error residual y no como prueba formal de chi-cuadrado."
)