"""Exploración, validación y pronóstico de series climáticas mensuales."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.datos import cargar_csv, cargar_metadatos, tarjeta_estudiante
from src.series import (
    cargar_serie,
    descomponer,
    ejecutar_autots,
    indices_estacionales,
    metodos_referencia,
    metricas,
    promedio_movil,
)
from src.exportar import descargar_excel, descargar_figuras, descargar_reporte_markdown
from src.validacion import (
    advertir_datos_faltantes,
    cargar_datos_seguros,
    mostrar_error_ajuste,
    validar_archivo_vacio,
    validar_columnas_seleccionadas,
    validar_minimo_observaciones,
    validar_respuesta_numerica,
    validar_serie_mensual,
)


# Se definen nombres de meses en español para tablas y diagramas de caja.
MESES_ES = [
    "Enero",
    "Febrero",
    "Marzo",
    "Abril",
    "Mayo",
    "Junio",
    "Julio",
    "Agosto",
    "Septiembre",
    "Octubre",
    "Noviembre",
    "Diciembre",
]
NOMBRES_VARIABLES = {
    "precipitacion_mm": "Precipitación (mm)",
    "temperatura_c": "Temperatura (°C)",
}
CARPETA_RESULTADOS = Path(__file__).resolve().parents[1] / "resultados"


# AutoTS se conserva como recurso entre reruns y se vuelve a ajustar solo al cambiar parámetros.
@st.cache_resource(show_spinner=False)
def ejecutar_autots_cacheado(
    horizonte: int,
    model_list: str,
    generaciones: int,
    validaciones: int,
    intervalo: float,
    semilla: int,
) -> dict:
    """Ejecuta AutoTS para precipitación y reutiliza el recurso asociado a esta configuración."""
    # El ensayo genético siempre compara pronósticos de precipitación mensual en milímetros.
    serie_precipitacion = cargar_serie("precipitacion_mm")

    # Los argumentos identifican la búsqueda y forman su clave de caché reproducible.
    return ejecutar_autots(
        serie_precipitacion,
        horizonte=horizonte,
        model_list=model_list,
        generaciones=generaciones,
        validaciones=validaciones,
        intervalo=intervalo,
        semilla=semilla,
    )


# Se mantiene la ficha del estudiante y se cargan la variable seleccionada y la muestra de 24 meses.
tarjeta_estudiante()
st.title("Series de tiempo")
variable = st.selectbox(
    "Variable climática",
    options=["precipitacion_mm", "temperatura_c"],
    format_func=lambda valor: NOMBRES_VARIABLES[valor],
)
serie = cargar_datos_seguros(cargar_serie, variable, "serie climática")
if serie is None:
    st.stop()
etiqueta_y = NOMBRES_VARIABLES[variable]
muestra24 = cargar_datos_seguros(
    cargar_csv,
    "muestra_series24.csv",
    "verificación manual de series",
)
if muestra24 is None:
    st.stop()
if not validar_archivo_vacio(serie.to_frame(), NOMBRES_VARIABLES[variable]):
    st.stop()
if not validar_respuesta_numerica(serie.to_frame(), variable, "serie de tiempo"):
    st.stop()
if not validar_serie_mensual(serie, minimo=24, contexto=NOMBRES_VARIABLES[variable]):
    st.stop()
if not advertir_datos_faltantes(serie.to_frame(), contexto="serie climática mensual"):
    st.info("La serie contiene meses sin observación; se detiene el análisis para evitar imputaciones automáticas.")
    st.stop()
if not validar_archivo_vacio(muestra24, "muestra_series24.csv"):
    st.stop()
if not validar_columnas_seleccionadas(
    muestra24,
    ["fecha", "precipitacion_mm", "conjunto", "periodo", "anio", "mes"],
    "verificación temporal",
):
    st.stop()
if not validar_respuesta_numerica(muestra24, "precipitacion_mm", "validación temporal"):
    st.stop()
if not validar_minimo_observaciones(muestra24["precipitacion_mm"], 24, "muestra de series"):
    st.stop()
if not advertir_datos_faltantes(muestra24, contexto="muestra temporal de 24 meses"):
    st.info("Completa o documenta los faltantes de la muestra antes de comparar pronósticos.")
    st.stop()
if (
    (muestra24["conjunto"] == "entrenamiento").sum() != 18
    or (muestra24["conjunto"] == "validacion").sum() != 6
):
    st.error(
        "La muestra debe tener 18 meses de entrenamiento y 6 de validación. "
        "Revisa la columna conjunto antes de calcular el concurso manual."
    )
    st.stop()

# 1. Se explora la serie con promedios móviles y distribuciones por mes calendario.
st.header("Exploración")
figura, eje = plt.subplots(figsize=(11, 5))
eje.plot(serie.index, serie, color="#287c78", alpha=0.55, label="Observada")
eje.plot(
    promedio_movil(serie, 3).index,
    promedio_movil(serie, 3),
    color="#e7a34b",
    linewidth=1.8,
    label="Promedio móvil 3 meses",
)
eje.plot(
    promedio_movil(serie, 12).index,
    promedio_movil(serie, 12),
    color="#b7473e",
    linewidth=2.2,
    label="Promedio móvil 12 meses",
)
eje.set_xlabel("Fecha")
eje.set_ylabel(etiqueta_y)
eje.set_title(f"Serie mensual: {etiqueta_y}")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se preparan etiquetas categóricas para que el diagrama respete enero-diciembre en español.
datos_mensuales = serie.rename("valor").to_frame()
datos_mensuales["Mes"] = pd.Categorical(
    [MESES_ES[fecha.month - 1] for fecha in datos_mensuales.index],
    categories=MESES_ES,
    ordered=True,
)
figura, eje = plt.subplots(figsize=(11, 5))
sns.boxplot(data=datos_mensuales, x="Mes", y="valor", order=MESES_ES, color="#e7a34b", ax=eje)
eje.set_xlabel("Mes")
eje.set_ylabel(etiqueta_y)
eje.set_title(f"Variación entre años por mes: {etiqueta_y}")
eje.tick_params(axis="x", rotation=35)
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se explican tendencia, estacionalidad e irregularidad con el régimen bimodal caribeño.
st.markdown(
    """
- **Tendencia:** cambios de largo plazo; el promedio móvil de 12 meses ayuda a separarlos del ciclo anual.
- **Estacionalidad:** patrón que se repite por mes. En la precipitación del Caribe colombiano suele haber dos temporadas lluviosas, abril-mayo y septiembre-noviembre, separadas por periodos relativamente secos.
- **Irregularidad:** variación restante por eventos y condiciones meteorológicas que no se repiten con calendario fijo.

La temperatura también puede presentar un ciclo anual, pero no debe interpretarse con el régimen bimodal de lluvia.
"""
)

# 2. Se descompone aditivamente la serie y se interpretan los doce índices mensuales.
st.header("Descomposición clásica")
try:
    descomposicion = descomponer(serie)
except Exception as error:
    mostrar_error_ajuste(error, "la descomposición clásica de la serie")
    st.stop()
componentes = [
    ("Observada", descomposicion.observed),
    ("Tendencia", descomposicion.trend),
    ("Estacional", descomposicion.seasonal),
    ("Residuo", descomposicion.resid),
]
figura, ejes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
for eje, (nombre, valores) in zip(ejes, componentes):
    eje.plot(valores.index, valores, color="#287c78" if nombre != "Residuo" else "#b7473e")
    eje.set_ylabel(etiqueta_y if nombre != "Residuo" else "Residuo")
    eje.set_title(nombre, loc="left", fontsize=10)
ejes[-1].set_xlabel("Fecha")
figura.suptitle("Descomposición aditiva (periodo = 12 meses)")
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se traducen los índices aditivos a una interpretación sobre el promedio general.
indices = indices_estacionales(serie)
tabla_indices = pd.DataFrame(
    {
        "Mes": MESES_ES,
        "Índice estacional": indices.to_numpy(),
        "Interpretación": [
            "Por encima del promedio" if valor > 0 else "Por debajo del promedio" if valor < 0 else "Igual al promedio"
            for valor in indices.to_numpy()
        ],
    }
)
st.dataframe(
    tabla_indices,
    width="stretch",
    hide_index=True,
    column_config={
        "Índice estacional": st.column_config.NumberColumn(format="%.4f"),
    },
)
st.caption(f"Suma de los doce índices aditivos: {indices.sum():.10f} (≈ 0).")

# 3. Se valida en los últimos doce meses sin mezclar datos futuros en el entrenamiento.
st.header("Métodos de referencia y validación")
entrenamiento = serie.iloc[:-12]
validacion = serie.iloc[-12:]
try:
    pronosticos_validacion = metodos_referencia(entrenamiento, h=len(validacion))
except Exception as error:
    mostrar_error_ajuste(error, "los métodos de referencia de la serie")
    st.stop()
pronosticos_validacion.index = validacion.index

# Se grafican observaciones y métodos de referencia exclusivamente en el periodo de validación.
figura, eje = plt.subplots(figsize=(11, 5))
eje.plot(serie.index, serie, color="#202f3c", label="Observada")
colores_metodos = {
    "ingenuo": "#b7473e",
    "estacional_ingenuo": "#287c78",
    "promedio": "#e7a34b",
}
for metodo in pronosticos_validacion.columns:
    eje.plot(
        pronosticos_validacion.index,
        pronosticos_validacion[metodo],
        linestyle="--",
        marker="o",
        color=colores_metodos[metodo],
        label=metodo.replace("_", " ").capitalize(),
    )
eje.axvline(validacion.index[0], color="#777777", linestyle=":", label="Inicio validación")
eje.set_xlabel("Fecha")
eje.set_ylabel(etiqueta_y)
eje.set_title("Validación temporal: últimos 12 meses")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se calculan métricas independientes para cada pronóstico y se ordenan por SMAPE ascendente.
filas_metricas = [
    {"Método": metodo, **metricas(validacion, pronosticos_validacion[metodo])}
    for metodo in pronosticos_validacion.columns
]
tabla_metricas = pd.DataFrame(filas_metricas).sort_values("SMAPE").reset_index(drop=True)
st.dataframe(
    tabla_metricas,
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.4f")
        for nombre in ["MAE", "RMSE", "SMAPE"]
    },
)

# 4. Se configura AutoTS en un formulario para que la búsqueda ocurra solo bajo solicitud.
st.header("Pronóstico automático con AutoTS")
st.caption("AutoTS se ejecuta sobre precipitación mensual (mm), con búsqueda y validación temporal.")
metadatos = cargar_metadatos()
semilla_autots = int(metadatos["codigo"]) % 10000
with st.form("formulario_autots"):
    columna_horizonte, columna_modelos = st.columns(2)
    with columna_horizonte:
        horizonte_autots = st.slider("Horizonte (meses)", 6, 24, 12)
    with columna_modelos:
        lista_modelos_autots = st.selectbox("Catálogo de modelos", ["superfast", "fast"])
    columna_generaciones, columna_validaciones, columna_intervalo = st.columns(3)
    with columna_generaciones:
        generaciones_autots = st.slider("Generaciones", 1, 10, 5)
    with columna_validaciones:
        validaciones_autots = st.slider("Validaciones temporales", 1, 3, 2)
    with columna_intervalo:
        intervalo_autots = st.selectbox("Intervalo predictivo", [0.8, 0.9, 0.95], index=1)
    ejecutar_formulario = st.form_submit_button("Ejecutar AutoTS")

# Se ejecuta o recupera el recurso AutoTS únicamente al enviar el formulario.
if ejecutar_formulario:
    try:
        with st.spinner("AutoTS está evaluando candidatos y generaciones; esta búsqueda puede tardar varios minutos..."):
            resultado_autots = ejecutar_autots_cacheado(
                horizonte_autots,
                lista_modelos_autots,
                generaciones_autots,
                validaciones_autots,
                intervalo_autots,
                semilla_autots,
            )
        st.session_state["resultado_autots"] = resultado_autots
        st.session_state["parametros_autots"] = {
            "horizonte": horizonte_autots,
            "model_list": lista_modelos_autots,
            "generaciones": generaciones_autots,
            "validaciones": validaciones_autots,
            "intervalo": intervalo_autots,
            "semilla": semilla_autots,
        }
    except Exception as error:
        mostrar_error_ajuste(
            error,
            "la búsqueda automática de AutoTS",
            "Confirma que haya al menos 24 meses continuos, revisa las opciones de búsqueda "
            "y vuelve a intentarlo. Algunos candidatos pueden fallar y ser descartados por AutoTS.",
        )

# Se muestran resultados de sesión si ya se ejecutó el formulario.
if "resultado_autots" in st.session_state:
    resultado_autots = st.session_state["resultado_autots"]
    st.subheader(f"Modelo ganador: {resultado_autots['mejor_modelo']['nombre']}")
    st.json(
        json.loads(json.dumps(resultado_autots["mejor_modelo"], ensure_ascii=False, default=str))
    )

    # Se dibuja la precipitación histórica con el pronóstico y su intervalo predictivo.
    serie_precipitacion = cargar_serie("precipitacion_mm")
    tabla_autots = resultado_autots["pronostico"]
    figura, eje = plt.subplots(figsize=(11, 5))
    eje.plot(serie_precipitacion.index, serie_precipitacion, color="#202f3c", label="Precipitación observada")
    eje.plot(
        tabla_autots["fecha"],
        tabla_autots["pronostico_mm"],
        color="#287c78",
        marker="o",
        label="Pronóstico AutoTS",
    )
    eje.fill_between(
        tabla_autots["fecha"],
        tabla_autots["limite_inferior_mm"],
        tabla_autots["limite_superior_mm"],
        color="#287c78",
        alpha=0.2,
        label=f"Intervalo {st.session_state['parametros_autots']['intervalo']:.0%}",
    )
    eje.set_xlabel("Fecha")
    eje.set_ylabel("Precipitación (mm)")
    eje.set_title("Pronóstico automático de precipitación")
    eje.legend()
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

    # Se muestran el pronóstico tabular y los diez candidatos mejor evaluados.
    st.dataframe(tabla_autots, width="stretch", hide_index=True)
    st.download_button(
        "Descargar pronóstico AutoTS (CSV)",
        data=tabla_autots.to_csv(index=False).encode("utf-8-sig"),
        file_name="autots_pronostico.csv",
        mime="text/csv",
    )
    st.subheader("Top 10 modelos de validación")
    st.dataframe(resultado_autots["top10"], width="stretch", hide_index=True)

# Se explica la búsqueda genética como evaluación repetida de variedades bajo los mismos datos.
st.info(
    "Analogía: AutoTS funciona como un ensayo de variedades. Prueba candidatos en parcelas "
    "temporales de entrenamiento y validación, compara sus métricas y conserva configuraciones "
    "competitivas para nuevas generaciones, como una selección genética. El modelo ganador es "
    "el mejor de los candidatos y condiciones probados, no una garantía de pronóstico perfecto."
)

# 5. Se inspeccionan archivos ya guardados sin llamar a AutoTS ni recalcular el pronóstico.
st.header("Resultados guardados")
archivos_resultados = {
    "autots_pronostico.csv": CARPETA_RESULTADOS / "autots_pronostico.csv",
    "autots_top10_modelos.csv": CARPETA_RESULTADOS / "autots_top10_modelos.csv",
    "autots_mejor_modelo.json": CARPETA_RESULTADOS / "autots_mejor_modelo.json",
}
existentes = {nombre: ruta for nombre, ruta in archivos_resultados.items() if ruta.is_file()}
if not existentes:
    st.info("Todavía no hay resultados guardados en la carpeta resultados/.")
else:
    for nombre, ruta in existentes.items():
        st.subheader(nombre)
        if ruta.suffix == ".csv":
            st.dataframe(pd.read_csv(ruta), width="stretch", hide_index=True)
        else:
            with ruta.open("r", encoding="utf-8") as archivo_json:
                st.json(json.load(archivo_json))

# 6. Se verifica a mano la muestra de 24 meses con PM3, índices, errores y métodos de referencia.
st.header("Verificación del cálculo a mano (24 meses)")
muestra_temporal = muestra24.copy()
muestra_serie = muestra_temporal.set_index("fecha")["precipitacion_mm"]
muestra_temporal["PM3"] = promedio_movil(muestra_serie, 3).to_numpy()
st.subheader("Muestra con promedio móvil de tres meses")
st.dataframe(
    muestra_temporal,
    width="stretch",
    hide_index=True,
    column_config={
        "precipitacion_mm": st.column_config.NumberColumn(format="%.4f"),
        "PM3": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se muestran índices estacionales y su suma para verificar la condición aditiva.
indices_muestra = indices_estacionales(muestra_serie)
tabla_indices_muestra = pd.DataFrame(
    {
        "Mes": MESES_ES,
        "Índice estacional (mm)": indices_muestra.to_numpy(),
    }
)
st.subheader("Índices estacionales aditivos")
st.dataframe(
    tabla_indices_muestra,
    width="stretch",
    hide_index=True,
    column_config={"Índice estacional (mm)": st.column_config.NumberColumn(format="%.4f")},
)
st.caption(f"Suma de los doce índices: {indices_muestra.sum():.10f} mm (≈ 0).")

# Se separan los 18 meses de entrenamiento y los seis meses de validación con sus pronósticos.
entrenamiento_muestra = muestra_temporal.loc[
    muestra_temporal["conjunto"] == "entrenamiento"
].set_index("fecha")["precipitacion_mm"]
validacion_muestra = muestra_temporal.loc[
    muestra_temporal["conjunto"] == "validacion"
].set_index("fecha")["precipitacion_mm"]
try:
    pronosticos_muestra = metodos_referencia(entrenamiento_muestra, h=len(validacion_muestra))
except Exception as error:
    mostrar_error_ajuste(error, "los pronósticos de verificación para la muestra 18/6")
    st.stop()
tabla_validacion = pd.DataFrame(
    {
        "fecha": validacion_muestra.index,
        "Real (mm)": validacion_muestra.to_numpy(),
    }
)
for metodo in pronosticos_muestra.columns:
    tabla_validacion[metodo] = pronosticos_muestra[metodo].to_numpy()
st.subheader("Pronósticos de julio a diciembre de 2025")
st.dataframe(
    tabla_validacion,
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.4f")
        for nombre in tabla_validacion.columns
        if nombre != "fecha"
    },
)

# Se despliegan por mes el error absoluto, el error cuadrático y el cociente de SMAPE.
tabla_errores = pd.DataFrame({"fecha": validacion_muestra.index, "Real (mm)": validacion_muestra.to_numpy()})
for metodo in pronosticos_muestra.columns:
    pronosticado = pronosticos_muestra[metodo].to_numpy()
    real = validacion_muestra.to_numpy()
    error = real - pronosticado
    denominador = np.abs(real) + np.abs(pronosticado)
    cociente_smape = np.divide(
        np.abs(error),
        denominador,
        out=np.zeros_like(denominador, dtype=float),
        where=denominador != 0,
    )
    tabla_errores[f"|e| {metodo}"] = np.abs(error)
    tabla_errores[f"e² {metodo}"] = error**2
    tabla_errores[f"|e|/(|y|+|ŷ|) {metodo}"] = cociente_smape
st.subheader("Errores mensuales por método")
st.dataframe(
    tabla_errores,
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.4f")
        for nombre in tabla_errores.columns
        if nombre != "fecha"
    },
)

# Se comparan MAE, RMSE y SMAPE, y se elige el menor SMAPE como ganador del concurso.
metricas_muestra = pd.DataFrame(
    [
        {"Método": metodo, **metricas(validacion_muestra, pronosticos_muestra[metodo])}
        for metodo in pronosticos_muestra.columns
    ]
).sort_values("SMAPE").reset_index(drop=True)
st.subheader("Métricas de validación (6 meses)")
st.dataframe(
    metricas_muestra,
    width="stretch",
    hide_index=True,
    column_config={nombre: st.column_config.NumberColumn(format="%.4f") for nombre in ["MAE", "RMSE", "SMAPE"]},
)
st.success(f"Método ganador por menor SMAPE: {metricas_muestra.iloc[0]['Método']}.")

# Se dibuja la partición de 18/6 meses y se marca con una línea vertical su frontera temporal.
figura, eje = plt.subplots(figsize=(11, 5))
eje.plot(muestra_serie.index, muestra_serie, color="#202f3c", marker="o", label="Real")
for metodo in pronosticos_muestra.columns:
    eje.plot(
        pronosticos_muestra.index,
        pronosticos_muestra[metodo],
        linestyle="--",
        marker="o",
        label=metodo.replace("_", " ").capitalize(),
    )
eje.axvline(validacion_muestra.index[0], color="#b7473e", linestyle=":", label="Inicio validación")
eje.set_xlabel("Fecha")
eje.set_ylabel("Precipitación (mm)")
eje.set_title("Verificación de la partición temporal 18/6")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se reúnen las salidas principales para exportarlas sin repetir análisis ni búsquedas.
tablas_exportacion = {
    "PM3 y muestra": muestra_temporal,
    "Índices completos": tabla_indices,
    "Métricas referencia": tabla_metricas,
    "Índices muestra 24": tabla_indices_muestra,
    "Pronóstico muestra": tabla_validacion,
    "Errores por mes": tabla_errores,
    "Métricas muestra": metricas_muestra,
}
if "resultado_autots" in st.session_state:
    tablas_exportacion["Pronóstico AutoTS"] = st.session_state["resultado_autots"]["pronostico"]
    tablas_exportacion["Top 10 AutoTS"] = st.session_state["resultado_autots"]["top10"]

# Se ofrecen Excel, la figura de validación y un resumen descargable.
with st.expander("Descargar resultados"):
    descargar_excel(
        tablas_exportacion,
        "series_resultados",
        key="descargar_series_excel",
    )
    descargar_figuras(figura, "series_validacion_referencias", key="descargar_series_figura")
    mejor_referencia = metricas_muestra.iloc[0]
    reporte_series = (
        f"Variable explorada: {NOMBRES_VARIABLES[variable]}.\n\n"
        f"Observaciones mensuales: {len(serie)}; suma de índices estacionales: "
        f"{indices.sum():.8f}.\n\n"
        f"Mejor método de referencia en la muestra: {mejor_referencia['Método']} "
        f"(MAE={mejor_referencia['MAE']:.4f}, RMSE={mejor_referencia['RMSE']:.4f}, "
        f"SMAPE={mejor_referencia['SMAPE']:.4f})."
    )
    descargar_reporte_markdown(
        "Resultados de series de tiempo",
        reporte_series,
        "series_reporte.md",
        key="descargar_series_reporte",
    )