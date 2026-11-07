"""Exploración univariada, cualitativa y bivariada de parcelas agrícolas."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.datos import cargar_csv, tarjeta_estudiante
from src.eda import resumen_numerico, tabla_frecuencias, vallas_tukey
from src.exportar import descargar_excel, descargar_figuras, descargar_reporte_markdown
from src.validacion import (
    advertir_datos_faltantes,
    cargar_datos_seguros,
    mostrar_error_ajuste,
    validar_archivo_vacio,
    validar_columnas_seleccionadas,
    validar_minimo_observaciones,
    validar_respuesta_numerica,
)


# Se definen las unidades conocidas para etiquetar correctamente los ejes.
ETIQUETAS_NUMERICAS = {
    "dosis_N": "Dosis de nitrógeno (kg/ha)",
    "lamina_riego": "Lámina de riego (mm)",
    "materia_organica": "Materia orgánica (%)",
    "ph": "pH (sin unidad)",
    "rendimiento": "Rendimiento (t/ha)",
}


# Se mantiene la ficha del estudiante y se cargan el conjunto completo y la muestra manual.
tarjeta_estudiante()
parcelas = cargar_datos_seguros(cargar_csv, "parcelas.csv", "EDA")
muestra_eda = cargar_datos_seguros(cargar_csv, "muestra_eda10.csv", "verificación manual de EDA")
if parcelas is None or muestra_eda is None:
    st.stop()
if not validar_columnas_seleccionadas(
    parcelas,
    ["parcela", "suelo", "riego", "rendimiento"],
    "análisis exploratorio",
):
    st.stop()
if not validar_columnas_seleccionadas(
    muestra_eda,
    ["parcela", "suelo", "riego", "rendimiento"],
    "verificación manual de EDA",
):
    st.stop()
advertir_datos_faltantes(parcelas, contexto="EDA de parcelas")

# Se ofrecen filtros categóricos comunes para los análisis del conjunto completo.
st.sidebar.header("Filtros del EDA")
opciones_suelo = sorted(parcelas["suelo"].dropna().unique().tolist())
opciones_riego = sorted(parcelas["riego"].dropna().unique().tolist())
suelos_seleccionados = st.sidebar.multiselect(
    "Suelo", options=opciones_suelo, default=opciones_suelo
)
riegos_seleccionados = st.sidebar.multiselect(
    "Riego", options=opciones_riego, default=opciones_riego
)

# Se filtran las parcelas una vez para que los apartados principales sean comparables.
datos = parcelas.loc[
    parcelas["suelo"].isin(suelos_seleccionados)
    & parcelas["riego"].isin(riegos_seleccionados)
].copy()

# Se presenta el título y se detiene el análisis si los filtros no dejan observaciones.
st.title("EDA: Análisis Exploratorio de Datos")
st.caption(f"Parcelas incluidas por los filtros: {len(datos)} de {len(parcelas)}.")
if datos.empty:
    st.warning("Selecciona al menos un tipo de suelo y una condición de riego.")
    st.stop()

# A. Se resumen variables numéricas y se muestran sus distribuciones y atípicos.
st.header("A. Análisis univariado")
variables_numericas = datos.select_dtypes(include="number").columns.tolist()
variable = st.selectbox(
    "Variable numérica",
    options=variables_numericas,
    index=(variables_numericas.index("rendimiento") if "rendimiento" in variables_numericas else 0),
)
if not validar_respuesta_numerica(datos, variable, "resumen univariado"):
    st.stop()
if not validar_minimo_observaciones(datos[variable], 1, "resumen univariado"):
    st.stop()
try:
    resumen = resumen_numerico(datos[variable])
except Exception as error:
    mostrar_error_ajuste(error, "el cálculo del resumen numérico")
    st.stop()
st.dataframe(
    resumen.rename_axis("Estadístico").reset_index(),
    width="stretch",
    hide_index=True,
)

# Se dibujan histograma y densidad con las mismas observaciones válidas.
columna_histograma, columna_caja = st.columns(2)
valores_variable = datos[variable].dropna()
with columna_histograma:
    figura, eje = plt.subplots()
    sns.histplot(
        data=datos,
        x=variable,
        stat="density",
        kde=valores_variable.nunique() > 1,
        ax=eje,
        color="#287c78",
    )
    eje.set_xlabel(ETIQUETAS_NUMERICAS.get(variable, variable))
    eje.set_ylabel("Densidad")
    eje.set_title("Histograma y curva de densidad")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

# Se representa el resumen de cinco números en un diagrama de caja.
with columna_caja:
    figura, eje = plt.subplots()
    sns.boxplot(data=datos, y=variable, ax=eje, color="#e7a34b")
    eje.set_ylabel(ETIQUETAS_NUMERICAS.get(variable, variable))
    eje.set_xlabel("")
    eje.set_title("Diagrama de caja")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

# Se identifican las observaciones exteriores a las vallas y se recuperan sus parcelas.
try:
    limite_inferior, limite_superior, atipicos = vallas_tukey(datos[variable])
except Exception as error:
    mostrar_error_ajuste(error, "el cálculo de las vallas de Tukey")
    st.stop()
st.caption(
    f"Vallas de Tukey: inferior = {limite_inferior:.4f}; "
    f"superior = {limite_superior:.4f}."
)
if atipicos.empty:
    st.info("No se detectaron valores atípicos para esta variable con la regla de Tukey.")
else:
    filas_atipicas = datos.loc[atipicos.index, ["parcela", "suelo", "riego", variable]]
    st.markdown("**Parcelas con valores atípicos**")
    st.dataframe(filas_atipicas, width="stretch", hide_index=True)

# B. Se muestran frecuencias y gráficos de barras de las variables cualitativas.
st.header("B. Análisis cualitativo")
columna_suelo, columna_riego = st.columns(2)
tablas_frecuencias = {}
for columna, nombre, contenedor in [
    ("suelo", "Suelo", columna_suelo),
    ("riego", "Riego", columna_riego),
]:
    frecuencias = tabla_frecuencias(datos[columna])
    tablas_frecuencias[nombre] = frecuencias
    with contenedor:
        st.subheader(nombre)
        st.dataframe(frecuencias, width="stretch", hide_index=True)

        # La altura se representa con frecuencias absolutas para facilitar el conteo.
        figura, eje = plt.subplots()
        sns.barplot(
            data=frecuencias,
            x="Categoría",
            y="Frecuencia absoluta",
            ax=eje,
            color="#287c78",
        )
        eje.set_xlabel(nombre)
        eje.set_ylabel("Frecuencia absoluta (parcelas)")
        eje.set_title(f"Frecuencia de {nombre.lower()}")
        figura.tight_layout()
        st.pyplot(figura)
        plt.close(figura)

# C. Se comparan rendimientos por grupos y se explora la asociación entre variables.
st.header("C. Análisis bivariado")
columna_caja_suelo, columna_caja_riego = st.columns(2)
for variable_grupo, nombre_grupo, contenedor in [
    ("suelo", "Suelo", columna_caja_suelo),
    ("riego", "Riego", columna_caja_riego),
]:
    with contenedor:
        figura, eje = plt.subplots()
        sns.boxplot(
            data=datos,
            x=variable_grupo,
            y="rendimiento",
            ax=eje,
            color="#e7a34b",
        )
        eje.set_xlabel(nombre_grupo)
        eje.set_ylabel("Rendimiento (t/ha)")
        eje.set_title(f"Rendimiento por {nombre_grupo.lower()}")
        figura.tight_layout()
        st.pyplot(figura)
        plt.close(figura)

# Se elige una variable explicativa numérica y se colorea cada punto por condición de riego.
variables_explicativas = [
    nombre for nombre in variables_numericas if nombre != "rendimiento"
]
if variables_explicativas:
    variable_x = st.selectbox(
        "Variable X para comparar con rendimiento",
        options=variables_explicativas,
        index=0,
    )
    figura, eje = plt.subplots()
    sns.scatterplot(
        data=datos,
        x=variable_x,
        y="rendimiento",
        hue="riego",
        ax=eje,
        palette="Set1",
    )
    eje.set_xlabel(ETIQUETAS_NUMERICAS.get(variable_x, variable_x))
    eje.set_ylabel("Rendimiento (t/ha)")
    eje.set_title("Relación entre variable X y rendimiento")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

# Se calcula y dibuja la matriz de correlación de Pearson entre columnas numéricas.
matriz_correlacion = datos.select_dtypes(include="number").corr(method="pearson")
matriz_etiquetada = matriz_correlacion.rename(
    columns=ETIQUETAS_NUMERICAS,
    index=ETIQUETAS_NUMERICAS,
)
if matriz_correlacion.shape[0] < 2 or matriz_correlacion.dropna(how="all").empty:
    st.info("No hay suficientes variables numéricas para calcular una matriz de correlación.")
else:
    figura, eje = plt.subplots()
    sns.heatmap(
        matriz_etiquetada,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        ax=eje,
    )
    eje.set_title("Matriz de correlación de Pearson")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

# D. Se localizan parcelas con faltantes y se explica por qué no deben borrarse automáticamente.
st.header("D. Datos faltantes y atípicos")
parcelas_con_faltantes = datos.loc[datos.isna().any(axis=1)].copy()
if parcelas_con_faltantes.empty:
    st.success("No hay parcelas con valores faltantes bajo los filtros seleccionados.")
else:
    parcelas_con_faltantes["Variables faltantes"] = parcelas_con_faltantes.apply(
        lambda fila: ", ".join(fila.index[fila.isna()].tolist()), axis=1
    )
    st.dataframe(
        parcelas_con_faltantes[["parcela", "Variables faltantes"]],
        width="stretch",
        hide_index=True,
    )
st.info(
    "Un faltante no debe borrarse sin justificar: primero verifica su origen y patrón, "
    "conserva el dato original y documenta cualquier exclusión o imputación. La elección "
    "depende de la variable, del mecanismo de ausencia y del objetivo del análisis."
)

# E. Se conserva la muestra didáctica completa para verificar los cálculos a mano.
st.header("E. Verificación del cálculo a mano (10 parcelas)")
if not validar_columnas_seleccionadas(
    muestra_eda,
    ["parcela", "suelo", "riego", "rendimiento"],
    "verificación con 10 parcelas",
):
    st.stop()
if not validar_respuesta_numerica(muestra_eda, "rendimiento", "verificación manual"):
    st.stop()
if not validar_minimo_observaciones(
    muestra_eda["rendimiento"], 10, "verificación con 10 parcelas"
):
    st.stop()
st.dataframe(muestra_eda, width="stretch", hide_index=True)

# Se presenta el resumen de rendimiento de las diez parcelas, usando cuatro decimales.
resumen_muestra = resumen_numerico(muestra_eda["rendimiento"])
st.markdown("**Resumen numérico del rendimiento (t/ha)**")
st.dataframe(
    resumen_muestra.rename_axis("Estadístico").reset_index(),
    width="stretch",
    hide_index=True,
    column_config={
        "Valor": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se desglosan las desviaciones de cada rendimiento respecto de la media muestral.
rendimiento_muestra = muestra_eda["rendimiento"]
media_muestra = float(rendimiento_muestra.mean())
tabla_desviaciones = pd.DataFrame(
    {
        "Parcela": muestra_eda["parcela"],
        "x (rendimiento, t/ha)": rendimiento_muestra,
        "x - media": rendimiento_muestra - media_muestra,
        "(x - media)^2": (rendimiento_muestra - media_muestra) ** 2,
    }
)
st.markdown(f"**Media de rendimiento para las desviaciones:** {media_muestra:.4f} t/ha")
st.dataframe(
    tabla_desviaciones,
    width="stretch",
    hide_index=True,
    column_config={
        "x (rendimiento, t/ha)": st.column_config.NumberColumn(format="%.4f"),
        "x - media": st.column_config.NumberColumn(format="%.4f"),
        "(x - media)^2": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se muestran las vallas y las parcelas que el criterio de Tukey marca como atípicas.
limite_muestra_inferior, limite_muestra_superior, atipicos_muestra = vallas_tukey(
    rendimiento_muestra
)
st.write(
    f"Valla inferior: {limite_muestra_inferior:.4f} t/ha; "
    f"valla superior: {limite_muestra_superior:.4f} t/ha."
)
if atipicos_muestra.empty:
    st.write("Atípicos según Tukey: ninguno.")
else:
    tabla_atipicos_muestra = muestra_eda.loc[
        atipicos_muestra.index, ["parcela", "suelo", "riego", "rendimiento"]
    ]
    st.markdown("**Atípicos según Tukey**")
    st.dataframe(
        tabla_atipicos_muestra,
        width="stretch",
        hide_index=True,
        column_config={
            "rendimiento": st.column_config.NumberColumn(format="%.4f"),
        },
    )

# Se comparan las medias de rendimiento por riego con cuatro decimales.
media_por_riego = (
    muestra_eda.groupby("riego", dropna=False)["rendimiento"]
    .mean()
    .rename("Media de rendimiento (t/ha)")
    .reset_index()
)
st.markdown("**Media de rendimiento por riego**")
st.dataframe(
    media_por_riego,
    width="stretch",
    hide_index=True,
    column_config={
        "Media de rendimiento (t/ha)": st.column_config.NumberColumn(format="%.4f"),
    },
)

# Se ofrecen tablas, figura y un resumen descargable sin cambiar los cálculos del EDA.
with st.expander("Descargar resultados"):
    descargar_excel(
        {
            "Resumen seleccionado": resumen.rename("Valor").rename_axis("Estadístico").reset_index(),
            "Frecuencias suelo": tablas_frecuencias["Suelo"],
            "Frecuencias riego": tablas_frecuencias["Riego"],
            "Muestra 10 parcelas": muestra_eda,
            "Desviaciones muestra": tabla_desviaciones,
            "Media por riego": media_por_riego,
        },
        "eda_resultados",
        key="descargar_eda_excel",
    )
    descargar_figuras(figura, "eda_matriz_correlacion", key="descargar_eda_figura")
    reporte_eda = (
        f"Variable analizada: {variable} ({ETIQUETAS_NUMERICAS.get(variable, variable)}).\n\n"
        f"Observaciones incluidas: {len(datos)}.\n\n"
        f"Media: {resumen['Media']:.4f}; mediana: {resumen['Mediana']:.4f}; "
        f"desviación estándar: {resumen['Desviación estándar muestral']:.4f}.\n\n"
        f"Vallas de Tukey: [{limite_inferior:.4f}, {limite_superior:.4f}].\n\n"
        f"Atípicos detectados: {len(atipicos)}."
    )
    descargar_reporte_markdown(
        "Resumen EDA",
        reporte_eda,
        "eda_reporte.md",
        key="descargar_eda_reporte",
    )