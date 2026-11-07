"""Exploración de correlación y regresión para las parcelas agrícolas."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats

from src.datos import cargar_csv, tarjeta_estudiante
from src.regresion import (
    ajustar_multiple,
    ajustar_rls,
    calculo_manual,
    diagnostico_residuos,
    matriz_correlacion,
    predecir,
)


# Se registran unidades para las ecuaciones, tablas y ejes de las gráficas.
UNIDADES = {
    "dosis_N": "kg/ha",
    "lamina_riego": "mm",
    "materia_organica": "%",
    "ph": "sin unidad",
    "rendimiento": "t/ha",
}

# Se carga la base completa y la muestra fija para los ejercicios manuales.
tarjeta_estudiante()
parcelas = cargar_csv("parcelas.csv")
muestra_reg8 = cargar_csv("muestra_reg8.csv")
respuesta = "rendimiento"

# A. Se calcula la matriz elegida y la correlación de cada predictor con rendimiento.
st.title("Regresión")
st.header("Correlación")
metodo_visible = st.selectbox("Método de correlación", ["Pearson", "Spearman"])
correlaciones = matriz_correlacion(parcelas, metodo_visible.lower())
etiquetas = {
    nombre: f"{nombre} ({UNIDADES.get(nombre, 'sin unidad')})"
    for nombre in correlaciones.columns
}
correlaciones_grafico = correlaciones.rename(index=etiquetas, columns=etiquetas)

# Se dibuja el mapa de calor con nombres y unidades de las variables cuantitativas.
figura, eje = plt.subplots(figsize=(9, 6))
sns.heatmap(
    correlaciones_grafico,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    vmin=-1,
    vmax=1,
    ax=eje,
)
eje.set_title(f"Matriz de correlación de {metodo_visible}")
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se ordenan los predictores por la magnitud absoluta de su correlación con rendimiento.
correlacion_rendimiento = (
    correlaciones[respuesta]
    .drop(index=respuesta)
    .rename("Correlación con rendimiento")
    .to_frame()
)
correlacion_rendimiento["|r|"] = correlacion_rendimiento[
    "Correlación con rendimiento"
].abs()
correlacion_rendimiento = correlacion_rendimiento.sort_values("|r|", ascending=False)
st.dataframe(correlacion_rendimiento, width="stretch")
st.info("La correlación describe asociación; por sí sola no demuestra causalidad.")

# B. Se selecciona un predictor y se ajusta rendimiento ~ X con MCO.
st.header("Regresión lineal simple")
variables_x = [
    nombre
    for nombre in parcelas.select_dtypes(include="number").columns
    if nombre != respuesta
]
indice_lamina = variables_x.index("lamina_riego") if "lamina_riego" in variables_x else 0
variable_x = st.selectbox("Variable explicativa X", variables_x, index=indice_lamina)
casos_simples = parcelas[[variable_x, respuesta, "riego"]].dropna().copy()
modelo_simple = ajustar_rls(parcelas, variable_x, respuesta)
manual_simple = calculo_manual(casos_simples[variable_x], casos_simples[respuesta])
pendiente = float(modelo_simple.params.iloc[1])
intercepto = float(modelo_simple.params.iloc[0])
error_pendiente = float(modelo_simple.bse.iloc[1])
estadistico_t = float(modelo_simple.tvalues.iloc[1])
p_pendiente = float(modelo_simple.pvalues.iloc[1])
intervalos_simple = modelo_simple.conf_int(alpha=0.05)
ic_pendiente = intervalos_simple.iloc[1].to_numpy(dtype=float)
rse_simple = float(np.sqrt(modelo_simple.mse_resid))

# Se grafica la dispersión y la recta ajustada en el rango observado.
figura, eje = plt.subplots(figsize=(9, 5))
sns.scatterplot(
    data=casos_simples,
    x=variable_x,
    y=respuesta,
    hue="riego",
    ax=eje,
    palette="Set1",
)
x_recta = np.linspace(casos_simples[variable_x].min(), casos_simples[variable_x].max(), 200)
y_recta = intercepto + pendiente * x_recta
eje.plot(x_recta, y_recta, color="#202f3c", linewidth=2, label="Recta MCO")
unidad_x = UNIDADES.get(variable_x, "sin unidad")
eje.set_xlabel(f"{variable_x} ({unidad_x})")
eje.set_ylabel("Rendimiento (t/ha)")
eje.set_title("Rendimiento y recta de regresión lineal simple")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se expresa la ecuación con unidades y se resumen correlación y ajuste.
st.markdown(
    f"**Ecuación:** ŷ (t/ha) = {intercepto:.4f} (t/ha) "
    f"+ ({pendiente:.6f} t/ha por {unidad_x}) × {variable_x} ({unidad_x})."
)
columna_r, columna_r2, columna_rse = st.columns(3)
columna_r.metric("r de Pearson", f"{manual_simple['r_pearson']:.4f}")
columna_r2.metric("R²", f"{modelo_simple.rsquared:.4f}")
columna_rse.metric("RSE (t/ha)", f"{rse_simple:.4f}")

# Se comparan coeficientes, errores estándar, pruebas t e intervalos al 95 %.
filas_coeficientes = []
for indice, nombre in enumerate(["Intercepto", f"Pendiente: {variable_x}"]):
    filas_coeficientes.append(
        {
            "Término": nombre,
            "Coeficiente": modelo_simple.params.iloc[indice],
            "Error estándar": modelo_simple.bse.iloc[indice],
            "t": modelo_simple.tvalues.iloc[indice],
            "p-valor": modelo_simple.pvalues.iloc[indice],
            "IC 95 % inferior": intervalos_simple.iloc[indice, 0],
            "IC 95 % superior": intervalos_simple.iloc[indice, 1],
        }
    )
st.dataframe(
    pd.DataFrame(filas_coeficientes),
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.6f")
        for nombre in filas_coeficientes[0]
        if nombre != "Término"
    },
)

# Se comunica la decisión de H0 y se traduce la pendiente a un cambio interpretable.
if p_pendiente < 0.05:
    decision_pendiente = (
        f"Al 5 %, se rechaza H0: β1 = 0 (p = {p_pendiente:.4f}); hay evidencia de "
        "asociación lineal en esta muestra, sin implicar causalidad."
    )
else:
    decision_pendiente = (
        f"Al 5 %, no se rechaza H0: β1 = 0 (p = {p_pendiente:.4f}); la muestra no aporta "
        "evidencia suficiente de una pendiente distinta de cero."
    )
st.info(decision_pendiente)
if variable_x == "lamina_riego":
    cambio_100_mm = pendiente * 100
    st.write(
        f"Interpretación: por cada 100 mm adicionales de lámina de riego, el rendimiento "
        f"estimado cambia {cambio_100_mm:.4f} t/ha, en promedio y dentro del rango observado."
    )
else:
    cambio_100_unidades = pendiente * 100
    st.write(
        f"Para {variable_x}, b1 representa {pendiente:.6f} t/ha por cada unidad de {unidad_x}; "
        f"por 100 unidades serían {cambio_100_unidades:.4f} t/ha. La interpretación por "
        "100 mm corresponde únicamente a lamina_riego."
    )

# C. Se visualizan residuos y se interpretan las pruebas de normalidad y varianza constante.
st.header("Diagnóstico de residuos")
diagnostico = diagnostico_residuos(modelo_simple)
residuos = diagnostico["residuos"].to_numpy()
ajustados = diagnostico["ajustados"].to_numpy()
columna_residuos, columna_histograma = st.columns(2)
with columna_residuos:
    figura, eje = plt.subplots(figsize=(7, 4))
    sns.scatterplot(x=ajustados, y=residuos, ax=eje, color="#287c78")
    eje.axhline(0, color="#b7473e", linestyle="--")
    eje.set_xlabel("Rendimiento ajustado (t/ha)")
    eje.set_ylabel("Residuo (t/ha)")
    eje.set_title("Residuos frente a valores ajustados")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

# Se revisa la forma de los residuos con un histograma y un gráfico Q-Q normal.
with columna_histograma:
    figura, eje = plt.subplots(figsize=(7, 4))
    sns.histplot(x=residuos, bins="auto", kde=True, ax=eje, color="#e7a34b")
    eje.set_xlabel("Residuo (t/ha)")
    eje.set_ylabel("Frecuencia")
    eje.set_title("Histograma de residuos")
    figura.tight_layout()
    st.pyplot(figura)
    plt.close(figura)

figura, eje = plt.subplots(figsize=(7, 4))
stats.probplot(residuos, dist="norm", plot=eje)
eje.set_xlabel("Cuantiles teóricos normales")
eje.set_ylabel("Residuos observados (t/ha)")
eje.set_title("Gráfico Q-Q de residuos")
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se muestran valores p y conclusiones sencillas para ambos diagnósticos.
shapiro = diagnostico["shapiro_wilk"]
breusch_pagan = diagnostico["breusch_pagan"]
columna_shapiro, columna_bp = st.columns(2)
columna_shapiro.metric("Shapiro-Wilk: p-valor", f"{shapiro['p_valor']:.4f}")
columna_bp.metric("Breusch-Pagan: p-valor", f"{breusch_pagan['p_LM']:.4f}")
if shapiro["p_valor"] >= 0.05:
    st.success("Shapiro-Wilk: no se rechaza normalidad de residuos al 5 %.")
else:
    st.warning("Shapiro-Wilk: hay evidencia de desviación de normalidad al 5 %.")
if breusch_pagan["p_LM"] >= 0.05:
    st.success("Breusch-Pagan: no se rechaza varianza constante al 5 %.")
else:
    st.warning("Breusch-Pagan: hay evidencia de heterocedasticidad al 5 %.")

# D. Se calcula dentro del rango observado y se avisa explícitamente ante extrapolación.
st.header("Predicción")
minimo_x = float(casos_simples[variable_x].min())
maximo_x = float(casos_simples[variable_x].max())
paso_x = max((maximo_x - minimo_x) / 100, 0.01)
valor_inicial = float(casos_simples[variable_x].median())
valor_slider = st.slider(
    f"X dentro del rango observado ({unidad_x})",
    min_value=minimo_x,
    max_value=maximo_x,
    value=valor_inicial,
    step=paso_x,
)
prediccion_slider = predecir(modelo_simple, valor_slider).iloc[0]
st.dataframe(
    pd.DataFrame(
        [
            {
                "X": valor_slider,
                "Predicción (t/ha)": prediccion_slider["Predicción"],
                "IC 95 % inferior": prediccion_slider["IC 95 % inferior"],
                "IC 95 % superior": prediccion_slider["IC 95 % superior"],
                "IP 95 % inferior": prediccion_slider["IP 95 % inferior"],
                "IP 95 % superior": prediccion_slider["IP 95 % superior"],
            }
        ]
    ),
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.4f")
        for nombre in [
            "X",
            "Predicción (t/ha)",
            "IC 95 % inferior",
            "IC 95 % superior",
            "IP 95 % inferior",
            "IP 95 % superior",
        ]
    },
)

# Se ofrece un valor libre y se avisa cuando queda fuera del rango muestral de X.
valor_libre = st.number_input(
    f"Valor libre de X ({unidad_x})",
    value=valor_inicial,
    step=paso_x,
)
if valor_libre < minimo_x or valor_libre > maximo_x:
    st.warning(
        f"EXTRAPOLACIÓN: {valor_libre:.4f} queda fuera del rango observado "
        f"[{minimo_x:.4f}, {maximo_x:.4f}]. La predicción puede no ser fiable."
    )
prediccion_libre = predecir(modelo_simple, valor_libre).iloc[0]
st.metric("Predicción para el valor libre (t/ha)", f"{prediccion_libre['Predicción']:.4f}")
st.write(
    f"IC 95 % de la media: [{prediccion_libre['IC 95 % inferior']:.4f}, "
    f"{prediccion_libre['IC 95 % superior']:.4f}] t/ha. "
    f"IP 95 % individual: [{prediccion_libre['IP 95 % inferior']:.4f}, "
    f"{prediccion_libre['IP 95 % superior']:.4f}] t/ha."
)

# E. Se presenta un modelo múltiple con cuatro predictores y casos completos.
st.header("Regresión múltiple (vista previa)")
variables_multiples = ["dosis_N", "lamina_riego", "materia_organica", "ph"]
modelo_multiple = ajustar_multiple(parcelas, respuesta, variables_multiples)
columna_r2_multi, columna_r2_adj_multi, columna_n_multi = st.columns(3)
columna_r2_multi.metric("R² múltiple", f"{modelo_multiple.rsquared:.4f}")
columna_r2_adj_multi.metric("R² ajustado", f"{modelo_multiple.rsquared_adj:.4f}")
columna_n_multi.metric("Filas completas", f"{int(modelo_multiple.nobs)}")

# Se traducen coeficientes, errores estándar, pruebas e intervalos a nombres legibles.
intervalos_multiple = modelo_multiple.conf_int(alpha=0.05)
filas_coeficientes_multiple = []
for indice, nombre in enumerate(["Intercepto", *variables_multiples]):
    filas_coeficientes_multiple.append(
        {
            "Término": nombre,
            "Coeficiente": modelo_multiple.params.iloc[indice],
            "Error estándar": modelo_multiple.bse.iloc[indice],
            "t": modelo_multiple.tvalues.iloc[indice],
            "p-valor": modelo_multiple.pvalues.iloc[indice],
            "IC 95 % inferior": intervalos_multiple.iloc[indice, 0],
            "IC 95 % superior": intervalos_multiple.iloc[indice, 1],
        }
    )
st.dataframe(
    pd.DataFrame(filas_coeficientes_multiple),
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.4f")
        for nombre in filas_coeficientes_multiple[0]
        if nombre != "Término"
    },
)
st.caption("El ajuste múltiple elimina filas con faltantes en cualquiera de sus variables.")

# F. Se verifica a mano la muestra de ocho parcelas y se compara con todas las parcelas.
st.header("Verificación del cálculo a mano (8 parcelas)")
variable_x_manual = "lamina_riego"
manual_muestra = calculo_manual(
    muestra_reg8[variable_x_manual],
    muestra_reg8[respuesta],
)
x_muestra = muestra_reg8[variable_x_manual].to_numpy(dtype=float)
y_muestra = muestra_reg8[respuesta].to_numpy(dtype=float)
desviacion_x = x_muestra - manual_muestra["media_x"]
desviacion_y = y_muestra - manual_muestra["media_y"]
tabla_pasos = muestra_reg8[["parcela", "suelo", "riego"]].copy()
tabla_pasos["x (lámina, mm)"] = x_muestra
tabla_pasos["y (rendimiento, t/ha)"] = y_muestra
tabla_pasos["x - x̄"] = desviacion_x
tabla_pasos["y - ȳ"] = desviacion_y
tabla_pasos["(x - x̄)(y - ȳ)"] = desviacion_x * desviacion_y
tabla_pasos["Rango x"] = pd.Series(x_muestra).rank(method="average").to_numpy()
tabla_pasos["Rango y"] = pd.Series(y_muestra).rank(method="average").to_numpy()
tabla_pasos["Ajustado ŷ (t/ha)"] = manual_muestra["ajustados"]
tabla_pasos["Residuo e (t/ha)"] = manual_muestra["residuos"]
st.dataframe(
    tabla_pasos,
    width="stretch",
    hide_index=True,
    column_config={
        nombre: st.column_config.NumberColumn(format="%.4f")
        for nombre in tabla_pasos.columns
        if nombre not in {"parcela", "suelo", "riego"}
    },
)

# Se listan todas las salidas escalares manuales con cuatro decimales y seis para b1.
resultados_manuales = [
    ("n", manual_muestra["n"]),
    ("Media de x", manual_muestra["media_x"]),
    ("Media de y", manual_muestra["media_y"]),
    ("Sxx", manual_muestra["Sxx"]),
    ("Syy", manual_muestra["Syy"]),
    ("Sxy", manual_muestra["Sxy"]),
    ("r de Pearson", manual_muestra["r_pearson"]),
    ("rho de Spearman", manual_muestra["rho_spearman"]),
    ("b1", manual_muestra["b1"]),
    ("b0", manual_muestra["b0"]),
    ("SCE", manual_muestra["SCE"]),
    ("SCR", manual_muestra["SCR"]),
    ("SCT", manual_muestra["SCT"]),
    ("R²", manual_muestra["R2"]),
    ("RSE", manual_muestra["RSE"]),
    ("SE(b1)", manual_muestra["SE_b1"]),
    ("t observado", manual_muestra["t"]),
    ("gl", manual_muestra["gl"]),
    ("t crítico bilateral (α=0.05)", manual_muestra["t_critico_bilateral_05"]),
    ("p-valor", manual_muestra["p_valor"]),
    ("IC 95 % b1 inferior", manual_muestra["IC95_b1"][0]),
    ("IC 95 % b1 superior", manual_muestra["IC95_b1"][1]),
]
tabla_resultados_manuales = pd.DataFrame(
    {
        "Resultado": [nombre for nombre, _ in resultados_manuales],
        "Valor": [
            f"{valor:.6f}" if nombre == "b1" else f"{valor:.4f}"
            for nombre, valor in resultados_manuales
        ],
    }
)
st.dataframe(tabla_resultados_manuales, width="stretch", hide_index=True)

# Se grafica la muestra de ocho puntos y su recta manual de mínimos cuadrados.
figura, eje = plt.subplots(figsize=(8, 4.5))
sns.scatterplot(x=x_muestra, y=y_muestra, ax=eje, color="#287c78", label="8 parcelas")
x_recta_muestra = np.linspace(x_muestra.min(), x_muestra.max(), 100)
eje.plot(
    x_recta_muestra,
    manual_muestra["b0"] + manual_muestra["b1"] * x_recta_muestra,
    color="#202f3c",
    label="Recta MCO manual",
)
eje.set_xlabel("Lámina de riego (mm)")
eje.set_ylabel("Rendimiento (t/ha)")
eje.set_title("Verificación manual: muestra de ocho parcelas")
eje.legend()
figura.tight_layout()
st.pyplot(figura)
plt.close(figura)

# Se comparan b1, r, R², t y p entre la muestra didáctica y todas las parcelas.
manual_todas = calculo_manual(parcelas[variable_x_manual], parcelas[respuesta])
comparacion_manual = pd.DataFrame(
    [
        {
            "Conjunto": "Muestra de 8 parcelas",
            "b1": manual_muestra["b1"],
            "r": manual_muestra["r_pearson"],
            "R²": manual_muestra["R2"],
            "t": manual_muestra["t"],
            "p-valor": manual_muestra["p_valor"],
        },
        {
            "Conjunto": "Todas las parcelas",
            "b1": manual_todas["b1"],
            "r": manual_todas["r_pearson"],
            "R²": manual_todas["R2"],
            "t": manual_todas["t"],
            "p-valor": manual_todas["p_valor"],
        },
    ]
)
st.dataframe(
    comparacion_manual,
    width="stretch",
    hide_index=True,
    column_config={
        "b1": st.column_config.NumberColumn(format="%.6f"),
        "r": st.column_config.NumberColumn(format="%.4f"),
        "R²": st.column_config.NumberColumn(format="%.4f"),
        "t": st.column_config.NumberColumn(format="%.4f"),
        "p-valor": st.column_config.NumberColumn(format="%.4f"),
    },
)