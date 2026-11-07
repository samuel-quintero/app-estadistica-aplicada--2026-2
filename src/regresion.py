"""Herramientas de correlación, regresión lineal e inferencia para datos agrícolas."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan


def matriz_correlacion(df: pd.DataFrame, metodo: str = "pearson") -> pd.DataFrame:
    """Devuelve la matriz Pearson o Spearman de las columnas numéricas del DataFrame."""
    # Se valida el método solicitado para evitar seleccionar accidentalmente otra medida.
    metodo_normalizado = metodo.lower()
    if metodo_normalizado not in {"pearson", "spearman"}:
        raise ValueError("metodo debe ser 'pearson' o 'spearman'.")

    # Solo se incluyen variables numéricas en la matriz de correlación.
    variables_numericas = df.select_dtypes(include="number")
    return variables_numericas.corr(method=metodo_normalizado)


def calculo_manual(x, y) -> dict:
    """Calcula a mano MCO simple, correlaciones, residuos e inferencia de la pendiente."""
    # Se convierten las entradas y se conservan únicamente pares finitos completos.
    valores_x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    valores_y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(dtype=float)
    if valores_x.size != valores_y.size:
        raise ValueError("x e y deben tener la misma cantidad de observaciones.")
    pares_validos = np.isfinite(valores_x) & np.isfinite(valores_y)
    valores_x = valores_x[pares_validos]
    valores_y = valores_y[pares_validos]
    n = int(valores_x.size)
    if n < 3:
        raise ValueError("Se requieren al menos tres pares completos para la inferencia MCO.")

    # Se calculan medias, desviaciones centradas y sumas de cuadrados/cruzados.
    media_x = float(np.mean(valores_x))
    media_y = float(np.mean(valores_y))
    diferencias_x = valores_x - media_x
    diferencias_y = valores_y - media_y
    sxx = float(np.sum(diferencias_x**2))
    syy = float(np.sum(diferencias_y**2))
    sxy = float(np.sum(diferencias_x * diferencias_y))
    if sxx == 0 or syy == 0:
        raise ValueError("x e y deben tener variación para calcular correlación y regresión.")

    # Se calculan las correlaciones de Pearson por sumas y Spearman por rangos.
    r_pearson = float(sxy / np.sqrt(sxx * syy))
    rho_spearman = float(stats.spearmanr(valores_x, valores_y).statistic)

    # La pendiente y el intercepto MCO usan las fórmulas cerradas de la clase.
    pendiente = float(sxy / sxx)
    intercepto = float(media_y - pendiente * media_x)
    ajustados = intercepto + pendiente * valores_x
    residuos = valores_y - ajustados

    # SCE mide error residual, SCR error explicado y SCT variación total observada.
    sce = float(np.sum(residuos**2))
    scr = float(np.sum((ajustados - media_y) ** 2))
    sct = syy
    r_cuadrado = float(1 - sce / sct)

    # RSE y SE(b1) estiman la dispersión residual y la incertidumbre de la pendiente.
    grados_libertad = n - 2
    rse = float(np.sqrt(sce / grados_libertad))
    error_pendiente = float(rse / np.sqrt(sxx))
    estadistico_t = float(pendiente / error_pendiente)

    # Se obtiene el crítico t bilateral, el p-valor y el intervalo de confianza del 95 %.
    t_critico = float(stats.t.ppf(1 - 0.05 / 2, df=grados_libertad))
    p_valor = float(2 * stats.t.sf(abs(estadistico_t), df=grados_libertad))
    intervalo_pendiente = (
        float(pendiente - t_critico * error_pendiente),
        float(pendiente + t_critico * error_pendiente),
    )

    # Se devuelven estadísticas escalares y vectores para contrastar cada paso manual.
    return {
        "n": n,
        "media_x": media_x,
        "media_y": media_y,
        "Sxx": sxx,
        "Syy": syy,
        "Sxy": sxy,
        "r_pearson": r_pearson,
        "rho_spearman": rho_spearman,
        "b1": pendiente,
        "b0": intercepto,
        "ajustados": ajustados,
        "residuos": residuos,
        "SCE": sce,
        "SCR": scr,
        "SCT": sct,
        "R2": r_cuadrado,
        "RSE": rse,
        "SE_b1": error_pendiente,
        "t": estadistico_t,
        "gl": grados_libertad,
        "t_critico_bilateral_05": t_critico,
        "p_valor": p_valor,
        "IC95_b1": intervalo_pendiente,
    }


def _nombre_formula(nombre: str) -> str:
    """Cita nombres de columnas arbitrarios usando la función Q de Patsy."""
    # repr crea una cadena válida incluso si el nombre contiene espacios o símbolos.
    return f"Q({nombre!r})"


def _guardar_variables(modelo, variables_x: list[str], variable_y: str):
    """Asocia al resultado de statsmodels los nombres originales de sus variables."""
    # Estos metadatos permiten que predecir construya correctamente la tabla de entrada.
    modelo._regresion_variables_x = list(variables_x)
    modelo._regresion_variable_y = variable_y
    return modelo


def ajustar_rls(df: pd.DataFrame, x: str, y: str):
    """Ajusta una regresión lineal simple por MCO con statsmodels formula API."""
    # Se eliminan filas sin x o y para que el ajuste y sus predicciones usen casos completos.
    datos = df[[x, y]].dropna().copy()
    if len(datos) < 3:
        raise ValueError("Se requieren al menos tres filas completas para ajustar la regresión.")

    # Se formula y ~ x con Q() para admitir nombres de columna no convencionales.
    formula = f"{_nombre_formula(y)} ~ {_nombre_formula(x)}"
    modelo = smf.ols(formula=formula, data=datos).fit()
    return _guardar_variables(modelo, [x], y)


def predecir(modelo, x0) -> pd.DataFrame:
    """Devuelve predicción media, IC de la media e intervalo de predicción al 95 %."""
    # Se recuperan los nombres originales guardados por las funciones de ajuste del módulo.
    variables_x = getattr(modelo, "_regresion_variables_x", None)
    if not variables_x:
        raise ValueError("El modelo debe haberse ajustado con ajustar_rls o ajustar_multiple.")

    # Se convierte x0 a un DataFrame con los nombres esperados por la fórmula de statsmodels.
    if isinstance(x0, pd.DataFrame):
        nuevos_datos = x0.copy()
    elif isinstance(x0, pd.Series):
        nuevos_datos = x0.to_frame().T
    elif isinstance(x0, dict):
        nuevos_datos = pd.DataFrame([x0])
    elif np.isscalar(x0):
        if len(variables_x) != 1:
            raise ValueError("Para regresión múltiple, x0 debe ser un dict o DataFrame.")
        nuevos_datos = pd.DataFrame({variables_x[0]: [x0]})
    else:
        matriz_x0 = np.asarray(x0)
        if len(variables_x) != 1:
            raise ValueError("Para regresión múltiple, x0 debe ser un dict o DataFrame.")
        nuevos_datos = pd.DataFrame({variables_x[0]: matriz_x0.reshape(-1)})

    # Se calcula la predicción y sus intervalos de confianza y predicción al 95 %.
    resumen = modelo.get_prediction(nuevos_datos).summary_frame(alpha=0.05)
    return resumen.rename(
        columns={
            "mean": "Predicción",
            "mean_se": "Error estándar de la media",
            "mean_ci_lower": "IC 95 % inferior",
            "mean_ci_upper": "IC 95 % superior",
            "obs_ci_lower": "IP 95 % inferior",
            "obs_ci_upper": "IP 95 % superior",
        }
    )


def diagnostico_residuos(modelo) -> dict:
    """Devuelve residuos, ajustados, Shapiro-Wilk y Breusch-Pagan del modelo."""
    # Se extraen residuos y valores ajustados con sus etiquetas originales de observación.
    datos_modelo = modelo.model.data
    etiquetas = getattr(datos_modelo, "row_labels", None)
    residuos = pd.Series(np.asarray(modelo.resid), index=etiquetas, name="residuo")
    ajustados = pd.Series(np.asarray(modelo.fittedvalues), index=etiquetas, name="ajustado")
    if residuos.size < 3:
        raise ValueError("Se requieren al menos tres residuos para Shapiro-Wilk.")

    # Shapiro-Wilk evalúa normalidad aproximada de los residuos.
    shapiro = stats.shapiro(residuos.to_numpy())

    # Breusch-Pagan contrasta varianza constante usando la matriz de diseño del modelo.
    bp_lm, bp_p_lm, bp_f, bp_p_f = het_breuschpagan(
        residuos.to_numpy(),
        modelo.model.exog,
    )
    return {
        "residuos": residuos,
        "ajustados": ajustados,
        "shapiro_wilk": {
            "estadistico": float(shapiro.statistic),
            "p_valor": float(shapiro.pvalue),
        },
        "breusch_pagan": {
            "LM": float(bp_lm),
            "p_LM": float(bp_p_lm),
            "F": float(bp_f),
            "p_F": float(bp_p_f),
        },
    }


def ajustar_multiple(df: pd.DataFrame, y: str, lista_x: list[str]):
    """Ajusta MCO múltiple por fórmula y elimina filas faltantes en variables del modelo."""
    # Se valida que haya predictores y se seleccionan solo variables incluidas en la fórmula.
    variables_x = list(lista_x)
    if not variables_x:
        raise ValueError("lista_x debe incluir al menos un predictor.")
    if len(set(variables_x)) != len(variables_x):
        raise ValueError("lista_x no debe contener predictores repetidos.")
    if y in variables_x:
        raise ValueError("La variable respuesta y no puede aparecer también en lista_x.")
    variables_modelo = [y, *variables_x]
    datos = df[variables_modelo].dropna().copy()
    if len(datos) <= len(variables_x) + 1:
        raise ValueError("No hay suficientes filas completas para ajustar el modelo múltiple.")

    # Se construye y ~ x1 + x2 + ... y se ajusta por MCO con statsmodels.
    predictores = " + ".join(_nombre_formula(nombre) for nombre in variables_x)
    formula = f"{_nombre_formula(y)} ~ {predictores}"
    modelo = smf.ols(formula=formula, data=datos).fit()
    return _guardar_variables(modelo, variables_x, y)


if __name__ == "__main__":
    # Se carga la muestra pequeña generada para verificar la regresión a mano.
    ruta_muestra = Path(__file__).resolve().parents[1] / "datos" / "muestra_reg8.csv"
    muestra = pd.read_csv(ruta_muestra)
    variable_x = "lamina_riego"
    variable_y = "rendimiento"

    # Se calculan las fórmulas manuales y el ajuste equivalente de statsmodels.
    manual = calculo_manual(muestra[variable_x], muestra[variable_y])
    modelo = ajustar_rls(muestra, variable_x, variable_y)
    parametros = modelo.params.to_numpy()
    valores_statsmodels = {
        "b0": float(parametros[0]),
        "b1": float(parametros[1]),
        "R2": float(modelo.rsquared),
        "RSE": float(np.sqrt(modelo.mse_resid)),
        "t": float(modelo.tvalues.iloc[1]),
    }

    # Se exige coincidencia numérica en b0, b1, R², RSE y t antes de imprimir el reporte.
    for nombre in ("b0", "b1", "R2", "RSE", "t"):
        np.testing.assert_allclose(manual[nombre], valores_statsmodels[nombre], rtol=1e-10)

    # Se muestran los resultados de ambos métodos y el estado de las verificaciones.
    print(f"Muestra: {ruta_muestra.name}; n={manual['n']}; x={variable_x}; y={variable_y}")
    print("Medida                  Manual          statsmodels")
    for nombre, etiqueta in (
        ("b0", "b0"),
        ("b1", "b1"),
        ("R2", "R²"),
        ("RSE", "RSE"),
        ("t", "t"),
    ):
        print(f"{etiqueta:<22} {manual[nombre]:>12.8f} {valores_statsmodels[nombre]:>16.8f}")
    print("Verificación manual y statsmodels: coinciden.")