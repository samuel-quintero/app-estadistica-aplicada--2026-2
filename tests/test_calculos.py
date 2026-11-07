"""Comprueba que los cálculos de src/ coincidan con los verificadores independientes."""

# runpy ejecuta los scripts de verificación sin importar funciones de la aplicación.
import runpy
from pathlib import Path

# NumPy compara resultados de punto flotante con tolerancia numérica.
import numpy as np
import pandas as pd

# Se importan únicamente las funciones de cálculo que deben contrastarse.
from src.eda import resumen_numerico, vallas_tukey
from src.distribuciones import (
    cuantil_retorno,
    gumbel_momentos,
    posiciones_graficas,
    riesgo_falla,
)
from src.regresion import calculo_manual
from src.series import indices_estacionales, metodos_referencia, metricas, promedio_movil


# Se localizan muestras y verificadores desde la raíz del repositorio.
RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "datos"
RUTA_VERIFICACION = RAIZ / "verificacion"


def test_eda_coincide_con_verificador_independiente():
    """Compara estadísticos descriptivos, Tukey y medias por riego."""
    # Se ejecuta el script manual independiente y se carga la misma muestra pequeña.
    verificador = runpy.run_path(str(RUTA_VERIFICACION / "verificar_eda.py"))
    datos = pd.read_csv(RUTA_DATOS / "muestra_eda10.csv")
    resumen = resumen_numerico(datos["rendimiento"])

    # Se cotejan centro, dispersión y cuantiles tipo 7 con las fórmulas del script.
    assert np.isclose(resumen["Media"], verificador["media"])
    assert np.isclose(resumen["Mediana"], verificador["mediana"])
    assert np.isclose(resumen["Varianza muestral"], verificador["varianza"])
    assert np.isclose(resumen["Desviación estándar muestral"], verificador["desviacion"])
    assert np.isclose(resumen["CV (%)"], verificador["cv"])
    assert np.isclose(resumen["Q1 (tipo 7)"], verificador["q1"])
    assert np.isclose(resumen["Q3 (tipo 7)"], verificador["q3"])

    # Se comparan RIC, vallas y parcelas que quedan fuera del rango de Tukey.
    inferior, superior, atipicos = vallas_tukey(datos["rendimiento"])
    assert np.isclose(resumen["RIC"], verificador["ric"])
    assert np.isclose(inferior, verificador["valla_inferior"])
    assert np.isclose(superior, verificador["valla_superior"])
    assert atipicos.index.tolist() == verificador["atipicos"].index.tolist()

    # Se comparan las medias de rendimiento por riego con el verificador independiente.
    medias_src = datos.groupby("riego")["rendimiento"].mean().to_dict()
    medias_verificador = verificador["medias_riego"].set_index("riego")["rendimiento"].to_dict()
    assert medias_src.keys() == medias_verificador.keys()
    for riego in medias_src:
        assert np.isclose(medias_src[riego], medias_verificador[riego])


def test_distribuciones_coincide_con_verificador_independiente():
    """Compara posiciones Weibull, Gumbel por momentos, cuantiles y riesgo."""
    # Se ejecuta el script de fórmulas y se lee su muestra ordenada.
    verificador = runpy.run_path(str(RUTA_VERIFICACION / "verificar_distribuciones.py"))
    datos = pd.read_csv(RUTA_DATOS / "muestra_dist10.csv")
    valores = datos["pmax24h_mm"]
    posiciones = posiciones_graficas(valores)

    # Las posiciones y retornos de Weibull deben coincidir observación por observación.
    assert np.allclose(posiciones["p Weibull"], verificador["probabilidad"])
    assert np.allclose(posiciones["T Weibull (años)"], verificador["periodo_retorno"])

    # Se comparan parámetros de Gumbel por momentos y cuantiles de diseño.
    parametros = gumbel_momentos(valores)
    assert np.isclose(parametros["mu"], verificador["mu"])
    assert np.isclose(parametros["beta"], verificador["beta"])
    for periodo, cuantil in zip([10, 50, 100], verificador["cuantiles_gumbel"]):
        assert np.isclose(cuantil_retorno("Gumbel", (parametros["mu"], parametros["beta"]), periodo), cuantil)

    # El riesgo de falla también debe coincidir para T=50 y una vida útil de 25 años.
    assert np.isclose(riesgo_falla(50, 25), verificador["riesgo"])


def test_regresion_coincide_con_verificador_independiente():
    """Compara sumas, correlaciones, MCO e inferencia de muestra_reg8.csv."""
    # Se ejecuta el verificador de fórmulas explícitas y se carga la misma muestra.
    verificador = runpy.run_path(str(RUTA_VERIFICACION / "verificar_regresion.py"))
    datos = pd.read_csv(RUTA_DATOS / "muestra_reg8.csv")
    resultado = calculo_manual(datos["lamina_riego"], datos["rendimiento"])

    # Se comparan sumas y correlaciones calculadas desde las desviaciones y rangos.
    for nombre in ["Sxx", "Syy", "Sxy"]:
        assert np.isclose(resultado[nombre], verificador[nombre.lower()])
    assert np.isclose(resultado["r_pearson"], verificador["r_pearson"])
    assert np.isclose(resultado["rho_spearman"], verificador["rho_spearman"])

    # Se comparan coeficientes, sumas de error y estadísticas inferenciales.
    comparaciones = {
        "b0": "b0",
        "b1": "b1",
        "SCE": "sce",
        "R2": "r_cuadrado",
        "RSE": "rse",
        "SE_b1": "se_b1",
        "t": "t_observado",
        "p_valor": "p_valor",
        "IC95_b1": "ic_inferior",
    }
    for nombre, nombre_verificador in comparaciones.items():
        if nombre == "IC95_b1":
            assert np.isclose(resultado[nombre][0], verificador[nombre_verificador])
            assert np.isclose(resultado[nombre][1], verificador["ic_superior"])
        else:
            assert np.isclose(resultado[nombre], verificador[nombre_verificador])


def test_series_coincide_con_verificador_independiente():
    """Compara PM3, índices aditivos, pronósticos y métricas de muestra_series24.csv."""
    # Se ejecuta el verificador temporal y se ordena la misma muestra por fecha.
    verificador = runpy.run_path(str(RUTA_VERIFICACION / "verificar_series.py"))
    datos = pd.read_csv(RUTA_DATOS / "muestra_series24.csv", parse_dates=["fecha"])
    datos = datos.sort_values("fecha").reset_index(drop=True)
    serie = datos.set_index("fecha")["precipitacion_mm"]

    # Se coteja PM3 calculado con rolling y los doce índices mensuales aditivos.
    pm3 = promedio_movil(serie, 3)
    assert np.allclose(pm3.to_numpy(), verificador["datos"]["PM3"].to_numpy(), equal_nan=True)
    indices = indices_estacionales(serie)
    assert np.allclose(indices.to_numpy(), verificador["indices"])

    # Se compara cada pronóstico de referencia usando exactamente los 18 meses de entrenamiento.
    entrenamiento = datos.loc[datos["conjunto"] == "entrenamiento"].set_index("fecha")[
        "precipitacion_mm"
    ]
    real = datos.loc[datos["conjunto"] == "validacion"].set_index("fecha")["precipitacion_mm"]
    pronosticos = metodos_referencia(entrenamiento, h=len(real))
    for nombre, clave in [
        ("ingenuo", "y_ingenuo"),
        ("estacional_ingenuo", "y_estacional"),
        ("promedio", "y_promedio"),
    ]:
        assert np.allclose(pronosticos[nombre].to_numpy(), verificador[clave])

        # Se cotejan MAE, RMSE y SMAPE derivados de los errores mensuales.
        metricas_src = metricas(real, pronosticos[nombre])
        metricas_verif = verificador["metricas_metodos"][nombre]
        for medida in ["MAE", "RMSE", "SMAPE"]:
            assert np.isclose(metricas_src[medida], metricas_verif[medida])
