"""Ajuste y evaluación de distribuciones para máximos anuales de precipitación."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# Se relacionan los nombres presentados al estudiante con las distribuciones SciPy.
CANDIDATAS = {
    "Normal": stats.norm,
    "Lognormal": stats.lognorm,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Gumbel": stats.gumbel_r,
    "GEV": stats.genextreme,
}

# Estas distribuciones usan soporte positivo y ubicación fijada en cero.
PARAMETROS_FIJOS = {
    "Lognormal": {"loc": 0.0},
    "Gamma": {"loc": 0.0},
    "Weibull": {"loc": 0.0},
}

# En scipy.stats.genextreme el parámetro c tiene signo opuesto al parámetro GEV ξ.


def _datos_validos(datos) -> np.ndarray:
    """Convierte la entrada en un vector numérico finito y valida que no esté vacío."""
    # Se convierten valores numéricos y se excluyen faltantes e infinitos.
    valores = pd.to_numeric(pd.Series(datos), errors="coerce").to_numpy(dtype=float)
    valores = valores[np.isfinite(valores)]
    if valores.size == 0:
        raise ValueError("Se requiere al menos una observación numérica finita.")
    return valores


def _distribucion(nombre: str):
    """Obtiene una candidata o informa los nombres admitidos."""
    # Se evita continuar con un nombre que no corresponda a una candidata definida.
    if nombre not in CANDIDATAS:
        raise ValueError(f"Modelo desconocido. Elige uno de: {', '.join(CANDIDATAS)}.")
    return CANDIDATAS[nombre]


def _nombres_parametros(nombre: str) -> list[str]:
    """Devuelve los nombres de forma, ubicación y escala en el orden de SciPy."""
    # SciPy entrega primero parámetros de forma y al final loc y scale.
    distribucion = _distribucion(nombre)
    nombres_forma = (
        [parte.strip() for parte in distribucion.shapes.split(",")]
        if distribucion.shapes
        else []
    )
    return nombres_forma + ["loc", "scale"]


def _cantidad_parametros_libres(nombre: str) -> int:
    """Cuenta parámetros estimados, sin incluir ubicaciones fijadas."""
    # k incluye cada parámetro de forma, scale y loc solo cuando no está fijado.
    nombres = _nombres_parametros(nombre)
    cantidad = len(nombres)
    if "loc" in PARAMETROS_FIJOS.get(nombre, {}):
        cantidad -= 1
    return cantidad


def _parametros_para_prueba(nombre: str, params: tuple) -> tuple[dict, dict]:
    """Separa parámetros ajustados y conocidos según la interfaz de SciPy."""
    # Se asocian los parámetros posicionales del ajuste a los nombres de SciPy.
    nombres = _nombres_parametros(nombre)
    parametros_ajustados = dict(zip(nombres, params))

    # Los parámetros fijos se mantienen conocidos en cada muestra Monte Carlo.
    parametros_conocidos = PARAMETROS_FIJOS.get(nombre, {}).copy()
    parametros_estimados = {
        clave: valor
        for clave, valor in parametros_ajustados.items()
        if clave not in parametros_conocidos
    }
    return parametros_conocidos, parametros_estimados


def ajustar(datos, nombre: str) -> tuple[tuple, int]:
    """Ajusta por máxima verosimilitud y devuelve (parámetros SciPy, k libre)."""
    # Se valida el modelo y se extraen solo las observaciones válidas.
    distribucion = _distribucion(nombre)
    valores = _datos_validos(datos)

    # Lognormal, Gamma y Weibull requieren precipitación estrictamente positiva.
    if nombre in PARAMETROS_FIJOS and np.any(valores <= 0):
        raise ValueError(f"{nombre} con loc=0 requiere datos estrictamente positivos.")

    # La estimación MLE fija loc=0 solo para los modelos indicados en la configuración.
    opciones_fit = {"floc": 0.0} if nombre in PARAMETROS_FIJOS else {}
    parametros = tuple(distribucion.fit(valores, **opciones_fit))
    cantidad_libres = _cantidad_parametros_libres(nombre)
    return parametros, cantidad_libres


def log_verosimilitud(datos, nombre: str, params: tuple) -> float:
    """Calcula ℓ = suma(log(pdf(x))) para los datos y parámetros indicados."""
    # Se evalúa la densidad de probabilidad en cada observación y se suman sus logaritmos.
    distribucion = _distribucion(nombre)
    valores = _datos_validos(datos)
    return float(np.sum(distribucion.logpdf(valores, *params)))


def criterios_informacion(datos, nombre: str, params: tuple) -> dict:
    """Calcula AIC, BIC y AICc usando el número de parámetros libres k."""
    # Se obtienen n, k y log-verosimilitud para aplicar las fórmulas solicitadas.
    valores = _datos_validos(datos)
    n = int(valores.size)
    k = _cantidad_parametros_libres(nombre)
    log_likelihood = log_verosimilitud(valores, nombre, params)

    # AIC penaliza k; BIC usa k·ln(n); ambos parten de menos dos veces ℓ.
    aic = 2 * k - 2 * log_likelihood
    bic = k * math.log(n) - 2 * log_likelihood

    # AICc añade 2k(k+1)/(n-k-1) y no está definido si n <= k+1.
    aicc = (
        aic + (2 * k * (k + 1)) / (n - k - 1)
        if n > k + 1
        else math.nan
    )
    return {
        "n": n,
        "k": k,
        "log_verosimilitud": log_likelihood,
        "AIC": float(aic),
        "BIC": float(bic),
        "AICc": float(aicc),
    }


def _bondad_con_parametros(
    valores: np.ndarray,
    nombre: str,
    params: tuple,
    n_mc: int,
    rng: np.random.Generator,
) -> dict:
    """Ejecuta KS/AD Monte Carlo con reestimación y calcula KS ingenuo."""
    # Se separan parámetros conocidos de los estimados para que SciPy los reestime.
    distribucion = _distribucion(nombre)
    parametros_conocidos, parametros_estimados = _parametros_para_prueba(
        nombre, params
    )

    # goodness_of_fit vuelve a ajustar parámetros desconocidos en cada réplica.
    resultado_ks = stats.goodness_of_fit(
        distribucion,
        valores,
        known_params=parametros_conocidos or None,
        fit_params=parametros_estimados,
        statistic="ks",
        n_mc_samples=n_mc,
        rng=rng,
    )
    resultado_ad = stats.goodness_of_fit(
        distribucion,
        valores,
        known_params=parametros_conocidos or None,
        fit_params=parametros_estimados,
        statistic="ad",
        n_mc_samples=n_mc,
        rng=rng,
    )

    # Este p-valor ingenuo no corrige que los parámetros se estimaron con la muestra.
    resultado_ks_ingenuo = stats.kstest(valores, distribucion.cdf, args=params)
    return {
        "KS": float(resultado_ks.statistic),
        "p_KS_MC": float(resultado_ks.pvalue),
        "KS_ingenuo": float(resultado_ks_ingenuo.statistic),
        "p_KS_ingenuo": float(resultado_ks_ingenuo.pvalue),
        "AD": float(resultado_ad.statistic),
        "p_AD_MC": float(resultado_ad.pvalue),
    }


def bondad_ajuste(datos, nombre: str, n_mc: int) -> dict:
    """Devuelve KS y AD corregidos por Monte Carlo y el p-valor KS ingenuo."""
    # Se exige al menos una réplica para estimar la distribución nula Monte Carlo.
    if n_mc < 1:
        raise ValueError("n_mc debe ser un entero positivo.")

    # Se ajusta el modelo observado y se usa una semilla fija para reproducibilidad.
    valores = _datos_validos(datos)
    params, _ = ajustar(valores, nombre)
    rng = np.random.default_rng(2026)
    return _bondad_con_parametros(valores, nombre, params, int(n_mc), rng)


def _interpretar_delta_aic(delta: float) -> str:
    """Traduce ΔAIC a una interpretación relativa de soporte del modelo."""
    # Se aplican rangos habituales y se nombran explícitamente los intervalos intermedios.
    if delta < 2:
        return "Soporte sustancial"
    if delta < 4:
        return "Soporte débil"
    if delta <= 7:
        return "Menor soporte"
    if delta <= 10:
        return "Soporte muy débil"
    return "Sin soporte"


def tabla_comparativa(datos, n_mc: int = 199) -> pd.DataFrame:
    """Compara candidatas con MLE, información, pruebas y decisión al 5 %."""
    # Se valida n_mc una vez y se preparan datos y generador reproducible compartido.
    if n_mc < 1:
        raise ValueError("n_mc debe ser un entero positivo.")
    valores = _datos_validos(datos)
    rng = np.random.default_rng(2026)
    filas = []

    # Para cada candidata se calculan parámetros, verosimilitud y criterios de información.
    for nombre in CANDIDATAS:
        params, k = ajustar(valores, nombre)
        criterios = criterios_informacion(valores, nombre, params)
        pruebas = _bondad_con_parametros(valores, nombre, params, int(n_mc), rng)
        nombres = _nombres_parametros(nombre)
        parametros_texto = ", ".join(
            f"{clave}={valor:.4f}" for clave, valor in zip(nombres, params)
        )

        # Se rechaza H0 si cualquiera de las pruebas corregidas tiene p < 0.05.
        pruebas_rechazadas = [
            prueba
            for prueba, p_valor in (
                ("KS", pruebas["p_KS_MC"]),
                ("AD", pruebas["p_AD_MC"]),
            )
            if p_valor < 0.05
        ]
        decision = (
            f"Sí ({' y '.join(pruebas_rechazadas)})"
            if pruebas_rechazadas
            else "No"
        )

        # Se conserva una fila con parámetros, ajuste, diagnóstico y valores p.
        filas.append(
            {
                "Modelo": nombre,
                "Parámetros": parametros_texto,
                "k": k,
                "ℓ": criterios["log_verosimilitud"],
                "AIC": criterios["AIC"],
                "BIC": criterios["BIC"],
                "AICc": criterios["AICc"],
                "KS": pruebas["KS"],
                "p KS (Monte Carlo)": pruebas["p_KS_MC"],
                "KS ingenuo": pruebas["KS_ingenuo"],
                "p KS ingenuo": pruebas["p_KS_ingenuo"],
                "AD": pruebas["AD"],
                "p AD (Monte Carlo)": pruebas["p_AD_MC"],
                "¿Se rechaza H0? (5 %)": decision,
            }
        )

    # Se mide la diferencia de AIC respecto del mínimo y se interpreta cada diferencia.
    tabla = pd.DataFrame(filas)
    tabla["ΔAIC"] = tabla["AIC"] - tabla["AIC"].min()
    tabla["Interpretación de ΔAIC"] = tabla["ΔAIC"].map(_interpretar_delta_aic)
    return tabla.sort_values("AIC").reset_index(drop=True)


def posiciones_graficas(datos) -> pd.DataFrame:
    """Calcula posiciones Weibull y Gringorten y sus periodos empíricos de retorno."""
    # Se ordena la muestra de menor a mayor y se numeran observaciones desde uno.
    valores = np.sort(_datos_validos(datos))
    n = valores.size
    orden = np.arange(1, n + 1)

    # Weibull usa i/(n+1); Gringorten usa (i-0.44)/(n+0.12).
    posicion_weibull = orden / (n + 1)
    posicion_gringorten = (orden - 0.44) / (n + 0.12)

    # El periodo de retorno empírico se deriva de la probabilidad de no excedencia p.
    return pd.DataFrame(
        {
            "pmax24h_mm": valores,
            "i": orden,
            "p Weibull": posicion_weibull,
            "T Weibull (años)": 1 / (1 - posicion_weibull),
            "p Gringorten": posicion_gringorten,
            "T Gringorten (años)": 1 / (1 - posicion_gringorten),
        }
    )


def cuantil_retorno(nombre: str, params: tuple, T: float) -> float:
    """Calcula x_T = F⁻¹(1-1/T) para un periodo de retorno T mayor que un año."""
    # Se verifica que el periodo genere una probabilidad estrictamente entre cero y uno.
    distribucion = _distribucion(nombre)
    if not np.isfinite(T) or T <= 1:
        raise ValueError("El periodo de retorno T debe ser finito y mayor que 1.")

    # Se devuelve el cuantil cuya probabilidad acumulada es 1 - 1/T.
    probabilidad = 1 - 1 / T
    return float(distribucion.ppf(probabilidad, *params))


def riesgo_falla(T: float, N: int) -> float:
    """Calcula R = 1-(1-1/T)^N para el riesgo de al menos una excedencia."""
    # Se validan el periodo de retorno y el horizonte entero de años.
    if not np.isfinite(T) or T <= 1:
        raise ValueError("El periodo de retorno T debe ser finito y mayor que 1.")
    if not isinstance(N, (int, np.integer)) or N < 0:
        raise ValueError("N debe ser un entero no negativo.")

    # La expresión acumula la probabilidad anual de excedencia bajo independencia.
    return float(1 - (1 - 1 / T) ** N)


def bootstrap_cuantil(
    datos,
    nombre: str,
    T: float,
    B: int = 500,
    semilla: int = 42,
) -> tuple[float, float]:
    """Estima el IC del 95 % del cuantil de retorno mediante bootstrap paramétrico."""
    # Se valida el número de réplicas y se ajusta el modelo a la muestra observada.
    if B < 2:
        raise ValueError("B debe ser al menos 2 para estimar un intervalo.")
    valores = _datos_validos(datos)
    params, _ = ajustar(valores, nombre)
    distribucion = _distribucion(nombre)
    rng = np.random.default_rng(semilla)
    cuantiles = []

    # Cada réplica se simula del modelo ajustado, se vuelve a ajustar y calcula x_T.
    for _ in range(B):
        try:
            simulacion = distribucion.rvs(
                *params,
                size=valores.size,
                random_state=rng,
            )
            parametros_bootstrap, _ = ajustar(simulacion, nombre)
            cuantiles.append(cuantil_retorno(nombre, parametros_bootstrap, T))
        except (ValueError, RuntimeError, FloatingPointError):
            # Se omiten réplicas que no convergen, sin alterar los datos originales.
            continue

    # Se exige una proporción mínima de réplicas válidas para evitar un IC inestable.
    if len(cuantiles) < math.ceil(0.8 * B):
        raise RuntimeError(
            f"Solo {len(cuantiles)} de {B} réplicas bootstrap produjeron un ajuste válido."
        )

    # Los percentiles 2.5 y 97.5 forman el intervalo percentil del 95 %.
    inferior, superior = np.quantile(cuantiles, [0.025, 0.975])
    return float(inferior), float(superior)


def gumbel_momentos(datos) -> dict:
    """Estima μ y β de Gumbel por momentos con s muestral y Euler γ≈0.5772."""
    # Se requiere variabilidad estimable para calcular la escala por momentos.
    valores = _datos_validos(datos)
    if valores.size < 2:
        raise ValueError("Se requieren al menos dos observaciones para calcular s muestral.")

    # β = s·√6/π convierte la desviación estándar muestral a escala Gumbel.
    desviacion_muestral = float(np.std(valores, ddof=1))
    beta = desviacion_muestral * math.sqrt(6) / math.pi

    # μ = media - 0.5772·β centra la distribución mediante la constante de Euler.
    mu = float(np.mean(valores) - 0.5772 * beta)
    return {"mu": mu, "beta": float(beta)}


if __name__ == "__main__":
    # Se carga la serie de máximos anuales relativa a la ubicación de este módulo.
    ruta_datos = Path(__file__).resolve().parents[1] / "datos" / "precip_max_anual.csv"
    serie_precipitacion = pd.read_csv(ruta_datos)["pmax24h_mm"]

    # Se imprimen las candidatas ordenadas por AIC con 199 réplicas Monte Carlo.
    comparacion = tabla_comparativa(serie_precipitacion, n_mc=199)
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(comparacion.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))