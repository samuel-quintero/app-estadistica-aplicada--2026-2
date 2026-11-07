"""Ajuste y comparación AICc de modelos semiempíricos de secado en capa delgada."""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# Se definen las cinco ecuaciones, parámetros iniciales y límites razonables de ajuste.
def _lewis(t, k):
    """Modelo de Lewis: MR = exp(-k·t)."""
    return np.exp(-k * t)


def _henderson_pabis(t, a, k):
    """Modelo Henderson-Pabis: MR = a·exp(-k·t)."""
    return a * np.exp(-k * t)


def _page(t, k, n):
    """Modelo de Page: MR = exp(-k·t^n)."""
    return np.exp(-k * np.power(t, n))


def _logaritmico(t, a, k, c):
    """Modelo logarítmico: MR = a·exp(-k·t) + c."""
    return a * np.exp(-k * t) + c


def _dos_terminos(t, a, k0, b, k1):
    """Modelo de dos términos: MR = a·exp(-k0·t) + b·exp(-k1·t)."""
    return a * np.exp(-k0 * t) + b * np.exp(-k1 * t)


# Se guardan iniciales y cotas para mantener parámetros de secado en rangos plausibles.
MODELOS = {
    "Lewis": {
        "funcion": _lewis,
        "parametros": ("k",),
        "p0": (0.005,),
        "bounds": ((0.0,), (1.0,)),
    },
    "Henderson-Pabis": {
        "funcion": _henderson_pabis,
        "parametros": ("a", "k"),
        "p0": (1.0, 0.005),
        "bounds": ((0.0, 0.0), (2.0, 1.0)),
    },
    "Page": {
        "funcion": _page,
        "parametros": ("k", "n"),
        "p0": (0.002, 1.2),
        "bounds": ((1e-12, 0.1), (1.0, 3.0)),
    },
    "Logarítmico": {
        "funcion": _logaritmico,
        "parametros": ("a", "k", "c"),
        "p0": (0.9, 0.005, 0.05),
        "bounds": ((0.0, 0.0, -1.0), (2.0, 1.0, 1.0)),
    },
    "Dos términos": {
        "funcion": _dos_terminos,
        "parametros": ("a", "k0", "b", "k1"),
        "p0": (0.6, 0.003, 0.4, 0.01),
        "bounds": ((0.0, 0.0, 0.0, 0.0), (2.0, 1.0, 2.0, 1.0)),
    },
}


def _validar_datos(t, MR) -> tuple[np.ndarray, np.ndarray]:
    """Convierte y valida tiempos y razones de humedad finitos de igual longitud."""
    # Se normalizan ambas entradas como vectores float para scipy.optimize.
    tiempos = np.asarray(t, dtype=float).reshape(-1)
    razones = np.asarray(MR, dtype=float).reshape(-1)
    if tiempos.size != razones.size:
        raise ValueError("t y MR deben tener la misma cantidad de observaciones.")

    # Se excluyen pares no finitos sin perder la correspondencia tiempo-humedad.
    validos = np.isfinite(tiempos) & np.isfinite(razones)
    tiempos = tiempos[validos]
    razones = razones[validos]
    if tiempos.size == 0:
        raise ValueError("Se requiere al menos un par finito de tiempo y MR.")
    return tiempos, razones


def ajustar_curva(t, MR, nombre: str) -> np.ndarray:
    """Ajusta por mínimos cuadrados no lineales el modelo elegido con curve_fit."""
    # Se valida el nombre y se obtienen observaciones finitas antes del ajuste.
    if nombre not in MODELOS:
        raise ValueError(f"Modelo desconocido. Elige uno de: {', '.join(MODELOS)}.")
    tiempos, razones = _validar_datos(t, MR)
    configuracion = MODELOS[nombre]

    # curve_fit busca parámetros que minimizan la suma de cuadrados de los residuos.
    parametros, _ = curve_fit(
        configuracion["funcion"],
        tiempos,
        razones,
        p0=configuracion["p0"],
        bounds=configuracion["bounds"],
        maxfev=20000,
    )
    return parametros


def metricas_curva(MR, MR_ajustado, k: int) -> dict:
    """Calcula R², RMSE, χ² reducido, AIC y AICc para una curva ajustada."""
    # Se verifica que las observaciones y predicciones sean vectores comparables.
    observados = np.asarray(MR, dtype=float).reshape(-1)
    ajustados = np.asarray(MR_ajustado, dtype=float).reshape(-1)
    if observados.size != ajustados.size:
        raise ValueError("MR y MR_ajustado deben tener la misma cantidad de valores.")
    validos = np.isfinite(observados) & np.isfinite(ajustados)
    observados = observados[validos]
    ajustados = ajustados[validos]
    n = int(observados.size)
    if n == 0:
        raise ValueError("Se requiere al menos una pareja finita para calcular métricas.")
    if k < 0:
        raise ValueError("k debe ser un entero no negativo.")

    # La SCE cuantifica el error total y SST el error respecto de la media observada.
    residuos = observados - ajustados
    sce = float(np.sum(residuos**2))
    suma_total = float(np.sum((observados - np.mean(observados)) ** 2))
    r_cuadrado = 1 - sce / suma_total if suma_total > 0 else np.nan
    rmse = float(np.sqrt(sce / n))

    # Sin varianzas de medición por punto, se reporta χ² reducido como SCE/(n-k).
    chi_cuadrado_reducido = sce / (n - k) if n > k else np.nan

    # AIC = n·ln(SCE/n)+2k bajo errores gaussianos; se define -inf si SCE=0.
    if sce == 0:
        aic = -np.inf
    else:
        aic = n * np.log(sce / n) + 2 * k

    # AICc corrige la muestra finita y no está definido cuando n <= k+1.
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1) if n > k + 1 else np.nan
    return {
        "n": n,
        "k": int(k),
        "SCE": sce,
        "R²": float(r_cuadrado),
        "RMSE": rmse,
        "χ² reducido": float(chi_cuadrado_reducido),
        "AIC": float(aic),
        "AICc": float(aicc),
    }


def tabla_modelos(t, MR) -> pd.DataFrame:
    """Ajusta cada modelo, reúne sus parámetros y métricas, y ordena por AICc."""
    # Se validan los datos una vez y se preparan filas para todas las ecuaciones.
    tiempos, razones = _validar_datos(t, MR)
    filas = []

    # Cada curva se ajusta con sus parámetros y se evalúa sobre los mismos datos.
    for nombre, configuracion in MODELOS.items():
        parametros = ajustar_curva(tiempos, razones, nombre)
        predicciones = configuracion["funcion"](tiempos, *parametros)
        k = len(configuracion["parametros"])
        metricas = metricas_curva(razones, predicciones, k)
        texto_parametros = ", ".join(
            f"{clave}={valor:.6g}"
            for clave, valor in zip(configuracion["parametros"], parametros)
        )

        # Se conserva una fila auditable con el nombre del modelo y sus resultados.
        filas.append(
            {
                "Modelo": nombre,
                "Parámetros": texto_parametros,
                **metricas,
            }
        )

    # AICc menor indica mejor soporte relativo entre modelos ajustados a los mismos datos.
    return pd.DataFrame(filas).sort_values("AICc", na_position="last").reset_index(drop=True)