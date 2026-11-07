"""Genera conjuntos simulados reproducibles para la app de Estadística Aplicada."""

from hashlib import md5
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Identidad usada para personalizar la semilla y los metadatos de esta ejecución.
CODIGO = "1034285837"
NOMBRE = "Samuel David Quintero Palencia"


def generar_parcelas(rng, cultivo, rendimiento_base, n_parcelas):
    """Simula parcelas y rendimientos agrícolas con dos observaciones atípicas."""
    # Se generan las características de manejo, suelo y sitio de cada parcela.
    suelos = np.array(["Franco", "Arcilloso", "Arenoso"])
    suelo = rng.choice(suelos, size=n_parcelas, p=[0.48, 0.30, 0.22])
    riego = rng.choice(np.array(["Sí", "No"]), size=n_parcelas, p=[0.45, 0.55])
    dosis_n = rng.uniform(0, 180, size=n_parcelas)
    lamina_riego = rng.uniform(35, 120, size=n_parcelas)
    lamina_riego[riego == "No"] = 0.0
    materia_organica_completa = rng.uniform(1.0, 5.5, size=n_parcelas)
    ph = rng.uniform(4.8, 7.8, size=n_parcelas)

    # El rendimiento responde a nitrógeno, agua, materia orgánica, pH y suelo.
    efecto_suelo = {"Franco": 0.45, "Arcilloso": 0.15, "Arenoso": -0.45}
    respuesta_n = 0.030 * dosis_n - 0.000085 * dosis_n**2
    respuesta_agua = 0.009 * np.minimum(lamina_riego, 100)
    respuesta_mo = 0.48 * (materia_organica_completa - 2.8)
    respuesta_ph = -0.20 * (ph - 6.3) ** 2
    ruido = rng.normal(0, 0.48, size=n_parcelas)
    rendimiento = (
        rendimiento_base
        + respuesta_n
        + respuesta_agua
        + respuesta_mo
        + respuesta_ph
        + np.array([efecto_suelo[valor] for valor in suelo])
        + ruido
    )
    rendimiento = np.maximum(rendimiento, 0.1)

    # Se reservan dos registros extremos identificables para los ejercicios de EDA.
    rendimiento[0] = rendimiento_base + 12.0
    rendimiento[1] = max(0.1, rendimiento_base * 0.02)

    # Se ocultan entre tres y seis mediciones de materia orgánica sin afectar el cálculo.
    materia_organica = materia_organica_completa.copy()
    n_faltantes = int(rng.integers(3, 7))
    indices_faltantes = rng.choice(n_parcelas, size=n_faltantes, replace=False)
    materia_organica[indices_faltantes] = np.nan

    return pd.DataFrame(
        {
            "parcela": [f"P{indice:03d}" for indice in range(1, n_parcelas + 1)],
            "suelo": suelo,
            "riego": riego,
            "dosis_N": dosis_n,
            "lamina_riego": lamina_riego,
            "materia_organica": materia_organica,
            "ph": ph,
            "rendimiento": rendimiento,
        }
    )


def generar_clima(rng):
    """Simula precipitación mensual bimodal y temperatura con ciclo anual."""
    # Se construye una observación al inicio de cada mes entre 2015 y 2025.
    fechas = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    anio = fechas.year.to_numpy()
    mes = fechas.month.to_numpy()

    # Las medias mayores corresponden a las temporadas lluviosas del Caribe.
    medias_precipitacion = np.array(
        [28, 22, 38, 125, 175, 105, 78, 92, 175, 195, 145, 48], dtype=float
    )
    sigma_lluvia = 0.48
    precipitacion = rng.lognormal(
        mean=np.log(medias_precipitacion[mes - 1]) - sigma_lluvia**2 / 2,
        sigma=sigma_lluvia,
    )

    # La temperatura aumenta levemente por año y oscila a lo largo de los meses.
    ciclo_anual = 1.15 * np.cos(2 * np.pi * (mes - 4) / 12)
    tendencia = 0.025 * (anio - 2015)
    temperatura = 27.0 + ciclo_anual + tendencia + rng.normal(0, 0.35, len(fechas))

    return pd.DataFrame(
        {
            "fecha": fechas,
            "precipitacion_mm": precipitacion,
            "temperatura_c": temperatura,
        }
    )


def imprimir_muestra(titulo, muestra):
    """Imprime una muestra pequeña en formato tabular legible."""
    print(f"\n{titulo}")
    print(muestra.to_string(index=False))


def main():
    """Genera y guarda los datos simulados personalizados y sus muestras."""
    # La semilla y las selecciones categóricas se derivan de la identidad estudiantil.
    semilla = int(CODIGO) % 2**32
    rng = np.random.default_rng(semilla)
    cultivos = [
        ("Maíz", 5.5),
        ("Arroz", 5.0),
        ("Yuca", 12.0),
        ("Ñame", 10.0),
        ("Ajonjolí", 0.9),
        ("Plátano", 15.0),
        ("Frijol", 1.7),
        ("Sorgo", 3.2),
    ]
    municipios = [
        "Sincelejo",
        "Corozal",
        "Sampués",
        "Los Palmitos",
        "San Marcos",
        "Majagual",
        "Tolú",
        "Ovejas",
    ]
    cultivo, rendimiento_base = cultivos[semilla % len(cultivos)]
    municipio = municipios[(semilla // len(cultivos)) % len(municipios)]
    n_parcelas = 80 + semilla % 41

    # La carpeta de salida queda junto al script, independientemente del directorio actual.
    carpeta_salida = Path(__file__).resolve().parent / "datos"
    carpeta_salida.mkdir(parents=True, exist_ok=True)

    # Se generan y guardan las parcelas; la materia orgánica mantiene sus faltantes.
    parcelas = generar_parcelas(rng, cultivo, rendimiento_base, n_parcelas)
    parcelas.to_csv(carpeta_salida / "parcelas.csv", index=False)

    # Cada año tiene un máximo de 24 horas simulado con una distribución Gumbel.
    mu_gumbel = float(rng.uniform(70, 100))
    beta_gumbel = float(rng.uniform(12, 20))
    anios = np.arange(1986, 2026)
    precip_max = pd.DataFrame(
        {
            "anio": anios,
            "pmax24h_mm": rng.gumbel(mu_gumbel, beta_gumbel, size=len(anios)),
        }
    )
    precip_max.to_csv(carpeta_salida / "precip_max_anual.csv", index=False)

    # El clima mensual combina lluvias bimodales, ruido lognormal y ciclo térmico.
    clima = generar_clima(rng)
    clima.to_csv(carpeta_salida / "clima_mensual.csv", index=False)

    # Se simula el secado de Page y se limita el ruido para conservar MR entre cero y uno.
    tiempos = np.arange(0, 301, 15)
    k_page = float(rng.uniform(0.0015, 0.0040))
    n_page = float(rng.uniform(0.85, 1.20))
    mr_ideal = np.exp(-k_page * tiempos**n_page)
    mr = np.clip(mr_ideal + rng.normal(0, 0.012, size=len(tiempos)), 0, 1)
    mr[0] = 1.0
    secado = pd.DataFrame({"tiempo_min": tiempos, "MR": mr})
    secado.to_csv(carpeta_salida / "secado.csv", index=False)

    # La muestra EDA incluye el atípico alto y nueve parcelas regulares aleatorias.
    indices_regulares = np.arange(2, n_parcelas)
    indices_eda = np.concatenate(
        ([0], rng.choice(indices_regulares, size=9, replace=False))
    )
    muestra_eda = parcelas.iloc[indices_eda][
        ["parcela", "suelo", "riego", "rendimiento"]
    ].sample(frac=1, random_state=semilla).reset_index(drop=True)
    muestra_eda.to_csv(carpeta_salida / "muestra_eda10.csv", index=False)

    # La muestra de distribución toma diez años al azar y se ordena cronológicamente.
    indices_dist = rng.choice(len(precip_max), size=10, replace=False)
    muestra_dist = precip_max.iloc[indices_dist].sort_values("anio").reset_index(drop=True)
    muestra_dist.to_csv(carpeta_salida / "muestra_dist10.csv", index=False)

    # Se excluyen los dos atípicos, se agrupan láminas positivas en ocho cuantiles,
    # y se selecciona una parcela aleatoria por grupo para cubrir el rango de riego.
    parcelas_sin_atipicos = parcelas.iloc[2:].copy()
    parcelas_con_riego = parcelas_sin_atipicos.loc[
        parcelas_sin_atipicos["lamina_riego"] > 0
    ].copy()
    parcelas_con_riego["grupo_lamina"] = pd.qcut(
        parcelas_con_riego["lamina_riego"], q=8, labels=False
    )
    indices_reg = [
        int(rng.choice(grupo.index.to_numpy()))
        for _, grupo in parcelas_con_riego.groupby("grupo_lamina", sort=True)
    ]
    muestra_reg = (
        parcelas.loc[indices_reg]
        .sort_values("lamina_riego")
        .reset_index(drop=True)
    )
    muestra_reg.to_csv(carpeta_salida / "muestra_reg8.csv", index=False)

    # La muestra temporal contiene 2024-2025, con 18 meses de entrenamiento y seis de validación.
    clima_ultimos_24 = clima.loc[clima["fecha"].dt.year >= 2024].copy().reset_index(drop=True)
    muestra_series = pd.DataFrame(
        {
            "periodo": np.arange(1, 25),
            "fecha": clima_ultimos_24["fecha"],
            "anio": clima_ultimos_24["fecha"].dt.year,
            "mes": clima_ultimos_24["fecha"].dt.month,
            "precipitacion_mm": clima_ultimos_24["precipitacion_mm"],
            "conjunto": ["entrenamiento"] * 18 + ["validacion"] * 6,
        }
    )
    muestra_series.to_csv(carpeta_salida / "muestra_series24.csv", index=False)

    # La huella identifica el código, el tamaño del conjunto y la suma de rendimientos.
    suma_rendimiento = float(parcelas["rendimiento"].sum())
    texto_huella = f"{CODIGO}|{n_parcelas}|{suma_rendimiento:.2f}"
    huella = md5(texto_huella.encode("utf-8")).hexdigest()[:8].upper()
    metadatos = {
        "codigo": CODIGO,
        "nombre": NOMBRE,
        "cultivo": cultivo,
        "municipio": municipio,
        "n_parcelas": n_parcelas,
        "huella": huella,
    }
    with (carpeta_salida / "metadatos.json").open("w", encoding="utf-8") as archivo:
        json.dump(metadatos, archivo, ensure_ascii=False, indent=2)

    # Se confirma la personalización y se muestran las cuatro muestras para revisión.
    print(f"Nombre: {NOMBRE}")
    print(f"Código: {CODIGO}")
    print(f"Cultivo: {cultivo}")
    print(f"Municipio: {municipio}")
    print(f"Huella: {huella}")
    imprimir_muestra("Muestra EDA (10 parcelas)", muestra_eda)
    imprimir_muestra("Muestra de distribución (10 años)", muestra_dist)
    imprimir_muestra("Muestra de regresión (8 parcelas)", muestra_reg)
    imprimir_muestra("Muestra de series (24 meses)", muestra_series)


if __name__ == "__main__":
    main()