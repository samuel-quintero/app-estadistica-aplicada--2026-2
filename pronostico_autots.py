"""Ejecuta AutoTS como un ensayo de evaluación y selección de variedades."""

# json permite leer la ficha del estudiante y guardar los parámetros elegidos por AutoTS.
import json

# Path construye rutas de archivos portables y crea la carpeta de resultados.
from pathlib import Path

# Se importa la carga de precipitación y el orquestador AutoTS del módulo de series.
from src.series import cargar_serie, ejecutar_autots


# La raíz del proyecto se obtiene desde este script, como ubicar el campo de ensayo.
RAIZ_PROYECTO = Path(__file__).resolve().parent

# La ficha entrega el código que identifica la semilla del experimento reproducible.
RUTA_METADATOS = RAIZ_PROYECTO / "datos" / "metadatos.json"

# Se abre la ficha en UTF-8 para conservar los caracteres del resto de metadatos.
with RUTA_METADATOS.open("r", encoding="utf-8") as archivo_metadatos:
    METADATOS = json.load(archivo_metadatos)

# Se transforma el código en semilla y se limita a cuatro dígitos como pidió el protocolo.
SEMILLA = int(METADATOS["codigo"]) % 10000

# Se toma la precipitación mensual de 2015 a 2025 como las observaciones del ensayo.
PRECIPITACION = cargar_serie("precipitacion_mm")

# Se crea la carpeta donde se guardarán los pronósticos, resultados y ficha ganadora.
CARPETA_RESULTADOS = RAIZ_PROYECTO / "resultados"
CARPETA_RESULTADOS.mkdir(parents=True, exist_ok=True)

# Se registran los controles de AutoTS: horizonte, generaciones, validaciones e intervalo.
PARAMETROS_BUSQUEDA = {
    "horizonte": 12,  # Equivale a evaluar la variedad durante los siguientes doce meses.
    "model_list": "superfast",  # Limita el catálogo a variedades de crecimiento rápido.
    "generaciones": 5,  # Número de rondas de selección y cruza de configuraciones candidatas.
    "validaciones": 2,  # Número de bloques temporales usados como parcelas de evaluación.
    "intervalo": 0.9,  # Cobertura nominal de los límites inferior y superior del pronóstico.
    "semilla": SEMILLA,  # Hace reproducible la selección y evaluación del ensayo.
}

# AutoTS prueba modelos de pronóstico como variedades bajo las mismas condiciones temporales.
RESULTADO = ejecutar_autots(
    PRECIPITACION,  # La precipitación mensual representa la variable de respuesta del ensayo.
    horizonte=PARAMETROS_BUSQUEDA["horizonte"],  # Evalúa el desempeño durante un año futuro.
    model_list=PARAMETROS_BUSQUEDA["model_list"],  # Escoge el catálogo superfast de candidatos.
    generaciones=PARAMETROS_BUSQUEDA["generaciones"],  # Realiza cinco rondas de selección genética.
    validaciones=PARAMETROS_BUSQUEDA["validaciones"],  # Compara cada candidato en dos cortes temporales.
    intervalo=PARAMETROS_BUSQUEDA["intervalo"],  # Solicita bandas predictivas del noventa por ciento.
    semilla=PARAMETROS_BUSQUEDA["semilla"],  # Repite la misma búsqueda cuando se ejecute de nuevo.
)

# Se asignan nombres de salida para que los productos del ensayo sean fáciles de localizar.
RUTA_PRONOSTICO = CARPETA_RESULTADOS / "autots_pronostico.csv"
RUTA_TOP10 = CARPETA_RESULTADOS / "autots_top10_modelos.csv"
RUTA_MEJOR_MODELO = CARPETA_RESULTADOS / "autots_mejor_modelo.json"

# El CSV del pronóstico conserva fecha, valor central y límites de incertidumbre.
RESULTADO["pronostico"].to_csv(RUTA_PRONOSTICO, index=False)

# El top 10 conserva métricas y puntuación para comparar variedades finalistas.
RESULTADO["top10"].to_csv(RUTA_TOP10, index=False)

# El JSON registra ganador, parámetros, transformaciones y condiciones del experimento.
MEJOR_MODELO = {
    **RESULTADO["mejor_modelo"],
    "semilla": SEMILLA,
    "parametros_busqueda": PARAMETROS_BUSQUEDA,
}

# default=str convierte valores numpy u objetos de configuración a texto serializable.
with RUTA_MEJOR_MODELO.open("w", encoding="utf-8") as archivo_mejor_modelo:
    json.dump(MEJOR_MODELO, archivo_mejor_modelo, ensure_ascii=False, indent=2, default=str)

# Se busca SeasonalNaive entre los diez finalistas, el equivalente a revisar una variedad testigo.
MASCARA_SEASONAL_NAIVE = RESULTADO["top10"]["Model"].astype(str).str.contains(
    "SeasonalNaive",
    case=False,
    regex=False,
)

# Se convierte su posición de índice a puesto comenzando desde uno.
PUESTOS_SEASONAL_NAIVE = RESULTADO["top10"].index[MASCARA_SEASONAL_NAIVE].tolist()

# Se informa si la variedad SeasonalNaive clasificó y cuál fue su posición.
if PUESTOS_SEASONAL_NAIVE:
    MENSAJE_SEASONAL_NAIVE = f"Puesto de SeasonalNaive en el top 10: {PUESTOS_SEASONAL_NAIVE[0] + 1}."
else:
    MENSAJE_SEASONAL_NAIVE = "SeasonalNaive no figura entre los diez primeros modelos."

# Se imprime la variedad ganadora y el resultado del contraste con el método estacional.
print(f"Semilla del ensayo: {SEMILLA}")
print(f"Modelo ganador: {MEJOR_MODELO['nombre']}")
print(MENSAJE_SEASONAL_NAIVE)
print("Top 10 por Score:")
print(RESULTADO["top10"].to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))
print("Archivos guardados:")
print(RUTA_PRONOSTICO.relative_to(RAIZ_PROYECTO))
print(RUTA_TOP10.relative_to(RAIZ_PROYECTO))
print(RUTA_MEJOR_MODELO.relative_to(RAIZ_PROYECTO))
