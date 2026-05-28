# =============================================================================
# Script: procesar_features.py
# Descripción: Agrupa features por valor de corriente y calcula estadísticos
#              (min, max, mean) para cada feature. Exporta el resultado a CSV.
# Autor: Desarrollador Senior Python
# =============================================================================

import pandas as pd


# =============================================================================
# >>> ZONA DE CONFIGURACIÓN
# =============================================================================

RUTA_ENTRADA = "data/data_USPL_2/processed/extracted_features.csv"       # Ruta del archivo CSV de entrada
RUTA_SALIDA  = "data/data_USPL_2/processed/datos_procesados.csv"    # Ruta del archivo CSV de salida
COLUMNA_OBJETIVO = "target_current"      # Nombre de la columna de agrupación
N_BINS = 4
# =============================================================================
# >>> FIN DE ZONA DE CONFIGURACIÓN <<<
# =============================================================================


def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el archivo CSV desde la ruta indicada.

    Args:
        ruta (str): Ruta al archivo CSV de entrada.

    Returns:
        pd.DataFrame: DataFrame con los datos cargados.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada.
    """
    try:
        df = pd.read_csv(ruta)
        print(f"[OK] Archivo cargado: {ruta}")
        print(f"     Filas: {df.shape[0]} | Columnas: {df.shape[1]}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")


def validar_columna_objetivo(df: pd.DataFrame, columna_objetivo: str) -> None:
    """
    Verifica que la columna objetivo exista en el DataFrame.

    Args:
        df (pd.DataFrame): DataFrame cargado.
        columna_objetivo (str): Nombre de la columna objetivo.

    Raises:
        ValueError: Si la columna no existe en el DataFrame.
    """
    if columna_objetivo not in df.columns:
        raise ValueError(
            f"La columna '{columna_objetivo}' no existe en el archivo. "
            f"Columnas disponibles: {list(df.columns)}"
        )
    print(f"[OK] Columna objetivo encontrada: '{columna_objetivo}'")


def obtener_features(df: pd.DataFrame, columna_objetivo: str) -> list:
    """
    Retorna la lista de columnas de features, excluyendo la columna objetivo.

    Args:
        df (pd.DataFrame): DataFrame completo.
        columna_objetivo (str): Columna a excluir.

    Returns:
        list: Lista de nombres de columnas que son features.
    """
    features = [col for col in df.columns if col != columna_objetivo]
    print(f"[OK] Features detectadas ({len(features)}): {features}")
    return features


def asignar_cuartiles(df: pd.DataFrame, columna_objetivo: str, n_bins: int) -> pd.DataFrame:
    """
    Asigna cada fila a un bin basado en la distribución cuantílica de la columna
    objetivo, usando pandas.qcut. Agrega la columna 'cuartil_corriente' al DataFrame.

    Se usa duplicates='drop' para manejar casos donde los límites de los bins
    coinciden (distribuciones con valores repetidos o concentrados).

    Args:
        df (pd.DataFrame): DataFrame con los datos originales.
        columna_objetivo (str): Columna numérica sobre la que se aplica qcut.
        n_bins (int): Número de bins cuantílicos a generar.

    Returns:
        pd.DataFrame: Copia del DataFrame con la columna 'cuartil_corriente' añadida.

    Raises:
        ValueError: Si tras eliminar duplicados quedan menos de 2 bins válidos.
    """
    df = df.copy()

    try:
        df["cuartil_corriente"] = pd.qcut(
            df[columna_objetivo],
            q=n_bins,
            duplicates="drop"   # Evita error si hay límites de bins duplicados
        )
    except ValueError as e:
        raise ValueError(
            f"No se pudo aplicar qcut sobre '{columna_objetivo}' con {n_bins} bins. "
            f"Detalle: {e}"
        )

    bins_resultantes = df["cuartil_corriente"].nunique()
    print(f"[OK] Cuartiles asignados.")
    print(f"     Bins solicitados: {n_bins} | Bins generados: {bins_resultantes}")
    if bins_resultantes < n_bins:
        print(f"     [AVISO] Se generaron menos bins por valores duplicados en los límites.")

    return df


def calcular_estadisticos(df: pd.DataFrame, columna_objetivo: str, features: list) -> pd.DataFrame:
    """
    Agrupa el DataFrame por la columna 'cuartil_corriente' (bins cuantílicos)
    y calcula min, max y mean para cada feature. Los NaN se ignoran en el cálculo.
    Añade también el conteo de muestras por cuartil.

    Args:
        df (pd.DataFrame): DataFrame con los datos, debe incluir 'cuartil_corriente'.
        columna_objetivo (str): Nombre original de la columna de corriente
                                (se usa para el log; la agrupación es por cuartil).
        features (list): Lista de features a procesar.

    Returns:
        pd.DataFrame: DataFrame con una fila por cuartil y columnas de estadísticos.
    """
    COLUMNA_BIN = "cuartil_corriente"

    print(f"\n[INFO] Calculando estadísticos agrupando por cuartiles de '{columna_objetivo}'...")

    # Agrupar por bin cuantílico y calcular los tres estadísticos
    # skipna=True (comportamiento por defecto) ignora los NaN automáticamente
    agrupado = df.groupby(COLUMNA_BIN, observed=True)[features].agg(["min", "max", "mean"])

    # Aplanar el índice jerárquico de columnas → feature_min, feature_max, feature_mean
    agrupado.columns = [f"{feature}_{stat}" for feature, stat in agrupado.columns]

    # Añadir columna con el número de muestras por cuartil (ignora NaN en el conteo)
    agrupado["n_muestras"] = df.groupby(COLUMNA_BIN, observed=True)[features[0]].count()

    # Resetear el índice para que el intervalo del cuartil quede como columna normal
    agrupado = agrupado.reset_index()

    # Renombrar la columna de bin a algo más descriptivo en el CSV de salida
    agrupado = agrupado.rename(columns={COLUMNA_BIN: "rango_corriente"})

    print(f"[OK] Estadísticos calculados.")
    print(f"     Cuartiles procesados: {agrupado['rango_corriente'].nunique()}")
    print(f"     Columnas generadas: {len(agrupado.columns) - 2} estadísticos "
          f"+ rango_corriente + n_muestras")

    return agrupado


def exportar_resultado(df_resultado: pd.DataFrame, ruta: str) -> None:
    """
    Exporta el DataFrame resultante a un archivo CSV.

    Args:
        df_resultado (pd.DataFrame): DataFrame procesado.
        ruta (str): Ruta donde se guardará el CSV de salida.
    """
    df_resultado.to_csv(ruta, index=False)
    print(f"\n[OK] Archivo exportado exitosamente: {ruta}")


def procesar_pipeline(ruta_entrada: str, ruta_salida: str, columna_objetivo: str, n_bins: int = 4) -> pd.DataFrame:
    """
    Ejecuta el pipeline completo: carga, validación, binning cuantílico,
    cálculo de estadísticos y exportación.

    Args:
        ruta_entrada (str): Ruta del CSV de entrada.
        ruta_salida (str): Ruta del CSV de salida.
        columna_objetivo (str): Nombre de la columna de agrupación (corriente).
        n_bins (int): Número de bins cuantílicos. Por defecto 4 (cuartiles).

    Returns:
        pd.DataFrame: DataFrame final con los estadísticos calculados.
    """
    print("=" * 60)
    print("  INICIO DEL PIPELINE DE PROCESAMIENTO")
    print("=" * 60)

    # Paso 1: Cargar datos
    df = cargar_datos(ruta_entrada)

    # Paso 2: Validar columna objetivo
    validar_columna_objetivo(df, columna_objetivo)

    # Paso 3: Obtener lista de features
    features = obtener_features(df, columna_objetivo)

    # Paso 4: Asignar cuartiles con qcut (NUEVO - reemplaza agrupación por valor exacto)
    df = asignar_cuartiles(df, columna_objetivo, n_bins)

    # Paso 5: Calcular estadísticos agrupados por cuartil
    df_resultado = calcular_estadisticos(df, columna_objetivo, features)

    # Paso 6: Exportar resultado
    exportar_resultado(df_resultado, ruta_salida)

    print("\n" + "=" * 60)
    print("  PIPELINE FINALIZADO CORRECTAMENTE")
    print("=" * 60)

    return df_resultado


# =============================================================================
# EJEMPLO DE USO
# =============================================================================

if __name__ == "__main__":

    # Ejecutar el pipeline con los parámetros definidos en la zona de configuración
    df_final = procesar_pipeline(
        ruta_entrada=RUTA_ENTRADA,
        ruta_salida=RUTA_SALIDA,
        columna_objetivo=COLUMNA_OBJETIVO,
        n_bins=N_BINS
    )

    # Mostrar una vista previa del resultado en consola
    print("\nVista previa del DataFrame resultante:")
    print(df_final.head())
    print(f"\nDimensiones finales: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")