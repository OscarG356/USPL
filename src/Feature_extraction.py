# -*- coding: utf-8 -*-
"""
Extracción y Preprocesamiento de Características de Señales de Osciloscopio

Este script procesa los archivos CSV de señales del osciloscopio, realiza un escalamiento MinMax global,
segmenta cada pulso en subsistemas, calcula la frecuencia de muestreo (fs), extrae características
utilizando TSFEL, limpia la redundancia por varianza/correlación y guarda el archivo resultante en:
data/data_USPL_<ID>/processed/extracted_features.csv
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# pyrefly: ignore [missing-import]
import tsfel


def natural_sort_key(s: str) -> list:
    """Clave para ordenamiento natural de nombres de archivos (ej. SENAL1, SENAL2, SENAL10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_and_preprocess_signals(data_folder: str):
    """
    Carga, ordena, escala y segmenta las señales de los archivos CSV en data_folder,
    y carga los valores objetivo de corriente desde el archivo Excel.
    """
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"No se encontró la carpeta de datos raw: {data_folder}")

    # 1. Eliminar archivos duplicados (que terminan en (1).CSV)
    files_to_delete = [f for f in os.listdir(data_folder) if f.endswith('(1).CSV')]
    for f in files_to_delete:
        file_p = os.path.join(data_folder, f)
        os.remove(file_p)
        print(f"[CLEANUP] Eliminado archivo duplicado: {f}")

    # 2. Filtrar y ordenar archivos CSV
    signal_files = [f for f in os.listdir(data_folder) if f.lower().endswith('.csv')]
    signal_files.sort(key=natural_sort_key)
    print(f"[INFO] Se encontraron {len(signal_files)} archivos CSV en '{data_folder}'")

    if len(signal_files) == 0:
        raise ValueError(f"No hay archivos CSV en la carpeta: {data_folder}")

    # 3. Leer la intensidad de cada archivo
    all_signals_data = []
    for file_name in signal_files:
        file_path = os.path.join(data_folder, file_name)
        signal_df = pd.read_csv(file_path, header=0)
        # La columna de intensidad se asume en el índice 1
        signal = signal_df.iloc[:, 1].astype(float)
        all_signals_data.append(signal)

    # Filtrar desde la muestra 191 según la metodología establecida
    all_signals_data = all_signals_data[191:]
    signals_matrix = pd.DataFrame(all_signals_data)
    print(f"[INFO] Matriz de señales seleccionadas (desde índice 191): {signals_matrix.shape}")

    # 4. Escalamiento MinMax Global (-1 a 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    signals_array = signals_matrix.values
    signals_scaled_reshaped = scaler.fit_transform(signals_array.reshape(-1, 1))
    signals_matrix_norm_array = signals_scaled_reshaped.reshape(signals_array.shape)
    signals_matrix_norm = pd.DataFrame(signals_matrix_norm_array, columns=signals_matrix.columns)

    # 5. Calcular frecuencia de muestreo (fs) a partir del primer archivo
    example_file_path = os.path.join(data_folder, signal_files[0])
    example_df = pd.read_csv(example_file_path, header=0)
    dt_s = example_df.iloc[1, 0] - example_df.iloc[0, 0]
    fs = 1 / dt_s
    print(f"[INFO] Intervalo dt: {dt_s:.2e} s | Frecuencia de muestreo fs: {fs:.2e} Hz")

    # 6. Cargar etiquetas objetivo (Corriente en mA)
    excel_path = os.path.join(data_folder, 'Datos-Corriente.xlsx')
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"No se encontró el archivo de etiquetas: {excel_path}")

    labels_df = pd.read_excel(excel_path)
    t_Corriente_raw = labels_df['Corriente (mA)'].values[191:]
    print(f"[INFO] Total de etiquetas cargadas: {len(t_Corriente_raw)}")

    # 7. Segmentación (2 segmentos de longitud 300 por pulso)
    segmented_signals_list = []
    segmented_labels_list = []
    segment_length = 300
    num_segments = 2

    for idx, (_, signal_series) in enumerate(signals_matrix_norm.iterrows()):
        arr = signal_series.values
        corriente = t_Corriente_raw[idx]
        if len(arr) >= num_segments * segment_length:
            for i in range(num_segments):
                start_index = i * segment_length
                end_index = start_index + segment_length
                segment = arr[start_index:end_index]
                segmented_signals_list.append(segment)
                segmented_labels_list.append(corriente)

    segmented_signals_df = pd.DataFrame(segmented_signals_list)
    t_Corriente = np.array(segmented_labels_list)
    print(f"[INFO] Matriz de señales segmentadas: {segmented_signals_df.shape} | Labels: {t_Corriente.shape}")

    return segmented_signals_df, t_Corriente, fs


def clean_features(df: pd.DataFrame, threshold_corr: float = 0.90, epsilon: float = 1e-10) -> pd.DataFrame:
    """
    Limpia características redundantemente correlacionadas o con varianza casi nula.
    """
    print(f"[CLEAN] Dimensiones antes de limpiar: {df.shape}")

    # 1. Filtro por varianza mínima
    variances = df.var()
    to_drop_var = variances[variances <= epsilon].index.tolist()
    df_var_cleaned = df.drop(columns=to_drop_var)

    # 2. Manejo de NaNs
    df_var_cleaned = df_var_cleaned.dropna(axis=1, how='all')

    # 3. Filtro por correlación (matriz triangular superior)
    corr_matrix = df_var_cleaned.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] >= threshold_corr)]

    df_final = df_var_cleaned.drop(columns=to_drop_corr)

    print(f"[CLEAN] Eliminadas por varianza <= {epsilon}: {len(to_drop_var)}")
    print(f"[CLEAN] Eliminadas por alta correlación (>= {threshold_corr}): {len(to_drop_corr)}")
    print(f"[CLEAN] Dimensiones finales tras limpieza: {df_final.shape}")

    return df_final


def extract_and_clean_features(segmented_signals_df: pd.DataFrame, t_Corriente: np.ndarray, fs: float, threshold_corr: float = 0.90) -> pd.DataFrame:
    """
    Extrae características de TSFEL para cada señal segmentada y las filtra.
    """
    print("[INFO] Obteniendo configuración predefinida de TSFEL...")
    cfg = tsfel.get_features_by_domain()

    all_extracted_features = []
    total_signals = len(segmented_signals_df)

    print(f"[INFO] Extrayendo características TSFEL para {total_signals} muestras de señal...")
    for idx, (_, signal_series) in enumerate(segmented_signals_df.iterrows()):
        features_for_signal = tsfel.time_series_features_extractor(cfg, signal_series.values, fs=fs, verbose=0)
        all_extracted_features.append(features_for_signal)

    features_df = pd.concat(all_extracted_features, ignore_index=True)
    print(f"[INFO] Matriz de características crudas extraídas: {features_df.shape}")

    # Limpieza de características
    df_limpio = clean_features(features_df, threshold_corr=threshold_corr)
    df_limpio['target_current'] = t_Corriente

    return df_limpio


def main():
    parser = argparse.ArgumentParser(description="Script de extracción de características TSFEL para datos USPL.")
    parser.add_argument("--uspl", type=int, default=2, help="ID del experimento USPL (1 o 2). Por defecto: 2")
    parser.add_argument("--threshold_corr", type=float, default=0.90, help="Umbral de correlación para descartar características redundantes. Por defecto: 0.90")
    parser.add_argument("--raw_dir", type=str, default=None, help="Ruta personalizada para la carpeta de datos raw.")
    parser.add_argument("--output_path", type=str, default=None, help="Ruta personalizada para guardar extracted_features.csv.")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Configuración de rutas predeterminadas
    uspl_id = f"USPL_{args.uspl}"
    raw_folder = args.raw_dir or os.path.join(base_dir, "data", f"data_{uspl_id}", "raw", "osciloscopio")
    output_path = args.output_path or os.path.join(base_dir, "data", f"data_{uspl_id}", "processed", "extracted_features.csv")

    print(f"============================================================")
    print(f"  EXTRACCIÓN DE CARACTERÍSTICAS - {uspl_id}")
    print(f"============================================================")
    print(f"Carpeta de entrada (raw):  {raw_folder}")
    print(f"Archivo de salida:          {output_path}")

    # Cargar y preprocesar
    segmented_signals_df, t_Corriente, fs = load_and_preprocess_signals(raw_folder)

    # Extraer y limpiar características
    features_df = extract_and_clean_features(segmented_signals_df, t_Corriente, fs, threshold_corr=args.threshold_corr)

    # Guardar resultado en processed/
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    features_df.to_csv(output_path, index=False)
    print(f"\n[OK] Características guardadas exitosamente en: {output_path}")
    print(f"[OK] Dimensiones del dataset final: {features_df.shape}")


if __name__ == "__main__":
    main()