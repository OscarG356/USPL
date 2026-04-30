import pandas as pd

# 1. Cargar el log de iteraciones con el formato estándar (separado por comas)
df_log = pd.read_csv('iteration_log.csv')

# 2. Filtrar solo las iteraciones exitosas (por seguridad)
if 'status' in df_log.columns:
    df_validos = df_log[df_log['status'] == 'ok'].copy()
else:
    df_validos = df_log.copy()

# 3. Identificar las columnas de las métricas
columnas_modelos = ['R2_SVR', 'R2_RF', 'R2_Bayesian', 'R2_XGB']

# Asegurarnos de que las columnas sean numéricas
for col in columnas_modelos:
    df_validos[col] = pd.to_numeric(df_validos[col], errors='coerce')

# 4. Calcular la media y la desviación estándar
estadisticas = pd.DataFrame({
    'Modelo': [col.replace('R2_', '') for col in columnas_modelos],
    'Media (R2)': df_validos[columnas_modelos].mean().values,
    'Std (Desviación)': df_validos[columnas_modelos].std().values
})

# 5. Imprimir los resultados
print("\n=== Resumen Estadístico de Modelos (R2) ===")
print(estadisticas.round(4).to_string(index=False))

# Guardar en CSV (lo guardamos en estándar para evitar problemas futuros)
estadisticas.to_csv('resumen_estadistico_modelos.csv', index=False)
print("\n[INFO] Resumen guardado en 'resumen_estadistico_modelos.csv'")