import pandas as pd

# Cargar el archivo
df = pd.read_csv('iteration_log.csv')

# Definimos las columnas de métricas (ignorando 'iter' y 'status')
metric_columns = [col for col in df.columns if col not in ['iter', 'status']]

# 1. Promedio total de cada columna (de todas las iteraciones)
promedios_por_modelo = df[metric_columns].mean()

# 2. Si quieres el promedio global por tipo de métrica (ej. promedio de todos los R2)
r2_cols = [c for c in df.columns if 'R2' in c]
mae_cols = [c for c in df.columns if 'MAE' in c]
rmse_cols = [c for c in df.columns if 'RMSE' in c]
mape_cols = [c for c in df.columns if 'MAPE' in c]

resumen_global = {
    "R2 Promedio Global": df[r2_cols].mean().mean(),
    "MAE Promedio Global": df[mae_cols].mean().mean(),
    "RMSE Promedio Global": df[rmse_cols].mean().mean(),
    "MAPE Promedio Global": df[mape_cols].mean().mean()
}

print("--- Promedio por cada modelo y métrica ---")
print(promedios_por_modelo)

print("\n--- Resumen Global de Métricas ---")
for metrica, valor in resumen_global.items():
    print(f"{metrica}: {valor:.4f}")
