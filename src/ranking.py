import pandas as pd

def procesar_ranking_general(file_path, col_rank, nombre_metrica):
    df = pd.read_csv(file_path)
    
    # 1. Agrupar por feature y modelo para obtener el promedio de los rankings
    # Esto consolida todas las iteraciones en un solo valor medio
    df_grouped = df.groupby(['feature', 'model'])[col_rank].mean().reset_index()
    
    # 2. Pivotar para tener los modelos como columnas
    df_pivot = df_grouped.pivot(index='feature', columns='model', values=col_rank)
    
    # 3. Renombrar columnas: rank_perm_svr, rank_perm_xgb, etc.
    df_pivot.columns = [f'rank_{nombre_metrica}_{col.lower()}' for col in df_pivot.columns]
    
    return df_pivot

# Procesar ambos archivos
resumen_perm = procesar_ranking_general('ranking_consenso_perm.csv', 'rank_perm', 'perm')
resumen_shap = procesar_ranking_general('ranking_consenso_shap.csv', 'rank_shap', 'shap')

# Unir los dos análisis (Permutation y SHAP) por la columna 'feature'
df_final = pd.merge(resumen_perm, resumen_shap, on='feature', how='outer')

# 4. Limitar a máximo 3 decimales
df_final = df_final.round(3)

# Guardar el resultado final
df_final.to_csv('ranking_general_por_modelo.csv')

print("Archivo 'ranking_general_por_modelo.csv' generado con éxito.")
print(df_final.head())
