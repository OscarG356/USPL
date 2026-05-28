import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Cargar los datos
df = pd.read_csv('/home/oagr/Documentos/Gita/USPL/data/data_USPL_2/outputs/run_20260505_183625/model_predictions_history.csv')

# --- CONFIGURACIÓN DE FILTROS ---
# Cambia estos valores para ajustar la resolución del eje Y
MIN_CURRENT = float(input("Ingrese el valor mínimo de corriente para visualizar (ej. 0): "))
MAX_CURRENT = float(input("Ingrese el valor máximo de corriente para visualizar (ej. 400): "))
# --------------------------------

def visualizar_predicciones(data, min_c, max_c):
    # 2. Filtrar por rango de corriente para eliminar atípicos
    filtered_df = data[(data['Actual_Current'] >= min_c) & (data['Actual_Current'] <= max_c)].copy()
    
    # 3. Reestructurar datos (Melt) para que Seaborn pueda graficar varios modelos a la vez
    # Pasamos de formato ancho a formato largo
    df_melted = filtered_df.melt(
        id_vars=['Iteration', 'Actual_Current'], 
        value_vars=['Pred_SVR', 'Pred_RF', 'Pred_Bayesian', 'Pred_XGBoost'],
        var_name='Model', 
        value_name='Prediction'
    )

    # 4. Configuración de la estética
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    # 5. Graficar con Intervalo de Confianza (CI 95%)
    # Seaborn calcula el CI automáticamente si hay múltiples valores para un mismo punto X
    plot = sns.lineplot(
        data=df_melted, 
        x='Actual_Current', 
        y='Prediction', 
        hue='Model', 
        errorbar=('ci', 95), # Esto genera la sombra del 95% de confianza
        alpha=0.8
    )

    # 6. Plot the "Ideal" reference line (Actual vs Actual)
    plt.plot(filtered_df['Actual_Current'], filtered_df['Actual_Current'], 
             color='black', linestyle='--', label='Actual (Ideal)', linewidth=1)

    # Personalización
    plt.title(f'Model Comparison (Range: {min_c} - {max_c} Current)', fontsize=14)
    plt.xlabel('Actual Current', fontsize=12)
    plt.ylabel('Prediction', fontsize=12)
    plt.legend(title='Models / Reference', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.show()

# Ejecutar la función
visualizar_predicciones(df, MIN_CURRENT, MAX_CURRENT)