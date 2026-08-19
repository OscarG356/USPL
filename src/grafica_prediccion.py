# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 1. Cargar los datos
df = pd.read_csv('/home/oagr/Documentos/Gita/USPL/data/data_USPL_2/outputs/run_20260818_113505_loco_chunks/model_predictions_history.csv')

# --- CONFIGURACIÓN DE FILTROS ---
# Cambia estos valores para ajustar la resolución del eje Y
#MIN_Current = float(input("Ingrese el valor mínimo de corriente para visualizar (ej. 0): "))
#MAX_Current = float(input("Ingrese el valor máximo de corriente para visualizar (ej. 400): "))
MIN_Current = 160.0
MAX_Current = 400.0    
# --------------------------------

def visualizar_predicciones(data, min_c, max_c):
    # 2. Filtrar por rango de corriente para eliminar atípicos
    filtered_df = data[(data['Actual_Current'] >= min_c) & (data['Actual_Current'] <= max_c)].copy()
    
    # 3. Reestructurar datos (Melt) para que Seaborn pueda graficar varios modelos a la vez
    # Pasamos de formato ancho a formato largo
    df_melted = filtered_df.melt(
        id_vars=['repetition', 'Actual_Current'], 
        value_vars=['Pred_SVR', 'Pred_Random Forest', 'Pred_Bayesian Ridge', 'Pred_XGBoost'],
        var_name='Model', 
        value_name='Prediction'
    )
    df_melted['Model'] = df_melted['Model'].replace({
        'Pred_SVR': 'SVR',
        'Pred_Random Forest': 'RFR',
        'Pred_Bayesian Ridge': 'BR',
        'Pred_XGBoost': 'XGBoost'
    })

    # 4. Configuración de la estética
    plt.figure(figsize=(8, 4))
    sns.set_style("whitegrid")

    # 5. Graficar con Intervalo de Confianza (CI 95%)
    # Seaborn calcula el CI automáticamente si hay múltiples valores para un mismo punto X
    sns.lineplot(
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
    plt.xlim(min_c, max_c)
    plt.ylim(min_c, max_c)

    # Personalización
    #plt.title(f'Model Comparison (Range: {min_c} - {max_c} Current)', fontsize=14)
    # Aumentamos fontsize (ej. de 12 a 14) y añadimos weight='bold'
    plt.xlabel('Actual $I_p$ (mA)', fontsize=14, weight='bold')
    plt.ylabel('Predicted $G_p$ (mA)', fontsize=14, weight='bold')

    plt.tick_params(axis='both', labelsize=12)

    # Para la leyenda, usamos prop para el texto y title_fontsize para el título
    plt.legend(
        title='Models / Reference', 
        loc='lower right', 
        prop={'size': 12, 'weight': 'bold'},  # Texto de la leyenda en negrita y tamaño 12
        title_fontsize=13                     # Título de la leyenda un poco más grande
    )

    # Ponemos el título de la leyenda en negrita (opcional, requiere una línea extra)
    plt.gca().get_legend().get_title().set_weight('bold')

    plt.tight_layout()    
    # Preguntar si guardar en PDF
    #save_pdf = input("¿Desea guardar la gráfica en PDF? (s/n): ").strip().lower()
    save_pdf = 's'  # Cambia a 's' para guardar automáticamente sin preguntar
    if save_pdf == 's':
        default_name = "grafica_prediccion.pdf"
        #out_path = input(f"Nombre de archivo de salida (ej. {default_name}): ").strip()
        out_path = ""  # Cambia a "" para usar el nombre por defecto sin preguntar
        if out_path == "":
            out_path = default_name
        try:
            plt.savefig(out_path, format='pdf', bbox_inches='tight')
            print(f"Gráfica guardada en: {out_path}")
        except Exception as e:  # noqa: BLE001
            print(f"No se pudo guardar el archivo PDF: {e}")

    plt.show()

# Ejecutar la función
visualizar_predicciones(df, MIN_Current, MAX_Current)