import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE RUTAS
# ══════════════════════════════════════════════════════════════

def setup_paths(uspl_id, run_id=None):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_laser_dir = os.path.join(base_dir, "data", f"data_USPL_{uspl_id}", "outputs")
    
    if run_id is None:
        runs = sorted([d for d in os.listdir(data_laser_dir) if d.startswith("run_")])
        if not runs: raise FileNotFoundError(f"No hay ejecuciones en {data_laser_dir}")
        run_id = runs[-1]
    
    run_path = os.path.join(data_laser_dir, run_id)
    report_path = os.path.join(base_dir, "reports", f"reports_USPL_{uspl_id}", "figures", run_id)
    os.makedirs(report_path, exist_ok=True)
    
    return {
        "run_dir": run_path,
        "perm_csv": os.path.join(run_path, "ranking_consenso_perm.csv"),
        "shap_csv": os.path.join(run_path, "ranking_consenso_shap.csv"),
        "output_csv": os.path.join(run_path, "tabla_consenso_final.csv")
    }

# ══════════════════════════════════════════════════════════════
# 2. LÓGICA DE CÁLCULO TRIPLE 80% (PERM, SHAP, GENERAL)
# ══════════════════════════════════════════════════════════════

def process_triple_80(paths):
    print(f"--- Calculando Triple Consenso 80% en: {os.path.basename(paths['run_dir'])} ---")
    
    if not os.path.exists(paths['perm_csv']) or not os.path.exists(paths['shap_csv']):
        print("[ERROR] Faltan archivos base.")
        return

    # Cargar
    df_p = pd.read_csv(paths['perm_csv'])
    df_s = pd.read_csv(paths['shap_csv'])

    # Normalización Min-Max (0-1) por modelo
    def normalize(df, col):
        df = df.copy()
        for mod in df['model'].unique():
            mask = df['model'] == mod
            vals = df.loc[mask, col]
            df.loc[mask, col] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
        return df

    df_p = normalize(df_p, 'importance_mean')
    df_s = normalize(df_s, 'shap_importance')

    # 1. Agrupar por Feature (Consensos base)
    c_perm = df_p.groupby('feature')['importance_mean'].mean().reset_index()
    c_perm.rename(columns={'importance_mean': 'Consenso_Permutation'}, inplace=True)

    c_shap = df_s.groupby('feature')['shap_importance'].mean().reset_index()
    c_shap.rename(columns={'shap_importance': 'Consenso_SHAP'}, inplace=True)

    # Unir todo en una tabla maestra
    df = pd.merge(c_perm, c_shap, on='feature')
    df['Consenso_General'] = (df['Consenso_Permutation'] + df['Consenso_SHAP']) / 2

    # --- FUNCIÓN PARA MARCAR EL 80% ---
    def get_80_stats(df_target, col_name):
        # Ordenar descendente
        temp = df_target[['feature', col_name]].sort_values(col_name, ascending=False).copy()
        # Peso relativo y acumulado
        total = temp[col_name].sum()
        temp[f'Acum_{col_name}'] = (temp[col_name] / total).cumsum()
        # Marcar Top 80 (True si el acumulado ANTERIOR era menor a 0.8)
        temp[f'Top80_{col_name}'] = temp[f'Acum_{col_name}'].shift(1).fillna(0) < 0.8
        return temp[['feature', f'Acum_{col_name}', f'Top80_{col_name}']]

    # Aplicar a los tres pilares
    stats_perm = get_80_stats(df, 'Consenso_Permutation')
    stats_shap = get_80_stats(df, 'Consenso_SHAP')
    stats_gen  = get_80_stats(df, 'Consenso_General')

    # Mergear las estadísticas de vuelta a la tabla principal
    df = df.merge(stats_perm, on='feature')\
           .merge(stats_shap, on='feature')\
           .merge(stats_gen, on='feature')

    # Ordenar la tabla final por el Consenso General para que sea legible
    df = df.sort_values('Consenso_General', ascending=False).round(4)

    # Guardar
    df.to_csv(paths['output_csv'], index=False)
    print(f"[OK] Tabla generada con éxito en: {paths['output_csv']}")
    
    # Resumen rápido por consola
    n_perm = df['Top80_Consenso_Permutation'].sum()
    n_shap = df['Top80_Consenso_SHAP'].sum()
    n_gen  = df['Top80_Consenso_General'].sum()
    print(f"--- RESUMEN TOP 80% ---")
    print(f"Features en Permutation: {int(n_perm)}")
    print(f"Features en SHAP:        {int(n_shap)}")
    print(f"Features en General:     {int(n_gen)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--uspl', type=int, required=True)
    parser.add_argument('--run', type=str, default=None)
    args = parser.parse_args()

    try:
        rutas = setup_paths(args.uspl, args.run)
        process_triple_80(rutas)
    except Exception as e:
        print(f"[ERROR] {e}")