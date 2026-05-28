
# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────

import gc
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import shap
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance


# ══════════════════════════════════════════════════════════════
# 0. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description='Entrenamiento Temporal OPNET')
parser.add_argument('--uspl', type=int, default=1,
                    help='ID del láser (1 o 2)')
# Argumento añadido: permite apuntar consenso_score a un run previo.
# No afecta al comportamiento original; el run actual sigue siendo TIMESTAMP.
parser.add_argument('--run',  type=str, default=None,
                    help='(Opcional) ID de un run previo para recalcular '
                         'consenso_score sin reentrenar. Ej: run_20240101_120000')
# Nuevo: régimen de operación (pulse o sc)
parser.add_argument('--regimen', type=str, choices=['pulse', 'sc'], default='pulse',
                help='Regimen de operación: Pulsado (pulse) o supercontinuo (sc)')
# Si regimen=sc, elegir modo: edfa o curr
parser.add_argument('--sc_mode', type=str, choices=['edfa', 'curr'], default='curr',
                help='Si regimen=sc, elegir modo de operación: EDFA (edfa) o Corriente (curr)')
args = parser.parse_args()

USPL_ID = f"USPL_{args.uspl}"   # "USPL_1" o "USPL_2"

# Runtime options from CLI
REGIMEN = args.regimen
SC_MODE = args.sc_mode if REGIMEN == 'sc' else None

BASE_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_DIR          = os.path.join(BASE_DIR, "data", f"data_{USPL_ID}", "outputs", f"run_{TIMESTAMP}")
DIR_REPORTS_FIGS = os.path.join(BASE_DIR, "reports", f"reports_{USPL_ID}", "figures", f"run_{TIMESTAMP}")
#PATH_FEATURES    = os.path.join(BASE_DIR, "data", f"data_{USPL_ID}", "processed", "extracted_features.csv")
if REGIMEN == 'pulse':
    PATH_FEATURES = os.path.join(BASE_DIR, "data", f"data_{USPL_ID}", "processed", "extracted_features.csv")
elif REGIMEN == 'sc' and SC_MODE == 'edfa':
    PATH_FEATURES = os.path.join(BASE_DIR, "data", f"data_{USPL_ID}", "processed", "SC", "Var_EDFA", "extracted_features.csv")
elif REGIMEN == 'sc' and SC_MODE == 'curr':
    PATH_FEATURES = os.path.join(BASE_DIR, "data", f"data_{USPL_ID}", "processed", "SC", "Var_Cur", "extracted_features.csv")

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(DIR_REPORTS_FIGS, exist_ok=True)

# ── Parámetro adicional para el pipeline de estadísticos descriptivos ────────
N_BINS_ESTADISTICOS = 4   # Bins cuantílicos para valorespromedio_features

print(f"--- Iniciando ejecución para: {USPL_ID} ---")
print(f"--- Salidas en: {RUN_DIR} ---")
print(f"--- Regimen: {REGIMEN} ---")
if SC_MODE is not None:
    print(f"--- SC Mode: {SC_MODE} ---")


# ══════════════════════════════════════════════════════════════
# 1. FUNCIONES DE APOYO
# ══════════════════════════════════════════════════════════════

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def save_to_run(df, filename):
    path = os.path.join(RUN_DIR, filename)
    if os.path.exists(path):
        df_old = pd.read_csv(path)
        pd.concat([df_old, df], ignore_index=True).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════
# 2. PIPELINE DE ESTADÍSTICOS POR FEATURE
# ══════════════════════════════════════════════════════════════

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
            duplicates="drop"
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
        columna_objetivo (str): Nombre original de la columna de corriente.
        features (list): Lista de features a procesar.

    Returns:
        pd.DataFrame: DataFrame con una fila por cuartil y columnas de estadísticos.
    """
    COLUMNA_BIN = "cuartil_corriente"
    print(f"\n[INFO] Calculando estadísticos agrupando por cuartiles de '{columna_objetivo}'...")
    agrupado = df.groupby(COLUMNA_BIN, observed=True)[features].agg(["min", "max", "mean"])
    agrupado.columns = [f"{feature}_{stat}" for feature, stat in agrupado.columns]
    agrupado["n_samples"] = df.groupby(COLUMNA_BIN, observed=True)[features[0]].count()
    agrupado = agrupado.reset_index()
    agrupado = agrupado.rename(columns={COLUMNA_BIN: "current_range"})
    print(f"[OK] Estadísticos calculados.")
    print(f"     Cuartiles procesados: {agrupado['current_range'].nunique()}")
    print(f"     Columnas generadas: {len(agrupado.columns) - 2} estadísticos "
        f"+ current_range + n_samples")
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


def procesar_pipeline(ruta_entrada: str, ruta_salida: str,
                      columna_objetivo: str, n_bins: int = 4) -> pd.DataFrame:
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
    print("  INICIO DEL PIPELINE DE ESTADÍSTICOS (valorespromedio)")
    print("=" * 60)
    df = cargar_datos(ruta_entrada)
    validar_columna_objetivo(df, columna_objetivo)
    features = obtener_features(df, columna_objetivo)
    df = asignar_cuartiles(df, columna_objetivo, n_bins)
    df_resultado = calcular_estadisticos(df, columna_objetivo, features)
    exportar_resultado(df_resultado, ruta_salida)
    print("\n" + "=" * 60)
    print("  PIPELINE DE ESTADÍSTICOS FINALIZADO")
    print("=" * 60)
    return df_resultado


# ══════════════════════════════════════════════════════════════
# 3. TRIPLE CONSENSO 80% (PERM + SHAP + GENERAL)
# ══════════════════════════════════════════════════════════════

def process_triple_80(paths: dict) -> None:
    """
    Calcula el consenso Triple 80% sobre importancias de Permutation y SHAP.

    Normaliza por modelo (Min-Max), agrega en consenso cross-model y marca
    las features que acumulan el 80% de la importancia en cada pilar.

    Args:
        paths (dict): Diccionario con claves:
            - 'run_dir'   : str — directorio base del run actual.
            - 'perm_csv'  : str — CSV con ranking de permutation importance.
            - 'shap_csv'  : str — CSV con ranking SHAP importance.
            - 'output_csv': str — ruta de salida de la tabla de consenso final.
    """
    print(f"\n--- Calculando Triple Consenso 80% en: {os.path.basename(paths['run_dir'])} ---")

    if not os.path.exists(paths['perm_csv']) or not os.path.exists(paths['shap_csv']):
        print("[ERROR] Faltan archivos base para consenso_score. "
              "Verifica que el loop de entrenamiento completó al menos 1 iteración.")
        return

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

    # Consensos base: media cross-model por feature
    c_perm = df_p.groupby('feature')['importance_mean'].mean().reset_index()
    c_perm.rename(columns={'importance_mean': 'Consensus_Permutation'}, inplace=True)

    c_shap = df_s.groupby('feature')['shap_importance'].mean().reset_index()
    c_shap.rename(columns={'shap_importance': 'Consensus_SHAP'}, inplace=True)

    df_consenso = pd.merge(c_perm, c_shap, on='feature')
    df_consenso['Consensus_General'] = (
        df_consenso['Consensus_Permutation'] + df_consenso['Consensus_SHAP']
    ) / 2

    # Marcar el Top 80% acumulado por pilar
    def get_80_stats(df_target, col_name):
        temp = df_target[['feature', col_name]].sort_values(col_name, ascending=False).copy()
        total = temp[col_name].sum()
        temp[f'Cum_{col_name}'] = (temp[col_name] / total).cumsum()
        temp[f'Top80_{col_name}'] = temp[f'Cum_{col_name}'].shift(1).fillna(0) < 0.8
        return temp[['feature', f'Cum_{col_name}', f'Top80_{col_name}']]

    stats_perm = get_80_stats(df_consenso, 'Consensus_Permutation')
    stats_shap = get_80_stats(df_consenso, 'Consensus_SHAP')
    stats_gen  = get_80_stats(df_consenso, 'Consensus_General')

    df_consenso = df_consenso \
        .merge(stats_perm, on='feature') \
        .merge(stats_shap, on='feature') \
        .merge(stats_gen,  on='feature')

    df_consenso = df_consenso.sort_values('Consensus_General', ascending=False).round(4)
    df_consenso.to_csv(paths['output_csv'], index=False)
    print(f"[OK] Consensus table generated at: {paths['output_csv']}")

    n_perm = df_consenso['Top80_Consensus_Permutation'].sum()
    n_shap = df_consenso['Top80_Consensus_SHAP'].sum()
    n_gen  = df_consenso['Top80_Consensus_General'].sum()
    print(f"--- RESUMEN TOP 80% ---")
    print(f"Features en Permutation: {int(n_perm)}")
    print(f"Features en SHAP:        {int(n_shap)}")
    print(f"Features en General:     {int(n_gen)}")


# ══════════════════════════════════════════════════════════════
# 4. CAPA DE ANÁLISIS Y GRÁFICAS
# ══════════════════════════════════════════════════════════════

def ejecutar_analisis_final():
    print(f"\n[PROCESO] Generando reportes finales...")
    
    # --- 2.1 Métricas (Media y Std) (Va a data/outputs/run...) ---
    log_path = os.path.join(RUN_DIR, 'iteration_log.csv')
    if os.path.exists(log_path):
        df_l = pd.read_csv(log_path)
        df_v = df_l[df_l['status'] == 'ok'] if 'status' in df_l.columns else df_l
        m_cols = [c for c in df_l.columns if any(m in c for m in ['R2', 'MAE', 'RMSE', 'MAPE'])]
        if m_cols:
            stats = pd.DataFrame({
                'Metric_Model': m_cols,
                'Mean': df_v[m_cols].mean().values,
                'Std': df_v[m_cols].std().values
            })
            stats.to_csv(os.path.join(RUN_DIR, 'resumen_estadistico_modelos.csv'), index=False)

    # --- 2.2 Ranking General con Aporte 80% (Va a data/outputs/run...) ---
    def calc_80(path, col_imp, suf):
        if not os.path.exists(path): return pd.DataFrame()
        df_g = pd.read_csv(path).groupby(['feature', 'model'])[col_imp].mean().reset_index()
        res = []
        for mod in df_g['model'].unique():
            dm = df_g[df_g['model'] == mod].copy().sort_values(col_imp, ascending=False)
            imp_norm = dm[col_imp].clip(lower=0)
            dm['acum'] = (imp_norm / (imp_norm.sum() + 1e-10)).cumsum()
            dm['top80'] = dm['acum'] <= 0.80
            if (dm['acum'] > 0.80).any(): dm.loc[(dm['acum'] > 0.80).idxmax(), 'top80'] = True
            m_id = mod.replace(' ', '_').lower()
            dm = dm.rename(columns={col_imp: f'{suf}_imp_{m_id}', 'acum': f'{suf}_acum_{m_id}', 'top80': f'{suf}_80_{m_id}'})
            res.append(dm.drop(columns=['model']))
        f = res[0]
        for i in range(1, len(res)): f = pd.merge(f, res[i], on='feature', how='outer')
        return f

    df_rank_p = calc_80(os.path.join(RUN_DIR, 'ranking_consenso_perm.csv'), 'importance_mean', 'perm')
    df_rank_s = calc_80(os.path.join(RUN_DIR, 'ranking_consenso_shap.csv'), 'shap_importance', 'shap')
    
    if not df_rank_p.empty and not df_rank_s.empty:
        final_rank = pd.merge(df_rank_p, df_rank_s, on='feature', how='outer').round(4)
        final_rank.to_csv(os.path.join(RUN_DIR, 'ranking_general_por_modelo.csv'), index=False)

    # --- 2.3 Convergencia Dual (SE DIBUJA DIRECTO EN reports/figures/) ---
    config_graficas = [
        ('ranking_consenso_perm.csv', 'rank_perm', 'importance_mean', 'PERMUTATION'),
        ('ranking_consenso_shap.csv', 'rank_shap', 'shap_importance', 'SHAP')
    ]

    for file_name, col_rank, col_imp, title in config_graficas:
        path = os.path.join(RUN_DIR, file_name)
        if not os.path.exists(path): continue
        
        df_r = pd.read_csv(path)
        n_f = df_r['feature'].nunique()
        n_m = df_r['model'].nunique()
        df_r['iteration'] = (df_r.index // (n_f * n_m)) + 1
        
        conv_list = []
        for n in range(1, df_r['iteration'].max() + 1):
            sub = df_r[df_r['iteration'] <= n].groupby('feature').agg({col_rank:'mean', col_imp:'mean'}).reset_index()
            sub['rank_abs'] = sub[col_rank].rank(method='min')
            sub['iteration'] = n
            conv_list.append(sub)
        
        df_plot = pd.concat(conv_list)
        feats = df_plot['feature'].unique()
        colors = cm.tab20(np.linspace(0, 1, len(feats)))
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle(f'Dual Convergence: {title}', fontsize=16, fontweight='bold')
        
        for f_name, color in zip(feats, colors):
            d = df_plot[df_plot['feature'] == f_name].sort_values('iteration')
            axes[0,0].plot(d['iteration'], d['rank_abs'], color=color, linewidth=2, label=f_name)
            axes[0,1].plot(d['iteration'], d['rank_abs'].diff().abs(), color=color, linewidth=1.5)
            axes[1,0].plot(d['iteration'], d[col_imp], color=color, linewidth=2)
            axes[1,1].plot(d['iteration'], d[col_imp].diff().abs(), color=color, linewidth=1.5)
        
        axes[0,0].set_title("Stability: Ranking Position"); axes[0,0].invert_yaxis(); axes[0,0].grid(alpha=0.3)
        axes[0,1].set_title("Rate of Change (Ranking)"); axes[0,1].grid(alpha=0.3)
        axes[1,0].set_title("Stability: Importance Magnitude"); axes[1,0].grid(alpha=0.3)
        axes[1,1].set_title("Rate of Change (Value)"); axes[1,1].grid(alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        img_filename = f'convergencia_{title.lower()}.png'
        dest_path = os.path.join(DIR_REPORTS_FIGS, img_filename)
        plt.savefig(dest_path, dpi=100)
        plt.close()
        print(f"[REPORTE] Gráfica generada en: {dest_path}")

    # --- 2.4 Consenso de Predicciones (SE DIBUJA DIRECTO EN reports/figures/) ---
    preds_path = os.path.join(RUN_DIR, 'model_predictions_history.csv')
    if os.path.exists(preds_path):
        df_preds = pd.read_csv(preds_path)
        df_melted = df_preds.melt(id_vars=['Actual_Current', 'Iteration'],
                                  var_name='Model', value_name='Predicted')

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.lineplot(data=df_melted, x='Actual_Current', y='Predicted',
                     hue='Model', errorbar=('ci', 95), linewidth=2, ax=ax)
        ax.plot([df_preds['Actual_Current'].min(), df_preds['Actual_Current'].max()],
                [df_preds['Actual_Current'].min(), df_preds['Actual_Current'].max()],
                color='black', linestyle='--', label='Ideal (Actual = Predicted)')
        
        iters_count = df_preds['Iteration'].nunique()
        ax.set_title(f'Prediction Consensus after {iters_count} Iterations (95% CI)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual Current')
        ax.set_ylabel('Predicted')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        img_cons_filename = 'consenso_predicciones.png'
        dest_cons_path = os.path.join(DIR_REPORTS_FIGS, img_cons_filename)
        plt.savefig(dest_cons_path, dpi=100)
        plt.close()
        print(f"[REPORTE] Gráfica de Consenso generada en: {dest_cons_path}")


# ══════════════════════════════════════════════════════════════
# 5. EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── PASO 0: Estadísticos descriptivos por cuartil de corriente ──────────
    # (valorespromedio_features.py integrado)
    # Se ejecuta antes del loop para tener contexto estadístico del dataset.
    # Si el archivo no existe, se aborta aquí (mismo comportamiento que temporal).
    if not os.path.exists(PATH_FEATURES):
        print(f"[ERROR] No existe el archivo de entrada en: {PATH_FEATURES}")
        exit()

    RUTA_SALIDA_STATS = os.path.join(RUN_DIR, "datos_procesados.csv")
    procesar_pipeline(
        ruta_entrada=PATH_FEATURES,
        ruta_salida=RUTA_SALIDA_STATS,
        columna_objetivo="target_current",
        n_bins=N_BINS_ESTADISTICOS
    )

    # ── NÚCLEO: Loop de Entrenamiento Temporal ───────────────────────────────
    # [ORIGINAL: temporallearning_opnet.py — INTACTO DESDE AQUÍ HASTA EL FIN DEL LOOP]

    df_raw = pd.read_csv(PATH_FEATURES)
    y = df_raw['target_current'].values
    X_df = df_raw.drop(columns=['target_current'])
    
    log = []
    TOTAL_ITERACIONES = int(input("Ingrese el número total de iteraciones a ejecutar: "))
    
    print(f"[INFO] Dataset con {X_df.shape[1]} features. Iniciando {TOTAL_ITERACIONES} iteraciones...")

    for i in range(TOTAL_ITERACIONES): 
        print(f"[ITER {i+1}/{TOTAL_ITERACIONES}] Procesando modelos...")
        try:
            X_train, X_test, y_train, y_test = train_test_split(X_df.values, y, test_size=0.2, stratify=pd.cut(y, bins=10))
            
            # --- Variables Globales de SHAP (Restauradas de tu código original) ---
            X_train_background = shap.sample(X_train, 100)
            X_test_global      = shap.sample(X_test, 75)
            
            # ══════════════════════════════════════════════════════
            # 1. SVR
            # ══════════════════════════════════════════════════════
            pipe_svr = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
            grid_svr = GridSearchCV(pipe_svr, {
                'svr__C':       [10, 100, 1000],
                'svr__gamma':   ['scale', 0.001, 0.005, 0.01],
                'svr__epsilon': [1, 2.5, 5]
            }, cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)
            
            p_svr = permutation_importance(grid_svr.best_estimator_, X_test, y_test, n_repeats=75, n_jobs=-1)
            df_p_svr = pd.DataFrame({'feature': X_df.columns, 'importance_mean': p_svr.importances_mean, 'model': 'SVR'}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
            df_p_svr['rank_perm'] = df_p_svr.index + 1
            
            # SHAP Original
            predict_fn = grid_svr.best_estimator_.predict
            explainer_svr = shap.KernelExplainer(predict_fn, X_train_background)
            shap_values_global = explainer_svr.shap_values(X_test_global)
            df_s_svr = pd.DataFrame({'feature': X_df.columns, 'shap_importance': np.mean(np.abs(shap_values_global), axis=0), 'model': 'SVR'}).sort_values('shap_importance', ascending=False).reset_index(drop=True)
            df_s_svr['rank_shap'] = df_s_svr.index + 1

            # ══════════════════════════════════════════════════════
            # 2. Random Forest
            # ══════════════════════════════════════════════════════
            pipe_rf = Pipeline([('rf', RandomForestRegressor(n_jobs=-1))])
            grid_rf = GridSearchCV(pipe_rf, {
                'rf__n_estimators':     [100, 300, 500],
                'rf__max_depth':        [5, 10, 15],
                'rf__min_samples_leaf': [4, 8, 16],
                'rf__max_features':     ['sqrt', 'log2']
            }, cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)
            
            p_rf = permutation_importance(grid_rf.best_estimator_, X_test, y_test, n_repeats=75, n_jobs=-1)
            df_p_rf = pd.DataFrame({'feature': X_df.columns, 'importance_mean': p_rf.importances_mean, 'model': 'Random Forest'}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
            df_p_rf['rank_perm'] = df_p_rf.index + 1
            
            # SHAP Original RF
            best_rf_model = grid_rf.best_estimator_.named_steps['rf']
            X_train_background_rf = shap.sample(X_train, 100)
            X_test_global_rf      = shap.sample(X_test, 75)
            
            explainer_rf = shap.TreeExplainer(best_rf_model, X_train_background_rf)
            shap_values_rf = explainer_rf.shap_values(X_test_global_rf)
            df_s_rf = pd.DataFrame({'feature': X_df.columns, 'shap_importance': np.mean(np.abs(shap_values_rf), axis=0), 'model': 'Random Forest'}).sort_values('shap_importance', ascending=False).reset_index(drop=True)
            df_s_rf['rank_shap'] = df_s_rf.index + 1

            # ══════════════════════════════════════════════════════
            # 3. Bayesian Ridge
            # ══════════════════════════════════════════════════════
            pipe_br = Pipeline([('scaler', StandardScaler()), ('bayesian', BayesianRidge())])
            grid_br = GridSearchCV(pipe_br, {
                'bayesian__max_iter': [300, 500, 1000],
                'bayesian__alpha_1':  [1e-6, 1e-4],
                'bayesian__alpha_2':  [1e-6, 1e-4],
                'bayesian__lambda_1': [1e-6, 1e-4],
                'bayesian__lambda_2': [1e-6, 1e-4],
            }, cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)
            
            p_br = permutation_importance(grid_br.best_estimator_, X_test, y_test, n_repeats=75, n_jobs=-1)
            df_p_br = pd.DataFrame({'feature': X_df.columns, 'importance_mean': p_br.importances_mean, 'model': 'Bayesian Ridge'}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
            df_p_br['rank_perm'] = df_p_br.index + 1
            
            # SHAP Original Bayesian
            best_rb_model = grid_br.best_estimator_.named_steps['bayesian']
            scaler_br      = grid_br.best_estimator_.named_steps['scaler']
            X_bg_scaled    = scaler_br.transform(X_train_background)
            X_test_scaled  = scaler_br.transform(X_test_global)
            explainer_rb   = shap.LinearExplainer(best_rb_model, X_bg_scaled)
            shap_values_rb = explainer_rb.shap_values(X_test_scaled)
            df_s_br = pd.DataFrame({'feature': X_df.columns, 'shap_importance': np.mean(np.abs(shap_values_rb), axis=0), 'model': 'Bayesian Ridge'}).sort_values('shap_importance', ascending=False).reset_index(drop=True)
            df_s_br['rank_shap'] = df_s_br.index + 1

            # ══════════════════════════════════════════════════════
            # 4. XGBoost
            # ══════════════════════════════════════════════════════
            pipe_xgb = Pipeline([('xgb', XGBRegressor(n_jobs=-1, random_state=42, verbosity=0))])
            grid_xgb = GridSearchCV(pipe_xgb, {
                'xgb__n_estimators':     [100, 300],
                'xgb__max_depth':        [3, 6],
                'xgb__learning_rate':    [0.05, 0.1],
                'xgb__subsample':        [0.8, 1.0],
                'xgb__colsample_bytree': [0.8, 1.0],
            }, cv=5, scoring='r2', n_jobs=-1).fit(X_train, y_train)
            
            p_xgb = permutation_importance(grid_xgb.best_estimator_, X_test, y_test, n_repeats=75, n_jobs=-1)
            df_p_xgb = pd.DataFrame({'feature': X_df.columns, 'importance_mean': p_xgb.importances_mean, 'model': 'XGBoost'}).sort_values('importance_mean', ascending=False).reset_index(drop=True)
            df_p_xgb['rank_perm'] = df_p_xgb.index + 1
            
            # SHAP Original XGBoost
            best_xgb_model = grid_xgb.best_estimator_.named_steps['xgb']
            explainer_xgb = shap.TreeExplainer(best_xgb_model, X_train_background)
            shap_values_xgb = explainer_xgb.shap_values(X_test_global)
            df_s_xgb = pd.DataFrame({'feature': X_df.columns, 'shap_importance': np.mean(np.abs(shap_values_xgb), axis=0), 'model': 'XGBoost'}).sort_values('shap_importance', ascending=False).reset_index(drop=True)
            df_s_xgb['rank_shap'] = df_s_xgb.index + 1

            # --- Guardar Consensos (Rankings) en data/outputs/run_XXX/ ---
            save_to_run(pd.concat([df_p_svr, df_p_rf, df_p_br, df_p_xgb]), 'ranking_consenso_perm.csv')
            save_to_run(pd.concat([df_s_svr, df_s_rf, df_s_br, df_s_xgb]), 'ranking_consenso_shap.csv')
            
            # --- Guardar Historial de Predicciones ---
            preds = {
                'Pred_SVR': grid_svr.predict(X_test), 'Pred_RF': grid_rf.predict(X_test),
                'Pred_Bayesian': grid_br.predict(X_test), 'Pred_XGBoost': grid_xgb.predict(X_test)
            }
            df_preds_iter = pd.DataFrame({
                'Iteration': i + 1,
                'Actual_Current': y_test.flatten()
            })
            for col_name, p_array in preds.items():
                df_preds_iter[col_name] = p_array
            save_to_run(df_preds_iter, 'model_predictions_history.csv')

            # --- Guardar Métricas de Iteración ---
            log_entry = {'iter': i+1, 'status': 'ok'}
            for mod_key, p_array in preds.items():
                mod = mod_key.replace("Pred_", "")
                if mod == "XGBoost": mod = "XGB" 
                log_entry.update({
                    f'R2_{mod}': r2_score(y_test, p_array), f'MAE_{mod}': mean_absolute_error(y_test, p_array),
                    f'RMSE_{mod}': np.sqrt(mean_squared_error(y_test, p_array)), f'MAPE_{mod}': mape(y_test, p_array)
                })
            log.append(log_entry)
            
        except Exception as e:
            log.append({'iter': i+1, 'status': 'error', 'detail': str(e)})
            print(f"Error en iteración: {e}")
        finally:
            gc.collect()

    # Disparar generación de archivos finales
    pd.DataFrame(log).to_csv(os.path.join(RUN_DIR, 'iteration_log.csv'), index=False)
    ejecutar_analisis_final()

    # ── PASO FINAL: Triple Consenso 80% normalizado ──────────────────────────
    # (consenso_score.py integrado)
    # Se ejecuta después de ejecutar_analisis_final() porque necesita los CSVs
    # de ranking que el loop genera iterativamente.
    # Produce: tabla_consenso_final.csv en el mismo RUN_DIR.
    paths_consenso = {
        "run_dir":    RUN_DIR,
        "perm_csv":   os.path.join(RUN_DIR, "ranking_consenso_perm.csv"),
        "shap_csv":   os.path.join(RUN_DIR, "ranking_consenso_shap.csv"),
        "output_csv": os.path.join(RUN_DIR, "tabla_consenso_final.csv"),
    }
    process_triple_80(paths_consenso)

    print(f"\n[OK] Ejecución finalizada con éxito.")