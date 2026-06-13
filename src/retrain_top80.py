
# ─────────────────────────────────────────────────────────────
# retrain_top80.py
# Etapa de reentrenamiento basada en selección de features Top 80%
# Compatible con: pulse | sc/edfa | sc/curr
# Uso:
#   python retrain_top80.py --run run_20260527_153000 --uspl 1
#   python retrain_top80.py --run run_20260527_153000 --uspl 2 --regimen sc --sc_mode edfa
# ─────────────────────────────────────────────────────────────

import gc
import os
import argparse
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# ══════════════════════════════════════════════════════════════
# 0. CLI
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description='Reentrenamiento Top 80%% — retrain_top80.py'
    )
    parser.add_argument('--run',     type=str, required=True,
                        help='ID del run previo. Ej: run_20260527_153000')
    parser.add_argument('--uspl',    type=int, required=True,
                        help='ID del láser (1 o 2)')
    parser.add_argument('--regimen', type=str,
                        choices=['pulse', 'sc'], default='pulse',
                        help='Régimen de operación: pulse o sc')
    parser.add_argument('--sc_mode', type=str,
                        choices=['edfa', 'curr'], default='curr',
                        help='Si regimen=sc: edfa o curr')
    parser.add_argument('--iters',   type=int, default=None,
                        help='Número de iteraciones de reentrenamiento. '
                             'Si no se especifica, se pide interactivamente.')
    return parser.parse_args()


# ══════════════════════════════════════════════════════════════
# 1. RUTAS Y CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

def build_paths(args) -> dict:
    """
    Construye todas las rutas del proyecto a partir de los argumentos CLI,
    replicando la lógica de ruta del script original.

    Returns:
        dict con claves: base_dir, uspl_id, run_id, run_dir_orig,
                         run_dir_new, figures_dir_new,
                         path_features, consensus_csv,
                         perm_csv, shap_csv, iter_log_csv,
                         predictions_csv
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    uspl_id  = f"USPL_{args.uspl}"
    regimen  = args.regimen
    sc_mode  = args.sc_mode if regimen == 'sc' else None

    # Ruta al CSV de features (idéntica al script original)
    if regimen == 'pulse':
        path_features = os.path.join(
            base_dir, "data", f"data_{uspl_id}", "processed",
            "extracted_features.csv"
        )
    elif regimen == 'sc' and sc_mode == 'edfa':
        path_features = os.path.join(
            base_dir, "data", f"data_{uspl_id}", "processed",
            "SC", "Var_EDFA", "extracted_features.csv"
        )
    else:  # sc / curr
        path_features = os.path.join(
            base_dir, "data", f"data_{uspl_id}", "processed",
            "SC", "Var_Cur", "extracted_features.csv"
        )

    # Directorio del run original
    run_dir_orig = os.path.join(
        base_dir, "data", f"data_{uspl_id}", "outputs", args.run
    )

    # Nuevo run derivado
    timestamp_new = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    new_run_name  = f"{args.run}_top80_{timestamp_new}"

    run_dir_new = os.path.join(
        base_dir, "data", f"data_{uspl_id}", "outputs", new_run_name
    )
    figures_dir_new = os.path.join(
        base_dir, "reports", f"reports_{uspl_id}", "figures", new_run_name
    )

    return {
        "base_dir":         base_dir,
        "uspl_id":          uspl_id,
        "run_id_orig":      args.run,
        "run_dir_orig":     run_dir_orig,
        "run_dir_new":      run_dir_new,
        "figures_dir_new":  figures_dir_new,
        "path_features":    path_features,
        # Archivos del run original
        "perm_csv":         os.path.join(run_dir_orig, "ranking_consenso_perm.csv"),
        "shap_csv":         os.path.join(run_dir_orig, "ranking_consenso_shap.csv"),
        "consensus_csv":    os.path.join(run_dir_orig, "tabla_consenso_final.csv"),
        "iter_log_csv":     os.path.join(run_dir_orig, "iteration_log.csv"),
        "predictions_csv":  os.path.join(run_dir_orig, "model_predictions_history.csv"),
    }


def validate_paths(paths: dict) -> None:
    """Verifica que los archivos críticos del run original existan."""
    required = ["perm_csv", "shap_csv", "path_features"]
    missing  = [k for k in required if not os.path.exists(paths[k])]
    if missing:
        raise FileNotFoundError(
            f"[ERROR] Archivos requeridos no encontrados:\n"
            + "\n".join(f"  [{k}] {paths[k]}" for k in missing)
        )
    print("[OK] Todos los archivos del run original localizados.")


# ══════════════════════════════════════════════════════════════
# 2. SELECCIÓN DE FEATURES TOP 80% POR MODELO
# ══════════════════════════════════════════════════════════════

MODEL_KEYS = {
    "SVR":            "SVR",
    "Random Forest":  "Random Forest",
    "Bayesian Ridge": "Bayesian Ridge",
    "XGBoost":        "XGBoost",
}


def _top80_per_model(df_ranking: pd.DataFrame,
                     col_importance: str,
                     model_map: dict) -> dict:
    """
    Calcula el subset Top 80% de importancia acumulada por modelo.

    Args:
        df_ranking     : DataFrame con columnas [feature, model, col_importance].
        col_importance : Nombre de la columna de importancia.
        model_map      : Diccionario {clave_interna: nombre_en_csv}.

    Returns:
        dict {clave_interna: [lista de features Top80]}
    """
    result = {}
    for key, model_name in model_map.items():
        sub = (
            df_ranking[df_ranking['model'] == model_name]
            .groupby('feature')[col_importance]
            .mean()
            .reset_index()
            .sort_values(col_importance, ascending=False)
        )
        sub['importance_clipped'] = sub[col_importance].clip(lower=0)
        total = sub['importance_clipped'].sum()
        if total < 1e-12:
            # Todos los valores son 0 o negativos — usar todas las features
            result[key] = sub['feature'].tolist()
            continue
        sub['cumulative'] = (sub['importance_clipped'] / total).cumsum()
        # Incluir la primera feature que supera el umbral (evitar 0 features)
        mask = sub['cumulative'].shift(1).fillna(0) < 0.80
        mask |= (~mask & (sub['cumulative'].shift(1).fillna(0) < 0.80)).shift(-1).fillna(False)
        # Estrategia simple y robusta: tomar todas hasta que el acumulado >= 0.80
        idx_cross = sub[sub['cumulative'] >= 0.80].index
        if len(idx_cross) == 0:
            selected = sub['feature'].tolist()
        else:
            cutoff = idx_cross[0]
            selected = sub.loc[sub.index <= cutoff, 'feature'].tolist()
        result[key] = selected
    return result


def load_feature_subsets(paths: dict) -> dict:
    """
    Lee los CSVs de ranking del run original y devuelve los subsets Top 80%
    por modelo y por método (SHAP y Permutation).

    Returns:
        dict con claves 'shap' y 'perm', cada una con un sub-dict
        {nombre_modelo: [features]}
    """
    df_perm = pd.read_csv(paths['perm_csv'])
    df_shap = pd.read_csv(paths['shap_csv'])

    subsets_shap = _top80_per_model(df_shap, 'shap_importance', MODEL_KEYS)
    subsets_perm = _top80_per_model(df_perm, 'importance_mean',  MODEL_KEYS)

    # Log
    for key in MODEL_KEYS:
        n_shap = len(subsets_shap[key])
        n_perm = len(subsets_perm[key])
        print(f"  [{key}] Top80 SHAP: {n_shap} features | Top80 Perm: {n_perm} features")

    return {"shap": subsets_shap, "perm": subsets_perm}


def save_feature_lists(subsets: dict, run_dir_new: str) -> None:
    """Persiste los subsets de features seleccionados como CSV."""
    rows_shap, rows_perm = [], []
    for model_key in MODEL_KEYS:
        for f in subsets['shap'][model_key]:
            rows_shap.append({'model': model_key, 'feature': f})
        for f in subsets['perm'][model_key]:
            rows_perm.append({'model': model_key, 'feature': f})

    pd.DataFrame(rows_shap).to_csv(
        os.path.join(run_dir_new, 'selected_features_shap.csv'), index=False
    )
    pd.DataFrame(rows_perm).to_csv(
        os.path.join(run_dir_new, 'selected_features_perm.csv'), index=False
    )
    print("[OK] Listas de features Top80 guardadas.")


# ══════════════════════════════════════════════════════════════
# 3. MÉTRICAS
# ══════════════════════════════════════════════════════════════

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "R2":   r2_score(y_true, y_pred),
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }


# ══════════════════════════════════════════════════════════════
# 4. DEFINICIÓN DE MODELOS (IDÉNTICA AL SCRIPT ORIGINAL)
# ══════════════════════════════════════════════════════════════

def build_grid(model_key: str) -> GridSearchCV:
    """
    Devuelve el GridSearchCV correspondiente al modelo, con los mismos
    hiperparámetros y configuración del script original.
    """
    if model_key == "SVR":
        pipe = Pipeline([('scaler', StandardScaler()), ('svr', SVR())])
        param_grid = {
            'svr__C':       [10, 100, 1000],
            'svr__gamma':   ['scale', 0.001, 0.005, 0.01],
            'svr__epsilon': [1, 2.5, 5],
        }
    elif model_key == "Random Forest":
        pipe = Pipeline([('rf', RandomForestRegressor(n_jobs=-1))])
        param_grid = {
            'rf__n_estimators':     [100, 300, 500],
            'rf__max_depth':        [5, 10, 15],
            'rf__min_samples_leaf': [4, 8, 16],
            'rf__max_features':     ['sqrt', 'log2'],
        }
    elif model_key == "Bayesian Ridge":
        pipe = Pipeline([('scaler', StandardScaler()), ('bayesian', BayesianRidge())])
        param_grid = {
            'bayesian__max_iter': [300, 500, 1000],
            'bayesian__alpha_1':  [1e-6, 1e-4],
            'bayesian__alpha_2':  [1e-6, 1e-4],
            'bayesian__lambda_1': [1e-6, 1e-4],
            'bayesian__lambda_2': [1e-6, 1e-4],
        }
    elif model_key == "XGBoost":
        pipe = Pipeline([('xgb', XGBRegressor(
            n_jobs=-1, random_state=42, verbosity=0
        ))])
        param_grid = {
            'xgb__n_estimators':     [100, 300],
            'xgb__max_depth':        [3, 6],
            'xgb__learning_rate':    [0.05, 0.1],
            'xgb__subsample':        [0.8, 1.0],
            'xgb__colsample_bytree': [0.8, 1.0],
        }
    else:
        raise ValueError(f"Modelo desconocido: {model_key}")

    return GridSearchCV(pipe, param_grid, cv=5, scoring='r2', n_jobs=-1)


# ══════════════════════════════════════════════════════════════
# 5. LOOP DE REENTRENAMIENTO
# ══════════════════════════════════════════════════════════════

def retrain_loop(X_df: pd.DataFrame,
                 y: np.ndarray,
                 subsets: dict,
                 n_iters: int,
                 run_dir_new: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ejecuta el loop de reentrenamiento para cada modelo con:
      - Todas las features
      - Top 80% SHAP
      - Top 80% Permutation

    Returns:
        (df_metrics_all_iters, df_predictions_all_iters)
    """
    all_metrics = []
    all_preds   = []

    # Historial de predicciones por iteración (para cálculo de IC / confianza)
    preds_history = []

    for i in range(n_iters):
        print(f"\n[ITER {i+1}/{n_iters}] Reentrenando con subsets Top80...")

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_df.values, y,
                test_size=0.2,
                stratify=pd.cut(y, bins=10)
            )
            feat_names = X_df.columns.tolist()

            iter_metrics = {"iter": i + 1, "status": "ok"}
            preds_row    = {"iter": i + 1, "y_test": y_test.tolist()}

            for model_key in MODEL_KEYS:
                print(f"  → {model_key}...")

                feat_shap = subsets['shap'][model_key]
                feat_perm = subsets['perm'][model_key]

                idx_all  = list(range(len(feat_names)))
                idx_shap = [feat_names.index(f) for f in feat_shap if f in feat_names]
                idx_perm = [feat_names.index(f) for f in feat_perm if f in feat_names]

                subsets_to_run = {
                    "All":   (idx_all,  X_train[:, idx_all],  X_test[:, idx_all]),
                    "SHAP80": (idx_shap, X_train[:, idx_shap], X_test[:, idx_shap]),
                    "PERM80": (idx_perm, X_train[:, idx_perm], X_test[:, idx_perm]),
                }

                for suffix, (_, Xtr, Xte) in subsets_to_run.items():
                    grid = build_grid(model_key)
                    grid.fit(Xtr, y_train)
                    y_pred = grid.predict(Xte)

                    m = compute_metrics(y_test, y_pred)
                    tag = f"{model_key}_{suffix}"

                    iter_metrics[f"R2_{tag}"]   = m["R2"]
                    iter_metrics[f"MAE_{tag}"]  = m["MAE"]
                    iter_metrics[f"RMSE_{tag}"] = m["RMSE"]
                    iter_metrics[f"MAPE_{tag}"] = m["MAPE"]

                    preds_row[f"pred_{tag}"] = y_pred.tolist()

                gc.collect()

            # Construir DataFrame de predicciones por iteración (formato similar a temporallearning_optnet)
            df_preds_iter = pd.DataFrame({
                'Iteration': i + 1,
                'Actual_Current': y_test
            })
            for col, val in preds_row.items():
                if col not in ('iter', 'y_test'):
                    col_name = col.replace('pred_', 'Pred_')
                    df_preds_iter[col_name] = val

            # Añadir columnas estándar compatibles con temporallearning_opnet.py
            model_short = {
                'SVR': 'SVR',
                'Random Forest': 'RF',
                'Bayesian Ridge': 'Bayesian',
                'XGBoost': 'XGBoost'
            }
            default_list = [np.nan] * len(y_test)
            for model_key, short in model_short.items():
                # Preferir la predicción con todas las features; si no existe, usar SHAP80 o PERM80
                pred_key_all = f'pred_{model_key}_All'
                pred_key_shap = f'pred_{model_key}_SHAP80'
                pred_key_perm = f'pred_{model_key}_PERM80'
                if pred_key_all in preds_row:
                    df_preds_iter[f'Pred_{short}'] = preds_row[pred_key_all]
                elif pred_key_shap in preds_row:
                    df_preds_iter[f'Pred_{short}'] = preds_row[pred_key_shap]
                elif pred_key_perm in preds_row:
                    df_preds_iter[f'Pred_{short}'] = preds_row[pred_key_perm]
                else:
                    df_preds_iter[f'Pred_{short}'] = default_list

            preds_history.append(df_preds_iter)

            all_metrics.append(iter_metrics)

            # Aplanar predicciones en filas (una por sample)
            n_test = len(y_test)
            for k in range(n_test):
                row = {"iter": i + 1, "actual": y_test[k]}
                for col, val in preds_row.items():
                    if col not in ("iter", "y_test"):
                        row[col] = val[k]
                all_preds.append(row)

        except Exception as e:
            all_metrics.append({"iter": i + 1, "status": "error", "detail": str(e)})
            print(f"  [ERROR] Iteración {i+1}: {e}")


    # Guardar historial de predicciones por iteración y calcular intervalos de confianza (2.5% - 97.5%)
    try:
        if len(preds_history) > 0:
            df_history = pd.concat(preds_history, ignore_index=True)
            df_history.to_csv(os.path.join(run_dir_new, 'model_predictions_history.csv'), index=False)

            ci_rows = []
            pred_cols = [c for c in df_history.columns if c not in ('Iteration', 'Actual_Current')]
            for col in pred_cols:
                grouped = df_history.groupby('Actual_Current')[col].agg(list).reset_index()
                for _, r in grouped.iterrows():
                    arr = np.array(r[col])
                    mean = float(np.mean(arr))
                    lower = float(np.percentile(arr, 2.5))
                    upper = float(np.percentile(arr, 97.5))
                    std = float(np.std(arr))
                    model_subset = col.replace('Pred_', '')
                    parts = model_subset.split('_')
                    model = parts[0]
                    subset = '_'.join(parts[1:]) if len(parts) > 1 else ''
                    ci_rows.append({
                        'Actual_Current': r['Actual_Current'], 'Model': model, 'Subset': subset,
                        'Mean': mean, 'CI_lower': lower, 'CI_upper': upper, 'Std': std
                    })
            df_ci = pd.DataFrame(ci_rows)
            df_ci.to_csv(os.path.join(run_dir_new, 'predictions_confidence_intervals.csv'), index=False)
            print('[OK] model_predictions_history.csv y predictions_confidence_intervals.csv guardados.')
    except Exception as e:
        print(f"[WARN] No se pudo generar historial de predicciones o CI: {e}")

    df_metrics = pd.DataFrame(all_metrics)
    df_preds   = pd.DataFrame(all_preds)

    df_metrics.to_csv(os.path.join(run_dir_new, "logs_retrain.csv"), index=False)
    df_preds.to_csv(os.path.join(run_dir_new, "predictions_comparison.csv"), index=False)

    print("\n[OK] Logs y predicciones guardados.")
    return df_metrics, df_preds


# ══════════════════════════════════════════════════════════════
# 6. TABLA COMPARATIVA DE MÉTRICAS
# ══════════════════════════════════════════════════════════════

def build_metrics_comparison(df_metrics: pd.DataFrame,
                              run_dir_new: str) -> pd.DataFrame:
    """
    Agrega las métricas por iteración (media) y genera la tabla comparativa
    en el formato solicitado:
        Model | R2_All | R2_SHAP80 | R2_PERM80 | MAE_All | ... | RMSE_PERM80
    """
    df_ok = df_metrics[df_metrics.get('status', 'ok') == 'ok'] \
        if 'status' in df_metrics.columns else df_metrics

    rows = []
    for model_key in MODEL_KEYS:
        row = {"Model": model_key}
        for metric in ["R2", "MAE", "RMSE", "MAPE"]:
            for subset in ["All", "SHAP80", "PERM80"]:
                col = f"{metric}_{model_key}_{subset}"
                if col in df_ok.columns:
                    row[f"{metric}_{subset}"] = round(df_ok[col].mean(), 4)
                else:
                    row[f"{metric}_{subset}"] = np.nan
        rows.append(row)

    cols_order = (
        ["Model"]
        + [f"R2_{s}"   for s in ["All", "SHAP80", "PERM80"]]
        + [f"MAE_{s}"  for s in ["All", "SHAP80", "PERM80"]]
        + [f"RMSE_{s}" for s in ["All", "SHAP80", "PERM80"]]
        + [f"MAPE_{s}" for s in ["All", "SHAP80", "PERM80"]]
    )

    df_comp = pd.DataFrame(rows)[[c for c in cols_order if c in pd.DataFrame(rows).columns]]
    df_comp.to_csv(os.path.join(run_dir_new, "metrics_comparison.csv"), index=False)
    print("[OK] metrics_comparison.csv guardado.")
    return df_comp


# ══════════════════════════════════════════════════════════════
# 7. GRÁFICAS COMPARATIVAS POR MODELO
# ══════════════════════════════════════════════════════════════

SUBSET_STYLES = {
    "All":    {"color": "#2196F3", "label": "All Features"},
    "SHAP80": {"color": "#FF5722", "label": "Top 80% SHAP"},
    "PERM80": {"color": "#4CAF50", "label": "Top 80% Permutation"},
}

MODEL_FILE_KEYS = {
    "SVR":            "svr",
    "Random Forest":  "rf",
    "Bayesian Ridge": "bayesian",
    "XGBoost":        "xgb",
}


def plot_model_comparison(df_preds: pd.DataFrame,
                          df_comp:  pd.DataFrame,
                          figures_dir: str) -> None:
    """
    Genera una figura por modelo con líneas continuas:
      - Actual vs Predicho para All Features, SHAP80 y PERM80.
      - Línea ideal (Actual = Predicted).
      - R² de cada subset en el subtítulo.
    """
    for model_key in MODEL_KEYS:
        file_key  = MODEL_FILE_KEYS[model_key]
        fig, ax   = plt.subplots(figsize=(9, 7))

        # Línea ideal
        all_actuals = df_preds['actual'].values
        lim_min = all_actuals.min() * 0.97
        lim_max = all_actuals.max() * 1.03
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                color='black', linestyle='--', linewidth=1.5,
                label='Ideal (Actual = Predicted)', zorder=5)

        # R² subtítulo
        r2_labels = []

        for subset, style in SUBSET_STYLES.items():
            pred_col = f"pred_{model_key}_{subset}"
            if pred_col not in df_preds.columns:
                continue

            df_plot = (
                df_preds[['actual', pred_col]]
                .dropna()
                .sort_values('actual')
            )

            sns.lineplot(
                data=df_plot,
                x='actual',
                y=pred_col,
                errorbar=('ci', 95),
                color=style['color'],
                linewidth=2,
                label=style['label'],
                ax=ax,
                zorder=4
            )

            # R² desde la tabla de comparativa
            r2_col = f"R2_{subset}"
            if r2_col in df_comp.columns:
                r2_val = df_comp.loc[df_comp['Model'] == model_key, r2_col].values
                r2_str = f"{r2_val[0]:.3f}" if len(r2_val) > 0 else "—"
            else:
                r2_str = "—"
            r2_labels.append(f"R²({style['label']})={r2_str}")

        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("Actual Current", fontsize=12)
        ax.set_ylabel("Predicted Current", fontsize=12)
        ax.set_title(
            f"Feature Subset Comparison — {model_key}\n"
            + " | ".join(r2_labels),
            fontsize=11, fontweight='bold'
        )
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        plt.tight_layout()

        fname = os.path.join(figures_dir, f"comparison_{file_key}_top80.png")
        plt.savefig(fname, dpi=120)
        plt.close()
        print(f"[REPORTE] Gráfica guardada: {fname}")


# ══════════════════════════════════════════════════════════════
# 8. EJECUCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

def main():
    args  = parse_args()
    paths = build_paths(args)

    print("\n" + "═" * 60)
    print(f"  RETRAIN TOP 80% — {paths['uspl_id']}")
    print(f"  Run original : {paths['run_id_orig']}")
    print(f"  Nuevo run    : {os.path.basename(paths['run_dir_new'])}")
    print("═" * 60)

    # Validar que los archivos existan
    validate_paths(paths)

    # Crear directorios del nuevo run
    os.makedirs(paths['run_dir_new'],    exist_ok=True)
    os.makedirs(paths['figures_dir_new'], exist_ok=True)

    # Número de iteraciones
    n_iters = args.iters
    if n_iters is None:
        n_iters = int(input("Ingrese el número de iteraciones de reentrenamiento: "))

    # Cargar datos
    print("\n[INFO] Cargando dataset de features...")
    df_raw = pd.read_csv(paths['path_features'])
    # Interpolación lineal para rellenar NaNs (ambas direcciones). Mantener columnas intactas.
    df_raw = df_raw.interpolate(method='linear', limit_direction='both', axis=0)
    print("[INFO] Se aplicó interpolación lineal a valores faltantes (NaN).")
    y      = df_raw['target_current'].values
    X_df   = df_raw.drop(columns=['target_current'])
    print(f"[OK] Dataset: {X_df.shape[0]} muestras × {X_df.shape[1]} features")

    # Selección de features Top 80%
    print("\n[INFO] Calculando subsets Top 80% desde run original...")
    subsets = load_feature_subsets(paths)
    save_feature_lists(subsets, paths['run_dir_new'])

    # Loop de reentrenamiento
    print(f"\n[INFO] Iniciando reentrenamiento ({n_iters} iteraciones × "
          f"{len(MODEL_KEYS)} modelos × 3 subsets)...")
    df_metrics, df_preds = retrain_loop(
        X_df, y, subsets, n_iters, paths['run_dir_new']
    )

    # Tabla comparativa de métricas
    print("\n[INFO] Generando tabla comparativa de métricas...")
    df_comp = build_metrics_comparison(df_metrics, paths['run_dir_new'])

    print("\n── Resumen de métricas medias ──────────────────────────")
    print(df_comp[["Model", "R2_All", "R2_SHAP80", "R2_PERM80"]].to_string(index=False))

    # Gráficas comparativas
    print("\n[INFO] Generando gráficas comparativas por modelo...")
    plot_model_comparison(df_preds, df_comp, paths['figures_dir_new'])

    print("\n" + "═" * 60)
    print("  RETRAIN TOP 80% COMPLETADO")
    print(f"  Outputs  → {paths['run_dir_new']}")
    print(f"  Figuras  → {paths['figures_dir_new']}")
    print("═" * 60)


if __name__ == "__main__":
    main()
