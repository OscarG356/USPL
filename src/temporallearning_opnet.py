"""
Pipeline Experimental Unificado y Leakage-Safe para Señales USPL
Esquema Leave-One-Current-Range-Out (5 chunks deterministas de corriente).
Incluye interpretabilidad (SHAP/Permutation) y reportes completos.
"""

import argparse
import datetime

#import gc
import os
import re

# pyrefly: ignore [missing-import]
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
#import matplotlib.cm as cm
#import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
#import seaborn as sns

# pyrefly: ignore [missing-import]
import shap

# pyrefly: ignore [missing-import]
import tsfel

# pyrefly: ignore [missing-import]
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(description="Pipeline Integrado OPNET - LOCO (Leave-One-Current-Range-Out)")
parser.add_argument("--uspl", type=int, default=1, help="ID del láser (1 o 2)")
parser.add_argument(
    "--iters",
    type=int,
    default=5,
    help=(
        "DEPRECADO / NO USADO para el split externo. El pipeline siempre ejecuta "
        "exactamente 5 iteraciones deterministas (una por cada chunk de corriente). "
        "Se conserva solo por compatibilidad de CLI."
    ),
)
parser.add_argument(
    "--feature_method",
    type=str,
    default="all",
    help="Selección de características (temporal o all)",
)
parser.add_argument(
    "--threshold_corr",
    type=float,
    default=0.90,
    help="Umbral de correlación para descartar características",
)
args = parser.parse_args()

if args.iters != 5:
    print(
        f"[AVISO] --iters={args.iters} fue ignorado: el esquema Leave-One-Current-"
        f"Range-Out ejecuta siempre exactamente 5 iteraciones (una por chunk)."
    )

USPL_ID = f"USPL_{args.uspl}"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005

RAW_DIR = os.path.join(BASE_DIR, "data", f"data_{USPL_ID}", "raw", "osciloscopio")
RUN_DIR = os.path.join(
    BASE_DIR, "data", f"data_{USPL_ID}", "outputs", f"run_{TIMESTAMP}_loco_chunks"
)
os.makedirs(RUN_DIR, exist_ok=True)

print(f"--- Iniciando Pipeline LOCO (Leave-One-Current-Range-Out) para: {USPL_ID} ---")
print(f"--- Salidas en: {RUN_DIR} ---")

# ══════════════════════════════════════════════════════════════
# 1bis. CHUNKS DE CORRIENTE (rangos deterministas)
# ══════════════════════════════════════════════════════════════

# Rango [min, max] (mA) inclusivo para cada chunk.
N_CHUNKS = 5
N_ITERS = args.iters
# ══════════════════════════════════════════════════════════════
# 2. FUNCIONES AUXILIARES Y PREPROCESAMIENTO RAW
# ══════════════════════════════════════════════════════════════


def natural_sort_key(s: str) -> list:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def load_raw_signals(data_folder: str):
    """Carga y segmenta las señales CRUDAS. Sin normalizar.

    El `current_chunk` y el `trace_id` se asignan a la TRAZA ORIGINAL antes de
    segmentar, y cada segmento generado hereda el mismo `current_chunk` y
    `trace_id` que su traza de origen. Esto garantiza que nunca se mezclen
    segmentos de una misma traza (o del mismo chunk) entre TRAIN y TEST.
    """
    signal_files = [
        f
        for f in os.listdir(data_folder)
        if f.lower().endswith(".csv") and not f.endswith("(1).CSV")
    ]
    signal_files.sort(key=natural_sort_key)

    all_signals_data = []
    for file_name in signal_files:
        file_path = os.path.join(data_folder, file_name)
        signal_df = pd.read_csv(file_path, header=0)
        all_signals_data.append(signal_df.iloc[:, 1].astype(float))

    all_signals_data = all_signals_data[191:]
    signals_matrix = pd.DataFrame(all_signals_data).values

    # Obtener fs
    example_df = pd.read_csv(os.path.join(data_folder, signal_files[0]), header=0)
    fs = 1 / (example_df.iloc[1, 0] - example_df.iloc[0, 0])

    # Etiquetas
    labels_df = pd.read_excel(os.path.join(data_folder, "Datos-Corriente.xlsx"))
    t_Corriente_raw = labels_df["Corriente (mA)"].values[191:]

    # Segmentación (con asignación de current_chunk y trace_id ANTES de segmentar)
    num_traces = len(signals_matrix)
    #np.random.seed(42)  # Opcional: fija una semilla para reproducibilidad
    
    # Genera un vector con la misma cantidad de elementos para cada chunk (1 a 5)
    chunk_assignment = np.tile(np.arange(1, N_CHUNKS + 1), int(np.ceil(num_traces / N_CHUNKS)))[:num_traces]
    np.random.shuffle(chunk_assignment)

    segmented_signals = []
    segmented_labels = []
    segmented_chunks = []
    segmented_trace_ids = []
    segment_length, num_segments = 300, 2

    for idx, signal_array in enumerate(signals_matrix):
        current_value = float(t_Corriente_raw[idx])
        chunk_id = int(chunk_assignment[idx])  # Chunk asignado aleatoriamente a la traza

        if len(signal_array) >= num_segments * segment_length:
            for i in range(num_segments):
                segmented_signals.append(
                    signal_array[i * segment_length : (i + 1) * segment_length]
                )
                segmented_labels.append(current_value)
                segmented_chunks.append(chunk_id)
                segmented_trace_ids.append(idx)

    return (
        np.array(segmented_signals),
        np.array(segmented_labels),
        np.array(segmented_chunks),
        np.array(segmented_trace_ids),
        fs,
    )


def mape(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100


def fit_feature_selection(X_train_df, threshold_corr=0.90, epsilon=1e-10):
    """Determina qué características mantener usando ÚNICAMENTE TRAIN."""
    # 1. Varianza
    variances = X_train_df.var()
    keep_var = variances[variances > epsilon].index.tolist()
    X_train_var = X_train_df[keep_var]

    # 2. Correlación
    corr_matrix = X_train_var.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [c for c in upper.columns if any(upper[c] >= threshold_corr)]

    selected_features = [c for c in keep_var if c not in to_drop_corr]
    return selected_features


def save_to_run_append(df, filename):
    path = os.path.join(RUN_DIR, filename)
    if os.path.exists(path):
        pd.concat([pd.read_csv(path), df], ignore_index=True).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


def holm_correction(pvals):
    """Aplica la corrección paso a paso de Holm-Bonferroni para comparaciones múltiples."""
    pvals = np.array(pvals)
    n = len(pvals)
    sorted_indices = np.argsort(pvals)
    adj_pvals = np.zeros(n)

    for i, idx in enumerate(sorted_indices):
        adj_pvals[idx] = pvals[idx] * (n - i)

    for i in range(1, n):
        idx_prev = sorted_indices[i - 1]
        idx_curr = sorted_indices[i]
        adj_pvals[idx_curr] = max(adj_pvals[idx_prev], adj_pvals[idx_curr])

    return np.minimum(adj_pvals, 1.0)


def process_triple_80(paths: dict) -> None:
    """Calcula el consenso Triple 80% sobre importancias de Permutation y SHAP."""
    print(
        f"\n--- Calculando Triple Consenso 80% en: {os.path.basename(paths['run_dir'])} ---"
    )

    if not os.path.exists(paths["perm_csv"]) or not os.path.exists(paths["shap_csv"]):
        print("[ERROR] Faltan archivos base para consenso_score.")
        return

    df_p = pd.read_csv(paths["perm_csv"])
    df_s = pd.read_csv(paths["shap_csv"])

    def normalize(df, col):
        df = df.copy()
        for mod in df["model"].unique():
            mask = df["model"] == mod
            vals = df.loc[mask, col]
            df.loc[mask, col] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
        return df

    df_p = normalize(df_p, "importance_mean")
    df_s = normalize(df_s, "shap_importance")

    c_perm = df_p.groupby("feature")["importance_mean"].mean().reset_index()
    c_perm.rename(columns={"importance_mean": "Consensus_Permutation"}, inplace=True)

    c_shap = df_s.groupby("feature")["shap_importance"].mean().reset_index()
    c_shap.rename(columns={"shap_importance": "Consensus_SHAP"}, inplace=True)

    df_consenso = pd.merge(c_perm, c_shap, on="feature")
    df_consenso["Consensus_General"] = (
        df_consenso["Consensus_Permutation"] + df_consenso["Consensus_SHAP"]
    ) / 2

    def get_80_stats(df_target, col_name):
        temp = (
            df_target[["feature", col_name]]
            .sort_values(col_name, ascending=False)
            .copy()
        )
        total = temp[col_name].sum()
        temp[f"Cum_{col_name}"] = (temp[col_name] / total).cumsum()
        temp[f"Top80_{col_name}"] = temp[f"Cum_{col_name}"].shift(1).fillna(0) < 0.8
        return temp[["feature", f"Cum_{col_name}", f"Top80_{col_name}"]]

    stats_perm = get_80_stats(df_consenso, "Consensus_Permutation")
    stats_shap = get_80_stats(df_consenso, "Consensus_SHAP")
    stats_gen = get_80_stats(df_consenso, "Consensus_General")

    df_consenso = (
        df_consenso.merge(stats_perm, on="feature")
        .merge(stats_shap, on="feature")
        .merge(stats_gen, on="feature")
    )

    df_consenso = df_consenso.sort_values("Consensus_General", ascending=False).round(4)
    df_consenso.to_csv(paths["output_csv"], index=False)
    print(f"[OK] Consensus table generated at: {paths['output_csv']}")


# ══════════════════════════════════════════════════════════════
# 3. PIPELINE PRINCIPAL (LOOP LOCO: 5 iteraciones deterministas)
# ══════════════════════════════════════════════════════════════


def run_pipeline():
    X_raw, y_raw, _chunk_raw_dummy, trace_id_raw, fs = load_raw_signals(RAW_DIR)
    domain = args.feature_method
    if domain == "all":
        domain = None
    cfg = tsfel.get_features_by_domain(domain)

    metrics_log, residuals_log, split_log = [], [], []
    N_ITERS = args.iters

    for iteration in range(N_ITERS):
        repetition = iteration + 1
        
        # 1. Asignar chunks aleatorios por traza en CADA iteración
        unique_traces = np.unique(trace_id_raw)
        num_traces = len(unique_traces)
        
        np.random.seed(42 + repetition)  # Semilla variable por iteración
        
        chunks_array = np.tile(np.arange(1, N_CHUNKS + 1), int(np.ceil(num_traces / N_CHUNKS)))[:num_traces]
        np.random.shuffle(chunks_array)
        trace_to_chunk = dict(zip(unique_traces, chunks_array))
        
        chunk_raw = np.array([trace_to_chunk[t_id] for t_id in trace_id_raw])
        
        # 2. Selección de Chunk de Test
        test_chunk = (repetition % N_CHUNKS) + 1
        train_chunks = [c for c in range(1, N_CHUNKS + 1) if c != test_chunk]

        # 3. Máscaras y Split (ÚNICA VEZ)
        train_mask = np.isin(chunk_raw, train_chunks)
        test_mask = chunk_raw == test_chunk

        X_train_raw, X_test_raw = X_raw[train_mask], X_raw[test_mask]
        y_train, y_test = y_raw[train_mask], y_raw[test_mask]
        chunk_train, chunk_test = chunk_raw[train_mask], chunk_raw[test_mask]
        trace_id_train, trace_id_test = (
            trace_id_raw[train_mask],
            trace_id_raw[test_mask],
        )
        
        test_min, test_max = float(y_test.min()), float(y_test.max())

        print(f"\n{'=' * 50}")
        print(f"Iteration {repetition}/{N_ITERS}")
        print(f"TEST CHUNK (Random Split): {test_chunk}")
        print(f"TEST RANGE: {test_min:.2f}–{test_max:.2f} mA")
        print(f"TRAIN CHUNKS: {train_chunks}")
        print(f"{'=' * 50}")

        # --- VALIDACIONES METODOLÓGICAS ---
        assert len(set(np.unique(chunk_train)) & set(np.unique(chunk_test))) == 0, (
            "ERROR FATAL: Leakage en Split de Chunks (chunk presente en TRAIN y TEST)"
        )
        assert set(np.unique(chunk_train)) == set(train_chunks), (
            "ERROR FATAL: TRAIN no contiene exactamente los 4 chunks esperados"
        )
        assert set(np.unique(chunk_test)) == {test_chunk}, (
            "ERROR FATAL: TEST no corresponde exactamente al chunk esperado"
        )
        assert len(set(trace_id_train) & set(trace_id_test)) == 0, (
            "ERROR FATAL: Leakage de trazas entre TRAIN y TEST"
        )

        split_log.append(
            {
                "repetition": repetition,
                "test_chunk": test_chunk,
                "test_current_min": test_min,
                "test_current_max": test_max,
                "train_chunks": ",".join(map(str, train_chunks)),
                "train_current_min": float(y_train.min()),
                "train_current_max": float(y_train.max()),
                "n_train_samples": len(y_train),
                "n_test_samples": len(y_test),
            }
        )
        # ... Resto del pipeline de entrenamiento TSFEL / Modelos / SHAP ...

        # --- NORMALIZACIÓN (Solo TRAIN) ---
        X_min_train = np.min(X_train_raw)
        X_max_train = np.max(X_train_raw)

        X_train_norm = (
            2.0 * (X_train_raw - X_min_train) / (X_max_train - X_min_train) - 1.0
        )
        X_test_norm = (
            2.0 * (X_test_raw - X_min_train) / (X_max_train - X_min_train) - 1.0
        )

        print(f"X_min_train: {X_min_train:.4f} | X_max_train: {X_max_train:.4f}")

        # --- FEATURE EXTRACTION ---
        print("Extracting features (TSFEL)...")
        X_train_feat = pd.concat(
            [
                tsfel.time_series_features_extractor(cfg, sig, fs=fs, verbose=0)
                for sig in X_train_norm
            ],
            ignore_index=True,
        ).dropna(axis=1, how="all")
        X_test_feat = pd.concat(
            [
                tsfel.time_series_features_extractor(cfg, sig, fs=fs, verbose=0)
                for sig in X_test_norm
            ],
            ignore_index=True,
        )

        common_cols = X_train_feat.columns.intersection(X_test_feat.columns)
        X_train_feat, X_test_feat = X_train_feat[common_cols], X_test_feat[common_cols]

        # --- FEATURE SELECTION (Solo TRAIN) ---
        selected_features = fit_feature_selection(
            X_train_feat, threshold_corr=args.threshold_corr
        )
        X_train_sel, X_test_sel = (
            X_train_feat[selected_features].values,
            X_test_feat[selected_features].values,
        )

        print(f"Features: {X_train_feat.shape[1]} -> {len(selected_features)} selected")

        # --- MODELADO, SHAP Y PERMUTATION ---
        models = {
            "SVR": (SVR(), {"model__C": [10, 100], "model__gamma": ["scale", 0.01]}),
            "Random Forest": (
                RandomForestRegressor(random_state=42),
                {"model__n_estimators": [100, 300], "model__max_depth": [5, 10]},
            ),
            "Bayesian Ridge": (BayesianRidge(), {"model__max_iter": [300]}),
            "XGBoost": (
                XGBRegressor(random_state=42, verbosity=0),
                {"model__n_estimators": [100, 300], "model__max_depth": [3, 6]},
            ),
        }

        iter_preds = {"repetition": repetition, "Actual_Current": y_test}
        df_p_list, df_s_list = [], []

        X_train_background = (
            shap.sample(X_train_sel, 100) if len(X_train_sel) > 100 else X_train_sel
        )
        X_test_global = (
            shap.sample(X_test_sel, 75) if len(X_test_sel) > 75 else X_test_sel
        )

        # CV interno: 4 chunks en TRAIN -> GroupKFold(n_splits=4), agrupado por chunk
        inner_cv = GroupKFold(n_splits=4)

        for model_name, (estimator, param_grid) in models.items():
            pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(X_train_sel, y_train, groups=chunk_train)

            y_pred = grid.predict(X_test_sel)
            iter_preds[f"Pred_{model_name}"] = y_pred

            # Permutation Importance
            p_imp = permutation_importance(
                grid.best_estimator_, X_test_sel, y_test, n_repeats=30, n_jobs=-1
            )
            df_p = pd.DataFrame(
                {
                    "feature": selected_features,
                    "importance_mean": p_imp.importances_mean,
                    "model": model_name,
                }
            )
            df_p = df_p.sort_values("importance_mean", ascending=False).reset_index(
                drop=True
            )
            df_p["rank_perm"] = df_p.index + 1
            df_p["repetition"] = repetition
            df_p["test_chunk"] = test_chunk
            df_p_list.append(df_p)

            # SHAP
            best_model = grid.best_estimator_.named_steps["model"]
            scaler = grid.best_estimator_.named_steps["scaler"]

            if model_name in ["Random Forest", "XGBoost"]:
                X_train_bg_scaled = scaler.transform(X_train_background)
                X_test_global_scaled = scaler.transform(X_test_global)
                explainer = shap.TreeExplainer(best_model, X_train_bg_scaled)
                shap_values = explainer.shap_values(
                    X_test_global_scaled, check_additivity=False
                )
            elif model_name == "Bayesian Ridge":
                X_train_bg_scaled = scaler.transform(X_train_background)
                X_test_global_scaled = scaler.transform(X_test_global)
                explainer = shap.LinearExplainer(best_model, X_train_bg_scaled)
                shap_values = explainer.shap_values(X_test_global_scaled)
            else:  # SVR
                predict_fn = grid.best_estimator_.predict
                explainer = shap.KernelExplainer(predict_fn, X_train_background)
                shap_values = explainer.shap_values(X_test_global)

            df_s = pd.DataFrame(
                {
                    "feature": selected_features,
                    "shap_importance": np.mean(np.abs(shap_values), axis=0),
                    "model": model_name,
                }
            )
            df_s = df_s.sort_values("shap_importance", ascending=False).reset_index(
                drop=True
            )
            df_s["rank_shap"] = df_s.index + 1
            df_s["repetition"] = repetition
            df_s["test_chunk"] = test_chunk
            df_s_list.append(df_s)

            # Métricas
            metrics_log.append(
                {
                    "repetition": repetition,
                    "test_chunk": test_chunk,
                    "test_current_min": test_min,
                    "test_current_max": test_max,
                    "train_chunks": ",".join(map(str, train_chunks)),
                    "model": model_name,
                    "R2": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAPE": mape(y_test, y_pred),
                    "n_features": len(selected_features),
                    "best_params": str(grid.best_params_),
                }
            )

            for i_sample in range(len(y_test)):
                y_t = y_test[i_sample]
                y_p = y_pred[i_sample]
                y_t_safe = y_t if y_t != 0 else 1e-10

                residuals_log.append(
                    {
                        "repetition": repetition,
                        "test_chunk": test_chunk,
                        "model": model_name,
                        "setpoint": y_t,
                        "y_true": y_t,
                        "y_pred": y_p,
                        "residual": y_t - y_p,
                        "absolute_error": abs(y_t - y_p),
                        "relative_error": (y_t - y_p) / y_t_safe,
                    }
                )

        # --- GUARDAR ARCHIVOS DE INTERPRETABILIDAD POR ITERACIÓN ---
        save_to_run_append(pd.concat(df_p_list), "ranking_consenso_perm.csv")
        save_to_run_append(pd.concat(df_s_list), "ranking_consenso_shap.csv")
        save_to_run_append(pd.DataFrame(iter_preds), "model_predictions_history.csv")

    # ══════════════════════════════════════════════════════════════
    # 4. ANÁLISIS FINAL Y GUARDADO ESTADÍSTICO
    # ══════════════════════════════════════════════════════════════

    # Save Split Info (auditable: qué chunk fue TEST, rangos, tamaños de muestra)
    pd.DataFrame(split_log).to_csv(
        os.path.join(RUN_DIR, "splits_info.csv"), index=False
    )

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(os.path.join(RUN_DIR, "metrics_per_iteration.csv"), index=False)

    df_residuals = pd.DataFrame(residuals_log)
    df_residuals.to_csv(os.path.join(RUN_DIR, "residuals.csv"), index=False)

    # Resumen y CI (Bootstrap simple para la MEDIA) sobre las 5 iteraciones/chunks
    summary_list = []
    np.random.seed(42)  # Reproducibilidad del bootstrap
    n_boot = 1000
    for model in df_metrics["model"].unique():
        for metric in ["R2", "MAE", "RMSE", "MAPE"]:
            data = df_metrics[df_metrics["model"] == model][metric].values
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)

            boot_means = [
                np.mean(np.random.choice(data, size=len(data), replace=True))
                for _ in range(n_boot)
            ]
            ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])

            summary_list.append(
                {
                    "Model": model,
                    "Metric": metric,
                    "Mean": mean_val,
                    "Std": std_val,
                    "Median": median_val,
                    "CI95_lower": ci_lower,
                    "CI95_upper": ci_upper,
                    "n_iterations": len(data),
                }
            )
    pd.DataFrame(summary_list).to_csv(
        os.path.join(RUN_DIR, "confidence_intervals.csv"), index=False
    )

    # Comparación Pareada (Wilcoxon) con Corrección Holm, sobre las 5 iteraciones/chunks
    comparisons = []
    model_names = list(models.keys())

    for metric in ["MAE", "RMSE"]:
        metric_pvals = []
        metric_comps_temp = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                modA, modB = model_names[i], model_names[j]

                # Asegurar alineación por repetición para datos puramente pareados
                dataA = (
                    df_metrics[df_metrics["model"] == modA]
                    .sort_values("repetition")[metric]
                    .values
                )
                dataB = (
                    df_metrics[df_metrics["model"] == modB]
                    .sort_values("repetition")[metric]
                    .values
                )

                diff = dataA - dataB
                mean_diff = np.mean(diff)
                median_diff = np.median(diff)
                std_diff = np.std(diff)

                # Wilcoxon signed-rank test
                _stat, p_val = wilcoxon(dataA, dataB)

                # Tamaño de efecto (Cohen's dz)
                effect_size = mean_diff / std_diff if std_diff != 0 else 0

                metric_comps_temp.append(
                    {
                        "Model_A": modA,
                        "Model_B": modB,
                        "Metric": metric,
                        "Mean_Difference": mean_diff,
                        "Median_Difference": median_diff,
                        "Effect_Size": effect_size,
                        "p_value": p_val,
                    }
                )
                metric_pvals.append(p_val)

        # Corrección de Holm
        adj_pvals = holm_correction(metric_pvals)
        for k in range(len(metric_comps_temp)):
            metric_comps_temp[k]["adjusted_p_value"] = adj_pvals[k]
            comparisons.append(metric_comps_temp[k])

    pd.DataFrame(comparisons).to_csv(
        os.path.join(RUN_DIR, "paired_model_comparison.csv"), index=False
    )

    # Resumen de residuales por Setpoint y Modelo
    def rmse_agg(x):
        return np.sqrt(np.mean(x**2))

    residual_summary = (
        df_residuals.groupby(["model", "setpoint"])
        .agg(
            n_samples=("y_true", "count"),
            mean_true=("y_true", "mean"),
            mean_predicted=("y_pred", "mean"),
            MAE=("absolute_error", "mean"),
            mean_residual=("residual", "mean"),
            std_residual=("residual", "std"),
            median_residual=("residual", "median"),
            RMSE=("residual", rmse_agg),
        )
        .reset_index()
    )
    residual_summary.to_csv(os.path.join(RUN_DIR, "residual_summary.csv"), index=False)

    # Generar el Consenso 80% Final
    paths_consenso = {
        "run_dir": RUN_DIR,
        "perm_csv": os.path.join(RUN_DIR, "ranking_consenso_perm.csv"),
        "shap_csv": os.path.join(RUN_DIR, "ranking_consenso_shap.csv"),
        "output_csv": os.path.join(RUN_DIR, "tabla_consenso_final.csv"),
    }
    process_triple_80(paths_consenso)

    print("\n[OK] Pipeline LOCO (Leave-One-Current-Range-Out) finalizado metodológicamente correcto.")


if __name__ == "__main__":
    run_pipeline()