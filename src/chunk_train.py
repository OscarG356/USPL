"""
Machine learning pipeline for pump current estimation using TSFEL features.

The pipeline trains four regression models:
*   - Support Vector Regression (SVR)
*   - Random Forest
*   - Bayesian Ridge
*   - XGBoost

SHAP and permutation importance are used to identify and compare
the most important features for each model.
"""

import argparse
import datetime

# import gc
import os
import re

# pyrefly: ignore [missing-import]
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
# pyrefly: ignore [missing-import]
# pyrefly: ignore [missing-import]
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt

# pyrefly: ignore [missing-import]
# import seaborn as sns

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

# --------------------------------------------------------------
# 1. Configuration
# --------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="USPL Pump Current Estimation Pipeline - Random Chunk"
)

parser.add_argument(
    "--uspl",
    type=int,
    default=1,
    help="Laser ID (1 or 2)",
)

parser.add_argument(
    "--iters",
    type=int,
    default=5,
    help="Number of iterations to run",
)

parser.add_argument(
    "--feature_method",
    type=str,
    default="all",
    help="Feature domain to use: temporal, statistical, spectral, or all",
)

parser.add_argument(
    "--threshold_corr",
    type=float,
    default=0.90,
    help="Correlation threshold for feature elimination",
)


parser.add_argument(
    "--operation_regime",
    type=str,
    default="mode-locking",
    help="Operating regime to train: mode-locking or supercontinuum",
)

args = parser.parse_args()

USPL_ID = f"USPL_{args.uspl}"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005

RAW_DIR = os.path.join(
    BASE_DIR, "data", f"data_{USPL_ID}", "raw", args.operation_regime, "temporal"
)
RUN_DIR = os.path.join(
    BASE_DIR,
    "data",
    f"data_{USPL_ID}",
    "outputs",
    args.feature_method,
    f"run_{TIMESTAMP}",
)
os.makedirs(RUN_DIR, exist_ok=True)

print(f"--- Starting Random Chunk Pipeline for: {USPL_ID} ---")
print(f"--- Outputs: {RUN_DIR} ---")


N_CHUNKS = 10  # 10 contiguous chunks: 8 for TRAIN, 2 randomly selected for TEST (~80/20)
N_TEST_CHUNKS = 2  # number of chunks randomly selected as TEST at each iteration

# --------------------------------------------------------------
# 2. Raw Data Preprocessing
# --------------------------------------------------------------


def natural_sort_key(s: str) -> list:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def load_raw_signals(data_folder: str, regime: str):
    """
    Load and cut RAW signals

    * current_chunk and trace_id are heredated by the segmented signal
    """
    signal_files = [
        f
        for f in os.listdir(data_folder)
        if f.lower().endswith(".csv") and not f.endswith("(1).CSV")
    ]
    signal_files.sort(key=natural_sort_key)

    if regime == 'mode-locking':
        extractor = lambda df: df.iloc[:, 1].astype(float)
    else:
        extractor = lambda df: df["intensity"].astype(float)       

    all_signals_data = []
    for file_name in signal_files:
        file_path = os.path.join(data_folder, file_name)
        signal_df = pd.read_csv(file_path, header=0)
        all_signals_data.append(extractor(signal_df))
        #all_signals_data.append(signal_df["intensity"].astype(float))
        #all_signals_data.append(signal_df.iloc[:, 1].astype(float))

    signals_matrix = pd.DataFrame(all_signals_data).values

    # Get fs
    example_df = pd.read_csv(os.path.join(data_folder, signal_files[0]), header=0)
    fs = 1 / (example_df.iloc[1, 0] - example_df.iloc[0, 0])

    # Labels
    labels_df = pd.read_excel(os.path.join(data_folder, "Datos-Corriente.xlsx"))
    if regime == "mode-locking":
        target_raw = labels_df["Corriente (mA)"].values[:]
    else:
        target_raw = labels_df["Ganancia-EDFA (dBm)"].values[:]

    # Segmentation
    num_traces = len(signals_matrix)

    if regime == "mode-locking":
        segment_length = 200
        num_segments = 3
    else:
        segment_length = signals_matrix.shape[1]
        num_segments = 1

    # Chunk division: contiguous blocks in the original (physical/current) order.
    # NOTE: this assignment is not used for the actual TRAIN/TEST split (that is
    # recomputed in run_pipeline from trace_id, once, before the iteration loop).
    # It is kept here only so the returned `segmented_chunks` array is consistent
    # with the real contiguous-chunk scheme instead of a leftover random one.
    chunk_assignment = np.zeros(num_traces, dtype=int)
    for chunk_id, idx_block in enumerate(
        np.array_split(np.arange(num_traces), N_CHUNKS), start=1
    ):
        chunk_assignment[idx_block] = chunk_id

    segmented_signals = []
    segmented_labels = []
    segmented_chunks = []
    segmented_trace_ids = []

    for idx, signal_array in enumerate(signals_matrix):
        target_value = float(target_raw[idx])
        chunk_id = int(chunk_assignment[idx])

        if len(signal_array) >= num_segments * segment_length:
            for i in range(num_segments):
                segmented_signals.append(
                    signal_array[i * segment_length : (i + 1) * segment_length]
                )
                segmented_labels.append(target_value)
                segmented_chunks.append(chunk_id)
                segmented_trace_ids.append(idx)

    return (
        np.array(segmented_signals),
        np.array(segmented_labels),
        np.array(segmented_chunks),
        np.array(segmented_trace_ids),
        fs,
    )


def create_contiguous_chunks(num_traces: int, n_chunks: int) -> np.ndarray:
    """
    Split `num_traces` traces (already in their original, physically-ordered
    sequence) into `n_chunks` contiguous, approximately-equal-sized blocks.

    Returns an array of length `num_traces` with the 1-indexed chunk id of
    each trace (position i -> chunk assigned to trace i). No shuffling is
    performed: trace i and trace i+1 are always in the same chunk unless
    trace i is the last element of its block.
    """
    chunk_assignment = np.zeros(num_traces, dtype=int)
    for chunk_id, idx_block in enumerate(
        np.array_split(np.arange(num_traces), n_chunks), start=1
    ):
        chunk_assignment[idx_block] = chunk_id
    return chunk_assignment


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def fit_feature_selection(X_train_df, threshold_corr=0.90, epsilon=1e-10):
    """Select features using variance and correlation filtering."""
    # 1. Variance
    variances = X_train_df.var()
    keep_var = variances[variances > epsilon].index.tolist()
    X_train_var = X_train_df[keep_var]

    # 2. Correlation
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
    """Apply the Holm-Bonferroni correction for multiple comparisons."""
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
    """Calculate feature importance consensus using permutation importance, SHAP."""
    print(
        f"\n--- Calculating Triple 80% Consensus in: {os.path.basename(paths['run_dir'])} ---"
    )

    if not os.path.exists(paths["perm_csv"]) or not os.path.exists(paths["shap_csv"]):
        print("[ERROR] Required files for consensus calculation are missing.")
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


# --------------------------------------------------------------
# 3. Main pipeline
# --------------------------------------------------------------


def run_pipeline():
    X_raw, y_raw, _chunk_raw_dummy, trace_id_raw, fs = load_raw_signals(
        RAW_DIR, args.operation_regime
    )
    domain = args.feature_method

    if domain == "all":
        cfg = tsfel.get_features_by_domain(domain=None)
    else:
        selected_domains = [d.strip() for d in domain.split(',') if d.strip()]
        
        # Validación rápida
        valid_domains = {'temporal', 'statistical', 'spectral', 'fractal'}
        for d in selected_domains:
            if d not in valid_domains:
                raise ValueError(f"Dominio '{d}' no válido. Usa: {valid_domains}")
        
        # ! IMPORTANTE: TSFEL solo acepta UN dominio, no una lista
        if len(selected_domains) == 1:
            cfg = tsfel.get_features_by_domain(domain=selected_domains[0])
        else:
            cfg = {}
            for d in selected_domains:
                cfg.update(tsfel.get_features_by_domain(domain=d))

    metrics_log, residuals_log, split_log = [], [], []
    N_ITERS = args.iters

    # --------------------------------------------------------------
    # Contiguous chunk assignment (Random Contiguous Chunk Split)
    # --------------------------------------------------------------
    # The chunks themselves are NOT random: they are N_CHUNKS consecutive
    # blocks of traces following the original (physically-ordered) sequence
    # of the dataset. This is computed only once, before the iteration loop,
    # because the chunk boundaries never change across iterations - only the
    # selection of which chunks act as TEST changes per iteration below.
    unique_traces = np.unique(trace_id_raw)
    num_traces = len(unique_traces)

    chunks_array = create_contiguous_chunks(num_traces, N_CHUNKS)
    trace_to_chunk = dict(zip(unique_traces, chunks_array))
    chunk_raw = np.array([trace_to_chunk[t_id] for t_id in trace_id_raw])

    # Sanity check: every chunk must be made of index-contiguous traces
    # (i.e. no interleaving/shuffling was introduced while building chunks).
    for chunk_id in range(1, N_CHUNKS + 1):
        traces_in_chunk = np.sort(unique_traces[chunks_array == chunk_id])
        if len(traces_in_chunk) > 1:
            assert np.all(np.diff(traces_in_chunk) == 1), (
                f"FATAL ERROR: Chunk {chunk_id} is not made of contiguous traces."
            )

    for iteration in range(N_ITERS):
        repetition = iteration + 1

        # --- Random selection of TEST chunks (only source of randomness) ---
        # A fresh, reproducible-but-different RNG per iteration: same
        # `repetition` always yields the same selection, but different
        # iterations yield different selections.
        rng = np.random.default_rng(42 + repetition)
        test_chunks = sorted(
            rng.choice(
                np.arange(1, N_CHUNKS + 1), size=N_TEST_CHUNKS, replace=False
            ).tolist()
        )
        train_chunks = [c for c in range(1, N_CHUNKS + 1) if c not in test_chunks]

        # --- Train/Test Split ---
        train_mask = np.isin(chunk_raw, train_chunks)
        test_mask = np.isin(chunk_raw, test_chunks)

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
        print(f"TEST CHUNKS: {test_chunks}")
        print(f"TEST RANGE: {test_min:.2f}–{test_max:.2f} mA")
        print(f"TRAIN CHUNKS: {train_chunks}")
        print(f"{'=' * 50}")

        # --- Leakage Checks ---
        assert len(test_chunks) == N_TEST_CHUNKS, (
            "FATAL ERROR: TEST does not contain exactly the expected number of chunks."
        )

        assert len(train_chunks) == N_CHUNKS - N_TEST_CHUNKS, (
            "FATAL ERROR: TRAIN does not contain exactly the expected number of chunks."
        )

        assert len(set(train_chunks) & set(test_chunks)) == 0, (
            "FATAL ERROR: Chunk leakage detected between TRAIN and TEST."
        )

        assert set(np.unique(chunk_train)) == set(train_chunks), (
            "FATAL ERROR: TRAIN does not contain exactly the expected chunks."
        )

        assert set(np.unique(chunk_test)) == set(test_chunks), (
            "FATAL ERROR: TEST does not correspond exactly to the expected chunks."
        )

        assert len(set(trace_id_train) & set(trace_id_test)) == 0, (
            "FATAL ERROR: Trace leakage detected between TRAIN and TEST."
        )

        split_log.append(
            {
                "repetition": repetition,
                "test_chunks": ",".join(map(str, test_chunks)),
                "train_chunks": ",".join(map(str, train_chunks)),
                "test_current_min": test_min,
                "test_current_max": test_max,
                "train_current_min": float(y_train.min()),
                "train_current_max": float(y_train.max()),
                "n_train_samples": len(y_train),
                "n_test_samples": len(y_test),
            }
        )

        # --- Train-Based Signal Normalization ---
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

        # --- FEATURE SELECTION (TRAIN) ---
        selected_features = fit_feature_selection(
            X_train_feat, threshold_corr=args.threshold_corr
        )
        X_train_sel, X_test_sel = (
            X_train_feat[selected_features].values,
            X_test_feat[selected_features].values,
        )

        print(f"Features: {X_train_feat.shape[1]} -> {len(selected_features)} selected")

        # --- Model Training and Explainability ---
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
                grid.best_estimator_, X_test_sel, y_test, n_repeats=75, n_jobs=-1
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
            df_p["test_chunks"] = ",".join(map(str, test_chunks))
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
            df_s["test_chunks"] = ",".join(map(str, test_chunks))
            df_s_list.append(df_s)

            # Metrics
            metrics_log.append(
                {
                    "repetition": repetition,
                    "test_chunks": ",".join(map(str, test_chunks)),
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
                        "test_chunks": ",".join(map(str, test_chunks)),
                        "model": model_name,
                        "setpoint": y_t,
                        "y_true": y_t,
                        "y_pred": y_p,
                        "residual": y_t - y_p,
                        "absolute_error": abs(y_t - y_p),
                        "relative_error": (y_t - y_p) / y_t_safe,
                    }
                )

        # --- Save files ---
        save_to_run_append(pd.concat(df_p_list), "ranking_consenso_perm.csv")
        save_to_run_append(pd.concat(df_s_list), "ranking_consenso_shap.csv")
        save_to_run_append(pd.DataFrame(iter_preds), "model_predictions_history.csv")

    # --------------------------------------------------------------
    # 4. Final Statistical Analysis
    # --------------------------------------------------------------

    # Save Split Info
    pd.DataFrame(split_log).to_csv(
        os.path.join(RUN_DIR, "splits_info.csv"), index=False
    )

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(os.path.join(RUN_DIR, "metrics_per_iteration.csv"), index=False)

    df_residuals = pd.DataFrame(residuals_log)
    df_residuals.to_csv(os.path.join(RUN_DIR, "residuals.csv"), index=False)

    # Sumary
    summary_list = []
    # //np.random.seed(42)
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
                    "Mean": mean_val,
                    "Std": std_val,
                    "Metric": metric,
                    "Median": median_val,
                    "CI95_lower": ci_lower,
                    "CI95_upper": ci_upper,
                    "n_iterations": len(data),
                }
            )
    pd.DataFrame(summary_list).to_csv(
        os.path.join(RUN_DIR, "confidence_intervals.csv"), index=False
    )

    # Comparision between models
    comparisons = []
    model_names = list(models.keys())

    for metric in ["MAE", "RMSE"]:
        metric_pvals = []
        metric_comps_temp = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                modA, modB = model_names[i], model_names[j]

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

        # Holm corrections
        adj_pvals = holm_correction(metric_pvals)
        for k in range(len(metric_comps_temp)):
            metric_comps_temp[k]["adjusted_p_value"] = adj_pvals[k]
            comparisons.append(metric_comps_temp[k])

    pd.DataFrame(comparisons).to_csv(
        os.path.join(RUN_DIR, "paired_model_comparison.csv"), index=False
    )

    # Residual summary
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

    # Generate final concensum
    paths_consenso = {
        "run_dir": RUN_DIR,
        "perm_csv": os.path.join(RUN_DIR, "ranking_consenso_perm.csv"),
        "shap_csv": os.path.join(RUN_DIR, "ranking_consenso_shap.csv"),
        "output_csv": os.path.join(RUN_DIR, "tabla_consenso_final.csv"),
    }
    process_triple_80(paths_consenso)

    print("\n[OK] Pipeline finish.")


if __name__ == "__main__":
    run_pipeline()