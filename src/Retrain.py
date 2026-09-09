"""
Experiment: Training models using the Top-80% most important features
(Random Contiguous Chunk Split methodology)

This script replicates EXACTLY the data loading, contiguous-chunk
construction, random TEST-chunk selection, leakage checks, TRAIN-only
normalization, TSFEL feature extraction and model/CV setup of the main
pipeline ("Random Contiguous Chunk Split"). It evaluates the impact of
reducing the number of input features by training each model using only
the subset of features that accounts for 80% of the cumulative importance,
based on previously computed SHAP and Permutation Importance rankings from
that main pipeline.

IMPORTANT: This version does NOT apply variance or correlation filtering.
It uses ALL features from the Top-80% ranking that are present in the
TSFEL-extracted common columns. This allows evaluation of the Top-80%
feature set without additional feature reduction steps.

The script requires, inside --original_run_dir:
1. 'ranking_consenso_perm.csv' and 'ranking_consenso_shap.csv' as the
   default sources for model-specific feature importance rankings.
2. 'tabla_consenso_final.csv' (optional, only used if it contains a
   'model' column and a 'Consensus_General' column).
3. 'metrics_per_iteration.csv' (optional). If available, it is used to
   compare the Top-80% results with the original ALL-FEATURES experiment.
   If it is not available, only the Top-80% results are generated.

SHAP and Permutation Importance are NOT recalculated here; the rankings
are loaded from a previous run of the main pipeline.

The ONLY methodological difference versus the main pipeline is: after
TSFEL feature extraction (and common column intersection between train
and test), each model uses ALL features from its Top-80% ranking that
are present in the common columns. No variance or correlation filtering
is applied.

NOTE ON WHAT WAS NOT PROVIDED: the exact `load_raw_signals` implementation
of the main pipeline (column name used for the signal, i.e. `intensity`,
target column names per regime, and the file-based `.CSV`/`.xlsx` loading
logic) was given to me directly in this conversation as part of the main
"Random Contiguous Chunk Split" pipeline script, and is reproduced here
verbatim so both scripts stay perfectly comparable. I did not have to
invent it. If your main pipeline's `load_raw_signals` has since changed,
this function must be updated to match it again.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys

import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
import tsfel
from sklearn.ensemble import RandomForestRegressor
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
    description=("Training models using the Top-80% most important features.")
)
parser.add_argument("--uspl", type=int, default=1, help="Laser ID (1 or 2)")
parser.add_argument("--iters", type=int, default=5, help="Number of iterations to run.")
parser.add_argument(
    "--feature_method",
    type=str,
    default="all",
    help=("Feature domain to use: temporal, statistical, spectral, or all."),
)
parser.add_argument(
    "--top80_threshold",
    type=float,
    default=0.80,
    help="Threshold for cumulative importance Top-N%.",
)
parser.add_argument(
    "--original_run_dir",
    type=str,
    required=True,
    help=("Path with the previous complete run of the main pipeline."),
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
    f"run_{TIMESTAMP}_top80",
)
os.makedirs(RUN_DIR, exist_ok=True)

MODEL_NAME_TO_FILESAFE = {
    "SVR": "SVR",
    "Random Forest": "Random_Forest",
    "Bayesian Ridge": "Bayesian_Ridge",
    "XGBoost": "XGBoost",
}

# Random Contiguous Chunk Split parameters (must match the main pipeline).
N_CHUNKS = 10
N_TEST_CHUNKS = 2

print(f"--- Top-{args.top80_threshold:.0%} Feature Experiment: {USPL_ID} ---")
print(f"--- Original Run Directory: {args.original_run_dir} ---")
print(f"--- New Output Directory: {RUN_DIR} ---")

# --------------------------------------------------------------
# 2. FUNCIONES AUXILIARES (idénticas al pipeline original donde aplica)
# --------------------------------------------------------------


def natural_sort_key(s: str) -> list:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def create_contiguous_chunks(num_traces: int, n_chunks: int) -> np.ndarray:
    """
    Split `num_traces` traces (already in their original, physically-ordered
    sequence) into `n_chunks` contiguous, approximately-equal-sized blocks.

    Returns an array of length `num_traces` with the 1-indexed chunk id of
    each trace. No shuffling is performed: trace i and trace i+1 are always
    in the same chunk unless trace i is the last element of its block.

    Identical to the main pipeline's `create_contiguous_chunks`.
    """
    chunk_assignment = np.zeros(num_traces, dtype=int)
    for chunk_id, idx_block in enumerate(
        np.array_split(np.arange(num_traces), n_chunks), start=1
    ):
        chunk_assignment[idx_block] = chunk_id
    return chunk_assignment


def load_raw_signals(data_folder: str, regime: str):
    """
    Load and cut RAW signals. Identical to the main "Random Contiguous
    Chunk Split" pipeline, so that trace_id, chunk_id and target are
    guaranteed to match exactly between the two scripts.

    * chunk_id returned here is a placeholder built from
      `create_contiguous_chunks` purely for structural consistency; the
      actual TRAIN/TEST chunk assignment used by the pipeline is
      recomputed once from `trace_id` in `run_pipeline`, before the
      iteration loop (see there).
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
        all_signals_data.append(signal_df["intensity"].astype(float))
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

    # Placeholder contiguous chunk assignment (not used for the actual
    # split, see docstring above).
    chunk_assignment = create_contiguous_chunks(num_traces, N_CHUNKS)

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


def mape(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100


def save_to_run_append(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(RUN_DIR, filename)
    if os.path.exists(path):
        pd.concat([pd.read_csv(path), df], ignore_index=True).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


# --------------------------------------------------------------
# 3. CONSTRUCCIÓN DEL CONJUNTO TOP-N% POR MODELO
# --------------------------------------------------------------


def _normalize_per_model(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    for mod in df["model"].unique():
        mask = df["model"] == mod
        vals = df.loc[mask, col]
        df.loc[mask, col] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-10)
    return df


def _select_top_n_percent(sub: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Aplica exactamente el criterio del pipeline original:
    1. Ordenar de mayor a menor importancia.
    2. Calcular contribución acumulada (relativa al total del modelo).
    3. Seleccionar features cuya contribución acumulada permite alcanzar
       el umbral (Top80 = cumsum_previo < threshold).
    4. Mantener al menos la primera feature.
    """
    sub = sub.sort_values("Consensus_General", ascending=False).reset_index(drop=True)
    total = sub["Consensus_General"].sum()
    if total <= 0:
        # Caso extremo: todas las importancias son 0 (o negativas tras
        # normalización). Se conserva únicamente la primera feature.
        sub["cumulative_importance"] = np.nan
        mask = pd.Series([False] * len(sub))
        mask.iloc[0] = True
    else:
        sub["cumulative_importance"] = (sub["Consensus_General"] / total).cumsum()
        cum_prev = sub["cumulative_importance"].shift(1).fillna(0)
        mask = cum_prev < threshold
        if not mask.any():
            mask.iloc[0] = True
    sub["rank"] = sub.index + 1
    sub["importance"] = sub["Consensus_General"]
    return sub.loc[
        mask, ["model", "feature", "rank", "importance", "cumulative_importance"]
    ]


def build_top80_from_consensus_table(consenso_csv: str, threshold: float):
    """Intenta usar tabla_consenso_final.csv SOLO si trae columna 'model'."""
    if not os.path.exists(consenso_csv):
        return None
    df = pd.read_csv(consenso_csv)
    if "model" not in df.columns:
        print(
            "[INFO] tabla_consenso_final.csv no contiene columna 'model' "
            "(es un consenso global). NO se usará como fuente del Top-"
            f"{threshold:.0%} por modelo, para evitar asumir que el "
            "consenso global equivale al de cada modelo individual."
        )
        return None

    if "Consensus_General" not in df.columns:
        print(
            "[INFO] tabla_consenso_final.csv tiene columna 'model' pero no "
            "'Consensus_General'; se ignora esta fuente y se usan los "
            "rankings de Permutation/SHAP en su lugar."
        )
        return None

    print("[OK] Usando tabla_consenso_final.csv (con columna 'model') como fuente.")
    results = [
        _select_top_n_percent(df[df["model"] == mod].copy(), threshold)
        for mod in df["model"].unique()
    ]
    return pd.concat(results, ignore_index=True)


def build_top80_from_perm_shap(perm_csv: str, shap_csv: str, threshold: float):
    if not os.path.exists(perm_csv) or not os.path.exists(shap_csv):
        raise FileNotFoundError(
            "No se encontraron ranking_consenso_perm.csv y/o "
            "ranking_consenso_shap.csv en --original_run_dir. Estos archivos "
            "son obligatorios (no se recalculan SHAP/Permutation en este "
            f"script). Buscados en:\n  {perm_csv}\n  {shap_csv}"
        )

    df_p = pd.read_csv(perm_csv)  # feature, importance_mean, model, rank_perm, ...
    df_s = pd.read_csv(shap_csv)  # feature, shap_importance, model, rank_shap, ...

    required_p = {"feature", "importance_mean", "model"}
    required_s = {"feature", "shap_importance", "model"}
    if not required_p.issubset(df_p.columns):
        raise ValueError(
            f"ranking_consenso_perm.csv no tiene las columnas esperadas "
            f"{required_p}. Columnas encontradas: {list(df_p.columns)}"
        )
    if not required_s.issubset(df_s.columns):
        raise ValueError(
            f"ranking_consenso_shap.csv no tiene las columnas esperadas "
            f"{required_s}. Columnas encontradas: {list(df_s.columns)}"
        )

    print(
        "[OK] Usando ranking_consenso_perm.csv + ranking_consenso_shap.csv "
        "(promediados por modelo/feature a través de las repeticiones)."
    )

    perm_avg = (
        df_p.groupby(["model", "feature"])["importance_mean"].mean().reset_index()
    )
    shap_avg = (
        df_s.groupby(["model", "feature"])["shap_importance"].mean().reset_index()
    )

    perm_avg = _normalize_per_model(perm_avg, "importance_mean")
    shap_avg = _normalize_per_model(shap_avg, "shap_importance")

    merged = pd.merge(perm_avg, shap_avg, on=["model", "feature"], how="outer")
    merged["importance_mean"] = merged["importance_mean"].fillna(0)
    merged["shap_importance"] = merged["shap_importance"].fillna(0)
    merged["Consensus_General"] = (
        merged["importance_mean"] + merged["shap_importance"]
    ) / 2

    results = [
        _select_top_n_percent(merged[merged["model"] == mod].copy(), threshold)
        for mod in merged["model"].unique()
    ]
    return pd.concat(results, ignore_index=True)


def get_top_n_percent_features(original_run_dir: str, threshold: float) -> pd.DataFrame:
    consenso_csv = os.path.join(original_run_dir, "tabla_consenso_final.csv")
    perm_csv = os.path.join(original_run_dir, "ranking_consenso_perm.csv")
    shap_csv = os.path.join(original_run_dir, "ranking_consenso_shap.csv")

    df_top = build_top80_from_consensus_table(consenso_csv, threshold)
    if df_top is None:
        df_top = build_top80_from_perm_shap(perm_csv, shap_csv, threshold)

    df_top = df_top.sort_values(["model", "rank"]).reset_index(drop=True)
    return df_top


# --------------------------------------------------------------
# 4. MODELOS (idénticos al pipeline original)
# --------------------------------------------------------------


def build_model_specs():
    return {
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


# --------------------------------------------------------------
# 5. PIPELINE PRINCIPAL
# --------------------------------------------------------------


def run_pipeline():
    # --- 5.1 Obtener Top-N% por modelo (sin recalcular interpretabilidad) ---
    df_top = get_top_n_percent_features(args.original_run_dir, args.top80_threshold)

    models = build_model_specs()
    model_names = list(models.keys())

    missing_models = [m for m in model_names if m not in df_top["model"].unique()]
    if missing_models:
        raise ValueError(
            f"Los siguientes modelos no tienen features Top-"
            f"{args.top80_threshold:.0%} disponibles en los rankings "
            f"cargados: {missing_models}. Verifica que "
            f"--original_run_dir corresponda a una corrida completa del "
            f"pipeline original con estos modelos."
        )

    top_features_per_model = {
        m: df_top[df_top["model"] == m].sort_values("rank")["feature"].tolist()
        for m in model_names
    }

    print(
        f"\n--- Features Top-{args.top80_threshold:.0%} por modelo (fuente: interpretabilidad previa) ---"
    )
    for m in model_names:
        feats = top_features_per_model[m]
        print(f"{m}: {len(feats)} features Top-{args.top80_threshold:.0%}")
        for f in feats:
            print(f"    - {f}")

    # Guardar archivo combinado
    df_top.to_csv(os.path.join(RUN_DIR, "features_top80_per_model.csv"), index=False)

    # Guardar archivos individuales por modelo
    for m in model_names:
        fname = f"features_top80_{MODEL_NAME_TO_FILESAFE[m]}.csv"
        df_top[df_top["model"] == m].to_csv(os.path.join(RUN_DIR, fname), index=False)

    # --- 5.2 Cargar datos crudos (idéntico al pipeline original) ---
    X_raw, y_raw, _chunk_raw_dummy, trace_id_raw, fs = load_raw_signals(
        RAW_DIR, args.operation_regime
    )
    domain = args.feature_method
    if domain == "all":
        domain = None
    cfg = tsfel.get_features_by_domain(domain)

    metrics_log, residuals_log, split_log = [], [], []
    iter_pred_records = []
    N_ITERS = args.iters

    # --------------------------------------------------------------
    # Contiguous chunk assignment (Random Contiguous Chunk Split)
    # --------------------------------------------------------------
    # Computed once, before the iteration loop: the chunk boundaries never
    # change across iterations, only the random selection of which 2 chunks
    # act as TEST changes per iteration (identical to the main pipeline).
    unique_traces = np.unique(trace_id_raw)
    num_traces = len(unique_traces)

    chunks_array = create_contiguous_chunks(num_traces, N_CHUNKS)
    trace_to_chunk = dict(zip(unique_traces, chunks_array))
    chunk_raw = np.array([trace_to_chunk[t_id] for t_id in trace_id_raw])

    # Sanity check: every chunk must be made of index-contiguous traces.
    for chunk_id in range(1, N_CHUNKS + 1):
        traces_in_chunk = np.sort(unique_traces[chunks_array == chunk_id])
        if len(traces_in_chunk) > 1:
            assert np.all(np.diff(traces_in_chunk) == 1), (
                f"FATAL ERROR: Chunk {chunk_id} is not made of contiguous traces."
            )

    for iteration in range(N_ITERS):
        repetition = iteration + 1

        # --- Random selection of TEST chunks (only source of randomness) ---
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

        # --- Leakage Checks (idénticos al pipeline original) ---
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

        # --- EXTRACCIÓN TSFEL (idéntica al pipeline original) ---
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
        print(f"Features extraídas (TSFEL, comunes train/test): {len(common_cols)}")

        iter_preds = {"repetition": repetition, "Actual_Current": y_test}

        for model_name, (estimator, param_grid) in models.items():
            top_feats_for_model = top_features_per_model[model_name]

            # A Top-80% feature that does not even exist after TSFEL /
            # common_cols is a data/config incompatibility, not a normal
            # variance/correlation exclusion: fail loudly and clearly.
            missing_from_tsfel = [
                f for f in top_feats_for_model if f not in common_cols
            ]
            if missing_from_tsfel:
                raise ValueError(
                    f"[ERROR] Iteración {repetition} / Modelo '{model_name}': "
                    f"las siguientes features Top-{args.top80_threshold:.0%} "
                    f"del ranking de interpretabilidad NO existen entre las "
                    f"features extraídas por TSFEL en esta corrida (tras la "
                    f"intersección train/test): {missing_from_tsfel}. Esto "
                    f"sugiere una incompatibilidad entre --feature_method / "
                    f"--operation_regime / datos crudos usados aquí y los "
                    f"usados para generar --original_run_dir. Se detiene la "
                    f"ejecución en lugar de eliminarlas silenciosamente."
                )

            # Usar TODAS las features Top-80% que existen en common_cols.
            # No se aplica filtro de varianza ni de correlación.
            final_features = [f for f in top_feats_for_model if f in common_cols]
            if not final_features:
                raise ValueError(
                    f"[ERROR] Iteración {repetition} / Modelo '{model_name}': "
                    f"no se encontraron features del Top-"
                    f"{args.top80_threshold:.0%} en los datos TSFEL comunes. "
                    f"No hay features disponibles para entrenar."
                )

            X_train_sel = X_train_feat[final_features].values
            X_test_sel = X_test_feat[final_features].values

            # --- ENTRENAMIENTO (idéntico al pipeline original) ---
            pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            inner_cv = GroupKFold(n_splits=4)
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

            metrics_log.append(
                {
                    "repetition": repetition,
                    "model": model_name,
                    "test_chunks": ",".join(map(str, test_chunks)),
                    "train_chunks": ",".join(map(str, train_chunks)),
                    "test_current_min": test_min,
                    "test_current_max": test_max,
                    "R2": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAPE": mape(y_test, y_pred),
                    "n_features": len(final_features),
                    "features_used": ";".join(final_features),
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

        iter_pred_records.append(iter_preds)

    # --------------------------------------------------------------
    # 6. GUARDADO DE ARCHIVOS PRINCIPALES
    # --------------------------------------------------------------

    pd.DataFrame(split_log).to_csv(
        os.path.join(RUN_DIR, "splits_info.csv"), index=False
    )

    df_metrics = pd.DataFrame(metrics_log)
    df_metrics.to_csv(
        os.path.join(RUN_DIR, "metrics_per_iteration_top80.csv"), index=False
    )

    df_residuals = pd.DataFrame(residuals_log)
    df_residuals.to_csv(os.path.join(RUN_DIR, "residuals_top80.csv"), index=False)

    df_preds = pd.DataFrame(iter_pred_records)
    save_to_run_append(df_preds, "model_predictions_history_top80.csv")

    # --- Resumen y CI 95% (bootstrap, idéntico al pipeline original: sin
    # semilla global fija, n_boot=1000) ---
    summary_list = []
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
                    "Feature_Set": "Top80",
                    "Metric": metric,
                    "Mean": mean_val,
                    "Std": std_val,
                    "Median": median_val,
                    "CI95_lower": ci_lower,
                    "CI95_upper": ci_upper,
                    "n_iterations": len(data),
                }
            )
    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(
        os.path.join(RUN_DIR, "confidence_intervals_top80.csv"), index=False
    )

    # Resumen de residuales por setpoint y modelo
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
    residual_summary.to_csv(
        os.path.join(RUN_DIR, "residual_summary_top80.csv"), index=False
    )

    # --------------------------------------------------------------
    # 7. COMPARACIÓN ALL vs TOP-80% (si el experimento original está disponible)
    # --------------------------------------------------------------

    all_metrics_path = os.path.join(args.original_run_dir, "metrics_per_iteration.csv")
    comparison_rows = []
    all_available = os.path.exists(all_metrics_path)

    def agg_stats(df, model, metric):
        vals = df[df["model"] == model][metric].values
        return np.mean(vals), np.std(vals)

    top80_stats = {}
    for model in model_names:
        top80_stats[model] = {
            "n_features": df_metrics[df_metrics["model"] == model]["n_features"].iloc[
                0
            ],
        }
        for metric in ["R2", "MAE", "RMSE"]:
            mean_v, std_v = agg_stats(df_metrics, model, metric)
            top80_stats[model][f"{metric}_Mean"] = mean_v
            top80_stats[model][f"{metric}_Std"] = std_v

        comparison_rows.append(
            {
                "Model": model,
                "Feature_Set": "Top80",
                "N_Features": top80_stats[model]["n_features"],
                "R2_Mean": top80_stats[model]["R2_Mean"],
                "R2_Std": top80_stats[model]["R2_Std"],
                "MAE_Mean": top80_stats[model]["MAE_Mean"],
                "MAE_Std": top80_stats[model]["MAE_Std"],
                "RMSE_Mean": top80_stats[model]["RMSE_Mean"],
                "RMSE_Std": top80_stats[model]["RMSE_Std"],
            }
        )

    all_stats = {}
    if all_available:
        print(f"\n[OK] Experimento original encontrado: {all_metrics_path}")
        df_all_metrics = pd.read_csv(all_metrics_path)
        for model in model_names:
            if model not in df_all_metrics["model"].unique():
                print(
                    f"[WARN] El modelo '{model}' no está en el experimento "
                    f"original ALL FEATURES. Se omite en la comparación."
                )
                continue
            all_stats[model] = {
                "n_features": df_all_metrics[df_all_metrics["model"] == model][
                    "n_features"
                ].iloc[0],
            }
            for metric in ["R2", "MAE", "RMSE"]:
                mean_v, std_v = agg_stats(df_all_metrics, model, metric)
                all_stats[model][f"{metric}_Mean"] = mean_v
                all_stats[model][f"{metric}_Std"] = std_v

            comparison_rows.append(
                {
                    "Model": model,
                    "Feature_Set": "All",
                    "N_Features": all_stats[model]["n_features"],
                    "R2_Mean": all_stats[model]["R2_Mean"],
                    "R2_Std": all_stats[model]["R2_Std"],
                    "MAE_Mean": all_stats[model]["MAE_Mean"],
                    "MAE_Std": all_stats[model]["MAE_Std"],
                    "RMSE_Mean": all_stats[model]["RMSE_Mean"],
                    "RMSE_Std": all_stats[model]["RMSE_Std"],
                }
            )
    else:
        print(
            f"\n[INFO] No se encontró metrics_per_iteration.csv en "
            f"--original_run_dir ({all_metrics_path}). Se genera únicamente "
            f"la tabla Top-80%; la estructura queda lista para combinarse "
            f"posteriormente con el experimento ALL FEATURES."
        )

    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(
        os.path.join(RUN_DIR, "comparison_all_vs_top80.csv"), index=False
    )

    # --- Archivo de impacto de la reducción de features ---
    impact_rows = []
    for model in model_names:
        n_top80 = top80_stats[model]["n_features"]
        if model in all_stats:
            n_all = all_stats[model]["n_features"]
            reduction_pct = 100 * (n_all - n_top80) / n_all if n_all else np.nan
            row = {
                "model": model,
                "n_features_all": n_all,
                "n_features_top80": n_top80,
                "feature_reduction_percent": reduction_pct,
                "R2_all": all_stats[model]["R2_Mean"],
                "R2_top80": top80_stats[model]["R2_Mean"],
                "delta_R2": top80_stats[model]["R2_Mean"] - all_stats[model]["R2_Mean"],
                "MAE_all": all_stats[model]["MAE_Mean"],
                "MAE_top80": top80_stats[model]["MAE_Mean"],
                "delta_MAE": top80_stats[model]["MAE_Mean"]
                - all_stats[model]["MAE_Mean"],
                "RMSE_all": all_stats[model]["RMSE_Mean"],
                "RMSE_top80": top80_stats[model]["RMSE_Mean"],
                "delta_RMSE": top80_stats[model]["RMSE_Mean"]
                - all_stats[model]["RMSE_Mean"],
            }
        else:
            row = {
                "model": model,
                "n_features_all": np.nan,
                "n_features_top80": n_top80,
                "feature_reduction_percent": np.nan,
                "R2_all": np.nan,
                "R2_top80": top80_stats[model]["R2_Mean"],
                "delta_R2": np.nan,
                "MAE_all": np.nan,
                "MAE_top80": top80_stats[model]["MAE_Mean"],
                "delta_MAE": np.nan,
                "RMSE_all": np.nan,
                "RMSE_top80": top80_stats[model]["RMSE_Mean"],
                "delta_RMSE": np.nan,
            }
        impact_rows.append(row)

    df_impact = pd.DataFrame(impact_rows)
    df_impact.to_csv(os.path.join(RUN_DIR, "feature_impact_analysis.csv"), index=False)

    # --------------------------------------------------------------
    # 8. RESUMEN FINAL POR CONSOLA
    # --------------------------------------------------------------
    print_final_summary(df_metrics, model_names)

    print(f"\n[OK] Experimento Top-{args.top80_threshold:.0%} finalizado.")
    print(f"[OK] Resultados guardados en: {RUN_DIR}")


# --------------------------------------------------------------
# 9. RESUMEN FINAL POR CONSOLA
# --------------------------------------------------------------


def print_final_summary(df_metrics, model_names):
    print("\n" + "=" * 50)
    print(f"TOP-{args.top80_threshold:.0%} FEATURE EXPERIMENT")
    print("=" * 50)
    for model in model_names:
        sub = df_metrics[df_metrics["model"] == model]
        n_feat = sub["n_features"].iloc[0]
        print(f"\n{model}")
        print(f"Features: {n_feat}")
        print(f"R²:   {sub['R2'].mean():.4f} ± {sub['R2'].std():.4f}")
        print(f"MAE:  {sub['MAE'].mean():.4f} ± {sub['MAE'].std():.4f}")
        print(f"RMSE: {sub['RMSE'].mean():.4f} ± {sub['RMSE'].std():.4f}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        print(f"\n[ERROR FATAL] {exc}", file=sys.stderr)
        raise