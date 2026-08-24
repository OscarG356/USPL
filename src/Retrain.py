"""
Experimento Independiente: Entrenamiento con features Top-80% por modelo
==========================================================================

Este script es un EXPERIMENTO SEPARADO que reutiliza la metodología
leakage-safe del pipeline original (`temporallearning_opnet.py`), pero
en lugar de usar todas las features seleccionadas tras el filtrado de
varianza/correlación, cada modelo se entrena usando ÚNICAMENTE el
subconjunto de features que representa hasta el 80% de su importancia
acumulada, según los rankings de interpretabilidad (Permutation + SHAP)
generados previamente por el pipeline original.

NO modifica ni sobrescribe ningún archivo del pipeline original.
NO recalcula SHAP ni Permutation Importance: los rankings ya existentes
se cargan desde el directorio de salida (`--original_run_dir`) de una
corrida previa del pipeline original.

────────────────────────────────────────────────────────────────────────
¿Qué archivos del pipeline original utiliza este script?
────────────────────────────────────────────────────────────────────────
Del directorio `--original_run_dir` (RUN_DIR de una corrida previa de
`temporallearning_opnet.py`) se leen, en este orden de prioridad:

1. `tabla_consenso_final.csv`
   - Si esta tabla contiene una columna `model`, se asume que ya trae el
     consenso desagregado POR MODELO y se usa directamente para construir
     el Top-80% de cada modelo.
   - En el pipeline original tal como está escrito, `tabla_consenso_final.csv`
     es un CONSENSO GLOBAL (promedia SHAP/Permutation sobre todas las
     iteraciones y TODOS los modelos juntos, ver `process_triple_80`), por
     lo que normalmente NO contiene la columna `model`. En ese caso este
     script NO la usa como fuente del Top-80% por modelo (para no asumir
     que un consenso global equivale al Top-80% de cada modelo individual).

2. `ranking_consenso_perm.csv` y `ranking_consenso_shap.csv`
   - Estos sí contienen la columna `model` (y `repetition`), por lo que son
     la fuente utilizada por defecto. Este script:
       a. Promedia `importance_mean` (Permutation) y `shap_importance`
          (SHAP) por (modelo, feature) a través de todas las repeticiones
          disponibles en esos archivos.
       b. Normaliza esos promedios con min-max, POR MODELO (igual que
          `process_triple_80` en el pipeline original, pero sin colapsar
          los modelos entre sí).
       c. Calcula `Consensus_General = mean(perm_norm, shap_norm)` por
          (modelo, feature).
       d. Para cada modelo, ordena descendentemente por `Consensus_General`,
          calcula la importancia relativa acumulada y selecciona las
          features hasta alcanzar el 80% acumulado, con el mismo criterio
          exacto usado en el pipeline original:
              Top80 = (cumsum_acumulado_previo < 0.80)
          manteniendo siempre al menos la primera feature.

Adicionalmente, si existe `metrics_per_iteration.csv` en el
`--original_run_dir`, se utiliza (sin volver a entrenar nada) como fuente
de la comparación "ALL FEATURES" contra la cual se compara el experimento
Top-80% (sección 8 del encargo). Si no existe, el script genera únicamente
los resultados Top-80% y deja preparada la estructura de comparación.

────────────────────────────────────────────────────────────────────────
Uso
────────────────────────────────────────────────────────────────────────
python top80_feature_experiment_opnet.py \
    --uspl 1 \
    --original_run_dir /ruta/a/data/data_USPL_1/outputs/run_XXXXXXXX_integrated \
    --iters 5 \
    --feature_method all \
    --top80_threshold 0.80
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pyrefly: ignore [missing-import]
import tsfel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# pyrefly: ignore [missing-import]
from xgboost import XGBRegressor

# ══════════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN / CLI
# ══════════════════════════════════════════════════════════════

parser = argparse.ArgumentParser(
    description=(
        "Experimento independiente: entrenamiento con features Top-80% "
        "por modelo (derivadas del análisis de interpretabilidad del "
        "pipeline original)."
    )
)
parser.add_argument("--uspl", type=int, default=1, help="ID del láser (1 o 2)")
parser.add_argument(
    "--iters", type=int, default=5, help="Número de repeticiones del experimento"
)
parser.add_argument(
    "--feature_method",
    type=str,
    default="all",
    help=(
        "Dominio TSFEL usado para extraer features. DEBE coincidir con el "
        "usado en la corrida original para que los nombres de features "
        "sean comparables."
    ),
)
parser.add_argument(
    "--threshold_corr",
    type=float,
    default=0.90,
    help=(
        "Umbral de correlación (informativo / heredado del pipeline "
        "original). NO se reaplica un filtro de correlación sobre las "
        "features Top-80%: éstas se tratan como una selección de "
        "características ya determinada por el análisis de "
        "interpretabilidad previo."
    ),
)
parser.add_argument(
    "--top80_threshold",
    type=float,
    default=0.80,
    help="Umbral de importancia acumulada para definir el conjunto Top-N%.",
)
parser.add_argument(
    "--original_run_dir",
    type=str,
    required=True,
    help=(
        "Directorio RUN_DIR de una corrida previa de "
        "temporallearning_opnet.py. Debe contener ranking_consenso_perm.csv "
        "y ranking_consenso_shap.csv (y opcionalmente tabla_consenso_final.csv "
        "y metrics_per_iteration.csv)."
    ),
)
parser.add_argument(
    "--raw_dir",
    type=str,
    default=None,
    help=(
        "Ruta explícita a la carpeta con las señales crudas del osciloscopio. "
        "Si no se especifica, se infiere igual que en el pipeline original: "
        "<BASE_DIR>/data/data_USPL_<uspl>/raw/osciloscopio"
    ),
)
args = parser.parse_args()

USPL_ID = f"USPL_{args.uspl}"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # noqa: DTZ005

RAW_DIR = args.raw_dir or os.path.join(
    BASE_DIR, "data", f"data_{USPL_ID}", "raw", "osciloscopio"
)
RUN_DIR = os.path.join(
    BASE_DIR, "data", f"data_{USPL_ID}", "outputs", f"run_{TIMESTAMP}_top80"
)
PLOTS_DIR = os.path.join(RUN_DIR, "plots")
os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

MODEL_NAME_TO_FILESAFE = {
    "SVR": "SVR",
    "Random Forest": "Random_Forest",
    "Bayesian Ridge": "Bayesian_Ridge",
    "XGBoost": "XGBoost",
}

print(f"--- Experimento Top-{args.top80_threshold:.0%} de features para: {USPL_ID} ---")
print(f"--- Directorio original (rankings): {args.original_run_dir} ---")
print(f"--- Salidas nuevas en: {RUN_DIR} ---")

# ══════════════════════════════════════════════════════════════
# 2. FUNCIONES AUXILIARES (idénticas al pipeline original donde aplica)
# ══════════════════════════════════════════════════════════════


def natural_sort_key(s: str) -> list:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def load_raw_signals(data_folder: str):
    """Carga y segmenta las señales CRUDAS. Sin normalizar.
    Idéntico al pipeline original para garantizar splits comparables."""
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

    example_df = pd.read_csv(os.path.join(data_folder, signal_files[0]), header=0)
    fs = 1 / (example_df.iloc[1, 0] - example_df.iloc[0, 0])

    labels_df = pd.read_excel(os.path.join(data_folder, "Datos-Corriente.xlsx"))
    t_Corriente_raw = labels_df["Corriente (mA)"].values[191:]

    segmented_signals, segmented_labels = [], []
    segment_length, num_segments = 300, 2
    for idx, signal_array in enumerate(signals_matrix):
        if len(signal_array) >= num_segments * segment_length:
            for i in range(num_segments):
                segmented_signals.append(
                    signal_array[i * segment_length : (i + 1) * segment_length]
                )
                segmented_labels.append(t_Corriente_raw[idx])

    return np.array(segmented_signals), np.array(segmented_labels), fs


def mape(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true_safe - y_pred) / y_true_safe)) * 100


def save_to_run_append(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(RUN_DIR, filename)
    if os.path.exists(path):
        pd.concat([pd.read_csv(path), df], ignore_index=True).to_csv(path, index=False)
    else:
        df.to_csv(path, index=False)


# ══════════════════════════════════════════════════════════════
# 3. CONSTRUCCIÓN DEL CONJUNTO TOP-N% POR MODELO
# ══════════════════════════════════════════════════════════════


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

    # Detectar la columna de consenso a usar
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

    df_p = pd.read_csv(
        perm_csv
    )  # feature, importance_mean, model, rank_perm, repetition
    df_s = pd.read_csv(
        shap_csv
    )  # feature, shap_importance, model, rank_shap, repetition

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


# ══════════════════════════════════════════════════════════════
# 4. MODELOS (idénticos al pipeline original)
# ══════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════
# 5. PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════


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
    X_raw, y_raw, fs = load_raw_signals(RAW_DIR)
    domain = args.feature_method
    if domain == "all":
        domain = None
    cfg = tsfel.get_features_by_domain(domain)

    metrics_log, residuals_log, split_log = [], [], []
    iter_pred_records = []

    for iteration in range(args.iters):
        repetition = iteration + 1
        print(f"\n{'=' * 50}\nIteration {repetition}/{args.iters}\n{'=' * 50}")

        # --- SPLIT LEAKAGE-SAFE (misma semilla que el pipeline original) ---
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=iteration)
        train_idx, test_idx = next(splitter.split(X_raw, y_raw, groups=y_raw))
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y_raw[train_idx], y_raw[test_idx]

        train_groups = np.unique(y_train)
        test_groups = np.unique(y_test)
        assert len(set(train_groups) & set(test_groups)) == 0, (
            "ERROR FATAL: Leakage en Split (setpoints de corriente "
            "compartidos entre train y test)"
        )

        split_log.append(
            {
                "repetition": repetition,
                "random_seed": iteration,
                "train_groups": train_groups.tolist(),
                "test_groups": test_groups.tolist(),
            }
        )

        # --- NORMALIZACIÓN (solo con TRAIN) ---
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

        # --- VALIDACIÓN: las features Top-N% deben existir tras TSFEL ---
        iter_preds = {"repetition": repetition, "Actual_Current": y_test}

        for model_name, (estimator, param_grid) in models.items():
            model_features = top_features_per_model[model_name]
            missing_feats = [f for f in model_features if f not in common_cols]
            if missing_feats:
                raise ValueError(
                    f"[ERROR] Iteración {repetition} / Modelo '{model_name}': "
                    f"las siguientes features Top-{args.top80_threshold:.0%} "
                    f"seleccionadas por el ranking de interpretabilidad NO "
                    f"existen entre las features extraídas por TSFEL en esta "
                    f"corrida: {missing_feats}. Verifica que "
                    f"--feature_method coincida con el usado en la corrida "
                    f"original y que los datos crudos sean los mismos. Se "
                    f"detiene la ejecución en lugar de eliminarlas "
                    f"silenciosamente."
                )

            X_train_sel_df = X_train_feat[model_features]
            X_test_sel_df = X_test_feat[model_features]

            # Verificación de columnas idénticas train/test
            assert list(X_train_sel_df.columns) == list(X_test_sel_df.columns), (
                f"[ERROR] Columnas de train/test no coinciden para "
                f"'{model_name}' en la iteración {repetition}."
            )

            # Imputación segura (ajustada SOLO con TRAIN) por si TSFEL
            # produce NaN puntuales en alguna fila/feature.
            train_means = X_train_sel_df.mean()
            X_train_sel = X_train_sel_df.fillna(train_means).values
            X_test_sel = X_test_sel_df.fillna(train_means).values

            # --- ENTRENAMIENTO ---
            pipe = Pipeline([("scaler", StandardScaler()), ("model", estimator)])
            inner_cv = GroupKFold(n_splits=5)
            grid = GridSearchCV(
                pipe,
                param_grid,
                cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            grid.fit(X_train_sel, y_train, groups=y_train)
            y_pred = grid.predict(X_test_sel)
            iter_preds[f"Pred_{model_name}"] = y_pred

            metrics_log.append(
                {
                    "repetition": repetition,
                    "model": model_name,
                    "feature_set": "Top80",
                    "R2": r2_score(y_test, y_pred),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAPE": mape(y_test, y_pred),
                    "n_features": len(model_features),
                    "features_used": ";".join(model_features),
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
                        "model": model_name,
                        "feature_set": "Top80",
                        "setpoint": y_t,
                        "y_true": y_t,
                        "y_pred": y_p,
                        "residual": y_t - y_p,
                        "absolute_error": abs(y_t - y_p),
                        "relative_error": (y_t - y_p) / y_t_safe,
                    }
                )

        iter_pred_records.append(iter_preds)

    # ══════════════════════════════════════════════════════════════
    # 6. GUARDADO DE ARCHIVOS PRINCIPALES
    # ══════════════════════════════════════════════════════════════

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

    # --- Resumen y CI 95% (bootstrap, igual metodología que el original) ---
    summary_list = []
    np.random.seed(42)
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

    # ══════════════════════════════════════════════════════════════
    # 7. COMPARACIÓN ALL vs TOP-80% (si el experimento original está disponible)
    # ══════════════════════════════════════════════════════════════

    all_metrics_path = os.path.join(args.original_run_dir, "metrics_per_iteration.csv")
    comparison_rows = []
    all_available = os.path.exists(all_metrics_path)

    # Estadísticos Top80 en formato ancho (para armar tabla comparativa y
    # el archivo de impacto de reducción de features)
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

    # ══════════════════════════════════════════════════════════════
    # 8. GRÁFICAS
    # ══════════════════════════════════════════════════════════════
    generate_plots(df_metrics, df_residuals, df_comparison, all_available, model_names)

    # ══════════════════════════════════════════════════════════════
    # 9. RESUMEN FINAL POR CONSOLA
    # ══════════════════════════════════════════════════════════════
    print_final_summary(df_metrics, model_names)

    print(f"\n[OK] Experimento Top-{args.top80_threshold:.0%} finalizado.")
    print(f"[OK] Resultados guardados en: {RUN_DIR}")


# ══════════════════════════════════════════════════════════════
# 10. GRÁFICAS
# ══════════════════════════════════════════════════════════════


def generate_plots(df_metrics, df_residuals, df_comparison, all_available, model_names):
    # A. Comparación de número de features (All vs Top80)
    if all_available:
        pivot_n = df_comparison.pivot(
            index="Model", columns="Feature_Set", values="N_Features"
        )
        pivot_n = pivot_n.reindex(model_names)
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot_n.plot(kind="bar", ax=ax)
        ax.set_ylabel("Número de features")
        ax.set_title(
            f"Número de features: All vs Top-{args.top80_threshold:.0%}"
        )
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "A_n_features_all_vs_top80.png"), dpi=150)
        plt.close(fig)
    else:
        n_top80 = df_metrics.groupby("model")["n_features"].first().reindex(model_names)
        fig, ax = plt.subplots(figsize=(8, 5))
        n_top80.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_ylabel("Número de features")
        ax.set_title(
            f"Número de features Top-{args.top80_threshold:.0%} por modelo"
        )
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "A_n_features_top80.png"), dpi=150)
        plt.close(fig)

    # B. Comparación de rendimiento (R2, MAE, RMSE): All vs Top80
    if all_available:
        for metric in ["R2", "MAE", "RMSE"]:
            pivot_m = df_comparison.pivot(
                index="Model", columns="Feature_Set", values=f"{metric}_Mean"
            ).reindex(model_names)
            pivot_std = df_comparison.pivot(
                index="Model", columns="Feature_Set", values=f"{metric}_Std"
            ).reindex(model_names)
            fig, ax = plt.subplots(figsize=(8, 5))
            pivot_m.plot(kind="bar", yerr=pivot_std, ax=ax, capsize=4)
            ax.set_ylabel(metric)
            ax.set_title(f"{metric}: All vs Top-{args.top80_threshold:.0%}")
            plt.tight_layout()
            fig.savefig(
                os.path.join(PLOTS_DIR, f"B_{metric}_all_vs_top80.png"), dpi=150
            )
            plt.close(fig)
    else:
        for metric in ["R2", "MAE", "RMSE"]:
            means = df_metrics.groupby("model")[metric].mean().reindex(model_names)
            stds = df_metrics.groupby("model")[metric].std().reindex(model_names)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(
                means.index,
                means.values,
                yerr=stds.values,
                capsize=4,
                color="darkorange",
            )
            ax.set_ylabel(metric)
            ax.set_title(f"{metric} (Top-{args.top80_threshold:.0%}) por modelo")
            plt.xticks(rotation=20)
            plt.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, f"B_{metric}_top80.png"), dpi=150)
            plt.close(fig)

    # C. Predicción vs valor real (Top80), por modelo
    for model in model_names:
        sub = df_residuals[df_residuals["model"] == model]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sub["y_true"], sub["y_pred"], alpha=0.4, s=15)
        lims = [
            min(sub["y_true"].min(), sub["y_pred"].min()),
            max(sub["y_true"].max(), sub["y_pred"].max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1)
        ax.set_xlabel("Corriente real (mA)")
        ax.set_ylabel("Corriente predicha (mA)")
        safe_name = MODEL_NAME_TO_FILESAFE[model]
        ax.set_title(f"{model} — Predicción vs Real (Top-{args.top80_threshold:.0%})")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"C_pred_vs_real_{safe_name}.png"), dpi=150)
        plt.close(fig)

    # D. Error por corriente (setpoint)
    for model in model_names:
        sub = df_residuals[df_residuals["model"] == model]
        if sub.empty:
            continue
        grouped = (
            sub.groupby("setpoint")
            .agg(
                MAE=("absolute_error", "mean"),
                RMSE=("residual", lambda x: np.sqrt(np.mean(x**2))),
            )
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(grouped["setpoint"], grouped["MAE"], marker="o", label="MAE")
        ax.plot(grouped["setpoint"], grouped["RMSE"], marker="s", label="RMSE")
        ax.set_xlabel("Setpoint de corriente (mA)")
        ax.set_ylabel("Error")
        safe_name = MODEL_NAME_TO_FILESAFE[model]
        ax.set_title(
            f"{model} — Error por setpoint de corriente (Top-{args.top80_threshold:.0%})"
        )
        ax.legend()
        plt.tight_layout()
        fig.savefig(
            os.path.join(PLOTS_DIR, f"D_error_por_corriente_{safe_name}.png"), dpi=150
        )
        plt.close(fig)

    print(f"[OK] Gráficas guardadas en: {PLOTS_DIR}")


# ══════════════════════════════════════════════════════════════
# 11. RESUMEN FINAL POR CONSOLA
# ══════════════════════════════════════════════════════════════


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
