"""
Gráfica de comparación de modelos (Actual vs Predicho) con banda de IC 95%
============================================================================

Script INDEPENDIENTE de solo-graficación. NO entrena nada, NO modifica el
pipeline original ni el script de Top-80%: únicamente lee el archivo
`model_predictions_history_top80.csv` que ya generó una corrida previa de
`top80_feature_experiment_opnet.py` (o el `model_predictions_history.csv`
del pipeline original, que tiene el mismo formato) y produce la gráfica.

Columnas esperadas en el CSV de entrada:
    repetition, Actual_Current, Pred_<Modelo1>, Pred_<Modelo2>, ...
(exactamente lo que guardan ambos pipelines).

Uso
----
python plot_model_comparison_ci.py \
    --input /ruta/a/model_predictions_history_top80.csv \
    --min_current 160 \
    --max_current 400 \
    --output grafica_prediccion.pdf
"""

import argparse
import os

# pyrefly: ignore [missing-import]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# pyrefly: ignore [missing-import]
import seaborn as sns

# Nombres cortos para la leyenda (opcional, se puede ajustar libremente)
SHORT_NAMES = {
    "Pred_SVR": "SVR",
    "Pred_Random Forest": "RFR",
    "Pred_Bayesian Ridge": "BR",
    "Pred_XGBoost": "XGBoost",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Grafica Actual vs Predicho (con IC 95%) a partir de un CSV de "
            "predicciones ya generado (model_predictions_history[_top80].csv)."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Ruta al CSV de predicciones (model_predictions_history_top80.csv).",
    )
    parser.add_argument(
        "--min_current",
        type=float,
        default=None,
        help="Corriente mínima (mA) a graficar. Por defecto: mínimo de los datos.",
    )
    parser.add_argument(
        "--max_current",
        type=float,
        default=None,
        help="Corriente máxima (mA) a graficar. Por defecto: máximo de los datos.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grafica_prediccion.pdf",
        help="Ruta de salida para el PDF. También se guarda un PNG con el mismo nombre base.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Si se pasa, muestra la gráfica en pantalla además de guardarla.",
    )
    return parser.parse_args()


def visualizar_predicciones(df: pd.DataFrame, min_c: float, max_c: float, out_path: str, show: bool):
    # 1. Filtrar por rango de corriente
    filtered_df = df[(df["Actual_Current"] >= min_c) & (df["Actual_Current"] <= max_c)].copy()
    if filtered_df.empty:
        raise ValueError(
            f"El rango [{min_c}, {max_c}] no contiene datos. "
            f"Rango disponible en el archivo: "
            f"[{df['Actual_Current'].min()}, {df['Actual_Current'].max()}]"
        )

    # 2. Detectar dinámicamente las columnas de predicción presentes en el CSV
    pred_cols = [c for c in filtered_df.columns if c.startswith("Pred_")]
    if not pred_cols:
        raise ValueError(
            "No se encontraron columnas 'Pred_<Modelo>' en el CSV de entrada."
        )

    # 3. Reestructurar (melt) a formato largo para graficar varios modelos a la vez
    df_melted = filtered_df.melt(
        id_vars=["repetition", "Actual_Current"],
        value_vars=pred_cols,
        var_name="Model",
        value_name="Prediction",
    )
    rename_map = {c: SHORT_NAMES.get(c, c.replace("Pred_", "")) for c in pred_cols}
    df_melted["Model"] = df_melted["Model"].replace(rename_map)

    # 4. Estética
    plt.figure(figsize=(8, 4))
    sns.set_style("whitegrid")

    # 5. Graficar con Intervalo de Confianza (CI 95%)
    sns.lineplot(
        data=df_melted,
        x="Actual_Current",
        y="Prediction",
        hue="Model",
        errorbar=("ci", 95),
        alpha=0.8,
    )

    # 6. Línea de referencia ideal (Actual vs Actual)
    plt.plot(
        filtered_df["Actual_Current"],
        filtered_df["Actual_Current"],
        color="black",
        linestyle="--",
        label="Actual (Ideal)",
        linewidth=1,
    )
    plt.xlim(min_c, max_c)
    plt.ylim(min_c, max_c)

    plt.xlabel("Actual $I_p$ (mA)", fontsize=14, weight="bold")
    plt.ylabel("Predicted $I_p$ (mA)", fontsize=14, weight="bold")
    plt.tick_params(axis="both", labelsize=12)

    plt.legend(
        title="Models / Reference",
        loc="lower right",
        prop={"size": 12, "weight": "bold"},
        title_fontsize=13,
    )
    plt.gca().get_legend().get_title().set_weight("bold")

    plt.tight_layout()

    # 7. Guardar (PDF y PNG)
    base, _ext = os.path.splitext(out_path)
    pdf_path = base + ".pdf"
    png_path = base + ".png"
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.savefig(png_path, format="png", dpi=150, bbox_inches="tight")
    print(f"Gráfica guardada en:\n  {pdf_path}\n  {png_path}")

    if show:
        plt.show()
    plt.close()


def main():
    args = parse_args()

    df = pd.read_csv('/home/oagr/Documentos/Gita/USPL/data/data_USPL_2/outputs/run_20260819_155209_top80/model_predictions_history_top80.csv')
    if "Actual_Current" not in df.columns:
        raise ValueError(
            f"El archivo no tiene la columna 'Actual_Current'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    min_c = args.min_current if args.min_current is not None else df["Actual_Current"].min()
    max_c = args.max_current if args.max_current is not None else df["Actual_Current"].max()

    visualizar_predicciones(df, min_c, max_c, args.output, args.show)


if __name__ == "__main__":
    main()