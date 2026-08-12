import os
import argparse

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib
matplotlib.use('Agg')
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt


def setup_paths(uspl_id: int, run_id: str | None = None) -> dict:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    outputs_dir = os.path.join(base_dir, 'data', f'data_USPL_{uspl_id}', 'outputs')

    if run_id is None:
        runs = sorted([d for d in os.listdir(outputs_dir) if d.startswith('run_')])
        if not runs:
            raise FileNotFoundError(f'No hay ejecuciones en {outputs_dir}')
        run_id = runs[-1]

    run_dir = os.path.join(outputs_dir, run_id)
    reports_dir = os.path.join(base_dir, 'reports', f'reports_USPL_{uspl_id}', 'figures', run_id)
    os.makedirs(reports_dir, exist_ok=True)

    return {
        'run_dir': run_dir,
        'reports_dir': reports_dir,
        'perm_csv': os.path.join(run_dir, 'ranking_consenso_perm.csv'),
        'shap_csv': os.path.join(run_dir, 'ranking_consenso_shap.csv'),
    }


def infer_iteration_column(df: pd.DataFrame) -> pd.Series:
    n_features = df['feature'].nunique()
    n_models = df['model'].nunique()
    block_size = max(n_features * n_models, 1)
    return (df.index // block_size) + 1


def build_convergence_summary(df: pd.DataFrame, rank_col: str, imp_col: str) -> pd.DataFrame:
    df = df.copy()
    df['iteration'] = infer_iteration_column(df)

    per_iteration = []
    previous = None

    for iteration in range(1, int(df['iteration'].max()) + 1):
        subset = df[df['iteration'] <= iteration].groupby('feature').agg({imp_col: 'mean'}).reset_index()

        if previous is None:
            per_iteration.append({
                'iteration': iteration,
                'mean_abs_delta_importance': 0.0,
                'n_features': int(subset.shape[0]),
            })
        else:
            merged = previous.merge(subset, on='feature', suffixes=('_prev', '_curr'))
            per_iteration.append({
                'iteration': iteration,
                'mean_abs_delta_importance': float((merged[f'{imp_col}_curr'] - merged[f'{imp_col}_prev']).abs().mean()),
                'n_features': int(subset.shape[0]),
            })

        previous = subset[['feature', imp_col]].copy()

    return pd.DataFrame(per_iteration)


def save_convergence_plot(df: pd.DataFrame, rank_col: str, imp_col: str, plot_title: str, output_path: str) -> None:
    df_r = df.copy()
    n_f = df_r['feature'].nunique()
    n_m = df_r['model'].nunique()
    df_r['iteration'] = (df_r.index // (n_f * n_m)) + 1

    conv_list = []
    for n in range(1, df_r['iteration'].max() + 1):
        sub = df_r[df_r['iteration'] <= n].groupby('feature').agg({rank_col: 'mean', imp_col: 'mean'}).reset_index()
        sub['iteration'] = n
        conv_list.append(sub)

    df_plot = pd.concat(conv_list)
    feats = df_plot['feature'].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(feats)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for f_name, color in zip(feats, colors):
        d = df_plot[df_plot['feature'] == f_name].sort_values('iteration')
        axes[0].plot(d['iteration'], d[imp_col], color=color, linewidth=2)
        axes[1].plot(d['iteration'], d[imp_col].diff().abs(), color=color, linewidth=1.5)

    axes[0].set_title('Stability: Importance Magnitude', fontsize=14)
    axes[0].set_xlabel('Iteration', fontsize=14)
    axes[0].set_ylabel('Importance Value', fontsize=14)
    axes[0].tick_params(axis='both', labelsize=14)
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Rate of Change (Magnitude)', fontsize=14)
    axes[1].set_xlabel('Iteration', fontsize=14)
    axes[1].set_ylabel('|Δ Importance Value|', fontsize=14)
    axes[1].tick_params(axis='both', labelsize=14)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close(fig)


def calculate_convergence_speed(df_summary: pd.DataFrame) -> dict:
    """Calculate convergence speed metrics."""
    speeds = {}
    
    for source in df_summary['source'].unique():
        data = df_summary[df_summary['source'] == source].copy()
        data = data.sort_values('iteration')
        
        if len(data) > 1:
            x = data['iteration'].values
            y = data['mean_abs_delta_importance'].values
            speed = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else 0
            threshold = y[0] * 0.01
            converged_iter = len(data[data['mean_abs_delta_importance'] < threshold])
            
            speeds[source] = {
                'velocidad_cambio': float(speed),
                'iteraciones_convergencia': int(converged_iter) if converged_iter > 0 else len(data),
                'delta_inicial': float(y[0]),
                'delta_final': float(y[-1]),
            }
    
    return speeds


def calcular_convergencia(paths: dict) -> tuple[pd.DataFrame, dict]:
    if not os.path.exists(paths['perm_csv']) or not os.path.exists(paths['shap_csv']):
        raise FileNotFoundError('Missing base CSV files: ranking_consenso_perm.csv or ranking_consenso_shap.csv')

    df_perm = pd.read_csv(paths['perm_csv'])
    df_shap = pd.read_csv(paths['shap_csv'])

    resumen_perm = build_convergence_summary(df_perm, 'rank_perm', 'importance_mean')
    resumen_perm['source'] = 'PERMUTATION'

    resumen_shap = build_convergence_summary(df_shap, 'rank_shap', 'shap_importance')
    resumen_shap['source'] = 'SHAP'

    resumen = pd.concat([resumen_perm, resumen_shap], ignore_index=True)
    resumen = resumen[['source', 'iteration', 'n_features', 'mean_abs_delta_importance']]

    save_convergence_plot(
        df_perm,
        'rank_perm',
        'importance_mean',
        'Permutation',
        os.path.join(paths['reports_dir'], 'convergence_permutation.pdf')
    )
    save_convergence_plot(
        df_shap,
        'rank_shap',
        'shap_importance',
        'SHAP',
        os.path.join(paths['reports_dir'], 'convergence_shap.pdf')
    )
    
    # Calcular velocidad de convergencia
    convergence_speed = calculate_convergence_speed(resumen)

    return resumen, convergence_speed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convergence analysis for a USPL run')
    parser.add_argument('--uspl', type=int, required=True, help='Laser ID (1 or 2)')
    parser.add_argument('--run', type=str, default=None, help='Run ID to analyze. If omitted, the latest run is used')
    args = parser.parse_args()

    try:
        rutas = setup_paths(args.uspl, args.run)
        resumen, velocidades = calcular_convergencia(rutas)
        print(f"[OK] Convergence figures saved in: {rutas['reports_dir']}")
        print("\n=== Convergence speed metrics ===")
        for source, metrics in velocidades.items():
            print(f"\n{source}:")
            print(f"  - Change speed: {metrics['velocidad_cambio']:.6f}")
            print(f"  - Convergence iterations (delta < 1%): {metrics['iteraciones_convergencia']}")
            print(f"  - Initial delta: {metrics['delta_inicial']:.6f}")
            print(f"  - Final delta: {metrics['delta_final']:.6f}")
        print("\n=== Last summary iterations ===")
        print(resumen.tail())
    except Exception as e:
        print(f"[ERROR] {e}")