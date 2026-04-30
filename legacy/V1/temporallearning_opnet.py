# ─────────────────────────────────────────────────────────────
#  MODO RENDIMIENTO MÁXIMO
#  - display() y plt.show() desactivados
#  - verbose=0 en todos los GridSearch
#  - prints intermedios silenciados
#  - try/except por iteración para que nunca se detenga solo
#  - gc.collect() al final de cada vuelta
#  - resumen final con resultados y errores
# ─────────────────────────────────────────────────────────────

# from IPython.display import display   # <-- no se usa en modo rendimiento
import gc
import os
import traceback

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Backend sin ventana, evita errores en PC sin display
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

loaded_df = pd.read_csv('extracted_features.csv')

t_Corriente = loaded_df['target_current'].values
features_df = loaded_df.drop(columns=['target_current'])

# print(f"Shape of the loaded features matrix: {features_df.shape}")
# print(f"Shape of loaded t_Corriente: {t_Corriente.shape}")
# display(features_df.head())

print(f"[INFO] Dataset cargado. Features: {features_df.shape}, Target: {t_Corriente.shape}")
print(f"[INFO] Iniciando loop de iteraciones...\n")

# ── Registro de resultados y errores ───────────────────────────
iteration_log = []   # {'iter': i, 'status': 'ok'/'error', 'detail': ...}

def mape(y_true, y_pred):
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ── LOOP PRINCIPAL ─────────────────────────────────────────────
for i in range(100):

    print(f"[ITER {i+1}] Iniciando...")

    try:

        # ── Train/Test Split ───────────────────────────────────
        from sklearn.model_selection import train_test_split

        X = features_df.values
        X_train, X_test, y_train, y_test = train_test_split(
            X, t_Corriente, test_size=0.2,
            stratify=pd.cut(t_Corriente, bins=10)
        )

        # ══════════════════════════════════════════════════════
        # SVR
        # ══════════════════════════════════════════════════════
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV

        pipeline = Pipeline([
            ('scaler_post', StandardScaler()),
            ('svr',         SVR(kernel='rbf'))
        ])

        param_grid = {
            'svr__C':       [10, 100, 1000],
            'svr__gamma':   ['scale', 0.001, 0.005, 0.01],
            'svr__epsilon': [1, 2.5, 5]
        }

        grid_search_svr = GridSearchCV(pipeline, param_grid, cv=5, verbose=0, scoring='r2', n_jobs=-1)
        grid_search_svr.fit(X_train, y_train)

        # print(grid_search_svr.best_params_)
        # print(f"Mejor R2 (CV train): {grid_search_svr.best_score_:.4f}")

        # ── Métricas SVR ──────────────────────────────────────
        y_pred_test  = grid_search_svr.predict(X_test)
        y_pred_train = grid_search_svr.predict(X_train)

        # print(f"R2 train:   {r2_score(y_train, y_pred_train):.4f}")
        # print(f"R2 test:    {r2_score(y_test, y_pred_test):.4f}")
        # print(f"MAE test:   {mean_absolute_error(y_test, y_pred_test):.2f} mA")
        # print(f"RMSE test:  {np.sqrt(mean_squared_error(y_test, y_pred_test)):.2f} mA")
        # print(f"MAPE test:  {mape(y_test, y_pred_test):.2f}%")

        # ── Plot SVR (guardado, NO mostrado) ──────────────────
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.scatter(y_test, y_pred_test, alpha=0.7, label='Predicted vs. Actual')
        # min_val = min(y_test.min(), y_pred_test.min())
        # max_val = max(y_test.max(), y_pred_test.max())
        # ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
        # ax.set_xlabel('Actual Current (mA)')
        # ax.set_ylabel('Predicted Current (mA)')
        # ax.set_title('Actual vs. Predicted Current Values - SVR (Test Set)')
        # ax.legend(); ax.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'svr_iter_{i+1}.png', dpi=80)
        # plt.close()

        # ── Permutation Importance SVR ────────────────────────
        from sklearn.inspection import permutation_importance

        result = permutation_importance(
            grid_search_svr.best_estimator_,
            X_test, y_test,
            n_repeats=75,
            n_jobs=-1
        )

        permutation_svr_df = pd.DataFrame({
            'feature':          features_df.columns,
            'importance_mean':  result.importances_mean,
            'importance_std':   result.importances_std
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

        permutation_svr_df['rank_perm'] = permutation_svr_df.index + 1

        # print("\n--- Features contributing to 80% of model explanation (based on Permutation Importance) ---")
        importancias_limpias = permutation_svr_df['importance_mean'].clip(lower=0)
        permutation_svr_df['normalized_importance'] = importancias_limpias / importancias_limpias.sum()
        permutation_svr_df['cumulative_importance']  = permutation_svr_df['normalized_importance'].cumsum()
        # display(permutation_svr_df[['rank_perm', 'feature', 'importance_mean', 'cumulative_importance']])
        # plt.clf()

        # ── SHAP SVR ──────────────────────────────────────────
        import shap

        best_svr_pipeline  = grid_search_svr.best_estimator_
        predict_fn         = best_svr_pipeline.predict

        X_train_background = shap.sample(X_train, 100)
        X_test_global      = shap.sample(X_test, 75)

        explainer_svr      = shap.KernelExplainer(predict_fn, X_train_background)
        shap_values_global = explainer_svr.shap_values(X_test_global)

        mean_abs_shap = np.mean(np.abs(shap_values_global), axis=0)

        shap_svr_df = (
            pd.DataFrame({
                'feature':         features_df.columns.tolist(),
                'shap_importance': mean_abs_shap
            })
            .sort_values('shap_importance', ascending=False)
            .reset_index(drop=True)
        )
        shap_svr_df['rank_shap'] = shap_svr_df.index + 1

        # print("=== Top Mean Absolute SHAP — SVR ===")
        shap_svr_df['normalized'] = shap_svr_df['shap_importance'] / shap_svr_df['shap_importance'].sum()
        shap_svr_df['cumulative'] = shap_svr_df['normalized'].cumsum()
        # display(shap_svr_df[['rank_shap', 'feature', 'shap_importance', 'cumulative']])

        # ══════════════════════════════════════════════════════
        # Random Forest
        # ══════════════════════════════════════════════════════
        from sklearn.ensemble import RandomForestRegressor

        rf_pipeline = Pipeline([
            ('rf', RandomForestRegressor(n_jobs=-1))
        ])

        param_grid_rf = {
            'rf__n_estimators':    [100, 300, 500],
            'rf__max_depth':       [5, 10, 15],
            'rf__min_samples_leaf':[4, 8, 16],
            'rf__max_features':    ['sqrt', 'log2']
        }

        # print("Iniciando entrenamiento de Random Forest (esto será más rápido que SVR)...")
        grid_search_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=5, verbose=0, scoring='r2', n_jobs=-1)
        grid_search_rf.fit(X_train, y_train)

        # print("\n=== Resultados de Random Forest ===")
        # print("Mejores Hiperparámetros:", grid_search_rf.best_params_)
        # print(f"Mejor R2 (CV train): {grid_search_rf.best_score_:.4f}")

        # ── Métricas RF ───────────────────────────────────────
        y_pred_rf_train = grid_search_rf.predict(X_train)
        y_pred_rf_test  = grid_search_rf.predict(X_test)

        # print("=== Random Forest Performance ===")
        # print(f"R2 train:   {r2_score(y_train, y_pred_rf_train):.4f}")
        # print(f"R2 test:    {r2_score(y_test,  y_pred_rf_test):.4f}")
        # print(f"MAE test:   {mean_absolute_error(y_test, y_pred_rf_test):.2f} mA")
        # print(f"RMSE test:  {np.sqrt(mean_squared_error(y_test, y_pred_rf_test)):.2f} mA")
        # print(f"MAPE test:  {mape(y_test, y_pred_rf_test):.2f}%")

        # ── Plot RF (comentado) ───────────────────────────────
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.scatter(y_test, y_pred_rf_test, alpha=0.7, label='Predicted vs. Actual')
        # min_val = min(y_test.min(), y_pred_rf_test.min())
        # max_val = max(y_test.max(), y_pred_rf_test.max())
        # ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
        # ax.set_xlabel('Actual Current (mA)'); ax.set_ylabel('Predicted Current (mA)')
        # ax.set_title('Actual vs. Predicted Current Values - Random Forest (Test Set)')
        # ax.legend(); ax.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'rf_iter_{i+1}.png', dpi=80)
        # plt.close()

        # ── Permutation Importance RF ─────────────────────────
        result_rf = permutation_importance(
            grid_search_rf.best_estimator_,
            X_test, y_test,
            n_repeats=75,
            n_jobs=-1
        )

        permutation_rf_df = pd.DataFrame({
            'feature':         features_df.columns,
            'importance_mean': result_rf.importances_mean,
            'importance_std':  result_rf.importances_std
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

        permutation_rf_df['rank_perm'] = permutation_rf_df.index + 1

        # print("\n--- Features contributing to 80% of model explanation (RF - Permutation Importance) ---")
        importancias_limpias_rf = permutation_rf_df['importance_mean'].clip(lower=0)
        permutation_rf_df['normalized_importance'] = importancias_limpias_rf / importancias_limpias_rf.sum()
        permutation_rf_df['cumulative_importance']  = permutation_rf_df['normalized_importance'].cumsum()
        # display(permutation_rf_df[['rank_perm', 'feature', 'importance_mean', 'cumulative_importance']])
        # plt.clf()

        # ── SHAP RF ───────────────────────────────────────────
        best_rf_model        = grid_search_rf.best_estimator_.named_steps['rf']
        X_train_background_rf = shap.sample(X_train, 100)
        X_test_global_rf      = shap.sample(X_test, 75)

        explainer_rf   = shap.TreeExplainer(best_rf_model, X_train_background_rf)
        shap_values_rf = explainer_rf.shap_values(X_test_global_rf)

        mean_abs_shap_rf = np.mean(np.abs(shap_values_rf), axis=0)

        shap_rf_df = (
            pd.DataFrame({
                'feature':         features_df.columns.tolist(),
                'shap_importance': mean_abs_shap_rf
            })
            .sort_values('shap_importance', ascending=False)
            .reset_index(drop=True)
        )
        shap_rf_df['rank_shap'] = shap_rf_df.index + 1

        # print("=== Top Mean Absolute SHAP — Random Forest (Regresor) ===")
        shap_rf_df['normalized'] = shap_rf_df['shap_importance'] / shap_rf_df['shap_importance'].sum()
        shap_rf_df['cumulative'] = shap_rf_df['normalized'].cumsum()
        # display(shap_rf_df[['rank_shap', 'feature', 'shap_importance', 'cumulative']])

        # ══════════════════════════════════════════════════════
        # Bayesian Ridge
        # ══════════════════════════════════════════════════════
        from sklearn.linear_model import BayesianRidge

        bayesian_pipeline = Pipeline([
            ('scaler',   StandardScaler()),
            ('bayesian', BayesianRidge())
        ])

        param_grid_bayesian = {
            'bayesian__max_iter': [300, 500, 1000],
            'bayesian__alpha_1':  [1e-6, 1e-4],
            'bayesian__alpha_2':  [1e-6, 1e-4],
            'bayesian__lambda_1': [1e-6, 1e-4],
            'bayesian__lambda_2': [1e-6, 1e-4],
        }

        grid_search_bayesian = GridSearchCV(bayesian_pipeline, param_grid_bayesian, cv=5, verbose=0, scoring='r2', n_jobs=-1)
        grid_search_bayesian.fit(X_train, y_train)

        # print("\n=== Resultados de Bayesian Ridge ===")
        # print("Mejores Hiperparámetros:", grid_search_bayesian.best_params_)
        # print(f"Mejor R2 (CV train): {grid_search_bayesian.best_score_:.4f}")

        # ── Métricas Bayesian ─────────────────────────────────
        y_pred_bayesian_train = grid_search_bayesian.predict(X_train)
        y_pred_bayesian_test  = grid_search_bayesian.predict(X_test)

        # print("=== Bayesian Ridge Performance ===")
        # print(f"R2 train:   {r2_score(y_train, y_pred_bayesian_train):.4f}")
        # print(f"R2 test:    {r2_score(y_test,  y_pred_bayesian_test):.4f}")
        # print(f"MAE test:   {mean_absolute_error(y_test, y_pred_bayesian_test):.2f} mA")
        # print(f"RMSE test:  {np.sqrt(mean_squared_error(y_test, y_pred_bayesian_test)):.2f} mA")
        # print(f"MAPE test:  {mape(y_test, y_pred_bayesian_test):.2f}%")

        # ── Plot Bayesian (comentado) ─────────────────────────
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.scatter(y_test, y_pred_bayesian_test, alpha=0.7, label='Predicted vs. Actual')
        # min_val = min(y_test.min(), y_pred_bayesian_test.min())
        # max_val = max(y_test.max(), y_pred_bayesian_test.max())
        # ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
        # ax.set_xlabel('Actual Current (mA)'); ax.set_ylabel('Predicted Current (mA)')
        # ax.set_title('Actual vs. Predicted Current Values - Bayesian Ridge (Test Set)')
        # ax.legend(); ax.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'bayesian_iter_{i+1}.png', dpi=80)
        # plt.close()

        # ── Permutation Importance Bayesian ───────────────────
        result_rb = permutation_importance(
            grid_search_bayesian.best_estimator_,
            X_test, y_test,
            n_repeats=75,
            n_jobs=-1
        )

        permutation_rb_df = pd.DataFrame({
            'feature':         features_df.columns,
            'importance_mean': result_rb.importances_mean,
            'importance_std':  result_rb.importances_std
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

        permutation_rb_df['rank_perm'] = permutation_rb_df.index + 1

        # print("\n--- Features contributing to 80% of model explanation (RB - Permutation) ---")
        importancias_limpias_rb = permutation_rb_df['importance_mean'].clip(lower=0)
        permutation_rb_df['normalized_importance'] = importancias_limpias_rb / importancias_limpias_rb.sum()
        permutation_rb_df['cumulative_importance']  = permutation_rb_df['normalized_importance'].cumsum()
        # display(permutation_rb_df[['rank_perm', 'feature', 'importance_mean', 'cumulative_importance']])
        # plt.clf()

        # ── SHAP Bayesian ─────────────────────────────────────
        best_rb_model  = grid_search_bayesian.best_estimator_.named_steps['bayesian']
        explainer_rb   = shap.LinearExplainer(best_rb_model, X_train_background)
        shap_values_rb = explainer_rb.shap_values(X_test_global)

        mean_abs_shap_rb = np.mean(np.abs(shap_values_rb), axis=0)

        shap_rb_df = (
            pd.DataFrame({
                'feature':         features_df.columns.tolist(),
                'shap_importance': mean_abs_shap_rb
            })
            .sort_values('shap_importance', ascending=False)
            .reset_index(drop=True)
        )
        shap_rb_df['rank_shap'] = shap_rb_df.index + 1

        # print("=== Top Mean Absolute SHAP — Bayesian Ridge ===")
        shap_rb_df['normalized'] = shap_rb_df['shap_importance'] / shap_rb_df['shap_importance'].sum()
        shap_rb_df['cumulative'] = shap_rb_df['normalized'].cumsum()
        # display(shap_rb_df[['rank_shap', 'feature', 'shap_importance', 'cumulative']])

        # ══════════════════════════════════════════════════════
        # XGBoost
        # ══════════════════════════════════════════════════════
        from xgboost import XGBRegressor

        xgb_pipeline = Pipeline([
            ('xgb', XGBRegressor(n_jobs=-1, random_state=42, verbosity=0))
        ])

        param_grid_xgb = {
            'xgb__n_estimators':    [100, 300],
            'xgb__max_depth':       [3, 6],
            'xgb__learning_rate':   [0.05, 0.1],
            'xgb__subsample':       [0.8, 1.0],
            'xgb__colsample_bytree':[0.8, 1.0],
        }

        grid_search_xgb = GridSearchCV(xgb_pipeline, param_grid_xgb, cv=5, verbose=0, scoring='r2', n_jobs=-1)
        grid_search_xgb.fit(X_train, y_train)

        # print("\n=== Resultados de XGBoost ===")
        # print("Mejores Hiperparámetros:", grid_search_xgb.best_params_)
        # print(f"Mejor R2 (CV train): {grid_search_xgb.best_score_:.4f}")

        # ── Métricas XGBoost ──────────────────────────────────
        y_pred_xgb_train = grid_search_xgb.predict(X_train)
        y_pred_xgb_test  = grid_search_xgb.predict(X_test)

        # print("=== XGBoost Performance ===")
        # print(f"R2 train:   {r2_score(y_train, y_pred_xgb_train):.4f}")
        # print(f"R2 test:    {r2_score(y_test,  y_pred_xgb_test):.4f}")
        # print(f"MAE test:   {mean_absolute_error(y_test, y_pred_xgb_test):.2f} mA")
        # print(f"RMSE test:  {np.sqrt(mean_squared_error(y_test, y_pred_xgb_test)):.2f} mA")
        # print(f"MAPE test:  {mape(y_test, y_pred_xgb_test):.2f}%")

        # ── Plot XGBoost (comentado) ──────────────────────────
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.scatter(y_test, y_pred_xgb_test, alpha=0.7, label='Predicted vs. Actual')
        # min_val = min(y_test.min(), y_pred_xgb_test.min())
        # max_val = max(y_test.max(), y_pred_xgb_test.max())
        # ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
        # ax.set_xlabel('Actual Current (mA)'); ax.set_ylabel('Predicted Current (mA)')
        # ax.set_title('Actual vs. Predicted Current Values - XGBoost (Test Set)')
        # ax.legend(); ax.grid(True)
        # plt.tight_layout()
        # plt.savefig(f'xgb_iter_{i+1}.png', dpi=80)
        # plt.close()

        # ── Permutation Importance XGBoost ────────────────────
        result_xgb = permutation_importance(
            grid_search_xgb.best_estimator_,
            X_test, y_test,
            n_repeats=75,
            n_jobs=-1
        )

        permutation_xgb_df = pd.DataFrame({
            'feature':         features_df.columns,
            'importance_mean': result_xgb.importances_mean,
            'importance_std':  result_xgb.importances_std
        }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

        permutation_xgb_df['rank_perm'] = permutation_xgb_df.index + 1

        # print("\n--- Features contributing to 80% of model explanation (XGB - Permutation) ---")
        importancias_limpias_xgb = permutation_xgb_df['importance_mean'].clip(lower=0)
        permutation_xgb_df['normalized_importance'] = importancias_limpias_xgb / importancias_limpias_xgb.sum()
        permutation_xgb_df['cumulative_importance']  = permutation_xgb_df['normalized_importance'].cumsum()
        # display(permutation_xgb_df[['rank_perm', 'feature', 'importance_mean', 'cumulative_importance']])
        # plt.clf()

        # ── SHAP XGBoost ──────────────────────────────────────
        best_xgb_model   = grid_search_xgb.best_estimator_.named_steps['xgb']
        explainer_xgb    = shap.TreeExplainer(best_xgb_model, X_train_background)
        shap_values_xgb  = explainer_xgb.shap_values(X_test_global)

        mean_abs_shap_xgb = np.mean(np.abs(shap_values_xgb), axis=0)

        shap_xgb_df = (
            pd.DataFrame({
                'feature':         features_df.columns.tolist(),
                'shap_importance': mean_abs_shap_xgb
            })
            .sort_values('shap_importance', ascending=False)
            .reset_index(drop=True)
        )
        shap_xgb_df['rank_shap'] = shap_xgb_df.index + 1

        # print("=== Top Mean Absolute SHAP — XGBoost ===")
        shap_xgb_df['normalized'] = shap_xgb_df['shap_importance'] / shap_xgb_df['shap_importance'].sum()
        shap_xgb_df['cumulative'] = shap_xgb_df['normalized'].cumsum()
        # display(shap_xgb_df[['rank_shap', 'feature', 'shap_importance', 'cumulative']])

        # ══════════════════════════════════════════════════════
        # Desempeño General — Guardado CSV
        # ══════════════════════════════════════════════════════
        y_pred_svr      = grid_search_svr.best_estimator_.predict(X_test)
        y_pred_rf       = grid_search_rf.best_estimator_.predict(X_test)
        y_pred_bayesian = grid_search_bayesian.best_estimator_.predict(X_test)
        y_pred_xgb      = grid_search_xgb.best_estimator_.predict(X_test)

        results_df = pd.DataFrame({
            'Iteration':      i + 1,
            'Actual_Current': y_test.flatten(),
            'Pred_SVR':       y_pred_svr,
            'Pred_RF':        y_pred_rf,
            'Pred_Bayesian':  y_pred_bayesian,
            'Pred_XGBoost':   y_pred_xgb
        })

        output_file = 'model_predictions_history.csv'
        if os.path.exists(output_file):
            results_df.to_csv(output_file, mode='a', header=False, index=False)
            # print(f"✅ Iteración {i+1} añadida a '{output_file}'.")
        else:
            results_df.to_csv(output_file, mode='w', header=True, index=False)
            # print(f"📂 Archivo nuevo '{output_file}' creado con iteración {i+1}.")

        # display(results_df.head())

        # ── Rankings consenso ─────────────────────────────────
        file_perm = 'ranking_consenso_perm.csv'
        file_shap = 'ranking_consenso_shap.csv'

        model_results = [
            ('SVR',           permutation_svr_df, shap_svr_df),
            ('Random Forest', permutation_rf_df,  shap_rf_df),
            ('Bayesian Ridge',permutation_rb_df,  shap_rb_df),
            ('XGBoost',       permutation_xgb_df, shap_xgb_df)
        ]

        data_perm, data_shap = [], []
        for model_name, df_perm, df_shap in model_results:
            temp_perm = df_perm[['feature', 'rank_perm', 'importance_mean']].copy()
            temp_perm['model'] = model_name
            data_perm.append(temp_perm)

            temp_shap = df_shap[['feature', 'rank_shap', 'shap_importance']].copy()
            temp_shap['model'] = model_name
            data_shap.append(temp_shap)

        df_iter_perm = pd.concat(data_perm, ignore_index=True)
        df_iter_shap = pd.concat(data_shap, ignore_index=True)

        def save_with_append(df_new, filename):
            if os.path.exists(filename):
                df_existing = pd.read_csv(filename)
                df_updated  = pd.concat([df_existing, df_new], ignore_index=True)
                df_updated.to_csv(filename, index=False)
                # print(f"✅ Datos añadidos a {filename}")
            else:
                df_new.to_csv(filename, index=False)
                # print(f"📂 Archivo {filename} creado desde cero.")

        save_with_append(df_iter_perm, file_perm)
        save_with_append(df_iter_shap, file_shap)

        # print(f"\nResumen de filas guardadas:")
        # print(f"Permutación: {len(df_iter_perm)} filas.")
        # print(f"SHAP: {len(df_iter_shap)} filas.")

        # ── Registro de éxito ─────────────────────────────────
        r2_svr  = r2_score(y_test, y_pred_svr)
        r2_rf   = r2_score(y_test, y_pred_rf)
        r2_bay  = r2_score(y_test, y_pred_bayesian)
        r2_xgb  = r2_score(y_test, y_pred_xgb)
        iteration_log.append({
            'iter': i + 1, 'status': 'ok',
            'R2_SVR': round(r2_svr, 4), 'R2_RF': round(r2_rf, 4),
            'R2_Bayesian': round(r2_bay, 4), 'R2_XGB': round(r2_xgb, 4)
        })
        print(f"[ITER {i+1}] ✅ OK  |  R2: SVR={r2_svr:.4f}  RF={r2_rf:.4f}  Bay={r2_bay:.4f}  XGB={r2_xgb:.4f}")

    except Exception as e:
        # Captura el error, lo registra y continúa con la siguiente iteración
        tb = traceback.format_exc()
        iteration_log.append({'iter': i + 1, 'status': 'ERROR', 'detail': str(e)})
        print(f"[ITER {i+1}] ❌ ERROR — {e}")
        print(f"           Traceback guardado en error_iter_{i+1}.txt")
        with open(f'error_iter_{i+1}.txt', 'w') as f:
            f.write(tb)

    finally:
        # ── Liberación de memoria ──────────────────────────────
        gc.collect()

# ══════════════════════════════════════════════════════════════
# ZONA ACME — Análisis post-loop
# ══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  RESUMEN FINAL DE ITERACIONES")
print("="*60)
ok_count  = sum(1 for r in iteration_log if r['status'] == 'ok')
err_count = sum(1 for r in iteration_log if r['status'] == 'ERROR')
print(f"  Total: {len(iteration_log)}  |  ✅ OK: {ok_count}  |  ❌ Errores: {err_count}")
print("-"*60)
for row in iteration_log:
    if row['status'] == 'ok':
        print(f"  Iter {row['iter']:>3}  ✅  "
              f"SVR={row['R2_SVR']}  RF={row['R2_RF']}  Bay={row['R2_Bayesian']}  XGB={row['R2_XGB']}")
    else:
        print(f"  Iter {row['iter']:>3}  ❌  {row['detail']}")
print("="*60)

# ── Guardar log de resultados ─────────────────────────────────
pd.DataFrame(iteration_log).to_csv('iteration_log.csv', index=False)
print("\n[INFO] Log guardado en 'iteration_log.csv'")

# ── Gráfico consenso (solo si hay datos) ─────────────────────
if os.path.exists('model_predictions_history.csv'):
    import seaborn as sns

    df_preds  = pd.read_csv('model_predictions_history.csv')
    df_melted = df_preds.melt(id_vars=['Actual_Current', 'Iteration'],
                              var_name='Model', value_name='Predicted')

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(data=df_melted, x='Actual_Current', y='Predicted',
                 hue='Model', errorbar=('ci', 95), linewidth=2, ax=ax)
    ax.plot([df_preds['Actual_Current'].min(), df_preds['Actual_Current'].max()],
            [df_preds['Actual_Current'].min(), df_preds['Actual_Current'].max()],
            color='black', linestyle='--', label='Ideal (Actual = Predicted)')
    ax.set_title(f'Consenso de Predicciones tras {ok_count} Iteraciones (IC 95%)', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('consenso_predicciones.png', dpi=100)
    plt.close()
    # plt.show()
    print("[INFO] Gráfico guardado en 'consenso_predicciones.png'")

# ── Tabla de consenso final ───────────────────────────────────
if os.path.exists('ranking_consenso_perm.csv') and os.path.exists('ranking_consenso_shap.csv'):
    df_perm_final = pd.read_csv('ranking_consenso_perm.csv')
    df_shap_final = pd.read_csv('ranking_consenso_shap.csv')

    resumen_perm = df_perm_final.groupby('feature').agg(
        {'rank_perm': 'mean', 'importance_mean': 'mean'}).reset_index()
    resumen_shap = df_shap_final.groupby('feature').agg(
        {'rank_shap': 'mean', 'shap_importance': 'mean'}).reset_index()

    tabla_consenso = pd.merge(resumen_perm, resumen_shap, on='feature')
    tabla_consenso['consenso_score'] = tabla_consenso['rank_perm'] + tabla_consenso['rank_shap']
    tabla_consenso = tabla_consenso.sort_values('consenso_score').reset_index(drop=True)

    tabla_consenso['perm_norm'] = (tabla_consenso['importance_mean'].clip(lower=0) /
                                   tabla_consenso['importance_mean'].clip(lower=0).sum())
    tabla_consenso['shap_norm'] = tabla_consenso['shap_importance'] / tabla_consenso['shap_importance'].sum()
    tabla_consenso['cum_importance_perm'] = tabla_consenso['perm_norm'].cumsum()
    tabla_consenso['cum_importance_shap'] = tabla_consenso['shap_norm'].cumsum()

    cols_finales = [
        'feature', 'consenso_score',
        'rank_perm', 'importance_mean', 'cum_importance_perm',
        'rank_shap', 'shap_importance', 'cum_importance_shap'
    ]

    # print("=== TABLA DE CONSENSO FINAL (Promedio de 100 ejecuciones) ===")
    # display(tabla_consenso[cols_finales].head(50))

    tabla_consenso[cols_finales].to_csv('tabla_consenso_final.csv', index=False)
    print("[INFO] Tabla de consenso guardada en 'tabla_consenso_final.csv'")

# ══════════════════════════════════════════════════════════════
# CURVA DE CONVERGENCIA DEL CONSENSO_SCORE
# ══════════════════════════════════════════════════════════════
if os.path.exists('ranking_consenso_perm.csv') and os.path.exists('ranking_consenso_shap.csv'):
    import matplotlib.cm as cm
 
 
    df_perm_conv = pd.read_csv('ranking_consenso_perm.csv')
    df_shap_conv = pd.read_csv('ranking_consenso_shap.csv')

    TOP_N = len(df_perm_conv['feature'].unique())
 
    # Reconstruir número de iteración a partir del orden de inserción
    n_features_conv  = df_perm_conv['feature'].nunique()
    n_models_conv    = df_perm_conv['model'].nunique()
    rows_per_iter    = n_features_conv * n_models_conv
    n_iters_conv     = len(df_perm_conv) // rows_per_iter
 
    df_perm_conv['iteration'] = (df_perm_conv.index // rows_per_iter) + 1
    df_shap_conv['iteration'] = (df_shap_conv.index // rows_per_iter) + 1
 
    df_perm_conv = df_perm_conv[df_perm_conv['iteration'] <= n_iters_conv]
    df_shap_conv = df_shap_conv[df_shap_conv['iteration'] <= n_iters_conv]
 
    # Calcular consenso_score acumulado iteración a iteración
    conv_records = []
    for n in range(1, n_iters_conv + 1):
        sub_p = df_perm_conv[df_perm_conv['iteration'] <= n]
        sub_s = df_shap_conv[df_shap_conv['iteration'] <= n]
 
        avg_p = sub_p.groupby('feature')['rank_perm'].mean().reset_index()
        avg_s = sub_s.groupby('feature')['rank_shap'].mean().reset_index()
 
        merged_conv = pd.merge(avg_p, avg_s, on='feature')
        merged_conv['consenso_score'] = merged_conv['rank_perm'] + merged_conv['rank_shap']
        merged_conv['iteration'] = n
        conv_records.append(merged_conv)
 
    df_conv = pd.concat(conv_records, ignore_index=True)
 
    # Top-N según iteración final
    final_scores  = df_conv[df_conv['iteration'] == n_iters_conv].sort_values('consenso_score')
    top_features  = final_scores['feature'].head(TOP_N).tolist()
    df_conv_top   = df_conv[df_conv['feature'].isin(top_features)]
 
    df_conv_top.to_csv('convergencia_consenso_data.csv', index=False)
 
    # Graficar
    colors = cm.tab10(np.linspace(0, 1, TOP_N))
    fig, axes = plt.subplots(1, 2, figsize=(24, 9))
    fig.suptitle(f'Curva de Convergencia del Consenso Score — Top {TOP_N} Features',
                 fontsize=14, fontweight='bold')
 
    ax1 = axes[0]
    for feat, color in zip(top_features, colors):
        sub = df_conv_top[df_conv_top['feature'] == feat].sort_values('iteration')
        ax1.plot(sub['iteration'], sub['consenso_score'],
                 label=feat, color=color, linewidth=1.8)
    ax1.set_xlabel('Iteraciones acumuladas', fontsize=11)
    ax1.set_ylabel('Consenso Score', fontsize=11)
    ax1.set_title('Evolución del Consenso Score', fontsize=12)
    ax1.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax1.grid(True, alpha=0.3)
 
    ax2 = axes[1]
    for feat, color in zip(top_features, colors):
        sub   = df_conv_top[df_conv_top['feature'] == feat].sort_values('iteration')
        delta = sub['consenso_score'].diff().abs()
        ax2.plot(sub['iteration'], delta,
                 label=feat, color=color, linewidth=1.5, alpha=0.85)
    ax2.axhline(y=0.05, color='red', linestyle='--', linewidth=1.2,
                label='Umbral 0.05 (convergencia)')
    ax2.set_xlabel('Iteraciones acumuladas', fontsize=11)
    ax2.set_ylabel('|Δ Consenso Score|', fontsize=11)
    ax2.set_title('Velocidad de Convergencia\n(cerca de 0 = estabilizado)', fontsize=12)
    ax2.legend(fontsize=7, loc='upper right', framealpha=0.7)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
 
    plt.tight_layout()
    plt.savefig('convergencia_consenso.png', dpi=120)
    plt.close()
    print("[INFO] Curva de convergencia guardada en 'convergencia_consenso.png'")
 
    # Diagnóstico automático
    umbral_conv = 0.05
    convergencia_iter = {}
    for feat in top_features:
        sub   = df_conv_top[df_conv_top['feature'] == feat].sort_values('iteration')
        delta = sub['consenso_score'].diff().abs().fillna(999)
        supera = sub['iteration'][delta > umbral_conv]
        convergencia_iter[feat] = int(supera.iloc[-1]) + 1 if len(supera) > 0 else 1
 
    max_conv_iter = max(convergencia_iter.values())
    print("\n" + "="*60)
    print("  DIAGNÓSTICO DE CONVERGENCIA")
    print("="*60)
    print(f"  {'Feature':<35} {'Converge en iter':>16}")
    print(f"  {'-'*53}")
    for feat in top_features:
        print(f"  {feat:<35} {convergencia_iter[feat]:>16}")
    print(f"\n  → Mínimo de iteraciones para convergencia total: {max_conv_iter}")
    print("="*60)
 
print("\n[INFO] Script finalizado. Todos los resultados están en los CSV.")
 
