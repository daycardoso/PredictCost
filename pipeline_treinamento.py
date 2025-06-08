import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def main():
    print("=== Iniciando pipeline de avaliação ===")

    # 1) Carregar CSV e criar bin de cost para estratificação
    print("Step 1: Carregando dados e criando cost_bin para estratificação...")
    df = pd.read_csv('df_unificado.csv')
    df['cost_bin'] = pd.qcut(df['cost'], q=10, labels=False, duplicates='drop')
    print("Step 1 completado.")

    # 2) Amostragem estratificada de 10%
    print("Step 2: Realizando amostragem estratificada de 10%...")
    _, df_sample = train_test_split(
        df,
        test_size=0.10,
        random_state=42,
        stratify=df['cost_bin']
    )
    df_sample = df_sample.drop(columns=['cost_bin'])
    print("Step 2 completado.")

    # 3) Preparar X e y
    print("Step 3: Preparando features (X) e alvo (y)...")
    X = df_sample.drop(columns=['cost'])
    y = df_sample['cost'].values
    print(f" - X shape: {X.shape}, y shape: {y.shape}")

    # 4) Escalonamento (útil p/ SVR e KNN)
    print("Step 4: Escalonando features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Step 4 completado.")

    # 5) Definir 5-Fold CV com embaralhamento
    print("Step 5: Configurando KFold CV (5 splits, shuffle=True)...")
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    print("Step 5 completado.")

    # 6) Modelos a avaliar
    print("Step 6: Definindo modelos para pré-seleção...")
    models = {
        'SVR': SVR(kernel='rbf'),
        'KNN': KNeighborsRegressor(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    }
    print(f" - Modelos definidos: {list(models.keys())}")
    print("Step 6 completado.")

    # 7) Grids de parâmetros para pré-seleção
    print("Step 7: Definindo grids de hiperparâmetros...")
    param_grids = {
        'SVR': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'epsilon': [0.01, 0.1, 0.2]
        },
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 13]
        },
        'DecisionTree': {
            'max_depth': [None, 5, 10, 15]
        },
        'RandomForest': {
            'max_depth': [None, 5, 10, 15]
        }
    }
    for name, grid in param_grids.items():
        print(f" - Grid {name}: {grid}")
    print("Step 7 completado.")

    # 8) Métricas – nomes de scoring do sklearn
    print("Step 8: Definindo métricas de scoring...")
    scoring = {
        'r2': 'r2',
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'mape': 'neg_mean_absolute_percentage_error',
        'medae': 'neg_median_absolute_error',
        'maxerr': 'max_error',
        'expvar': 'explained_variance'
    }
    print(f" - Métricas definidas: {list(scoring.keys())}")
    print("Step 8 completado.")

    # 9) Pré-seleção de hiperparâmetros via GridSearch
    print("Step 9: Iniciando pré-seleção de hiperparâmetros via GridSearchCV...")
    best_models = {}
    for name, model in models.items():
        print(f"   --> Step 9.{name}: iniciando GridSearchCV com {len(param_grids[name]) if isinstance(param_grids[name], dict) else 'N/A'} parâmetros...")
        X_input = X_scaled if name in ['SVR', 'KNN'] else X
        print(f"       > Usando dados {'escalonados' if name in ['SVR','KNN'] else 'originais'} com shape {X_input.shape}")
        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            cv=cv,
            scoring='r2',
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_input, y)
        best_models[name] = grid.best_estimator_
        print(f"       > {name} melhores parâmetros: {grid.best_params_}")
        print(f"       > {name} CV R2: {grid.best_score_:.4f}")
    print("Step 9 completado.")

    # 10) Avaliação final dos modelos pré-selecionados
    print("Step 10: Avaliando modelos pré-selecionados...")
    for name, model in best_models.items():
        print(f"   --> Step 10.{name}: iniciando avaliação final...")
        X_input = X_scaled if name in ['SVR', 'KNN'] else X
        print(f"       > Usando dados {'escalonados' if name in ['SVR','KNN'] else 'originais'} com shape {X_input.shape}")
        print(f"       > Executando cross_validate para {name}...")
        cv_results = cross_validate(
            model,
            X_input,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False
        )
        print(f"       > cross_validate concluído para {name}.")
        for metric, scores in cv_results.items():
            if metric.startswith('test_'):
                mname = metric.replace('test_', '')
                vals = scores
                if mname in ['mse', 'mae', 'mape', 'medae']:
                    vals = -vals
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                print(f"           * {mname:6s} → {mean_val:.4f} (± {std_val:.4f})")
        print(f"       > fit_time   → {cv_results['fit_time'].mean():.4f}s (± {cv_results['fit_time'].std():.4f}s)")
        print(f"       > score_time → {cv_results['score_time'].mean():.4f}s (± {cv_results['score_time'].std():.4f}s)")
        print(f"   <-- Step 10.{name} completado.")
    print("Step 10 completado.")

    print("=== Pipeline de avaliação concluído ===")

if __name__ == '__main__':
    main()
