# PredictCost

Este projeto explora como prever o custo de manobra entre duas poses de robô usando dados sintéticos. O repositório contém um pipeline completo de treinamento em Python, notebooks de exemplo e modelos pré-treinados.

## Dataset

Cada amostra representa uma pose inicial e final em um espaço 2D:

* **Features**: `xi`, `yi`, `thetai`, `betai`, `xf`, `yf`, `thetaf`, `betaf`

  * `x` e `y` em metros
  * `theta` e `beta` em radianos
* **Target**: custo de manobra (valor real)

O arquivo `df_unificado.csv` deve estar presente para executar o pipeline de treinamento e pode ser encontrado em [df_unificado](https://www.kaggle.com/datasets/dayanacardoso/pose-inicial-pose-final-custo/data).

## Estrutura do Repositório

* `pipeline_treinamento.py` – script principal para treinamento e avaliação de modelos.
* `PredictCostRandomForestRegressor.ipynb` – notebook com experimentos para Random Forest.
* `PredictCostDecisionTreeRegressor.ipynb` – notebook explorando Decision Trees.
* `getting_started_tutorials/cuml_sklearn_colab_demo.ipynb` – exemplo usando GPUs.
* `relatorio_gpu.json` – relatório com métricas de cross-validation e melhores hiperparâmetros.
* `rf_final_gpu.pkl` e `xgb_final_gpu.pkl` – modelos pré-treinados (Random Forest e XGBoost).
* `LICENSE` – licença MIT.

## Pipeline de Treinamento

O script `pipeline_treinamento.py` executa os seguintes passos:

1. Carrega dados de `df_unificado.csv` e cria uma coluna `cost_bin` para amostragem estratificada.
2. Amostra 10% do dataset usando *train-test split* estratificado.
3. Prepara `X` (features) e `y` (target).
4. Escala as features com `StandardScaler` para algoritmos que se beneficiam de normalização (SVR e KNN).
5. Define um *KFold* embaralhado de 5 folds.
6. Realiza *grid search* de hiperparâmetros para quatro modelos:

   * Support Vector Regressor (SVR)
   * K‑Nearest Neighbors (KNN)
   * Decision Tree Regressor
   * Random Forest Regressor
7. Seleciona o melhor estimador de cada grid e avalia com `cross_validate`, obtendo métricas como R², MSE, MAE, MAPE, erro absoluto mediano, erro máximo e variância explicada.

As métricas de um experimento usando GPU estão em `relatorio_gpu.json`, com Random Forest alcançando R² de \~0.90.

## Uso

1. Coloque `df_unificado.csv` no diretório raiz do repositório.
2. Instale as dependências:

   ```bash
   pip install scikit-learn pandas numpy cuml xgboost
   ```
3. Execute:

   ```bash
   python pipeline_treinamento.py
   ```

Os notebooks fornecem experimentação adicional e podem ser executados no Jupyter ou Google Colab.

## Resultados

Exemplo de avaliação em GPU (veja `relatorio_gpu.json`):

```
RandomForest:
  R²   0.9062
  MSE  0.0003569
  MAE  0.0115
XGBoost:
  R²   0.8746
  MSE  0.0004772
  MAE  0.0146
```

Modelos pré-treinados `rf_final_gpu.pkl` e `xgb_final_gpu.pkl` estão disponíveis para inferência rápida.

## Licença

Este projeto está licenciado sob os termos da Licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.
