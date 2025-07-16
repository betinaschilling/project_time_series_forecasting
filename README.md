# Projeto de Forecasting de Vendas

Este README descreve em detalhes cada etapa do pipeline de forecasting de vendas, as motivações de design, as principais funções envolvidas e exemplos de logs de execução.

---

## Sumário

1. [Visão Geral](#visão-geral)  
2. [Estrutura de Pastas e Scripts](#estrutura-de-pastas-e-scripts)  
3. [Etapas do Pipeline](#etapas-do-pipeline)  
   1. [Ingestão (`ingest.py`)](#1-ingestão-ingestpy)  
   2. [Pré-processamento (`preprocess.py`)](#2-pré-processamento-preprocesspy)  
   3. [Geração de Features (`make_features.py`)](#3-geração-de-features-make_featurespy)  
   4. [Treino LightGBM (`train_lgbm.py`)](#4-treino-lightgbm-train_lgbmpy)  
   5. [Treino CatBoost (`train_catboost.py`)](#5-treino-catboost-train_catboostpy)  
   6. [Reconciliação (`reconcile.py`)](#6-reconciliação-reconcilepy)  
   7. [Avaliação de Métricas (`metrics.py`)](#7-avaliação-de-métricas-metricspy)  
   8. [Dashboard Streamlit (`app.py`)](#8-dashboard-streamlit-apppy)  
4. [Como Executar](#como-executar)  
5. [Exemplos de Logs](#exemplos-de-logs)  

---

## Visão Geral

Construímos um pipeline completo que:

- **Ingestão** de CSV bruto → Delta  
- **Pré-processamento**: expansão de datas, zero-fill, limpeza  
- **Feature Engineering**: lags, médias móveis, calendário, flags  
- **Modelagem**: LightGBM e CatBoost (in-sample + rolling forecast)  
- **Reconciliação**: média simples entre previsões granular  
- **Avaliação**: RMSE, MAE, MAPE, WMAPE e R² em granular e agregado  
- **Visualização**: Streamlit com filtros, tabelas e gráficos  

Cada etapa é acionada por um script CLI instalado via `setup.py` (ex: `forecast-ingest`, `forecast-features`, `forecast-train-lgbm`, etc.).

---

## Estrutura de Pastas e Scripts

project_time_series_forecasting/
├─ data/
│  ├─ processed/
│  │  └─ vendas_processed.csv/.delta/…
│  ├─ features/
│  │  └─ vendas_features.csv/.delta/…
│  ├─ interim/
│  └─ models/
│     ├─ ml_forecast.csv
│     ├─ catboost_forecast.csv
│     ├─ reconciled_sku_forecast.csv
│     └─ reconciled_daily_summary.csv
├─ notebooks/
├─ src/
│  ├─ ingestion/
│  │  └─ loader.py
│  ├─ preprocessing/
│  │  └─ clean.py
│  ├─ features/
│  │  └─ make_features.py
│  ├─ models/
│  │  ├─ train_lgbm.py
│  │  ├─ train_catboost.py
│  │  └─ reconcile.py --> Média do desempenho dos modelos
│  ├─ evaluation/
│  │  └─ metrics.py
│  ├─ scripts/
│  │  └─ run_sku_forecaster.py --> Função para Gerar o forecast por SKU.
│  └─ visualization/
│     └─ app.py --> Realizado x Previsto
├─ setup.py
└─ README.md


---

## Etapas do Pipeline

### 1. Ingestão (`ingest.py`)
- **Objetivo**: ler CSV bruto, salvar em CSV/Delta intermediário.  
- **Função principal**: `Ingestor.run()`.  
- **Por quê?**: garantir rastreabilidade e persistência de raw → interim.

### 2. Pré-processamento (`preprocess.py`)
- **Objetivo**: agregação, malha completa de datas, imputação zeros.  
- **Funções**:
  - `expand_dates()`
  - `fill_missing()`
- **Por quê?**: criar histórico consistente por SKU/data.

### 3. Geração de Features (`make_features.py`)
- **Objetivo**: criar lags, rolling means, calendário e flags.  
- **Passos**:
  1. `add_imputation_flag()`  
  2. `add_lag_features()`  
  3. `add_rolling_features()`  
  4. `add_calendar_features()`  
- **Entrada**: `FeatureEngineer.run()`.  

### 4. Treino LightGBM (`train_lgbm.py`)
- **Objetivo**: treinar, cross-validate e fazer rolling forecast.  
- **Funções**:
  - `cross_validate(df)`
  - `train_and_save(df)`
  - `rolling_forecast(model, df)`
- **Por quê?**: modelo rápido e robusto.

### 5. Treino CatBoost (`train_catboost.py`)
- Mesma lógica do LightGBM, mas usando `CatBoostRegressor`.  
- **Funções**: `cross_validate`, `train_and_save`, `predict_future`.

### 6. Reconciliação (`reconcile.py`)
- **Objetivo**: unir forecasts LGBM + CatBoost + histórico → granular e daily.  
- **Funções**:
  - `load_data()`
  - `compute_reconciled()`
  - `save_outputs()`

### 7. Avaliação de Métricas (`metrics.py`)
- **Objetivo**: calcular RMSE, MAE, MAPE, WMAPE, R² em granular e agregado.  
- **Funções**:
  - `load_all()`
  - `_compute_metrics(y_true, y_pred)`
  - `compute()`
  - `save(df_metrics)`

### 8. Dashboard Streamlit (`app.py`)
- **Visões**:
  1. **Resumo APlicação**  
  2. **Resumo Diário**  
  3. **Por SKU**  
  4. **Métricas**  
- **Entrada**: `forecast-dashboard` → `visualization.app.main()`.


### 9. Forecast por SKU
- **O quê**  
  Gera previsões diárias de vendas para um único SKU, incluindo:  
  1. **Fitted** (in-sample) – aplica os modelos treinados sobre o histórico de features.  
  2. **Forecast** (horizonte futuro) – reconstrói recursivamente as features dos próximos dias e prevê as vendas.

- **Por que**  
  Permite analisar individualmente o comportamento de cada produto, validar o ajuste dos modelos no histórico e planejar estoque específico por SKU.

- **Principais funções/métodos**  
  - `main()` – recebe o código do SKU e caminhos de entrada/saída, instancia `SKUForecaster` e chama `run()`.  
  - `SKUForecaster.load_models()` – carrega os objetos LightGBM e CatBoost do disco.  
  - `SKUForecaster.load_data()` – lê e filtra histórico processado e features apenas para o SKU selecionado.  
  - `SKUForecaster.forecast()` –  
    - monta previsões “in-sample” (fitted) usando todo o histórico de features,  
    - gera previsões recursivas para o horizonte configurado (por padrão 7 dias),  
    - concatena ambos em um único DataFrame.  
  - `SKUForecaster.save()` – salva o resultado em `data/models/forecast_sku_<SKU>.csv` com colunas:  
    `sku`, `data`, `lgbm_fitted`, `cb_fitted`, `lgbm_forecast`, `cb_forecast`.  
---

## Como Executar

```bash
pip install -e .        # instala dependências e entry points
forecast-load
forecast-clean
forecast-features
forecast-train
forecast-train
forecast-reconcile
forecast-evaluate
forecast-sku
forecast-dashboard


