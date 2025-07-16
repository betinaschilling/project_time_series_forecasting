# Documentação do Projeto de Forecasting de Vendas

Esta documentação descreve todo o fluxo de trabalho implementado para a solução de previsão de demanda, detalhando as etapas, motivações, ferramentas escolhidas e as funções Python responsáveis por cada parte do processo.

---

## 1. Ingestão de Dados

**Objetivo:** Ler os dados brutos de vendas e armazenar em formatos intermediários (CSV e Delta).

**Motivação:** Garantir idempotência e rastreabilidade, usando Delta Lake para versionamento.

**Ferramentas:** Spark com Delta Lake, Python.

**Chamadas de função:**
- `scripts/ingest.py` → `main()` inicia o processo.
- `Ingestor.run()` coordena a ingestão:
  - `Ingestor.load_raw()` lê CSV bruto.
  - `Ingestor.expand_dates()` expande datas faltantes.
  - `Ingestor.save_interim_csv()` grava CSV em `data/interim/`.
  - `Ingestor.save_interim_delta()` grava Delta em `data/interim/`.

---

## 2. Pré-processamento

**Objetivo:** Limpar, agregar e preparar o histórico de vendas para engenharia de features.

**Motivação:** Garantir qualidade dos dados e uniformidade antes de gerar features.

**Ferramentas:** Spark com Delta Lake, Python.

**Chamadas de função:**
- `scripts/preprocess.py` → `main()`.
- `Preprocessor.run()` executa as transformações:
  - `Preprocessor.read_delta()` carrega dados de `data/interim/*.delta`.
  - `Preprocessor.deduplicate()` agrega duplicatas.
  - `Preprocessor.mark_imputation()` sinaliza valores imputados.
  - `Preprocessor.fill_missing_sales()` preenche lacunas com zero.
  - `Preprocessor.sort_and_save_csv()` salva CSV em `data/processed/`.
  - `Preprocessor.sort_and_save_delta()` salva Delta em `data/processed/`.

---

## 3. Engenharia de Features

**Objetivo:** Gerar lags, médias móveis e variáveis de calendário para alimentar os modelos.

**Motivação:** Modelos de séries temporais se beneficiam de informações de dependência temporal e sazonalidade.

**Ferramentas:** Spark com Delta Lake, Python.

**Chamadas de função:**
- `src/features/make_features.py` → `main()`.
- `FeatureEngineer.run()`:
  - `FeatureEngineer.load_data()` lê Delta de `data/processed/`.
  - `FeatureEngineer.add_imputation_flag()` propaga flag de imputação.
  - `FeatureEngineer.add_lag_features()` gera `lag_1`, `lag_7`, `lag_14`.
  - `FeatureEngineer.add_rolling_features()` gera `roll_mean_7`, `roll_mean_14`, `roll_mean_30`.
  - `FeatureEngineer.add_calendar_features()` cria `weekday`, `is_weekend`, `month`.
  - Salva CSV em `data/features/` via `write.csv`.
  - Salva Delta em `data/features/` via `df.write.format("delta")`.

---

## 4. Modelagem (LightGBM e CatBoost)

**Objetivo:** Treinar e validar dois modelos de regressão para previsão granular por SKU.

**Motivação:** Comparar desempenho de algoritmos baseados em árvores.

**Ferramentas:** LightGBM, CatBoost, scikit-learn, pandas.

**Chamadas de função (LightGBM):**
- `src/models/train_lgbm.py` → `main()`.
- `LGBMTrainer.run()`:
  - `LGBMTrainer.load_data()` carrega `data/features/vendas_features.csv`.
  - `LGBMTrainer.cross_validate()` executa `TimeSeriesSplit` e registra RMSE de cada fold.
  - `LGBMTrainer.train_and_save()` treina em todo o conjunto e salva modelo em `data/models/lgbm_model.pkl`.
  - `LGBMTrainer.rolling_forecast()` gera forecast recursivo e salva `data/models/ml_forecast.csv`.

**Chamadas de função (CatBoost):**
- `src/models/train_catboost.py` → `main()`.
- `CatBoostTrainer.run()`:
  - `CatBoostTrainer.load_data()` carrega features CSV.
  - `CatBoostTrainer.cross_validate()` valida com `TimeSeriesSplit`.
  - `CatBoostTrainer.train_and_save()` treina e salva em `data/models/catboost_model.cbm`.
  - `CatBoostTrainer.predict_future()` gera forecasts e salva `data/models/catboost_forecast.csv`.

---

## 5. Reconciliação

**Objetivo:** Combinar previsões LGBM e CatBoost garantindo consistência via média simples.

**Motivação:** Obter robustez e reduzir viés de cada modelo.

**Ferramentas:** pandas.

**Chamadas de função:**
- `src/models/reconcile.py` → `main()`.
- `Reconciler.run()`:
  - `Reconciler.load_data()` lê `ml_forecast.csv`, `catboost_forecast.csv`, `vendas_processed.csv`.
  - `Reconciler.compute_reconciled()` calcula `reconciled` como média de `ml_pred` e `cb_pred`.
  - `Reconciler.save_sku_forecast()` salva `data/models/reconciled_sku_forecast.csv`.
  - `Reconciler.save_daily_summary()` salva `data/models/reconciled_daily_summary.csv` (agregado diário).

---

## 6. Avaliação de Métricas

**Objetivo:** Calcular RMSE, MAE, MAPE, WMAPE e R² nos níveis granular e agregado.

**Motivação:** Quantificar a performance e comparar modelos.

**Ferramentas:** scikit-learn, pandas.

**Chamadas de função:**
- `src/evaluation/metrics.py` → `main()`.
- `MetricsEvaluator.run()`:
  - `MetricsEvaluator.load_all()` lê históricos e forecasts.
  - `MetricsEvaluator._compute_metrics()` calcula métricas.
  - `MetricsEvaluator.compute()` itera por modelo e nível.
  - `MetricsEvaluator.save()` salva `data/models/metrics_report.csv`.

---

## 7. Visualização (Streamlit)

**Objetivo:** Dashboard interativo para storytelling, análises diárias, por SKU e métricas.

**Motivação:** Facilitar exploração e comunicação dos resultados.

**Ferramentas:** Streamlit, Altair, pandas.

**Chamadas de função:**
- `src/visualization/app.py` → `main()`.
- `load_csv()` carrega CSV com cache.
- Abas definidas por `st.sidebar.radio()`:
  - **Resumo Aplicação**: Storytelling.
  - **Resumo Diário**: tabela e gráfico de `reconciled_daily_summary.csv`.
  - **Por SKU**: tabela e gráfico de `reconciled_sku_forecast.csv`.
  - **Métricas**: tabela e gráfico de `metrics_report.csv`.

---

*Atualize esta documentação sempre que novas etapas ou funções forem adicionadas ao projeto.*  
