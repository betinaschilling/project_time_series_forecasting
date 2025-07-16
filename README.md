# projeto-forecast

Este repositório contém um pipeline completo de previsão de demanda de varejo, implementado em Python de forma modular e escalável.

## visão geral

O objetivo deste projeto é demonstrar todas as etapas de um fluxo de trabalho de forecasting:

- ingestão e versionamento de dados brutos  
- exploração e análise exploratória das séries históricas  
- definição e ajuste de baseline ARIMA/SARIMA  
- geração de features de séries temporais para modelos de machine learning  
- treino e comparação de modelos de ML (regressão regularizada, LightGBM, CatBoost)  
- avaliação de desempenho com métricas adequadas (RMSE, WMAPE, R²)  
- visualização de resultados e análise de resíduos  
- função de input por SKU e previsão para horizontes definidos

## estrutura de pastas

```plaintext
projeto-forecast/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_arima.ipynb
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   ├── visualization/
│   \└── utils/
├── scripts/
│   ├── run_ingest.py
│   ├── run_preprocess.py
│   \└── run_train.py
├── tests/
├── docs/
├── requirements.txt
├── setup.py
└── .gitignore
```

## pré-requisitos

- Python 3.8 ou superior  
- pip instalado

Instalar dependências:

```bash
pip install -r requirements.txt
```

## uso

1. carregar dados brutos em `data/raw/`  
2. executar ingestão:
   ```bash
   python scripts/run_ingest.py
   ```  
3. executar pré-processamento:
   ```bash
   python scripts/run_preprocess.py
   ```  
4. treinar modelos e gerar previsões:
   ```bash
   python scripts/run_train.py
   ```  
5. revisar notebooks em `notebooks/` para análises interativas

## testes

Executar testes unitários:

```bash
pytest
```

## contribuição

Sugestões de melhorias e correções podem ser feitas via pull request. Abra uma issue para discutir mudanças maiores.

## licença

Este projeto está licenciado sob os termos do MIT License.
